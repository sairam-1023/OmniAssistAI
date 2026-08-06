"""
Full document ingestion pipeline: classify (Vision) -> extract text
(OCR, routed by doc_type) -> add to the FAISS index (Language) for
semantic search and RAG. This replaces the Week 4 hand-written mock
documents with real, extracted content from actual uploaded images.

Usage:
    python3 -m modules.ocr.ingest path/to/image.jpg
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # mitigates a known faiss/torch
                                               # OpenMP conflict on macOS

import json
import pickle
import sys
import uuid

from modules.core.logging_config import configure_logging, get_logger
from modules.ocr.extract import extract_text
from modules.vision.predict import predict as classify_document  # torch-based; import before faiss

# IMPORTANT: faiss must be imported AFTER the torch-based modules above.
# Importing faiss first caused a silent segfault on macOS (Apple
# Silicon) — no Python traceback, just a hard crash — due to a native
# (C++) library conflict between faiss and torch's bundled OpenMP
# runtimes. Confirmed via isolated testing: Vision's predict() worked
# perfectly alone, and only crashed once faiss was imported earlier in
# the same process. See docs/ocr_model_notes.md for the full
# debugging trail.
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

configure_logging()
logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "models/language/vector_index"


def ingest_document(image_path: str, filename: str | None = None) -> dict:
    """
    Runs the full pipeline on one image and adds its extracted text to
    the FAISS index. Returns a summary dict of what was ingested.
    """
    filename = filename or image_path.split("/")[-1]
    document_id = f"doc_{uuid.uuid4().hex[:8]}"

    logger.info(f"Ingesting {filename} (document_id={document_id})")

    vision_result = classify_document(image_path)
    doc_type = vision_result["doc_type"]
    logger.info(f"Vision classified as: {doc_type} (confidence={vision_result['confidence']:.3f})")

    ocr_result = extract_text(image_path, doc_type)
    text = ocr_result["text"]
    logger.info(f"OCR ({ocr_result['engine']}) extracted {len(text)} chars, confidence={ocr_result['confidence']:.3f}")

    if not text.strip():
        logger.warning(f"No text extracted from {filename} — skipping index addition.")
        return {
            "document_id": document_id, "filename": filename, "doc_type": doc_type,
            "text": "", "indexed": False,
        }

    # Load the existing index (built in Week 4) and append this
    # document's text as one new chunk. Same "short document = one
    # chunk" reasoning as index.py's chunk_document() — real documents
    # here are typically short enough not to need splitting; longer
    # real-world documents would need the same line-count-based
    # splitting logic from modules/language/index.py.
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    with open(f"{INDEX_DIR}/metadata.json") as f:
        metadata = json.load(f)

    # Guard against re-ingesting the same file (e.g. a user re-uploading,
    # a retried request) silently duplicating index entries — found as
    # a real bug via a failing test after this file was ingested twice
    # during development. See docs/ocr_model_notes.md.
    existing_filenames = {m["filename"] for m in metadata}
    if filename in existing_filenames:
        logger.warning(f"{filename} is already indexed — skipping duplicate ingestion.")
        return {
            "document_id": document_id, "filename": filename, "doc_type": doc_type,
            "text": text, "confidence": ocr_result["confidence"], "engine": ocr_result["engine"],
            "indexed": False, "reason": "duplicate_filename",
        }

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    
    new_embedding = embedder.encode([text], convert_to_numpy=True).astype("float32")
    index.add(new_embedding)

    chunks.append(text)
    metadata.append({"document_id": document_id, "filename": filename, "doc_type": doc_type})

    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(f"{INDEX_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Added to index. Index now contains {index.ntotal} chunks.")

    return {
        "document_id": document_id, "filename": filename, "doc_type": doc_type,
        "text": text, "confidence": ocr_result["confidence"], "engine": ocr_result["engine"],
        "indexed": True,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 -m modules.ocr.ingest path/to/image.jpg")
        sys.exit(1)

    result = ingest_document(sys.argv[1])
    print(json.dumps({k: v for k, v in result.items() if k != "text"}, indent=2))
    print(f"\nExtracted text:\n{result['text']}")