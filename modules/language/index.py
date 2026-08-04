"""
Builds a FAISS vector index over document text chunks, using
sentence-transformers for embeddings. This is the "retrieval" half
of RAG: given a query, find the most semantically relevant chunks.

Usage:
    python3 -m modules.language.index
"""

import json
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from data.documents.sample_docs import SAMPLE_DOCUMENTS
from modules.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, well-regarded general embedding model
INDEX_DIR = "models/language/vector_index"


def chunk_document(text: str) -> list[str]:
    """
    Splits document text into chunks. For short documents (our mock
    invoices/receipts/forms are all under ~20 lines), splitting at all
    risks separating related facts (e.g., "Due Date" from "Total Due")
    into different chunks that can't both be retrieved together. So we
    only split if a document exceeds a generous line threshold —
    otherwise the whole document is one chunk.

    Real OCR'd documents (Week 6) may be much longer and will need
    smarter, semantic-boundary-aware chunking rather than this
    line-count heuristic.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    max_lines_per_chunk = 20  # generous — larger than any current mock document
    if len(lines) <= max_lines_per_chunk:
        return ["\n".join(lines)]

    chunks = []
    for i in range(0, len(lines), max_lines_per_chunk):
        chunk_text = "\n".join(lines[i:i + max_lines_per_chunk])
        chunks.append(chunk_text)
    return chunks


def build_index():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded embedding model: {EMBEDDING_MODEL_NAME}")

    all_chunks = []       # the actual text of each chunk
    chunk_metadata = []   # which document/doc_type each chunk came from

    for doc in SAMPLE_DOCUMENTS:
        chunks = chunk_document(doc["text"])
        for chunk_text in chunks:
            all_chunks.append(chunk_text)
            chunk_metadata.append({
                "document_id": doc["document_id"],
                "filename": doc["filename"],
                "doc_type": doc["doc_type"],
            })

    logger.info(f"Total chunks to index: {len(all_chunks)} (from {len(SAMPLE_DOCUMENTS)} documents)")

    # Embed all chunks at once — more efficient than one at a time.
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")  # FAISS requires float32

    # FAISS index: IndexFlatL2 does exact (brute-force) nearest-neighbor
    # search using Euclidean distance. Simple and exact — the right
    # choice at our scale (a few thousand chunks); larger-scale systems
    # use approximate indexes (e.g., IndexIVFFlat) that trade a little
    # accuracy for speed.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info(f"Built FAISS index: {index.ntotal} vectors, dimension {dimension}")

    # Persist everything needed to search later: the index itself,
    # the raw chunk text (FAISS only stores vectors, not the text),
    # and metadata for each chunk.
    import os
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(all_chunks, f)
    with open(f"{INDEX_DIR}/metadata.json", "w") as f:
        json.dump(chunk_metadata, f, indent=2)

    logger.info(f"Saved index and chunk data to {INDEX_DIR}/")


if __name__ == "__main__":
    build_index()