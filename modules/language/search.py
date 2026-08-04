"""
Semantic search over the FAISS document index: given a query, returns
the most relevant text chunks. This is retrieval only — no LLM
involved yet, that's modules/language/rag.py.

Interface:
    search(query: str, top_k: int = 3) -> list[dict]
"""

import json
import pickle
from functools import lru_cache

import faiss
from sentence_transformers import SentenceTransformer

from modules.core.exceptions import ModelNotLoadedError
from modules.core.logging_config import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_DIR = "models/language/vector_index"


@lru_cache
def _load_index():
    try:
        index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
        with open(f"{INDEX_DIR}/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(f"{INDEX_DIR}/metadata.json") as f:
            metadata = json.load(f)
    except (FileNotFoundError, RuntimeError) as e:
        raise ModelNotLoadedError(
            f"Vector index not found at {INDEX_DIR}. Run `python3 -m modules.language.index` first."
        ) from e

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return index, chunks, metadata, embedder


def search(query: str, top_k: int = 3) -> list[dict]:
    """
    Returns the top_k most semantically relevant chunks for the query,
    each with its source document metadata and a distance score
    (lower distance = more similar).
    """
    index, chunks, metadata, embedder = _load_index()

    query_vector = embedder.encode([query], convert_to_numpy=True).astype("float32")

    # FAISS search returns two arrays: distances and indices of the
    # top_k nearest vectors to our query vector.
    distances, indices = index.search(query_vector, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "chunk_text": chunks[idx],
            "distance": float(distances[0][rank]),
            "document_id": metadata[idx]["document_id"],
            "filename": metadata[idx]["filename"],
            "doc_type": metadata[idx]["doc_type"],
        })

    logger.info(f"Search query: {query!r} -> {len(results)} results")
    return results