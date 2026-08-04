"""
The RAG (Retrieval-Augmented Generation) pipeline: retrieves relevant
document chunks via semantic search, then asks an LLM to answer the
user's question using only that retrieved context, with source
citations.

Uses Groq's API (free tier, OpenAI-compatible client shape) running
Llama 3.3 70B.

Interface:
    answer_query(query: str, top_k: int = 3) -> dict
        {"answer": str, "sources": list[dict]}
"""

from groq import Groq

from modules.core.config import get_settings
from modules.core.exceptions import OmniAssistError
from modules.core.logging_config import get_logger
from modules.language.search import search

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are OmniAssist AI, a document assistant. Answer the user's question "
    "using ONLY the document excerpts provided below. "
    "If the excerpts don't contain enough information to answer, say so clearly "
    "instead of guessing. Always mention which document(s) your answer is based on."
)


def _build_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a labeled context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk['filename']} ({chunk['doc_type']})]\n{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def answer_query(query: str, top_k: int = 3) -> dict:
    settings = get_settings()
    if not settings.groq_api_key:
        raise OmniAssistError("GROQ_API_KEY is not set in .env — cannot call the LLM.")

    retrieved_chunks = search(query, top_k=top_k)
    context = _build_context(retrieved_chunks)

    user_prompt = (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the excerpts above. Cite the source filename(s) you used."
    )

    client = Groq(api_key=settings.groq_api_key)

    logger.info(f"Calling Groq for query: {query!r} with {len(retrieved_chunks)} retrieved chunks")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # strong open-source model, free tier, fast LPU inference
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature: favor consistent, grounded answers over creativity
    )

    answer_text = response.choices[0].message.content

    sources = [
        {"filename": c["filename"], "document_id": c["document_id"], "doc_type": c["doc_type"]}
        for c in retrieved_chunks
    ]

    return {"answer": answer_text, "sources": sources}