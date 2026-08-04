"""
The RAG (Retrieval-Augmented Generation) pipeline: retrieves relevant
document chunks via semantic search, then asks an LLM to answer the
user's question using only that retrieved context, with source
citations.

Uses the intent classifier to choose how many chunks to retrieve
(top_k) based on the TYPE of question being asked, rather than a
single fixed top_k for every query — see docs/rag_model_notes.md for
why this matters (queries needing facts spread across multiple chunks
were silently losing information under a fixed top_k).

Uses Groq's API (free tier, OpenAI-compatible client shape) running
Llama 3.3 70B.

Interface:
    answer_query(query: str) -> dict
        {"answer": str, "sources": list[dict], "intent": str}
"""

from groq import Groq

from modules.core.config import get_settings
from modules.core.exceptions import OmniAssistError
from modules.core.logging_config import get_logger
from modules.language.predict import predict as predict_intent
from modules.language.search import search

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are OmniAssist AI, a document assistant. Answer the user's question "
    "using ONLY the document excerpts provided below. "
    "If the excerpts don't contain enough information to answer, say so clearly "
    "instead of guessing. Always mention which document(s) your answer is based on."
)

GENERAL_CHAT_SYSTEM_PROMPT = (
    "You are OmniAssist AI, a friendly document assistant. Respond briefly and "
    "naturally. You don't have document context for this message — if the user "
    "seems to be asking about a specific document, gently prompt them to ask a "
    "document-related question."
)

# How many chunks to retrieve, based on the classified intent.
# document_qa: usually one specific fact, small buffer for retrieval imperfection.
# summarize: needs broad coverage of a document's content.
# search_documents: deliberately wide net across many documents.
# general_chat: retrieval skipped entirely (see answer_query).
TOP_K_BY_INTENT = {
    "document_qa": 3,
    "summarize": 5,
    "search_documents": 8,
}


def _build_context(chunks: list[dict]) -> str:
    """Formats retrieved chunks into a labeled context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}: {chunk['filename']} ({chunk['doc_type']})]\n{chunk['chunk_text']}"
        )
    return "\n\n".join(parts)


def _call_groq(system_prompt: str, user_prompt: str, api_key: str) -> str:
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def answer_query(query: str) -> dict:
    settings = get_settings()
    if not settings.groq_api_key:
        raise OmniAssistError("GROQ_API_KEY is not set in .env — cannot call the LLM.")

    intent_result = predict_intent(query)
    intent = intent_result["intent"]
    logger.info(f"Routed query {query!r} -> intent={intent} (confidence={intent_result['confidence']:.3f})")

    if intent == "general_chat":
        # No document retrieval at all — this is the "skip retrieval
        # entirely" case discussed: nothing document-related to search for.
        answer_text = _call_groq(GENERAL_CHAT_SYSTEM_PROMPT, query, settings.groq_api_key)
        return {"answer": answer_text, "sources": [], "intent": intent}

    top_k = TOP_K_BY_INTENT.get(intent, 3)  # default fallback, e.g. if a new intent is added later
    retrieved_chunks = search(query, top_k=top_k)
    context = _build_context(retrieved_chunks)

    user_prompt = (
        f"Document excerpts:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer the question using only the excerpts above. Cite the source filename(s) you used."
    )

    logger.info(f"Calling Groq for query: {query!r} with {len(retrieved_chunks)} retrieved chunks (top_k={top_k})")
    answer_text = _call_groq(SYSTEM_PROMPT, user_prompt, settings.groq_api_key)

    sources = [
        {"filename": c["filename"], "document_id": c["document_id"], "doc_type": c["doc_type"]}
        for c in retrieved_chunks
    ]

    return {"answer": answer_text, "sources": sources, "intent": intent}