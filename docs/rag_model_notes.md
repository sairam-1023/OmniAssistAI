# Language / RAG Module — Notes (Week 4)

## Components
- Intent classifier: DistilBERT, 4 intents (see language_model_notes.md)
- Embedding model: all-MiniLM-L6-v2 (sentence-transformers), 384-dim vectors
- Vector store: FAISS IndexFlatL2, exact nearest-neighbor search
- Generation: Groq API, llama-3.3-70b-versatile (free tier, OpenAI-compatible client)

## Semantic search validation
Tested with queries sharing no exact words with target document text
(e.g., "how much do I owe" -> found "Total Due: $4,050.00"; "coffee
shop purchase" -> found Blue Bottle Coffee receipt). Confirms embeddings
are capturing meaning, not just keyword overlap.

## Known bug found and fixed: chunk boundary fact-splitting
Initial chunking strategy (fixed 4-line groups) split short documents
mid-fact — e.g., the Acme invoice's "Due Date" and "Total Due" landed
in different chunks. A query asking for both retrieved only the chunk
containing neither, and the LLM correctly refused to guess rather than
hallucinate ("the excerpts don't contain enough information").

Fix: for documents under 20 lines (all current mock documents), index
the entire document as one chunk instead of splitting. After the fix,
the same query correctly retrieved the full invoice in one chunk and
answered both facts correctly with accurate source citation.

## Bug found and fixed: top_k exceeding index size
After adding intent-aware top_k routing (search_documents -> top_k=8),
duplicate documents started appearing in results (e.g., drivers_license.png
listed 3 times in one response). Root cause: the index only contains 6
chunks total (one per short mock document, per the earlier chunking fix),
but top_k=8 requested more neighbors than exist. FAISS's IndexFlatL2
doesn't error in this case — it silently returns duplicate/padded matches
to fill the requested count.

Fix: search() now caps top_k at index.ntotal (the actual number of
indexed chunks), logging a warning when the cap is applied. Verified:
search_documents queries now return 6 unique sources with no duplicates,
and the warning fires exactly when top_k > available chunks.

This bug will naturally recur, differently, once the corpus grows in
Week 6+ — worth re-testing this cap logic once real documents make
top_k=8 a reasonable, satisfiable request rather than an edge case.

## Intent-aware top_k routing
answer_query() now calls the intent classifier first, then chooses
top_k based on the classified intent, rather than a single fixed
top_k for every query:
  - document_qa: 3 (specific fact, usually in one chunk)
  - summarize: 5 (broader document coverage)
  - search_documents: 8 (wide net across documents)
  - general_chat: retrieval skipped entirely

This directly addresses the earlier concern that a fixed top_k could
silently omit chunks needed to fully answer broader queries.

## Provider choice: Groq over OpenAI
Started with OpenAI (gpt-4o-mini) but hit `insufficient_quota` — the
API requires billing setup separate from a ChatGPT subscription.
Switched to Groq (free tier, no card required, OpenAI-compatible
client), running Llama 3.3 70B. Swapping providers required changing
only the client import, instantiation, and model name — the RAG logic
itself (prompt construction, grounding instructions) was untouched,
confirming that isolating "call the LLM" behind one function was a
good design choice.

## Known limitation
All testing so far uses 6 hand-written mock documents. Real OCR'd
documents (Week 6) will likely be longer and messier, requiring a
smarter chunking strategy (by sentence or semantic section) than the
current line-count heuristic.