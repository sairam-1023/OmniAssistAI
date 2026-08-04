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

## Secondary observation (not yet fixed)
`search()` occasionally returns the same document's chunk twice within
one `top_k` result set (observed in the buggy run above, both slots 2
and 3 were `electric_bill_invoice.png`). Worth investigating once real,
larger document sets exist — may just reflect having only 6 documents
total right now, giving few genuinely distinct chunks to fill top_k=3.

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