from modules.language.rag import answer_query

test_queries = [
    "How much do I owe on the Acme invoice, and when is it due?",  # document_qa -> top_k=3
    "Summarize the patient intake form.",                           # summarize -> top_k=5
    "Find all documents related to invoices or bills.",             # search_documents -> top_k=8
    "Hi, what can you help me with?",   
    " list all file names",                            # general_chat -> no retrieval
]

for q in test_queries:
    result = answer_query(q)
    print(f"\nQuery: {q!r}")
    print(f"Intent: {result['intent']}")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {[s['filename'] for s in result['sources']]}")