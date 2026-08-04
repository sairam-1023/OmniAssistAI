from modules.language.search import search

test_queries = [
    "how much do I owe on the invoice",       # no doc says "owe" — doc says "Total Due"
    "what medication was I prescribed",        # doc says "Medication:", not "prescribed"
    "coffee shop purchase",                    # doc never says "coffee shop", just "Blue Bottle Coffee"
]

for q in test_queries:
    print(f"\nQuery: {q!r}")
    results = search(q, top_k=2)
    for r in results:
        print(f"  [{r['doc_type']}] {r['filename']} (distance={r['distance']:.3f})")
        print(f"    {r['chunk_text'][:80]}...")