from modules.language.rag import answer_query

result = answer_query("How much do I owe on the Acme invoice, and when is it due?")
print("ANSWER:", result["answer"])
print("\nSOURCES:", result["sources"])