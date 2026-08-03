from modules.language.predict import predict

test_queries = [
    "What is the total on this bill?",
    "Can you TL;DR this for me?",
    "Pull up every form from this week",
    "yo whats good",
    "how much did i pay",
]

for q in test_queries:
    result = predict(q)
    print(f"{q!r} -> {result}")