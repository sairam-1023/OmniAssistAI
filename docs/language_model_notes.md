# Language Module — Intent Classifier Notes (Week 4)

## Model
DistilBERT (`distilbert-base-uncased`), fine-tuned classification head
for 4 intents: document_qa, summarize, search_documents, general_chat.

## Training data
- v1: ~65 hand-written example utterances (13-20 per intent).
- v2: ~120 examples (~30 per intent), after v1 was found to generalize
  poorly — see below. Added casual, abbreviated, and lowercase phrasing
  deliberately, not just more formal variations.

## v1 results (before fix)
Validation accuracy: 84.6%, F1-macro: 0.81 — looked strong, but this
was measured on data sharing the same shallow surface patterns as
training data (similar sentence structure/formality per class).

Stress-tested against 5 novel, differently-phrased queries (not
present in training data): **3/5 correct**, with weak confidence even
on correct predictions (0.33-0.65) — barely above the 0.25 floor of
random guessing among 4 classes. Notably failed on "Can you TL;DR this
for me?" despite "TL;DR this invoice." being an actual training
example — indicating memorization of surface patterns rather than
semantic understanding of "TL;DR" = summarize intent.

## v2 results (after adding phrasing diversity)
Same 5 stress-test queries: **5/5 correct**, confidence improved
across the board (0.49-0.81). The previously-failed TL;DR case now
correctly predicts `summarize`, though still the least confident
(0.49) — a genuinely ambiguous, borderline case even for a human
(could plausibly be read as casual chat).

## Known limitation
Total training data (~120 examples) is still small for a transformer
model. Borderline/ambiguous phrasings will likely still be
misclassified or low-confidence. Production logging of real user
queries (once live) should be used to continuously expand this
dataset — the model will only get more robust with more real,
diverse examples.