"""
Generates a synthetic dataset for training the Analytics classifier.

Real uploaded-document metadata will replace this later — the point
right now is to have realistic, learnable data so we can build and
test the training pipeline honestly.
"""

import numpy as np
import pandas as pd

np.random.seed(42)  # reproducibility: same "random" data every run

N = 800
categories = ["invoice", "receipt", "id_card", "form", "prescription", "generic_photo"]

rows = []
for _ in range(N):
    category = np.random.choice(categories)

    # Feature generation, loosely realistic per category
    if category == "invoice":
        page_count = np.random.randint(1, 5)
        word_count = np.random.randint(150, 600)
    elif category == "receipt":
        page_count = 1
        word_count = np.random.randint(20, 120)
    elif category == "id_card":
        page_count = 1
        word_count = np.random.randint(10, 40)
    elif category == "form":
        page_count = np.random.randint(1, 3)
        word_count = np.random.randint(80, 300)
    elif category == "prescription":
        page_count = 1
        word_count = np.random.randint(15, 80)
    else:  # generic_photo
        page_count = 1
        word_count = np.random.randint(0, 20)

    file_size_kb = page_count * np.random.randint(150, 500)
    ocr_confidence = np.clip(np.random.normal(0.85, 0.12), 0.3, 0.99)
    upload_hour = np.random.randint(0, 24)

    # Priority logic: low OCR confidence or handwritten-prone types → higher priority for human review
    risk_score = (1 - ocr_confidence) + (0.3 if category in ("form", "prescription") else 0)
    if risk_score > 0.55:
        priority = "high"
    elif risk_score > 0.3:
        priority = "medium"
    else:
        priority = "low"

    rows.append({
        "page_count": page_count,
        "file_size_kb": file_size_kb,
        "word_count": word_count,
        "ocr_confidence": round(ocr_confidence, 3),
        "upload_hour": upload_hour,
        "category": category,
        "priority": priority,
    })

df = pd.DataFrame(rows)
df.to_csv("data/sample_documents.csv", index=False)
print(f"Wrote {len(df)} rows to data/sample_documents.csv")
print(df.head())
print("\nCategory distribution:\n", df["category"].value_counts())
print("\nPriority distribution:\n", df["priority"].value_counts())