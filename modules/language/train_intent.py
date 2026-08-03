"""
Fine-tunes DistilBERT to classify a user query into one of four intents:
document_qa, summarize, search_documents, general_chat.

Usage:
    python3 -m modules.language.train_intent
"""

import json

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data.intents.training_data import INTENT_EXAMPLES
from modules.core.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

MODEL_NAME = "distilbert-base-uncased"
INTENTS = sorted(INTENT_EXAMPLES.keys())  # fixed, alphabetical order — must match predict.py
LABEL2ID = {label: i for i, label in enumerate(INTENTS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build_dataset():
    """Flattens INTENT_EXAMPLES into (text, label_id) pairs, then splits train/val."""
    texts, labels = [], []
    for intent, examples in INTENT_EXAMPLES.items():
        for text in examples:
            texts.append(text)
            labels.append(LABEL2ID[intent])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return train_texts, val_texts, train_labels, val_labels


def tokenize_dataset(texts, labels, tokenizer):
    """
    Tokenizes a list of raw strings into DistilBERT's expected input format,
    and wraps it as a HuggingFace Dataset object that Trainer can consume.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
    dataset = Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })
    return dataset


def compute_metrics(eval_pred):
    """Called by Trainer after each evaluation pass, to report accuracy/F1."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }


def main():
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Intents (fixed order): {INTENTS}")

    train_texts, val_texts, train_labels, val_labels = build_dataset()
    logger.info(f"Train examples: {len(train_texts)}, Val examples: {len(val_texts)}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = tokenize_dataset(train_texts, train_labels, tokenizer)
    val_dataset = tokenize_dataset(val_texts, val_labels, tokenizer)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(INTENTS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir="models/language/intent_checkpoints",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",   # we'll save the final model manually below
        logging_strategy="epoch",
        learning_rate=2e-5,
        report_to=[],  # disable auto-integrations (e.g. wandb); we use MLflow manually
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    final_metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {final_metrics}")

    # Save the fine-tuned model + tokenizer together — predict.py needs both.
    save_path = "models/language/intent_classifier"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save the label mapping explicitly too, so predict.py doesn't need
    # to re-import training_data.py just to know the label order.
    with open(f"{save_path}/label_map.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)

    logger.info(f"Saved intent classifier to {save_path}")


if __name__ == "__main__":
    main()
