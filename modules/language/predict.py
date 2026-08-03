"""
Loads the fine-tuned intent classifier and exposes a clean predict()
interface.

Interface (per project plan, Section 3.4):
    predict(query_text: str) -> {intent, confidence}
"""

import json
from functools import lru_cache

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from modules.core.exceptions import ModelNotLoadedError
from modules.core.logging_config import get_logger

logger = get_logger(__name__)

MODEL_PATH = "models/language/intent_classifier"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@lru_cache
def _load_model():
    try:
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        with open(f"{MODEL_PATH}/label_map.json") as f:
            label_map = json.load(f)
    except OSError as e:
        raise ModelNotLoadedError(
            f"Intent classifier not found at {MODEL_PATH}. "
            f"Run `python3 -m modules.language.train_intent` first."
        ) from e

    model.eval()
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    return model, tokenizer, id2label


def predict(query_text: str) -> dict:
    model, tokenizer, id2label = _load_model()

    inputs = tokenizer(query_text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]
        pred_id = probs.argmax().item()
        confidence = probs[pred_id].item()

    intent = id2label[pred_id]
    logger.info(f"Predicted intent={intent}, confidence={confidence:.3f} for query: {query_text!r}")

    return {"intent": intent, "confidence": confidence}