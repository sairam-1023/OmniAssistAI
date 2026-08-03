"""
Loads the trained Vision model and exposes a clean predict() interface
for a single image path.

Interface (per project plan, Section 3.3):
    predict(image_path: str) -> {doc_type, confidence}
"""

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from modules.core.exceptions import ModelNotLoadedError
from modules.core.logging_config import get_logger
from modules.vision.train import CATEGORIES, DEVICE, build_model, val_transform

logger = get_logger(__name__)

MODEL_PATH = "models/vision/resnet18_finetuned.pt"


@lru_cache
def _load_model():
    model = build_model(num_classes=len(CATEGORIES))
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError as e:
        raise ModelNotLoadedError(
            f"Vision model not found at {MODEL_PATH}. Run `python3 -m modules.vision.train` first."
        ) from e
    model.eval()
    return model


def predict(image_path: str) -> dict:
    model = _load_model()

    image = np.array(Image.open(image_path).convert("RGB"))
    transformed = val_transform(image=image)["image"]
    batch = transformed.unsqueeze(0).to(DEVICE)  # add batch dimension: [3,224,224] -> [1,3,224,224]

    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

    doc_type = CATEGORIES[pred_idx]
    logger.info(f"Predicted doc_type={doc_type}, confidence={confidence:.3f} for {image_path}")

    # Also log the full probability breakdown — useful for debugging
    # low-confidence or out-of-distribution predictions.
    all_probs = {cat: round(probs[i].item(), 3) for i, cat in enumerate(CATEGORIES)}
    logger.info(f"Full probability breakdown: {all_probs}")

    return {"doc_type": doc_type, "confidence": confidence}