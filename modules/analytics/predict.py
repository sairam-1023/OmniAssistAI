"""
Loads trained Analytics models and exposes a clean predict() interface.

Interface (per project plan, Section 3.2):
    predict(features: dict) -> {category, priority, confidence}
"""

from functools import lru_cache

import joblib
import pandas as pd

from modules.core.config import get_settings
from modules.core.exceptions import ModelNotLoadedError
from modules.core.logging_config import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = ["page_count", "file_size_kb", "word_count", "ocr_confidence", "upload_hour"]


@lru_cache
def _load_models():
    """
    Loads both trained models once and caches them in memory — avoids
    re-reading the .joblib files from disk on every single prediction.
    """
    try:
        category_model = joblib.load("models/analytics/category_model.joblib")
        priority_model = joblib.load("models/analytics/priority_model.joblib")
    except FileNotFoundError as e:
        raise ModelNotLoadedError(
            f"Analytics model file not found: {e}. Run `python3 -m modules.analytics.train` first."
        ) from e
    return category_model, priority_model


def predict(features: dict) -> dict:
    """
    features: dict with keys page_count, file_size_kb, word_count,
              ocr_confidence, upload_hour.
    Returns: {"category": str, "priority": str, "confidence": float}
    """
    category_model, priority_model = _load_models()

    # Build a single-row DataFrame — scikit-learn expects 2D input,
    # even for one prediction, and column order must match training.
    X = pd.DataFrame([features])[FEATURE_COLUMNS]

    category_pred = category_model.predict(X)[0]
    priority_pred = priority_model.predict(X)[0]

    # predict_proba gives per-class probabilities; we take the max as
    # a simple confidence score for the priority prediction (the one
    # this module is primarily responsible for).
    priority_proba = priority_model.predict_proba(X)[0]
    confidence = float(priority_proba.max())

    logger.info(f"Predicted category={category_pred}, priority={priority_pred}, confidence={confidence:.3f}")

    return {"category": category_pred, "priority": priority_pred, "confidence": confidence}