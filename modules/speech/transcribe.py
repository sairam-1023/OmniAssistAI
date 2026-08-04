"""
Speech-to-text via OpenAI's Whisper (runs locally, no API key needed).

Interface (per project plan, Section 3.5):
    transcribe(audio_path: str) -> {text, language, confidence}
"""

from functools import lru_cache

import whisper

from modules.core.exceptions import TranscriptionError
from modules.core.logging_config import get_logger

logger = get_logger(__name__)

WHISPER_MODEL_SIZE = "small"


@lru_cache
def _load_model():
    logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} (first call only, cached after)")
    return whisper.load_model(WHISPER_MODEL_SIZE)


def transcribe(audio_path: str) -> dict:
    model = _load_model()

    try:
        result = model.transcribe(audio_path)
    except Exception as e:
        raise TranscriptionError(f"Failed to transcribe {audio_path}: {e}") from e

    # Whisper's segments include per-segment log-probabilities; we
    # average them as a rough overall confidence signal. Not a true
    # probability (log-probs aren't bounded [0,1]), so we convert via
    # exponentiation - a common practical approximation.
    segments = result.get("segments", [])
    if segments:
        avg_logprob = sum(s["avg_logprob"] for s in segments) / len(segments)
        confidence = min(1.0, max(0.0, pow(2.718281828, avg_logprob)))  # e^avg_logprob, clamped
    else:
        confidence = 0.0

    text = result["text"].strip()
    language = result.get("language", "unknown")

    logger.info(f"Transcribed {audio_path}: language={language}, confidence={confidence:.3f}")
    logger.info(f"Text: {text!r}")

    return {"text": text, "language": language, "confidence": confidence}