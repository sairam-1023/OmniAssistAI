"""
Handwritten-text OCR via Gemini's multimodal API. Replaces an earlier
TrOCR-based approach — TrOCR (trained on single cropped text lines)
struggled badly with real, messy multi-line handwriting (prescription
samples), including repetition loops and fluent-sounding hallucination
even on controlled, clearly-written test input. See
docs/ocr_model_notes.md for the full comparison.

Gemini's multimodal understanding handles full document images
directly — no line-segmentation preprocessing needed.

Interface:
    extract_handwritten_text(image_path: str) -> {text, confidence}
"""

import json

from google import genai
from google.genai import types

from modules.core.config import get_settings
from modules.core.exceptions import OCRExtractionError
from modules.core.logging_config import get_logger

logger = get_logger(__name__)

MODEL_NAME = "gemini-3.6-flash"

PROMPT = (
    "Read all handwritten and printed text in this image exactly as written. "
    "Respond ONLY with valid JSON in this exact format, no other text: "
    '{"text": "<all text found, preserving line breaks as \\n>", '
    '"confidence": <your confidence this transcription is fully accurate, 0.0 to 1.0>}'
)


def extract_handwritten_text(image_path: str) -> dict:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise OCRExtractionError("GEMINI_API_KEY is not set in .env — cannot call Gemini.")

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[PROMPT, image_part],
        )

        # Gemini sometimes wraps JSON in markdown code fences despite
        # instructions not to — strip those defensively before parsing.
        raw_text = response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        parsed = json.loads(raw_text)
        text = parsed.get("text", "")
        confidence = float(parsed.get("confidence", 0.0))

    except Exception as e:
        raise OCRExtractionError(f"Gemini handwriting extraction failed on {image_path}: {e}") from e

    logger.info(f"Gemini extracted text from {image_path}, self-reported confidence={confidence:.3f}")
    return {"text": text, "confidence": confidence}