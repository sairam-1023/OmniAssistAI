"""
Routes a document to the right OCR engine based on its Vision-classified
doc_type: printed-text documents go to Tesseract (fast, free, local,
strong on clean printed text); handwriting-prone documents go to
Gemini (better real-world handwriting understanding — see
docs/ocr_model_notes.md for why TrOCR was replaced).

Interface (per project plan, Section 3.6):
    extract_text(image_path: str, doc_type: str) -> {text, confidence, engine}
"""

from modules.core.logging_config import get_logger
from modules.ocr.extract_handwritten import extract_handwritten_text
from modules.ocr.extract_printed import extract_printed_text

logger = get_logger(__name__)

# Document types where handwriting is likely/possible enough to route
# to Gemini instead of Tesseract. Based on Vision's DocumentType enum
# (Week 1/3) — prescriptions and filled-in forms are the categories
# most likely to contain real handwriting; invoices/receipts/id_cards
# are reliably printed/typed.
HANDWRITING_LIKELY_TYPES = {"prescription", "form"}


def extract_text(image_path: str, doc_type: str) -> dict:
    if doc_type in HANDWRITING_LIKELY_TYPES:
        logger.info(f"Routing {image_path} (doc_type={doc_type}) -> Gemini (handwriting path)")
        result = extract_handwritten_text(image_path)
        engine = "gemini"
    else:
        logger.info(f"Routing {image_path} (doc_type={doc_type}) -> Tesseract (printed path)")
        result = extract_printed_text(image_path)
        engine = "tesseract"

    return {"text": result["text"], "confidence": result["confidence"], "engine": engine}