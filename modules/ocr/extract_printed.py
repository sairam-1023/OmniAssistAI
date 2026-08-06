"""
Printed-text OCR via Tesseract (pytesseract). Best suited to clean,
printed text (invoices, receipts, forms) — not handwriting, which
TrOCR (extract_handwritten.py) handles instead.

Interface:
    extract_printed_text(image_path: str) -> {text, confidence}
"""

import pytesseract
from PIL import Image

from modules.core.exceptions import OCRExtractionError
from modules.core.logging_config import get_logger
from modules.ocr.preprocess import preprocess_image

logger = get_logger(__name__)


def extract_printed_text(image_path: str) -> dict:
    try:
        preprocessed = preprocess_image(image_path)
        pil_image = Image.fromarray(preprocessed)

        # image_to_data gives per-word confidence scores, not just raw
        # text — lets us compute a genuine average confidence, same
        # spirit as Whisper's per-segment log-probs in Week 5.
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    except Exception as e:
        raise OCRExtractionError(f"Tesseract failed on {image_path}: {e}") from e

    words = []
    confidences = []
    for i, word in enumerate(data["text"]):
        conf = int(data["conf"][i])
        if word.strip() and conf >= 0:  # Tesseract uses -1 for non-text regions
            words.append(word)
            confidences.append(conf)

    text = " ".join(words)
    avg_confidence = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

    logger.info(f"Tesseract extracted {len(words)} words from {image_path}, avg_confidence={avg_confidence:.3f}")

    return {"text": text, "confidence": avg_confidence}