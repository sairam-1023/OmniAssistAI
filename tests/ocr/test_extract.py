"""Tests for modules/ocr/extract.py (routing) and extract_printed.py"""

from PIL import Image

from modules.ocr.extract import HANDWRITING_LIKELY_TYPES, extract_text
from modules.ocr.extract_printed import extract_printed_text


def test_extract_printed_returns_expected_keys(tmp_path):
    # Blank image — testing interface contract, not extraction accuracy,
    # same philosophy as Vision/Language/Speech's contract-only tests.
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (300, 400), color="white").save(img_path)

    result = extract_printed_text(str(img_path))
    assert set(result.keys()) == {"text", "confidence"}


def test_extract_printed_confidence_is_valid_range(tmp_path):
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (300, 400), color="white").save(img_path)

    result = extract_printed_text(str(img_path))
    assert 0.0 <= result["confidence"] <= 1.0


def test_routing_sends_printed_types_to_tesseract(tmp_path):
    img_path = tmp_path / "blank.png"
    Image.new("RGB", (300, 400), color="white").save(img_path)

    result = extract_text(str(img_path), doc_type="invoice")
    assert result["engine"] == "tesseract"


def test_handwriting_likely_types_are_correctly_defined():
    # Guards the routing logic's core assumption — if this set ever
    # changes, it's a deliberate decision, not an accidental edit.
    assert "prescription" in HANDWRITING_LIKELY_TYPES
    assert "invoice" not in HANDWRITING_LIKELY_TYPES