"""Tests for modules/analytics/predict.py"""

from modules.analytics.predict import predict


def test_predict_returns_expected_keys():
    result = predict({
        "page_count": 1,
        "file_size_kb": 300,
        "word_count": 30,
        "ocr_confidence": 0.9,
        "upload_hour": 10,
    })
    assert set(result.keys()) == {"category", "priority", "confidence"}


def test_predict_low_confidence_leans_high_priority():
    result = predict({
        "page_count": 1,
        "file_size_kb": 300,
        "word_count": 30,
        "ocr_confidence": 0.35,   # deliberately very low OCR confidence
        "upload_hour": 10,
    })
    assert result["priority"] == "high"


def test_predict_confidence_is_valid_probability():
    result = predict({
        "page_count": 2,
        "file_size_kb": 500,
        "word_count": 200,
        "ocr_confidence": 0.9,
        "upload_hour": 10,
    })
    assert 0.0 <= result["confidence"] <= 1.0