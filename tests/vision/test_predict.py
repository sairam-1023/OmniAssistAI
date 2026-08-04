"""Tests for modules/vision/predict.py"""

from PIL import Image

from modules.vision.predict import predict
from modules.vision.train import CATEGORIES


def test_predict_returns_expected_keys(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (300, 400), color="white").save(img_path)

    result = predict(str(img_path))
    assert set(result.keys()) == {"doc_type", "confidence"}


def test_predict_returns_valid_category(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (300, 400), color="white").save(img_path)

    result = predict(str(img_path))
    assert result["doc_type"] in CATEGORIES


def test_predict_confidence_is_valid_probability(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (300, 400), color="white").save(img_path)

    result = predict(str(img_path))
    assert 0.0 <= result["confidence"] <= 1.0