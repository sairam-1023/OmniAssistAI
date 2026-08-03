"""Tests for modules/language/predict.py (intent classifier)"""

from modules.language.predict import predict


def test_predict_returns_expected_keys():
    result = predict("What is the total on this invoice?")
    assert set(result.keys()) == {"intent", "confidence"}


def test_predict_returns_valid_intent():
    valid_intents = {"document_qa", "summarize", "search_documents", "general_chat"}
    result = predict("Summarize this document please.")
    assert result["intent"] in valid_intents


def test_predict_confidence_is_valid_probability():
    result = predict("Hello!")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_clear_document_qa_query():
    # Unambiguous, high-signal example — reasonable to assert the exact label
    result = predict("What is the invoice number on this document?")
    assert result["intent"] == "document_qa"