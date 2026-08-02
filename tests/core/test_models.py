"""Tests for modules/core/models.py"""

import pytest
from pydantic import ValidationError

from modules.core.models import AnalysisResult, Document, DocumentType


def test_document_creates_with_valid_data():
    doc = Document(id="1", filename="invoice.png", doc_type=DocumentType.INVOICE)
    assert doc.id == "1"
    assert doc.doc_type == DocumentType.INVOICE
    assert doc.ocr_text is None  # optional field, defaults to None


def test_document_rejects_invalid_doc_type():
    with pytest.raises(ValidationError):
        Document(id="1", filename="invoice.png", doc_type="not_a_real_type")


def test_analysis_result_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        AnalysisResult(
            document_id="1", category="invoice", priority="high", confidence=1.5
        )


def test_analysis_result_accepts_valid_confidence():
    result = AnalysisResult(
        document_id="1", category="invoice", priority="high", confidence=0.87
    )
    assert result.confidence == 0.87