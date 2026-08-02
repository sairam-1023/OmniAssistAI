"""
Domain models shared across every OmniAssist AI module.

These are the "nouns" of the system: Document, MediaFile, UserQuery,
AnalysisResult. Every module (vision, ocr, nlp, speech, analytics)
passes these objects around instead of raw dicts, so data shape is
validated once, at creation time, rather than trusted blindly deep
inside the system.
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """The document categories the Vision module can classify into."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    ID_CARD = "id_card"
    FORM = "form"
    PRESCRIPTION = "prescription"
    GENERIC_PHOTO = "generic_photo"


class Document(BaseModel):
    """A single uploaded file, tracked through its whole lifecycle."""
    id: str
    filename: str
    doc_type: DocumentType | None = None
    ocr_text: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MediaFile(BaseModel):
    """A raw uploaded file before/independent of OCR processing."""
    id: str
    original_filename: str
    content_type: str          # e.g. "image/png", "application/pdf"
    size_bytes: int
    storage_path: str          # where it lives on disk


class UserQuery(BaseModel):
    """A single question a user asked, and how it was asked."""
    id: str
    document_id: str | None = None   # None = general chat, not doc-specific
    query_text: str
    via_voice: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AnalysisResult(BaseModel):
    """Output of the Week-2 Analytics classifier for one document."""
    document_id: str
    category: str
    priority: str
    confidence: float = Field(ge=0.0, le=1.0)
    predicted_at: datetime = Field(default_factory=datetime.utcnow)