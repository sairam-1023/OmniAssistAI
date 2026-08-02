"""
Custom exception hierarchy for OmniAssist AI.

Every module-specific error subclasses OmniAssistError, so calling code
(especially the FastAPI layer in Week 7) can catch broad or narrow as
needed, and so each error type can be mapped to the right HTTP status.
"""


class OmniAssistError(Exception):
    """Base class for every custom error in this project."""


class ModelNotLoadedError(OmniAssistError):
    """Raised when an ML model file is missing or fails to load."""


class DocumentNotFoundError(OmniAssistError):
    """Raised when a requested document ID doesn't exist."""


class OCRExtractionError(OmniAssistError):
    """Raised when OCR processing fails on a document."""


class ClassificationError(OmniAssistError):
    """Raised when the Vision or Analytics classifier fails to produce a result."""


class TranscriptionError(OmniAssistError):
    """Raised when speech-to-text fails."""


class SynthesisError(OmniAssistError):
    """Raised when text-to-speech fails."""



    