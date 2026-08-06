"""
Pydantic models defining the shape of API requests and responses.
FastAPI uses these to validate incoming data automatically and to
generate the /docs documentation.
"""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    doc_type: str
    confidence: float
    engine: str
    indexed: bool


class QueryRequest(BaseModel):
    query: str


class SourceInfo(BaseModel):
    filename: str
    document_id: str
    doc_type: str


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list[SourceInfo]


class VoiceQueryResponse(BaseModel):
    transcribed_text: str
    answer: str
    intent: str
    sources: list[SourceInfo]
    audio_answer_url: str