"""
OmniAssist AI — FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # see docs/ocr_model_notes.md

import shutil
import uuid
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.schemas import IngestResponse, QueryRequest, QueryResponse, SourceInfo, VoiceQueryResponse
from modules.core.config import get_settings
from modules.core.exceptions import OmniAssistError
from modules.core.logging_config import configure_logging, get_logger
from modules.ocr.ingest import ingest_document  # torch-based; import before faiss-touching modules
from modules.speech.synthesize import synthesize
from modules.speech.transcribe import transcribe

from modules.language.rag import answer_query  # noqa: E402 (faiss-touching; kept after torch imports)

configure_logging()
logger = get_logger(__name__)

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="A multimodal document, voice & knowledge assistant.",
    version="0.1.0",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

UPLOAD_DIR = Path("storage/uploads")
AUDIO_OUTPUT_DIR = Path("storage/audio_outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health_check():
    """Simple liveness check — used by deployment platforms and monitoring."""
    return {"status": "ok", "app": settings.app_name, "environment": settings.environment}


@app.post("/documents/upload", response_model=IngestResponse)
@limiter.limit("10/minute")
def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Uploads a document image, runs it through Vision classification and
    OCR extraction, and adds it to the searchable index.
    """
    file_ext = Path(file.filename).suffix
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{file_ext}"

    with open(saved_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_document(str(saved_path), filename=file.filename)
    except OmniAssistError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return IngestResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        doc_type=result["doc_type"],
        confidence=result.get("confidence", 0.0),
        engine=result.get("engine", "none"),
        indexed=result["indexed"],
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
def query_documents(request: Request, request_body: QueryRequest):
    """Answers a text question using RAG over indexed documents."""
    try:
        result = answer_query(request_body.query)
    except OmniAssistError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return QueryResponse(
        answer=result["answer"],
        intent=result["intent"],
        sources=[SourceInfo(**s) for s in result["sources"]],
    )


@app.post("/query/voice", response_model=VoiceQueryResponse)
@limiter.limit("10/minute")
def query_by_voice(request: Request, file: UploadFile = File(...)):
    """
    Accepts a spoken question as an audio file, transcribes it,
    answers it via RAG, and returns both the text answer and a URL
    to a spoken audio response.
    """
    audio_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.wav"
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        transcription = transcribe(str(audio_path))
        result = answer_query(transcription["text"])

        answer_audio_filename = f"{uuid.uuid4().hex}.mp3"
        answer_audio_path = AUDIO_OUTPUT_DIR / answer_audio_filename
        synthesize(result["answer"], str(answer_audio_path))
    except OmniAssistError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return VoiceQueryResponse(
        transcribed_text=transcription["text"],
        answer=result["answer"],
        intent=result["intent"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        audio_answer_url=f"/audio/{answer_audio_filename}",
    )


@app.get("/audio/{filename}")
def get_audio_answer(filename: str):
    """Serves a generated spoken-answer audio file."""
    from fastapi.responses import FileResponse

    path = AUDIO_OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/mpeg")