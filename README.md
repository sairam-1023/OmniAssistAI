# OmniAssist AI

A multimodal document, voice & knowledge assistant. Upload a scanned
document, image, or PDF — OmniAssist classifies it, extracts and
structures its text (including handwriting), indexes it for semantic
search, and lets you ask questions about it by typing or speaking.

Built module-by-module across an 8-week plan, shipped to a live,
public HTTPS URL.

## Problem

Knowledge work is scattered across formats: scanned invoices,
handwritten forms, PDFs, photos of documents, and voice notes. This
project puts all of that in one place — searchable, queryable, and
answerable by text or voice — without manually reading every page.

## Status

🚧 Under active development. Currently on **Week 1: Core foundation**.

- [x] Week 1 — Core app (domain models, config, logging, exceptions)
- [ ] Week 2 — Analytics module
- [ ] Week 3 — Vision module
- [ ] Week 4 — Language / RAG module
- [ ] Week 5 — Speech module
- [ ] Week 6 — OCR / Ingestion module
- [ ] Week 7 — Platform layer (FastAPI, Docker, CI, deploy)
- [ ] Week 8 — Frontend + hardening + v1.0 release

## Architecture

Full write-up coming in `docs/` as modules are built. In short:
upload → Vision classifies doc type → OCR extracts text/tables →
indexed into a vector store → user asks a question (text or voice) →
RAG retrieves relevant chunks → LLM answers with sources → optionally
synthesized back to speech.

## Local setup

```bash
git clone https://github.com/sairam-1023/OmniAssistAI.git
cd OmniAssistAI
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest -v
```

## Tech stack

Python 3.11 · Pydantic v2 · FastAPI (Week 7) · scikit-learn/XGBoost ·
PyTorch (ResNet18) · PaddleOCR/TrOCR · FAISS/Chroma · Whisper ·
XTTS-v2 · Docker · GitHub Actions

## License

TBD