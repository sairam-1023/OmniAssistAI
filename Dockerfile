# Base image: official Python 3.11 slim build — matches our local dev
# Python version exactly (Week 1 decision), "slim" keeps image size down
# by excluding unnecessary OS packages.
FROM python:3.11-slim

# System-level dependencies our Python packages need under the hood:
# - tesseract-ocr: for modules/ocr/extract_printed.py (pytesseract)
# - ffmpeg: for modules/speech/transcribe.py (Whisper audio decoding)
# - libgl1: OpenCV (cv2) needs this for image processing even in a
#   headless (no display) server environment
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy and install Python dependencies FIRST, before copying the rest
# of the code. Docker caches each instruction as a "layer" — if only
# our application code changes (not requirements.txt), Docker reuses
# the cached dependency-install layer instead of reinstalling
# everything from scratch, making rebuilds much faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the actual application code.
COPY . .

# Hugging Face Spaces expects the app to listen on port 7860 by default.
EXPOSE 7860

# Run the FastAPI app via uvicorn, binding to all interfaces (0.0.0.0)
# so it's reachable from outside the container, on the port HF Spaces expects.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]