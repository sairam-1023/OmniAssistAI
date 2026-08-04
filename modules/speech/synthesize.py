"""
Text-to-speech via gTTS (Google Text-to-Speech). Lightweight, free,
requires internet access. XTTS-v2 (voice cloning, higher quality,
fully local/offline) is a documented future upgrade — see
docs/speech_model_notes.md.

Interface (per project plan, Section 3.5):
    synthesize(text: str, output_path: str) -> str  (returns output_path)
"""

from gtts import gTTS

from modules.core.exceptions import SynthesisError
from modules.core.logging_config import get_logger

logger = get_logger(__name__)


def synthesize(text: str, output_path: str) -> str:
    if not text or not text.strip():
        raise SynthesisError("Cannot synthesize empty text.")

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(output_path)
    except Exception as e:
        raise SynthesisError(f"Failed to synthesize speech: {e}") from e

    logger.info(f"Synthesized {len(text)} chars of text -> {output_path}")
    return output_path