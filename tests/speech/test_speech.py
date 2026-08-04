"""Tests for modules/speech/transcribe.py and synthesize.py"""

import numpy as np
import soundfile as sf

from modules.speech.synthesize import synthesize
from modules.speech.transcribe import transcribe


def test_transcribe_returns_expected_keys(tmp_path):
    # Generate a short silent audio file — we're testing the interface
    # contract (correct output shape), not transcription accuracy,
    # same philosophy as Vision/Language's contract-only tests.
    audio_path = tmp_path / "silence.wav"
    silence = np.zeros(16000 * 2, dtype=np.float32)  # 2 seconds of silence
    sf.write(str(audio_path), silence, 16000)

    result = transcribe(str(audio_path))
    assert set(result.keys()) == {"text", "language", "confidence"}


def test_transcribe_confidence_is_valid_range(tmp_path):
    audio_path = tmp_path / "silence.wav"
    silence = np.zeros(16000 * 2, dtype=np.float32)
    sf.write(str(audio_path), silence, 16000)

    result = transcribe(str(audio_path))
    assert 0.0 <= result["confidence"] <= 1.0


def test_synthesize_creates_valid_audio_file(tmp_path):
    output_path = tmp_path / "output.mp3"
    synthesize("This is a test.", str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_synthesize_rejects_empty_text(tmp_path):
    import pytest
    from modules.core.exceptions import SynthesisError

    output_path = tmp_path / "output.mp3"
    with pytest.raises(SynthesisError):
        synthesize("", str(output_path))