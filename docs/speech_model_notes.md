# Speech Module — Model Notes (Week 5)

## Components
- **STT (speech-to-text):** OpenAI Whisper, `small` model size, runs
  fully locally (no API key). ~500MB model, downloaded once and cached.
- **TTS (text-to-speech):** gTTS (Google Text-to-Speech). Lightweight
  cloud-API wrapper, requires internet access per call, no local model.

## STT results
Tested on a real recorded voice sample (own microphone, macOS,
`sounddevice`/`soundfile`, 16kHz mono WAV):

Spoken: "What is the invoice number?"
Transcribed: "What is the invoice number?" — exact match.
Detected language: en
Confidence: 0.591 (see note below on what this number means)

## Confidence caveat
Whisper does not output a bounded [0,1] confidence score natively —
it outputs per-segment log-probabilities. We approximate a confidence
score via `e^(average log-probability)`, clamped to [0,1]. This is a
reasonable practical approximation but should not be treated as a true
calibrated probability the way the Analytics/Vision/Language modules'
confidence scores are (those come from softmax/predict_proba, which
are properly calibrated probability distributions).

## Known issue: running on CPU, not MPS
Whisper defaulted to CPU execution (`FP16 is not supported on CPU;
using FP32 instead` warning), not the MPS GPU acceleration used by
Vision/Language modules. This is a performance gap, not a correctness
issue — transcription is accurate, just likely slower than necessary.

## TTS results
Synthesized multiple test phrases successfully, confirmed via `afinfo`
and audible playback via `afplay`. Verified end-to-end in a live,
interactive voice conversation loop (voice_chat.py) — recorded question
-> Whisper transcription -> RAG answer (with correct handling of an
ambiguous multi-invoice query) -> gTTS synthesis -> spoken response,
with no noticeable lag.

## Next steps (future work, not blocking Week 5 completion)
- Explicitly set Whisper to use MPS device, matching the pattern used
  in Vision/Language modules, for faster local transcription.
- Consider XTTS-v2 as a TTS upgrade: voice cloning, fully offline,
  higher perceived quality — tradeoff is significantly larger model
  size and slower local inference versus gTTS's lightweight cloud call.
- Test STT with background noise, accents, and non-English speech to
  establish real-world robustness (only clean, single-speaker English
  tested so far).
