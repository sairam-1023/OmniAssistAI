"""
Records a short audio clip from your microphone for testing Whisper.
Run, speak when prompted, and it saves to data/audio/test_recording.wav
"""

import sounddevice as sd
import soundfile as sf

DURATION_SECONDS = 6
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio

print(f"Recording for {DURATION_SECONDS} seconds... speak now!")
audio = sd.rec(int(DURATION_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()  # block until recording finishes
print("Done recording.")

sf.write("data/audio/test_recording.wav", audio, SAMPLE_RATE)
print("Saved to data/audio/test_recording.wav")