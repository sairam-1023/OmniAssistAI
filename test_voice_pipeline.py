"""
End-to-end test: record your voice, transcribe it, run it through RAG,
speak the answer back. Ties together Weeks 4 and 5.
"""

from modules.speech.transcribe import transcribe
from modules.language.rag import answer_query
from modules.speech.synthesize import synthesize

print("Transcribing your recorded question...")
transcription = transcribe("data/audio/test_recording.wav")
print(f"You asked: {transcription['text']!r}")

print("\nRunning RAG pipeline...")
result = answer_query(transcription["text"])
print(f"Intent: {result['intent']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {[s['filename'] for s in result['sources']]}")

print("\nSynthesizing spoken answer...")
synthesize(result["answer"], "data/audio/pipeline_answer.mp3")
print("Saved to data/audio/pipeline_answer.mp3 — playing now...")