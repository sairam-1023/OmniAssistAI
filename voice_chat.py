"""
Interactive voice conversation with OmniAssist AI.

Press Enter to record a question, speak, then listen to the spoken
answer. Repeat. Type 'q' + Enter (instead of just Enter) to quit.
"""

import sounddevice as sd
import soundfile as sf

from modules.language.rag import answer_query
from modules.speech.synthesize import synthesize
from modules.speech.transcribe import transcribe

RECORDING_PATH = "data/audio/live_question.wav"
ANSWER_PATH = "data/audio/live_answer.mp3"
RECORD_SECONDS = 6
SAMPLE_RATE = 16000


def record_question():
    print(f"\nRecording for {RECORD_SECONDS} seconds... speak now!")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    sf.write(RECORDING_PATH, audio, SAMPLE_RATE)
    print("Got it.")


def play_answer(path: str):
    # Reuse macOS's afplay via subprocess — reliable, confirmed working
    # earlier (unlike `open`, which had default-app issues on this machine).
    import subprocess
    subprocess.run(["afplay", path])


def main():
    print("=== OmniAssist AI — Voice Chat ===")
    print("Press Enter to ask a question by voice. Type 'q' then Enter to quit.\n")

    while True:
        user_input = input("Press Enter to speak (or 'q' to quit): ").strip().lower()
        if user_input == "q":
            print("Goodbye!")
            break

        record_question()

        transcription = transcribe(RECORDING_PATH)
        question_text = transcription["text"]
        print(f"\nYou asked: {question_text!r} (confidence: {transcription['confidence']:.2f})")

        if not question_text:
            print("Didn't catch that — nothing transcribed. Try again.")
            continue

        result = answer_query(question_text)
        print(f"[intent: {result['intent']}]")
        print(f"Answer: {result['answer']}")
        if result["sources"]:
            print(f"Sources: {[s['filename'] for s in result['sources']]}")

        synthesize(result["answer"], ANSWER_PATH)
        print("Speaking answer...")
        play_answer(ANSWER_PATH)


if __name__ == "__main__":
    main()