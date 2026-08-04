from modules.speech.transcribe import transcribe

from modules.speech.synthesize import synthesize

synthesize("Your Acme invoice total is $4,050, due on April 2nd.", "data/audio/test_output.mp3")

result = transcribe("data/audio/test_recording.wav")
print(result)