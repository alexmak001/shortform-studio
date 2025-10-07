# speech to text
import whisper
import os 

model_name = os.getenv("WHISPER_MODEL", "small")
model = whisper.load_model(model_name)

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]