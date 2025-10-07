# backend/tts_service.py
import os
from io import BytesIO
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

load_dotenv()

# Initialize ElevenLabs client
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)

def speak_text(text: str, voice_id="pNInz6obpgDQGcFmaJgB"):
    """
    Convert text to speech using ElevenLabs streaming API.
    Returns a BytesIO stream with audio data.
    """
    response = elevenlabs.text_to_speech.stream(
        voice_id=voice_id,
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )

    audio_stream = BytesIO()
    for chunk in response:
        if chunk:
            audio_stream.write(chunk)
    audio_stream.seek(0)
    return audio_stream