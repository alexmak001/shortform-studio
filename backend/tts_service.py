"""Text-to-speech via the ElevenLabs API."""

from __future__ import annotations

from io import BytesIO
import os
from typing import Iterable

from dotenv import load_dotenv
from elevenlabs import ElevenLabs


load_dotenv()

_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not _API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in environment/.env file.")

_DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "fQfFscOlHYWduZPLp3YY")
_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")

_client = ElevenLabs(api_key=_API_KEY)


def _resolve_voice_id(voice: str | None) -> str:
    voice_id = voice or _DEFAULT_VOICE_ID
    if not voice_id:
        raise RuntimeError(
            "No ElevenLabs voice configured. Pass speaker argument or set ELEVENLABS_VOICE_ID."
        )
    return voice_id


def _accumulate_audio(chunks: Iterable[bytes]) -> BytesIO:
    buffer = BytesIO()
    for chunk in chunks:
        buffer.write(chunk)
    buffer.seek(0)
    return buffer


def speak_text(
    text: str,
    speaker_wav: str | None = None,
    speaker: str | None = None,
    language: str | None = None,
) -> BytesIO:
    """Generate speech audio using ElevenLabs voices."""
    if speaker_wav:
        raise NotImplementedError("Voice cloning via speaker_wav is not supported with the ElevenLabs API.")

    voice_id = _resolve_voice_id(speaker)
    try:
        audio_chunks = _client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=_MODEL_ID,
            text=text,
            output_format=_OUTPUT_FORMAT,
            optimize_streaming_latency="0",
        )
    except Exception as exc:  # pragma: no cover - depends on external API
        raise RuntimeError(f"ElevenLabs text-to-speech failed: {exc}") from exc

    return _accumulate_audio(audio_chunks)
