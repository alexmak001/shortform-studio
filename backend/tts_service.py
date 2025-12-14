"""Text-to-speech via the ElevenLabs API."""

from __future__ import annotations

from io import BytesIO
import os
from typing import Optional, Sequence

from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from pydub import AudioSegment


load_dotenv()

_API_KEY = os.getenv("ELEVENLABS_API_KEY")
if not _API_KEY:
    raise RuntimeError("Missing ELEVENLABS_API_KEY in environment/.env file.")

_DEFAULT_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
_VOICE_ID_CARTOON_DAD = os.getenv("VOICE_ID_CARTOON_DAD")
_VOICE_ID_DECAURIE = os.getenv("VOICE_ID_DECAURIE")

_client = ElevenLabs(api_key=_API_KEY)


def voice_id_for(speaker: str) -> str:
    """Return the configured ElevenLabs voice ID for a Duo Mode speaker."""
    speaker_key = speaker.strip().upper()
    if speaker_key == "DECAURIE":
        voice_id = _VOICE_ID_DECAURIE
    elif speaker_key == "CARTOON_DAD":
        voice_id = _VOICE_ID_CARTOON_DAD
    else:
        voice_id = None

    if not voice_id:
        voice_id = _DEFAULT_VOICE_ID
    if not voice_id:
        raise RuntimeError(
            "Missing ElevenLabs voice mapping. "
            "Set VOICE_ID_DECAURIE and VOICE_ID_CARTOON_DAD (or ELEVENLABS_VOICE_ID as fallback)."
        )
    return voice_id


def speak_text(text: str, voice_id: Optional[str] = None) -> bytes:
    """Generate speech audio using ElevenLabs voices and return mp3 bytes."""
    resolved_voice_id = voice_id or voice_id_for("DECAURIE")
    try:
        audio_chunks = _client.text_to_speech.convert(
            voice_id=resolved_voice_id,
            model_id=_MODEL_ID,
            text=text,
            output_format=_OUTPUT_FORMAT,
            optimize_streaming_latency="0",
        )
    except Exception as exc:  # pragma: no cover - depends on external API
        raise RuntimeError(f"ElevenLabs text-to-speech failed: {exc}") from exc

    return b"".join(audio_chunks)


def stitch_mp3_chunks(chunks: Sequence[bytes], pause_ms: int = 250) -> bytes:
    """Combine mp3 chunks with small pauses and return a single mp3 payload."""
    if not chunks:
        raise ValueError("No audio chunks were provided for stitching.")

    combined: AudioSegment | None = None
    pause = AudioSegment.silent(duration=max(pause_ms, 0))

    for chunk in chunks:
        segment = AudioSegment.from_file(BytesIO(chunk), format="mp3")
        if combined is None:
            combined = segment
        else:
            combined += pause + segment

    assert combined is not None  # for mypy-like tools
    buffer = BytesIO()
    combined.export(buffer, format="mp3")
    return buffer.getvalue()
