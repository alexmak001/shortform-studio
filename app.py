import streamlit as st
import tempfile
from backend.ai_service import generate_topic_explanation
from backend.stt_service import transcribe_audio
from backend.tts_service import speak_text

st.title("🎓 AI Voice Tutor")

# Upload or record audio
audio = st.audio_input("Upload your voice (topic request)")

audio_path = "temp/recorded_audio.wav"

if audio:
    with open(audio_path, "wb") as f:
        f.write(audio.getbuffer())

    # Transcribe audio to text
    topic = transcribe_audio(audio_path)
    st.write(f"Detected topic: **{topic}**")

    # Generate explanation from AI
    explanation = generate_topic_explanation(topic)
    st.write(f"**Explanation:** {explanation}")

    # Speak with ElevenLabs
    speak_text(explanation, voice="Bella")