import os

import streamlit as st
from backend.ai_service import generate_topic_explanation
from backend.stt_service import transcribe_audio
from backend.tts_service import speak_text

st.title("🎓 AI Voice Tutor")

# Upload or record audio
audio = st.audio_input("Upload your voice (topic request)")

temp_dir = "temp"
audio_path = os.path.join(temp_dir, "recorded_audio.wav")

if audio:
    st.write("Audio captured")
    st.audio(audio)  # play back what Streamlit captured

    os.makedirs(temp_dir, exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio.getbuffer())

    # Transcribe audio to texts
    topic = transcribe_audio(audio_path)
    # topic = "decision trees"
    st.write(f"Detected topic: **{topic}**")

    # Generate explanation from AI
    explanation = generate_topic_explanation(topic)
    
    st.write(f"**Explanation:** {explanation}")

    # Synthesize locally and play back
    audio_stream = speak_text(explanation)
    st.write("Audio stream generated")
    st.audio(audio_stream, format="audio/wav")
