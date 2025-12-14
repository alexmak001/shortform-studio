import os

import streamlit as st
from backend.ai_service import generate_dialogue
from backend.stt_service import transcribe_audio
from backend.tts_service import speak_text, stitch_mp3_chunks, voice_id_for

st.title("🎭 Duo Mode Voice Tutor")
st.caption("Record a question and let DECAURIE and CARTOON_DAD bring the lesson to life.")

audio = st.audio_input("Upload or record your voice (topic request)")

temp_dir = "temp"
audio_path = os.path.join(temp_dir, "recorded_audio.wav")
pause_between_lines_ms = 250

if audio:
    st.write("Audio captured")
    st.audio(audio)

    os.makedirs(temp_dir, exist_ok=True)
    with open(audio_path, "wb") as f:
        f.write(audio.getbuffer())

    with st.status("Running Duo Mode...", expanded=True) as status:
        status.write("Transcribing your topic...")
        topic = transcribe_audio(audio_path)
        topic = "decision trees"
        st.write(f"Detected topic: **{topic}**")

        status.update(label="Generating dialogue script...", state="running")
        dialogue = generate_dialogue(topic)
        status.write("Dialogue ready. Preview it below before synthesis.")

        st.subheader("Dialogue Script")
        for turn in dialogue:
            st.markdown(f"**{turn['speaker']}**: {turn['line']}")

    #     status.update(label="Synthesizing duo voices...", state="running")
    #     audio_chunks: list[bytes] = []
    #     for idx, turn in enumerate(dialogue, start=1):
    #         speaker_voice = voice_id_for(turn["speaker"])
    #         status.write(f"Generating line {idx} for {turn['speaker']}...")
    #         chunk = speak_text(turn["line"], voice_id=speaker_voice)
    #         audio_chunks.append(chunk)

    #     final_audio = stitch_mp3_chunks(audio_chunks, pause_ms=pause_between_lines_ms)
    #     status.update(label="Duo Mode complete!", state="complete")

    # st.audio(final_audio, format="audio/mp3")
    # st.download_button(
    #     "Download Duo Dialogue",
    #     data=final_audio,
    #     file_name="duo-mode-dialogue.mp3",
    #     mime="audio/mpeg",
    # )
