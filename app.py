import logging
import os
from datetime import datetime
from typing import Tuple
import json


import streamlit as st
from backend.ai_service import generate_dialogue
from backend.shorts_renderer import render_shorts_video
from backend.stt_service import transcribe_audio
from backend.tts_service import mp3_duration_seconds, speak_text, stitch_mp3_chunks, voice_id_for



def _setup_logger(base_dir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(base_dir, f"duo_mode_{timestamp}.log")

    logger = logging.getLogger("duo_mode_app")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False
    logger.info("Logger initialized; writing to %s", log_path)
    return logger, log_path


st.title("🎭 Shorts Generator for Topics")

temp_dir = "logs"
temp_media_dir = "temp"
audio_path = os.path.join(temp_dir, "recorded_audio.wav")
pause_between_lines_ms = 250
logger, log_path = _setup_logger(temp_dir)
os.makedirs(temp_media_dir, exist_ok=True)

st.caption("Record a question and let JOHN and CARTOON_DAD bring the lesson to life.")
st.caption(f"Session logs saved to `{log_path}`")

# audio = st.audio_input("Upload or record your voice (topic request)")

# if audio:
#     st.write("Audio captured")
#     st.audio(audio)
#     logger.info("Audio clip received from user.")

#     os.makedirs(temp_dir, exist_ok=True)
#     with open(audio_path, "wb") as f:
#         f.write(audio.getbuffer())
#     logger.info("Audio saved to %s", audio_path)

#     with st.status("Running Duo Mode...", expanded=True) as status:
#         status.write("Transcribing your topic...")
#         topic = transcribe_audio(audio_path, logger=logger)
#         # topic = "decision trees and how they generate branches"
#         st.write(f"Detected topic: **{topic}**")
#         logger.info("Detected topic: %s", topic)

#         status.update(label="Generating dialogue script...", state="running")
#         dialogue = generate_dialogue(topic, logger=logger)
#         status.write("Dialogue ready. Preview it below before synthesis.")

#         st.subheader("Dialogue Script")
#         for turn in dialogue:
#             st.markdown(f"**{turn['speaker']}**: {turn['line']}")

#         status.update(label="Synthesizing duo voices...", state="running")
#         audio_chunks: list[bytes] = []
#         timed_dialogue: list[dict[str, object]] = []
#         current_start = 0.0
#         pause_seconds = max(pause_between_lines_ms, 0) / 1000.0
#         for idx, turn in enumerate(dialogue, start=1):
#             speaker_voice = voice_id_for(turn["speaker"])
#             status.write(f"Generating line {idx} for {turn['speaker']}...")
#             logger.info("Generating line %s for %s", idx, turn["speaker"])
#             chunk = speak_text(turn["line"], voice_id=speaker_voice, logger=logger)
#             audio_chunks.append(chunk)
#             duration = mp3_duration_seconds(chunk)
#             timed_dialogue.append(
#                 {
#                     "speaker": turn["speaker"],
#                     "text": turn["line"],
#                     "start": current_start,
#                     "duration": duration,
#                 }
#             )
#             logger.info(
#                 "Line timing speaker=%s duration=%.2fs start=%.2fs",
#                 turn["speaker"],
#                 duration,
#                 current_start,
#             )
#             current_start += duration + pause_seconds

#         final_audio = stitch_mp3_chunks(
#             audio_chunks,
#             pause_ms=pause_between_lines_ms,
#             logger=logger,
#         )
#         duo_audio_path = os.path.join(temp_media_dir, "duo_audio.mp3")
#         with open(duo_audio_path, "wb") as f:
#             f.write(final_audio)
#         status.update(label="Duo Mode complete!", state="complete")
#         logger.info("Dialogue audio stitched successfully.")
#         st.session_state["timed_dialogue"] = timed_dialogue
#         st.session_state["duo_audio_path"] = duo_audio_path

#         st.write(timed_dialogue)
#         st.write(duo_audio_path)

#     st.audio(final_audio, format="audio/mp3")
#     st.download_button(
#         "Download Duo Dialogue",
#         data=final_audio,
#         file_name="duo-mode-dialogue.mp3",
#         mime="audio/mpeg",
#     )

if True:
    st.session_state["timed_dialogue"] = json.loads('''
[
  {
    "speaker": "CARTOON_DAD",
    "text": "Hey, John, funny thought: if a decision tree were a garden hose, would it tell me to water my cactus today based on sun exposure and soil moisture? And could you sketch a tiny, concrete example tree for that exact scenario?",
    "start": 0,
    "duration": 13.740408163265306
  },
  {
    "speaker": "JOHN",
    "text": "Sure thing. A decision tree is a flow of yes/no questions that splits data into branches until a decision is reached. Start at the root with the plant's features, then ask: is soil moisture low? is sun high? From those answers you reach 'water' or 'do not water'.",
    "start": 13.990408163265306,
    "duration": 22.151836734693877
  },
  {
    "speaker": "CARTOON_DAD",
    "text": "Got it! So decision trees use simple if-then paths to guide choices, with each split using a feature like soil moisture or sun. Thanks for the clear demo, John, now I can read trees as friendly guides.",
    "start": 36.39224489795918,
    "duration": 12.016326530612245
  }
]
    ''')
    st.session_state["duo_audio_path"] = "temp/duo_audio.mp3"

    if st.button("Render Shorts Video"):
        if "timed_dialogue" not in st.session_state or "duo_audio_path" not in st.session_state:
            st.error("Missing audio or timing data. Please run Duo Mode first.")
        else:
            with st.status("Rendering Shorts video...", expanded=True) as status:
                output_path = os.path.join(temp_media_dir, "output_short.mp4")
                status.write("Compositing video and captions...")
                render_shorts_video(
                    st.session_state["timed_dialogue"],
                    audio_path=st.session_state["duo_audio_path"],
                    output_path=output_path,
                )
                status.update(label="Shorts video ready!", state="complete")
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Shorts Video",
                    data=f,
                    file_name="output_short.mp4",
                    mime="video/mp4",
                )
