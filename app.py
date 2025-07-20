import streamlit as st
import numpy as np
import torch
import os
from utils import extract_audio, transcribe_audio, summarize_text, clip_video, match_summary_to_segments

os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(layout="centered")

st.sidebar.write("✅ Numpy:", np._version_)
st.sidebar.write("✅ Torch:", torch._version_)
st.sidebar.write("✅ Whisper loaded")

st.title("🎥 Video Summary from Long Lectures")

video_file = st.file_uploader("Upload a lecture video", type=["mp4", "mov", "avi", "webm"])

import tempfile

if video_file:
    temp_video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")

    # Save uploaded video file in temp path
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    # Confirm file exists before extracting audio
    if os.path.exists(temp_video_path):
        st.success("✅ Video saved successfully!")
        try:
            audio_path = extract_audio(temp_video_path)
            st.success("✅ Audio extracted successfully!")
        except Exception as e:
            st.error(str(e))
    else:
        st.error("❌ Video file not found after saving.")

    st.video(temp_video_path)

    if st.button("🔊 Extract & Transcribe Audio"):
        st.info("Processing audio and transcription...")
        try:
            audio_path = extract_audio(temp_video_path)
            transcript, segments = transcribe_audio(audio_path)
            st.session_state['transcript'] = transcript
            st.session_state['segments'] = segments
            st.success("Transcription complete!")
            st.text_area("📝 Transcript", transcript, height=300)
        except Exception as e:
            st.error(str(e))

    if "transcript" in st.session_state:
        if st.button("🧠 Generate Summary"):
            try:
                summary = summarize_text(st.session_state['transcript'])
                st.session_state['summary'] = summary
                st.success("Summary generated!")
                st.text_area("🧾 Summary", summary, height=200)
            except Exception as e:
                st.error(str(e))

    if 'transcript' in st.session_state:
        st.download_button("📥 Download Transcript", st.session_state['transcript'], file_name="transcript.txt")

    if 'summary' in st.session_state:
        st.download_button("📥 Download Summary", st.session_state['summary'], file_name="summary.txt")

        st.header("🎯 Extract a Short Video Clip")
        col1, col2 = st.columns(2)
        with col1:
            start_time = st.number_input("Start time (seconds)", min_value=0)
        with col2:
            end_time = st.number_input("End time (seconds)", min_value=1)

        if st.button("✂ Create Clip"):
            if start_time < end_time:
                try:
                    clip_path = clip_video(temp_video_path, start_time, end_time)
                    st.video(clip_path)
                    st.success("Clip created successfully!")
                except Exception as e:
                    st.error(str(e))
            else:
                st.error("Start time must be less than end time.")

    if "summary" in st.session_state and "segments" in st.session_state:
        if st.button("🎞 Generate Final Summary Clip"):
            st.warning("⚠ This feature is currently disabled. Coming soon!")