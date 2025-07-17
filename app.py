import streamlit as st
import numpy as np
import torch
import os
import moviepy.video.io.ffmpeg_tools as ffmpeg_tools
from utils import extract_audio, transcribe_audio, summarize_text, clip_video, match_summary_to_segments


os.environ["TOKENIZERS_PARALLELISM"] = "false"
st.set_page_config(layout="centered")

st.sidebar.write("✅ Numpy:", np.__version__)
st.sidebar.write("✅ Torch:", torch.__version__)
st.sidebar.write("✅ Whisper loaded")

st.title("🎥 Video Summary from Long Lectures")

video_file = st.file_uploader("Upload a lecture video", type=["mp4", "mov", "avi", "webm"])

if video_file:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(video_file.read())
    st.video("uploaded_video.mp4")

    if st.button("🔊 Extract & Transcribe Audio"):
        st.info("Processing audio and transcription...")
        audio_path = extract_audio("uploaded_video.mp4")
        transcript, segments = transcribe_audio(audio_path)
        st.session_state['transcript'] = transcript
        st.session_state['segments'] = segments
        st.success("Transcription complete!")
        st.text_area("📝 Transcript", transcript, height=300)

    if "transcript" in st.session_state:
        if st.button("🧠 Generate Summary"):
            summary = summarize_text(st.session_state['transcript'])
            st.session_state['summary'] = summary
            st.success("Summary generated!")
            st.text_area("🧾 Summary", summary, height=200)

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

        if st.button("✂️ Create Clip"):
            if start_time < end_time:
                clip_path = clip_video("uploaded_video.mp4", start_time, end_time)
                st.video(clip_path)
                st.success("Clip created successfully!")
            else:
                st.error("Start time must be less than end time.")

    if "summary" in st.session_state and "segments" in st.session_state:
        if st.button("🎞 Generate Final Summary Clip"):
            st.warning("⚠️ This feature is currently disabled. Coming soon!")