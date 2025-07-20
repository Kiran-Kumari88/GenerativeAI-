import whisper
import torchaudio
import subprocess
import os
import imageio_ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from transformers import pipeline
from difflib import SequenceMatcher

# Load Whisper model once at import
model = whisper.load_model("base")

def extract_audio(video_path):
    audio_path = os.path.join("/tmp", "temp_audio.wav")

    # Use imageio-ffmpeg's built-in ffmpeg binary
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        audio_path
    ]

    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        raise RuntimeError(f"❌ ffmpeg failed to extract audio: {result.stderr.decode()}")

    if not os.path.exists(audio_path):
        raise RuntimeError("❌ Audio file was not created.")
    return audio_path

def transcribe_audio(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.numel() == 0:
            raise ValueError("Audio file is empty.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load audio: {e}") from e

    try:
        result = model.transcribe(audio_path, verbose=False)
        return result["text"], result.get("segments", [])
    except Exception as e:
        raise RuntimeError(f"❌ Whisper failed to transcribe audio: {e}") from e

def summarize_text(text):
    if len(text) > 8000:
        text = text[:8000]

    try:
        summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load summarization model: {e}") from e

    chunks = [text[i:i + 800] for i in range(0, len(text), 800)]
    summary = ""
    for chunk in chunks:
        try:
            result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
            summary += result[0]['summary_text'].strip() + "\n"
        except Exception as e:
            print(f"⚠ Error during summarization of chunk: {e}")
    return summary.strip()

def clip_video(video_path, start_time, end_time, output_path="/tmp/short_clip.mp4"):
    try:
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)
        if not os.path.exists(output_path):
            raise RuntimeError("❌ Clip was not created.")
        return output_path
    except Exception as e:
        raise RuntimeError(f"❌ Failed to clip video: {e}") from e

def match_summary_to_segments(summary, segments):
    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    matched_segments = []
    for seg in segments:
        seg_text = seg.get("text", "")
        if similar(summary, seg_text) > 0.3 or any(word in seg_text.lower() for word in summary.lower().split()):
            matched_segments.append((seg['start'], seg['end']))

    if not matched_segments:
        return []

    merged_segments = []
    current_start, current_end = matched_segments[0]

    for start, end in matched_segments[1:]:
        if start - current_end <= 2:
            current_end = end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end

    merged_segments.append((current_start, current_end))
    return merged_segments