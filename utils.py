import whisper
import torchaudio
import moviepy.video.io.ffmpeg_tools as ffmpeg_tools
from transformers import pipeline
import subprocess
from difflib import SequenceMatcher

model = whisper.load_model("base")  # Load once globally

def extract_audio(video_path):
    audio_path = "temp_audio.wav"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def transcribe_audio(audio_path):
    # Load audio to check if it's valid
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        print("✅ Audio shape:", waveform.shape)
        print("✅ Sample rate:", sample_rate)

        if waveform.numel() == 0:
            raise ValueError("Audio seems empty.")
    except Exception as e:
        raise RuntimeError("❌ Failed to load audio. It might be corrupted or silent.") from e

    # Whisper transcription
    result = model.transcribe(audio_path, verbose=False)
    text = result["text"]
    segments = result.get("segments", [])  # Will be empty in openai-whisper
    return text, segments

def summarize_text(text):
    if len(text) > 8000:
        text = text[:8000]
    summarizer = pipeline("summarization", model="Falconsai/text_summarization", tokenizer="Falconsai/text_summarization")
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n"
    return summary

def clip_video(video_path, start_time, end_time, output_path="short_clip.mp4"):
    ffmpeg_tools.ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_path)
    return output_path

def match_summary_to_segments(summary, segments):
    """
    Match summary to segment texts and return combined smart clip range.
    """
    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    matched_segments = []
    for seg in segments:
        seg_text = seg.get("text", "")
        if similar(summary, seg_text) > 0.3 or any(word in seg_text.lower() for word in summary.lower().split()):
            matched_segments.append((seg['start'], seg['end']))

    if not matched_segments:
        return []

    # Merge nearby segments into one continuous clip
    merged_segments = []
    current_start, current_end = matched_segments[0]

    for start, end in matched_segments[1:]:
        if start - current_end <= 2:  # if within 2 seconds, merge
            current_end = end
        else:
            merged_segments.append((current_start, current_end))
            current_start, current_end = start, end

    merged_segments.append((current_start, current_end))
    return merged_segments