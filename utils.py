import whisper
import torchaudio
import subprocess
from transformers import pipeline
from difflib import SequenceMatcher

model = whisper.load_model("base")  # Load once globally

def extract_audio(video_path):
    audio_path = "temp_audio.wav"
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
    ]
    subprocess.run(command, check=True)
    return audio_path

def transcribe_audio(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        print("✅ Audio shape:", waveform.shape)
        print("✅ Sample rate:", sample_rate)

        if waveform.numel() == 0:
            raise ValueError("Audio seems empty.")
    except Exception as e:
        raise RuntimeError("❌ Failed to load audio. It might be corrupted or silent.") from e

    result = model.transcribe(audio_path, verbose=False)
    text = result["text"]
    segments = result.get("segments", [])
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
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-ss", str(start_time), "-to", str(end_time),
        "-c", "copy", output_path
    ]
    subprocess.run(command, check=True)
    return output_path

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