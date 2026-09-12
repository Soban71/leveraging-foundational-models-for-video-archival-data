import os
import re
import subprocess
import tempfile
from pathlib import Path

import chromadb
import numpy as np
import open_clip
import torch


CLIP_FOLDER = "clips"
VIDEO_FOLDER = "videos"
SUBTITLE_FOLDER = "subtitles"
TRANSCRIPT_FOLDER = "transcripts"
CHROMA_PATH = "chroma_store"
TEXT_COLLECTION_NAME = "video_text_embeddings"
CLIP_SECONDS = 30
WHISPER_MODEL = "base"


def timestamp_to_seconds(value: str) -> float:
    """Convert SRT/VTT timestamp text to seconds."""
    value = value.strip().replace(",", ".")
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours = "0"
        minutes, seconds = parts
    else:
        return 0.0
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def read_subtitle_segments(subtitle_path: Path):
    """Read normal SRT or VTT subtitles without needing another package."""
    content = subtitle_path.read_text(encoding="utf-8", errors="ignore")
    content = content.replace("\ufeff", "")
    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    segments = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        time_line_index = next(
            (index for index, line in enumerate(lines) if "-->" in line),
            None,
        )
        if time_line_index is None:
            continue

        start_text, end_text = lines[time_line_index].split("-->", maxsplit=1)
        end_text = end_text.strip().split()[0]
        text = " ".join(lines[time_line_index + 1:])
        text = re.sub(r"<[^>]+>", "", text)

        if text:
            segments.append((timestamp_to_seconds(start_text), timestamp_to_seconds(end_text), text))

    return segments


def parse_clip_name(clip_file: str):
    """Return the original source name and zero-based start time for SOURCE_clip_N."""
    match = re.match(r"^(.*)_clip_(\d+)$", Path(clip_file).stem)
    if not match:
        return Path(clip_file).stem, 0
    return match.group(1), (int(match.group(2)) - 1) * CLIP_SECONDS


def clip_text_from_segments(segments, start_seconds: int):
    end_seconds = start_seconds + CLIP_SECONDS
    return " ".join(
        text for start, end, text in segments
        if end >= start_seconds and start <= end_seconds
    ).strip()


def find_sidecar_subtitle(clip_file: str, source_name: str):
    """Prefer clip subtitles, then a subtitle file belonging to the original video."""
    clip_stem = Path(clip_file).stem
    candidates = []
    for folder in (Path(CLIP_FOLDER), Path(SUBTITLE_FOLDER), Path(VIDEO_FOLDER)):
        for stem in (clip_stem, source_name):
            for extension in (".srt", ".vtt"):
                candidates.append(folder / f"{stem}{extension}")

    return next((path for path in candidates if path.is_file()), None)


def find_source_video(source_name: str):
    for extension in (".mp4", ".mov", ".avi", ".mkv"):
        path = Path(VIDEO_FOLDER) / f"{source_name}{extension}"
        if path.is_file():
            return path
    return None


def extract_embedded_subtitle(source_video: Path):
    """Extract subtitle stream 0 from a source video, returning subtitle segments."""
    if source_video is None:
        return []

    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        command = [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source_video),
            "-map", "0:s:0", str(temporary_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not temporary_path.exists():
            return []
        return read_subtitle_segments(temporary_path)
    except FileNotFoundError:
        print("ffmpeg is not installed, so embedded subtitles cannot be read.")
        return []
    finally:
        temporary_path.unlink(missing_ok=True)


def has_audio_stream(video_path: Path) -> bool:
    """Check for audio before giving a file to Whisper."""
    if video_path is None or not video_path.is_file():
        return False
    try:
        completed = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0 and "audio" in completed.stdout.lower()
    except FileNotFoundError:
        return False


def whisper_transcript(source_video: Path, start_seconds: int, whisper_model):
    """Transcribe only this clip's audio from the original source video.

    OpenCV creates the 30-second clips without audio, so Whisper must read the
    matching 30-second section from the original video in the videos folder.
    """
    if not has_audio_stream(source_video):
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary_file:
        audio_path = Path(temporary_file.name)

    try:
        command = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start_seconds), "-t", str(CLIP_SECONDS),
            "-i", str(source_video), "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", str(audio_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size == 0:
            return ""
        result = whisper_model.transcribe(
            str(audio_path), fp16=torch.cuda.is_available()
        )
        return " ".join(segment["text"].strip() for segment in result["segments"]).strip()
    finally:
        audio_path.unlink(missing_ok=True)


def load_openclip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer, device


def create_text_embedding(text, model, tokenizer, device):
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy().astype(np.float32)


def main():
    Path(TRANSCRIPT_FOLDER).mkdir(exist_ok=True)
    clip_files = sorted(path for path in Path(CLIP_FOLDER).glob("*.mp4"))
    if not clip_files:
        raise FileNotFoundError(f"No .mp4 clips found in '{CLIP_FOLDER}'.")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(name=TEXT_COLLECTION_NAME)
        print(f"Deleted old collection: {TEXT_COLLECTION_NAME}")
    except Exception:
        pass
    collection = client.create_collection(name=TEXT_COLLECTION_NAME)

    model, tokenizer, device = load_openclip()
    whisper_model = None

    for number, clip_path in enumerate(clip_files, start=1):
        clip_file = clip_path.name
        clip_id = clip_path.stem
        source_name, start_seconds = parse_clip_name(clip_file)
        transcript_path = Path(TRANSCRIPT_FOLDER) / f"{clip_id}.txt"
        subtitle_path = find_sidecar_subtitle(clip_file, source_name)
        transcript = ""
        transcript_source = ""

        # Reuse an already-created transcript. This avoids running Whisper again
        # when the script is re-run to rebuild text embeddings.
        if transcript_path.is_file():
            transcript = transcript_path.read_text(encoding="utf-8", errors="ignore").strip()
            if transcript:
                transcript_source = f"existing transcript: {transcript_path.name}"

        if not transcript and subtitle_path:
            # Clip-level sidecars already start at 00:00. Source-video sidecars
            # must be cut to this clip's 30-second time window.
            subtitle_start = 0 if subtitle_path.stem == clip_id else start_seconds
            transcript = clip_text_from_segments(
                read_subtitle_segments(subtitle_path), subtitle_start
            )
            transcript_source = f"subtitle: {subtitle_path.name}"

        if not transcript:
            embedded_segments = extract_embedded_subtitle(find_source_video(source_name))
            transcript = clip_text_from_segments(embedded_segments, start_seconds)
            if transcript:
                transcript_source = "embedded subtitle"

        if not transcript:
            source_video = find_source_video(source_name)
            if not has_audio_stream(source_video):
                transcript_source = "no audio found"
            elif whisper_model is None:
                try:
                    import whisper
                except ImportError as error:
                    raise ImportError(
                        "Whisper is required when no subtitles are available. "
                        "Install it with: pip install -U openai-whisper"
                    ) from error
                print(f"Loading Whisper model: {WHISPER_MODEL}")
                whisper_model = whisper.load_model(WHISPER_MODEL)
            if has_audio_stream(source_video):
                transcript = whisper_transcript(
                    source_video, start_seconds, whisper_model
                )
                transcript_source = (
                    f"Whisper ({WHISPER_MODEL})" if transcript else "no speech detected"
                )

        if not transcript:
            transcript = "No speech or subtitle text was found for this clip."
            transcript_source = "no speech detected"

        # Save only newly created text; never overwrite an existing transcript.
        if not transcript_path.is_file():
            transcript_path.write_text(transcript, encoding="utf-8")
        embedding = create_text_embedding(transcript, model, tokenizer, device)
        collection.add(
            ids=[f"{clip_id}__text"],
            embeddings=[embedding.tolist()],
            documents=[transcript],
            metadatas=[{
                "clip_id": clip_id,
                "clip_file": clip_file,
                "clip_path": str(clip_path),
                "transcript_source": transcript_source,
            }],
        )
        print(f"[{number}/{len(clip_files)}] {clip_file} -> {transcript_source}")

    print(f"\nText embeddings saved to: {TEXT_COLLECTION_NAME}")


if __name__ == "__main__":
    main()
