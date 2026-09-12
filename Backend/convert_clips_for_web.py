import shutil
import subprocess
from pathlib import Path


INPUT_FOLDER = Path("clips")
OUTPUT_FOLDER = Path("clips_web")


def main():
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg was not found. Install FFmpeg or make sure it is on your PATH."
        )
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(f"'{INPUT_FOLDER}' folder was not found.")

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    clips = sorted(INPUT_FOLDER.glob("*.mp4"))
    if not clips:
        raise FileNotFoundError("No MP4 clips were found in the 'clips' folder.")

    for number, clip_path in enumerate(clips, start=1):
        output_path = OUTPUT_FOLDER / clip_path.name
        print(f"[{number}/{len(clips)}] Converting: {clip_path.name}")

        command = [
            "ffmpeg", "-y", "-i", str(clip_path),
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "128k",
            str(output_path),
        ]
        completed = subprocess.run(command)
        if completed.returncode != 0:
            raise RuntimeError(f"Could not convert: {clip_path.name}")

    print(f"\nDone. Browser-compatible videos are in: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()