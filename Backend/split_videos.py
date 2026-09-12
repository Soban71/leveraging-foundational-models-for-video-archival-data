import cv2
import os
from pathlib import Path

VIDEO_FOLDER = "videos"
CLIP_FOLDER = "clips"
CLIP_SECONDS = 30

os.makedirs(CLIP_FOLDER, exist_ok=True)

video_files = [
    file for file in os.listdir(VIDEO_FOLDER)
    if file.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
]

for video_file in video_files:
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    video_name = Path(video_file).stem

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open video: {video_file}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(fps * CLIP_SECONDS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    clip_number = 1
    start_frame = 0

    while start_frame < total_frames:
        end_frame = min(start_frame + frames_per_clip, total_frames)

        clip_name = f"{video_name}_clip_{clip_number}.mp4"
        clip_path = os.path.join(CLIP_FOLDER, clip_name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame

        while current_frame < end_frame:
            success, frame = cap.read()

            if not success:
                break

            out.write(frame)
            current_frame += 1

        out.release()

        print(f"Created clip: {clip_name}")

        start_frame = end_frame
        clip_number += 1

    cap.release()

print("All videos have been split into 30-second clips.")