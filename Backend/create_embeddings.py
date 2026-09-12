import os
import cv2
import torch
import open_clip
import chromadb
import numpy as np
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

CLIP_FOLDER = "clips"

CHROMA_PATH = "chroma_store"

COLLECTION_NAME = "video_frame_embeddings"

FRAME_INTERVAL_SECONDS = 2


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")


# ============================================================
# LOAD OPENCLIP
# ============================================================

print("Loading OpenCLIP model...")

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

model = model.to(device)
model.eval()

print("OpenCLIP model loaded.")


# ============================================================
# LOAD CHROMADB
# ============================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


# Delete old frame collection if it exists.
# This makes sure the embeddings are rebuilt.
try:
    client.delete_collection(
        name=COLLECTION_NAME
    )

    print(
        f"Deleted old collection: {COLLECTION_NAME}"
    )

except Exception:
    pass


collection = client.create_collection(
    name=COLLECTION_NAME
)


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames_every_2_seconds(
    clip_path: str
):
    cap = cv2.VideoCapture(clip_path)

    if not cap.isOpened():
        print(
            f"Could not open clip: {clip_path}"
        )

        return []

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        cap.release()

        print(
            f"Invalid FPS: {clip_path}"
        )

        return []

    frame_interval = max(
        1,
        int(
            fps * FRAME_INTERVAL_SECONDS
        )
    )

    frames = []

    frame_number = 0

    while frame_number < total_frames:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

        success, frame = cap.read()

        if success:

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                frame_rgb
            )

            frames.append(
                image
            )

        frame_number += frame_interval

    cap.release()

    return frames


# ============================================================
# CREATE FRAME EMBEDDINGS
# ============================================================

def create_frame_embeddings(
    clip_path: str
):

    frames = extract_frames_every_2_seconds(
        clip_path
    )

    if len(frames) == 0:

        return [], 0

    frame_embeddings = []

    for frame_index, frame in enumerate(
        frames
    ):

        image = preprocess(
            frame
        ).unsqueeze(0).to(device)

        with torch.no_grad():

            embedding = model.encode_image(
                image
            )

        # Normalize embedding
        embedding = (
            embedding
            / embedding.norm(
                dim=-1,
                keepdim=True
            )
        )

        frame_embeddings.append(
            embedding[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return (
        frame_embeddings,
        len(frames)
    )


# ============================================================
# FIND VIDEO CLIPS
# ============================================================

clip_files = sorted(
    [
        file
        for file in os.listdir(
            CLIP_FOLDER
        )
        if file.lower().endswith(".mp4")
    ]
)


print()
print(
    f"Found {len(clip_files)} video clips."
)

print()


# ============================================================
# PROCESS EVERY CLIP
# ============================================================

total_frames_stored = 0

for clip_number, clip_file in enumerate(
    clip_files,
    start=1
):

    clip_path = os.path.join(
        CLIP_FOLDER,
        clip_file
    )

    # Remove .mp4 but preserve the
    # COMPLETE clip name.
    clip_id = os.path.splitext(
        clip_file
    )[0]

    print(
        "=" * 80
    )

    print(
        f"[{clip_number}/{len(clip_files)}]"
    )

    print(
        f"Processing: {clip_file}"
    )

    # --------------------------------------------------------
    # CREATE FRAME EMBEDDINGS
    # --------------------------------------------------------

    frame_embeddings, frame_count = (
        create_frame_embeddings(
            clip_path
        )
    )

    if frame_count == 0:

        print(
            "Skipping clip because "
            "no frames were extracted."
        )

        continue

    # --------------------------------------------------------
    # STORE EVERY FRAME
    # --------------------------------------------------------

    ids = []

    embeddings = []

    documents = []

    metadatas = []

    for frame_index, embedding in enumerate(
        frame_embeddings
    ):

        # Unique ID for every frame.
        #
        # Example:
        #
        # clip name + frame number
        #
        # This prevents clips from different
        # videos being mixed together.

        frame_id = (
            f"{clip_id}__frame_{frame_index}"
        )

        ids.append(
            frame_id
        )

        embeddings.append(
            embedding.tolist()
        )

        documents.append(
            clip_file
        )

        metadatas.append(
            {
                "clip_id": clip_id,
                "clip_file": clip_file,
                "clip_path": clip_path,
                "frame_index": frame_index,
                "frames_used": frame_count
            }
        )

    # --------------------------------------------------------
    # ADD TO CHROMADB
    # --------------------------------------------------------

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    total_frames_stored += frame_count

    print(
        f"Stored {frame_count} frames."
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print(
    "=" * 80
)

print(
    "EMBEDDING CREATION COMPLETE"
)

print(
    "=" * 80
)

print(
    f"Video clips processed: "
    f"{len(clip_files)}"
)

print(
    f"Total frames stored: "
    f"{total_frames_stored}"
)

print(
    f"Chroma collection: "
    f"{COLLECTION_NAME}"
)

print()
print(
    "You can now run search_clips.py"
)