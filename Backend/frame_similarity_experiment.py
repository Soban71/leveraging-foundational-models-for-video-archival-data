import csv
import json
from pathlib import Path

import chromadb
import numpy as np
import open_clip
import torch

from evaluation import calculate_metrics


# ============================================================
# FRAME-LEVEL COSINE-SIMILARITY RETRIEVAL EXPERIMENT
# ============================================================

TOP_K = 5
GROUND_TRUTH_FILES = ["evaluation_dataset.json", "evaluation_dataset(6).json"]

# Original OpenCLIP ViT-B/32 LAION frame-embedding experiment.
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "video_frame_embeddings"

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
MODEL_LABEL = "OpenCLIP ViT-B/32 (LAION-2B) - Frame Similarity"

OUTPUT_FILE = "results_frame_similarity_openclip_vit_b32.csv"


def load_ground_truth():
    path = next(
        (Path(name) for name in GROUND_TRUTH_FILES if Path(name).is_file()),
        None,
    )

    if path is None:
        raise FileNotFoundError("evaluation_dataset.json was not found.")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_frame_embeddings():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    data = collection.get(include=["embeddings", "metadatas"])

    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    metadatas = data["metadatas"]

    print(f"Loaded {len(embeddings)} frame embeddings from ChromaDB.")
    return embeddings, metadatas


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME} ({PRETRAINED})")

    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED,
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    return model, tokenizer, device


def encode_query(query, model, tokenizer, device):
    tokens = tokenizer([query]).to(device)

    with torch.inference_mode():
        embedding = model.encode_text(tokens)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding[0].cpu().numpy().astype(np.float32)


def rank_clips_by_best_frame(query_embedding, frame_embeddings, metadatas):
    """
    Calculate cosine similarity against EVERY stored frame embedding.
    Each clip receives the score of its highest-similarity frame.
    """
    frame_scores = frame_embeddings @ query_embedding
    best_frame_for_clip = {}

    for score, metadata in zip(frame_scores, metadatas):
        clip_id = metadata["clip_id"]

        if (
            clip_id not in best_frame_for_clip
            or score > best_frame_for_clip[clip_id]["visual_score"]
        ):
            best_frame_for_clip[clip_id] = {
                "clip_id": clip_id,
                "visual_score": float(score),
                "best_frame_index": metadata.get("frame_index", 0),
            }

    return sorted(
        best_frame_for_clip.values(),
        key=lambda result: result["visual_score"],
        reverse=True,
    )[:TOP_K]


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    ground_truth = load_ground_truth()
    frame_embeddings, metadatas = load_frame_embeddings()
    model, tokenizer, device = load_model()

    print("\n" + "=" * 92)
    print("FRAME-LEVEL COSINE-SIMILARITY RETRIEVAL EXPERIMENT")
    print("=" * 92)
    print(f"Model:      {MODEL_LABEL}")
    print(f"Database:   {CHROMA_PATH}")
    print(f"Process:    Query compared with every stored frame embedding")
    print(f"Clip score: Highest cosine-similarity frame in that clip")
    print(f"Evaluation: {len(ground_truth)} queries; Top-{TOP_K} clips")
    print("=" * 92 + "\n")

    precision_scores, recall_scores, f1_scores, hit_scores = [], [], [], []
    rows = []

    for number, query_data in enumerate(ground_truth, start=1):
        query = query_data["query"]

        print(f"[{number}/{len(ground_truth)}] {query_data['query_id']}: {query}")

        query_embedding = encode_query(query, model, tokenizer, device)
        results = rank_clips_by_best_frame(
            query_embedding,
            frame_embeddings,
            metadatas,
        )

        retrieved_ids = [result["clip_id"] for result in results]
        precision, recall, f1, hit = calculate_metrics(
            retrieved_ids,
            query_data["ground_truth"],
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        hit_scores.append(hit)

        rows.append({
            "row_type": "QUERY RESULT",
            "model_name": MODEL_LABEL,
            "pretrained_checkpoint": PRETRAINED,
            "retrieval_type": "Frame-level cosine similarity; best frame per clip",
            "query_id": query_data["query_id"],
            "query": query,
            "ground_truth_clips": " | ".join(query_data["ground_truth"]),
            "retrieved_top_5_clips": " | ".join(retrieved_ids),
            "best_frame_indices": " | ".join(
                str(result["best_frame_index"]) for result in results
            ),
            "precision_at_5": f"{precision:.4f}",
            "recall_at_5": f"{recall:.4f}",
            "f1_at_5": f"{f1:.4f}",
            "hit_at_5": hit,
        })

    total = len(ground_truth)
    summary = {
        "row_type": "AVERAGE RESULTS",
        "model_name": MODEL_LABEL,
        "pretrained_checkpoint": PRETRAINED,
        "retrieval_type": "Frame-level cosine similarity; best frame per clip",
        "query_id": f"All {total} queries",
        "query": "Average of all evaluation queries",
        "ground_truth_clips": "",
        "retrieved_top_5_clips": "",
        "best_frame_indices": "",
        "precision_at_5": f"{sum(precision_scores) / total:.4f}",
        "recall_at_5": f"{sum(recall_scores) / total:.4f}",
        "f1_at_5": f"{sum(f1_scores) / total:.4f}",
        "hit_at_5": f"{sum(hit_scores) / total:.4f}",
    }

    write_csv(OUTPUT_FILE, rows + [summary])

    print("\n" + "=" * 92)
    print("FRAME-LEVEL COSINE-SIMILARITY RESULTS")
    print("=" * 92)
    print(f"Average Precision@5:  {summary['precision_at_5']}")
    print(f"Average Recall@5:     {summary['recall_at_5']}")
    print(f"Average F1@5:         {summary['f1_at_5']}")
    print(f"Average Hit@5:        {summary['hit_at_5']}")
    print(f"Saved results to:     {OUTPUT_FILE}")
    print("=" * 92)


if __name__ == "__main__":
    main()
