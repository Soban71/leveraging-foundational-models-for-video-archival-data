import numpy as np
import open_clip
import torch

from vector_database import load_text_vector_data, load_vector_data

TOP_K = 5
TOP_FRAMES_PER_CLIP = 3

TOP_FRAME_WEIGHT = 0.70
OVERALL_FRAME_WEIGHT = 0.30


# 80% visual and 20% textperforming better
VISUAL_WEIGHT = 0.80
TEXT_WEIGHT = 0.20


def load_openclip():
    # Load the OpenCLIP model 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def encode_query(query, model, tokenizer, device):
    
    # Convert the user's text query into  OpenCLIP embedding.

    tokens = tokenizer([query]).to(device)

    with torch.no_grad():
        query_embedding = model.encode_text(tokens)

    query_embedding = query_embedding / query_embedding.norm(
        dim=-1,
        keepdim=True
    )

    return query_embedding[0].cpu().numpy().astype(np.float32)


def rank_clips(query_embedding, embeddings, metadatas):
    
    # Compare the query with all stored frame embeddings
    # and create one score for each video clip.
    
    frame_embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalise stored frame embeddings
    frame_embeddings = frame_embeddings / np.maximum(
        np.linalg.norm(frame_embeddings, axis=1, keepdims=True),
        1e-12
    )

    # Query is already normalised, but normalise again for safety
    query_embedding = query_embedding / max(
        np.linalg.norm(query_embedding),
        1e-12
    )

    # Cosine similarity
    similarities = frame_embeddings @ query_embedding

    # Store all frame scores for each clip
    clip_scores = {}
    clip_metadata = {}

    for score, metadata in zip(similarities, metadatas):
        clip_id = metadata["clip_id"]

        if clip_id not in clip_scores:
            clip_scores[clip_id] = []
            clip_metadata[clip_id] = metadata

        clip_scores[clip_id].append(float(score))

    results = []

    for clip_id, scores in clip_scores.items():
        scores.sort(reverse=True)

        strongest_scores = scores[:TOP_FRAMES_PER_CLIP]

        top_frame_average = float(np.mean(strongest_scores))
        overall_average = float(np.mean(scores))

        final_score = (
            TOP_FRAME_WEIGHT * top_frame_average
            + OVERALL_FRAME_WEIGHT * overall_average
        )

        results.append({
            "clip_id": clip_id,
            "visual_score": final_score,
            "metadata": clip_metadata[clip_id]
        })

    return results


def rank_transcripts(query_embedding, embeddings, metadatas):
    """Calculate an OpenCLIP text similarity score for each clip transcript."""
    text_embeddings = np.asarray(embeddings, dtype=np.float32)
    text_embeddings = text_embeddings / np.maximum(
        np.linalg.norm(text_embeddings, axis=1, keepdims=True), 1e-12
    )
    query_embedding = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
    similarities = text_embeddings @ query_embedding

    return {
        metadata["clip_id"]: float(score)
        for score, metadata in zip(similarities, metadatas)
    }


def combine_visual_and_text(visual_results, text_scores):
    """Combine visual and transcript retrieval using the configured weights."""
    results = []
    for visual_result in visual_results:
        clip_id = visual_result["clip_id"]
        visual_score = visual_result["visual_score"]
        text_score = text_scores.get(clip_id, 0.0)
        final_score = VISUAL_WEIGHT * visual_score + TEXT_WEIGHT * text_score

        results.append({
            "clip_id": clip_id,
            "visual_score": visual_score,
            "text_score": text_score,
            "final_score": final_score,
            "metadata": visual_result["metadata"],
        })

    results.sort(
        key=lambda result: result["final_score"],
        reverse=True
    )

    return results[:TOP_K]


def search(query, model, tokenizer, device, embeddings, metadatas,
           text_embeddings, text_metadatas):

    # Run one video search query and return the top 5 clips.

    query_embedding = encode_query(
        query,
        model,
        tokenizer,
        device
    )

    visual_results = rank_clips(
        query_embedding,
        embeddings,
        metadatas
    )
    text_scores = rank_transcripts(
        query_embedding,
        text_embeddings,
        text_metadatas,
    )
    return combine_visual_and_text(visual_results, text_scores)
def display_results(results):
    print("\n" + "=" * 80)
    print("TOP RETRIEVED CLIPS")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        metadata = result["metadata"]

        print(f"\nRank:        {rank}")
        print(f"Clip ID:     {result['clip_id']}")
        print(f"Clip file:   {metadata.get('clip_file', 'Not available')}")
        print(f"Clip path:   {metadata.get('clip_path', 'Not available')}")
        print(f"Visual score:{result['visual_score']:.4f}")
        print(f"Text score:  {result['text_score']:.4f}")
        print(f"Final score: {result['final_score']:.4f}")

        print("-" * 80)

def main():
    embeddings, metadatas = load_vector_data()
    text_embeddings, text_metadatas = load_text_vector_data()
    model, tokenizer, device = load_openclip()

    print("\nVideo search is ready.")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Enter your search query: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(
            query,
            model,
            tokenizer,
            device,
            embeddings,
            metadatas,
            text_embeddings,
            text_metadatas,
        )

        display_results(results)


if __name__ == "__main__":
    main()