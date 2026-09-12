import json

from search import (
    combine_visual_and_text,
    encode_query,
    load_openclip,
    rank_clips,
    rank_transcripts,
    search,
)
from vector_database import load_text_vector_data, load_vector_data

GROUND_TRUTH_FILE = "evaluation_dataset.json"
TOP_K = 5


def normalise_clip_id(clip_id):
    
    
    clip_id = str(clip_id).replace("\\", "/")
    clip_id = clip_id.split("/")[-1]

    if clip_id.lower().endswith(".mp4"):
        clip_id = clip_id[:-4]

    return clip_id.strip()


def load_ground_truth():
    
    # Load the manually labelled evaluation queries.
    
    with open(
        GROUND_TRUTH_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def calculate_metrics(retrieved_ids, relevant_ids):
    
    # Calculating Precision, Recall, F1 and Hit
    
    retrieved = {
        normalise_clip_id(clip_id)
        for clip_id in retrieved_ids
    }

    relevant = {
        normalise_clip_id(clip_id)
        for clip_id in relevant_ids
    }

    correct = len(retrieved & relevant)

    precision = correct / len(retrieved) if retrieved else 0.0
    recall = correct / len(relevant) if relevant else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    hit = 1 if correct > 0 else 0

    return precision, recall, f1, hit


def evaluate_query(
    query_data,
    model,
    tokenizer,
    device,
    embeddings,
    metadatas,
    text_embeddings,
    text_metadatas
):
    """Compare visual-only, text-only and multimodal retrieval for one query."""
    query = query_data["query"]
    relevant_ids = query_data["ground_truth"]
    query_embedding = encode_query(query, model, tokenizer, device)

    # Visual-only: current frame-ranking method.
    all_visual_results = rank_clips(query_embedding, embeddings, metadatas)
    visual_results = sorted(
        all_visual_results,
        key=lambda result: result["visual_score"],
        reverse=True
    )[:TOP_K]

    # Text-only: rank the subtitle/Whisper transcript of every clip.
    text_scores = rank_transcripts(
        query_embedding, text_embeddings, text_metadatas
    )
    text_results = [
        {
            "clip_id": metadata["clip_id"],
            "text_score": text_scores[metadata["clip_id"]],
            "metadata": metadata,
        }
        for metadata in text_metadatas
    ]
    text_results.sort(key=lambda result: result["text_score"], reverse=True)
    text_results = text_results[:TOP_K]

    # Combined: the existing 70% visual + 30% text method.
    multimodal_results = combine_visual_and_text(all_visual_results, text_scores)

    methods = [
        ("Visual only", visual_results),
        ("Text only", text_results),
        ("Visual + text", multimodal_results),
    ]

    print("\n" + "=" * 80)
    print(f"RETRIEVAL COMPARISON - {query_data['query_id']}")
    print("=" * 80)
    print(f"Query: {query}\n")
    print(f"{'Method':<16} {'Precision@5':>12} {'Recall@5':>10} {'F1@5':>8} {'Hit@5':>8}")
    print("-" * 80)

    for method_name, results in methods:
        retrieved_ids = [result["clip_id"] for result in results]
        precision, recall, f1, hit = calculate_metrics(retrieved_ids, relevant_ids)
        print(
            f"{method_name:<16} {precision:>12.3f} {recall:>10.3f} "
            f"{f1:>8.3f} {hit:>8}"
        )

    print("\nTOP 5 CLIPS")
    for method_name, results in methods:
        clip_ids = [result["clip_id"] for result in results]
        print(f"{method_name:<16}: {' | '.join(clip_ids)}")
    print("=" * 80)


def evaluate_all_queries(
    ground_truth,
    model,
    tokenizer,
    device,
    embeddings,
    metadatas,
    text_embeddings,
    text_metadatas
):
    """Run every query once and show average results for all three methods."""

    scores = {
        "Visual only": {"precision": [], "recall": [], "f1": [], "hit": []},
        "Text only": {"precision": [], "recall": [], "f1": [], "hit": []},
        "Visual + text": {"precision": [], "recall": [], "f1": [], "hit": []},
    }

    print("\nRunning all evaluation queries...")

    for number, query_data in enumerate(ground_truth, start=1):
        query = query_data["query"]
        relevant_ids = query_data["ground_truth"]

        print(
            f"[{number}/{len(ground_truth)}] "
            f"{query_data['query_id']}: {query}"
        )

        query_embedding = encode_query(query, model, tokenizer, device)

        # Visual-only retrieval
        all_visual_results = rank_clips(
            query_embedding,
            embeddings,
            metadatas
        )

        visual_results = sorted(
            all_visual_results,
            key=lambda result: result["visual_score"],
            reverse=True
        )[:TOP_K]

        # Text-only retrieval
        text_scores = rank_transcripts(
            query_embedding,
            text_embeddings,
            text_metadatas
        )

        text_results = [
            {
                "clip_id": metadata["clip_id"],
                "text_score": text_scores.get(metadata["clip_id"], 0.0),
            }
            for metadata in text_metadatas
        ]

        text_results = sorted(
            text_results,
            key=lambda result: result["text_score"],
            reverse=True
        )[:TOP_K]

        # Combined visual + text retrieval
        multimodal_results = combine_visual_and_text(
            all_visual_results,
            text_scores
        )

        methods = [
            ("Visual only", visual_results),
            ("Text only", text_results),
            ("Visual + text", multimodal_results),
        ]

        for method_name, results in methods:
            retrieved_ids = [result["clip_id"] for result in results]

            precision, recall, f1, hit = calculate_metrics(
                retrieved_ids,
                relevant_ids
            )

            scores[method_name]["precision"].append(precision)
            scores[method_name]["recall"].append(recall)
            scores[method_name]["f1"].append(f1)
            scores[method_name]["hit"].append(hit)

    total_queries = len(ground_truth)

    print("\n" + "=" * 80)
    print(f"OVERALL RETRIEVAL EVALUATION RESULTS - ALL {total_queries} QUERIES")
    print("=" * 80)
    print(
        f"{'Method':<16} {'Precision@5':>12} "
        f"{'Recall@5':>10} {'F1@5':>8} {'Hit@5':>8}"
    )
    print("-" * 80)

    for method_name in ["Visual only", "Text only", "Visual + text"]:
        method_scores = scores[method_name]

        average_precision = sum(method_scores["precision"]) / total_queries
        average_recall = sum(method_scores["recall"]) / total_queries
        average_f1 = sum(method_scores["f1"]) / total_queries
        average_hit = sum(method_scores["hit"]) / total_queries

        print(
            f"{method_name:<16} {average_precision:>12.3f} "
            f"{average_recall:>10.3f} {average_f1:>8.3f} "
            f"{average_hit:>8.3f}"
        )

    print("=" * 80)
    


def main():

    # Load ground-truth dataset
    ground_truth = load_ground_truth()

    # Load vector data
    embeddings, metadatas = load_vector_data()
    text_embeddings, text_metadatas = load_text_vector_data()

    # Load OpenCLIP
    model, tokenizer, device = load_openclip()

    print("\n" + "=" * 80)
    print("VIDEO RETRIEVAL EVALUATION")
    print("=" * 80)

    print("\nAvailable evaluation queries:")

    for item in ground_truth:
        print(f"{item['query_id']}: {item['query']}")

    print("\nType a Query ID such as Q01.")
    print("Type 'all' to evaluate every query and show average results.")
    print("Type 'exit' to stop.")

    # Keep running until user types exit
    while True:

        query_id = input(
            "\nEnter Query ID: "
        ).strip()

        if query_id.lower() == "exit":
            print("\nEvaluation closed.")
            break

        if query_id.lower() == "all":
            evaluate_all_queries(
                ground_truth,
                model,
                tokenizer,
                device,
                embeddings,
                metadatas,
                text_embeddings,
                text_metadatas
            )
            continue

        query_id = query_id.upper()

        # Find query in evaluation dataset
        query_data = next(
            (
                item
                for item in ground_truth
                if item["query_id"].upper() == query_id
            ),
            None
        )

        if query_data is None:
            print("Query ID not found. Please try again.")
            continue

        # Run evaluation
        evaluate_query(
            query_data,
            model,
            tokenizer,
            device,
            embeddings,
            metadatas,
            text_embeddings,
            text_metadatas
        )
    


if __name__ == "__main__":
    main()
