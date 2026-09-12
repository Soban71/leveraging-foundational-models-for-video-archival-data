import json
import socket
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from search import load_openclip, search
from vector_database import load_text_vector_data, load_vector_data


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
TOP_K = 5
TRANSCRIPT_FOLDER = "transcripts"
CLIP_DESCRIPTION_FILES = [
    "clip_descriptions.json",
    "clip_descriptions(9).json",
]
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.1:8b"
MAX_TRANSCRIPT_CHARACTERS = 700
OLLAMA_TIMEOUT_SECONDS = 900
OLLAMA_CONTEXT_WINDOW = 4096
MAX_ANSWER_TOKENS = 300


def load_clip_descriptions():
    """Load manually produced visual descriptions and timestamps, if present."""
    path = next(
        (Path(name) for name in CLIP_DESCRIPTION_FILES if Path(name).is_file()),
        None,
    )
    if path is None:
        names = " or ".join(f"'{name}'" for name in CLIP_DESCRIPTION_FILES)
        print(f"Warning: {names} was not found.")
        return {}

    with path.open("r", encoding="utf-8") as file:
        items = json.load(file)
    return {item["clip_id"]: item for item in items}


def read_transcript(clip_id):
    """Read the text generated from subtitles or Whisper for one clip."""
    path = Path(TRANSCRIPT_FOLDER) / f"{clip_id}.txt"
    if not path.is_file():
        return "No transcript is available for this clip."

    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return text[:MAX_TRANSCRIPT_CHARACTERS] or "No speech was detected for this clip."


def seconds_to_time(seconds):
    seconds = int(seconds or 0)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_context(results, descriptions):
    """Turn the retrieved clips into evidence for the LLM."""
    sections = []

    for rank, result in enumerate(results, start=1):
        clip_id = result["clip_id"]
        description = descriptions.get(clip_id, {})
        visual_description = description.get("description", "No visual description available.")
        start = seconds_to_time(description.get("start_time_seconds"))
        end = seconds_to_time(description.get("end_time_seconds"))
        transcript = read_transcript(clip_id)

        sections.append(
            f"SOURCE {rank}\n"
            f"Clip ID: {clip_id}\n"
            f"Time: {start} to {end}\n"
            f"Visual description: {visual_description}\n"
            f"Transcript: {transcript}"
        )

    return "\n\n".join(sections)


def build_prompt(question, context):
    return f"""You answer questions about a video archive.

Use only the retrieved sources below. The \"Visual description\" field is
valid evidence about what can be seen in a clip, and the \"Transcript\" field
is valid evidence about what is said in a clip. If either field directly
answers the question, answer it clearly. For example, if a visual description
says children are practising football, answer that children are playing or
practising football. Do not say that you cannot answer when a source directly
supports the answer. Only say \"I cannot answer this from the retrieved clips.\"
when none of the sources contains relevant evidence. Do not invent facts.

Give a clear answer in 2 to 4 sentences. Include relevant visual details and
spoken information from the transcripts. Include a clip timestamp when it is
available and useful. At the end, add a line called \"Sources:\" and list the
exact Clip ID(s) you used, not source numbers such as \"SOURCE 1\".

Question: {question}

Retrieved sources:
{context}
"""


def generate_answer(prompt):
    """Call a local Ollama server running Llama 3.1 8B."""
    payload = json.dumps({
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": OLLAMA_CONTEXT_WINDOW,
            "num_predict": MAX_ANSWER_TOKENS,
        },
    }).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # An 8B model can take several minutes when it runs on a CPU.
        with urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("response", "The model did not return an answer.").strip()
    except (URLError, TimeoutError, socket.timeout) as error:
        raise RuntimeError(
            f"Ollama did not return an answer within {OLLAMA_TIMEOUT_SECONDS // 60} "
            "minutes. Make sure Ollama is running and that your computer has "
            f"enough memory for {LLM_MODEL}. Details: {error}"
        ) from error


def answer_question(question, model, tokenizer, device, visual_embeddings,
                    visual_metadatas, text_embeddings, text_metadatas,
                    descriptions):
    results = search(
        question,
        model,
        tokenizer,
        device,
        visual_embeddings,
        visual_metadatas,
        text_embeddings,
        text_metadatas,
    )

    context = create_context(results, descriptions)
    answer = generate_answer(build_prompt(question, context))
    return answer, results


def main():
    visual_embeddings, visual_metadatas = load_vector_data()
    text_embeddings, text_metadatas = load_text_vector_data()
    model, tokenizer, device = load_openclip()
    descriptions = load_clip_descriptions()

    print("\nVideo RAG is ready. Type 'exit' to stop.")
    print(f"Retrieval: 70% visual + 30% text | Generator: {LLM_MODEL}\n")

    while True:
        question = input("Ask a question about the video archive: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        try:
            answer, results = answer_question(
                question, model, tokenizer, device,
                visual_embeddings, visual_metadatas,
                text_embeddings, text_metadatas, descriptions,
            )
        except RuntimeError as error:
            print(f"\nRAG error: {error}\n")
            continue

        print("\n" + "=" * 80)
        print("ANSWER")
        print("=" * 80)
        print(answer)
        print("\nRETRIEVED CLIPS")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result['clip_id']} (score: {result['final_score']:.4f})")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()