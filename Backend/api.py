
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import (
    answer_question,
    load_clip_descriptions,
    load_openclip,
    load_text_vector_data,
    load_vector_data,
)


CLIPS_FOLDER = Path("clips_web")
rag_resources = {}


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)


def get_clip_url(result):
    """Return a safe browser URL for a retrieved clip file."""
    metadata = result.get("metadata", {})
    clip_file = (
        metadata.get("clip_file")
        or metadata.get("clip_path")
        or f"{result['clip_id']}.mp4"
    )
    return f"/api/clip/{quote(Path(str(clip_file)).name)}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models and embeddings once when the server starts."""
    print("Loading RAG resources. This may take a moment...")
    visual_embeddings, visual_metadatas = load_vector_data()
    text_embeddings, text_metadatas = load_text_vector_data()
    model, tokenizer, device = load_openclip()

    rag_resources.update({
        "visual_embeddings": visual_embeddings,
        "visual_metadatas": visual_metadatas,
        "text_embeddings": text_embeddings,
        "text_metadatas": text_metadatas,
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "descriptions": load_clip_descriptions(),
    })
    print("RAG API is ready.")
    yield
    rag_resources.clear()


app = FastAPI(
    title="Video Archive RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

# Lets a React development server (normally port 5173 or 3000) call this API.

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://archivelens-video.janjuasoban846.chatgpt.site",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not CLIPS_FOLDER.is_dir():
    raise RuntimeError("The 'clips' folder was not found beside api.py.")
app.mount("/clips", StaticFiles(directory=str(CLIPS_FOLDER)), name="clips")


@app.get("/api/clip/{clip_file}")
def stream_clip(clip_file: str):
    """Serve a retrieved MP4 file to the browser video player."""
    safe_name = Path(clip_file).name
    clip_path = CLIPS_FOLDER / safe_name
    if not clip_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Clip file was not found: {safe_name}",
        )
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        headers={"Content-Disposition": "inline"},
    )


@app.get("/api/health")
def health_check():
    return {"status": "ready", "generator": "llama3.1:8b"}


def format_clips(results):
    clips = []
    for rank, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        clips.append({
            "rank": rank,
            "clip_id": result["clip_id"],
            "score": round(result["final_score"], 4),
            "visual_score": round(result["visual_score"], 4),
            "text_score": round(result["text_score"], 4),
            "clip_url": get_clip_url(result),
            "clip_file": metadata.get("clip_file"),
        })
    return clips


@app.post("/api/retrieve")
def retrieve_video_clips(request: SearchRequest):
    """Quickly return relevant clips before the slower Llama answer is ready."""
    try:
        from search import search

        results = search(
            request.question,
            rag_resources["model"],
            rag_resources["tokenizer"],
            rag_resources["device"],
            rag_resources["visual_embeddings"],
            rag_resources["visual_metadatas"],
            rag_resources["text_embeddings"],
            rag_resources["text_metadatas"],
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"question": request.question, "clips": format_clips(results)}


@app.post("/api/generate")
def generate_rag_answer(request: AnswerRequest):
    """Generate the slower, grounded Ollama answer after retrieval is visible."""
    try:
        answer, _ = answer_question(
            request.question,
            rag_resources["model"],
            rag_resources["tokenizer"],
            rag_resources["device"],
            rag_resources["visual_embeddings"],
            rag_resources["visual_metadatas"],
            rag_resources["text_embeddings"],
            rag_resources["text_metadatas"],
            rag_resources["descriptions"],
        )
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return {"question": request.question, "answer": answer}
