import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings

DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"
CHARACTER_COLLECTION_NAME = "characters"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_PROMPT = "What is shown in this image?"
CHARACTER_EMBED_PROMPT = "Describe the main character, creature, or person in this image, including their appearance, species, design, and any distinctive visual features."

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

genai_client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})
char_collection = chroma_client.get_or_create_collection(CHARACTER_COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})

# Build filename → path lookup from ChromaDB metadata
_meta = collection.get(include=["metadatas"])
PATH_LOOKUP = {m["filename"]: m["path"] for m in _meta["metadatas"]}

app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    top_n: int = 12
    min_similarity: float = 35.0  # percent, 0–100


@app.post("/search")
async def search(req: SearchRequest):
    response = genai_client.models.embed_content(model=EMBED_MODEL, contents=[req.query])
    query_vector = list(response.embeddings[0].values)

    # Fetch more than needed so we can filter by threshold after
    fetch_n = min(req.top_n * 3, collection.count())
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_n,
        include=["metadatas", "distances"],
    )

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        # Gemini embeddings are unit-normalised → L2 dist ∈ [0, 2]
        # cosine_similarity = 1 − dist²/2, mapped to 0–100
        sim = round(max(0.0, (1.0 - dist)) * 100, 1)
        if sim < req.min_similarity:
            continue
        output.append({"filename": meta["filename"], "similarity": sim})
        if len(output) == req.top_n:
            break

    return {"results": output}


@app.get("/image/{filename:path}")
async def serve_image(filename: str):
    path = PATH_LOOKUP.get(filename)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    ext = Path(path).suffix.lower()
    mime = MIME_MAP.get(ext, "image/jpeg")
    return FileResponse(path, media_type=mime, headers={"Cache-Control": "max-age=86400"})


@app.get("/open/{filename:path}")
async def open_in_finder(filename: str):
    path = PATH_LOOKUP.get(filename)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    subprocess.Popen(["open", "-R", path])
    return {"ok": True}


@app.get("/count")
async def count():
    return {"count": collection.count()}


@app.post("/search-by-image")
async def search_by_image(
    file: UploadFile = File(...),
    top_n: int = Form(12),
    min_similarity: float = Form(35.0),
    description: str = Form(""),
):
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    mime = MIME_MAP.get(ext, "image/jpeg")
    image_bytes = await file.read()

    contents = [CHARACTER_EMBED_PROMPT]
    if description.strip():
        contents.append(description.strip())
    contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))

    response = genai_client.models.embed_content(
        model=EMBED_MODEL,
        contents=contents,
    )
    query_vector = list(response.embeddings[0].values)

    fetch_n = min(top_n * 3, char_collection.count()) if char_collection.count() > 0 else min(top_n * 3, collection.count())
    target = char_collection if char_collection.count() > 0 else collection
    results = target.query(
        query_embeddings=[query_vector],
        n_results=fetch_n,
        include=["metadatas", "distances"],
    )

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        sim = round(max(0.0, (1.0 - dist)) * 100, 1)
        if sim < min_similarity:
            continue
        output.append({"filename": meta["filename"], "similarity": sim})
        if len(output) == top_n:
            break

    return {"results": output}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
