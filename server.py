import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai
import chromadb
from chromadb.config import Settings

DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"
EMBED_MODEL = "gemini-embedding-2-preview"

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

genai_client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME, embedding_function=None)

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
        sim = round(max(0.0, (1.0 - dist ** 2 / 2.0)) * 100, 1)
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


app.mount("/", StaticFiles(directory="static", html=True), name="static")
