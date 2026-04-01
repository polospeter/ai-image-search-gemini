import json
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
VIDEO_COLLECTION_NAME = "videos"
FRAMES_DIR = "./video_frames"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_PROMPT = "What is shown in this image?"
CHARACTER_EMBED_PROMPT = "Describe the main character, creature, or person in this image, including their appearance, species, design, and any distinctive visual features."

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
VIDEO_MIME_MAP = {".mp4": "video/mp4", ".mov": "video/quicktime"}
VISUAL_COLLECTION_NAME = "visual"
VISUAL_MODEL_NAME = "apple/aimv2-3B-patch14-336"

OBSIDIAN_VAULT = "/Users/peter.polos/Documents/Obsidian Vault"
CHARACTER_AVATAR_DIRS = [
    os.path.join(OBSIDIAN_VAULT, "Age of Automata", "Characters", "attachments"),
    os.path.join(OBSIDIAN_VAULT, "Age of Automata", "Characters"),
    os.path.join(OBSIDIAN_VAULT, "Age of Automata"),
]

CHARACTERS = [
    {
        "id": "hooded_hunter",
        "name": "Hooded Hunter",
        "full_name": "Kael Veyr",
        "avatar": "20260218_1052_Image Generation_remix_01khr2kdfne88bgwvafj4hyfad.png",
        "prompt": "frontal cinematic portrait of a hooded hunter in a frozen wasteland, half human half machine, pale gaunt face, sleepless eyes, exposed metal plates and seam welds along the cheek and neck, dark weathered hood dusted with snow, ominous, hyperreal",
    },
    {
        "id": "steampunk_pirate",
        "name": "Steampunk Pirate",
        "full_name": "Steampunk Pirate",
        "avatar": "20251212_2330_Steampunk Pirate on Ice_simple_compose_01kcaawe22ejartqjpx8n4f0ea 1.png",
        "prompt": "steampunk pirate standing on ice in a frozen wasteland, dramatic steampunk costume, goggles, mechanical details, cold environment",
    },
    {
        "id": "tribal_woman",
        "name": "Tribal Woman",
        "full_name": "Nara of the White Veil",
        "avatar": "Zoom_in_on_2k_202602222207.jpeg",
        "prompt": "wide cinematic shot of a pale tribal oracle woman in a blizzard, long white hair, yellow eyes, intricate blue ritual tattoos across face and body, fur-and-leather primitive armor, calm but dangerous expression, frozen mountains in background, hyperreal, shallow depth of field",
    },
    {
        "id": "mecha_wolves",
        "name": "Mecha Wolves",
        "full_name": "The Whitefang Pack",
        "avatar": "close_up_shot_202603211155.jpg",
        "prompt": "cinematic shot of robotic wolves in a snowy forest at night, skeletal mechanical bodies, frosted steel limbs, glowing amber eyes, sleek black metal plates, predatory pack formation, cold blue moonlight, hyperreal, menacing",
    },
    {
        "id": "iron_huntsman",
        "name": "Iron Huntsman",
        "full_name": "Brennan Duskforge",
        "avatar": "realistic_extreme_close_202603231808.png",
        "prompt": "cinematic portrait of a weathered monster hunter in a frozen wasteland, wide-brimmed leather hat with rivets dusted in snow, dark round steampunk goggles, crimson scarf wrapped around the lower face, heavy black tattered cloak billowing in blizzard wind, two fully mechanical prosthetic arms of rusted scratched steel with articulated fingers, scoped hunting rifle, vials and pouches on belt, dark shoulder armor plates, hyperreal, gritty, ominous",
    },
    {
        "id": "carrier_automata",
        "name": "Carrier Automata",
        "full_name": "Mules",
        "avatar": "Remove_the_flames_in_the_robots_eyes_2k_delpmaspu_2.png",
        "prompt": "cinematic shot of a simple humanoid robot in a dark snowy forest at night, rusted bronze-copper plating, round goggle-like eyes, small furnace glowing with fire in the chest cavity, heavy backpack with rolled blanket and rope, utilitarian design, no weapons, snow falling, hyperreal, warm amber glow against cold blue tones",
    },
    {
        "id": "frostspine_beast",
        "name": "Frostspine Beast",
        "full_name": "Frostspine",
        "avatar": "Put_this_scary_2k_202602121207.png",
        "prompt": "cinematic wide shot of a massive white wolf-like beast in a frozen birch forest, pale white fur, blank white eyes, bony frost-covered spines and branch-like antlers erupting from the skull and spine, snarling teeth, low aggressive stance, deep snow, blizzard, hyperreal, terrifying, desaturated cold tones",
    },
    {
        "id": "valgr_dragon",
        "name": "Valgr",
        "full_name": "Valgr, the Last Dragon",
        "avatar": "Whisk_d607561e36d6e3eb23840a2fa743a988eg.png",
        "prompt": "cinematic shot of a massive dark serpentine dragon head emerging from thick fog, dark blue-black scales, single glowing amber eye, long tendrils and horn-like protrusions flowing from the skull, enormous scale against frozen mountain ruins and crumbling stone pillars, cold desaturated blue atmosphere, hyperreal, ominous, mythic",
    },
]

genai_client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})
char_collection = chroma_client.get_or_create_collection(CHARACTER_COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})
video_collection = chroma_client.get_or_create_collection(VIDEO_COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})
visual_collection = chroma_client.get_or_create_collection(VISUAL_COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"})

# Build filename → path lookups from ChromaDB metadata
_meta = collection.get(include=["metadatas"])
PATH_LOOKUP = {m["filename"]: m["path"] for m in _meta["metadatas"]}

_video_meta = video_collection.get(include=["metadatas"])
VIDEO_PATH_LOOKUP  = {m["filename"]: m["video_path"] for m in _video_meta["metadatas"]}
VIDEO_FRAME_LOOKUP = {m["filename"]: m["frame_path"] for m in _video_meta["metadatas"]}

_visual_meta = visual_collection.get(include=["metadatas"])
VISUAL_PATH_LOOKUP = {m["filename"]: m["path"] for m in _visual_meta["metadatas"]}

EXCLUSIONS_PATH = "./exclusions.json"


def _load_exclusions() -> dict:
    try:
        with open(EXCLUSIONS_PATH) as f:
            raw = json.load(f)
        return {cid: set(fns) for cid, fns in raw.items()}
    except FileNotFoundError:
        return {}


exclusions: dict[str, set] = _load_exclusions()

# ── AIMv2 visual model (lazy-loaded on first request) ─────────────────────────
_aim_model = None
_aim_processor = None


def _get_aim_model():
    global _aim_model, _aim_processor
    if _aim_model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        _aim_processor = AutoImageProcessor.from_pretrained(VISUAL_MODEL_NAME)
        _aim_model = AutoModel.from_pretrained(VISUAL_MODEL_NAME)
        _device = "mps" if torch.backends.mps.is_available() else "cpu"
        _aim_model = _aim_model.to(_device).eval()
    return _aim_model, _aim_processor


app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    top_n: int = 12
    min_similarity: float = 35.0  # percent, 0–100


class CharacterSearchRequest(BaseModel):
    character_id: str
    mode: str = "images"  # "images" or "videos"
    top_n: int = 30
    min_similarity: float = 35.0


class FlagRequest(BaseModel):
    character_id: str
    filename: str


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


@app.post("/search-videos")
async def search_videos(req: SearchRequest):
    response = genai_client.models.embed_content(model=EMBED_MODEL, contents=[req.query])
    query_vector = list(response.embeddings[0].values)

    fetch_n = min(req.top_n * 3, video_collection.count())
    if fetch_n == 0:
        return {"results": []}

    results = video_collection.query(
        query_embeddings=[query_vector],
        n_results=fetch_n,
        include=["metadatas", "distances"],
    )

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        sim = round(max(0.0, (1.0 - dist)) * 100, 1)
        if sim < req.min_similarity:
            continue
        output.append({"filename": meta["filename"], "similarity": sim})
        if len(output) == req.top_n:
            break
    return {"results": output}


@app.get("/frame/{filename:path}")
async def serve_frame(filename: str):
    frame_path = VIDEO_FRAME_LOOKUP.get(filename)
    if not frame_path or not os.path.exists(frame_path):
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(frame_path, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})


@app.get("/video/{filename:path}")
async def serve_video(filename: str):
    path = VIDEO_PATH_LOOKUP.get(filename)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    ext = Path(path).suffix.lower()
    mime = VIDEO_MIME_MAP.get(ext, "video/mp4")
    return FileResponse(path, media_type=mime)


@app.get("/open-video/{filename:path}")
async def open_video_in_finder(filename: str):
    path = VIDEO_PATH_LOOKUP.get(filename)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    subprocess.Popen(["open", "-R", path])
    return {"ok": True}


@app.get("/count-videos")
async def count_videos():
    return {"count": video_collection.count()}


@app.get("/characters")
async def get_characters():
    return {
        "characters": [
            {"id": c["id"], "name": c["name"], "full_name": c["full_name"]}
            for c in CHARACTERS
        ]
    }


@app.get("/character-avatar/{char_id}")
async def character_avatar(char_id: str):
    char = next((c for c in CHARACTERS if c["id"] == char_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    avatar_fn = char["avatar"]

    # Try indexed collection first
    if avatar_fn in PATH_LOOKUP:
        path = PATH_LOOKUP[avatar_fn]
        if os.path.exists(path):
            ext = Path(path).suffix.lower()
            return FileResponse(path, media_type=MIME_MAP.get(ext, "image/jpeg"), headers={"Cache-Control": "max-age=86400"})

    # Fall back to Obsidian vault directories
    for dir_path in CHARACTER_AVATAR_DIRS:
        path = os.path.join(dir_path, avatar_fn)
        if os.path.exists(path):
            ext = Path(path).suffix.lower()
            return FileResponse(path, media_type=MIME_MAP.get(ext, "image/jpeg"), headers={"Cache-Control": "max-age=86400"})

    raise HTTPException(status_code=404, detail="Avatar image not found")


@app.post("/search-by-character")
async def search_by_character(req: CharacterSearchRequest):
    char = next((c for c in CHARACTERS if c["id"] == req.character_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    response = genai_client.models.embed_content(model=EMBED_MODEL, contents=[char["prompt"]])
    query_vector = list(response.embeddings[0].values)

    target = video_collection if req.mode == "videos" else collection
    count = target.count()
    if count == 0:
        return {"results": []}

    fetch_n = min(req.top_n * 3, count)
    results = target.query(
        query_embeddings=[query_vector],
        n_results=fetch_n,
        include=["metadatas", "distances"],
    )

    char_exclusions = exclusions.get(req.character_id, set())
    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        if meta["filename"] in char_exclusions:
            continue
        sim = round(max(0.0, (1.0 - dist)) * 100, 1)
        if sim < req.min_similarity:
            continue
        output.append({"filename": meta["filename"], "similarity": sim})
        if len(output) == req.top_n:
            break

    return {"results": output}


@app.post("/flag")
async def flag_image(req: FlagRequest):
    if req.character_id not in {c["id"] for c in CHARACTERS}:
        raise HTTPException(status_code=400, detail="Unknown character_id")
    if req.filename not in PATH_LOOKUP:
        raise HTTPException(status_code=400, detail="Unknown filename")
    exclusions.setdefault(req.character_id, set()).add(req.filename)
    with open(EXCLUSIONS_PATH, "w") as f:
        json.dump({cid: list(fns) for cid, fns in exclusions.items()}, f, indent=2)
    return {"ok": True}


@app.post("/search-by-visual-image")
async def search_by_visual_image(
    file: UploadFile = File(...),
    top_n: int = Form(30),
    min_similarity: float = Form(30.0),
):
    import asyncio
    import torch
    import numpy as np
    from PIL import Image as PILImage
    import io

    image_bytes = await file.read()
    img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")

    aim_model, aim_processor = await asyncio.get_event_loop().run_in_executor(
        None, _get_aim_model
    )

    def _embed():
        inputs = aim_processor(images=img, return_tensors="pt")
        device = next(aim_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = aim_model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)[0]
            emb = emb / emb.norm()
        return emb.cpu().float().numpy().tolist()

    query_vector = await asyncio.get_event_loop().run_in_executor(None, _embed)

    count = visual_collection.count()
    if count == 0:
        return {"results": []}

    fetch_n = min(top_n * 3, count)
    results = visual_collection.query(
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


@app.get("/count-visual")
async def count_visual():
    return {"count": visual_collection.count()}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
