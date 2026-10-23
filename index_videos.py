import asyncio
import os

import cv2
from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

VIDEO_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/002_RAW_VIDEOS"
FRAMES_DIR = "./video_frames"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "videos"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_PROMPT = "What is shown in this image?"
CONCURRENCY = 10
VIDEO_EXTS = {".mp4", ".mov"}

genai_client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

os.makedirs(FRAMES_DIR, exist_ok=True)

# Collect video files
all_videos = [
    f for f in os.scandir(VIDEO_DIR)
    if f.is_file() and os.path.splitext(f.name)[1].lower() in VIDEO_EXTS
]
all_videos.sort(key=lambda f: f.stat().st_mtime, reverse=True)

# Skip already-indexed
existing_ids = set(collection.get(include=[])["ids"])
to_process = [v for v in all_videos if v.name not in existing_ids]
print(f"Videos to index: {len(to_process)} (skipping {len(all_videos) - len(to_process)} already indexed)")

# ── Phase 1: Extract first frames (sync) ─────────────────────────────────────
to_embed = []
for entry in tqdm(to_process, desc="extracting frames", unit="vid"):
    frame_path = os.path.join(FRAMES_DIR, entry.name + ".jpg")
    if os.path.exists(frame_path):
        to_embed.append((entry, os.path.abspath(frame_path)))
        continue
    cap = cv2.VideoCapture(entry.path)
    if not cap.isOpened():
        tqdm.write(f"Cannot open: {entry.name} — skipping")
        cap.release()
        continue
    ret, frame = cap.read()
    cap.release()
    if not ret:
        tqdm.write(f"Cannot read frame: {entry.name} — skipping")
        continue
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    to_embed.append((entry, os.path.abspath(frame_path)))

print(f"Frames ready: {len(to_embed)}")

# ── Phase 2: Embed frames (async) ────────────────────────────────────────────
semaphore = None


async def embed_and_store(video_entry, frame_path, progress):
    async with semaphore:
        with open(frame_path, "rb") as f:
            jpeg_bytes = f.read()

        for attempt in range(3):
            try:
                response = await genai_client.aio.models.embed_content(
                    model=EMBED_MODEL,
                    contents=[EMBED_PROMPT, types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")],
                )
                vector = list(response.embeddings[0].values)
                collection.upsert(
                    ids=[video_entry.name],
                    embeddings=[vector],
                    metadatas=[{
                        "filename": video_entry.name,
                        "video_path": video_entry.path,
                        "frame_path": frame_path,
                    }],
                )
                progress.update(1)
                return
            except Exception as e:
                if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                    wait = 60 * (attempt + 1)
                    progress.write(f"Rate limit hit, sleeping {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    progress.write(f"Error on {video_entry.name}: {e}")
                    return


async def main():
    global semaphore
    semaphore = asyncio.Semaphore(CONCURRENCY)
    with atqdm(total=len(to_embed), desc="embedding", unit="vid") as progress:
        await asyncio.gather(*[embed_and_store(entry, fp, progress) for entry, fp in to_embed])
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} videos.")


asyncio.run(main())
