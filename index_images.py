import os
import time

from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings

IMAGE_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/001_RAW_IMAGES"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "images"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_PROMPT = "What is shown in this image?"
MAX_IMAGES = 200
RATE_LIMIT_DELAY = 0.5

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

genai_client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=None,
)

# Collect image files sorted by most recent first
all_files = [
    f for f in os.scandir(IMAGE_DIR)
    if f.is_file() and os.path.splitext(f.name)[1].lower() in MIME_MAP
]
all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
to_index = all_files[:MAX_IMAGES]

# Skip already-indexed files
existing_ids = set(collection.get(include=[])["ids"])
to_index = [f for f in to_index if f.name not in existing_ids]

print(f"Images to index: {len(to_index)} (skipping {MAX_IMAGES - len(to_index)} already indexed)")

for i, entry in enumerate(to_index, 1):
    ext = os.path.splitext(entry.name)[1].lower()
    mime_type = MIME_MAP[ext]

    try:
        with open(entry.path, "rb") as f:
            image_bytes = f.read()

        response = genai_client.models.embed_content(
            model=EMBED_MODEL,
            contents=[EMBED_PROMPT, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        )
        vector = list(response.embeddings[0].values)

        collection.upsert(
            ids=[entry.name],
            embeddings=[vector],
            metadatas=[{"filename": entry.name, "path": entry.path}],
        )

        if i % 10 == 0 or i == len(to_index):
            print(f"[{i}/{len(to_index)}] Indexed: {entry.name}")

        time.sleep(RATE_LIMIT_DELAY)

    except Exception as e:
        if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
            print(f"Rate limit hit, sleeping 60s...")
            time.sleep(60)
        else:
            print(f"Error on {entry.name}: {e}")

print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} images.")
