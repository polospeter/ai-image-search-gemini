import asyncio
import os
from tqdm.asyncio import tqdm

from google import genai
from google.genai import types
import chromadb
from chromadb.config import Settings

IMAGE_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/001_RAW_IMAGES"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "characters"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_PROMPT = "Describe the main character, creature, or person in this image, including their appearance, species, design, and any distinctive visual features."
CONCURRENCY = 10

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

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

# Collect image files sorted by most recent first
all_files = [
    f for f in os.scandir(IMAGE_DIR)
    if f.is_file() and os.path.splitext(f.name)[1].lower() in MIME_MAP
]
all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
to_index = all_files

# Skip already-indexed files
existing_ids = set(collection.get(include=[])["ids"])
to_index = [f for f in to_index if f.name not in existing_ids]

print(f"Images to index: {len(to_index)} (skipping {len(all_files) - len(to_index)} already indexed)")

semaphore = None  # set in main()


async def embed_and_store(entry, progress):
    ext = os.path.splitext(entry.name)[1].lower()
    mime_type = MIME_MAP[ext]

    async with semaphore:
        with open(entry.path, "rb") as f:
            image_bytes = f.read()

        for attempt in range(3):
            try:
                response = await genai_client.aio.models.embed_content(
                    model=EMBED_MODEL,
                    contents=[EMBED_PROMPT, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
                )
                vector = list(response.embeddings[0].values)
                collection.upsert(
                    ids=[entry.name],
                    embeddings=[vector],
                    metadatas=[{"filename": entry.name, "path": entry.path}],
                )
                progress.update(1)
                return
            except Exception as e:
                if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                    wait = 60 * (attempt + 1)
                    progress.write(f"Rate limit hit, sleeping {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    progress.write(f"Error on {entry.name}: {e}")
                    return


async def main():
    global semaphore
    semaphore = asyncio.Semaphore(CONCURRENCY)
    with tqdm(total=len(to_index), desc="characters", unit="img") as progress:
        await asyncio.gather(*[embed_and_store(entry, progress) for entry in to_index])
    print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} images.")


asyncio.run(main())
