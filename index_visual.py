import os
from pathlib import Path

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

IMAGE_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/001_RAW_IMAGES"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "visual"
MODEL_NAME = "apple/aimv2-3B-patch14-336"
BATCH_SIZE = 4

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

# ── Device ────────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} …")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
print("Model ready.")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
chroma_client = chromadb.PersistentClient(
    path=DB_PATH,
    settings=Settings(anonymized_telemetry=False),
)
collection = chroma_client.get_or_create_collection(
    COLLECTION_NAME,
    embedding_function=None,
    metadata={"hnsw:space": "cosine"},
)

# ── File list ─────────────────────────────────────────────────────────────────
all_files = [
    f for f in os.scandir(IMAGE_DIR)
    if f.is_file() and Path(f.name).suffix.lower() in MIME_MAP
]
all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

existing_ids = set(collection.get(include=[])["ids"])
to_index = [f for f in all_files if f.name not in existing_ids]
print(f"Images to index: {len(to_index)} (skipping {len(all_files) - len(to_index)} already indexed)")


# ── Embed ─────────────────────────────────────────────────────────────────────
def embed_batch(entries):
    images, valid_entries = [], []
    for entry in entries:
        try:
            images.append(Image.open(entry.path).convert("RGB"))
            valid_entries.append(entry)
        except Exception as e:
            tqdm.write(f"Cannot open {entry.name}: {e}")

    if not images:
        return

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean-pool over patch tokens → (batch, 3072)
        embs = outputs.last_hidden_state.mean(dim=1)
        # L2 normalise so cosine similarity = dot product
        embs = embs / embs.norm(dim=1, keepdim=True)
        embs = embs.cpu().float().numpy()

    collection.upsert(
        ids=[e.name for e in valid_entries],
        embeddings=embs.tolist(),
        metadatas=[{"filename": e.name, "path": e.path} for e in valid_entries],
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
for i in tqdm(range(0, len(to_index), BATCH_SIZE), desc="visual", unit="batch"):
    batch = to_index[i : i + BATCH_SIZE]
    try:
        embed_batch(batch)
    except Exception as e:
        tqdm.write(f"Batch error: {e}")

print(f"\nDone. Collection '{COLLECTION_NAME}' now has {collection.count()} images.")
