import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import chromadb
from chromadb.config import Settings
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

IMAGE_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/001_RAW_IMAGES"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "visual"
MODEL_NAME = "apple/aimv2-3B-patch14-336"
BATCH_SIZE = 4
GEM_P = 3.0
SAM3_PROMPT = "character, person, creature, animal, robot"
BG_COLOR = (128, 128, 128)  # neutral grey

MIME_MAP = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

# ── Device ────────────────────────────────────────────────────────────────────
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

# ── SAM3 ──────────────────────────────────────────────────────────────────────
print("Loading SAM3 …")
_sam3_model = build_sam3_image_model()
_sam3_processor = Sam3Processor(_sam3_model, confidence_threshold=0.5)
print("SAM3 ready.")

# ── AIMv2 ─────────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} …")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
print("AIMv2 ready.")

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


# ── Subject extraction ────────────────────────────────────────────────────────
def extract_subject(img: Image.Image) -> Image.Image:
    state = _sam3_processor.set_image(img)
    state = _sam3_processor.set_text_prompt(SAM3_PROMPT, state)
    masks = state.get("masks", [])
    scores = state.get("scores", [])
    if not len(masks):
        return img  # fallback: embed full image

    best_idx = int(np.argmax(scores))
    mask = np.array(masks[best_idx]).squeeze()  # ensure (H, W)
    mask_img = Image.fromarray((mask * 255).astype("uint8"), mode="L")
    rgba = img.convert("RGBA")
    rgba.putalpha(mask_img)
    bg = Image.new("RGB", img.size, BG_COLOR)
    bg.paste(rgba, mask=mask_img)
    return bg


# ── Embed ─────────────────────────────────────────────────────────────────────
def embed_batch(entries):
    images, valid_entries = [], []
    for entry in entries:
        try:
            img = Image.open(entry.path).convert("RGB")
            images.append(extract_subject(img))
            valid_entries.append(entry)
        except Exception as e:
            tqdm.write(f"Cannot open {entry.name}: {e}")

    if not images:
        return

    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        patches = outputs.last_hidden_state  # (batch, 576, 3072)
        embs = patches.clamp(min=1e-6).pow(GEM_P).mean(dim=1).pow(1.0 / GEM_P)
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
