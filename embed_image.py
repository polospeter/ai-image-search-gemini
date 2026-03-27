from google import genai
from google.genai import types
import os

client = genai.Client(
    vertexai=True, project="ai-search-agent-491514", location="us-central1",
)

IMAGE_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/001_RAW_IMAGES"

# Pick the latest image file in the directory
image_files = [
    f for f in os.scandir(IMAGE_DIR)
    if f.is_file() and os.path.splitext(f.name)[1].lower() in {".jpg", ".jpeg", ".png", ".webp"}
]
IMAGE_PATH = max(image_files, key=lambda f: f.stat().st_mtime).path

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()

# Detect mime type from extension
ext = os.path.splitext(IMAGE_PATH)[1].lower()
mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
mime_type = mime_map.get(ext, "image/jpeg")

model = "gemini-embedding-2-preview"
response = client.models.embed_content(
    model=model,
    contents=[
        "What is shown in this image?",
        types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type,
        ),
    ],
)
print(f"Image: {os.path.basename(IMAGE_PATH)}")
print(f"Embedding dimensions: {len(response.embeddings[0].values)}")
print(response.embeddings[0])
