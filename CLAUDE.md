# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the server
uvicorn server:app --reload

# Index images (run each once, re-run to pick up new files)
python3 index_images.py        # general content search → 'images' collection
python3 index_characters.py    # character/creature search → 'characters' collection
python3 index_videos.py        # sampled video search (default: every 2 seconds) → 'videos' collection
python3 index_visual.py        # pure visual similarity → 'visual' collection (AIMv2, local)

# Install dependencies
pip3 install fastapi uvicorn google-genai chromadb opencv-python tqdm transformers torch torchvision pillow
```

## Architecture

**Three-layer stack:** indexing scripts → ChromaDB vector store → FastAPI + vanilla JS frontend.

### Collections

| Collection | Indexer | Embedding backend | Prompt / method |
|---|---|---|---|
| `images` | `index_images.py` | Gemini `gemini-embedding-2-preview` (Vertex AI) | `"What is shown in this image?"` |
| `characters` | `index_characters.py` | Gemini | Character appearance description |
| `videos` | `index_videos.py` | Gemini | Sampled frames throughout each video |
| `visual` | `index_visual.py` | AIMv2 3B (local, MPS/CPU) | Mean-pooled patch tokens, L2-normalised |

All collections use cosine space (`hnsw:space: cosine`). Similarity is computed as `(1 − cosine_distance) × 100`.

### server.py

- Loads all four ChromaDB collections and builds filename→path dicts (`PATH_LOOKUP`, `VIDEO_PATH_LOOKUP`, etc.) at startup.
- AIMv2 model is **lazy-loaded** on first `/search-by-visual-image` request and cached in module globals.
- Gemini client connects to Vertex AI (`project="ai-search-agent-491514"`, `location="us-central1"`); requires Application Default Credentials (`gcloud auth application-default login`).
- `exclusions.json` persists per-character flagged filenames; loaded at startup, updated live via `POST /flag`.
- `CHARACTERS` list (8 entries) in server.py is the source of truth for character search — add new characters here.
- Static files are served last via `app.mount("/", StaticFiles(...))` — all API routes must be registered before that line.

### Indexers

- All skip already-indexed files by checking `collection.get(include=[])["ids"]`.
- `index_images.py` / `index_characters.py`: async concurrent Gemini calls (concurrency=10), tqdm progress, 3-attempt backoff on 429.
- `index_videos.py`: OpenCV samples every `--interval` seconds (default 2), plus the last frame; Gemini embeds samples concurrently. IDs include a file/interval signature, reruns resume, and successful upgrades remove old records. `video_search.py` deduplicates matches to the best frame per clip. Repeat `--video-dir` and add `--recursive` for scattered folders; `--dry-run` lists scope. Metadata `video_id` identifies each full source path, preserving same-named clips.
- `index_visual.py`: sync batched inference (batch_size=4) using HuggingFace `transformers`, MPS device on Apple Silicon.

### Frontend (`static/index.html`)

Single-file vanilla JS, no build step. Four search modes dispatched from the form submit handler:

- **Images** — `POST /search` (Gemini text query)
- **Videos** — `POST /search-videos` (Gemini text query)
- **Images + image drop** — `POST /search-by-image` (Gemini multimodal, uses `characters` collection)
- **Visual** — `POST /search-by-visual-image` (AIMv2, auto-submits on image drop/paste, text input disabled)
- **Character pills** — `POST /search-by-character` (Gemini, filters via `exclusions`)

`renderResults` renders image cards for all non-video modes; video cards show matching timestamps and start hover-to-play inline previews at the matched frame.

### Key paths (hardcoded, update per environment)

- `IMAGE_DIR` in indexers — source images
- `VIDEO_DIR` in `index_videos.py` — source videos
- `OBSIDIAN_VAULT` / `CHARACTER_AVATAR_DIRS` in `server.py` — avatar image fallback directories
- `DB_PATH = "./chroma_db"` — relative to working directory; always start uvicorn from repo root
