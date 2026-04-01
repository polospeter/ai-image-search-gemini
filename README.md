# AI Image Search — Gemini

A local semantic search tool for images and videos powered by Google Gemini multimodal embeddings and ChromaDB. Search your media library with natural language, by dropping an image, or by selecting a predefined character — all through a minimal dark-mode web UI.

## Overview

The system embeds images and video frames using the `gemini-embedding-2-preview` model via Google Vertex AI, stores the resulting vectors in a local ChromaDB instance, and exposes a FastAPI server that the browser UI queries in real time.

Three separate ChromaDB collections serve different search modes:

| Collection | Indexer | Embed Prompt | Purpose |
|---|---|---|---|
| `images` | `index_images.py` | `"What is shown in this image?"` | General content search |
| `characters` | `index_characters.py` | Character appearance description | Character / creature similarity search |
| `videos` | `index_videos.py` | `"What is shown in this image?"` (on first frame) | Video clip search |

---

## Features

- **Text search** — type a natural language query to find matching images
- **Image drop search** — drag and drop (or paste) an image to find visually similar results using the `characters` collection
- **Character search** — select a predefined character (e.g. Hooded Hunter, Tribal Woman, Mecha Wolves) to retrieve the best matching images or videos for that character's visual profile
- **Video search** — find video clips by text query; results show a thumbnail of the first frame with a hover-to-play inline preview
- **Exclusions / flagging** — mark individual results as irrelevant for a character; flagged files are persisted to `exclusions.json` and filtered out of future searches
- **Open in Finder** — click the folder icon on any result to reveal the source file in macOS Finder
- **Incremental indexing** — all indexers skip already-indexed files so re-running is fast

---

## Project Structure

```
.
├── server.py              # FastAPI backend — search endpoints + file serving
├── index_images.py        # Index all images → "images" collection
├── index_characters.py    # Index all images → "characters" collection (character-focused prompt)
├── index_videos.py        # Extract first frames + index videos → "videos" collection
├── static/
│   └── index.html         # Single-page browser UI
├── chroma_db/             # Local ChromaDB vector store (git-ignored)
├── video_frames/          # Extracted first-frame JPEGs for each video (git-ignored)
├── exclusions.json        # Per-character flagged filename lists
└── .gitignore
```

---

## Requirements

- Python 3.11+
- Google Cloud project with Vertex AI enabled
- Application Default Credentials configured (`gcloud auth application-default login`)
- OpenCV (`cv2`) for video frame extraction

Install Python dependencies:

```bash
pip install fastapi uvicorn google-genai chromadb opencv-python tqdm
```

---

## Configuration

Before running the indexers or server, update the directory paths at the top of each script to point to your media:

**`index_images.py` / `index_characters.py`**
```python
IMAGE_DIR = "/path/to/your/images"
```

**`index_videos.py`**
```python
VIDEO_DIR = "/path/to/your/videos"
```

**`server.py`** — update the Vertex AI project and character avatar source directories:
```python
genai_client = genai.Client(vertexai=True, project="your-gcp-project", location="us-central1")

OBSIDIAN_VAULT = "/path/to/your/avatar/source"
CHARACTER_AVATAR_DIRS = [...]
```

---

## Indexing

Run each indexer once (and again whenever you add new files). Already-indexed files are automatically skipped.

```bash
# Index images for text/content search
python index_images.py

# Index images for character/visual-similarity search
python index_characters.py

# Extract first frames and index videos
python index_videos.py
```

All three scripts run with up to 10 concurrent Gemini embedding requests (`CONCURRENCY = 10`) and display a `tqdm` progress bar. Rate-limit errors (HTTP 429) are handled with automatic back-off: 60 s on the first hit, 120 s on the second, then 180 s.

---

## Running the Server

```bash
uvicorn server:app --reload
```

Then open `http://localhost:8000` in your browser.

---

## API Reference

### Search

| Method | Endpoint | Body / Params | Description |
|---|---|---|---|
| `POST` | `/search` | `{query, top_n, min_similarity}` | Text search over the `images` collection |
| `POST` | `/search-videos` | `{query, top_n, min_similarity}` | Text search over the `videos` collection |
| `POST` | `/search-by-image` | `multipart/form-data`: `file`, `top_n`, `min_similarity`, `description` | Image drop search using the `characters` collection |
| `POST` | `/search-by-character` | `{character_id, mode, top_n, min_similarity}` | Search by character profile prompt; `mode` is `"images"` or `"videos"` |

All search endpoints return:
```json
{
  "results": [
    { "filename": "example.jpg", "similarity": 87.3 }
  ]
}
```

Similarity is on a 0–100 scale derived from cosine distance (`similarity = (1 − cosine_distance) × 100`).

### Serving Files

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/image/{filename}` | Serve an indexed image file |
| `GET` | `/frame/{filename}` | Serve an extracted video frame thumbnail |
| `GET` | `/video/{filename}` | Serve the original video file |
| `GET` | `/open/{filename}` | Reveal the image in macOS Finder |
| `GET` | `/open-video/{filename}` | Reveal the video in macOS Finder |

### Counts

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/count` | Number of entries in the `images` collection |
| `GET` | `/count-videos` | Number of entries in the `videos` collection |

### Characters

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/characters` | List all character IDs, names, and full names |
| `GET` | `/character-avatar/{char_id}` | Serve the character's avatar image |

### Exclusions

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/flag` | `{character_id, filename}` | Add a filename to a character's exclusion list |

Exclusions are stored in `exclusions.json` and persist across server restarts. The `/search-by-character` endpoint automatically filters out all flagged filenames for the requested character.

---

## Characters

The following characters are defined in `server.py` and available for the character search mode. Each character has a descriptive prompt used to generate the search embedding.

| ID | Name | Full Name |
|---|---|---|
| `hooded_hunter` | Hooded Hunter | Kael Veyr |
| `steampunk_pirate` | Steampunk Pirate | Steampunk Pirate |
| `tribal_woman` | Tribal Woman | Nara of the White Veil |
| `mecha_wolves` | Mecha Wolves | The Whitefang Pack |
| `iron_huntsman` | Iron Huntsman | Brennan Duskforge |
| `carrier_automata` | Carrier Automata | Mules |
| `frostspine_beast` | Frostspine Beast | Frostspine |
| `valgr_dragon` | Valgr | Valgr, the Last Dragon |

To add a new character, append an entry to the `CHARACTERS` list in `server.py` with `id`, `name`, `full_name`, `avatar` (filename of the avatar image), and `prompt` (the visual description used for embedding).

---

## Similarity Threshold

The default `min_similarity` is **35%**. Results below this threshold are discarded before being returned. You can adjust this per-request via the API, or change the default in the `SearchRequest` / `CharacterSearchRequest` model definitions in `server.py`.

---

## Notes

- The `open` endpoints use `subprocess.Popen(["open", "-R", path])` and are **macOS-only**.
- The ChromaDB store lives in `./chroma_db/` and is excluded from git. Back it up separately if you want to preserve your index.
- `video_frames/` is also excluded from git; it is regenerated automatically when you re-run `index_videos.py`.
- Avatar images are looked up first in the ChromaDB `PATH_LOOKUP` (already-indexed files), then by falling back to the paths listed in `CHARACTER_AVATAR_DIRS`.
