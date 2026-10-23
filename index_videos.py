"""Index sampled video frames; reruns resume and upgrade first-frame indexes."""

import argparse
import asyncio
import hashlib
import math
from pathlib import Path

from video_search import sample_frames

VIDEO_DIR = "/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/002_RAW_VIDEOS"
FRAMES_DIR = "./video_frames"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "videos"
EMBED_MODEL = "gemini-embedding-2-preview"
EMBED_PROMPT = "What is shown in this image?"
CONCURRENCY = 10
VIDEO_EXTS = {".mp4", ".mov"}


async def index_video(path, interval, frames_dir, collection, client, semaphore, progress):
    import cv2
    from google.genai import types

    stat = path.stat()
    # File changes or a different interval produce a new, resumable generation.
    signature = hashlib.sha256(
        f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{interval}:v2".encode()
    ).hexdigest()
    existing = collection.get(where={"video_path": str(path)}, include=["metadatas"])
    existing_ids = set(existing["ids"])
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open video")
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(count):
            raise ValueError("Unknown frame count")
        indices = sample_frames(int(count), fps, interval)
        wanted = {f"{signature}:{index}" for index in indices}
        pending = []
        for index in indices:
            frame_id = f"{signature}:{index}"
            if frame_id in existing_ids:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot read frame {index}; rerun to retry")
            frame_path = frames_dir / f"{signature}-{index}.jpg"
            if not cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise ValueError(f"Cannot save frame {index}")
            pending.append((frame_id, frame_path, index / fps))
    finally:
        cap.release()

    async def embed(frame_id, frame_path, timestamp):
        async with semaphore:
            for attempt in range(3):
                try:
                    response = await client.aio.models.embed_content(
                        model=EMBED_MODEL,
                        contents=[EMBED_PROMPT, types.Part.from_bytes(
                            data=frame_path.read_bytes(), mime_type="image/jpeg")],
                    )
                    collection.upsert(
                        ids=[frame_id], embeddings=[list(response.embeddings[0].values)],
                        metadatas=[{
                            "filename": path.name, "video_path": str(path),
                            "video_id": hashlib.sha256(str(path).encode()).hexdigest(),
                            "frame_path": str(frame_path), "timestamp": timestamp,
                        }],
                    )
                    return True
                except Exception as exc:
                    if attempt < 2 and ("429" in str(exc) or "ResourceExhausted" in type(exc).__name__):
                        await asyncio.sleep(60 * (attempt + 1))
                    else:
                        progress.write(f"Error on {path.name} at {timestamp:.2f}s: {exc}")
                        return False

    outcomes = await asyncio.gather(*(embed(*frame) for frame in pending))
    if not all(outcomes):
        # Keep old records until the replacement is fully indexed.
        return False
    stale = existing_ids - wanted
    if stale:
        collection.delete(ids=sorted(stale))
    progress.write(f"{path.name}: {len(indices)} frames ({len(pending)} new)")
    return True


def collect_videos(directories, recursive=False):
    paths = set()
    for directory in directories:
        entries = directory.rglob("*") if recursive else directory.iterdir()
        paths.update(p.resolve() for p in entries if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    return sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)


async def main(args):
    import chromadb
    from chromadb.config import Settings
    from google import genai
    from tqdm import tqdm

    paths = collect_videos(args.video_dir, args.recursive)
    print(f"Found {len(paths)} videos; sampling every {args.interval:g} seconds.", flush=True)
    if args.dry_run:
        for path in paths:
            print(path)
        return 0
    frames_dir = Path(FRAMES_DIR).resolve()
    frames_dir.mkdir(parents=True, exist_ok=True)
    client = genai.Client(vertexai=True, project="ai-search-agent-491514", location="us-central1")
    db = chromadb.PersistentClient(path=DB_PATH, settings=Settings(anonymized_telemetry=False))
    collection = db.get_or_create_collection(
        COLLECTION_NAME, embedding_function=None, metadata={"hnsw:space": "cosine"},
    )
    semaphore = asyncio.Semaphore(CONCURRENCY)
    failed = 0
    with tqdm(total=len(paths), desc="indexing videos", unit="vid") as progress:
        for path in paths:
            try:
                if not await index_video(path, args.interval, frames_dir, collection, client, semaphore, progress):
                    failed += 1
            except Exception as exc:
                progress.write(f"Error on {path.name}: {exc}")
                failed += 1
            progress.update(1)
    print(f"Done. {collection.count()} indexed frames; {failed} failed videos. Restart the server to reload media paths.")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, action="append", help="Source folder; repeat for multiple folders")
    parser.add_argument("--recursive", action="store_true", help="Include all source subfolders")
    parser.add_argument("--dry-run", action="store_true", help="List videos without embedding or changing the index")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between samples (default: 2)")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be positive and finite")
    args.video_dir = args.video_dir or [Path(VIDEO_DIR)]
    if any(not directory.is_dir() for directory in args.video_dir):
        parser.error("Every --video-dir must be an existing directory")
    raise SystemExit(asyncio.run(main(args)))
