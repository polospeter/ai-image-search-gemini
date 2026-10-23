"""Sampling and retrieval helpers shared by the video indexer and server."""

import math
from pathlib import Path


def video_key(metadata):
    return metadata.get("video_id", metadata.get("video_path", metadata["filename"]))


def sample_frames(frame_count, fps, interval):
    if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Video must have a valid frame count and frame rate")
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("Sampling interval must be positive and finite")
    step = max(1, round(interval * fps))
    frames = list(range(0, frame_count, step))
    if frames[-1] != frame_count - 1:
        frames.append(frame_count - 1)
    return frames


def search_video_frames(collection, vector, top_n, min_similarity, excluded=()):
    """Return the best frame per video, expanding past duplicate frame hits."""
    count = collection.count()
    if count == 0 or top_n <= 0:
        return []
    fetch_n = min(top_n * 3, count)
    while True:
        results = collection.query(
            query_embeddings=[vector], n_results=fetch_n,
            include=["metadatas", "distances"],
        )
        output, seen = [], set()
        below_threshold = False
        for frame_id, meta, distance in zip(
            results["ids"][0], results["metadatas"][0], results["distances"][0]
        ):
            similarity = round(max(0.0, 1.0 - distance) * 100, 1)
            if similarity < min_similarity:
                below_threshold = True
                break
            filename = meta["filename"]
            key = video_key(meta)
            if key in seen or key in excluded or filename in excluded:
                continue
            seen.add(key)
            output.append({
                "filename": filename, "similarity": similarity, "video_id": key,
                "folder": Path(meta["video_path"]).parent.name if "video_path" in meta else "",
                "timestamp": meta.get("timestamp", 0.0), "frame_id": frame_id,
            })
            if len(output) == top_n:
                return output
        if fetch_n == count or below_threshold:
            return output
        fetch_n = min(fetch_n * 2, count)
