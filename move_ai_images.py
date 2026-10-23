#!/usr/bin/env python3
"""Move AI-generated images from Downloads to the image library."""

import argparse
import re
import shutil
from pathlib import Path

SOURCE_DIR = Path.home() / "Downloads"
DEST_DIR = Path("/Users/peter.polos/Documents/Peter_Projects/Cineautoma/Movie_Projects/001_RAW_IMAGES")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

PATTERNS = [
    # ChatGPT: "ChatGPT Image May 22, 2026, 10_21_13 PM.png"
    re.compile(r"^ChatGPT Image .+$", re.IGNORECASE),
    # Midjourney: "username_prompt_words_<uuid>_<n>.png"
    re.compile(r"^.+_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_\d+$", re.IGNORECASE),
    # Timestamp suffix: "prompt_words_202605230649.jpeg"
    re.compile(r"^.+_\d{10,14}$"),
    # Grok: pure UUID-like hex string "84366593-7bb1-48de-b35f8-c18dc1187a94.png"
    re.compile(r"^[0-9a-f]{6,}(-[0-9a-f]+)+$", re.IGNORECASE),
]


def is_ai_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False
    stem = path.stem
    return any(p.match(stem) for p in PATTERNS)


def main():
    parser = argparse.ArgumentParser(description="Move AI-generated images from Downloads to image library.")
    parser.add_argument("--source", type=Path, default=SOURCE_DIR, help="Source folder (default: ~/Downloads)")
    parser.add_argument("--dest", type=Path, default=DEST_DIR, help="Destination folder")
    parser.add_argument("--move", action="store_true", help="Actually move files (default is dry-run)")
    args = parser.parse_args()

    candidates = [f for f in args.source.iterdir() if f.is_file() and is_ai_image(f)]

    if not candidates:
        print("No matching AI-generated images found.")
        return

    skipped = []
    to_move = []
    for f in sorted(candidates):
        dest_path = args.dest / f.name
        if dest_path.exists():
            skipped.append(f)
        else:
            to_move.append(f)

    print(f"{'DRY RUN — ' if not args.move else ''}Found {len(to_move)} file(s) to move, {len(skipped)} already exist.\n")

    for f in to_move:
        dest_path = args.dest / f.name
        print(f"  {'MOVE' if args.move else 'WOULD MOVE'}: {f.name}")
        if args.move:
            shutil.move(str(f), dest_path)

    if skipped:
        print(f"\nSkipped (already in destination):")
        for f in skipped:
            print(f"  {f.name}")

    if not args.move and to_move:
        print(f"\nRun with --move to actually move the files.")


if __name__ == "__main__":
    main()
