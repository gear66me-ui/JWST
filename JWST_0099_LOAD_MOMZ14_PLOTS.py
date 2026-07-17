#!/usr/bin/env python3
"""
JWST_0099_LOAD_MOMZ14_PLOTS.py

Display existing MoM-z14 PNG products already present in the active Colab
runtime. This script does not query MAST, create plots, modify data, or use AI.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from IPython.display import Image, Markdown, display

VERSION = "JWST_0099"

SEARCH_DIRS = [
    Path("/content/JWST_OUTPUT/PNG"),
    Path("/content/drive/MyDrive/Colab Notebooks/JWST/MoM-14"),
]

PRIORITY_PATTERNS = [
    "JWST_0098_*_RAW_NATIVE.png",
    "JWST_0097_*.png",
    "JWST_0096_*.png",
    "*MOMZ14*.png",
    "*MoM-z14*.png",
    "*MoM-14*.png",
]


def find_images() -> list[Path]:
    ordered: list[Path] = []
    seen: set[str] = set()
    for directory in SEARCH_DIRS:
        if not directory.exists():
            continue
        for pattern in PRIORITY_PATTERNS:
            for path in sorted(directory.glob(pattern)):
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key not in seen:
                    seen.add(key)
                    ordered.append(path)
    return ordered


def main() -> None:
    images = find_images()
    if not images:
        searched = "\n".join(f"  {path}" for path in SEARCH_DIRS)
        raise FileNotFoundError(
            "No existing MoM-z14 PNG files were found.\n"
            "The active Colab runtime may have restarted.\n"
            f"Searched:\n{searched}\n"
            "Rerun JWST_0098 first to recreate the MAST-native plots."
        )

    display(Markdown(f"# MoM-z14 existing plot files\nFound **{len(images)}** PNG file(s)."))
    for index, path in enumerate(images, start=1):
        size_mb = path.stat().st_size / (1024 * 1024)
        display(Markdown(f"## {index:02d}. `{path.name}`\n`{path}`  ·  {size_mb:.2f} MB"))
        display(Image(filename=str(path)))

    print()
    print(f"CODE OUTPUT: {VERSION}")
    print(f"IMAGES DISPLAYED {len(images)}")
    for path in images:
        print(f"PNG             {path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
