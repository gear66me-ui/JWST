#!/usr/bin/env python3
"""
Move the four existing MoM-z14 FITS images from the active Colab runtime into:
/content/drive/MyDrive/Colab Notebooks/JWST/MoM-14

No uploads, no file picker, no redownload, and no AI imagery.
"""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

VERSION = "JWST_0082"
DRIVE_MOUNT = Path("/content/drive")
DESTINATION = DRIVE_MOUNT / "MyDrive" / "Colab Notebooks" / "JWST" / "MoM-14"
FILTERS = ("F115W", "F150W", "F277W", "F444W")
SEARCH_ROOTS = (
    Path("/content/JWST_OUTPUT/FITS"),
    Path("/content"),
)


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This script must run inside Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)


def find_runtime_file(filter_name: str) -> Path:
    exact = Path(f"/content/JWST_OUTPUT/FITS/JWST_0080_MOMZ14_{filter_name}.fits")
    if exact.exists():
        return exact

    candidates: list[Path] = []
    patterns = (
        f"*MOMZ14*{filter_name}*.fits",
        f"*MOM*14*{filter_name}*.fits",
        f"*{filter_name}*.fits",
    )
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if "/content/drive/" in str(path):
                    continue
                if path.is_file():
                    candidates.append(path)
        if candidates:
            break

    unique = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not unique:
        raise FileNotFoundError(
            f"Could not find the existing {filter_name} FITS image anywhere in /content."
        )
    return unique[0]


def move_file(source: Path, filter_name: str) -> Path:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / f"MoM-14_{filter_name}.fits"
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"Move verification failed for {filter_name}.")
    return target


def main() -> None:
    mount_drive()
    DESTINATION.mkdir(parents=True, exist_ok=True)

    moved: list[tuple[str, Path, Path]] = []
    for filter_name in FILTERS:
        source = find_runtime_file(filter_name)
        target = move_file(source, filter_name)
        moved.append((filter_name, source, target))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"DRIVE FOLDER    {DESTINATION}")
    print("ACTION          moved from Colab runtime to Google Drive")
    for filter_name, source, target in moved:
        print(f"{filter_name:<8} {source}")
        print(f"         -> {target}")
    print(f"FILES MOVED     {len(moved)}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
