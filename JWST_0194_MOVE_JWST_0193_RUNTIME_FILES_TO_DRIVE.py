# JWST_0194
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import shutil

from google.colab import drive

VERSION = "JWST_0194"
SRC_ROOT = Path("/content")
STAMP = datetime.now(ZoneInfo("America/Bogota")).strftime("%Y%m%d_%H%M%S")
DST_ROOT = Path(f"/content/drive/MyDrive/JWST/RUNTIME_BACKUP/{VERSION}_{STAMP}")

TOKENS = (
    "JWST_0193",
    "X362b",
    "X362B",
    "A001_X362b_Xae6",
    "JADES-GS-z11-0",
    "JADES_GS_Z11_0",
)


def human_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024


def is_candidate(path):
    s = str(path)
    if "/content/drive/" in s:
        return False
    if not path.is_file():
        return False
    return any(token in s for token in TOKENS)


def copy_verify_delete(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_size = src.stat().st_size
    dst_size = dst.stat().st_size
    if src_size != dst_size:
        raise IOError(f"size mismatch: source={src_size}, destination={dst_size}")
    src.unlink()
    return src_size


def remove_empty_dirs(root):
    dirs = [p for p in root.rglob("*") if p.is_dir() and "/content/drive/" not in str(p)]
    for p in sorted(dirs, key=lambda x: len(x.parts), reverse=True):
        try:
            p.rmdir()
        except OSError:
            pass


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print("Automatic move mode: copy to Drive -> size verify -> delete runtime original.")
    drive.mount("/content/drive", force_remount=False)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in SRC_ROOT.rglob("*") if is_candidate(p)])
    total = sum(p.stat().st_size for p in files)
    print(f"Drive destination: {DST_ROOT}")
    print(f"Files selected: {len(files)} | total={human_bytes(total)}")

    moved = 0
    moved_bytes = 0
    failed = []

    for i, src in enumerate(files, 1):
        rel = src.relative_to(SRC_ROOT)
        dst = DST_ROOT / rel
        try:
            size = copy_verify_delete(src, dst)
            moved += 1
            moved_bytes += size
            print(f"[{i:04d}/{len(files):04d}] {human_bytes(size):>10} | moved and verified | {rel}")
        except Exception as exc:
            failed.append((str(src), str(exc)))
            print(f"[{i:04d}/{len(files):04d}] FAILED | {rel} | {exc}")

    remove_empty_dirs(SRC_ROOT)

    manifest = DST_ROOT / f"{VERSION}_MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"CODE OUTPUT: {VERSION}\n")
        f.write(f"Destination: {DST_ROOT}\n")
        f.write(f"Moved files: {moved}\n")
        f.write(f"Moved bytes: {moved_bytes}\n")
        f.write(f"Failed files: {len(failed)}\n")
        for path, error in failed:
            f.write(f"FAILED\t{path}\t{error}\n")

    stat = shutil.disk_usage("/content")
    print(f"Moved and verified: {moved}")
    print(f"Failed: {len(failed)}")
    print(f"Moved total: {human_bytes(moved_bytes)}")
    print(f"Runtime free now: {human_bytes(stat.free)} of {human_bytes(stat.total)}")
    print(f"Drive folder: {DST_ROOT}")
    print(f"Manifest: {manifest}")
    print(f"Timestamp Colombia: {datetime.now(ZoneInfo('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"# {VERSION}")


main()
