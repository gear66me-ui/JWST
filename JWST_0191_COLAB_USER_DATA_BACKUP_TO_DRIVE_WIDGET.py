# JWST_0191
import os, shutil, hashlib, csv, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from google.colab import drive
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION = "JWST_0191"
SOURCE_ROOT = Path("/content")
DRIVE_BASE = Path("/content/drive/MyDrive/JWST/RUNTIME_BACKUP")

# Only user-created/scientific data are eligible. Runtime OS, Python packages,
# caches, mounted Drive, and Colab sample data are never copied.
EXCLUDED_TOP_LEVEL = {
    "drive", "sample_data", ".config", ".cache", "__pycache__",
}
EXCLUDED_SUFFIXES = {".pyc", ".tmp", ".lock"}

scan_btn = widgets.Button(
    description="Scan backup candidates",
    button_style="info",
    layout=widgets.Layout(width="260px"),
)
backup_btn = widgets.Button(
    description="Back up user data to Drive",
    button_style="success",
    layout=widgets.Layout(width="300px"),
)
out = widgets.Output()


def fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def runtime_usage():
    total, used, free = shutil.disk_usage("/content")
    return total, used, free


def is_excluded(path: Path) -> bool:
    try:
        rel = path.relative_to(SOURCE_ROOT)
    except ValueError:
        return True
    if not rel.parts:
        return True
    if rel.parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def candidate_roots():
    roots = []
    for p in SOURCE_ROOT.iterdir():
        if is_excluded(p):
            continue
        # Preserve user/science directories and non-widget files.
        if p.is_dir():
            roots.append(p)
        elif p.is_file() and not p.name.startswith("JWST_"):
            roots.append(p)
    return sorted(roots, key=lambda p: p.name.lower())


def enumerate_files():
    files = []
    for root in candidate_roots():
        if root.is_file():
            try:
                files.append((root, root.stat().st_size))
            except OSError:
                pass
            continue
        for base, dirs, names in os.walk(root, followlinks=False):
            base_path = Path(base)
            dirs[:] = [d for d in dirs if not is_excluded(base_path / d)]
            for name in names:
                p = base_path / name
                if is_excluded(p) or p.is_symlink():
                    continue
                try:
                    files.append((p, p.stat().st_size))
                except OSError:
                    pass
    files.sort(key=lambda x: x[1], reverse=True)
    return files


def destination_for(src: Path, backup_root: Path) -> Path:
    return backup_root / src.relative_to(SOURCE_ROOT)


def copy_verified(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    copied = 0
    size = src.stat().st_size
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        while True:
            chunk = fin.read(16 * 1024 * 1024)
            if not chunk:
                break
            fout.write(chunk)
            copied += len(chunk)
        fout.flush()
        os.fsync(fout.fileno())
    if copied != size or tmp.stat().st_size != size:
        raise IOError(f"size verification failed ({copied} != {size})")
    tmp.replace(dst)
    os.sync()
    if dst.stat().st_size != size:
        raise IOError("destination size changed after finalization")


def scan(_=None):
    with out:
        clear_output(wait=True)
        print(f"CODE OUTPUT: {VERSION}")
        total, used, free = runtime_usage()
        print(f"Runtime disk: used={fmt_bytes(used)} | free={fmt_bytes(free)} | total={fmt_bytes(total)}")
        files = enumerate_files()
        total_bytes = sum(s for _, s in files)
        print(f"Eligible user/science files: {len(files)}")
        print(f"Eligible backup size: {fmt_bytes(total_bytes)}")
        print("\nLargest candidates:")
        for p, size in files[:30]:
            print(f"  {fmt_bytes(size):>10} | {p}")
        print("\nNot included: Colab operating system, installed packages, caches, sample_data, or /content/drive.")
        print(f"Timestamp Colombia: {datetime.now(ZoneInfo('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"# {VERSION}")


def backup(_=None):
    with out:
        clear_output(wait=True)
        print(f"CODE OUTPUT: {VERSION}")
        print("Safe backup mode: copy -> size verify -> delete runtime original.")
        drive.mount("/content/drive", force_remount=False)
        stamp = datetime.now(ZoneInfo("America/Bogota")).strftime("%Y%m%d_%H%M%S")
        backup_root = DRIVE_BASE / f"{VERSION}_{stamp}"
        backup_root.mkdir(parents=True, exist_ok=True)
        print(f"Drive destination: {backup_root}")

        files = enumerate_files()
        total_bytes = sum(s for _, s in files)
        print(f"Files selected: {len(files)} | total={fmt_bytes(total_bytes)}")
        if not files:
            print("Nothing eligible to back up.")
            print(f"# {VERSION}")
            return

        manifest_rows = []
        moved = 0
        failed = 0
        moved_bytes = 0
        for i, (src, size) in enumerate(files, 1):
            dst = destination_for(src, backup_root)
            print(f"[{i:04d}/{len(files):04d}] {fmt_bytes(size):>10} | {src}")
            try:
                if dst.exists() and dst.stat().st_size == size:
                    print("  Verified existing Drive copy; deleting runtime duplicate.")
                else:
                    copy_verified(src, dst)
                    print(f"  Copied and verified -> {dst}")
                src.unlink()
                moved += 1
                moved_bytes += size
                manifest_rows.append([str(src), str(dst), size, "moved_verified"])
            except Exception as exc:
                failed += 1
                manifest_rows.append([str(src), str(dst), size, f"ERROR: {exc}"])
                print(f"  ERROR: {exc}")

        # Remove empty source directories, deepest first.
        roots = candidate_roots()
        for root in roots:
            if root.is_dir() and root.exists():
                for base, dirs, names in os.walk(root, topdown=False):
                    p = Path(base)
                    try:
                        if not any(p.iterdir()):
                            p.rmdir()
                    except OSError:
                        pass

        manifest = backup_root / f"{VERSION}_MANIFEST.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source_runtime_path", "drive_backup_path", "size_bytes", "status"])
            w.writerows(manifest_rows)

        os.sync()
        total, used, free = runtime_usage()
        print("\nBACKUP SUMMARY")
        print(f"Moved and verified : {moved}")
        print(f"Failed             : {failed}")
        print(f"Runtime freed      : {fmt_bytes(moved_bytes)}")
        print(f"Runtime now        : used={fmt_bytes(used)} | free={fmt_bytes(free)} | total={fmt_bytes(total)}")
        print(f"Manifest           : {manifest}")
        print(f"Timestamp Colombia : {datetime.now(ZoneInfo('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"# {VERSION}")


scan_btn.on_click(scan)
backup_btn.on_click(backup)
display(widgets.HBox([scan_btn, backup_btn]), out)
