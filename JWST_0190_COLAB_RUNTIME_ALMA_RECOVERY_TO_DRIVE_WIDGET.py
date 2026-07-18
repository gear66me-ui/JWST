# JWST_0190
import os, shutil, subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from google.colab import drive
import ipywidgets as widgets
from IPython.display import display, clear_output

VERSION = 'JWST_0190'
DRIVE_ROOT = Path('/content/drive/MyDrive/JWST/ALMA/RUNTIME_RECOVERY')
CONTENT_ROOT = Path('/content')
MIN_SIZE = 100 * 1024**2

run_btn = widgets.Button(
    description='Move ALMA/JWST files to Drive + free runtime disk',
    button_style='danger',
    layout=widgets.Layout(width='460px')
)
out = widgets.Output()


def fmt_bytes(n):
    n = float(n)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.2f} {unit}'
        n /= 1024
    return f'{n:.2f} PB'


def disk_status(path='/content'):
    total, used, free = shutil.disk_usage(path)
    return total, used, free


def is_candidate(path):
    s = str(path).lower()
    name = path.name.lower()
    archive_ext = name.endswith(('.tar', '.tar.gz', '.tgz', '.zip', '.fits', '.fits.gz', '.fit', '.fz', '.part'))
    jwst_alma = any(k in s for k in ['jwst', 'alma', '2023.1.00336.s', 'a001_x3667_xe0', 'a001_x362b_xae6'])
    return archive_ext and jwst_alma


def scan_candidates():
    found = []
    skip_prefixes = ['/content/drive', '/content/sample_data']
    for root, dirs, files in os.walk(CONTENT_ROOT):
        root_s = str(root)
        if any(root_s.startswith(p) for p in skip_prefixes):
            dirs[:] = []
            continue
        for name in files:
            p = Path(root) / name
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size >= MIN_SIZE and is_candidate(p):
                found.append((size, p))
    return sorted(found, reverse=True)


def unique_destination(src):
    dest = DRIVE_ROOT / src.name
    if not dest.exists():
        return dest
    stem = src.name
    i = 1
    while True:
        candidate = DRIVE_ROOT / f'{stem}.runtime_copy_{i:02d}'
        if not candidate.exists():
            return candidate
        i += 1


def remove_safe_caches():
    removed = 0
    for p in [Path('/root/.cache/pip'), Path('/root/.cache/matplotlib')]:
        if p.exists():
            try:
                size = sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
                shutil.rmtree(p)
                removed += size
            except Exception:
                pass
    return removed


def run(_=None):
    with out:
        clear_output(wait=True)
        print(f'CODE OUTPUT: {VERSION}')
        print('Purpose: recover large JWST/ALMA files from Colab runtime and move them to persistent Google Drive.')

        drive.mount('/content/drive', force_remount=True)
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f'Drive destination: {DRIVE_ROOT}')

        total0, used0, free0 = disk_status()
        print(f'Runtime before: used={fmt_bytes(used0)} | free={fmt_bytes(free0)} | total={fmt_bytes(total0)}')

        candidates = scan_candidates()
        print(f'Large ALMA/JWST runtime files found: {len(candidates)}')

        moved_total = 0
        for size, src in candidates:
            dest = unique_destination(src)
            print(f'Moving {fmt_bytes(size):>10} | {src}')
            try:
                shutil.move(str(src), str(dest))
                moved_total += size
                print(f'  -> {dest}')
            except Exception as e:
                print(f'  ERROR: {e}')

        cache_removed = remove_safe_caches()
        subprocess.run(['sync'], check=False)

        total1, used1, free1 = disk_status()
        print()
        print(f'Moved to Drive: {fmt_bytes(moved_total)}')
        print(f'Safe caches removed: {fmt_bytes(cache_removed)}')
        print(f'Runtime after: used={fmt_bytes(used1)} | free={fmt_bytes(free1)} | total={fmt_bytes(total1)}')
        print(f'Freed runtime space: {fmt_bytes(max(0, free1-free0))}')
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')


run_btn.on_click(run)
display(run_btn, out)
