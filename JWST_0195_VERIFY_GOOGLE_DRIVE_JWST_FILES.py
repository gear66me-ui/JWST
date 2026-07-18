# JWST_0195
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os
from google.colab import drive

VERSION = "JWST_0195"
ROOT = Path("/content/drive/MyDrive/JWST")
CHECKS = [
    ROOT / "ALMA/JADES_GS_Z11_0_2023.1.00336.S/X362B_SCIENCE/ARCHIVE/2023.1.00336.S_uid___A001_X362b_Xae6_001_of_001.tar",
    ROOT / "ALMA/JADES_GS_Z11_0_2023.1.00336.S/METADATA/JWST_0192_ALMA_PRODUCT_INVENTORY.csv",
    ROOT / "RUNTIME_BACKUP/JWST_0194_20260717_220101/JWST_0194_MANIFEST.txt",
    ROOT / "RUNTIME_BACKUP/JWST_0194_20260717_220101/JWST_0193_JADES_GS_Z11_0_ALMA_X362B_SCIENCE_ARCHIVE_5MHZ_CENTROID.py",
]

def human(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024 or u == units[-1]:
            return f"{x:.2f} {u}"
        x /= 1024

def main():
    print(f"CODE OUTPUT: {VERSION}")
    drive.mount("/content/drive", force_remount=True)
    print(f"Drive root: {ROOT}")
    print()
    ok = 0
    for p in CHECKS:
        exists = p.is_file()
        size = p.stat().st_size if exists else 0
        status = "FOUND" if exists else "MISSING"
        print(f"{status:7} | {human(size):>10} | {p}")
        ok += int(exists and size > 0)
    print()
    recent = sorted(
        [p for p in ROOT.rglob("*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:25]
    print("Most recent files visible through the mounted Google Drive:")
    for p in recent:
        st = p.stat()
        stamp = datetime.fromtimestamp(st.st_mtime, ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{stamp} | {human(st.st_size):>10} | {p}")
    print()
    print(f"Verification result: {ok}/{len(CHECKS)} required files found and non-empty")
    print(f"Timestamp Colombia: {datetime.now(ZoneInfo('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"# {VERSION}")

main()
