from pathlib import Path
from datetime import datetime, timezone, timedelta
import os

VERSION = "JWST_0200"
BASE = Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
ARCHIVE = BASE / "ARCHIVE/2023.1.00336.S_uid___A001_X3667_Xe0_001_of_001.tar"
CUBE = BASE / "SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
CSV = BASE / "CSV/JWST_0199_X3667_CUBE_WCS_INVENTORY.csv"
EXPECTED_ARCHIVE_GB = 15.07
EXPECTED_CUBE_GB = 2.83

def human_bytes(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

def status_line(path, expected_gb=None):
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    nonempty = exists and size > 0
    status = "FOUND" if nonempty else "MISSING"
    note = ""
    if expected_gb is not None and nonempty:
        actual_gb = size / 1e9
        delta = actual_gb - expected_gb
        note = f" | expected ~{expected_gb:.2f} GB | delta {delta:+.2f} GB"
    print(f"{status:<7} | {human_bytes(size):>10} | {path}{note}")
    return nonempty, size

def fits_signature_ok(path):
    if not path.exists() or path.stat().st_size < 2880:
        return False, "file missing or too small"
    with open(path, "rb") as f:
        head = f.read(80)
    ok = head.startswith(b"SIMPLE") or head.startswith(b"XTENSION")
    return ok, head[:30].decode("ascii", errors="replace")

def tar_signature_ok(path):
    if not path.exists() or path.stat().st_size < 512:
        return False, "file missing or too small"
    with open(path, "rb") as f:
        f.seek(257)
        sig = f.read(8)
    ok = sig.startswith(b"ustar")
    return ok, sig.decode("ascii", errors="replace")

def main():
    print(f"CODE OUTPUT: {VERSION}")
    print("Drive mount reused; no remount requested.")
    print(f"Verification root: {BASE}\n")

    a_ok, a_size = status_line(ARCHIVE, EXPECTED_ARCHIVE_GB)
    c_ok, c_size = status_line(CUBE, EXPECTED_CUBE_GB)
    i_ok, _ = status_line(CSV)

    print()
    tar_ok, tar_sig = tar_signature_ok(ARCHIVE)
    fits_ok, fits_sig = fits_signature_ok(CUBE)
    print(f"TAR header signature valid: {tar_ok} | signature={tar_sig!r}")
    print(f"FITS header signature valid: {fits_ok} | header={fits_sig!r}")

    archive_reasonable = a_ok and a_size > 14_000_000_000
    cube_reasonable = c_ok and c_size > 2_000_000_000
    all_ok = archive_reasonable and cube_reasonable and i_ok and tar_ok and fits_ok

    print()
    print(f"Archive size check (>14 GB): {archive_reasonable}")
    print(f"Selected cube size check (>2 GB): {cube_reasonable}")
    print(f"Verification result: {'PASS' if all_ok else 'FAIL'}")
    if all_ok:
        print("The 15.07 GB X3667 archive and the extracted SPW23 pbcor cube are present and structurally recognizable.")
    else:
        print("One or more required files failed existence, size, or header-signature validation.")

    colombia = timezone(timedelta(hours=-5))
    print(f"Timestamp Colombia: {datetime.now(colombia).strftime('%Y-%m-%d %H:%M:%S %z')}")
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
