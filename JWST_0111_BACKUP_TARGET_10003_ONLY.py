# JWST_0111
# Audit reference: back up only the three verified HLSP GLASS spectra for source 10003.

from pathlib import Path
import hashlib
import os
import shutil
import subprocess
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

VERSION = "JWST_0111"
print(f"CODE OUTPUT: {VERSION}")

TARGET_RA = 3.6171694
TARGET_DEC = -30.4255494
MAX_OFFSET_ARCSEC = 0.10

SOURCE_ROOT = Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS")
REPO_DIR = Path("/content/JWST_BACKUP_REPO")
DEST_DIR = REPO_DIR / "FITS_DATA" / "MOM_Z14_SOURCE_10003"

EXPECTED = {
    "G140H": "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f100lp-g140h_v1_spec.fits",
    "G235H": "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits",
    "G395H": "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f290lp-g395h_v1_spec.fits",
}

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(SOURCE_ROOT)
if not (REPO_DIR / ".git").exists():
    raise RuntimeError(f"Git repository not found: {REPO_DIR}")

os.chdir("/content")
DEST_DIR.mkdir(parents=True, exist_ok=True)

selected = []
print("VERIFYING TARGET FILES")

for mode, filename in EXPECTED.items():
    matches = list(SOURCE_ROOT.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}; found {len(matches)}")

    path = matches[0]
    with fits.open(path, memmap=False) as hdul:
        hdr = hdul[0].header
        ra = hdr.get("RA")
        dec = hdr.get("DEC")
        if ra is None or dec is None:
            raise RuntimeError(f"Missing RA/DEC in {filename}")

        target = SkyCoord(TARGET_RA * u.deg, TARGET_DEC * u.deg)
        measured = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
        offset = measured.separation(target).arcsec

    if offset > MAX_OFFSET_ARCSEC:
        raise RuntimeError(f"Coordinate mismatch for {filename}: {offset:.6f} arcsec")

    size_mb = path.stat().st_size / 1024 / 1024
    selected.append((mode, path, offset, size_mb))
    print(f"{mode:5s}  offset={offset:.6f} arcsec  size={size_mb:.3f} MB")

print()
print("COPYING VERIFIED FILES")

for old in DEST_DIR.glob("*.fits"):
    old.unlink()

manifest_lines = [
    f"VERSION={VERSION}",
    "SOURCE_ID=10003",
    f"TARGET_RA_DEG={TARGET_RA}",
    f"TARGET_DEC_DEG={TARGET_DEC}",
    "ADOPTED_REDSHIFT=9.31102",
    "",
]

for mode, path, offset, size_mb in selected:
    destination = DEST_DIR / path.name
    shutil.copy2(path, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest_lines.append(
        f"{mode}\t{path.name}\toffset_arcsec={offset:.6f}\tsize_mb={size_mb:.3f}\tsha256={digest}"
    )

manifest = DEST_DIR / "MANIFEST.txt"
manifest.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

print("FILES COPIED:", len(selected))
print("TOTAL SIZE MB:", f"{sum(x[3] for x in selected):.3f}")
print("DESTINATION:", DEST_DIR)

subprocess.run(["git", "-C", str(REPO_DIR), "config", "user.name", "GEAR66"], check=True)
subprocess.run([
    "git", "-C", str(REPO_DIR), "config", "user.email",
    "gear66@users.noreply.github.com"
], check=True)
subprocess.run([
    "git", "-C", str(REPO_DIR), "add",
    str(DEST_DIR.relative_to(REPO_DIR))
], check=True)

status = subprocess.run(
    ["git", "-C", str(REPO_DIR), "status", "--porcelain"],
    check=True,
    text=True,
    capture_output=True,
).stdout.strip()

if status:
    subprocess.run([
        "git", "-C", str(REPO_DIR), "commit", "-m",
        "Back up verified MoM-z14 source 10003 spectra"
    ], check=True)
    print("PUSHING VERIFIED FILES")
    subprocess.run(["git", "-C", str(REPO_DIR), "push", "origin", "main"], check=True)
    print("UPLOAD COMPLETE")
else:
    print("NO CHANGES: VERIFIED FILES ALREADY PRESENT")

print("REPOSITORY FOLDER: FITS_DATA/MOM_Z14_SOURCE_10003")
print(f"END {VERSION}")
