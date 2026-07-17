# JWST_0127
# Download the highest-resolution archived MoM-z14 detector/source products and audit native arrays.

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import quote
import re
import json
import requests
import numpy as np
import pandas as pd
from astropy.io import fits

VERSION = "JWST_0127"
SOURCE_ID = "277193"
SOURCE_STEM = "jw05224-o004_s000277193_nirspec_clear-prism"
MAST_BASE = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

ROOT = Path("/content/JWST_OUTPUT")
FITS_DIR = ROOT / "DATA" / VERSION / "FITS"
CSV_DIR = ROOT / "CSV"
for directory in (FITS_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 FULL DETECTOR / MAXIMUM ARCHIVED SAMPLING DOWNLOAD")
print("-" * 112)


def mast_download(filename, required=False):
    path = FITS_DIR / filename
    if path.exists() and path.stat().st_size > 2880:
        print(f"CACHED:     {filename}  {path.stat().st_size/1e6:.3f} MB")
        return path
    uri = "mast:JWST/product/" + filename
    url = MAST_BASE + quote(uri, safe="")
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            if response.status_code != 200:
                if required:
                    response.raise_for_status()
                print(f"NOT FOUND:  {filename}  HTTP {response.status_code}")
                return None
            with open(path, "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        with open(path, "rb") as handle:
            magic = handle.read(8)
        if not magic.startswith(b"SIMPLE"):
            path.unlink(missing_ok=True)
            print(f"REJECTED:   {filename}  response was not FITS")
            return None
        print(f"DOWNLOADED: {filename}  {path.stat().st_size/1e6:.3f} MB")
        return path
    except Exception as exc:
        path.unlink(missing_ok=True)
        if required:
            raise
        print(f"FAILED:     {filename}  {exc}")
        return None


# Source-level calibrated products. CRF is the least-resampled public source product.
source_files = [
    f"{SOURCE_STEM}_crf.fits",
    f"{SOURCE_STEM}_s2d.fits",
    f"{SOURCE_STEM}_x1d.fits",
]
paths = []
for index, filename in enumerate(source_files):
    path = mast_download(filename, required=(index == 0))
    if path:
        paths.append(path)

crf_path = FITS_DIR / f"{SOURCE_STEM}_crf.fits"
if not crf_path.exists():
    raise RuntimeError("The required source CRF FITS file could not be downloaded.")

# Discover contributing exposure roots from every header and table string in the CRF product.
text_tokens = set()
with fits.open(crf_path, memmap=False) as hdul:
    for hdu in hdul:
        for value in hdu.header.values():
            if isinstance(value, str):
                text_tokens.add(value)
        data = hdu.data
        names = list(getattr(data, "names", []) or [])
        for name in names:
            try:
                column = np.asarray(data[name]).ravel()
                if column.dtype.kind in "SUO":
                    for value in column:
                        text_tokens.add(str(value))
            except Exception:
                pass

joined = "\n".join(text_tokens)
# Exposure products generally look like jw05224004001_..._nrs1_cal.fits or nrs2_cal.fits.
found_filenames = set(re.findall(r"jw\d{11,}[A-Za-z0-9_\-]*_nrs[12]_[A-Za-z0-9]+\.fits", joined, flags=re.I))
roots = set()
for filename in found_filenames:
    roots.add(re.sub(r"_(uncal|rate|rateints|cal|crf)\.fits$", "", filename, flags=re.I))

print(f"\nCONTRIBUTING EXPOSURE ROOTS DISCOVERED: {len(roots)}")
for root in sorted(roots):
    print(f"  {root}")

# Try maximum detector-level calibrated products first; rate/rateints are less processed and much larger.
parent_candidates = []
for root in sorted(roots):
    for suffix in ("cal.fits", "rate.fits", "rateints.fits"):
        parent_candidates.append(f"{root}_{suffix}")

for filename in parent_candidates:
    path = mast_download(filename, required=False)
    if path:
        paths.append(path)

# Audit every downloaded array and identify the maximum native detector dimensions.
rows = []
for path in sorted(set(paths)):
    try:
        with fits.open(path, memmap=True) as hdul:
            for index, hdu in enumerate(hdul):
                data = hdu.data
                shape = tuple(data.shape) if data is not None and hasattr(data, "shape") else None
                elements = int(np.prod(shape)) if shape else 0
                rows.append({
                    "filename": path.name,
                    "hdu_index": index,
                    "hdu_name": hdu.name,
                    "shape": str(shape),
                    "ndim": len(shape) if shape else 0,
                    "rows": shape[-2] if shape and len(shape) >= 2 else np.nan,
                    "columns": shape[-1] if shape else np.nan,
                    "elements": elements,
                    "size_mb": path.stat().st_size / 1e6,
                })
    except Exception as exc:
        print(f"AUDIT FAILED: {path.name}: {exc}")

audit = pd.DataFrame(rows)
audit_csv = CSV_DIR / f"{VERSION}_DOWNLOADED_FITS_ARRAY_AUDIT.csv"
audit.to_csv(audit_csv, index=False)

print("\nDOWNLOADED ARRAY INVENTORY")
if len(audit):
    for _, row in audit.iterrows():
        if row["elements"] > 0:
            print(f"{row['filename'][:66]:66s}  {row['hdu_name'][:12]:12s}  shape={row['shape']:18s}  elements={int(row['elements'])}")

image_arrays = audit[(audit["ndim"] >= 2) & (audit["elements"] > 0)].copy()
if len(image_arrays):
    maximum = image_arrays.sort_values(["columns", "elements"], ascending=False).iloc[0]
    print("\nMAXIMUM ARCHIVED ARRAY FOUND")
    print(f"  FILE       {maximum['filename']}")
    print(f"  HDU        {maximum['hdu_name']}")
    print(f"  SHAPE      {maximum['shape']}")
    print(f"  ROWS       {int(maximum['rows'])}")
    print(f"  COLUMNS    {int(maximum['columns'])}")
    print(f"  ELEMENTS   {int(maximum['elements'])}")
else:
    print("\nNo two-dimensional image arrays were readable.")

manifest = {
    "version": VERSION,
    "source_id": SOURCE_ID,
    "source_stem": SOURCE_STEM,
    "downloaded_files": [str(path) for path in sorted(set(paths))],
    "exposure_roots": sorted(roots),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
manifest_path = CSV_DIR / f"{VERSION}_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2))

print("\nOUTPUT SUMMARY")
print(f"FITS directory: {FITS_DIR}")
print(f"Array audit:    {audit_csv}")
print(f"Manifest:       {manifest_path}")
print(f"Timestamp UTC:  {manifest['timestamp_utc']}")
print(f"# {VERSION}")
