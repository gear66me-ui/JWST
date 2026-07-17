# JWST_0116
# Direct MAST download and native-array audit for verified source 10003 G235H products.

from pathlib import Path
from datetime import datetime
from urllib.parse import quote
import csv
import hashlib
import json
import requests
import numpy as np
from astropy.io import fits
from astroquery.mast import Observations

VERSION = "JWST_0116"
print(f"CODE OUTPUT: {VERSION}")

TARGET_RA = 3.6171694
TARGET_DEC = -30.4255494
SOURCE_ID = "10003"
GRATING = "g235h"
FILTER = "f170lp"

ROOT = Path("/content/JWST_OUTPUT/DATA/JWST_0116")
FITS_DIR = ROOT / "FITS"
CSV_DIR = Path("/content/JWST_OUTPUT/CSV")
FITS_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

BACKUP_FILE = Path(
    "/content/JWST_BACKUP_REPO/FITS_DATA/MOM_Z14_SOURCE_10003/"
    "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits"
)
RUNTIME_ROOT = Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS")
HLSP_NAME = "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits"

source_candidates = []
if BACKUP_FILE.exists():
    source_candidates.append(BACKUP_FILE)
if RUNTIME_ROOT.exists():
    source_candidates.extend(RUNTIME_ROOT.rglob(HLSP_NAME))
if not source_candidates:
    raise FileNotFoundError(HLSP_NAME)
source_fits = source_candidates[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_fits(path: Path):
    rows = []
    with fits.open(path, memmap=False) as hdul:
        for idx, hdu in enumerate(hdul):
            name = hdu.name or f"HDU{idx}"
            data = hdu.data
            shape = None
            elements = 0
            dtype = ""
            finite = 0
            if data is not None:
                try:
                    shape = tuple(int(x) for x in np.shape(data))
                    elements = int(np.size(data))
                    dtype = str(getattr(data, "dtype", ""))
                    if isinstance(data, np.ndarray) and data.dtype.names is None and np.issubdtype(data.dtype, np.number):
                        finite = int(np.isfinite(data).sum())
                except Exception:
                    pass
            rows.append({
                "file": path.name,
                "hdu_index": idx,
                "hdu_name": name,
                "shape": str(shape),
                "elements": elements,
                "finite_numeric": finite,
                "dtype": dtype,
            })
    return rows


def read_spec1d(path: Path):
    with fits.open(path, memmap=False) as hdul:
        if "SPEC1D" not in hdul:
            return None
        tab = hdul["SPEC1D"].data
        names = {n.lower(): n for n in (tab.dtype.names or [])}
        wave_key = names.get("wave") or names.get("wavelength")
        flux_key = names.get("flux")
        err_key = names.get("err") or names.get("error")
        if wave_key is None:
            return None
        wave = np.asarray(tab[wave_key], float).ravel()
        flux = np.asarray(tab[flux_key], float).ravel() if flux_key else np.full(wave.shape, np.nan)
        err = np.asarray(tab[err_key], float).ravel() if err_key else np.full(wave.shape, np.nan)
        return wave, flux, err


print("SOURCE FITS:", source_fits)
print("\n1/4 READING COMPLETE NATIVE 1-D ARRAY")
spec = read_spec1d(source_fits)
if spec is None:
    raise RuntimeError("SPEC1D wavelength array not found")
wave, flux, err = spec
native_valid = np.isfinite(wave)
print(f"Native wavelength samples       {wave.size}")
print(f"Finite wavelength samples       {native_valid.sum()}")
print(f"Wavelength range [um]           {np.nanmin(wave):.6f} to {np.nanmax(wave):.6f}")

native_csv = CSV_DIR / "JWST_0116_G235H_NATIVE_1D_SOURCE10003.csv"
with native_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_index", "wavelength_um", "flux_native", "error_native"])
    for i, (w, y, e) in enumerate(zip(wave, flux, err)):
        writer.writerow([i, f"{w:.10f}", f"{y:.12e}", f"{e:.12e}"])

print("\n2/4 AUDITING ALL 2-D AND HIGHER ARRAYS")
audit_rows = audit_fits(source_fits)
for row in audit_rows:
    if row["elements"]:
        print(f"HDU {row['hdu_index']:2d}  {row['hdu_name']:<16s} shape={row['shape']:<18s} elements={row['elements']}")

audit_csv = CSV_DIR / "JWST_0116_FITS_HDU_NATIVE_ARRAY_AUDIT.csv"
with audit_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
    writer.writeheader()
    writer.writerows(audit_rows)

print("\n3/4 QUERYING MAST PRODUCT TABLE")
obs = Observations.query_region(f"{TARGET_RA} {TARGET_DEC}", radius="3 arcsec")
products = Observations.get_product_list(obs)

selected = []
for row in products:
    filename = str(row["productFilename"]) if "productFilename" in row.colnames else ""
    data_uri = str(row["dataURI"]) if "dataURI" in row.colnames else ""
    text = (filename + " " + data_uri).lower()
    if SOURCE_ID in text and GRATING in text and FILTER in text and filename.lower().endswith(".fits"):
        selected.append({
            "filename": filename,
            "data_uri": data_uri,
            "product_type": str(row["productType"]) if "productType" in row.colnames else "",
            "description": str(row["description"]) if "description" in row.colnames else "",
            "size_bytes": int(row["size"]) if "size" in row.colnames and row["size"] is not None else 0,
        })

unique = {}
for item in selected:
    unique[item["data_uri"] or item["filename"]] = item
selected = list(unique.values())
print(f"Source-specific G235H products  {len(selected)}")
for item in selected:
    print(f"  {item['filename']}  {item['size_bytes'] / 1024 / 1024:.3f} MB")

product_csv = CSV_DIR / "JWST_0116_MAST_G235H_SOURCE10003_PRODUCTS.csv"
with product_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "data_uri", "product_type", "description", "size_bytes"])
    writer.writeheader()
    writer.writerows(selected)

print("\n4/4 DIRECT-DOWNLOADING MISSING PRODUCTS")
download_rows = []
for item in selected:
    uri = item["data_uri"]
    filename = item["filename"] or Path(uri).name
    destination = FITS_DIR / filename
    status = "already present"
    if not destination.exists() or destination.stat().st_size == 0:
        if not uri:
            status = "skipped: missing dataURI"
        else:
            url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(uri, safe="")
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with destination.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
            status = "downloaded"
    checksum = sha256(destination) if destination.exists() else ""
    print(f"{status:<24s} {filename}")
    download_rows.append({
        **item,
        "local_path": str(destination),
        "status": status,
        "downloaded_size_bytes": destination.stat().st_size if destination.exists() else 0,
        "sha256": checksum,
    })

for item in download_rows:
    path = Path(item["local_path"])
    if path.exists():
        audit_rows.extend(audit_fits(path))

all_audit_csv = CSV_DIR / "JWST_0116_ALL_DOWNLOADED_FITS_HDU_AUDIT.csv"
with all_audit_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
    writer.writeheader()
    writer.writerows(audit_rows)

manifest = ROOT / "JWST_0116_MANIFEST.json"
manifest.write_text(json.dumps({
    "version": VERSION,
    "source_id": SOURCE_ID,
    "target_ra_deg": TARGET_RA,
    "target_dec_deg": TARGET_DEC,
    "source_fits": str(source_fits),
    "native_1d_samples": int(wave.size),
    "products": download_rows,
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
}, indent=2), encoding="utf-8")

print("\nOUTPUT SUMMARY")
print("Native 1-D CSV:", native_csv)
print("Initial HDU audit:", audit_csv)
print("MAST product table:", product_csv)
print("Downloaded products:", FITS_DIR)
print("Complete HDU audit:", all_audit_csv)
print("Manifest:", manifest)
print(f"END {VERSION}")
# JWST_0116
