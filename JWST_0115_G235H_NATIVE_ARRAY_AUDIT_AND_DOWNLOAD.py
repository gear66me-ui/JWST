# JWST_0115
# Read the complete native G235H arrays, count detector samples,
# inspect all 2-D spectral HDUs, and download finer source-10003 products when available.

from pathlib import Path
from datetime import datetime
import json
import os
import numpy as np
import pandas as pd
from astropy.io import fits

VERSION = "JWST_0115"
print(f"CODE OUTPUT: {VERSION}")

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except Exception:
    pass

try:
    from astroquery.mast import Observations
except Exception as exc:
    raise RuntimeError("astroquery is required. Run: !pip -q install astroquery") from exc

SOURCE_ID = "10003"
PROGRAM_ID = "1324"
GRATING = "G235H"
FILTER = "F170LP"
TARGET_FILENAME = "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits"

OUT_ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = OUT_ROOT / "PNG"
CSV_DIR = OUT_ROOT / "CSV"
DATA_DIR = OUT_ROOT / "DATA" / VERSION / "NATIVE_PRODUCTS"
for p in (PNG_DIR, CSV_DIR, DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEARCH_ROOTS = [
    Path("/content/JWST_BACKUP_REPO/FITS_DATA/MOM_Z14_SOURCE_10003"),
    Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS"),
    Path("/content"),
]

matches = []
for root in SEARCH_ROOTS:
    if root.exists():
        matches.extend(root.rglob(TARGET_FILENAME))

unique = []
seen = set()
for p in matches:
    rp = str(p.resolve())
    if rp not in seen:
        seen.add(rp)
        unique.append(p)

if not unique:
    raise FileNotFoundError(TARGET_FILENAME)

source_path = unique[0]
print("SOURCE FITS:", source_path)

hdu_rows = []
array_rows = []
extracted_tables = []

with fits.open(source_path, memmap=False) as hdul:
    for index, hdu in enumerate(hdul):
        name = hdu.name or f"HDU{index}"
        data = hdu.data
        shape = None if data is None else tuple(np.shape(data))
        dtype = None if data is None else str(getattr(data, "dtype", type(data)))
        ndim = 0 if data is None else int(np.ndim(data))
        size = 0 if data is None else int(np.size(data))

        hdu_rows.append({
            "hdu_index": index,
            "hdu_name": name,
            "class": type(hdu).__name__,
            "shape": str(shape),
            "ndim": ndim,
            "element_count": size,
            "dtype": dtype,
        })

        if data is None:
            continue

        names = list(getattr(getattr(data, "dtype", None), "names", []) or [])
        if names:
            for col in names:
                try:
                    arr = np.asarray(data[col])
                except Exception:
                    continue
                finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else np.ones(arr.shape, dtype=bool)
                row = {
                    "hdu_index": index,
                    "hdu_name": name,
                    "array_name": col,
                    "shape": str(tuple(arr.shape)),
                    "ndim": int(arr.ndim),
                    "element_count": int(arr.size),
                    "finite_count": int(np.count_nonzero(finite)),
                    "dtype": str(arr.dtype),
                }
                if np.issubdtype(arr.dtype, np.number) and np.count_nonzero(finite):
                    vals = arr[finite].astype(float)
                    row["min"] = float(np.nanmin(vals))
                    row["max"] = float(np.nanmax(vals))
                array_rows.append(row)

            lower = {c.lower(): c for c in names}
            wave_col = next((lower[k] for k in ("wave", "wavelength", "lambda") if k in lower), None)
            flux_col = next((lower[k] for k in ("flux", "flam", "fnu") if k in lower), None)
            err_col = next((lower[k] for k in ("err", "error", "flux_error", "flux_err") if k in lower), None)

            if wave_col:
                wave = np.asarray(data[wave_col]).ravel()
                table_dict = {"wavelength_native": wave}
                if flux_col:
                    table_dict["flux_native"] = np.asarray(data[flux_col]).ravel()
                if err_col:
                    table_dict["error_native"] = np.asarray(data[err_col]).ravel()
                table = pd.DataFrame(table_dict)
                table.insert(0, "sample_index", np.arange(len(table), dtype=int))
                table.insert(1, "hdu_name", name)
                out_csv = CSV_DIR / f"{VERSION}_{name}_NATIVE_1D.csv"
                table.to_csv(out_csv, index=False)
                extracted_tables.append(out_csv)
        else:
            arr = np.asarray(data)
            finite = np.isfinite(arr) if np.issubdtype(arr.dtype, np.number) else np.ones(arr.shape, dtype=bool)
            row = {
                "hdu_index": index,
                "hdu_name": name,
                "array_name": name,
                "shape": str(tuple(arr.shape)),
                "ndim": int(arr.ndim),
                "element_count": int(arr.size),
                "finite_count": int(np.count_nonzero(finite)),
                "dtype": str(arr.dtype),
            }
            if np.issubdtype(arr.dtype, np.number) and np.count_nonzero(finite):
                vals = arr[finite].astype(float)
                row["min"] = float(np.nanmin(vals))
                row["max"] = float(np.nanmax(vals))
            array_rows.append(row)

hdu_df = pd.DataFrame(hdu_rows)
array_df = pd.DataFrame(array_rows)
hdu_csv = CSV_DIR / f"{VERSION}_HDU_INVENTORY.csv"
array_csv = CSV_DIR / f"{VERSION}_NATIVE_ARRAY_COUNTS.csv"
hdu_df.to_csv(hdu_csv, index=False)
array_df.to_csv(array_csv, index=False)

spec1d_rows = array_df[(array_df["hdu_name"].str.upper() == "SPEC1D")]
wave_rows = spec1d_rows[spec1d_rows["array_name"].str.lower().isin(["wave", "wavelength", "lambda"])]
full_native_samples = int(wave_rows["element_count"].max()) if not wave_rows.empty else 0

image_2d = array_df[array_df["ndim"] >= 2].copy()
image_2d_csv = CSV_DIR / f"{VERSION}_2D_ARRAY_INVENTORY.csv"
image_2d.to_csv(image_2d_csv, index=False)

print()
print("NATIVE ARRAY AUDIT")
print("-" * 88)
print(f"HDU count                     {len(hdu_df):8d}")
print(f"Complete native 1-D samples  {full_native_samples:8d}")
print(f"2-D or higher arrays         {len(image_2d):8d}")
for _, row in image_2d.iterrows():
    print(f"  {row['hdu_name']:<20s} {row['array_name']:<24s} shape={row['shape']:<18s} elements={int(row['element_count']):d}")

print()
print("QUERYING MAST FOR SOURCE-SPECIFIC G235H PRODUCTS")
print("-" * 88)

obs = Observations.query_criteria(
    obs_collection="JWST",
    proposal_id=PROGRAM_ID,
    instrument_name="NIRSPEC*",
)

products = Observations.get_product_list(obs)
prod_df = products.to_pandas()

filename_col = next((c for c in ["productFilename", "productfilename"] if c in prod_df.columns), None)
if filename_col is None:
    raise RuntimeError("MAST product table has no productFilename column")

name_series = prod_df[filename_col].astype(str)
mask_source = name_series.str.contains("10003|000010003", case=False, regex=True, na=False)
mask_mode = name_series.str.contains("g235h|f170lp", case=False, regex=True, na=False)

candidate_df = prod_df[mask_source & mask_mode].copy()

allowed_subgroups = {"CAL", "S2D", "X1D", "X1DINTS", "RATE", "RATEINTS"}
if "productSubGroupDescription" in candidate_df.columns:
    candidate_df = candidate_df[
        candidate_df["productSubGroupDescription"].astype(str).str.upper().isin(allowed_subgroups)
    ].copy()

candidate_csv = CSV_DIR / f"{VERSION}_MAST_SOURCE10003_G235H_CANDIDATES.csv"
candidate_df.to_csv(candidate_csv, index=False)

print(f"Candidate products found      {len(candidate_df):8d}")

manifest_path = None
if len(candidate_df):
    manifest = Observations.download_products(
        candidate_df,
        download_dir=str(DATA_DIR),
        cache=True,
        mrp_only=False,
    )
    manifest_df = manifest.to_pandas()
    manifest_path = CSV_DIR / f"{VERSION}_DOWNLOAD_MANIFEST.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Products requested            {len(manifest_df):8d}")
else:
    manifest_df = pd.DataFrame()
    print("No additional source-specific G235H CAL/S2D/X1D products were exposed by this query.")

print()
print("DOWNLOADED PRODUCT SAMPLE COUNTS")
print("-" * 88)

download_rows = []
for fits_path in DATA_DIR.rglob("*.fits"):
    try:
        with fits.open(fits_path, memmap=False) as hdul:
            for i, hdu in enumerate(hdul):
                data = hdu.data
                if data is None:
                    continue
                names = list(getattr(getattr(data, "dtype", None), "names", []) or [])
                if names:
                    for col in names:
                        arr = np.asarray(data[col])
                        if col.lower() in {"wave", "wavelength", "lambda", "flux", "err", "error"}:
                            download_rows.append({
                                "file": fits_path.name,
                                "path": str(fits_path),
                                "hdu_index": i,
                                "hdu_name": hdu.name,
                                "array_name": col,
                                "shape": str(tuple(arr.shape)),
                                "element_count": int(arr.size),
                                "finite_count": int(np.count_nonzero(np.isfinite(arr))) if np.issubdtype(arr.dtype, np.number) else int(arr.size),
                            })
                else:
                    arr = np.asarray(data)
                    if arr.ndim >= 2:
                        download_rows.append({
                            "file": fits_path.name,
                            "path": str(fits_path),
                            "hdu_index": i,
                            "hdu_name": hdu.name,
                            "array_name": hdu.name,
                            "shape": str(tuple(arr.shape)),
                            "element_count": int(arr.size),
                            "finite_count": int(np.count_nonzero(np.isfinite(arr))) if np.issubdtype(arr.dtype, np.number) else int(arr.size),
                        })
    except Exception as exc:
        download_rows.append({"file": fits_path.name, "path": str(fits_path), "error": str(exc)})

download_df = pd.DataFrame(download_rows)
download_csv = CSV_DIR / f"{VERSION}_DOWNLOADED_NATIVE_ARRAY_COUNTS.csv"
download_df.to_csv(download_csv, index=False)

if not download_df.empty:
    compact = download_df[[c for c in ["file", "hdu_name", "array_name", "shape", "element_count", "finite_count"] if c in download_df.columns]]
    print(compact.to_string(index=False, max_rows=80))
else:
    print("No additional FITS arrays downloaded.")

summary = {
    "version": VERSION,
    "source_fits": str(source_path),
    "complete_native_1d_samples": full_native_samples,
    "two_dimensional_array_count": int(len(image_2d)),
    "mast_candidate_count": int(len(candidate_df)),
    "downloaded_fits_count": int(len(list(DATA_DIR.rglob('*.fits')))),
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
}
summary_json = CSV_DIR / f"{VERSION}_SUMMARY.json"
summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print()
print("OUTPUT FILES")
print("-" * 88)
for p in [hdu_csv, array_csv, image_2d_csv, candidate_csv, download_csv, summary_json, *extracted_tables]:
    print(p)
if manifest_path:
    print(manifest_path)
print(DATA_DIR)
print()
print("NOTE: R≈2700 is resolving power, not a guaranteed count of 2700 samples.")
print("This script reports every real native array element available and downloads finer public products when MAST exposes them.")
print(datetime.utcnow().strftime("UTC %Y-%m-%d %H:%M:%S"))
print(f"# {VERSION}")
