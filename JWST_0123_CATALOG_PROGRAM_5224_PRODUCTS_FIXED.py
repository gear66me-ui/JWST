# JWST_0123
# Catalog JWST Program 5224 NIRSpec products once, without duplicate batch retrieval.

from pathlib import Path
from datetime import datetime, timezone
import time
import numpy as np
import pandas as pd
from astroquery.mast import Observations

VERSION = "JWST_0123"
PROGRAM = "5224"
TARGET_SOURCE_ID = "277193"
ROOT = Path("/content/JWST_OUTPUT")
CSV_DIR = ROOT / "CSV"
CSV_DIR.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print(f"PROGRAM: {PROGRAM}")
print("CATALOGING PROGRAM PRODUCTS — SINGLE RETRIEVAL, NO DUPLICATE BATCH LOOP")
print("-" * 110, flush=True)


def retry(label, fn, attempts=5):
    last = None
    for i in range(1, attempts + 1):
        try:
            print(f"{label} — attempt {i}/{attempts}", flush=True)
            return fn()
        except Exception as exc:
            last = exc
            print(f"  retry after: {type(exc).__name__}: {exc}", flush=True)
            if i < attempts:
                time.sleep(4 * i)
    raise last


obs = retry(
    "QUERY PROGRAM 5224",
    lambda: Observations.query_criteria(obs_collection="JWST", proposal_id=PROGRAM),
)

odf = obs.to_pandas()
for col in odf.columns:
    if odf[col].dtype == object:
        odf[col] = odf[col].fillna("").astype(str)

if "instrument_name" in odf.columns:
    odf = odf[odf["instrument_name"].str.upper().str.contains("NIRSPEC", na=False)].copy()

print(f"NIRSPEC OBSERVATION RECORDS: {len(odf)}", flush=True)
if len(odf) == 0:
    raise RuntimeError("No NIRSpec observations found for program 5224.")

obs_csv = CSV_DIR / f"{VERSION}_PROGRAM_5224_NIRSPEC_OBSERVATIONS.csv"
odf.to_csv(obs_csv, index=False)

# Rebuild an Astropy table containing only the selected observation rows.
keep_obsids = set(odf.get("obsid", pd.Series(dtype=str)).astype(str))
if "obsid" in obs.colnames and keep_obsids:
    mask = np.array([str(v) in keep_obsids for v in obs["obsid"]])
    obs_selected = obs[mask]
else:
    obs_selected = obs

products = retry(
    "FETCH COMPLETE PRODUCT TABLE ONCE",
    lambda: Observations.get_product_list(obs_selected),
)

pdf = products.to_pandas()
for col in pdf.columns:
    if pdf[col].dtype == object:
        pdf[col] = pdf[col].fillna("").astype(str)

print(f"RAW PRODUCT ROWS: {len(pdf)}", flush=True)

# Deduplicate defensively by URI first, filename second.
if "dataURI" in pdf.columns:
    before = len(pdf)
    pdf = pdf.drop_duplicates(subset=["dataURI"], keep="first").copy()
    print(f"UNIQUE PRODUCT URIs: {len(pdf)}  removed duplicates: {before-len(pdf)}", flush=True)
elif "productFilename" in pdf.columns:
    before = len(pdf)
    pdf = pdf.drop_duplicates(subset=["productFilename"], keep="first").copy()
    print(f"UNIQUE FILENAMES: {len(pdf)}  removed duplicates: {before-len(pdf)}", flush=True)

name_col = "productFilename"
if name_col not in pdf.columns:
    raise RuntimeError("MAST product table has no productFilename column.")

names = pdf[name_col].str.lower()
pdf["is_fits"] = names.str.endswith(".fits")
pdf["is_x1d"] = names.str.endswith("_x1d.fits")
pdf["is_s2d"] = names.str.endswith("_s2d.fits")
pdf["is_cal"] = names.str.endswith("_cal.fits")
pdf["is_crf"] = names.str.endswith("_crf.fits")
pdf["contains_target_source_id"] = names.str.contains(TARGET_SOURCE_ID, regex=False)
pdf["contains_prism"] = names.str.contains("prism", regex=False)

all_csv = CSV_DIR / f"{VERSION}_PROGRAM_5224_ALL_UNIQUE_PRODUCTS.csv"
pdf.to_csv(all_csv, index=False)

# Compact extension summary.
ext_rows = []
for label, flag in [
    ("FITS", "is_fits"),
    ("X1D", "is_x1d"),
    ("S2D", "is_s2d"),
    ("CAL", "is_cal"),
    ("CRF", "is_crf"),
    ("SOURCE_277193", "contains_target_source_id"),
]:
    ext_rows.append({"category": label, "count": int(pdf[flag].sum())})
summary = pd.DataFrame(ext_rows)
summary_csv = CSV_DIR / f"{VERSION}_PRODUCT_TYPE_SUMMARY.csv"
summary.to_csv(summary_csv, index=False)

# Candidate MoM-z14 products: exact source ID in filename, then all x1d for manual audit.
target_products = pdf[pdf["contains_target_source_id"]].copy()
target_csv = CSV_DIR / f"{VERSION}_SOURCE_277193_PRODUCTS.csv"
target_products.to_csv(target_csv, index=False)

x1d_products = pdf[pdf["is_x1d"]].copy()
x1d_csv = CSV_DIR / f"{VERSION}_ALL_X1D_PRODUCTS.csv"
x1d_products.to_csv(x1d_csv, index=False)

print("\nPRODUCT SUMMARY")
for _, row in summary.iterrows():
    print(f"  {row['category']:<18} {int(row['count']):>7}")

print(f"\nSOURCE {TARGET_SOURCE_ID} PRODUCT ROWS: {len(target_products)}")
if len(target_products):
    cols = [c for c in ["productFilename", "description", "productType", "calib_level", "size", "dataURI"] if c in target_products.columns]
    for _, row in target_products[cols].head(100).iterrows():
        print(f"  {row.get('productFilename', '')}")
else:
    print("  No filename contains source ID 277193.")
    print("  Use the all-X1D CSV to map source identity from FITS headers or MSA metadata.")

print("\nOUTPUT SUMMARY")
print(f"Observations: {obs_csv}")
print(f"All unique products: {all_csv}")
print(f"Product summary: {summary_csv}")
print(f"Source 277193 products: {target_csv}")
print(f"All X1D products: {x1d_csv}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
