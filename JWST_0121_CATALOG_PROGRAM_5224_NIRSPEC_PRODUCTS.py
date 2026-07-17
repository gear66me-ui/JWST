# JWST_0121
# Catalog all publicly available JWST/NIRSpec products for program 5224 without any pointing-center cone cut.

from pathlib import Path
from datetime import datetime, timezone
import time
import json
import numpy as np
import pandas as pd
from astroquery.mast import Observations

VERSION = "JWST_0121"
PROGRAM = "5224"
ROOT = Path("/content/JWST_OUTPUT")
CSV_DIR = ROOT / "CSV"
CSV_DIR.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}", flush=True)
print(f"PROGRAM: {PROGRAM}", flush=True)
print("CATALOGING ALL JWST/NIRSPEC OBSERVATIONS AND PRODUCTS", flush=True)
print("NO TARGET-CENTER CONE CUT WILL BE APPLIED", flush=True)
print("-" * 110, flush=True)


def retry(label, fn, attempts=5, delay=8):
    last = None
    for i in range(1, attempts + 1):
        try:
            print(f"{label} — attempt {i}/{attempts}", flush=True)
            return fn()
        except Exception as exc:
            last = exc
            print(f"  retry after error: {type(exc).__name__}: {exc}", flush=True)
            if i < attempts:
                time.sleep(delay * i)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


obs = retry(
    "QUERY PROGRAM 5224",
    lambda: Observations.query_criteria(
        obs_collection="JWST",
        proposal_id=PROGRAM,
    ),
)

obs_df = obs.to_pandas()
for col in obs_df.columns:
    if obs_df[col].dtype == object:
        obs_df[col] = obs_df[col].fillna("").astype(str)

print(f"PROGRAM OBSERVATION RECORDS: {len(obs_df)}", flush=True)

instrument_col = "instrument_name" if "instrument_name" in obs_df.columns else None
if instrument_col:
    nirspec_mask = obs_df[instrument_col].str.upper().str.contains("NIRSPEC", na=False)
else:
    nirspec_mask = pd.Series(False, index=obs_df.index)

nirspec_obs_df = obs_df[nirspec_mask].copy()
print(f"NIRSPEC OBSERVATION RECORDS: {len(nirspec_obs_df)}", flush=True)
if len(nirspec_obs_df) == 0:
    raise RuntimeError("Program 5224 returned no NIRSpec observation records.")

obs_csv = CSV_DIR / f"{VERSION}_PROGRAM_5224_NIRSPEC_OBSERVATIONS.csv"
nirspec_obs_df.to_csv(obs_csv, index=False)

print("\nOBSERVATION SUMMARY", flush=True)
summary_fields = [
    "instrument_name", "filters", "dataproduct_type", "calib_level",
    "intentType", "target_name", "obs_id", "proposal_pi"
]
for field in summary_fields:
    if field in nirspec_obs_df.columns:
        counts = nirspec_obs_df[field].replace("", "<blank>").value_counts(dropna=False).head(25)
        print(f"\n{field}", flush=True)
        for value, count in counts.items():
            print(f"  {str(value)[:82]:82s} {int(count):6d}", flush=True)

# Fetch products in small batches so one large MAST response cannot stall the whole run.
print("\nFETCHING PRODUCT TABLES IN BATCHES", flush=True)
BATCH = 20
product_frames = []
for start in range(0, len(nirspec_obs_df), BATCH):
    stop = min(start + BATCH, len(nirspec_obs_df))
    batch_table = obs[start:stop] if len(obs) == len(nirspec_obs_df) else obs[np.where(nirspec_mask.to_numpy())[0][start:stop]]
    label = f"PRODUCT BATCH {start + 1}-{stop} OF {len(nirspec_obs_df)}"
    try:
        products = retry(label, lambda bt=batch_table: Observations.get_product_list(bt), attempts=4, delay=6)
        pdf = products.to_pandas()
        if len(pdf):
            product_frames.append(pdf)
        print(f"  products returned: {len(pdf)}", flush=True)
    except Exception as exc:
        print(f"  BATCH FAILED AND WAS SKIPPED: {exc}", flush=True)

if not product_frames:
    raise RuntimeError("No product tables were returned for the NIRSpec observations.")

products_df = pd.concat(product_frames, ignore_index=True, sort=False)
products_df = products_df.drop_duplicates().reset_index(drop=True)
for col in products_df.columns:
    if products_df[col].dtype == object:
        products_df[col] = products_df[col].fillna("").astype(str)

name_col = "productFilename" if "productFilename" in products_df.columns else None
if not name_col:
    raise RuntimeError("MAST product table does not contain productFilename.")

name = products_df[name_col].str.lower()
products_df["is_fits"] = name.str.endswith(".fits")
products_df["is_prism"] = name.str.contains("prism", na=False)
products_df["is_clear"] = name.str.contains("clear", na=False)
products_df["is_x1d"] = name.str.endswith("_x1d.fits")
products_df["is_s2d"] = name.str.endswith("_s2d.fits")
products_df["is_cal"] = name.str.endswith("_cal.fits")
products_df["is_crf"] = name.str.endswith("_crf.fits")
products_df["is_rate"] = name.str.endswith("_rate.fits")
products_df["is_rateints"] = name.str.endswith("_rateints.fits")
products_df["mentions_277193"] = name.str.contains("277193", na=False)
products_df["mentions_mom"] = name.str.contains("mom", na=False)
products_df["mentions_z14"] = name.str.contains("z14", na=False)

if "size" in products_df.columns:
    size_num = pd.to_numeric(products_df["size"], errors="coerce")
    products_df["size_MB"] = size_num / 1_000_000.0
else:
    products_df["size_MB"] = np.nan

all_csv = CSV_DIR / f"{VERSION}_PROGRAM_5224_ALL_NIRSPEC_PRODUCTS.csv"
products_df.to_csv(all_csv, index=False)

science_mask = products_df["is_fits"] & (
    products_df["is_prism"] |
    products_df["is_x1d"] |
    products_df["is_s2d"] |
    products_df["is_cal"] |
    products_df["is_crf"] |
    products_df["mentions_277193"]
)
science_df = products_df[science_mask].copy()
science_csv = CSV_DIR / f"{VERSION}_PROGRAM_5224_SCIENCE_CANDIDATES.csv"
science_df.to_csv(science_csv, index=False)

print("\nPRODUCT CATALOG SUMMARY", flush=True)
print(f"UNIQUE PRODUCTS:              {len(products_df):8d}", flush=True)
print(f"FITS PRODUCTS:                {int(products_df['is_fits'].sum()):8d}", flush=True)
print(f"PRISM-NAMED PRODUCTS:         {int(products_df['is_prism'].sum()):8d}", flush=True)
print(f"CLEAR-NAMED PRODUCTS:         {int(products_df['is_clear'].sum()):8d}", flush=True)
print(f"X1D PRODUCTS:                 {int(products_df['is_x1d'].sum()):8d}", flush=True)
print(f"S2D PRODUCTS:                 {int(products_df['is_s2d'].sum()):8d}", flush=True)
print(f"CAL PRODUCTS:                 {int(products_df['is_cal'].sum()):8d}", flush=True)
print(f"CRF PRODUCTS:                 {int(products_df['is_crf'].sum()):8d}", flush=True)
print(f"RATE PRODUCTS:                {int(products_df['is_rate'].sum()):8d}", flush=True)
print(f"RATEINTS PRODUCTS:            {int(products_df['is_rateints'].sum()):8d}", flush=True)
print(f"FILENAMES MENTIONING 277193:  {int(products_df['mentions_277193'].sum()):8d}", flush=True)
print(f"FILENAMES MENTIONING MOM:     {int(products_df['mentions_mom'].sum()):8d}", flush=True)
print(f"FILENAMES MENTIONING Z14:     {int(products_df['mentions_z14'].sum()):8d}", flush=True)

for field in ["productSubGroupDescription", "productType", "description", "calib_level"]:
    if field in products_df.columns:
        counts = products_df[field].replace("", "<blank>").value_counts(dropna=False).head(30)
        print(f"\n{field}", flush=True)
        for value, count in counts.items():
            print(f"  {str(value)[:82]:82s} {int(count):6d}", flush=True)

print("\nMOST RELEVANT CANDIDATE FILENAMES", flush=True)
ranked = science_df.copy()
ranked["rank"] = (
    ranked["mentions_277193"].astype(int) * 100 +
    ranked["is_x1d"].astype(int) * 50 +
    ranked["is_s2d"].astype(int) * 40 +
    ranked["is_prism"].astype(int) * 30 +
    ranked["is_clear"].astype(int) * 20 +
    ranked["is_cal"].astype(int) * 10
)
ranked = ranked.sort_values(["rank", "size_MB"], ascending=[False, True])
for _, row in ranked.head(100).iterrows():
    size_text = "" if not np.isfinite(row["size_MB"]) else f"{row['size_MB']:.3f} MB"
    print(f"  {row[name_col]}  {size_text}", flush=True)

manifest = {
    "version": VERSION,
    "program": PROGRAM,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "nirspec_observation_records": int(len(nirspec_obs_df)),
    "unique_products": int(len(products_df)),
    "fits_products": int(products_df["is_fits"].sum()),
    "x1d_products": int(products_df["is_x1d"].sum()),
    "s2d_products": int(products_df["is_s2d"].sum()),
    "prism_named_products": int(products_df["is_prism"].sum()),
    "source_277193_filename_hits": int(products_df["mentions_277193"].sum()),
    "outputs": {
        "observations_csv": str(obs_csv),
        "all_products_csv": str(all_csv),
        "science_candidates_csv": str(science_csv),
    },
}
manifest_path = CSV_DIR / f"{VERSION}_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("\nOUTPUT SUMMARY", flush=True)
print(f"Observation catalog: {obs_csv}", flush=True)
print(f"All products:        {all_csv}", flush=True)
print(f"Science candidates:  {science_csv}", flush=True)
print(f"Manifest:            {manifest_path}", flush=True)
print(f"Timestamp UTC:       {manifest['timestamp_utc']}", flush=True)
print(f"# {VERSION}", flush=True)
