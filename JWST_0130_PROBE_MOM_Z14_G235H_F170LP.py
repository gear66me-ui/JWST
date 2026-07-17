# JWST_0130
from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess

VERSION = "JWST_0130"
PROGRAM = "5224"
SOURCE_ID = "277193"
TARGET_RA = 150.09354167
TARGET_DEC = 2.27325000
OUT = Path("/content/JWST_OUTPUT")
CSV_DIR = OUT / "CSV"
DATA_DIR = OUT / "DATA" / VERSION / "FITS"
CSV_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 G235H / F170LP ARCHIVE PROBE")
print("-" * 112)

try:
    from astroquery.mast import Observations
    import pandas as pd
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "astroquery", "pandas", "astropy"])
    from astroquery.mast import Observations
    import pandas as pd


def col_text(table, name):
    if name not in table.colnames:
        return pd.Series([""] * len(table))
    return pd.Series(["" if v is None else str(v) for v in table[name]])

print(f"PROGRAM: {PROGRAM}")
print(f"MOM-z14 SOURCE ID: {SOURCE_ID}")
print(f"ADOPTED COORDINATES: RA={TARGET_RA:.8f} deg  Dec={TARGET_DEC:+.8f} deg")
print("QUERYING MAST OBSERVATIONS...")

obs = Observations.query_criteria(proposal_id=PROGRAM, instrument_name="NIRSPEC*")
print(f"NIRSPEC OBSERVATION RECORDS: {len(obs)}")

obs_df = obs.to_pandas()
obs_path = CSV_DIR / f"{VERSION}_PROGRAM_{PROGRAM}_NIRSPEC_OBSERVATIONS.csv"
obs_df.to_csv(obs_path, index=False)

filters = col_text(obs, "filters").str.upper()
gratings = col_text(obs, "grating").str.upper()
if gratings.eq("").all():
    gratings = col_text(obs, "gratings").str.upper()
mode_text = filters + " " + gratings + " " + col_text(obs, "obs_id").str.upper()
mode_mask = mode_text.str.contains("F170LP", regex=False) & mode_text.str.contains("G235H", regex=False)
mode_obs = obs[mode_mask.to_numpy()]

print(f"G235H/F170LP OBSERVATION RECORDS: {len(mode_obs)}")

mode_obs_df = mode_obs.to_pandas() if len(mode_obs) else pd.DataFrame()
mode_obs_path = CSV_DIR / f"{VERSION}_G235H_F170LP_OBSERVATIONS.csv"
mode_obs_df.to_csv(mode_obs_path, index=False)

if len(mode_obs) == 0:
    print("\nRESULT")
    print("  No G235H/F170LP observation records were found in program 5224.")
    print("  This means there is no same-program high-resolution product to download for source 277193.")
else:
    print("RETRIEVING PRODUCTS FOR G235H/F170LP OBSERVATIONS...")
    products = Observations.get_product_list(mode_obs)
    prod_df = products.to_pandas()
    if "dataURI" in prod_df.columns:
        prod_df = prod_df.drop_duplicates(subset=["dataURI"])
    elif "productFilename" in prod_df.columns:
        prod_df = prod_df.drop_duplicates(subset=["productFilename"])
    products_path = CSV_DIR / f"{VERSION}_G235H_F170LP_ALL_PRODUCTS.csv"
    prod_df.to_csv(products_path, index=False)

    names = prod_df.get("productFilename", pd.Series([""] * len(prod_df))).fillna("").astype(str)
    uris = prod_df.get("dataURI", pd.Series([""] * len(prod_df))).fillna("").astype(str)
    exact_mask = names.str.contains(f"s000{SOURCE_ID}", case=False, regex=False) | names.str.contains(f"s{SOURCE_ID}", case=False, regex=False)
    exact = prod_df[exact_mask].copy()
    exact_path = CSV_DIR / f"{VERSION}_MOM_Z14_SOURCE_{SOURCE_ID}_G235H_F170LP_PRODUCTS.csv"
    exact.to_csv(exact_path, index=False)

    print(f"UNIQUE G235H/F170LP PRODUCTS: {len(prod_df)}")
    print(f"EXACT SOURCE-{SOURCE_ID} PRODUCTS: {len(exact)}")

    if len(exact):
        keep_suffixes = ("_x1d.fits", "_s2d.fits", "_crf.fits", "_cal.fits")
        selected = exact[exact["productFilename"].astype(str).str.lower().str.endswith(keep_suffixes)].copy()
        print("\nEXACT MATCHES")
        for _, row in exact.iterrows():
            print(f"  {row.get('productFilename', '')}")
        if len(selected):
            print("\nDOWNLOADING SCIENCE PRODUCTS...")
            manifest = Observations.download_products(selected, download_dir=str(DATA_DIR), cache=True)
            manifest_path = CSV_DIR / f"{VERSION}_DOWNLOAD_MANIFEST.csv"
            manifest.to_pandas().to_csv(manifest_path, index=False)
            print(f"DOWNLOADED/ATTEMPTED: {len(manifest)}")
        else:
            print("No x1d/s2d/crf/cal FITS products were present among the exact matches.")
    else:
        print("\nRESULT")
        print("  G235H/F170LP exists in the program, but no level-3 product filename matches source 277193.")
        print("  Therefore we cannot claim that MoM-z14 itself has a G235H/F170LP spectrum from this program.")

print("\nOUTPUT SUMMARY")
print(f"All NIRSpec observations: {obs_path}")
print(f"G235H/F170LP observations: {mode_obs_path}")
print(f"FITS directory: {DATA_DIR}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
