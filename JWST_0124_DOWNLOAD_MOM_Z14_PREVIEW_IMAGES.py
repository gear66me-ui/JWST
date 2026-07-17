# JWST_0124
# Download and display official MAST preview images for MoM-z14 source 277193.

from pathlib import Path
from urllib.parse import quote
from datetime import datetime, timezone
import time
import requests
import pandas as pd
from IPython.display import display, Image

VERSION = "JWST_0124"
SOURCE_ID = "277193"
CATALOG = Path("/content/JWST_OUTPUT/CSV/JWST_0123_SOURCE_277193_PRODUCTS.csv")
OUT = Path(f"/content/JWST_OUTPUT/PNG/{VERSION}_MOM_Z14_PREVIEWS")
CSV = Path("/content/JWST_OUTPUT/CSV")
OUT.mkdir(parents=True, exist_ok=True)
CSV.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print(f"SOURCE: {SOURCE_ID}")
print("DOWNLOADING OFFICIAL MAST PREVIEW IMAGES")
print("-" * 100)

if not CATALOG.exists():
    raise FileNotFoundError(
        f"Missing {CATALOG}. Run JWST_0123 first so the exact MAST product URIs are available."
    )

products = pd.read_csv(CATALOG)
name_col = "productFilename" if "productFilename" in products.columns else None
uri_col = "dataURI" if "dataURI" in products.columns else None
if not name_col or not uri_col:
    raise RuntimeError("Catalog does not contain productFilename and dataURI columns.")

mask = products[name_col].astype(str).str.lower().str.endswith((".jpg", ".jpeg", ".png"))
images = products.loc[mask, [name_col, uri_col]].drop_duplicates().copy()
if images.empty:
    raise RuntimeError("No JPEG or PNG preview products were listed for source 277193.")

session = requests.Session()
session.headers.update({"User-Agent": "JWST-0124-MoM-z14-preview-downloader"})
rows = []

for _, row in images.iterrows():
    filename = str(row[name_col])
    uri = str(row[uri_col])
    path = OUT / filename
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + quote(uri, safe="")
    status = "FAILED"
    error = ""

    for attempt in range(1, 5):
        try:
            print(f"{filename} — attempt {attempt}/4", flush=True)
            with session.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with open(path, "wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if path.exists() and path.stat().st_size > 0:
                status = "DOWNLOADED"
                break
        except Exception as exc:
            error = str(exc)
            time.sleep(2 * attempt)

    size_kb = path.stat().st_size / 1024 if path.exists() else 0.0
    rows.append({
        "filename": filename,
        "data_uri": uri,
        "status": status,
        "size_kb": size_kb,
        "local_path": str(path),
        "error": error,
    })

manifest = pd.DataFrame(rows)
manifest_path = CSV / f"{VERSION}_MOM_Z14_PREVIEW_MANIFEST.csv"
manifest.to_csv(manifest_path, index=False)

print("\nDOWNLOADED PREVIEWS")
for _, row in manifest.iterrows():
    print(f"{row['status']:<11} {row['size_kb']:9.1f} kB  {row['local_path']}")

print("\nINLINE PREVIEW")
for _, row in manifest[manifest["status"] == "DOWNLOADED"].iterrows():
    print(row["filename"])
    display(Image(filename=row["local_path"]))

print("\nOUTPUT SUMMARY")
print(f"Preview folder: {OUT}")
print(f"Manifest: {manifest_path}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
