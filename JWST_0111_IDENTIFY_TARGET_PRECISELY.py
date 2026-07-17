# JWST_0111_IDENTIFY_TARGET_PRECISELY.py

from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
from astropy.io import fits

VERSION = "JWST_0111"
TARGET_RA_DEG = 3.6171694
TARGET_DEC_DEG = -30.4255494
TARGET_Z = 9.31102
ROOT = Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS")
OUTDIR = Path("/content/JWST_OUTPUT/CSV")
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTCSV = OUTDIR / f"{VERSION}_TARGET_IDENTITY_AUDIT.csv"

print(f"CODE OUTPUT: {VERSION}")

RA_KEYS = [
    "TARG_RA", "RA_TARG", "TARGETRA", "RA_OBJ", "OBJ_RA", "SRC_RA",
    "SOURCE_RA", "RA", "CRVAL1"
]
DEC_KEYS = [
    "TARG_DEC", "DEC_TARG", "TARGETDEC", "DEC_OBJ", "OBJ_DEC", "SRC_DEC",
    "SOURCE_DEC", "DEC", "CRVAL2"
]
ID_KEYS = [
    "SOURCEID", "SOURCE_ID", "SRCID", "SRC_ID", "TARGETID", "TARGET_ID",
    "TARGNAME", "OBJECT", "OBJNAME", "SLTNAME", "MSA_ID"
]
MODE_KEYS = [
    "GRATING", "FILTER", "EXP_TYPE", "INSTRUME", "DETECTOR", "PROGRAM",
    "OBSERVTN", "VISIT", "TEMPLATE"
]
Z_KEYS = ["REDSHIFT", "Z", "SPECZ", "ZSPEC", "PHOTOZ"]


def to_float(value):
    try:
        if value is None:
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def angular_sep_arcsec(ra1, dec1, ra2, dec2):
    if not all(np.isfinite([ra1, dec1, ra2, dec2])):
        return np.nan
    r1 = math.radians(ra1)
    d1 = math.radians(dec1)
    r2 = math.radians(ra2)
    d2 = math.radians(dec2)
    cosang = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    cosang = max(-1.0, min(1.0, cosang))
    return math.degrees(math.acos(cosang)) * 3600.0


def first_header_value(headers, keys):
    for header in headers:
        for key in keys:
            if key in header:
                value = header.get(key)
                if value not in (None, ""):
                    return value, key
    return None, None


def filename_source_id(name):
    patterns = [
        r"abell2744-1324-(\d+)",
        r"_s(\d+)_",
        r"source[-_](\d+)",
        r"src[-_](\d+)"
    ]
    lower = name.lower()
    for pattern in patterns:
        m = re.search(pattern, lower)
        if m:
            return m.group(1)
    return ""


def table_identity_values(hdul):
    found = {}
    wanted = {
        "source_id", "sourceid", "src_id", "srcid", "target_id", "targetid",
        "ra", "dec", "source_ra", "source_dec", "targ_ra", "targ_dec",
        "redshift", "z", "specz", "zspec", "slit_name", "slitname"
    }
    for hdu_index, hdu in enumerate(hdul):
        data = hdu.data
        names = getattr(data, "names", None)
        if data is None or not names:
            continue
        for name in names:
            if name.lower() not in wanted:
                continue
            try:
                arr = np.asarray(data[name]).ravel()
                arr = arr[:20]
                values = []
                for value in arr:
                    if isinstance(value, bytes):
                        value = value.decode(errors="ignore")
                    text = str(value).strip()
                    if text and text not in values:
                        values.append(text)
                if values:
                    found[f"HDU{hdu_index}:{name}"] = "|".join(values[:5])
            except Exception:
                pass
    return found


rows = []
files = sorted(ROOT.rglob("*.fits"))

for path in files:
    try:
        with fits.open(path, memmap=False) as hdul:
            headers = [hdu.header for hdu in hdul]

            ra_value, ra_key = first_header_value(headers, RA_KEYS)
            dec_value, dec_key = first_header_value(headers, DEC_KEYS)
            id_value, id_key = first_header_value(headers, ID_KEYS)
            z_value, z_key = first_header_value(headers, Z_KEYS)

            ra = to_float(ra_value)
            dec = to_float(dec_value)
            z = to_float(z_value)
            sep = angular_sep_arcsec(TARGET_RA_DEG, TARGET_DEC_DEG, ra, dec)

            mode_values = {}
            for key in MODE_KEYS:
                value, used_key = first_header_value(headers, [key])
                if used_key:
                    mode_values[key] = str(value)

            table_values = table_identity_values(hdul)

            fname_id = filename_source_id(path.name)
            header_id = "" if id_value is None else str(id_value).strip()

            identity_text = " ".join([
                path.name.lower(),
                fname_id.lower(),
                header_id.lower(),
                " ".join(str(v).lower() for v in table_values.values())
            ])

            id_10003_match = "10003" in identity_text
            coordinate_match = bool(np.isfinite(sep) and sep <= 0.30)
            strong_coordinate_match = bool(np.isfinite(sep) and sep <= 0.10)

            score = 0
            if id_10003_match:
                score += 1000
            if strong_coordinate_match:
                score += 500
            elif coordinate_match:
                score += 250
            if np.isfinite(z) and abs(z - TARGET_Z) <= 0.01:
                score += 100
            if "g140h" in identity_text or "g235h" in identity_text or "g395h" in identity_text:
                score += 10

            rows.append({
                "score": score,
                "file": path.name,
                "full_path": str(path),
                "filename_source_id": fname_id,
                "header_id_key": id_key or "",
                "header_id_value": header_id,
                "ra_key": ra_key or "",
                "ra_deg": ra,
                "dec_key": dec_key or "",
                "dec_deg": dec,
                "separation_arcsec": sep,
                "coordinate_match_le_0p30_arcsec": coordinate_match,
                "strong_match_le_0p10_arcsec": strong_coordinate_match,
                "redshift_key": z_key or "",
                "redshift_value": z,
                "contains_source_10003": id_10003_match,
                "grating": mode_values.get("GRATING", ""),
                "filter": mode_values.get("FILTER", ""),
                "instrument": mode_values.get("INSTRUME", ""),
                "detector": mode_values.get("DETECTOR", ""),
                "program": mode_values.get("PROGRAM", ""),
                "observation": mode_values.get("OBSERVTN", ""),
                "table_identity_fields": "; ".join(f"{k}={v}" for k, v in table_values.items())
            })

    except Exception as exc:
        rows.append({
            "score": -1,
            "file": path.name,
            "full_path": str(path),
            "filename_source_id": "",
            "header_id_key": "",
            "header_id_value": "",
            "ra_key": "",
            "ra_deg": np.nan,
            "dec_key": "",
            "dec_deg": np.nan,
            "separation_arcsec": np.nan,
            "coordinate_match_le_0p30_arcsec": False,
            "strong_match_le_0p10_arcsec": False,
            "redshift_key": "",
            "redshift_value": np.nan,
            "contains_source_10003": False,
            "grating": "",
            "filter": "",
            "instrument": "",
            "detector": "",
            "program": "",
            "observation": "",
            "table_identity_fields": f"ERROR: {exc}"
        })


df = pd.DataFrame(rows)
df = df.sort_values(
    ["score", "separation_arcsec", "file"],
    ascending=[False, True, True],
    na_position="last"
).reset_index(drop=True)
df.to_csv(OUTCSV, index=False)

print(f"TARGET COORDINATES      RA={TARGET_RA_DEG:.7f} deg  Dec={TARGET_DEC_DEG:.7f} deg")
print(f"TARGET REDSHIFT        {TARGET_Z:.5f}")
print(f"FITS FILES AUDITED     {len(df)}")
print()

show_cols = [
    "score", "file", "filename_source_id", "header_id_value",
    "ra_deg", "dec_deg", "separation_arcsec", "redshift_value",
    "grating", "filter", "contains_source_10003"
]

with pd.option_context("display.max_colwidth", 90, "display.width", 240):
    print(df[show_cols].head(30).to_string(index=False))

print()
print("BEST MATCH PER GRATING")
for grating in ["G140H", "G235H", "G395H"]:
    mask = (
        df["file"].str.contains(grating, case=False, na=False) |
        df["full_path"].str.contains(grating, case=False, na=False) |
        df["grating"].astype(str).str.contains(grating, case=False, na=False)
    )
    sub = df[mask]
    if sub.empty:
        print(f"{grating:6s}  NONE")
        continue
    best = sub.iloc[0]
    print(
        f"{grating:6s}  score={int(best['score']):4d}  "
        f"sep={best['separation_arcsec'] if np.isfinite(best['separation_arcsec']) else np.nan:.6f} arcsec  "
        f"source={best['filename_source_id'] or best['header_id_value'] or 'UNKNOWN'}  "
        f"{best['file']}"
    )

print()
print(f"AUDIT CSV              {OUTCSV}")
print(f"END {VERSION}")
