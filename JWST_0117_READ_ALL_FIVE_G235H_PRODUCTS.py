# JWST_0117
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

VERSION = "JWST_0117"
print(f"CODE OUTPUT: {VERSION}")

TARGET_RA = 3.6171694
TARGET_DEC = -30.4255494
TARGET = SkyCoord(TARGET_RA * u.deg, TARGET_DEC * u.deg)

ROOTS = [
    Path("/content/JWST_OUTPUT/DATA/JWST_0116/FITS"),
    Path("/content/JWST_BACKUP_REPO/FITS_DATA/MOM_Z14_SOURCE_10003"),
]
OUT = Path("/content/JWST_OUTPUT/CSV")
OUT.mkdir(parents=True, exist_ok=True)

NAMES = [
    "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits",
    "jw01324-o006_s000010003_nirspec_f170lp-g235h_cal.fits",
    "jw01324-o006_s000010003_nirspec_f170lp-g235h_crf.fits",
    "jw01324-o006_s000010003_nirspec_f170lp-g235h_s2d.fits",
    "jw01324-o006_s000010003_nirspec_f170lp-g235h_x1d.fits",
]

RA_KEYS = ["RA_TARG", "TARG_RA", "SRCRA", "RA_OBJ", "RA"]
DEC_KEYS = ["DEC_TARG", "TARG_DEC", "SRCDEC", "DEC_OBJ", "DEC"]
META_KEYS = [
    "SOURCEID", "SRCNAME", "TARGNAME", "SLTNAME", "MSAMETFL",
    "GRATING", "FILTER", "EXP_TYPE", "PROGRAM", "OBSERVTN",
    "VISIT", "EXPOSURE", "DETECTOR", "APERNAME"
]


def find_file(name):
    hits = []
    for root in ROOTS:
        if root.exists():
            hits.extend(root.rglob(name))
    hits = list(dict.fromkeys(hits))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


def first_header_value(hdul, keys):
    for idx, hdu in enumerate(hdul):
        for key in keys:
            val = hdu.header.get(key)
            if val not in (None, ""):
                return val, f"HDU{idx}:{hdu.name}:{key}"
    return None, ""


def collect_metadata(hdul):
    out = {}
    for key in META_KEYS:
        val, src = first_header_value(hdul, [key])
        out[key] = val
        out[f"{key}_SOURCE"] = src
    return out


def array_shape(data):
    try:
        return tuple(np.shape(data))
    except Exception:
        return ()


def inspect_table_columns(hdu):
    names = []
    if getattr(hdu, "columns", None) is not None and hdu.columns.names:
        names = list(hdu.columns.names)
    return names

summary_rows = []
hdu_rows = []

print("READING ALL FIVE G235H PRODUCTS")
print("-" * 110)

for filename in NAMES:
    path = find_file(filename)
    with fits.open(path, memmap=False) as hdul:
        ra, ra_src = first_header_value(hdul, RA_KEYS)
        dec, dec_src = first_header_value(hdul, DEC_KEYS)
        offset = np.nan
        if ra is not None and dec is not None:
            try:
                measured = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
                offset = measured.separation(TARGET).arcsec
            except Exception:
                pass

        meta = collect_metadata(hdul)
        one_d_samples = 0
        finite_wave_samples = 0
        wave_min = np.nan
        wave_max = np.nan
        largest_array = 0
        largest_shape = ""

        for i, hdu in enumerate(hdul):
            data = hdu.data
            shape = array_shape(data)
            elems = int(np.prod(shape)) if shape else 0
            if elems > largest_array:
                largest_array = elems
                largest_shape = str(shape)

            columns = inspect_table_columns(hdu)
            wave_col = next((c for c in columns if c.lower() in {"wave", "wavelength"}), None)
            if wave_col and data is not None:
                try:
                    wave = np.asarray(data[wave_col], dtype=float).ravel()
                    one_d_samples = max(one_d_samples, wave.size)
                    finite = wave[np.isfinite(wave)]
                    finite_wave_samples = max(finite_wave_samples, finite.size)
                    if finite.size:
                        wave_min = np.nanmin([wave_min, finite.min()]) if np.isfinite(wave_min) else finite.min()
                        wave_max = np.nanmax([wave_max, finite.max()]) if np.isfinite(wave_max) else finite.max()
                except Exception:
                    pass

            hdu_rows.append({
                "file": filename,
                "hdu_index": i,
                "hdu_name": hdu.name,
                "shape": str(shape),
                "elements": elems,
                "columns": ", ".join(columns),
                "naxis": hdu.header.get("NAXIS"),
                "bunit": hdu.header.get("BUNIT"),
                "extver": hdu.header.get("EXTVER"),
            })

        row = {
            "file": filename,
            "path": str(path),
            "size_mb": path.stat().st_size / 1024 / 1024,
            "hdu_count": len(hdul),
            "ra_deg": ra,
            "dec_deg": dec,
            "ra_source": ra_src,
            "dec_source": dec_src,
            "offset_arcsec": offset,
            "native_1d_samples": one_d_samples,
            "finite_wave_samples": finite_wave_samples,
            "wave_min_um": wave_min,
            "wave_max_um": wave_max,
            "largest_array_elements": largest_array,
            "largest_array_shape": largest_shape,
            **meta,
        }
        summary_rows.append(row)

        status = "MATCH" if np.isfinite(offset) and offset <= 0.10 else "MISMATCH"
        print(filename)
        print(f"  RA/DEC       {ra}  {dec}")
        print(f"  OFFSET       {offset:.6f} arcsec  [{status}]" if np.isfinite(offset) else "  OFFSET       unavailable")
        print(f"  SOURCE ID    {meta.get('SOURCEID')}")
        print(f"  GRATING      {meta.get('GRATING')}   FILTER {meta.get('FILTER')}")
        print(f"  HDUs         {len(hdul)}")
        print(f"  1-D SAMPLES  {one_d_samples}")
        if np.isfinite(wave_min):
            print(f"  WAVE RANGE   {wave_min:.6f} to {wave_max:.6f} um")
        print(f"  LARGEST      {largest_shape} = {largest_array} elements")
        print()

summary_df = pd.DataFrame(summary_rows)
hdu_df = pd.DataFrame(hdu_rows)
summary_csv = OUT / f"{VERSION}_FIVE_PRODUCT_IDENTITY_SUMMARY.csv"
hdu_csv = OUT / f"{VERSION}_FIVE_PRODUCT_HDU_AUDIT.csv"
summary_df.to_csv(summary_csv, index=False)
hdu_df.to_csv(hdu_csv, index=False)

manifest = {
    "version": VERSION,
    "target_ra_deg": TARGET_RA,
    "target_dec_deg": TARGET_DEC,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "files": summary_rows,
}
manifest_path = OUT / f"{VERSION}_MANIFEST.json"
manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

print("OUTPUT SUMMARY")
print("Identity summary:", summary_csv)
print("HDU audit:", hdu_csv)
print("Manifest:", manifest_path)
print("Timestamp UTC:", datetime.now(timezone.utc).isoformat())
print(f"# {VERSION}")
