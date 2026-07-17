# JWST_0132
"""Route A: query DJA for the individual MoM-z14 msaexp spectrum and reproduce a paper-style figure."""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from astropy.io import fits

VERSION = "JWST_0132"
RA_DEG = 150.09354167
DEC_DEG = 2.27325000
Z = 14.44
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
DATA_DIR = ROOT / "DATA" / VERSION
for d in (PNG_DIR, CSV_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 ROUTE A — FIXED DJA INDIVIDUAL MSAEXP SPECTRUM RETRIEVAL")
print("-" * 112)

session = requests.Session()
session.headers.update({"User-Agent": f"{VERSION}/1.0"})


def get_bytes(url: str, timeout: int = 120) -> bytes:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def query_dja() -> pd.DataFrame:
    urls = [
        f"https://dawn-cph.github.io/dja/api/nirspec_extractions?coords={RA_DEG},{DEC_DEG}&size=2.0&output=csv",
        f"https://dja-api.onrender.com/nirspec_extractions?coords={RA_DEG},{DEC_DEG}&size=2.0&output=csv",
        f"https://grizli-cutout.herokuapp.com/nirspec_extractions?coords={RA_DEG},{DEC_DEG}&size=2.0&output=csv",
    ]
    errors = []
    for url in urls:
        try:
            raw = get_bytes(url, 90)
            text = raw.decode("utf-8", errors="replace")
            if "file" in text.lower() and len(text.splitlines()) > 1:
                df = pd.read_csv(io.StringIO(text))
                print(f"DJA QUERY OK: {url}")
                return df
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All DJA extraction API endpoints failed:\n" + "\n".join(errors))


def choose_candidate(df: pd.DataFrame) -> pd.Series:
    print(f"DJA CANDIDATES WITHIN QUERY RADIUS: {len(df)}")
    if len(df) == 0:
        raise RuntimeError("DJA returned no spectra near the adopted MoM-z14 coordinates.")
    low = {c: c.lower() for c in df.columns}
    file_col = next((c for c in df.columns if c.lower() == "file"), None)
    root_col = next((c for c in df.columns if c.lower() == "root"), None)
    src_col = next((c for c in df.columns if c.lower() in ("srcid", "source_id", "id")), None)
    grat_col = next((c for c in df.columns if c.lower() == "grating"), None)
    if file_col is None:
        raise RuntimeError(f"DJA response lacks a file column. Columns: {list(df.columns)}")
    score = np.zeros(len(df), dtype=float)
    text = df[file_col].astype(str).str.lower()
    score += text.str.contains("prism").astype(float) * 20
    score += text.str.contains("277193").astype(float) * 50
    if root_col:
        roots = df[root_col].astype(str).str.lower()
        score += roots.str.contains("mom").astype(float) * 30
        score += roots.str.contains("cos").astype(float) * 10
    if src_col:
        score += (df[src_col].astype(str).str.replace(".0", "", regex=False) == "277193").astype(float) * 100
    if grat_col:
        score += df[grat_col].astype(str).str.upper().str.contains("PRISM").astype(float) * 30
    if "ra" in low.values() and "dec" in low.values():
        ra_col = next(c for c in df.columns if c.lower() == "ra")
        de_col = next(c for c in df.columns if c.lower() == "dec")
        sep = np.hypot((df[ra_col].astype(float)-RA_DEG)*np.cos(np.deg2rad(DEC_DEG)), df[de_col].astype(float)-DEC_DEG)*3600
        score += np.clip(10-sep, 0, 10)
    df2 = df.copy()
    df2["selection_score"] = score
    show_cols = [c for c in [root_col, file_col, src_col, grat_col, "ra", "dec", "wmin", "wmax", "z"] if c and c in df2.columns]
    print(df2.sort_values("selection_score", ascending=False)[show_cols + ["selection_score"]].head(12).to_string(index=False))
    df2.to_csv(CSV_DIR / f"{VERSION}_DJA_QUERY_CANDIDATES.csv", index=False)
    return df2.iloc[int(np.argmax(score))]


def download_spec(row: pd.Series) -> Path:
    file_name = str(row[[c for c in row.index if c.lower() == "file"][0]])
    root_name = str(row[[c for c in row.index if c.lower() == "root"][0]]) if any(c.lower()=="root" for c in row.index) else ""
    candidates = []
    if file_name.startswith("http"):
        candidates.append(file_name)
    candidates += [
        f"https://s3.amazonaws.com/msaexp-nirspec/extractions/{root_name}/{file_name}",
        f"https://msaexp-nirspec.s3.amazonaws.com/extractions/{root_name}/{file_name}",
    ]
    errors = []
    for url in candidates:
        try:
            print(f"TRYING: {url}")
            raw = get_bytes(url, 180)
            if len(raw) < 10000:
                raise RuntimeError(f"response too small ({len(raw)} bytes)")
            out = DATA_DIR / Path(file_name).name
            out.write_bytes(raw)
            print(f"DOWNLOADED DJA SPEC: {out}  {out.stat().st_size/1e6:.3f} MB")
            return out
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Unable to download individual DJA spectrum:\n" + "\n".join(errors))


def read_spec(path: Path):
    with fits.open(path, memmap=False) as hdul:
        print("\nDJA FITS HDU INVENTORY")
        for i, h in enumerate(hdul):
            print(f"{i:3d}  {h.name:12s}  shape={getattr(h.data, 'shape', None)}")
        if "SPEC1D" not in hdul:
            raise RuntimeError("Individual DJA file lacks SPEC1D extension.")
        tab = hdul["SPEC1D"].data
        names = [n.lower() for n in tab.names]
        def col(*opts):
            for o in opts:
                if o.lower() in names:
                    return np.asarray(tab[tab.names[names.index(o.lower())]], dtype=float)
            return None
        wave = col("wave", "wavelength", "lambda")
        flux = col("flux", "flam", "f_lambda")
        err = col("err", "error", "full_err")
        if wave is None or flux is None:
            raise RuntimeError(f"SPEC1D missing wave/flux columns: {tab.names}")
        sci = np.asarray(hdul["SCI"].data, dtype=float) if "SCI" in hdul else None
        if "ERR" in hdul:
            err2d = np.asarray(hdul["ERR"].data, dtype=float)
        elif "WHT" in hdul:
            wht = np.asarray(hdul["WHT"].data, dtype=float)
            err2d = np.where(wht > 0, 1/np.sqrt(wht), np.nan)
        else:
            err2d = None
        header = dict(hdul[0].header)
    if np.nanmedian(wave) > 100:
        wave_um = wave / 1e4 if np.nanmedian(wave) > 1000 else wave / 1000
    else:
        wave_um = wave
    return wave_um, flux, err, sci, err2d, header


def fetch_cutout(filter_name: str) -> np.ndarray | None:
    urls = [
        "https://grizli-cutout.herokuapp.com/thumb?" +
        f"size=3&scl=0.08&invert=False&filters={filter_name.lower()}-clear&pl=2&coord={RA_DEG}%20{DEC_DEG}&dpi_scale=4",
    ]
    for url in urls:
        try:
            raw = get_bytes(url, 60)
            img = plt.imread(io.BytesIO(raw), format="png")
            return img
        except Exception:
            pass
    return None


df = query_dja()
row = choose_candidate(df)
spec_path = download_spec(row)
wave_um, flux, err, sci, err2d, hdr = read_spec(spec_path)

mask = np.isfinite(wave_um) & np.isfinite(flux)
wave_um = wave_um[mask]
flux = flux[mask]
if err is not None:
    err = np.asarray(err)[mask]

# Save exact native spectrum
out_df = pd.DataFrame({"wavelength_um": wave_um, "wavelength_nm": wave_um*1000, "flux": flux})
if err is not None:
    out_df["error"] = err
out_df.to_csv(CSV_DIR / f"{VERSION}_MOM_Z14_DJA_NATIVE_SPECTRUM.csv", index=False)

filters = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
cutouts = [fetch_cutout(f) for f in filters]

fig = plt.figure(figsize=(16, 10), constrained_layout=False)
gs = fig.add_gridspec(3, 7, height_ratios=[1.05, 0.78, 3.2], hspace=0.05, wspace=0.03)
for j, (filt, img) in enumerate(zip(filters, cutouts)):
    ax = fig.add_subplot(gs[0, j])
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "cutout\nunavailable", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(filt, fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])

ax2 = fig.add_subplot(gs[1, :])
if sci is not None:
    if err2d is not None and err2d.shape == sci.shape:
        sn = np.divide(sci, err2d, out=np.full_like(sci, np.nan), where=np.isfinite(err2d) & (err2d > 0))
    else:
        med = np.nanmedian(np.abs(sci - np.nanmedian(sci, axis=0))) * 1.4826
        sn = sci / med if med > 0 else sci
    lo, hi = np.nanpercentile(sn[np.isfinite(sn)], [3, 99.5])
    ax2.imshow(sn, aspect="auto", origin="lower", cmap="coolwarm", vmin=lo, vmax=hi,
               extent=[wave_um.min(), wave_um.max(), 0, sn.shape[0]])
ax2.set_xlim(0.8, 5.3)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_ylabel("2-D S/N")

ax3 = fig.add_subplot(gs[2, :])
scale = 1.0
# DJA flux is commonly microJy; preserve native units and label honestly
ax3.step(wave_um, flux*scale, where="mid", lw=1.2, label="DJA msaexp native extraction")
if err is not None:
    good = np.isfinite(err)
    ax3.fill_between(wave_um[good], flux[good]-err[good], flux[good]+err[good], step="mid", alpha=0.25, label="1σ")
lines_nm = {
    "Lyα": 121.567,
    "N IV]": 148.7,
    "C IV": 154.95,
    "He II": 164.0,
    "O III]": 166.3,
    "N III]": 175.0,
    "C III]": 190.8,
}
for name, rest_nm in lines_nm.items():
    obs_um = rest_nm*(1+Z)/1000
    ax3.axvline(obs_um, ls="--", lw=0.8, alpha=0.55)
    ax3.text(obs_um, 0.98, name, rotation=90, ha="right", va="top", transform=ax3.get_xaxis_transform(), fontsize=9)
ax3.axhline(0, lw=0.7, alpha=0.6)
ax3.set_xlim(0.8, 5.3)
ax3.set_xlabel("Observed wavelength [μm]")
ax3.set_ylabel("Native DJA flux")
ax3.set_title("MoM-z14 — DJA msaexp Route A reproduction (source 277193, z = 14.44)")
ax3.grid(alpha=0.16)
ax3.legend(loc="upper right", fontsize=9)

png = PNG_DIR / f"{VERSION}_MOM_Z14_DJA_PAPER_STYLE_ROUTE_A.png"
fig.savefig(png, dpi=350, bbox_inches="tight")
plt.show()
plt.close(fig)

manifest = {
    "version": VERSION,
    "coordinates_deg": [RA_DEG, DEC_DEG],
    "redshift": Z,
    "selected_file": spec_path.name,
    "native_points": int(len(wave_um)),
    "wavelength_um_min": float(np.nanmin(wave_um)),
    "wavelength_um_max": float(np.nanmax(wave_um)),
    "plot": str(png),
}
(CSV_DIR / f"{VERSION}_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

print("\nOUTPUT SUMMARY")
print(f"Selected DJA file: {spec_path}")
print(f"Native points:      {len(wave_um)}")
print(f"Wavelength range:   {np.nanmin(wave_um):.6f} to {np.nanmax(wave_um):.6f} um")
print(f"Paper-style PNG:    {png}")
print(f"Native CSV:         {CSV_DIR / f'{VERSION}_MOM_Z14_DJA_NATIVE_SPECTRUM.csv'}")
print(f"Timestamp UTC:      {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
