# JWST_0131
"""Route A: retrieve the DAWN JWST Archive PRISM spectrum for MoM-z14 and build a paper-style figure.

Uses only public scientific data and matplotlib. No AI-generated imagery.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

VERSION = "JWST_0131"
SOURCE_KEY = "5224_277193"
SOURCE_ID = "277193"
RA_DEG = 150.09333
DEC_DEG = 2.27316
Z_SPEC = 14.44
ZENODO_RECORD = "15472354"
ZENODO_PRISM_NAME = "dja_msaexp_emission_lines_v4.4.prism_spectra.fits"
MAST_S2D = "jw05224-o004_s000277193_nirspec_clear-prism_s2d.fits"
ROOT = Path("/content/JWST_OUTPUT")
DATA_DIR = ROOT / "DATA" / VERSION
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for p in (DATA_DIR, PNG_DIR, CSV_DIR):
    p.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 ROUTE A — DAWN JWST ARCHIVE PAPER-STYLE SPECTRUM")
print("-" * 112)


def get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def stream_download(url: str, out: Path, expected: int | None = None) -> Path:
    tmp = out.with_suffix(out.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    mode = "ab" if done else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
        if done and r.status_code == 200:
            done = 0
            mode = "wb"
        r.raise_for_status()
        total = expected or int(r.headers.get("Content-Length", "0")) + done
        last = time.time()
        with open(tmp, mode) as f:
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 2:
                    pct = 100.0 * done / total if total else float("nan")
                    print(f"  {done/1e6:9.1f} MB / {total/1e6:9.1f} MB  ({pct:5.1f}%)")
                    last = time.time()
    tmp.replace(out)
    return out


def zenodo_file(name: str) -> Path:
    meta = get_json(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
    files = meta.get("files", [])
    hit = next((x for x in files if x.get("key") == name), None)
    if hit is None:
        raise RuntimeError(f"Zenodo file not found: {name}")
    out = DATA_DIR / name
    size = int(hit.get("size", 0))
    if out.exists() and (size == 0 or out.stat().st_size == size):
        print(f"USING CACHED: {out.name}  {out.stat().st_size/1e6:.1f} MB")
        return out
    url = hit.get("links", {}).get("content") or hit.get("links", {}).get("self")
    print(f"DOWNLOADING DAWN MERGED PRISM FILE: {name}  ({size/1e6:.1f} MB)")
    return stream_download(url, out, size)


def mast_download(filename: str) -> Path:
    out = DATA_DIR / filename
    if out.exists() and out.stat().st_size > 1000:
        return out
    uri = f"mast:JWST/product/{filename}"
    url = "https://mast.stsci.edu/api/v0.1/Download/file?uri=" + requests.utils.quote(uri, safe=":/")
    print(f"DOWNLOADING MAST PRODUCT: {filename}")
    return stream_download(url, out)


def text_match(value: Any) -> bool:
    try:
        if isinstance(value, bytes):
            value = value.decode("utf-8", "ignore")
        s = str(value)
        return SOURCE_KEY in s or SOURCE_ID in s
    except Exception:
        return False


def candidate_id_columns(names: Iterable[str]) -> list[str]:
    pats = ("id", "root", "source", "file", "name", "msa", "dataset", "program")
    return [n for n in names if any(p in n.lower() for p in pats)]


def locate_source_row(path: Path) -> tuple[int, int, dict[str, Any]]:
    print("SCANNING DAWN FITS TABLE FOR SOURCE 5224_277193 ...")
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        for hi, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or [])
            if data is None or not names:
                continue
            ids = candidate_id_columns(names)
            for col in ids:
                arr = data[col]
                for i, val in enumerate(arr):
                    if text_match(val):
                        meta = {n: data[n][i] for n in ids if np.ndim(data[n][i]) == 0}
                        print(f"FOUND SOURCE: HDU={hi} ROW={i} MATCH_COLUMN={col} VALUE={val}")
                        return hi, i, meta
    raise RuntimeError("Source 5224_277193 was not found in the DAWN merged PRISM table.")


def flatten_numeric(v: Any) -> np.ndarray:
    try:
        a = np.asarray(v)
        if a.dtype.kind in "iuf":
            return np.asarray(a, dtype=float).ravel()
    except Exception:
        pass
    return np.array([], dtype=float)


def score_column(name: str, kind: str) -> int:
    n = name.lower()
    groups = {
        "wave": (("wave", 8), ("lambda", 8), ("lam", 4)),
        "flux": (("flux", 8), ("flam", 7), ("fnu", 7), ("spec", 3)),
        "err": (("err", 8), ("sigma", 8), ("unc", 7), ("error", 8)),
    }
    s = sum(w for token, w in groups[kind] if token in n)
    if kind == "flux" and any(x in n for x in ("err", "model", "cont", "line")):
        s -= 8
    return s


def extract_spectrum(path: Path, hdu_index: int, row_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        hdu = hdul[hdu_index]
        row = hdu.data[row_index]
        names = list(hdu.data.names)
        vectors: dict[str, np.ndarray] = {}
        for n in names:
            a = flatten_numeric(row[n])
            if a.size >= 20:
                vectors[n] = a
        if not vectors:
            raise RuntimeError("No vector-valued spectrum columns found in matched DAWN row.")

        def choose(kind: str, length: int | None = None) -> str:
            options = []
            for n, a in vectors.items():
                if length is not None and a.size != length:
                    continue
                options.append((score_column(n, kind), n))
            options.sort(reverse=True)
            if not options or options[0][0] <= 0:
                raise RuntimeError(f"Could not identify {kind} vector. Available: {list(vectors)}")
            return options[0][1]

        wave_name = choose("wave")
        wave = vectors[wave_name]
        flux_name = choose("flux", wave.size)
        flux = vectors[flux_name]
        try:
            err_name = choose("err", wave.size)
            err = vectors[err_name]
        except Exception:
            err_name = ""
            err = np.full_like(flux, np.nan)

        unit = ""
        try:
            unit = hdu.columns[flux_name].unit or ""
        except Exception:
            pass
        # Normalize wavelength to microns using value scale.
        finite = wave[np.isfinite(wave)]
        med = float(np.nanmedian(finite))
        if med > 1000:
            wave_um = wave / 1e4      # Angstrom -> micron
        elif med > 100:
            wave_um = wave / 1000.0   # nm -> micron
        else:
            wave_um = wave
        info = {"wave_column": wave_name, "flux_column": flux_name, "error_column": err_name, "flux_unit": unit}
        print("DAWN SPECTRUM COLUMNS:", info)
        return wave_um, flux, err, unit, info


def load_s2d(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(path, memmap=False) as hdul:
        sci = np.asarray(hdul["SCI"].data, float)
        err = np.asarray(hdul["ERR"].data, float)
        wav = np.asarray(hdul["WAVELENGTH"].data, float)
    sn = np.divide(sci, err, out=np.full_like(sci, np.nan), where=np.isfinite(err) & (err > 0))
    wave_1d = np.nanmedian(wav, axis=0)
    return wave_1d, sci, sn


def robust_limits(a: np.ndarray, lo: float = 2, hi: float = 98) -> tuple[float, float]:
    f = a[np.isfinite(a)]
    if f.size == 0:
        return -1, 1
    return tuple(np.nanpercentile(f, [lo, hi]))


def get_hips_candidates(filter_name: str) -> list[str]:
    # Query CDS MOCServer for HiPS surveys mentioning JWST/COSMOS and the requested filter.
    expr = f"ID=*JWST* & ID=*{filter_name}*"
    url = "https://alasky.cds.unistra.fr/MocServer/query"
    try:
        r = requests.get(url, params={"expr": expr, "get": "record", "fmt": "json"}, timeout=30)
        r.raise_for_status()
        obj = r.json()
        records = obj if isinstance(obj, list) else obj.get("records", [])
        ids = []
        for rec in records:
            sid = rec.get("ID") or rec.get("hips_service_url") or rec.get("obs_title")
            text = json.dumps(rec).lower()
            if sid and filter_name.lower() in text and ("cosmos" in text or "jwst" in text):
                ids.append(str(sid))
        return ids
    except Exception:
        return []


def hips_cutout(filter_name: str, npix: int = 128, fov_arcsec: float = 3.0) -> np.ndarray | None:
    ids = get_hips_candidates(filter_name)
    for hips in ids[:8]:
        try:
            params = {
                "hips": hips,
                "width": npix,
                "height": npix,
                "fov": fov_arcsec / 3600.0,
                "projection": "TAN",
                "coordsys": "icrs",
                "ra": RA_DEG,
                "dec": DEC_DEG,
                "format": "fits",
            }
            r = requests.get("https://alasky.cds.unistra.fr/hips-image-services/hips2fits", params=params, timeout=90)
            if r.ok and len(r.content) > 1000:
                tmp = DATA_DIR / f"{VERSION}_{filter_name}_hips.fits"
                tmp.write_bytes(r.content)
                with fits.open(tmp, memmap=False) as h:
                    arr = np.squeeze(np.asarray(h[0].data, float))
                if arr.ndim == 2 and np.isfinite(arr).sum() > 50:
                    print(f"NIRCAM CUTOUT: {filter_name} from {hips}")
                    return arr
        except Exception:
            continue
    return None


def draw_stamp(ax: plt.Axes, arr: np.ndarray | None, label: str) -> None:
    ax.set_title(label, fontsize=13, pad=3)
    ax.set_xticks([]); ax.set_yticks([])
    if arr is None:
        ax.set_facecolor("#d7e8f5")
        ax.text(0.5, 0.5, "cutout\nunavailable", ha="center", va="center", fontsize=8, transform=ax.transAxes)
        return
    finite = arr[np.isfinite(arr)]
    if finite.size:
        med = np.nanmedian(finite)
        sig = 1.4826 * np.nanmedian(np.abs(finite - med))
        scaled = np.arcsinh((arr - med) / max(sig * 2.5, 1e-30))
        vmin, vmax = robust_limits(scaled, 2, 99.7)
        ax.imshow(scaled, origin="lower", cmap="cool", vmin=vmin, vmax=vmax, interpolation="nearest")


# 1) DAWN merged spectrum
prism_path = zenodo_file(ZENODO_PRISM_NAME)
hdu_idx, row_idx, source_meta = locate_source_row(prism_path)
wave_um, flux, err, flux_unit, col_info = extract_spectrum(prism_path, hdu_idx, row_idx)
mask = np.isfinite(wave_um) & np.isfinite(flux)
wave_um, flux, err = wave_um[mask], flux[mask], err[mask]
order = np.argsort(wave_um)
wave_um, flux, err = wave_um[order], flux[order], err[order]

# 2) Standard 2-D source product for the upper strip
s2d_path = mast_download(MAST_S2D)
s2d_wave, s2d_sci, s2d_sn = load_s2d(s2d_path)

# 3) Try public per-filter HiPS cutouts
filters = ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
stamps = {f: hips_cutout(f) for f in filters}

# 4) Save exact extracted arrays
spec_df = pd.DataFrame({"wavelength_um": wave_um, "wavelength_nm": wave_um * 1000.0, "flux": flux, "flux_error": err})
spec_csv = CSV_DIR / f"{VERSION}_MOM_Z14_DJA_ROUTE_A_SPECTRUM.csv"
spec_df.to_csv(spec_csv, index=False)
meta_out = {
    "version": VERSION,
    "source": SOURCE_KEY,
    "ra_deg": RA_DEG,
    "dec_deg": DEC_DEG,
    "z_spec": Z_SPEC,
    "zenodo_record": ZENODO_RECORD,
    "zenodo_file": ZENODO_PRISM_NAME,
    "matched_hdu": hdu_idx,
    "matched_row": row_idx,
    "columns": col_info,
    "source_metadata": {k: str(v) for k, v in source_meta.items()},
}
meta_json = CSV_DIR / f"{VERSION}_MOM_Z14_ROUTE_A_METADATA.json"
meta_json.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")

# 5) Paper-style composite
plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8, "savefig.bbox": "tight"})
fig = plt.figure(figsize=(15.5, 10.5), constrained_layout=False)
gs = GridSpec(4, 7, figure=fig, height_ratios=[1.1, 0.9, 0.08, 3.2], hspace=0.08, wspace=0.025)
for j, f in enumerate(filters):
    draw_stamp(fig.add_subplot(gs[0, j]), stamps[f], f)

ax2 = fig.add_subplot(gs[1, :])
lo2, hi2 = robust_limits(s2d_sn, 3, 99.5)
extent = [float(np.nanmin(s2d_wave)), float(np.nanmax(s2d_wave)), 0, s2d_sn.shape[0]]
ax2.imshow(s2d_sn, origin="lower", aspect="auto", extent=extent, cmap="cool", vmin=lo2, vmax=hi2, interpolation="nearest")
ax2.set_xlim(0.8, 5.3)
ax2.set_xticks([]); ax2.set_yticks([])
ax2.set_ylabel("2-D S/N", fontsize=10)

ax = fig.add_subplot(gs[3, :])
# Use a step plot to show the native bins exactly as in published-style spectra.
ax.step(wave_um, flux, where="mid", linewidth=1.15, color="navy", label="DAWN msaexp v4.4")
if np.isfinite(err).any():
    ax.fill_between(wave_um, flux - err, flux + err, step="mid", alpha=0.28, color="skyblue", linewidth=0, label="1σ uncertainty")
ax.axhline(0, color="0.6", linewidth=0.6)
ax.set_xlim(0.8, 5.3)
finite_flux = flux[np.isfinite(flux)]
ylo, yhi = np.nanpercentile(finite_flux, [1, 99])
pad = 0.18 * max(yhi - ylo, 1e-30)
ax.set_ylim(ylo - pad, yhi + pad)
ax.set_xlabel(r"Observed wavelength $\lambda_{\rm obs}$ [$\mu$m]", fontsize=13)
ylabel = "Flux density"
if flux_unit:
    ylabel += f" [{flux_unit}]"
ax.set_ylabel(ylabel, fontsize=12)
ax.grid(axis="x", linewidth=0.35, alpha=0.25)

lines_nm = {
    "Lyα": 121.567,
    "N IV]": 148.7,
    "C IV": 154.95,
    "He II": 164.0,
    "O III]": 166.35,
    "N III]": 175.0,
    "C III]": 190.8,
}
ytop = ax.get_ylim()[1]
for name, rest_nm in lines_nm.items():
    obs_um = rest_nm * (1 + Z_SPEC) / 1000.0
    ax.axvline(obs_um, color="0.55", linestyle="--", linewidth=0.7, alpha=0.65)
    ax.text(obs_um, ytop - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]), name, rotation=90,
            ha="right", va="top", fontsize=9, color="0.25")

ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.suptitle("MoM-z14 — Route A DAWN JWST Archive spectrum and public JWST cutouts", fontsize=16, y=0.995)
fig.text(0.5, 0.008,
         "Spectrum: DAWN JWST Archive msaexp v4.4 merged PRISM release (source 5224_277193). "
         "2-D strip: STScI s2d source product. Cutouts: public CDS HiPS services when available.",
         ha="center", va="bottom", fontsize=8)

png_path = PNG_DIR / f"{VERSION}_MOM_Z14_ROUTE_A_PAPER_STYLE.png"
fig.savefig(png_path, dpi=350)
plt.show()
plt.close(fig)

print("\nROUTE A RESULT")
print(f"  SOURCE                         {SOURCE_KEY}")
print(f"  DAWN MATCH                     HDU {hdu_idx}, row {row_idx}")
print(f"  NATIVE SPECTRAL SAMPLES        {len(wave_um)}")
print(f"  WAVELENGTH RANGE [um]          {np.nanmin(wave_um):.6f} to {np.nanmax(wave_um):.6f}")
print(f"  FILTER CUTOUTS RETRIEVED       {sum(v is not None for v in stamps.values())}/{len(filters)}")
print("\nOUTPUT SUMMARY")
print(f"  Paper-style PNG:               {png_path}")
print(f"  Route-A spectrum CSV:          {spec_csv}")
print(f"  Metadata JSON:                 {meta_json}")
print(f"  DAWN merged FITS:              {prism_path}")
print(f"  2-D source FITS:               {s2d_path}")
print(f"  Timestamp UTC:                 {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
