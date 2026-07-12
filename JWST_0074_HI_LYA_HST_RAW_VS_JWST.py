# JWST_0074
# Real HST/COS H I Ly-alpha profile versus real MoM-z14 JWST/PRISM samples.
# No AI images, Gaussian profiles, smoothing, or raster digitization.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0074"
GALAXY = "MoM-z14"
Z_MOM = 14.44
Z_REF = 0.020598
LYA_NM = 121.56701
C_NM_THz = 299792.458
ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
OUT = ROOT / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION
HELPER = ROOT / "JWST_0060_MOMZ14_FAST_CONE_CLASSY.py"

BG, AX, GRID = "#050712", "#07101f", "#1f6f8b"
TEXT, MUTED = "#e6f4ff", "#8fb3c7"
LAB, OBS, BLUE, RED, POINT = "#ffd84d", "#ff9d2e", "#6ee7ff", "#ff5a66", "#d9edf7"


def need(name, pip_name=None):
    try:
        __import__(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or name])


def nu(nm):
    return C_NM_THz / np.asarray(nm, float)


def lam(thz):
    return C_NM_THz / np.asarray(thz, float)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_jwst():
    hits = sorted(CSV.glob("JWST_*_MoM-z14_EXACT_JWST.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if hits:
        return hits[0], "coordinate-matched cached JWST X1D", None
    subprocess.run([
        "curl", "-fsSL", "-o", str(HELPER),
        "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0060_MOMZ14_FAST_CONE_CLASSY.py"
    ], check=True)
    mast = load_module("m0060", HELPER)
    mast.VERSION, mast.OUT, mast.PNG, mast.CSV, mast.DATA = VERSION, OUT, PNG, CSV, DATA / "JWST_MAST"
    mast.MAX_JWST_X1D = 24
    base = mast.load_base(mast.ensure_base())
    _, meta = mast.exact_momz14_cone(base)
    return Path(meta["exact_csv"]), f"coordinate-verified GO-5224 X1D; separation={meta['sep']:.6f} arcsec", meta


def load_jwst(path):
    d = pd.read_csv(path)
    lookup = {str(c).lower(): c for c in d.columns}
    wc = next((lookup[k] for k in ["wavelength_um", "wavelength_nm", "wavelength", "wave"] if k in lookup), None)
    fc = next((lookup[k] for k in ["flux", "raw_flux", "jwst_flux"] if k in lookup), None)
    ec = next((lookup[k] for k in ["flux_error", "error", "err"] if k in lookup), None)
    if wc is None or fc is None:
        raise RuntimeError(f"Missing wavelength/flux columns in {path.name}")
    w = pd.to_numeric(d[wc], errors="coerce").to_numpy(float)
    y = pd.to_numeric(d[fc], errors="coerce").to_numpy(float)
    e = pd.to_numeric(d[ec], errors="coerce").to_numpy(float) if ec else np.full_like(y, np.nan)
    good = np.isfinite(w) & np.isfinite(y) & (w > 0)
    w, y, e = w[good], y[good], e[good]
    med, name = float(np.nanmedian(w)), str(wc).lower()
    wnm = w * 1000 if "_um" in name or med < 20 else (w if "_nm" in name or med < 10000 else w / 10)
    order = np.argsort(wnm)
    return wnm[order], y[order], e[order], str(wc), str(fc)


def read_x1d(path):
    from astropy.io import fits
    parts = []
    with fits.open(path, memmap=False) as hdul:
        for hdu_i, hdu in enumerate(hdul):
            if getattr(hdu, "data", None) is None or getattr(hdu, "columns", None) is None:
                continue
            names = {str(n).upper(): n for n in hdu.columns.names}
            if "WAVELENGTH" not in names or "FLUX" not in names:
                continue
            w = np.asarray(hdu.data[names["WAVELENGTH"]], float).ravel()
            f = np.asarray(hdu.data[names["FLUX"]], float).ravel()
            en = next((names[k] for k in ["ERROR", "ERR", "FLUX_ERROR"] if k in names), None)
            e = np.asarray(hdu.data[en], float).ravel() if en else np.full_like(f, np.nan)
            n = min(len(w), len(f), len(e))
            good = np.isfinite(w[:n]) & np.isfinite(f[:n]) & (w[:n] > 0)
            if good.any():
                parts.append(pd.DataFrame({"wave_A": w[:n][good], "flux": f[:n][good], "error": e[:n][good], "hdu": hdu_i}))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).sort_values("wave_A")


def fetch_haro11():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astroquery.mast import Observations

    cache = CSV / f"{VERSION}_HARO11_HST_COS_RAW_LYA.csv"
    audit = CSV / f"{VERSION}_HARO11_PRODUCT_AUDIT.csv"
    if cache.exists() and cache.stat().st_size > 1000:
        return pd.read_csv(cache), cache, audit, "cached HST/COS X1D"

    coord = SkyCoord(9.21875 * u.deg, -33.55472 * u.deg)
    obs = Observations.query_region(coord, radius=0.04 * u.deg)
    odf = obs.to_pandas()
    keep = odf["obs_collection"].astype(str).str.upper().eq("HST")
    keep &= odf["instrument_name"].astype(str).str.upper().str.contains("COS")
    obs = obs[np.asarray(keep)]
    if len(obs) == 0:
        raise RuntimeError("No HST/COS observations found for Haro 11")

    products = Observations.get_product_list(obs)
    pdf = products.to_pandas()
    subgroup = pdf.get("productSubGroupDescription", pd.Series("", index=pdf.index)).astype(str).str.upper()
    filename = pdf.get("productFilename", pd.Series("", index=pdf.index)).astype(str).str.lower()
    products = products[np.asarray(subgroup.eq("X1D") | filename.str.endswith("_x1d.fits"))]
    if len(products) == 0:
        raise RuntimeError("No Haro 11 COS X1D products found")

    DATA.mkdir(parents=True, exist_ok=True)
    rows, best = [], None
    for i, product in enumerate(products[:36], 1):
        name, uri = str(product["productFilename"]), str(product["dataURI"])
        local = DATA / name
        try:
            if not local.exists() or local.stat().st_size < 1000:
                Observations.download_file(uri, local_path=str(local), cache=True)
            d = read_x1d(local)
            if d is None:
                rows.append((i, name, "NO_TABLE", 0, np.nan)); continue
            rest = d["wave_A"].to_numpy(float) / (1 + Z_REF) / 10
            flux = d["flux"].to_numpy(float)
            window = (rest >= 118.8) & (rest <= 124.3) & np.isfinite(flux)
            count = int(window.sum())
            if count < 30:
                rows.append((i, name, "TOO_FEW", count, np.nan)); continue
            yy = flux[window]
            med = float(np.nanmedian(yy)); mad = float(np.nanmedian(np.abs(yy - med))) * 1.4826
            mad = mad if np.isfinite(mad) and mad > 0 else max(float(np.nanstd(yy)), 1e-30)
            core = np.abs(rest[window] - LYA_NM) <= 1.2
            contrast = (float(np.nanmax(yy[core])) - med) / mad if core.any() else 0.0
            score = count + 250 * max(contrast, 0)
            rows.append((i, name, "USABLE", count, score))
            if best is None or score > best[0]:
                best = (score, name, d, rest, window)
        except Exception as exc:
            rows.append((i, name, f"{type(exc).__name__}: {exc}", 0, np.nan))

    pd.DataFrame(rows, columns=["index", "filename", "status", "samples", "score"]).to_csv(audit, index=False)
    if best is None:
        raise RuntimeError(f"No usable Haro 11 COS X1D product; audit={audit}")

    _, name, d, rest, window = best
    out = d.loc[window].copy()
    out["rest_wavelength_nm"] = rest[window]
    out["rest_frequency_THz"] = nu(out["rest_wavelength_nm"])
    out["source_product"] = name
    out = out.sort_values("rest_wavelength_nm")
    out.to_csv(cache, index=False)
    return out, cache, audit, f"HST/COS X1D {name}"


def pick_ref_peak(d):
    x, y = d["rest_wavelength_nm"].to_numpy(float), d["flux"].to_numpy(float)
    core = np.abs(x - LYA_NM) <= 1.2
    ids = np.flatnonzero(core if core.any() else np.isfinite(x) & np.isfinite(y))
    i = int(ids[np.nanargmax(y[ids])])
    return float(x[i]), float(y[i])


def pick_break(x, y):
    dx, dy = np.diff(x), np.diff(y)
    mid = 0.5 * (x[:-1] + x[1:])
    expected = LYA_NM * (1 + Z_MOM)
    ids = np.where((dx > 0) & np.isfinite(dy) & (np.abs(mid - expected) <= 90))[0]
    positive = ids[dy[ids] > 0]
    if len(positive): ids = positive
    if len(ids) == 0: ids = np.arange(len(dy))
    i = int(ids[np.nanargmax(dy[ids])])
    return float(mid[i]), float(0.5 * dx[i]), float(0.5 * (y[i] + y[i + 1]))


def style(a):
    a.set_facecolor(AX); a.grid(True, color=GRID, lw=.48, alpha=.45); a.tick_params(colors=TEXT, labelsize=8.5)
    a.xaxis.label.set_color(TEXT); a.yaxis.label.set_color(TEXT); a.title.set_color(TEXT)
    for s in a.spines.values(): s.set_color("#4fa8c8"); s.set_linewidth(.8)


def limits(y):
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y)); p = .06 * (hi - lo) if hi > lo else 1e-30
    return lo - p, hi + p


def plot(ref, ref_status, jx, jy, je, jpath, jstatus):
    rx, ry, re = ref["rest_wavelength_nm"].to_numpy(float), ref["flux"].to_numpy(float), ref["error"].to_numpy(float)
    peak_nm, peak_flux = pick_ref_peak(ref)
    break_nm, break_unc, break_flux = pick_break(jx, jy)
    rf, jf = nu(rx), nu(jx); ro, jo = np.argsort(rf), np.argsort(jf)

    fig, (l, r) = plt.subplots(1, 2, figsize=(15.9, 6.6), facecolor=BG); style(l); style(r)
    l.step(rf[ro], ry[ro], where="mid", color=LAB, lw=.62); l.scatter(rf[ro], ry[ro], s=4.5, color=LAB, alpha=.45)
    l.axvline(float(nu(peak_nm)), color=BLUE, ls=(0, (3, 5)), lw=.55, alpha=.58)
    r.step(jf[jo], jy[jo], where="mid", color=OBS, lw=.70); r.scatter(jf[jo], jy[jo], s=32, color=POINT, edgecolor=BG, lw=.32)
    r.axvline(float(nu(break_nm)), color=RED, ls=(0, (3, 5)), lw=.55, alpha=.58)
    l.set(xlim=(rf.min(), rf.max()), ylim=limits(ry), title="REAL HST/COS H I Lyα PROFILE — REST FRAME", xlabel="Rest frequency, THz", ylabel="HST/COS X1D flux")
    r.set(xlim=(jf.min(), jf.max()), ylim=limits(jy), title="REAL MoM-z14 JWST/PRISM — Lyα BREAK", xlabel="Observed frequency, THz", ylabel="JWST/NIRSpec X1D flux")
    lt = l.secondary_xaxis("top", functions=(lam, nu)); lt.set_xlabel("Rest wavelength, nm", color=TEXT); lt.tick_params(colors=TEXT, labelsize=8)
    rt = r.secondary_xaxis("top", functions=(lam, nu)); rt.set_xlabel("Observed wavelength, nm", color=TEXT); rt.tick_params(colors=TEXT, labelsize=8)
    l.text(.018, .045, f"all numerical X1D samples = {len(rx)}\nmeasured Lyα peak = {peak_nm:.6f} nm\nreference = Haro 11, z={Z_REF:.6f}", transform=l.transAxes, color=TEXT, fontsize=7.3, bbox=dict(boxstyle="round", fc=AX, ec=LAB, alpha=.94))
    r.text(.018, .045, f"all PRISM bins = {len(jx)}\nsample-bracketed break = {break_nm:.3f} ± {break_unc:.3f} nm\npublished-z Lyα = {LYA_NM*(1+Z_MOM):.3f} nm", transform=r.transAxes, color=TEXT, fontsize=7.3, bbox=dict(boxstyle="round", fc=AX, ec=RED, alpha=.94))
    fig.suptitle(f"{VERSION} — H I LYMAN-α: NUMERICAL REST PROFILE versus MoM-z14", color=TEXT, fontsize=14.8, fontweight="bold", y=.982)
    fig.text(.5, .914, "Independent numerical spectra. Left is an empirical HST/COS Lyα profile transformed to rest frame; right is every available MoM-z14 PRISM bin. No smoothing or Gaussian model.", ha="center", color=MUTED, fontsize=8.3)
    fig.text(.5, .017, f"Reference: {ref_status} | JWST: {jpath.name} ({jstatus})", ha="center", color=MUTED, fontsize=7.3)
    fig.subplots_adjust(left=.075, right=.985, top=.825, bottom=.12, wspace=.16)
    pp = PNG / f"{VERSION}_{GALAXY}_HI_LYA_HST_COS_VS_JWST.png"; fig.savefig(pp, dpi=245, facecolor=BG); plt.show(); plt.close(fig)
    rc = CSV / f"{VERSION}_HARO11_HI_LYA_REST_RAW.csv"; oc = CSV / f"{VERSION}_{GALAXY}_HI_LYA_RAW_PRISM.csv"
    pd.DataFrame({"rest_wavelength_nm": rx, "rest_frequency_THz": nu(rx), "hst_cos_flux": ry, "hst_cos_error": re}).to_csv(rc, index=False)
    pd.DataFrame({"observed_wavelength_nm": jx, "observed_frequency_THz": nu(jx), "jwst_flux": jy, "jwst_error": je}).to_csv(oc, index=False)
    return pp, rc, oc, peak_nm, break_nm, break_unc


def main():
    for n in ["astropy", "astroquery"]: need(n)
    PNG.mkdir(parents=True, exist_ok=True); CSV.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    print(f"CODE OUTPUT: {VERSION}")
    print("STEP 1/4 | Load coordinate-verified MoM-z14 spectrum")
    jp, js, meta = locate_jwst(); w, y, e, wc, fc = load_jwst(jp)
    m = (w >= 1760) & (w <= 1995); jx, jy, je = w[m], y[m], e[m]
    if len(jx) < 6: raise RuntimeError(f"Only {len(jx)} JWST bins in Ly-alpha window")
    print("STEP 2/4 | Query MAST for real Haro 11 HST/COS X1D data")
    ref, ref_csv, product_audit, ref_status = fetch_haro11()
    print("STEP 3/4 | Preserve every sample and transform only the HST wavelength axis")
    print("STEP 4/4 | Plot H I Ly-alpha profile beside MoM-z14 break")
    pp, rc, oc, peak, brk, unc = plot(ref, ref_status, jx, jy, je, jp, js)
    audit = CSV / f"{VERSION}_SOURCE_AUDIT.csv"
    pd.DataFrame([{"reference_type": "empirical HST/COS Ly-alpha profile, not laboratory", "reference_object": "Haro 11", "reference_z": Z_REF, "reference_samples": len(ref), "reference_peak_nm": peak, "jwst_source": str(jp), "jwst_bins": len(jx), "jwst_break_nm": brk, "jwst_break_half_spacing_nm": unc}]).to_csv(audit, index=False)
    print(f"REFERENCE SAMPLES    : {len(ref)}"); print(f"REFERENCE PEAK       : {peak:.6f} nm"); print(f"JWST RAW BINS        : {len(jx)}"); print(f"JWST BREAK           : {brk:.6f} ± {unc:.6f} nm")
    print(f"PLOT PNG             : {pp}"); print(f"REFERENCE CSV        : {rc}"); print(f"JWST CSV             : {oc}"); print(f"PRODUCT AUDIT CSV    : {product_audit}"); print(f"SOURCE AUDIT CSV     : {audit}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")); print(f"# {VERSION}")


if __name__ == "__main__": main()
