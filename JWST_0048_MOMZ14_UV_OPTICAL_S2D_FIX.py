# JWST_0048
# MoM-z14 real-data workflow with S2D fallback fix.
# Real MAST data only for spectra. No AI images. No synthetic spectra. No smoothing. No continuum subtraction.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib, warnings

VERSION = "JWST_0048"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

TARGET = "MoM-z14"
Z_REF = 14.44
PROGRAM_ID = "5224"

UV_LINES = [
    (1, "N IV] 1486", 0.148600, "UV nitrogen"),
    (2, "C IV 1549", 0.154900, "UV carbon blend"),
    (3, "He II 1640", 0.164000, "UV helium"),
    (4, "O III] 1666", 0.166600, "UV oxygen"),
    (5, "C III] 1908", 0.190800, "UV carbon blend"),
]

OPTICAL_LINES = [
    (6, "[N II] 6548", 0.654805, "optical nitrogen"),
    (7, "H-alpha 6563", 0.656281, "optical hydrogen"),
    (8, "[N II] 6583", 0.658345, "optical nitrogen"),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "astroquery", "ipywidgets"]:
        need(pkg)
    for p in [PNG, CSV, DATA]:
        p.mkdir(parents=True, exist_ok=True)


def query_products():
    import pandas as pd
    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection="JWST", proposal_id=PROGRAM_ID)
    obs_df = obs.to_pandas() if hasattr(obs, "to_pandas") else pd.DataFrame(obs)
    obs_path = CSV / f"{VERSION}_PROGRAM_{PROGRAM_ID}_OBSERVATIONS.csv"
    obs_df.to_csv(obs_path, index=False)
    if len(obs) == 0:
        return [], obs_path, None
    prod = Observations.get_product_list(obs)
    prod_df = prod.to_pandas() if hasattr(prod, "to_pandas") else pd.DataFrame(prod)
    prod_path = CSV / f"{VERSION}_PROGRAM_{PROGRAM_ID}_PRODUCTS.csv"
    prod_df.to_csv(prod_path, index=False)
    if prod_df.empty:
        return [], obs_path, prod_path
    fn_col = "productFilename" if "productFilename" in prod_df.columns else None
    sub_col = "productSubGroupDescription" if "productSubGroupDescription" in prod_df.columns else None
    if fn_col is None:
        candidates = prod_df.copy()
    else:
        name = prod_df[fn_col].astype(str).str.lower()
        mask_x1d = name.str.contains("x1d") & name.str.endswith(".fits")
        mask_s2d = name.str.contains("s2d") & name.str.endswith(".fits")
        mask_c1d = name.str.contains("c1d") & name.str.endswith(".fits")
        if sub_col:
            sub = prod_df[sub_col].astype(str).str.upper()
            mask_x1d |= sub.eq("X1D")
            mask_s2d |= sub.eq("S2D")
            mask_c1d |= sub.eq("C1D")
        candidates = prod_df[mask_x1d | mask_c1d | mask_s2d].copy()
        def rank_filename(x):
            x = str(x).lower()
            if "x1d" in x: return 0
            if "c1d" in x: return 1
            if "s2d" in x: return 2
            return 9
        candidates["_priority"] = candidates[fn_col].map(rank_filename)
        candidates = candidates.sort_values("_priority")
    cand_path = CSV / f"{VERSION}_PROGRAM_{PROGRAM_ID}_SPECTRUM_CANDIDATES.csv"
    candidates.to_csv(cand_path, index=False)
    return candidates.to_dict("records"), obs_path, cand_path


def download_product(record):
    from astroquery.mast import Observations
    uri = record.get("dataURI") or record.get("dataUri") or record.get("uri")
    fname = record.get("productFilename") or record.get("productfilename") or "mast_product.fits"
    local = DATA / str(fname)
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    if not uri:
        raise RuntimeError("Product record has no dataURI.")
    Observations.download_file(uri, local_path=str(local))
    if not local.exists() or local.stat().st_size < 100000:
        raise RuntimeError(f"Downloaded file missing or too small: {fname}")
    return local, "downloaded-from-mast"


def finite_sorted(wave, flux):
    import numpy as np
    wave = np.ravel(np.asarray(wave, dtype=float))
    flux = np.ravel(np.asarray(flux, dtype=float))
    m = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[m], flux[m]
    o = np.argsort(wave)
    return wave[o], flux[o]


def unit_to_micron(v):
    import numpy as np
    arr = np.asarray(v, dtype=float)
    med = np.nanmedian(arr)
    if 0.1 < med < 50:
        return arr
    if 100 < med < 500000:
        return arr / 10000.0
    if 1e-7 < med < 5e-5:
        return arr * 1e6
    return arr


def read_table_spectrum(hdul):
    import numpy as np
    waves, fluxes = [], []
    unit = "native"
    used = []
    for i, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        names = list(getattr(data, "names", []) or []) if data is not None else []
        if "WAVELENGTH" in names and "FLUX" in names:
            unit = str(hdu.header.get("TUNIT2", unit))
            used.append(f"{i}:{getattr(hdu, 'name', '')}:TABLE")
            wcol = data["WAVELENGTH"]
            fcol = data["FLUX"]
            try:
                for w, f in zip(wcol, fcol):
                    waves.append(np.ravel(np.asarray(w, dtype=float)))
                    fluxes.append(np.ravel(np.asarray(f, dtype=float)))
            except TypeError:
                waves.append(np.ravel(np.asarray(wcol, dtype=float)))
                fluxes.append(np.ravel(np.asarray(fcol, dtype=float)))
    if not waves:
        return None
    wave = unit_to_micron(np.concatenate(waves))
    flux = np.concatenate(fluxes)
    wave, flux = finite_sorted(wave, flux)
    return wave, flux, unit, "; ".join(used), "TABLE_X1D_OR_C1D"


def wave_from_wcs(header, shape, row):
    import numpy as np
    from astropy.wcs import WCS
    wcs = WCS(header)
    nx = shape[1]
    x = np.arange(nx, dtype=float)
    y = np.full(nx, float(row))
    vals = wcs.all_pix2world(x, y, 0)
    if not isinstance(vals, (tuple, list)):
        vals = [vals]
    candidates = []
    for arr in vals:
        a = np.asarray(arr, dtype=float)
        am = unit_to_micron(a)
        finite = np.isfinite(am)
        if finite.sum() > max(10, nx // 5):
            lo, hi = np.nanmin(am[finite]), np.nanmax(am[finite])
            if 0.2 <= lo <= 30 and 0.2 <= hi <= 30 and abs(hi - lo) > 0.01:
                candidates.append(am)
    if not candidates:
        return None
    candidates.sort(key=lambda a: abs((np.nanmax(a) - np.nanmin(a))), reverse=True)
    return candidates[0]


def read_s2d_spectrum(hdul):
    import numpy as np
    for i, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is None:
            continue
        arr = np.asarray(data, dtype=float)
        if arr.ndim != 2:
            continue
        name = str(getattr(hdu, "name", "")).upper()
        if name not in ["SCI", "FLUX", "EXTRACT1D"] and i != 1:
            continue
        finite = np.isfinite(arr)
        if finite.sum() < 100:
            continue
        # Choose a single raw detector/WCS row. No smoothing, no continuum subtraction, no row summing.
        row_scores = np.nanpercentile(np.where(finite, arr, np.nan), 95, axis=1)
        if not np.isfinite(row_scores).any():
            row_scores = np.nanmean(np.abs(np.where(finite, arr, np.nan)), axis=1)
        row = int(np.nanargmax(row_scores))
        wave = wave_from_wcs(hdu.header, arr.shape, row)
        if wave is None:
            continue
        flux = arr[row, :]
        wave, flux = finite_sorted(wave, flux)
        unit = str(hdu.header.get("BUNIT", "native"))
        used = f"{i}:{name}:S2D_SINGLE_RAW_ROW_{row}"
        return wave, flux, unit, used, "S2D_SINGLE_RAW_ROW_FALLBACK"
    return None


def read_any_spectrum(path):
    from astropy.io import fits
    with fits.open(path) as hdul:
        table = read_table_spectrum(hdul)
        if table is not None and len(table[0]) > 20:
            return table
        s2d = read_s2d_spectrum(hdul)
        if s2d is not None and len(s2d[0]) > 20:
            return s2d
    raise RuntimeError(f"No readable WAVELENGTH/FLUX table or WCS-calibrated S2D SCI row found in {path.name}")


def choose_best(records):
    uv_expected = [rest * (1 + Z_REF) for _, _, rest, _ in UV_LINES]
    attempts = []
    for rec in records[:40]:
        try:
            path, status = download_product(rec)
            wave, flux, unit, used, mode = read_any_spectrum(path)
            cover = sum(min(wave) <= x <= max(wave) for x in uv_expected)
            priority = 0 if mode.startswith("TABLE") else 1
            attempts.append((cover, -priority, len(wave), path, status, wave, flux, unit, used, mode, rec))
        except Exception as exc:
            warnings.warn(str(exc))
    if not attempts:
        raise RuntimeError("No readable spectrum product found. Candidate inventory CSV was saved.")
    attempts.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return attempts[0]


def peak_sample(wave, flux, center, half=0.060):
    import numpy as np
    m = (wave >= center - half) & (wave <= center + half)
    if m.sum() == 0:
        return float("nan"), float("nan"), 0
    ww, ff = wave[m], flux[m]
    idx = int(np.nanargmax(ff))
    return float(ww[idx]), float(ff[idx]), int(m.sum())


def line_tables(wave, flux):
    import pandas as pd
    rows = []
    for n, label, rest, species in UV_LINES:
        exp = rest * (1 + Z_REF)
        pw, pf, cnt = peak_sample(wave, flux, exp)
        rows.append({"n": n, "line": label, "species": species, "group": "UV", "rest_um": rest, "expected_z14p44_um": exp, "raw_peak_sample_um": pw, "raw_peak_flux_native": pf, "sample_count": cnt, "slope_m": pw/rest if pw==pw else float("nan"), "z_from_raw_peak": pw/rest-1 if pw==pw else float("nan"), "status": "measured from real spectrum window" if cnt else "outside downloaded wavelength range"})
    uv = pd.DataFrame(rows)
    rows = []
    for n, label, rest, species in OPTICAL_LINES:
        exp = rest * (1 + Z_REF)
        if min(wave) <= exp <= max(wave):
            pw, pf, cnt = peak_sample(wave, flux, exp, 0.080)
            status = "measured from real spectrum window" if cnt else "inside range but no samples"
        else:
            pw, pf, cnt, status = float("nan"), float("nan"), 0, "out of downloaded wavelength range; not faked"
        rows.append({"n": n, "line": label, "species": species, "group": "H-alpha/[N II] optical", "rest_um": rest, "expected_z14p44_um": exp, "raw_peak_sample_um": pw, "raw_peak_flux_native": pf, "sample_count": cnt, "slope_m": pw/rest if pw==pw else float("nan"), "z_from_raw_peak": pw/rest-1 if pw==pw else float("nan"), "status": status})
    opt = pd.DataFrame(rows)
    return uv, opt


def freq_thz_from_um(x): return 299.792458 / x

def um_from_freq_thz(x): return 299.792458 / x


def style_dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.45, alpha=0.65)
    ax.tick_params(colors="#dbeafe", labelsize=8.5)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for sp in ax.spines.values():
        sp.set_color("#94a3b8")
        sp.set_linewidth(0.65)


def style_light(ax):
    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linewidth=0.45)
    ax.tick_params(colors="0.10", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("0.25")
        sp.set_linewidth(0.65)


def legend_dark(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=7.5, facecolor="#020617", edgecolor="#475569")
    for txt in leg.get_texts(): txt.set_color("#f8fafc")


def plot_combined_uv(wave, flux, unit, uv, avg_obs):
    import matplotlib.pyplot as plt
    valid = uv[uv["sample_count"] > 0]
    if valid.empty: return None
    xmin = max(float(min(wave)), float(valid["expected_z14p44_um"].min() - 0.16))
    xmax = min(float(max(wave)), float(valid["expected_z14p44_um"].max() + 0.16))
    m = (wave >= xmin) & (wave <= xmax)
    fig, ax = plt.subplots(figsize=(16.5, 8.0))
    style_dark(ax)
    ax.plot(wave[m], flux[m], color="#e0f2fe", linewidth=0.62, label="real spectrum samples")
    for row in valid.itertuples():
        ax.axvline(row.expected_z14p44_um, color="#64748b", linestyle=":", linewidth=0.75)
        ax.axvline(row.raw_peak_sample_um, color="#f8fafc", linestyle="--", linewidth=0.70)
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux_native], s=34, color="#f8fafc", edgecolor="#020617", zorder=5)
        ax.text(row.raw_peak_sample_um, row.raw_peak_flux_native, f" {int(row.n)}", color="#f8fafc", fontsize=9.5, fontweight="bold", va="bottom")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.35, label=f"UV average raw peak = {avg_obs:.6f} µm")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz"); top.xaxis.label.set_color("#f8fafc"); top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, native unit: {unit}")
    ax.set_title(f"{VERSION} — {TARGET}: combined real UV spectrum windows at z≈14.44")
    legend_dark(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMBINED_REAL_UV_LINES.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_optical_audit(opt, wave):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    style_light(ax)
    y = list(range(len(opt)))
    ax.axvspan(float(min(wave)), float(max(wave)), color="#e0f2fe", alpha=0.55, label="downloaded real spectrum coverage")
    ax.scatter(opt["expected_z14p44_um"], y, s=45, color="black", zorder=5, label="predicted λ at z=14.44")
    for yi, row in zip(y, opt.itertuples()):
        ax.text(row.expected_z14p44_um, yi + 0.12, f"{row.line}\n{row.expected_z14p44_um:.3f} µm", ha="center", va="bottom", fontsize=8.3)
    ax.set_yticks(y); ax.set_yticklabels(opt["line"].tolist())
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_title(f"{VERSION} — H-alpha/[N II] at z=14.44: out-of-band audit, no fake spectrum")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HALPHA_NII_OUT_OF_BAND_AUDIT.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def plot_slopes(df, avg_slope, avg_z):
    import numpy as np, matplotlib.pyplot as plt
    valid = df[df["sample_count"] > 0].copy()
    xs = np.linspace(valid["rest_um"].min()-0.004, valid["rest_um"].max()+0.004, 280)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    style_light(ax)
    styles = ["-", "--", ":", "-.", (0,(5,1))]
    for style, row in zip(styles, valid.itertuples()):
        ax.plot(xs, row.slope_m * xs, color="0.12", linewidth=0.70, linestyle=style, label=f"{int(row.n)} {row.line}: m={row.slope_m:.6f}")
        ax.scatter([row.rest_um], [row.raw_peak_sample_um], s=36, color="black", zorder=6)
        ax.text(row.rest_um, row.raw_peak_sample_um, f" {int(row.n)}", fontsize=9.5, fontweight="bold")
    ax.plot(xs, avg_slope * xs, color="darkorange", linewidth=1.30, label=f"average: m={avg_slope:.6f}, z={avg_z:.6f}")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed raw peak wavelength, micron")
    ax.set_title(f"{VERSION} — real UV-line slopes plus orange average")
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_UV_SLOPES_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor="white")
    plt.show()
    return path


def plot_summary(uv, opt, avg_rest, avg_obs, avg_slope, avg_z):
    import matplotlib.pyplot as plt
    rows = []
    valid = uv[uv["sample_count"] > 0]
    for r in uv.itertuples():
        obs = f"{r.raw_peak_sample_um:.6f}" if r.raw_peak_sample_um == r.raw_peak_sample_um else "OUT"
        sm = f"{r.slope_m:.6f}" if r.slope_m == r.slope_m else ""
        zz = f"{r.z_from_raw_peak:.6f}" if r.z_from_raw_peak == r.z_from_raw_peak else ""
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", obs, sm, zz])
    rows.append(["UV SUM", "SUM", f"{valid['rest_um'].sum():.6f}", f"{valid['raw_peak_sample_um'].sum():.6f}", "", ""])
    rows.append(["UV AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"])
    for r in opt.itertuples():
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", f"pred {r.expected_z14p44_um:.6f}", "out-of-band", "not measured"])
    fig, ax = plt.subplots(figsize=(15.2, 6.4)); fig.patch.set_facecolor("white"); ax.axis("off")
    ax.set_title(f"{VERSION} — UV measurements + H-alpha/[N II] out-of-band audit", fontsize=12.5, pad=10)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "obs/raw λ µm", "slope m", "z"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(7.9); table.scale(1, 1.24)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor("0.55"); cell.set_linewidth(0.42)
        if r == 0:
            cell.set_facecolor("0.90"); cell.get_text().set_fontweight("bold")
        elif r in [len(UV_LINES)+1, len(UV_LINES)+2]:
            cell.set_facecolor("#fff3e0"); cell.get_text().set_fontweight("bold")
        elif r > len(UV_LINES)+2:
            cell.set_facecolor("#eef2ff")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SUMMARY_TABLE_UV_AND_OPTICAL.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def make_widget(wave, flux, unit, uv, opt, avg_obs):
    try:
        import numpy as np, matplotlib.pyplot as plt, ipywidgets as widgets
        from IPython.display import display, clear_output
    except Exception as exc:
        print(f"Widget skipped: {exc}")
        return
    mn, mx = float(np.nanmin(wave)), float(np.nanmax(wave))
    uv_min = max(mn, float(uv["expected_z14p44_um"].min()-0.2))
    uv_max = min(mx, float(uv["expected_z14p44_um"].max()+0.2))
    rng = widgets.FloatRangeSlider(value=[uv_min, uv_max], min=mn, max=mx, step=(mx-mn)/1000, description="λ range", layout=widgets.Layout(width="95%"), readout_format=".4f")
    out = widgets.Output()
    def redraw(*_):
        with out:
            clear_output(wait=True)
            m = (wave >= rng.value[0]) & (wave <= rng.value[1])
            fig, ax = plt.subplots(figsize=(16.0, 7.2)); style_dark(ax)
            ax.plot(wave[m], flux[m], color="#e0f2fe", linewidth=0.62, label="real spectrum samples")
            for r in uv.itertuples():
                if r.sample_count > 0:
                    ax.axvline(r.raw_peak_sample_um, color="#f8fafc", linestyle="--", linewidth=0.70)
                    ax.text(r.raw_peak_sample_um, np.nanmax(flux[m]) if m.any() else 0, str(int(r.n)), color="#f8fafc", fontsize=9, va="top")
            ax.axvline(avg_obs, color="#f97316", linewidth=1.3, label="UV average")
            ax.set_xlabel("Observed wavelength, micron"); ax.set_ylabel(f"Flux, native unit: {unit}")
            ax.set_title(f"{VERSION} — interactive real-data spectrum window")
            legend_dark(ax, "upper right"); fig.tight_layout(); plt.show()
    rng.observe(redraw, names="value")
    display(widgets.VBox([rng, out])); redraw()


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows: widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i,h in enumerate(headers)))
    print("-" * (sum(widths) + 3*(len(widths)-1)))
    for row in rows: print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    records, obs_path, cand_path = query_products()
    if not records:
        print(f"CODE OUTPUT: {VERSION}\nNo MAST candidates found.\n{obs_path}")
        print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")); print(f"# {VERSION}"); return
    cover, negpri, nrows, path, status, wave, flux, unit, used, mode, rec = choose_best(records)
    uv, opt = line_tables(wave, flux)
    valid = uv[uv["sample_count"] > 0].copy()
    if valid.empty:
        raise RuntimeError("Readable spectrum found, but none of the UV windows are covered by this product.")
    avg_rest, avg_obs = float(valid["rest_um"].mean()), float(valid["raw_peak_sample_um"].mean())
    avg_slope, avg_z = avg_obs / avg_rest, avg_obs / avg_rest - 1

    import pandas as pd
    raw_csv = CSV / f"{VERSION}_REAL_SPECTRUM_READOUT.csv"
    uv_csv = CSV / f"{VERSION}_UV_MEASUREMENTS.csv"
    opt_csv = CSV / f"{VERSION}_OPTICAL_HALPHA_NII_AUDIT.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    pd.DataFrame({"wavelength_um": wave, f"flux_{unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
    uv.to_csv(uv_csv, index=False); opt.to_csv(opt_csv, index=False)
    pd.DataFrame([{"target": TARGET, "z_ref": Z_REF, "program": PROGRAM_ID, "product": path.name, "mode": mode, "used_hdus": used, "data_rule": "real MAST samples only; no smoothing; no synthetic spectra", "uv_measured_lines": len(valid), "uv_avg_obs_um": avg_obs, "uv_avg_slope": avg_slope, "uv_avg_z": avg_z, "optical_triplet_status": "predicted near 10.1 um at z=14.44; not faked if out of range"}]).to_csv(summary_csv, index=False)

    p1 = plot_combined_uv(wave, flux, unit, uv, avg_obs)
    p2 = plot_optical_audit(opt, wave)
    p3 = plot_slopes(uv, avg_slope, avg_z)
    p4 = plot_summary(uv, opt, avg_rest, avg_obs, avg_slope, avg_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Target", TARGET), ("Reference z", f"{Z_REF:.6f}"), ("MAST program", PROGRAM_ID),
        ("Product", path.name), ("Reader mode", mode), ("Data rule", "REAL MAST samples only"),
        ("No smoothing", "TRUE"), ("No synthetic spectra", "TRUE"), ("UV measured lines", len(valid)),
        ("UV observed sum um", f"{valid['raw_peak_sample_um'].sum():.6f}"),
        ("UV observed average um", f"{avg_obs:.6f}"), ("UV average slope", f"{avg_slope:.6f}"), ("UV average z", f"{avg_z:.6f}"),
        ("Combined UV plot", str(p1)), ("Optical H-alpha/[N II] audit", str(p2)), ("Slope plot", str(p3)),
        ("Summary table PNG", str(p4)), ("Raw CSV", str(raw_csv)), ("UV CSV", str(uv_csv)), ("Optical CSV", str(opt_csv))
    ], ["Field", "Value"])

    print("\nUV MEASUREMENTS AND AVERAGE")
    rows = []
    for r in valid.itertuples(): rows.append((int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}", f"{r.slope_m:.6f}", f"{r.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{valid['rest_um'].sum():.6f}", f"{valid['raw_peak_sample_um'].sum():.6f}", "", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw obs um", "Slope m", "z"])
    print("\nOPTICAL H-ALPHA / [N II]")
    print_table([(int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.expected_z14p44_um:.6f}", r.status) for r in opt.itertuples()], ["#", "Line", "Rest um", "Pred obs um", "Status"])
    print("\nWIDGET")
    print("Interactive widget below uses the same real spectrum arrays; controls only change display range.")
    make_widget(wave, flux, unit, uv, opt, avg_obs)
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")); print(f"# {VERSION}")

if __name__ == "__main__":
    main()
