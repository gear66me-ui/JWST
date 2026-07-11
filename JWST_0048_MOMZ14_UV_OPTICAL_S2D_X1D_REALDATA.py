# JWST_0048
# MoM-z14 real-data workflow with X1D table support and S2D 2D-spectrum support.
# No AI images. No synthetic spectra. No smoothing. No continuum subtraction.
# Spectral plots use raw MAST flux samples from WAVELENGTH/FLUX tables, or raw S2D pixel-row slices through the raw peak pixel.

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
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "astroquery"]:
        need(pkg)
    for p in [PNG, CSV, DATA]:
        p.mkdir(parents=True, exist_ok=True)


def query_mast_products():
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
    subgroup = "productSubGroupDescription" if "productSubGroupDescription" in prod_df.columns else None
    mask = pd.Series(False, index=prod_df.index)
    if fn_col:
        names = prod_df[fn_col].astype(str).str.lower()
        mask |= (names.str.contains("x1d") | names.str.contains("s2d")) & names.str.endswith(".fits")
    if subgroup:
        mask |= prod_df[subgroup].astype(str).str.upper().isin(["X1D", "S2D"])
    candidates = prod_df[mask].copy()
    if fn_col and not candidates.empty:
        names = candidates[fn_col].astype(str).str.lower()
        candidates["_priority"] = names.str.contains("x1d").astype(int) * 2 + names.str.contains("s2d").astype(int)
        candidates = candidates.sort_values("_priority", ascending=False).drop(columns=["_priority"])
    cand_path = CSV / f"{VERSION}_PROGRAM_{PROGRAM_ID}_SPECTRUM_CANDIDATES.csv"
    candidates.to_csv(cand_path, index=False)
    return candidates.to_dict("records"), obs_path, cand_path


def download_product(record):
    from astroquery.mast import Observations
    uri = record.get("dataURI") or record.get("dataUri") or record.get("uri")
    filename = record.get("productFilename") or record.get("productfilename") or "mast_product.fits"
    local = DATA / str(filename)
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    if not uri:
        raise RuntimeError("No dataURI in MAST product record.")
    Observations.download_file(uri, local_path=str(local))
    if not local.exists() or local.stat().st_size < 100000:
        raise RuntimeError(f"Downloaded file missing or too small: {filename}")
    return local, "downloaded-from-mast"


def normalize_wave_units(arr):
    import numpy as np
    a = np.asarray(arr, dtype=float).copy()
    med = abs(float(np.nanmedian(a))) if a.size else float("nan")
    if med != med:
        return a
    if 1e-7 < med < 1e-3:
        return a * 1e6
    if 1000.0 < med < 300000.0:
        return a / 10000.0
    if 100.0 < med < 30000.0:
        return a / 1000.0
    return a


def wavelength_axis_from_wcs(header, nx, ny):
    import numpy as np
    from astropy.wcs import WCS
    wcs = WCS(header)
    x = np.arange(nx, dtype=float)
    y = np.full(nx, float(ny // 2))
    world = wcs.all_pix2world(x, y, 0)
    if not isinstance(world, (list, tuple)):
        world = [world]
    candidates = []
    for arr in world:
        arr = normalize_wave_units(np.ravel(np.asarray(arr, dtype=float)))
        if len(arr) != nx or not np.isfinite(arr).any():
            continue
        med = float(np.nanmedian(arr))
        span = float(np.nanmax(arr) - np.nanmin(arr))
        diffs = np.diff(arr[np.isfinite(arr)])
        mono = max((diffs > 0).mean() if len(diffs) else 0, (diffs < 0).mean() if len(diffs) else 0)
        if 0.2 <= med <= 30.0 and span > 0.05:
            candidates.append((mono, span, arr))
    if not candidates:
        raise RuntimeError("Could not derive wavelength axis from S2D WCS header.")
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return candidates[0][2]


def read_product(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                wave = np.asarray(data["WAVELENGTH"], dtype=float).ravel()
                flux = np.asarray(data["FLUX"], dtype=float).ravel()
                mask = np.isfinite(wave) & np.isfinite(flux)
                order = np.argsort(wave[mask])
                return {
                    "kind": "X1D_TABLE",
                    "path": path,
                    "wave": wave[mask][order],
                    "flux": flux[mask][order],
                    "flux_unit": str(hdu.header.get("TUNIT2", "native")),
                    "hdu_info": f"{i}:{getattr(hdu, 'name', '')}",
                }
        for i, hdu in enumerate(hdul):
            name = str(getattr(hdu, "name", "")).upper()
            data = getattr(hdu, "data", None)
            if data is None:
                continue
            arr = np.asarray(data, dtype=float)
            if arr.ndim == 2 and arr.size > 100 and (name in ["SCI", "S2D", "DATA"] or i > 0):
                ny, nx = arr.shape
                wave_axis = wavelength_axis_from_wcs(hdu.header, nx, ny)
                return {
                    "kind": "S2D_IMAGE",
                    "path": path,
                    "image": arr,
                    "wave_axis": wave_axis,
                    "flux_unit": str(hdu.header.get("BUNIT", "native")),
                    "hdu_info": f"{i}:{name}",
                }
    raise RuntimeError(f"No readable X1D table or S2D image spectrum in {path.name}")


def coverage_score(product):
    expected = [rest * (1 + Z_REF) for _, _, rest, _ in UV_LINES]
    if product["kind"] == "X1D_TABLE":
        lo, hi = float(product["wave"].min()), float(product["wave"].max())
    else:
        lo, hi = float(min(product["wave_axis"])), float(max(product["wave_axis"]))
    return sum(lo <= x <= hi for x in expected), lo, hi


def choose_best_product(records):
    attempts = []
    for rec in records[:40]:
        try:
            path, status = download_product(rec)
            prod = read_product(path)
            score, lo, hi = coverage_score(prod)
            n = len(prod["wave"]) if prod["kind"] == "X1D_TABLE" else prod["image"].size
            attempts.append((score, n, path, status, prod, rec, lo, hi))
        except Exception as exc:
            warnings.warn(str(exc))
    if not attempts:
        raise RuntimeError("No readable X1D or S2D spectrum product found. Candidate inventory CSV was saved.")
    attempts.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return attempts[0]


def peak_x1d(product, center, half_width=0.060):
    import numpy as np
    w, f = product["wave"], product["flux"]
    mask = (w >= center - half_width) & (w <= center + half_width)
    if mask.sum() == 0:
        return float("nan"), float("nan"), 0, None, None
    x, y = w[mask], f[mask]
    i = int(np.nanargmax(y))
    return float(x[i]), float(y[i]), int(mask.sum()), None, None


def peak_s2d(product, center, half_width=0.060):
    import numpy as np
    wave = product["wave_axis"]
    img = product["image"]
    cols = np.where((wave >= center - half_width) & (wave <= center + half_width))[0]
    if len(cols) == 0:
        return float("nan"), float("nan"), 0, None, None
    sub = img[:, cols]
    if not np.isfinite(sub).any():
        return float("nan"), float("nan"), int(sub.size), None, None
    flat = int(np.nanargmax(sub))
    row, relcol = np.unravel_index(flat, sub.shape)
    col = int(cols[relcol])
    return float(wave[col]), float(img[row, col]), int(sub.size), int(row), int(col)


def build_tables(product):
    import pandas as pd
    uv_rows = []
    for n, label, rest, species in UV_LINES:
        expected = rest * (1 + Z_REF)
        if product["kind"] == "X1D_TABLE":
            peak_w, peak_f, samples, row, col = peak_x1d(product, expected)
        else:
            peak_w, peak_f, samples, row, col = peak_s2d(product, expected)
        uv_rows.append({
            "n": n,
            "line": label,
            "species": species,
            "line_group": "UV line/reference",
            "rest_um": rest,
            "expected_at_z14p44_um": expected,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux_native": peak_f,
            "sample_count_in_window": samples,
            "s2d_peak_row": row,
            "s2d_peak_col": col,
            "slope_m": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1.0 if peak_w == peak_w else float("nan"),
            "status": "measured from raw samples" if samples else "outside product wavelength coverage",
        })
    opt_rows = []
    if product["kind"] == "X1D_TABLE":
        lo, hi = float(product["wave"].min()), float(product["wave"].max())
    else:
        lo, hi = float(min(product["wave_axis"])), float(max(product["wave_axis"]))
    for n, label, rest, species in OPTICAL_LINES:
        expected = rest * (1 + Z_REF)
        covered = lo <= expected <= hi
        opt_rows.append({
            "n": n,
            "line": label,
            "species": species,
            "line_group": "optical H-alpha/[N II]",
            "rest_um": rest,
            "expected_at_z14p44_um": expected,
            "status": "covered by downloaded product; can measure if real flux exists" if covered else "out of downloaded real-data range; not faked",
        })
    return pd.DataFrame(uv_rows), pd.DataFrame(opt_rows)


def spectrum_slice(product, center, half_width=0.120, s2d_row=None):
    import numpy as np
    if product["kind"] == "X1D_TABLE":
        w, f = product["wave"], product["flux"]
        mask = (w >= center - half_width) & (w <= center + half_width)
        return w[mask], f[mask]
    wave = product["wave_axis"]
    img = product["image"]
    row = int(s2d_row) if s2d_row is not None and s2d_row == s2d_row else img.shape[0] // 2
    mask = (wave >= center - half_width) & (wave <= center + half_width)
    return wave[mask], img[row, mask]


def freq_thz_from_um(x):
    return 299.792458 / x


def um_from_freq_thz(x):
    return 299.792458 / x


def light(ax):
    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linewidth=0.45)
    ax.tick_params(colors="0.1", labelsize=9)
    for sp in ax.spines.values():
        sp.set_color("0.25")
        sp.set_linewidth(0.65)


def dark(ax):
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


def plot_combined_uv(product, uv_df, avg_obs):
    import matplotlib.pyplot as plt
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    if valid.empty:
        return None
    xmin = float(valid["expected_at_z14p44_um"].min() - 0.16)
    xmax = float(valid["expected_at_z14p44_um"].max() + 0.16)
    fig, ax = plt.subplots(figsize=(16.4, 8.0))
    dark(ax)
    for r in valid.itertuples():
        x, y = spectrum_slice(product, r.expected_at_z14p44_um, 0.115, r.s2d_peak_row)
        ax.plot(x, y, linewidth=0.62, label=f"{int(r.n)} {r.line} raw slice")
        ax.axvline(r.raw_peak_sample_um, color="#f8fafc", linestyle="--", linewidth=0.60)
        ax.scatter([r.raw_peak_sample_um], [r.raw_peak_flux_native], s=28, color="#f8fafc", edgecolor="#020617", zorder=5)
        ax.text(r.raw_peak_sample_um, r.raw_peak_flux_native, f" {int(r.n)}", color="#f8fafc", fontsize=9, fontweight="bold", va="bottom")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.35, label=f"average raw peak λ = {avg_obs:.6f} µm")
    ax.set_xlim(xmin, xmax)
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Raw flux, FITS unit: {product['flux_unit']}")
    ax.set_title(f"{VERSION} — {TARGET}: combined UV real-data line windows, z≈14.44")
    leg = ax.legend(loc="upper right", fontsize=7.4, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMBINED_UV_REALDATA_LINES.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_uv_windows(product, uv_df):
    import matplotlib.pyplot as plt
    paths = []
    for r in uv_df.itertuples():
        if r.sample_count_in_window <= 0:
            continue
        x, y = spectrum_slice(product, r.expected_at_z14p44_um, 0.095, r.s2d_peak_row)
        fig, ax = plt.subplots(figsize=(12.6, 5.4))
        light(ax)
        ax.plot(x, y, color="black", linewidth=0.55, label="raw real-data spectrum slice")
        ax.axvline(r.expected_at_z14p44_um, color="0.55", linestyle=":", linewidth=0.80, label="expected at z=14.44")
        ax.axvline(r.raw_peak_sample_um, color="black", linestyle="--", linewidth=0.85, label="raw local peak sample")
        ax.scatter([r.raw_peak_sample_um], [r.raw_peak_flux_native], s=30, color="black", zorder=5)
        ax.text(r.raw_peak_sample_um, r.raw_peak_flux_native, f"  {int(r.n)}", fontsize=12, fontweight="bold", va="bottom")
        row_note = f"\nS2D row={r.s2d_peak_row}" if product["kind"] == "S2D_IMAGE" else ""
        ax.text(0.02, 0.95, f"{int(r.n)} {r.line}\nrest λ={r.rest_um:.6f} µm\nraw peak λ={r.raw_peak_sample_um:.6f} µm\nz={r.z_from_raw_peak:.6f}{row_note}", transform=ax.transAxes, ha="left", va="top", fontsize=8.7, bbox=dict(facecolor="white", edgecolor="0.45", boxstyle="round,pad=0.32"))
        top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
        top.set_xlabel("Frequency, THz")
        ax.set_xlabel("Observed wavelength, micron")
        ax.set_ylabel(f"Raw flux, FITS unit: {product['flux_unit']}")
        ax.set_title(f"{VERSION} — real raw UV window: {r.line}")
        ax.legend(loc="best", fontsize=7.8, frameon=True, edgecolor="0.55")
        fig.tight_layout()
        safe = r.line.replace("[", "").replace("]", "").replace(" ", "_").replace("-", "_")
        path = PNG / f"{VERSION}_UV_LINE_{int(r.n)}_{safe}_WINDOW.png"
        fig.savefig(path, dpi=260, facecolor="white")
        plt.show()
        paths.append(path)
    return paths


def plot_s2d_overview(product, uv_df):
    if product["kind"] != "S2D_IMAGE":
        return None
    import numpy as np, matplotlib.pyplot as plt
    img = product["image"]
    wave = product["wave_axis"]
    finite = np.isfinite(img)
    lo, hi = np.nanpercentile(img[finite], [1, 99]) if finite.any() else (0, 1)
    fig, ax = plt.subplots(figsize=(14.5, 6.2))
    light(ax)
    extent = [float(min(wave)), float(max(wave)), 0, img.shape[0]-1]
    ax.imshow(img, origin="lower", aspect="auto", extent=extent, vmin=lo, vmax=hi, cmap="gray")
    for r in uv_df.itertuples():
        if r.sample_count_in_window > 0:
            ax.axvline(r.raw_peak_sample_um, color="darkorange", linewidth=0.75)
            ax.text(r.raw_peak_sample_um, img.shape[0]*0.94, str(int(r.n)), ha="center", va="top", color="darkorange", fontweight="bold")
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel("S2D spatial pixel row")
    ax.set_title(f"{VERSION} — raw S2D 2D spectrum image with UV peak markers")
    fig.tight_layout()
    path = PNG / f"{VERSION}_RAW_S2D_2D_OVERVIEW.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def plot_optical_audit(opt_df, product):
    import matplotlib.pyplot as plt
    if product["kind"] == "X1D_TABLE":
        lo, hi = float(product["wave"].min()), float(product["wave"].max())
    else:
        lo, hi = float(min(product["wave_axis"])), float(max(product["wave_axis"]))
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    light(ax)
    y = list(range(len(opt_df)))
    ax.axvspan(lo, hi, color="#e0f2fe", alpha=0.55, label="downloaded real-data wavelength coverage")
    ax.scatter(opt_df["expected_at_z14p44_um"], y, s=44, color="black", zorder=5, label="predicted observed λ at z=14.44")
    for yi, r in zip(y, opt_df.itertuples()):
        ax.text(r.expected_at_z14p44_um, yi + 0.13, f"{r.line}\n{r.expected_at_z14p44_um:.3f} µm", ha="center", va="bottom", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(opt_df["line"].tolist())
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_title(f"{VERSION} — H-alpha/[N II] at z=14.44: wavelength audit, no fake spectrum")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_OPTICAL_HALPHA_NII_AUDIT.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def plot_slopes(uv_df, avg_slope, avg_z):
    import numpy as np, matplotlib.pyplot as plt
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    xs = np.linspace(valid["rest_um"].min() - 0.004, valid["rest_um"].max() + 0.004, 280)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    light(ax)
    styles = ["-", "--", ":", "-.", (0, (5, 1))]
    for style, r in zip(styles, valid.itertuples()):
        ax.plot(xs, r.slope_m * xs, color="0.12", linewidth=0.70, linestyle=style, label=f"{int(r.n)} {r.line}: m={r.slope_m:.6f}")
        ax.scatter([r.rest_um], [r.raw_peak_sample_um], s=36, color="black", zorder=6)
        ax.text(r.rest_um, r.raw_peak_sample_um, f" {int(r.n)}", fontsize=9.5, fontweight="bold")
    ax.plot(xs, avg_slope * xs, color="darkorange", linewidth=1.30, label=f"average: m={avg_slope:.6f}, z={avg_z:.6f}")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed raw peak wavelength, micron")
    ax.set_title(f"{VERSION} — UV-line slopes plus orange average")
    ax.legend(loc="upper left", fontsize=7.6, frameon=True, facecolor="white", edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_UV_SLOPES_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor="white")
    plt.show()
    return path


def plot_summary_table(uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z):
    import matplotlib.pyplot as plt
    rows = []
    for r in uv_df.itertuples():
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}" if r.raw_peak_sample_um == r.raw_peak_sample_um else "OUT", f"{r.slope_m:.6f}" if r.slope_m == r.slope_m else "", f"{r.z_from_raw_peak:.6f}" if r.z_from_raw_peak == r.z_from_raw_peak else ""])
    valid = uv_df[uv_df["sample_count_in_window"] > 0]
    rows.append(["UV SUM", "SUM", f"{valid['rest_um'].sum():.6f}", f"{valid['raw_peak_sample_um'].sum():.6f}", "", ""])
    rows.append(["UV AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"])
    for r in opt_df.itertuples():
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", f"pred {r.expected_at_z14p44_um:.6f}", "out-of-band", "not measured"])
    fig, ax = plt.subplots(figsize=(15.0, 6.4))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(f"{VERSION} — UV measurements + H-alpha/[N II] audit", fontsize=12.5, pad=10)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "obs/raw λ µm", "slope m", "z"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.25)
    for (rr, cc), cell in table.get_celld().items():
        cell.set_edgecolor("0.55")
        cell.set_linewidth(0.42)
        if rr == 0:
            cell.set_facecolor("0.90")
            cell.get_text().set_fontweight("bold")
        elif rr in [len(uv_df)+1, len(uv_df)+2]:
            cell.set_facecolor("#fff3e0")
            cell.get_text().set_fontweight("bold")
        elif rr > len(uv_df)+2:
            cell.set_facecolor("#eef2ff")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SUMMARY_TABLE_UV_AND_OPTICAL.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    records, obs_path, cand_path = query_mast_products()
    if not records:
        print(f"CODE OUTPUT: {VERSION}\nNo public spectrum candidates found for JWST Program {PROGRAM_ID}.")
        print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
        print(f"# {VERSION}")
        return
    score, n, path, status, product, record, lo, hi = choose_best_product(records)
    uv_df, opt_df = build_tables(product)
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    if valid.empty:
        raise RuntimeError("Downloaded product is readable but covers none of the requested UV line windows at z=14.44.")
    avg_rest = float(valid["rest_um"].mean())
    avg_obs = float(valid["raw_peak_sample_um"].mean())
    avg_slope = avg_obs / avg_rest
    avg_z = avg_slope - 1.0

    import pandas as pd
    raw_csv = CSV / f"{VERSION}_REAL_RAW_SPECTRUM.csv"
    uv_csv = CSV / f"{VERSION}_UV_LINE_MEASUREMENTS.csv"
    optical_csv = CSV / f"{VERSION}_OPTICAL_HALPHA_NII_AUDIT.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    if product["kind"] == "X1D_TABLE":
        pd.DataFrame({"wavelength_um_raw": product["wave"], f"flux_raw_{product['flux_unit']}": product["flux"], "frequency_thz": freq_thz_from_um(product["wave"])}).to_csv(raw_csv, index=False)
    else:
        pd.DataFrame(product["image"]).to_csv(raw_csv, index=False)
        pd.DataFrame({"wavelength_um": product["wave_axis"]}).to_csv(CSV / f"{VERSION}_S2D_WAVELENGTH_AXIS.csv", index=False)
    uv_df.to_csv(uv_csv, index=False)
    opt_df.to_csv(optical_csv, index=False)
    pd.DataFrame([{
        "target": TARGET,
        "reference_z": Z_REF,
        "mast_program": PROGRAM_ID,
        "product": path.name,
        "product_kind": product["kind"],
        "download_status": status,
        "hdu_info": product["hdu_info"],
        "data_rule": "raw MAST flux samples; no smoothing; no synthetic spectra",
        "uv_lines_measured": int(len(valid)),
        "uv_average_rest_um": avg_rest,
        "uv_average_observed_um": avg_obs,
        "uv_average_slope": avg_slope,
        "uv_average_z": avg_z,
        "wavelength_coverage_min_um": lo,
        "wavelength_coverage_max_um": hi,
        "optical_triplet_status": "H-alpha/[N II] predicted near 10.1 um at z=14.44; plotted only as wavelength audit unless real data cover it",
    }]).to_csv(summary_csv, index=False)

    combined_uv = plot_combined_uv(product, uv_df, avg_obs)
    uv_windows = plot_uv_windows(product, uv_df)
    s2d_overview = plot_s2d_overview(product, uv_df)
    optical_audit = plot_optical_audit(opt_df, product)
    slope_plot = plot_slopes(uv_df, avg_slope, avg_z)
    table_plot = plot_summary_table(uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Target", TARGET),
        ("Reference z", f"{Z_REF:.6f}"),
        ("MAST program", PROGRAM_ID),
        ("Product", path.name),
        ("Product kind", product["kind"]),
        ("Data rule", "RAW MAST samples only; no smoothing; no synthetic spectra"),
        ("Coverage um", f"{lo:.6f} to {hi:.6f}"),
        ("UV measured lines", len(valid)),
        ("UV observed sum um", f"{valid['raw_peak_sample_um'].sum():.6f}"),
        ("UV observed average um", f"{avg_obs:.6f}"),
        ("UV average slope", f"{avg_slope:.6f}"),
        ("UV average z", f"{avg_z:.6f}"),
        ("Combined UV plot", str(combined_uv)),
        ("S2D overview", str(s2d_overview)),
        ("Optical H-alpha/[N II] audit", str(optical_audit)),
        ("Slope plot", str(slope_plot)),
        ("Summary table PNG", str(table_plot)),
        ("Raw CSV", str(raw_csv)),
        ("UV CSV", str(uv_csv)),
        ("Optical CSV", str(optical_csv)),
    ], ["Field", "Value"])

    print("\nUV LINE WINDOWS")
    print_table([(i+1, str(p)) for i, p in enumerate(uv_windows)], ["#", "Path"])
    print("\nUV MEASUREMENTS AND AVERAGE")
    rows = []
    for r in valid.itertuples():
        rows.append((int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}", f"{r.slope_m:.6f}", f"{r.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{valid['rest_um'].sum():.6f}", f"{valid['raw_peak_sample_um'].sum():.6f}", "", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw obs um", "Slope m", "z"])
    print("\nOPTICAL H-ALPHA / [N II] STATUS")
    print_table([(int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.expected_at_z14p44_um:.6f}", r.status) for r in opt_df.itertuples()], ["#", "Line", "Rest um", "Pred obs um", "Status"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
