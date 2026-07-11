# JWST_0047
# MoM-z14 real-data UV + optical audit with JWST_0035 color/style pattern.
# No AI images. No synthetic spectra. Spectral plots use raw MAST WAVELENGTH and FLUX samples.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib, warnings

VERSION = "JWST_0047"
PROJECT = "MOMZ14 UV + OPTICAL AUDIT, JWST_0035 STYLE"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

TARGET = "MoM-z14"
Z_REF = 14.44
PROGRAM_ID = "5224"

# JWST_0035 visual system
BG = "#050712"
PANEL = "#020617"
GRID = "#334155"
SPINE = "#94a3b8"
TEXT = "#f8fafc"
SUBTEXT = "#dbeafe"
MUTED = "#94a3b8"
OBS_BLUE = "#93c5fd"
RESID_PINK = "#fb7185"
SMOOTH_ORANGE = "#f97316"
FIT_GREEN = "#22c55e"
CONT_YELLOW = "#facc15"
BAND_GRAY = "#64748b"
LINE_COLORS = ["#38bdf8", "#fb7185", "#a78bfa", "#facc15", "#22c55e"]

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


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def dark_axis(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.68)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for s in ax.spines.values():
        s.set_color(SPINE)


def legend(ax, loc="best", fontsize=8.0):
    leg = ax.legend(loc=loc, fontsize=fontsize, facecolor=PANEL, edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color(TEXT)
    return leg


def freq_thz_from_um(x):
    return 299.792458 / x


def um_from_freq_thz(x):
    return 299.792458 / x


def query_mast_products():
    import pandas as pd
    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection="JWST", proposal_id=PROGRAM_ID)
    obs_df = obs.to_pandas() if hasattr(obs, "to_pandas") else pd.DataFrame(obs)
    obs_path = CSV / f"{VERSION}_PROGRAM_{PROGRAM_ID}_OBSERVATIONS.csv"
    obs_df.to_csv(obs_path, index=False)
    if len(obs) == 0:
        return [], obs_path, None
    products = Observations.get_product_list(obs)
    prod_df = products.to_pandas() if hasattr(products, "to_pandas") else pd.DataFrame(products)
    prod_path = CSV / f"{VERSION}_PROGRAM_{PROGRAM_ID}_PRODUCTS.csv"
    prod_df.to_csv(prod_path, index=False)
    if prod_df.empty:
        return [], obs_path, prod_path
    fn_col = "productFilename" if "productFilename" in prod_df.columns else None
    group_col = "productSubGroupDescription" if "productSubGroupDescription" in prod_df.columns else None
    mask = pd.Series(False, index=prod_df.index)
    if fn_col:
        names = prod_df[fn_col].astype(str).str.lower()
        mask |= names.str.contains("x1d") & names.str.endswith(".fits")
        mask |= names.str.contains("s2d") & names.str.endswith(".fits")
    if group_col:
        mask |= prod_df[group_col].astype(str).str.upper().isin(["X1D", "S2D"])
    candidates = prod_df[mask].copy()
    if candidates.empty and fn_col:
        candidates = prod_df[prod_df[fn_col].astype(str).str.lower().str.endswith(".fits")].copy()
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
        raise RuntimeError("MAST product record has no dataURI.")
    Observations.download_file(uri, local_path=str(local))
    if not local.exists() or local.stat().st_size < 100000:
        raise RuntimeError(f"downloaded file missing or too small: {filename}")
    return local, "downloaded-from-mast"


def read_wavelength_flux(path):
    import numpy as np
    from astropy.io import fits
    waves, fluxes = [], []
    unit = "native"
    used_hdus = []
    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                unit = str(hdu.header.get("TUNIT2", unit))
                used_hdus.append(f"{i}:{getattr(hdu, 'name', '')}")
                w = data["WAVELENGTH"]
                f = data["FLUX"]
                try:
                    for wi, fi in zip(w, f):
                        waves.append(np.ravel(np.asarray(wi, dtype=float)))
                        fluxes.append(np.ravel(np.asarray(fi, dtype=float)))
                except TypeError:
                    waves.append(np.ravel(np.asarray(w, dtype=float)))
                    fluxes.append(np.ravel(np.asarray(f, dtype=float)))
    if not waves:
        raise RuntimeError(f"No WAVELENGTH/FLUX table found in {path.name}")
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    m = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[m], flux[m]
    if unit.lower() == "jy":
        flux = flux * 1.0e6
        unit = "microJy"
    order = np.argsort(wave)
    return wave[order], flux[order], unit, "; ".join(used_hdus)


def choose_best_spectrum(records):
    expected = [rest * (1 + Z_REF) for _, _, rest, _ in UV_LINES]
    attempts = []
    for rec in records[:40]:
        try:
            path, status = download_product(rec)
            wave, flux, unit, used = read_wavelength_flux(path)
            if len(wave) < 30:
                continue
            coverage = sum((min(wave) <= x <= max(wave)) for x in expected)
            attempts.append((coverage, len(wave), path, status, wave, flux, unit, used, rec))
        except Exception as exc:
            warnings.warn(str(exc))
    if not attempts:
        raise RuntimeError("No readable public spectrum product found. Candidate inventory CSV was saved.")
    attempts.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return attempts[0]


def peak_sample(wave, flux, center, half_width=0.060):
    import numpy as np
    m = (wave >= center - half_width) & (wave <= center + half_width)
    if m.sum() == 0:
        return float("nan"), float("nan"), 0
    ww, ff = wave[m], flux[m]
    i = int(np.nanargmax(ff))
    return float(ww[i]), float(ff[i]), int(m.sum())


def build_line_tables(wave, flux):
    import pandas as pd
    uv_rows = []
    for n, label, rest, species in UV_LINES:
        expected = rest * (1 + Z_REF)
        peak_w, peak_f, nwin = peak_sample(wave, flux, expected, 0.060)
        uv_rows.append({
            "n": n,
            "line": label,
            "species": species,
            "rest_um": rest,
            "expected_at_z14p44_um": expected,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux": peak_f,
            "sample_count_in_window": nwin,
            "slope_m": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1 if peak_w == peak_w else float("nan"),
            "status": "raw spectrum window measured" if nwin else "outside downloaded wavelength range",
        })
    opt_rows = []
    for n, label, rest, species in OPTICAL_LINES:
        expected = rest * (1 + Z_REF)
        covered = bool(min(wave) <= expected <= max(wave))
        peak_w, peak_f, nwin = peak_sample(wave, flux, expected, 0.080) if covered else (float("nan"), float("nan"), 0)
        opt_rows.append({
            "n": n,
            "line": label,
            "species": species,
            "rest_um": rest,
            "expected_at_z14p44_um": expected,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux": peak_f,
            "sample_count_in_window": nwin,
            "slope_m": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1 if peak_w == peak_w else float("nan"),
            "status": "raw spectrum window measured" if nwin else "out-of-band at z=14.44; not faked",
        })
    return pd.DataFrame(uv_rows), pd.DataFrame(opt_rows)


def plot_combined_uv(wave, flux, unit, uv_df, avg_obs):
    import matplotlib.pyplot as plt
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    if valid.empty:
        return None
    xmin = max(float(min(wave)), float(valid["expected_at_z14p44_um"].min() - 0.18))
    xmax = min(float(max(wave)), float(valid["expected_at_z14p44_um"].max() + 0.18))
    m = (wave >= xmin) & (wave <= xmax)
    fig, ax = plt.subplots(figsize=(18.0, 10.2))
    dark_axis(fig, ax)
    ax.plot(wave[m], flux[m], color=OBS_BLUE, lw=1.05, label="Observed raw flux")
    for i, row in enumerate(valid.itertuples()):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        ax.axvline(row.expected_at_z14p44_um, color=color, lw=0.95, ls=":", alpha=0.80)
        ax.axvline(row.raw_peak_sample_um, color=color, lw=1.05, alpha=0.95, label=f"{int(row.n)} {row.line}: raw peak {row.raw_peak_sample_um:.6f} µm")
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux], s=76, color=color, edgecolor=TEXT, zorder=8)
        ax.text(row.raw_peak_sample_um, ax.get_ylim()[1], f" {int(row.n)}", color=BG, fontsize=11, weight="bold", ha="left", va="top", bbox=dict(boxstyle="circle,pad=0.24", facecolor=color, edgecolor=TEXT))
    ax.axvline(avg_obs, color=SMOOTH_ORANGE, lw=2.05, label=f"orange: average raw peak λ={avg_obs:.6f} µm")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color(TEXT)
    top.tick_params(colors=SUBTEXT, labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Raw flux, {unit}")
    ax.set_title(f"{VERSION} — {TARGET}: combined real UV emission-line spectrum, JWST_0035 color style")
    legend(ax, "upper left", 7.7)
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMBINED_REAL_UV_LINES_0035_STYLE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_uv_windows(wave, flux, unit, uv_df):
    import matplotlib.pyplot as plt
    paths = []
    for i, row in enumerate(uv_df.itertuples()):
        if row.sample_count_in_window <= 0:
            continue
        color = LINE_COLORS[i % len(LINE_COLORS)]
        c = row.expected_at_z14p44_um
        m = (wave >= c - 0.095) & (wave <= c + 0.095)
        fig, ax = plt.subplots(figsize=(15.8, 7.2))
        dark_axis(fig, ax)
        ax.plot(wave[m], flux[m], color=OBS_BLUE, lw=1.05, label="Observed raw flux")
        ax.axvline(row.expected_at_z14p44_um, color=CONT_YELLOW, lw=1.05, ls=":", label="expected at z=14.44")
        ax.axvline(row.raw_peak_sample_um, color=color, lw=1.15, label="raw local peak sample")
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux], s=82, color=color, edgecolor=TEXT, zorder=9)
        ax.text(row.raw_peak_sample_um, row.raw_peak_flux, f"  {int(row.n)}", color=TEXT, fontsize=12, weight="bold", va="bottom")
        ax.text(0.025, 0.955, f"{int(row.n)}  {row.line}\nλrest = {row.rest_um:.6f} µm\nλraw peak = {row.raw_peak_sample_um:.6f} µm\nz = {row.z_from_raw_peak:.6f}\nm = {row.slope_m:.6f}", transform=ax.transAxes, ha="left", va="top", color=TEXT, fontsize=9.3, bbox=dict(boxstyle="round,pad=0.42", facecolor=PANEL, edgecolor="#475569"))
        top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
        top.set_xlabel("Frequency, THz")
        top.xaxis.label.set_color(TEXT)
        top.tick_params(colors=SUBTEXT, labelsize=8)
        ax.set_xlabel("Observed wavelength, micron")
        ax.set_ylabel(f"Raw flux, {unit}")
        ax.set_title(f"{VERSION} — real raw UV line window {int(row.n)}: {row.line}")
        legend(ax, "upper right")
        fig.tight_layout()
        safe = row.line.replace("[", "").replace("]", "").replace(" ", "_").replace("-", "_")
        path = PNG / f"{VERSION}_UV_LINE_{int(row.n)}_{safe}_0035_STYLE.png"
        fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
        plt.show()
        paths.append(path)
    return paths


def plot_optical_audit(opt_df, wave):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16.0, 6.8))
    dark_axis(fig, ax)
    y = list(range(len(opt_df)))
    ax.axvspan(float(min(wave)), float(max(wave)), color=BAND_GRAY, alpha=0.16, label="downloaded real spectrum coverage")
    for i, row in enumerate(opt_df.itertuples()):
        color = LINE_COLORS[i % 3]
        ax.scatter([row.expected_at_z14p44_um], [i], s=95, color=color, edgecolor=TEXT, zorder=5)
        ax.text(row.expected_at_z14p44_um, i + 0.12, f"{row.line}\n{row.expected_at_z14p44_um:.3f} µm", color=TEXT, ha="center", va="bottom", fontsize=9.0)
    ax.set_yticks(y)
    ax.set_yticklabels(opt_df["line"].tolist(), color=SUBTEXT)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_title(f"{VERSION} — H-alpha/[N II] at z=14.44: out-of-band audit, no fake spectrum")
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_OPTICAL_HALPHA_NII_OUT_OF_BAND_0035_STYLE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def fit_shared_slope(valid):
    import numpy as np
    x = np.asarray(valid["rest_um"], dtype=float)
    y = np.asarray(valid["raw_peak_sample_um"], dtype=float)
    return float(np.sum(x * y) / np.sum(x * x))


def plot_slopes(uv_df, avg_slope, avg_z, fit_slope):
    import numpy as np
    import matplotlib.pyplot as plt
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    xs = np.linspace(valid["rest_um"].min() - 0.0045, valid["rest_um"].max() + 0.0045, 280)
    fig, ax = plt.subplots(figsize=(16.8, 9.0))
    dark_axis(fig, ax)
    ylist = []
    for i, row in enumerate(valid.itertuples()):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        y = row.slope_m * xs
        ylist.extend(y)
        ax.plot(xs, y, color=color, lw=1.05, alpha=0.92, label=f"{int(row.n)} {row.line}: m={row.slope_m:.6f}")
        ax.scatter([row.rest_um], [row.raw_peak_sample_um], s=96, color=color, edgecolor=TEXT, zorder=9)
        ax.text(row.rest_um, row.raw_peak_sample_um, f"  {int(row.n)}", color=TEXT, fontsize=10, weight="bold", va="center")
    ax.plot(xs, avg_slope * xs, color=SMOOTH_ORANGE, lw=2.25, label=f"orange: mean slope={avg_slope:.6f}; mean z={avg_z:.6f}")
    ax.plot(xs, fit_slope * xs, color=FIT_GREEN, lw=1.70, ls="--", label=f"green dashed: shared least-squares slope={fit_slope:.6f}; z={fit_slope-1:.6f}")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed raw peak wavelength, micron")
    ax.set_title(f"{VERSION} — UV redshift slopes, orange average, green shared fit")
    legend(ax, "upper left", 7.8)
    fig.tight_layout()
    path = PNG / f"{VERSION}_UV_SLOPES_AVERAGE_GREEN_FIT_0035_STYLE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_summary_table(uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z, fit_slope):
    import matplotlib.pyplot as plt
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    rows = []
    for row in valid.itertuples():
        rows.append([int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.slope_m:.6f}", f"{row.z_from_raw_peak:.6f}", "UV raw"])
    rows.append(["SUM", "SUM", f"{valid['rest_um'].sum():.6f}", f"{valid['raw_peak_sample_um'].sum():.6f}", "", "", "UV"])
    rows.append(["AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}", "orange"])
    rows.append(["FIT", "SHARED SLOPE", "through origin", "", f"{fit_slope:.6f}", f"{fit_slope-1:.6f}", "green dashed"])
    for row in opt_df.itertuples():
        rows.append([int(row.n), row.line, f"{row.rest_um:.6f}", f"pred {row.expected_at_z14p44_um:.6f}", "out-of-band", "not measured", "optical audit"])
    fig, ax = plt.subplots(figsize=(18.0, 6.7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(f"{VERSION} — UV measurements + average + H-alpha/[N II] audit | JWST_0035 color style", color=TEXT, fontsize=13.5, pad=12)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "obs/raw λ µm", "slope m", "z", "status"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.75)
    table.scale(1, 1.34)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.48)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color(TEXT)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(PANEL if r % 2 else "#0f172a")
            cell.get_text().set_color(SUBTEXT)
            label = str(rows[r - 1][0]) if r - 1 < len(rows) else ""
            if label in ["SUM", "AVG", "FIT"]:
                cell.set_facecolor("#431407" if label != "FIT" else "#052e16")
                cell.get_text().set_color("#fed7aa" if label != "FIT" else "#bbf7d0")
                cell.get_text().set_weight("bold")
            if r - 1 >= len(valid) + 4:
                cell.set_facecolor("#0f172a")
                cell.get_text().set_color("#cbd5e1")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SUMMARY_TABLE_0035_STYLE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    records, obs_path, cand_path = query_mast_products()
    if not records:
        print(f"CODE OUTPUT: {VERSION}\n")
        print_table([
            ("Status", "No public spectrum candidates found"),
            ("Observation inventory", obs_path),
            ("Candidate inventory", cand_path),
        ], ["Field", "Value"])
        print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
        print(f"# {VERSION}")
        return

    coverage, nrows, path, status, wave, flux, unit, used_hdus, record = choose_best_spectrum(records)
    uv_df, opt_df = build_line_tables(wave, flux)
    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    avg_rest = float(valid["rest_um"].mean())
    avg_obs = float(valid["raw_peak_sample_um"].mean())
    avg_slope = avg_obs / avg_rest
    avg_z = avg_slope - 1.0
    fit_slope = fit_shared_slope(valid)

    import pandas as pd
    raw_csv = CSV / f"{VERSION}_REAL_RAW_SPECTRUM.csv"
    uv_csv = CSV / f"{VERSION}_UV_LINE_MEASUREMENTS.csv"
    optical_csv = CSV / f"{VERSION}_OPTICAL_HALPHA_NII_AUDIT.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    pd.DataFrame({"wavelength_um_raw": wave, f"flux_raw_{unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
    uv_df.to_csv(uv_csv, index=False)
    opt_df.to_csv(optical_csv, index=False)
    pd.DataFrame([{
        "target": TARGET,
        "reference_z": Z_REF,
        "mast_program": PROGRAM_ID,
        "product": path.name,
        "download_status": status,
        "used_hdus": used_hdus,
        "data_rule": "raw MAST wavelength/flux for spectral plots; no smoothing; no synthetic spectra",
        "style_rule": "JWST_0035 dark background and color palette",
        "uv_lines_measured": int(len(valid)),
        "uv_observed_sum_um": valid["raw_peak_sample_um"].sum(),
        "uv_observed_average_um": avg_obs,
        "uv_average_slope": avg_slope,
        "uv_average_z": avg_z,
        "uv_shared_fit_slope": fit_slope,
        "uv_shared_fit_z": fit_slope - 1.0,
        "optical_triplet_status": "H-alpha/[N II] predicted near 10.1 um at z=14.44; plotted only as out-of-band audit unless real data cover it",
    }]).to_csv(summary_csv, index=False)

    combined_uv = plot_combined_uv(wave, flux, unit, uv_df, avg_obs)
    uv_windows = plot_uv_windows(wave, flux, unit, uv_df)
    optical_audit = plot_optical_audit(opt_df, wave)
    slope_plot = plot_slopes(uv_df, avg_slope, avg_z, fit_slope)
    table_plot = plot_summary_table(uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z, fit_slope)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Target", TARGET),
        ("Reference z", f"{Z_REF:.6f}"),
        ("MAST program", PROGRAM_ID),
        ("Product", path.name),
        ("Style", "JWST_0035 palette: dark background, cyan/pink/purple, orange average, green fit"),
        ("Data rule", "RAW MAST wavelength/flux only for spectra"),
        ("No smoothing", "TRUE"),
        ("No synthetic spectra", "TRUE"),
        ("UV measured lines", len(valid)),
        ("UV observed sum um", f"{valid['raw_peak_sample_um'].sum():.6f}"),
        ("UV observed average um", f"{avg_obs:.6f}"),
        ("UV average slope", f"{avg_slope:.6f}"),
        ("UV average z", f"{avg_z:.6f}"),
        ("Shared fit slope", f"{fit_slope:.6f}"),
        ("Shared fit z", f"{fit_slope - 1.0:.6f}"),
        ("Combined UV plot", str(combined_uv)),
        ("Optical audit plot", str(optical_audit)),
        ("Slope plot", str(slope_plot)),
        ("Summary table PNG", str(table_plot)),
        ("Raw CSV", str(raw_csv)),
        ("UV CSV", str(uv_csv)),
        ("Optical CSV", str(optical_csv)),
    ], ["Field", "Value"])

    print("\nUV LINE WINDOW PLOTS")
    print_table([(i + 1, str(p)) for i, p in enumerate(uv_windows)], ["#", "Path"])

    print("\nUV MEASUREMENTS, SUM, AVERAGE")
    rows = []
    for row in valid.itertuples():
        rows.append((int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.slope_m:.6f}", f"{row.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{valid['rest_um'].sum():.6f}", f"{valid['raw_peak_sample_um'].sum():.6f}", "", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"))
    rows.append(("FIT", "SHARED FIT", "through origin", "", f"{fit_slope:.6f}", f"{fit_slope-1:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw obs um", "Slope m", "z"])

    print("\nOPTICAL H-ALPHA / [N II] STATUS")
    print_table([(int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.expected_at_z14p44_um:.6f}", r.status) for r in opt_df.itertuples()], ["#", "Line", "Rest um", "Pred obs um", "Status"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
