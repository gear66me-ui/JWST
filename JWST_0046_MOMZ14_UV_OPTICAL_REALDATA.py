# JWST_0046
# MoM-z14 real-data workflow: UV emission-line spectrum windows + slope plot + H-alpha/[N II] out-of-band audit.
# No AI images. No synthetic spectra. Raw MAST wavelength/flux samples only for spectral plots.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib, warnings

VERSION = "JWST_0046"
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
        mask |= names.str.contains("x1d") & names.str.endswith(".fits")
        mask |= names.str.contains("s2d") & names.str.endswith(".fits")
    if subgroup:
        mask |= prod_df[subgroup].astype(str).str.upper().isin(["X1D", "S2D"])
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
        raise RuntimeError("No dataURI in MAST product record.")
    Observations.download_file(uri, local_path=str(local))
    if not local.exists() or local.stat().st_size < 100000:
        raise RuntimeError(f"Downloaded file missing or too small: {filename}")
    return local, "downloaded-from-mast"


def read_wavelength_flux(path):
    import numpy as np
    from astropy.io import fits
    waves, fluxes = [], []
    unit = "native"
    used = []
    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                unit = str(hdu.header.get("TUNIT2", unit))
                used.append(f"{i}:{getattr(hdu, 'name', '')}")
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
    m = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[m], flux[m]
    order = np.argsort(wave)
    return wave[order], flux[order], unit, "; ".join(used)


def choose_best_spectrum(records):
    uv_expected = [rest * (1 + Z_REF) for _, _, rest, _ in UV_LINES]
    attempts = []
    for rec in records[:30]:
        try:
            path, status = download_product(rec)
            wave, flux, unit, used = read_wavelength_flux(path)
            if len(wave) < 20:
                continue
            coverage_score = sum((min(wave) <= x <= max(wave)) for x in uv_expected)
            attempts.append((coverage_score, path, status, wave, flux, unit, used, rec))
        except Exception as exc:
            warnings.warn(str(exc))
    if not attempts:
        raise RuntimeError("No readable public spectrum product found. Candidate CSV was saved.")
    attempts.sort(key=lambda x: (x[0], len(x[3])), reverse=True)
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
        peak_w, peak_f, sample_count = peak_sample(wave, flux, expected)
        uv_rows.append({
            "n": n,
            "line": label,
            "species": species,
            "line_group": "rest-UV emission/reference line",
            "rest_um": rest,
            "expected_at_z14p44_um": expected,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux_native": peak_f,
            "sample_count_in_window": sample_count,
            "slope_m": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1 if peak_w == peak_w else float("nan"),
            "status": "raw spectrum window measured" if sample_count else "out of downloaded wavelength range",
        })
    opt_rows = []
    for n, label, rest, species in OPTICAL_LINES:
        expected = rest * (1 + Z_REF)
        covered = bool(min(wave) <= expected <= max(wave))
        peak_w, peak_f, sample_count = peak_sample(wave, flux, expected, 0.080) if covered else (float("nan"), float("nan"), 0)
        opt_rows.append({
            "n": n,
            "line": label,
            "species": species,
            "line_group": "optical H-alpha/[N II] triplet",
            "rest_um": rest,
            "expected_at_z14p44_um": expected,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux_native": peak_f,
            "sample_count_in_window": sample_count,
            "slope_m": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1 if peak_w == peak_w else float("nan"),
            "status": "raw spectrum window measured" if sample_count else "out of NIRSpec/raw product range; not faked",
        })
    return pd.DataFrame(uv_rows), pd.DataFrame(opt_rows)


def freq_thz_from_um(x):
    return 299.792458 / x


def um_from_freq_thz(x):
    return 299.792458 / x


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
    leg = ax.legend(loc=loc, fontsize=7.6, facecolor="#020617", edgecolor="#475569")
    for txt in leg.get_texts():
        txt.set_color("#f8fafc")


def plot_combined_uv(wave, flux, unit, uv_df, avg_uv_obs):
    import matplotlib.pyplot as plt
    valid = uv_df[uv_df["sample_count_in_window"] > 0]
    if valid.empty:
        return None
    xmin = max(float(min(wave)), float(valid["expected_at_z14p44_um"].min() - 0.16))
    xmax = min(float(max(wave)), float(valid["expected_at_z14p44_um"].max() + 0.16))
    m = (wave >= xmin) & (wave <= xmax)
    fig, ax = plt.subplots(figsize=(16.4, 8.0))
    style_dark(ax)
    ax.plot(wave[m], flux[m], color="#e0f2fe", linewidth=0.62, label="raw JWST/MAST spectrum samples")
    for row in valid.itertuples():
        ax.axvline(row.expected_at_z14p44_um, color="#64748b", linestyle=":", linewidth=0.75)
        ax.axvline(row.raw_peak_sample_um, color="#f8fafc", linestyle="--", linewidth=0.70)
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux_native], s=34, color="#f8fafc", edgecolor="#020617", zorder=5)
        ax.text(row.raw_peak_sample_um, row.raw_peak_flux_native, f" {int(row.n)}", color="#f8fafc", fontsize=9.5, fontweight="bold", va="bottom")
    ax.axvline(avg_uv_obs, color="#f97316", linewidth=1.35, label=f"UV average raw peak λ = {avg_uv_obs:.6f} µm")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Raw flux, FITS unit: {unit}")
    ax.set_title(f"{VERSION} — {TARGET}: combined real UV-line spectrum windows at z≈14.44")
    legend_dark(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMBINED_REAL_UV_LINES_RAW_SPECTRUM.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_uv_line_windows(wave, flux, unit, uv_df):
    import matplotlib.pyplot as plt
    paths = []
    for row in uv_df.itertuples():
        if row.sample_count_in_window <= 0:
            continue
        c = row.expected_at_z14p44_um
        m = (wave >= c - 0.095) & (wave <= c + 0.095)
        fig, ax = plt.subplots(figsize=(12.6, 5.4))
        style_light(ax)
        ax.plot(wave[m], flux[m], color="black", linewidth=0.55, label="raw spectrum")
        ax.axvline(row.expected_at_z14p44_um, color="0.55", linestyle=":", linewidth=0.80, label="expected at z=14.44")
        ax.axvline(row.raw_peak_sample_um, color="black", linestyle="--", linewidth=0.85, label="raw local peak sample")
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux_native], s=30, color="black", zorder=5)
        ax.text(row.raw_peak_sample_um, row.raw_peak_flux_native, f"  {int(row.n)}", fontsize=12, fontweight="bold", va="bottom")
        ax.text(0.02, 0.95, f"{int(row.n)} {row.line}\nrest λ={row.rest_um:.6f} µm\nraw peak λ={row.raw_peak_sample_um:.6f} µm\nz={row.z_from_raw_peak:.6f}", transform=ax.transAxes, ha="left", va="top", fontsize=8.8, bbox=dict(facecolor="white", edgecolor="0.45", boxstyle="round,pad=0.32"))
        top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
        top.set_xlabel("Frequency, THz")
        ax.set_xlabel("Observed wavelength, micron")
        ax.set_ylabel(f"Raw flux, FITS unit: {unit}")
        ax.set_title(f"{VERSION} — real raw UV line window: {row.line}")
        ax.legend(loc="best", fontsize=7.8, frameon=True, edgecolor="0.55")
        fig.tight_layout()
        safe = row.line.replace("[", "").replace("]", "").replace(" ", "_").replace("-", "_")
        path = PNG / f"{VERSION}_UV_LINE_{int(row.n)}_{safe}_RAW_WINDOW.png"
        fig.savefig(path, dpi=260, facecolor="white")
        plt.show()
        paths.append(path)
    return paths


def plot_optical_audit(opt_df, wave):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.8, 5.6))
    style_light(ax)
    y = list(range(len(opt_df)))
    ax.scatter(opt_df["expected_at_z14p44_um"], y, s=44, color="black", zorder=5, label="predicted observed λ at z=14.44")
    for yi, row in zip(y, opt_df.itertuples()):
        ax.text(row.expected_at_z14p44_um, yi + 0.12, f"{row.line}\n{row.expected_at_z14p44_um:.3f} µm", ha="center", va="bottom", fontsize=8.5)
    ax.axvspan(float(min(wave)), float(max(wave)), color="#e0f2fe", alpha=0.55, label="downloaded real spectrum wavelength coverage")
    ax.set_yticks(y)
    ax.set_yticklabels(opt_df["line"].tolist())
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_title(f"{VERSION} — H-alpha/[N II] at z=14.44: wavelength positions only, no fake spectrum")
    ax.legend(loc="upper left", fontsize=8.0, frameon=True, edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_OPTICAL_HALPHA_NII_OUT_OF_BAND_AUDIT.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def plot_slopes(meas_df, avg_slope, avg_z):
    import numpy as np, matplotlib.pyplot as plt
    valid = meas_df[meas_df["sample_count_in_window"] > 0].copy()
    xs = np.linspace(valid["rest_um"].min() - 0.004, valid["rest_um"].max() + 0.004, 280)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    style_light(ax)
    styles = ["-", "--", ":", "-.", (0, (5, 1))]
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


def plot_summary_table(uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z):
    import matplotlib.pyplot as plt
    rows = []
    for r in uv_df.itertuples():
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}" if r.raw_peak_sample_um == r.raw_peak_sample_um else "OUT", f"{r.slope_m:.6f}" if r.slope_m == r.slope_m else "", f"{r.z_from_raw_peak:.6f}" if r.z_from_raw_peak == r.z_from_raw_peak else ""])
    rows.append(["UV SUM", "SUM", f"{uv_df['rest_um'].sum():.6f}", f"{uv_df['raw_peak_sample_um'].sum():.6f}", "", ""])
    rows.append(["UV AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"])
    for r in opt_df.itertuples():
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", f"pred {r.expected_at_z14p44_um:.6f}", "out-of-band", "not measured"])
    fig, ax = plt.subplots(figsize=(15.0, 6.4))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(f"{VERSION} — UV measurements + optical H-alpha/[N II] out-of-band audit", fontsize=12.5, pad=10)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "obs/raw λ µm", "slope m", "z"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.25)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("0.55")
        cell.set_linewidth(0.42)
        if r == 0:
            cell.set_facecolor("0.90")
            cell.get_text().set_fontweight("bold")
        elif 6 <= r <= 7:
            cell.set_facecolor("#fff3e0")
            cell.get_text().set_fontweight("bold")
        elif r >= 8:
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
        print(f"CODE OUTPUT: {VERSION}\n")
        print(f"No public spectrum candidates found for JWST Program {PROGRAM_ID}.")
        print(f"Observation inventory: {obs_path}")
        print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
        print(f"# {VERSION}")
        return

    coverage_score, path, status, wave, flux, unit, used_hdus, record = choose_best_spectrum(records)
    uv_df, opt_df = build_line_tables(wave, flux)
    valid_uv = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    avg_rest = float(valid_uv["rest_um"].mean())
    avg_obs = float(valid_uv["raw_peak_sample_um"].mean())
    avg_slope = avg_obs / avg_rest
    avg_z = avg_slope - 1.0

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
        "data_rule": "raw MAST wavelength vs flux only for spectral plots; no smoothing; no synthetic spectra",
        "uv_lines_measured": int(len(valid_uv)),
        "uv_average_rest_um": avg_rest,
        "uv_average_observed_um": avg_obs,
        "uv_average_slope": avg_slope,
        "uv_average_z": avg_z,
        "optical_triplet_status": "H-alpha/[N II] predicted near 10.1 um at z=14.44; plotted only as out-of-band audit unless real data cover it",
    }]).to_csv(summary_csv, index=False)

    combined_uv = plot_combined_uv(wave, flux, unit, uv_df, avg_obs)
    uv_windows = plot_uv_line_windows(wave, flux, unit, uv_df)
    optical_audit = plot_optical_audit(opt_df, wave)
    slope_plot = plot_slopes(uv_df, avg_slope, avg_z)
    table_plot = plot_summary_table(uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Target", TARGET),
        ("Reference z", f"{Z_REF:.6f}"),
        ("MAST program", PROGRAM_ID),
        ("Product", path.name),
        ("Data rule", "RAW MAST wavelength/flux only for spectra"),
        ("No smoothing", "TRUE"),
        ("No synthetic spectra", "TRUE"),
        ("UV measured lines", len(valid_uv)),
        ("UV observed sum um", f"{valid_uv['raw_peak_sample_um'].sum():.6f}"),
        ("UV observed average um", f"{avg_obs:.6f}"),
        ("UV average slope", f"{avg_slope:.6f}"),
        ("UV average z", f"{avg_z:.6f}"),
        ("Combined UV plot", str(combined_uv)),
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
    for r in valid_uv.itertuples():
        rows.append((int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}", f"{r.slope_m:.6f}", f"{r.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{valid_uv['rest_um'].sum():.6f}", f"{valid_uv['raw_peak_sample_um'].sum():.6f}", "", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw obs um", "Slope m", "z"])

    print("\nOPTICAL H-ALPHA / [N II] STATUS")
    print_table([(int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.expected_at_z14p44_um:.6f}", r.status) for r in opt_df.itertuples()], ["#", "Line", "Rest um", "Pred obs um", "Status"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
