# JWST_0044
# Audit: Four clean plots from REAL JWST/MAST x1d data. Raw WAVELENGTH vs raw FLUX only. No smoothing, no continuum subtraction, no synthetic spectra.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0044"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"
HIGH_Z_REFERENCE = 14.44

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen"),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha"),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen"),
]

SCAN_LINES = [
    ("H-beta", 0.486133, 0.45),
    ("[O III] 4959", 0.495891, 0.45),
    ("[O III] 5007", 0.500684, 0.90),
    ("[N II] 6548", 0.654805, 0.45),
    ("H-alpha", 0.656281, 1.00),
    ("[N II] 6583", 0.658345, 0.70),
    ("[S II] 6716", 0.671647, 0.30),
    ("[S II] 6731", 0.673085, 0.30),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "requests"]:
        need(pkg)
    for folder in [PNG, CSV, DATA]:
        folder.mkdir(parents=True, exist_ok=True)


def download_x1d():
    import requests
    local = DATA / PRODUCT
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    r = requests.get(MAST_URL, timeout=180)
    r.raise_for_status()
    local.write_bytes(r.content)
    if local.stat().st_size < 100000:
        raise RuntimeError(f"MAST download too small: {local.stat().st_size} bytes")
    return local, "downloaded-from-mast"


def read_raw_fits(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path) as hdul:
        chosen = None
        hdu_index = None
        columns = []
        flux_unit = "native"
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                chosen = hdu
                hdu_index = idx
                columns = names
                flux_unit = str(hdu.header.get("TUNIT2", "native"))
                break
        if chosen is None:
            raise RuntimeError("No FITS table with WAVELENGTH and FLUX columns found.")
        wave = np.asarray(chosen.data["WAVELENGTH"], dtype=float)
        flux = np.asarray(chosen.data["FLUX"], dtype=float)
    return wave, flux, flux_unit, hdu_index, columns


def finite_xy(wave, flux):
    import numpy as np
    mask = np.isfinite(wave) & np.isfinite(flux)
    w = wave[mask]
    f = flux[mask]
    order = np.argsort(w)
    return w[order], f[order]


def raw_line_comb_seed(wave, flux):
    import numpy as np
    w, f = finite_xy(wave, flux)
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(f)
    normalized_raw = (f - med) / (scale if scale and scale == scale else 1.0)
    z_grid = np.linspace(2.0, 3.0, 2001)
    score = []
    for z in z_grid:
        total = 0.0
        weight_sum = 0.0
        for label, rest, weight in SCAN_LINES:
            x = rest * (1.0 + z)
            if w.min() <= x <= w.max():
                total += weight * np.interp(x, w, normalized_raw)
                weight_sum += weight
        score.append(total / weight_sum if weight_sum else float("nan"))
    score = np.asarray(score)
    z_seed = float(z_grid[int(np.nanargmax(score))])
    return z_grid, score, z_seed


def measure_raw_peaks(wave, flux, z_seed):
    import numpy as np
    import pandas as pd
    w, f = finite_xy(wave, flux)
    rows = []
    for n, label, rest, species in TRIPLET:
        center = rest * (1.0 + z_seed)
        half_width = 0.020
        mask = (w >= center - half_width) & (w <= center + half_width)
        if mask.sum() == 0:
            peak_w = float("nan")
            peak_f = float("nan")
        else:
            ww = w[mask]
            ff = f[mask]
            i = int(np.nanargmax(ff))
            peak_w = float(ww[i])
            peak_f = float(ff[i])
        rows.append({
            "n": n,
            "line": label,
            "species": species,
            "rest_um": rest,
            "raw_search_center_um": center,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux_native": peak_f,
            "slope_m": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1.0 if peak_w == peak_w else float("nan"),
            "high_z_14p44_predicted_um": rest * (1.0 + HIGH_Z_REFERENCE),
        })
    return pd.DataFrame(rows)


def freq_thz_from_um(x):
    return 299.792458 / x


def um_from_freq_thz(x):
    return 299.792458 / x


def set_paper_axis(ax):
    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linewidth=0.45)
    ax.tick_params(colors="0.12", labelsize=9)
    ax.xaxis.label.set_color("0.06")
    ax.yaxis.label.set_color("0.06")
    ax.title.set_color("0.04")
    for spine in ax.spines.values():
        spine.set_color("0.2")
        spine.set_linewidth(0.7)


def plot_single_line(wave, flux, flux_unit, row, plot_index):
    import numpy as np
    import matplotlib.pyplot as plt
    w, f = finite_xy(wave, flux)
    center = row.raw_peak_sample_um
    half_width = 0.050
    mask = (w >= center - half_width) & (w <= center + half_width)
    x = w[mask]
    y = f[mask]
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    set_paper_axis(ax)
    ax.plot(x, y, color="black", linewidth=0.55, label="raw JWST x1d flux samples")
    ax.axvline(center, color="0.05", linewidth=0.90, linestyle="--")
    ax.scatter([center], [row.raw_peak_flux_native], s=36, color="black", zorder=6)
    ax.text(center, row.raw_peak_flux_native, f"  {int(row.n)}", fontsize=13, fontweight="bold", va="bottom", color="black")
    ax.text(0.02, 0.96,
            f"{int(row.n)}  {row.line}\nrest λ = {row.rest_um:.6f} µm\nraw peak λ = {row.raw_peak_sample_um:.6f} µm\nz = {row.z_from_raw_peak:.6f}\nslope m = {row.slope_m:.6f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
            bbox=dict(facecolor="white", edgecolor="0.35", boxstyle="round,pad=0.35"))
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Raw flux, FITS unit: {flux_unit}")
    ax.set_title(f"{VERSION} — plot {plot_index}: raw spectrum around {row.line}")
    fig.tight_layout()
    safe = row.line.replace("[", "").replace("]", "").replace(" ", "_").replace("-", "_")
    path = PNG / f"{VERSION}_PLOT_{plot_index}_{safe}_RAW_SPECTRUM.png"
    fig.savefig(path, dpi=280, facecolor="white")
    plt.show()
    return path


def plot_slopes(trip, avg_slope, avg_z):
    import numpy as np
    import matplotlib.pyplot as plt
    x_min = float(trip["rest_um"].min() - 0.0014)
    x_max = float(trip["rest_um"].max() + 0.0014)
    xs = np.linspace(x_min, x_max, 250)
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    set_paper_axis(ax)
    styles = ["-", "--", ":"]
    for style, row in zip(styles, trip.itertuples()):
        ax.plot(xs, row.slope_m * xs, color="0.12", linewidth=0.70, linestyle=style,
                label=f"{int(row.n)} {row.line}: m={row.slope_m:.6f}")
        ax.scatter([row.rest_um], [row.raw_peak_sample_um], s=35, color="black", zorder=6)
        ax.text(row.rest_um, row.raw_peak_sample_um, f" {int(row.n)}", fontsize=10, fontweight="bold", va="center")
    ax.plot(xs, avg_slope * xs, color="darkorange", linewidth=1.25,
            label=f"average: m={avg_slope:.6f}, z={avg_z:.6f}")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed raw peak wavelength, micron")
    ax.set_title(f"{VERSION} — plot 4: three raw spectral slopes plus average")
    ax.legend(loc="upper left", fontsize=8.2, frameon=True, facecolor="white", edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_PLOT_4_THREE_SLOPES_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor="white")
    plt.show()
    return path


def plot_table_image(trip, avg_rest, avg_obs, avg_slope, avg_z):
    import matplotlib.pyplot as plt
    rows = []
    for row in trip.itertuples():
        rows.append([int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.slope_m:.6f}", f"{row.z_from_raw_peak:.6f}"])
    rows.append(["SUM", "SUM", f"{trip['rest_um'].sum():.6f}", f"{trip['raw_peak_sample_um'].sum():.6f}", "", ""])
    rows.append(["AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"])
    fig, ax = plt.subplots(figsize=(13.5, 4.7))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(f"{VERSION} — raw triplet slopes, sum, and average", fontsize=12.5, pad=10)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "raw peak λ µm", "slope m", "z"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("0.5")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("0.90")
            cell.get_text().set_fontweight("bold")
        elif r >= 4:
            cell.set_facecolor("#fff3e0")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    path = PNG / f"{VERSION}_RAW_SLOPE_TABLE.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for r in rows:
        widths = [max(widths[i], len(str(r[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    fitspath, download_status = download_x1d()
    wave, flux, flux_unit, hdu_index, columns = read_raw_fits(fitspath)
    z_grid, score, z_seed = raw_line_comb_seed(wave, flux)
    trip = measure_raw_peaks(wave, flux, z_seed)
    avg_rest = float(trip["rest_um"].mean())
    avg_obs = float(trip["raw_peak_sample_um"].mean())
    avg_slope = avg_obs / avg_rest
    avg_z = avg_slope - 1.0

    import pandas as pd
    raw_csv = CSV / f"{VERSION}_RAW_FITS_WAVELENGTH_FLUX.csv"
    trip_csv = CSV / f"{VERSION}_RAW_TRIPLET_SLOPES.csv"
    score_csv = CSV / f"{VERSION}_RAW_LINE_COMB_SCORE.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    pd.DataFrame({"wavelength_um_raw": wave, f"flux_raw_{flux_unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
    trip.to_csv(trip_csv, index=False)
    pd.DataFrame({"z": z_grid, "raw_score": score}).to_csv(score_csv, index=False)
    pd.DataFrame([{
        "product": PRODUCT,
        "download_status": download_status,
        "hdu_index": hdu_index,
        "columns": ", ".join(columns),
        "data_plotted": "raw WAVELENGTH column vs raw FLUX column",
        "no_smoothing": True,
        "no_continuum_subtraction": True,
        "no_synthetic_spectrum": True,
        "z_seed_from_raw_line_comb": z_seed,
        "observed_sum_um": trip["raw_peak_sample_um"].sum(),
        "observed_average_um": avg_obs,
        "rest_sum_um": trip["rest_um"].sum(),
        "rest_average_um": avg_rest,
        "average_slope": avg_slope,
        "average_z": avg_z,
        "high_z_reference": HIGH_Z_REFERENCE,
        "high_z_note": "H-alpha/[N II] at z=14.44 would be near 10.1 um and is not present in this NIRSpec x1d product. No fake high-z spectrum is generated.",
    }]).to_csv(summary_csv, index=False)

    plot_paths = []
    for idx, row in enumerate(trip.itertuples(), start=1):
        plot_paths.append(plot_single_line(wave, flux, flux_unit, row, idx))
    slope_path = plot_slopes(trip, avg_slope, avg_z)
    table_path = plot_table_image(trip, avg_rest, avg_obs, avg_slope, avg_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Data source", "REAL JWST/MAST x1d FITS"),
        ("Product", PRODUCT),
        ("Download", download_status),
        ("Plotted data", "raw WAVELENGTH vs raw FLUX"),
        ("No smoothing", "TRUE"),
        ("No continuum subtraction", "TRUE"),
        ("No synthetic spectrum", "TRUE"),
        ("Flux unit", flux_unit),
        ("Raw line-comb seed z", f"{z_seed:.6f}"),
        ("Observed sum um", f"{trip['raw_peak_sample_um'].sum():.6f}"),
        ("Observed average um", f"{avg_obs:.6f}"),
        ("Average slope", f"{avg_slope:.6f}"),
        ("Average z", f"{avg_z:.6f}"),
        ("High-z z=14.44 note", "not plotted as spectrum; H-alpha/[N II] would be near 10.1 um, outside this NIRSpec product"),
        ("Raw CSV", str(raw_csv)),
        ("Triplet CSV", str(trip_csv)),
        ("Table PNG", str(table_path)),
    ], ["Field", "Value"])

    print("\nFOUR REQUESTED PLOTS")
    rows = [(1, "[N II] 6548 raw spectral plot", str(plot_paths[0])),
            (2, "H-alpha raw spectral plot", str(plot_paths[1])),
            (3, "[N II] 6583 raw spectral plot", str(plot_paths[2])),
            (4, "three slopes plus orange average", str(slope_path))]
    print_table(rows, ["#", "Plot", "Path"])

    print("\nRAW PEAKS, SLOPES, SUM, AVERAGE")
    rows = []
    for row in trip.itertuples():
        rows.append((int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.slope_m:.6f}", f"{row.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{trip['rest_um'].sum():.6f}", f"{trip['raw_peak_sample_um'].sum():.6f}", "", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw obs um", "Slope m", "z"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
