# JWST_0042
# Audit: REAL JWST/MAST x1d spectrum. Full raw spectrum readout. No smoothing. No continuum subtraction. No synthetic peaks.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib

VERSION = "JWST_0042"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"

TRIPLET = [
    (1, "[N II] 6548", 0.654805),
    (2, "H-alpha 6563", 0.656281),
    (3, "[N II] 6583", 0.658345),
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
    for p in [PNG, CSV, DATA]:
        p.mkdir(parents=True, exist_ok=True)


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
        hdu_used = None
        names_used = None
        unit = "native"
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                hdu_used = hdu
                names_used = names
                unit = str(hdu.header.get("TUNIT2", "native"))
                break
        if hdu_used is None:
            raise RuntimeError("No FITS table with WAVELENGTH and FLUX columns was found.")
        wave = np.asarray(hdu_used.data["WAVELENGTH"], dtype=float)
        flux = np.asarray(hdu_used.data["FLUX"], dtype=float)
    return wave, flux, unit, names_used


def finite_xy(wave, flux):
    import numpy as np
    mask = np.isfinite(wave) & np.isfinite(flux)
    return wave[mask], flux[mask]


def raw_redshift_seed(wave, flux):
    import numpy as np
    w, f = finite_xy(wave, flux)
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(f)
    y = (f - med) / (scale if scale and scale == scale else 1.0)
    z_grid = np.linspace(2.00, 3.00, 2001)
    score = []
    for z in z_grid:
        s = 0.0
        wsum = 0.0
        for label, rest, wt in SCAN_LINES:
            x = rest * (1.0 + z)
            if w.min() <= x <= w.max():
                s += wt * np.interp(x, w, y)
                wsum += wt
        score.append(s / wsum if wsum else float("nan"))
    score = np.asarray(score)
    z = float(z_grid[int(np.nanargmax(score))])
    return z_grid, score, z


def raw_triplet_peaks(wave, flux, z_seed):
    import numpy as np, pandas as pd
    w, f = finite_xy(wave, flux)
    rows = []
    for n, label, rest in TRIPLET:
        pred = rest * (1.0 + z_seed)
        window = 0.020
        mask = (w >= pred - window) & (w <= pred + window)
        if mask.sum() == 0:
            peak_w = float("nan")
            peak_f = float("nan")
        else:
            ww = w[mask]
            ff = f[mask]
            idx = int(np.nanargmax(ff))
            peak_w = float(ww[idx])
            peak_f = float(ff[idx])
        rows.append({
            "n": n,
            "line": label,
            "rest_um": rest,
            "predicted_center_from_raw_score_um": pred,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux_native": peak_f,
            "z_from_raw_peak": peak_w / rest - 1.0 if peak_w == peak_w else float("nan"),
        })
    return pd.DataFrame(rows)


def freq_thz_from_um(x):
    return 299.792458 / x


def um_from_freq_thz(x):
    return 299.792458 / x


def dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.45, alpha=0.65)
    ax.tick_params(colors="#dbeafe", labelsize=8.5)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for s in ax.spines.values():
        s.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.0, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_full_raw(wave, flux, unit, trip, avg_obs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18.0, 9.0))
    dark(ax)
    ax.plot(wave, flux, linewidth=0.62, color="#e0f2fe", label="full raw MAST x1d readout: WAVELENGTH vs FLUX")
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, trip.itertuples()):
        ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.05, alpha=0.90, label=f"{int(row.n)} {row.line}: {row.raw_peak_sample_um:.6f} um")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.65, linestyle="--", label=f"average of 3 raw peak wavelengths = {avg_obs:.6f} um")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
    ax.set_title(f"{VERSION} — FULL RAW JWST/MAST SPECTRUM: no smoothing, no continuum subtraction, no synthetic data")
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FULL_RAW_SPECTRUM_WITH_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_full_raw_display_zoom(wave, flux, unit, trip, avg_obs):
    import numpy as np, matplotlib.pyplot as plt
    w, f = finite_xy(wave, flux)
    lo, hi = np.nanpercentile(f, [0.5, 99.5])
    pad = 0.06 * (hi - lo)
    fig, ax = plt.subplots(figsize=(18.0, 9.0))
    dark(ax)
    ax.plot(wave, flux, linewidth=0.62, color="#e0f2fe", label="same raw data; display y-axis clipped for visibility only")
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, trip.itertuples()):
        ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.05, alpha=0.90, label=f"{int(row.n)} {row.line}")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.65, linestyle="--", label=f"average = {avg_obs:.6f} um")
    ax.set_ylim(lo - pad, hi + pad)
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
    ax.set_title(f"{VERSION} — FULL RAW SPECTRUM, visibility y-window; data unchanged")
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FULL_RAW_SPECTRUM_DISPLAY_YWINDOW.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_raw_triplet_zoom(wave, flux, unit, trip, avg_obs):
    import matplotlib.pyplot as plt
    xmin = trip["raw_peak_sample_um"].min() - 0.090
    xmax = trip["raw_peak_sample_um"].max() + 0.090
    mask = (wave >= xmin) & (wave <= xmax)
    fig, ax = plt.subplots(figsize=(17.0, 8.4))
    dark(ax)
    ax.plot(wave[mask], flux[mask], linewidth=0.90, color="#e0f2fe", label="raw MAST x1d readout, triplet region")
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, trip.itertuples()):
        ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.25, alpha=0.93, label=f"{int(row.n)} {row.line}")
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux_native], s=48, color=color, edgecolor="#f8fafc", zorder=8)
    ax.axvline(avg_obs, color="#f97316", linewidth=1.75, linestyle="--", label=f"average = {avg_obs:.6f} um")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
    ax.set_title(f"{VERSION} — raw H-alpha/[N II] region with average line")
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_RAW_TRIPLET_REGION_WITH_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
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
    wave, flux, unit, columns = read_raw_fits(fitspath)
    z_grid, score, z_seed = raw_redshift_seed(wave, flux)
    trip = raw_triplet_peaks(wave, flux, z_seed)
    avg_obs = float(trip["raw_peak_sample_um"].mean())
    avg_rest = float(trip["rest_um"].mean())
    avg_z = avg_obs / avg_rest - 1.0

    import pandas as pd
    raw_csv = CSV / f"{VERSION}_FULL_RAW_FITS_READOUT.csv"
    trip_csv = CSV / f"{VERSION}_RAW_PEAKS_SUM_AVERAGE.csv"
    score_csv = CSV / f"{VERSION}_RAW_LINE_SCORE.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    pd.DataFrame({"wavelength_um_raw": wave, f"flux_raw_{unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
    trip.to_csv(trip_csv, index=False)
    pd.DataFrame({"z": z_grid, "raw_score": score}).to_csv(score_csv, index=False)
    pd.DataFrame([{
        "product": PRODUCT,
        "download_status": download_status,
        "raw_fits_columns": ", ".join(columns),
        "no_smoothing": True,
        "no_continuum_subtraction": True,
        "no_synthetic_spectrum": True,
        "raw_line_comb_seed_z": z_seed,
        "observed_sum_um": trip["raw_peak_sample_um"].sum(),
        "observed_average_um": avg_obs,
        "rest_sum_um": trip["rest_um"].sum(),
        "rest_average_um": avg_rest,
        "z_from_average_raw_peak_wavelengths": avg_z,
        "flux_unit_native": unit,
    }]).to_csv(summary_csv, index=False)

    p1 = plot_full_raw(wave, flux, unit, trip, avg_obs)
    p2 = plot_full_raw_display_zoom(wave, flux, unit, trip, avg_obs)
    p3 = plot_raw_triplet_zoom(wave, flux, unit, trip, avg_obs)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Product", PRODUCT),
        ("Download", download_status),
        ("Data plotted", "FULL RAW FITS WAVELENGTH + FLUX"),
        ("No smoothing", "TRUE"),
        ("No continuum subtraction", "TRUE"),
        ("No synthetic spectrum", "TRUE"),
        ("Flux unit", unit),
        ("Raw line-comb seed z", f"{z_seed:.6f}"),
        ("Observed sum um", f"{trip['raw_peak_sample_um'].sum():.6f}"),
        ("Observed average um", f"{avg_obs:.6f}"),
        ("z from average wavelengths", f"{avg_z:.6f}"),
        ("Full raw plot", str(p1)),
        ("Full raw y-window plot", str(p2)),
        ("Raw triplet zoom", str(p3)),
        ("Raw readout CSV", str(raw_csv)),
        ("Triplet average CSV", str(trip_csv)),
    ], ["Field", "Value"])

    print("\nRAW PEAKS, SUM, AVERAGE")
    rows = []
    for row in trip.itertuples():
        rows.append((int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{trip['rest_um'].sum():.6f}", f"{trip['raw_peak_sample_um'].sum():.6f}", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw observed um", "z"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
