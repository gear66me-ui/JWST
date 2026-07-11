# JWST_0041
# Audit: REAL JWST/MAST x1d spectrum only. Raw wavelength/flux plotted directly. No synthetic spectrum. No smoothing. No continuum subtraction.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib

VERSION = "JWST_0041"
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


def download_file():
    import requests
    local = DATA / PRODUCT
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    r = requests.get(MAST_URL, timeout=180)
    r.raise_for_status()
    local.write_bytes(r.content)
    return local, "downloaded-from-mast"


def read_raw_x1d(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path) as hdul:
        table = None
        unit_flux = "native"
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                table = data
                unit_flux = str(hdu.header.get("TUNIT2", "native"))
                break
        if table is None:
            raise RuntimeError("No FITS extension with WAVELENGTH and FLUX columns found.")
        wave = np.asarray(table["WAVELENGTH"], dtype=float)
        flux = np.asarray(table["FLUX"], dtype=float)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[good], flux[good]
    order = np.argsort(wave)
    return wave[order], flux[order], unit_flux


def raw_score_redshift(wave, flux):
    import numpy as np
    med = np.nanmedian(flux)
    mad = np.nanmedian(np.abs(flux - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(flux)
    y = (flux - med) / (scale if scale and scale == scale else 1.0)
    z_grid = np.linspace(2.00, 3.00, 2001)
    scores = []
    for z in z_grid:
        s = 0.0
        wsum = 0.0
        for label, rest, wt in SCAN_LINES:
            x = rest * (1.0 + z)
            if wave.min() <= x <= wave.max():
                s += wt * np.interp(x, wave, y)
                wsum += wt
        scores.append(s / wsum if wsum else float("nan"))
    scores = np.asarray(scores)
    return z_grid, scores, float(z_grid[int(np.nanargmax(scores))])


def measure_triplet_from_raw_samples(wave, flux, z_seed):
    import numpy as np, pandas as pd
    rows = []
    for n, label, rest in TRIPLET:
        pred = rest * (1.0 + z_seed)
        win = 0.020
        m = (wave >= pred - win) & (wave <= pred + win)
        if m.sum() == 0:
            peak_wave, peak_flux = float("nan"), float("nan")
        else:
            ww, ff = wave[m], flux[m]
            i = int(np.nanargmax(ff))
            peak_wave, peak_flux = float(ww[i]), float(ff[i])
        rows.append({
            "n": n,
            "line": label,
            "rest_um": rest,
            "predicted_from_raw_scan_um": pred,
            "raw_peak_sample_um": peak_wave,
            "raw_peak_flux_native": peak_flux,
            "z_from_raw_peak": peak_wave / rest - 1.0 if peak_wave == peak_wave else float("nan"),
        })
    df = pd.DataFrame(rows)
    return df


def freq_thz_from_um(w):
    return 299.792458 / w


def um_from_freq_thz(f):
    return 299.792458 / f


def dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.5, alpha=0.65)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for s in ax.spines.values():
        s.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.2, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_full_raw(wave, flux, unit, triplet_df, avg_obs):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(17.0, 8.8))
    dark(ax)
    ax.plot(wave, flux, linewidth=0.72, color="#e0f2fe", label="RAW MAST x1d flux samples")
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, triplet_df.itertuples()):
        ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.1, alpha=0.88, label=f"{int(row.n)} {row.line}: raw peak {row.raw_peak_sample_um:.6f} um")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.6, linestyle="--", label=f"average of 3 raw peak wavelengths = {avg_obs:.6f} um")
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
    ax.set_title(f"{VERSION} — FULL RAW JWST/MAST spectrum readout, no smoothing, no continuum subtraction")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FULL_RAW_MAST_SPECTRUM.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_raw_zoom(wave, flux, unit, triplet_df, avg_obs):
    import matplotlib.pyplot as plt
    xmin = triplet_df["raw_peak_sample_um"].min() - 0.085
    xmax = triplet_df["raw_peak_sample_um"].max() + 0.085
    m = (wave >= xmin) & (wave <= xmax)
    fig, ax = plt.subplots(figsize=(16.5, 8.2))
    dark(ax)
    ax.plot(wave[m], flux[m], linewidth=0.95, color="#e0f2fe", label="RAW MAST x1d flux samples, zoom")
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, triplet_df.itertuples()):
        ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.25, alpha=0.92, label=f"{int(row.n)} {row.line}")
        ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux_native], s=48, color=color, edgecolor="#f8fafc", zorder=6)
        ax.text(row.raw_peak_sample_um, row.raw_peak_flux_native, f" {int(row.n)}", color="#f8fafc", fontsize=10, weight="bold", va="bottom")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.75, linestyle="--", label=f"average = {avg_obs:.6f} um")
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
    ax.set_title(f"{VERSION} — RAW H-alpha/[N II] region from the actual JWST spectrum")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_RAW_HALPHA_NII_ZOOM_WITH_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_rest_observed_lines(triplet_df, avg_z):
    import numpy as np, matplotlib.pyplot as plt
    x0 = triplet_df["rest_um"].min() - 0.0012
    x1 = triplet_df["rest_um"].max() + 0.0012
    xs = np.linspace(x0, x1, 200)
    fig, ax = plt.subplots(figsize=(15.6, 8.1))
    dark(ax)
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, triplet_df.itertuples()):
        mi = row.raw_peak_sample_um / row.rest_um
        ax.plot(xs, mi * xs, color=color, linewidth=1.05, alpha=0.80, label=f"{int(row.n)} raw slope m={mi:.6f}")
        ax.scatter([row.rest_um], [row.raw_peak_sample_um], s=86, color=color, edgecolor="#f8fafc", zorder=7)
        ax.text(row.rest_um, row.raw_peak_sample_um, f" {int(row.n)}", color="#f8fafc", fontsize=10, weight="bold")
    ax.plot(xs, (1.0 + avg_z) * xs, color="#f97316", linewidth=2.2, label=f"average line: m=1+z_avg={1+avg_z:.6f}")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed wavelength from raw sample peak, micron")
    ax.set_title(f"{VERSION} — line slopes computed from raw sample peak wavelengths")
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_RAW_REST_VS_OBSERVED_SLOPES.png"
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
    fitspath, status = download_file()
    wave, flux, unit = read_raw_x1d(fitspath)
    z_grid, score, z_seed = raw_score_redshift(wave, flux)
    trip = measure_triplet_from_raw_samples(wave, flux, z_seed)
    avg_obs = float(trip["raw_peak_sample_um"].mean())
    avg_rest = float(trip["rest_um"].mean())
    avg_z = avg_obs / avg_rest - 1.0

    raw_csv = CSV / f"{VERSION}_FULL_RAW_MAST_SPECTRUM.csv"
    trip_csv = CSV / f"{VERSION}_RAW_TRIPLET_PEAKS_AVERAGE.csv"
    score_csv = CSV / f"{VERSION}_RAW_LINE_COMB_SCORE.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"

    import pandas as pd
    pd.DataFrame({"wavelength_um": wave, f"flux_raw_{unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
    trip.to_csv(trip_csv, index=False)
    pd.DataFrame({"z": z_grid, "raw_score": score}).to_csv(score_csv, index=False)
    pd.DataFrame([{
        "product": PRODUCT,
        "download_status": status,
        "z_seed_raw_line_comb": z_seed,
        "raw_peak_sum_observed_um": trip["raw_peak_sample_um"].sum(),
        "raw_peak_average_observed_um": avg_obs,
        "rest_sum_um": trip["rest_um"].sum(),
        "rest_average_um": avg_rest,
        "z_from_average_raw_peak_wavelengths": avg_z,
        "flux_unit_native": unit,
        "note": "Spectrum plots use raw FITS wavelength and flux samples. No smoothing, no continuum subtraction, no synthetic peaks."
    }]).to_csv(summary_csv, index=False)

    p1 = plot_full_raw(wave, flux, unit, trip, avg_obs)
    p2 = plot_raw_zoom(wave, flux, unit, trip, avg_obs)
    p3 = plot_rest_observed_lines(trip, avg_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Product", PRODUCT),
        ("Download", status),
        ("Plot data", "RAW WAVELENGTH + RAW FLUX from MAST x1d FITS"),
        ("No smoothing", "TRUE"),
        ("No continuum subtraction", "TRUE"),
        ("No synthetic peaks", "TRUE"),
        ("Flux unit", unit),
        ("Raw line-comb seed z", f"{z_seed:.6f}"),
        ("Observed sum um", f"{trip['raw_peak_sample_um'].sum():.6f}"),
        ("Observed average um", f"{avg_obs:.6f}"),
        ("z from average wavelengths", f"{avg_z:.6f}"),
        ("Full raw plot", str(p1)),
        ("Raw zoom plot", str(p2)),
        ("Raw slope plot", str(p3)),
        ("Raw spectrum CSV", str(raw_csv)),
        ("Triplet CSV", str(trip_csv)),
    ], ["Field", "Value"])

    print("\nRAW SAMPLE PEAKS, SUM, AVERAGE")
    rows = []
    for row in trip.itertuples():
        rows.append((int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.z_from_raw_peak:.6f}"))
    rows.append(("Σ", "SUM", f"{trip['rest_um'].sum():.6f}", f"{trip['raw_peak_sample_um'].sum():.6f}", ""))
    rows.append(("μ", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw peak obs um", "z"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
