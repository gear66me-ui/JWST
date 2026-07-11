# JWST_0043
# Audit: REAL JWST/MAST x1d spectrum widget. Raw WAVELENGTH vs raw FLUX only. No smoothing, no continuum subtraction, no synthetic data.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0043"
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
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "requests", "ipywidgets"]:
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
        unit = "native"
        hdu_index = None
        columns = []
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                hdu_used = hdu
                hdu_index = i
                columns = names
                unit = str(hdu.header.get("TUNIT2", "native"))
                break
        if hdu_used is None:
            raise RuntimeError("No FITS table with WAVELENGTH and FLUX columns was found.")
        wave = np.asarray(hdu_used.data["WAVELENGTH"], dtype=float)
        flux = np.asarray(hdu_used.data["FLUX"], dtype=float)
    return wave, flux, unit, hdu_index, columns


def finite_mask(wave, flux):
    import numpy as np
    return np.isfinite(wave) & np.isfinite(flux)


def raw_line_score(wave, flux):
    import numpy as np
    mask = finite_mask(wave, flux)
    w = wave[mask]
    f = flux[mask]
    med = np.nanmedian(f)
    mad = np.nanmedian(np.abs(f - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(f)
    raw_scaled = (f - med) / (scale if scale and scale == scale else 1.0)
    z_grid = np.linspace(2.00, 3.00, 2001)
    score = []
    for z in z_grid:
        s = 0.0
        wsum = 0.0
        for label, rest, wt in SCAN_LINES:
            x = rest * (1.0 + z)
            if w.min() <= x <= w.max():
                s += wt * np.interp(x, w, raw_scaled)
                wsum += wt
        score.append(s / wsum if wsum else float("nan"))
    score = np.asarray(score)
    z_seed = float(z_grid[int(np.nanargmax(score))])
    return z_grid, score, z_seed


def raw_triplet_peaks(wave, flux, z_seed):
    import numpy as np
    import pandas as pd
    mask = finite_mask(wave, flux)
    w = wave[mask]
    f = flux[mask]
    rows = []
    for n, label, rest in TRIPLET:
        center = rest * (1.0 + z_seed)
        win = 0.020
        m = (w >= center - win) & (w <= center + win)
        if m.sum() == 0:
            peak_w = float("nan")
            peak_f = float("nan")
        else:
            ww = w[m]
            ff = f[m]
            idx = int(np.nanargmax(ff))
            peak_w = float(ww[idx])
            peak_f = float(ff[idx])
        rows.append({
            "n": n,
            "line": label,
            "rest_um": rest,
            "raw_search_center_um": center,
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


def plot_raw_static(wave, flux, unit, trip, avg_obs, avg_flux, y_clip=False):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18.0, 9.0))
    dark(ax)
    ax.plot(wave, flux, linewidth=0.62, color="#e0f2fe", label="full raw WAVELENGTH vs FLUX from MAST x1d FITS")
    ax.axhline(avg_flux, color="#fbbf24", linewidth=1.15, alpha=0.85, label=f"mean raw flux = {avg_flux:.6g}")
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for color, row in zip(colors, trip.itertuples()):
        ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.0, alpha=0.86, label=f"{int(row.n)} {row.line}: {row.raw_peak_sample_um:.6f} um")
    ax.axvline(avg_obs, color="#f97316", linewidth=1.65, linestyle="--", label=f"avg of 3 raw peak wavelengths = {avg_obs:.6f} um")
    if y_clip:
        mask = finite_mask(wave, flux)
        lo, hi = np.nanpercentile(flux[mask], [0.5, 99.5])
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
    suffix = "display y-window only" if y_clip else "full native y-range"
    ax.set_title(f"{VERSION} — REAL RAW FULL SPECTRUM, {suffix}; no smoothing / no synthetic data")
    legend(ax, "upper right")
    fig.tight_layout()
    name = f"{VERSION}_FULL_RAW_SPECTRUM_{'YWINDOW' if y_clip else 'NATIVE_Y'}.png"
    path = PNG / name
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def make_widget(wave, flux, unit, trip, avg_obs, avg_flux):
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except Exception as exc:
        print(f"Interactive widget skipped: {exc}")
        return

    mask = finite_mask(wave, flux)
    wmin, wmax = float(np.nanmin(wave[mask])), float(np.nanmax(wave[mask]))
    f_lo, f_hi = np.nanpercentile(flux[mask], [0.5, 99.5])
    f_min, f_max = float(np.nanmin(flux[mask])), float(np.nanmax(flux[mask]))

    x_slider = widgets.FloatRangeSlider(value=[wmin, wmax], min=wmin, max=wmax, step=(wmax-wmin)/1000, description="λ range", layout=widgets.Layout(width="95%"), readout_format=".4f")
    y_mode = widgets.Dropdown(options=[("native full y", "native"), ("visibility y-window", "window")], value="window", description="Y axis")
    show_mean_flux = widgets.Checkbox(value=True, description="mean flux line")
    show_triplet = widgets.Checkbox(value=True, description="triplet raw peak lines")
    show_avg_wave = widgets.Checkbox(value=True, description="avg wavelength line")
    out = widgets.Output()

    def redraw(*args):
        with out:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(17.5, 8.0))
            dark(ax)
            ax.plot(wave, flux, linewidth=0.58, color="#e0f2fe", label="raw FITS flux samples")
            if show_mean_flux.value:
                ax.axhline(avg_flux, color="#fbbf24", linewidth=1.1, label=f"mean raw flux = {avg_flux:.6g}")
            if show_triplet.value:
                colors = ["#38bdf8", "#fb7185", "#a78bfa"]
                for color, row in zip(colors, trip.itertuples()):
                    ax.axvline(row.raw_peak_sample_um, color=color, linewidth=1.05, alpha=0.9, label=f"{int(row.n)} {row.line}")
            if show_avg_wave.value:
                ax.axvline(avg_obs, color="#f97316", linewidth=1.7, linestyle="--", label=f"avg wavelength = {avg_obs:.6f} um")
            ax.set_xlim(x_slider.value[0], x_slider.value[1])
            if y_mode.value == "window":
                ax.set_ylim(float(f_lo), float(f_hi))
            else:
                ax.set_ylim(f_min, f_max)
            top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
            top.set_xlabel("Frequency, THz")
            top.xaxis.label.set_color("#f8fafc")
            top.tick_params(colors="#dbeafe", labelsize=8)
            ax.set_xlabel("Observed wavelength, micron")
            ax.set_ylabel(f"Flux, raw FITS unit: {unit}")
            ax.set_title(f"{VERSION} — RAW JWST/MAST spectrum widget; display controls only, data unchanged")
            legend(ax, "upper right")
            fig.tight_layout()
            plt.show()

    for control in [x_slider, y_mode, show_mean_flux, show_triplet, show_avg_wave]:
        control.observe(redraw, names="value")
    display(widgets.VBox([x_slider, widgets.HBox([y_mode, show_mean_flux, show_triplet, show_avg_wave]), out]))
    redraw()


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
    wave, flux, unit, hdu_index, columns = read_raw_fits(fitspath)
    z_grid, score, z_seed = raw_line_score(wave, flux)
    trip = raw_triplet_peaks(wave, flux, z_seed)
    mask = finite_mask(wave, flux)
    avg_flux = float(__import__("numpy").nanmean(flux[mask]))
    avg_obs = float(trip["raw_peak_sample_um"].mean())
    avg_rest = float(trip["rest_um"].mean())
    avg_z = avg_obs / avg_rest - 1.0

    import pandas as pd
    raw_csv = CSV / f"{VERSION}_FULL_RAW_FITS_READOUT_UNMODIFIED.csv"
    trip_csv = CSV / f"{VERSION}_RAW_PEAKS_AND_AVERAGES.csv"
    score_csv = CSV / f"{VERSION}_RAW_LINE_SCORE.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    pd.DataFrame({"wavelength_um_raw": wave, f"flux_raw_{unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
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
        "no_synthetic_data": True,
        "mean_raw_flux": avg_flux,
        "raw_line_comb_seed_z_for_overlays_only": z_seed,
        "observed_sum_um": trip["raw_peak_sample_um"].sum(),
        "observed_average_um": avg_obs,
        "rest_sum_um": trip["rest_um"].sum(),
        "rest_average_um": avg_rest,
        "z_from_average_raw_peak_wavelengths": avg_z,
    }]).to_csv(summary_csv, index=False)

    p1 = plot_raw_static(wave, flux, unit, trip, avg_obs, avg_flux, y_clip=False)
    p2 = plot_raw_static(wave, flux, unit, trip, avg_obs, avg_flux, y_clip=True)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Product", PRODUCT),
        ("Download", download_status),
        ("Plotted data", "RAW WAVELENGTH vs RAW FLUX"),
        ("No smoothing", "TRUE"),
        ("No continuum subtraction", "TRUE"),
        ("No synthetic data", "TRUE"),
        ("Flux unit", unit),
        ("Mean raw flux", f"{avg_flux:.8g}"),
        ("Observed wavelength sum um", f"{trip['raw_peak_sample_um'].sum():.6f}"),
        ("Observed wavelength average um", f"{avg_obs:.6f}"),
        ("z from average wavelengths", f"{avg_z:.6f}"),
        ("Native full plot", str(p1)),
        ("Visibility y-window plot", str(p2)),
        ("Raw readout CSV", str(raw_csv)),
        ("Raw peak average CSV", str(trip_csv)),
    ], ["Field", "Value"])

    print("\nRAW PEAKS, SUM, AVERAGE")
    rows = []
    for row in trip.itertuples():
        rows.append((int(row.n), row.line, f"{row.rest_um:.6f}", f"{row.raw_peak_sample_um:.6f}", f"{row.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{trip['rest_um'].sum():.6f}", f"{trip['raw_peak_sample_um'].sum():.6f}", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw observed um", "z"])

    print("\nWIDGET")
    print("Interactive raw-spectrum widget below uses display controls only. The plotted data arrays are the raw FITS WAVELENGTH and FLUX columns.")
    make_widget(wave, flux, unit, trip, avg_obs, avg_flux)

    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
