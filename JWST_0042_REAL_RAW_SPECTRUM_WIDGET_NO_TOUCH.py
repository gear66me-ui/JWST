# JWST_0042
# Audit: REAL JWST/MAST x1d spectrum only. Raw wavelength/flux plotted directly. No synthetic peaks. No smoothing. No continuum subtraction. No line fitting.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0042"
PROJECT = "REAL RAW JWST FULL SPECTRUM WIDGET NO TOUCH"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "requests", "ipywidgets"]:
        need(pkg)
    for folder in [PNG, CSV, DATA]:
        folder.mkdir(parents=True, exist_ok=True)


def download_raw_fits():
    import requests
    local = DATA / PRODUCT
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    r = requests.get(MAST_URL, timeout=180)
    r.raise_for_status()
    local.write_bytes(r.content)
    return local, "downloaded-from-mast"


def read_raw_table(fits_path):
    import numpy as np
    import pandas as pd
    from astropy.io import fits
    with fits.open(fits_path) as hdul:
        table = None
        header = None
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                table = data
                header = hdu.header
                break
        if table is None:
            raise RuntimeError("No FITS table extension with WAVELENGTH and FLUX columns found.")
        names = list(table.names)
        raw = {}
        for name in names:
            try:
                arr = np.asarray(table[name])
                if arr.ndim == 1:
                    raw[name] = arr
            except Exception:
                pass
        df = pd.DataFrame(raw)
        flux_unit = str(header.get("TUNIT2", "native")) if header is not None else "native"
        wave_unit = str(header.get("TUNIT1", "native")) if header is not None else "native"
    return df, wave_unit, flux_unit


def finite_plot_arrays(df):
    import numpy as np
    wave = np.asarray(df["WAVELENGTH"], dtype=float)
    flux = np.asarray(df["FLUX"], dtype=float)
    finite = np.isfinite(wave) & np.isfinite(flux)
    order = np.argsort(wave[finite])
    return wave[finite][order], flux[finite][order], int((~finite).sum())


def freq_thz_from_um(w_um):
    return 299.792458 / w_um


def um_from_freq_thz(f_thz):
    return 299.792458 / f_thz


def dark_axis(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.50, alpha=0.62)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.3, facecolor="#020617", edgecolor="#475569")
    for text in leg.get_texts():
        text.set_color("#f8fafc")


def plot_raw_full(wave, flux, flux_unit, mean_flux):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(17.5, 9.0))
    dark_axis(ax)
    ax.plot(wave, flux, linewidth=0.65, color="#dbeafe", label="raw FITS samples: WAVELENGTH vs FLUX")
    ax.axhline(mean_flux, color="#f97316", linewidth=1.45, linestyle="--", label=f"raw arithmetic mean flux = {mean_flux:.6e}")
    ax.set_xlabel("Observed wavelength, micron / native FITS WAVELENGTH")
    ax.set_ylabel(f"Raw FLUX, native FITS unit: {flux_unit}")
    ax.set_title(f"{VERSION} — full raw JWST/MAST x1d spectrum, unmodified data")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FULL_RAW_SPECTRUM_NO_TOUCH.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_raw_full_points(wave, flux, flux_unit, mean_flux):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(17.5, 9.0))
    dark_axis(ax)
    ax.plot(wave, flux, linewidth=0.38, color="#93c5fd", alpha=0.75, label="raw line connection")
    ax.scatter(wave, flux, s=3.2, color="#e0f2fe", alpha=0.72, label="raw samples")
    ax.axhline(mean_flux, color="#f97316", linewidth=1.45, linestyle="--", label=f"raw arithmetic mean flux = {mean_flux:.6e}")
    ax.set_xlabel("Observed wavelength, micron / native FITS WAVELENGTH")
    ax.set_ylabel(f"Raw FLUX, native FITS unit: {flux_unit}")
    ax.set_title(f"{VERSION} — full raw spectrum with every finite sample point shown")
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    top.xaxis.label.set_color("#f8fafc")
    top.tick_params(colors="#dbeafe", labelsize=8)
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FULL_RAW_SPECTRUM_POINTS_NO_TOUCH.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def create_colab_widget(wave, flux, flux_unit, mean_flux):
    try:
        import matplotlib.pyplot as plt
        from IPython.display import display, Markdown
        import ipywidgets as widgets
        wmin = float(wave.min())
        wmax = float(wave.max())
        slider = widgets.FloatRangeSlider(
            value=[wmin, wmax],
            min=wmin,
            max=wmax,
            step=(wmax - wmin) / 500.0,
            description="λ range",
            continuous_update=False,
            layout=widgets.Layout(width="95%"),
            readout_format=".4f",
        )
        display(Markdown("### RAW JWST spectrum widget — full data, no smoothing, no synthetic peaks"))
        def draw(lam_range):
            lo, hi = lam_range
            mask = (wave >= lo) & (wave <= hi)
            fig, ax = plt.subplots(figsize=(15.8, 7.2))
            dark_axis(ax)
            ax.plot(wave[mask], flux[mask], linewidth=0.65, color="#dbeafe", label="raw FITS samples")
            ax.axhline(mean_flux, color="#f97316", linewidth=1.25, linestyle="--", label="full-spectrum raw mean flux")
            ax.set_xlabel("Observed wavelength, micron")
            ax.set_ylabel(f"Raw FLUX, native FITS unit: {flux_unit}")
            ax.set_title(f"{VERSION} raw zoom widget: {lo:.4f} to {hi:.4f} micron")
            legend(ax, "upper right")
            fig.tight_layout()
            plt.show()
        widgets.interact(draw, lam_range=slider)
        return "widget-displayed"
    except Exception as exc:
        return f"widget-skipped: {exc}"


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
    fits_path, download_status = download_raw_fits()
    df, wave_unit, flux_unit = read_raw_table(fits_path)
    raw_csv = CSV / f"{VERSION}_FULL_RAW_X1D_TABLE.csv"
    df.to_csv(raw_csv, index=False)
    wave, flux, nonfinite_count = finite_plot_arrays(df)
    import numpy as np
    mean_flux = float(np.nanmean(flux))
    mean_wave = float(np.nanmean(wave))
    summary_csv = CSV / f"{VERSION}_RAW_SUMMARY.csv"
    import pandas as pd
    pd.DataFrame([{
        "product": PRODUCT,
        "download_status": download_status,
        "row_count_raw_csv": len(df),
        "finite_plot_sample_count": len(wave),
        "nonfinite_excluded_from_plot_count": nonfinite_count,
        "wavelength_min": float(wave.min()),
        "wavelength_max": float(wave.max()),
        "raw_mean_wavelength": mean_wave,
        "raw_mean_flux": mean_flux,
        "wave_unit_native": wave_unit,
        "flux_unit_native": flux_unit,
        "processing": "none: no smoothing, no continuum subtraction, no peak finding, no redshift fitting, no synthetic data",
    }]).to_csv(summary_csv, index=False)

    p1 = plot_raw_full(wave, flux, flux_unit, mean_flux)
    p2 = plot_raw_full_points(wave, flux, flux_unit, mean_flux)
    widget_status = create_colab_widget(wave, flux, flux_unit, mean_flux)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("MAST product", PRODUCT),
        ("Download status", download_status),
        ("Data plotted", "raw WAVELENGTH and raw FLUX columns from FITS x1d table"),
        ("Smoothing", "NO"),
        ("Continuum subtraction", "NO"),
        ("Peak finding", "NO"),
        ("Synthetic data", "NO"),
        ("Finite plotted samples", len(wave)),
        ("Nonfinite excluded from plot", nonfinite_count),
        ("Raw mean flux line", f"{mean_flux:.9e}"),
        ("Raw full CSV", str(raw_csv)),
        ("Summary CSV", str(summary_csv)),
        ("Full raw PNG", str(p1)),
        ("Full raw points PNG", str(p2)),
        ("Widget", widget_status),
    ], ["Field", "Value"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
