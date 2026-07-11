# JWST_0061
# One figure, two panels only.
# LEFT: real JWST spectrum remapped to the rest frame.
# RIGHT: the same real JWST spectrum in the observed frame.
# Bottom axes show wavelength; top axes show frequency.
# No AI images. Matplotlib only. No network queries.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0061"
GALAXY = "MoM-z14"
Z = 14.44
STRETCH = 1.0 + Z
C_NM_THz = 299792.458

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
REST_COLOR = "#18c7d8"
OBS_COLOR = "#ff9d2e"
POINT_COLOR = "#d9edf7"

REST_WINDOW_NM = (145.0, 195.0)

COMPONENTS = [
    ("N IV] 1483", 148.332),
    ("N IV] 1487", 148.650),
    ("C IV 1548", 154.820),
    ("C IV 1551", 155.077),
    ("He II 1640", 164.042),
    ("O III] 1661", 166.081),
    ("O III] 1666", 166.615),
    ("N III] 1747", 174.682),
    ("N III] 1749", 174.865),
    ("N III] 1750", 174.967),
    ("N III] 1752", 175.216),
    ("N III] 1754", 175.399),
    ("C III] 1907", 190.668),
    ("C III] 1909", 190.873),
]


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def newest(pattern):
    matches = sorted(CSV.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def locate_spectrum():
    exact = newest("JWST_*_MoM-z14_EXACT_JWST.csv")
    if exact is not None:
        return exact, "coordinate-matched JWST extraction"

    for filename in [
        "JWST_0048_REAL_RAW_SPECTRUM.csv",
        "JWST_0047_REAL_RAW_SPECTRUM.csv",
        "JWST_0046_REAL_RAW_SPECTRUM.csv",
    ]:
        candidate = CSV / filename
        if candidate.exists() and candidate.stat().st_size > 100:
            return candidate, "cached JWST/MAST spectrum"

    fallback = newest("JWST_*_REAL_RAW_SPECTRUM.csv")
    if fallback is not None:
        return fallback, "cached JWST/MAST spectrum"

    raise FileNotFoundError(
        "No JWST spectrum CSV was found in /content/JWST_OUTPUT/CSV. "
        "Run an earlier MoM-z14 spectrum script first."
    )


def find_column(frame, choices, prefix=False):
    columns = list(frame.columns)
    lower = {str(column).lower(): column for column in columns}

    for choice in choices:
        if choice.lower() in lower:
            return lower[choice.lower()]

    if prefix:
        for column in columns:
            text = str(column).lower()
            if any(text.startswith(choice.lower()) for choice in choices):
                return column

    return None


def load_spectrum(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    wavelength_column = find_column(
        frame,
        ["wavelength_um", "wavelength", "wave_um", "wave"],
        prefix=True,
    )
    flux_column = find_column(
        frame,
        ["flux", "flux_raw_", "raw_flux"],
        prefix=True,
    )

    if wavelength_column is None or flux_column is None:
        raise RuntimeError(
            f"Could not identify wavelength and flux columns in {path.name}. "
            f"Columns: {list(frame.columns)}"
        )

    wavelength = frame[wavelength_column].to_numpy(float)
    flux = frame[flux_column].to_numpy(float)
    finite = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
    wavelength = wavelength[finite]
    flux = flux[finite]

    median = float(np.nanmedian(wavelength))
    if median > 1000.0:
        observed_nm = wavelength
    elif median > 10.0:
        observed_nm = wavelength
    else:
        observed_nm = wavelength * 1000.0

    order = np.argsort(observed_nm)
    return observed_nm[order], flux[order], str(flux_column)


def frequency_from_wavelength_nm(wavelength_nm):
    import numpy as np

    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    return C_NM_THz / wavelength_nm


def wavelength_nm_from_frequency(frequency_thz):
    import numpy as np

    frequency_thz = np.asarray(frequency_thz, dtype=float)
    return C_NM_THz / frequency_thz


def style_axis(axis):
    axis.set_facecolor(AX_BG)
    axis.grid(True, color=GRID, linewidth=0.55, alpha=0.50)
    axis.tick_params(colors=TEXT, labelsize=8.5)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.85)


def add_component_markers(axis, observed=False):
    for index, (label, rest_nm) in enumerate(COMPONENTS):
        x_value = rest_nm * STRETCH if observed else rest_nm
        axis.axvline(
            x_value,
            color=MUTED,
            linestyle="--",
            linewidth=0.72,
            alpha=0.78,
            zorder=1,
        )
        y_position = 0.965 if index % 2 == 0 else 0.875
        axis.text(
            x_value,
            y_position,
            label,
            transform=axis.get_xaxis_transform(),
            rotation=90,
            ha="right",
            va="top",
            fontsize=5.7,
            color=TEXT,
            alpha=0.94,
        )


def robust_limits(flux):
    import numpy as np

    low = float(np.nanpercentile(flux, 1.0))
    high = float(np.nanpercentile(flux, 99.0))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(flux))
        high = float(np.nanmax(flux))
    padding = 0.08 * (high - low) if high > low else max(abs(high) * 0.1, 1.0e-8)
    return low - padding, high + padding


def build_plot(observed_nm, flux, source_label):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    rest_nm = observed_nm / STRETCH
    mask = (
        np.isfinite(rest_nm)
        & np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (rest_nm >= REST_WINDOW_NM[0])
        & (rest_nm <= REST_WINDOW_NM[1])
    )

    if int(mask.sum()) < 10:
        raise RuntimeError(
            f"Only {int(mask.sum())} samples fall inside the 145–195 nm rest-frame window."
        )

    rest_plot = rest_nm[mask]
    observed_plot = observed_nm[mask]
    flux_plot = flux[mask]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(18.5, 8.2),
        sharey=True,
        facecolor=BG,
    )
    left, right = axes
    style_axis(left)
    style_axis(right)

    left.plot(rest_plot, flux_plot, color=REST_COLOR, linewidth=0.82, alpha=0.94)
    left.scatter(
        rest_plot,
        flux_plot,
        s=13,
        color=POINT_COLOR,
        edgecolor=BG,
        linewidth=0.30,
        zorder=4,
    )

    right.plot(observed_plot, flux_plot, color=OBS_COLOR, linewidth=0.82, alpha=0.94)
    right.scatter(
        observed_plot,
        flux_plot,
        s=13,
        color=POINT_COLOR,
        edgecolor=BG,
        linewidth=0.30,
        zorder=4,
    )

    add_component_markers(left, observed=False)
    add_component_markers(right, observed=True)

    left.set_xlim(REST_WINDOW_NM)
    right.set_xlim(REST_WINDOW_NM[0] * STRETCH, REST_WINDOW_NM[1] * STRETCH)
    y_min, y_max = robust_limits(flux_plot)
    left.set_ylim(y_min, y_max)

    left.set_title("REST FRAME — wavelength compressed by 15.44×", fontsize=12.2, pad=12)
    right.set_title("OBSERVED FRAME — real JWST measurement", fontsize=12.2, pad=12)
    left.set_xlabel("Rest wavelength, nm", fontsize=10)
    right.set_xlabel("Observed wavelength, nm", fontsize=10)
    left.set_ylabel("JWST flux samples", fontsize=10)

    left_top = left.secondary_xaxis(
        "top",
        functions=(frequency_from_wavelength_nm, wavelength_nm_from_frequency),
    )
    left_top.set_xlabel("Rest frequency, THz", color=TEXT, fontsize=9)
    left_top.tick_params(colors=TEXT, labelsize=8)

    right_top = right.secondary_xaxis(
        "top",
        functions=(frequency_from_wavelength_nm, wavelength_nm_from_frequency),
    )
    right_top.set_xlabel("Observed frequency, THz", color=TEXT, fontsize=9)
    right_top.tick_params(colors=TEXT, labelsize=8)

    left.text(
        0.018,
        0.055,
        "Same measured JWST flux values\nOnly the x-axis is transformed",
        transform=left.transAxes,
        ha="left",
        va="bottom",
        color=TEXT,
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#07111f",
            edgecolor=REST_COLOR,
            alpha=0.95,
        ),
    )

    right.text(
        0.018,
        0.055,
        f"λobserved = {STRETCH:.2f} × λrest\nνrest = {STRETCH:.2f} × νobserved",
        transform=right.transAxes,
        ha="left",
        va="bottom",
        color=TEXT,
        fontsize=8.2,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#07111f",
            edgecolor=OBS_COLOR,
            alpha=0.95,
        ),
    )

    figure.suptitle(
        f"{VERSION} — {GALAXY}: REST-FRAME versus OBSERVED JWST SPECTRUM",
        color=TEXT,
        fontsize=16.0,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.938,
        (
            "One real spectrum shown in two coordinate frames. "
            "Rest: 145–195 nm. Observed: "
            f"{REST_WINDOW_NM[0] * STRETCH:.0f}–{REST_WINDOW_NM[1] * STRETCH:.0f} nm."
        ),
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    figure.text(
        0.5,
        0.020,
        f"Source: {source_label}. The matching curve shape is expected because the same photons are being relabeled in the rest and observed frames.",
        ha="center",
        color=MUTED,
        fontsize=8.4,
    )

    figure.subplots_adjust(
        left=0.065,
        right=0.985,
        top=0.875,
        bottom=0.105,
        wspace=0.12,
    )

    png_path = PNG / f"{VERSION}_{GALAXY}_TWO_PANEL_REST_OBSERVED.png"
    figure.savefig(png_path, dpi=250, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(figure)

    data_path = CSV / f"{VERSION}_{GALAXY}_REST_OBSERVED_MAPPING.csv"
    pd.DataFrame(
        {
            "rest_wavelength_nm": rest_plot,
            "rest_frequency_THz": frequency_from_wavelength_nm(rest_plot),
            "observed_wavelength_nm": observed_plot,
            "observed_frequency_THz": frequency_from_wavelength_nm(observed_plot),
            "jwst_flux": flux_plot,
            "stretch_factor": STRETCH,
            "redshift_z": Z,
        }
    ).to_csv(data_path, index=False)

    return png_path, data_path, len(rest_plot)


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    spectrum_path, source_label = locate_spectrum()
    observed_nm, flux, flux_column = load_spectrum(spectrum_path)
    plot_path, data_path, sample_count = build_plot(observed_nm, flux, source_label)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"SOURCE CSV           : {spectrum_path}")
    print(f"SOURCE STATUS        : {source_label}")
    print(f"REST WINDOW          : {REST_WINDOW_NM[0]:.3f} to {REST_WINDOW_NM[1]:.3f} nm")
    print(
        f"OBSERVED WINDOW      : {REST_WINDOW_NM[0] * STRETCH:.3f} "
        f"to {REST_WINDOW_NM[1] * STRETCH:.3f} nm"
    )
    print(f"STRETCH FACTOR       : {STRETCH:.6f}")
    print(f"SAMPLES PLOTTED      : {sample_count}")
    print(f"FLUX COLUMN          : {flux_column}")
    print(f"PLOT PNG             : {plot_path}")
    print(f"MAPPING CSV          : {data_path}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
