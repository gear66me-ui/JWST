# JWST_0056
# MoM-z14 narrow frequency zoom dashboard.
# Five rows x two columns:
#   left  = real JWST samples in observed frequency
#   right = the same samples shifted back to rest-frame frequency
# No AI images. Matplotlib only. No new MAST query.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0056"
GALAXY = "MoM-z14"
PUBLISHED_Z = 14.44
C_UM_THz = 299.792458

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
CYAN = "#18c7d8"
ORANGE = "#ff9d2e"
CURVE = "#d9edf7"
POINTS = "#58bfff"


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def newest(pattern):
    matches = sorted(CSV.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def locate_inputs():
    line_csv = CSV / "JWST_0052_MoM-z14_LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv"
    if not line_csv.exists():
        line_csv = newest("JWST_0052_*LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv")

    raw_csv = None
    for filename in [
        "JWST_0048_REAL_RAW_SPECTRUM.csv",
        "JWST_0047_REAL_RAW_SPECTRUM.csv",
        "JWST_0046_REAL_RAW_SPECTRUM.csv",
    ]:
        candidate = CSV / filename
        if candidate.exists() and candidate.stat().st_size > 100:
            raw_csv = candidate
            break
    if raw_csv is None:
        raw_csv = newest("JWST_*_REAL_RAW_SPECTRUM.csv")

    if line_csv is None or not line_csv.exists():
        raise FileNotFoundError("Run JWST_0052 first; its line-complex CSV is missing.")
    if raw_csv is None or not raw_csv.exists():
        raise FileNotFoundError("No cached real JWST spectrum CSV was found.")
    return line_csv, raw_csv


def find_column(frame, prefix):
    for column in frame.columns:
        if str(column).lower().startswith(prefix.lower()):
            return column
    return None


def load_inputs(line_csv, raw_csv):
    import numpy as np
    import pandas as pd

    lines = pd.read_csv(line_csv)
    required = [
        "n",
        "line_complex",
        "centroid_rest_um",
        "centroid_expected_observed_um",
        "raw_local_peak_um_exploratory",
        "raw_local_peak_flux",
    ]
    missing = [column for column in required if column not in lines.columns]
    if missing:
        raise RuntimeError(f"Line CSV missing columns: {missing}")

    keep = (
        np.isfinite(lines["centroid_rest_um"])
        & np.isfinite(lines["centroid_expected_observed_um"])
        & np.isfinite(lines["raw_local_peak_um_exploratory"])
    )
    lines = lines[keep].copy().sort_values("n").reset_index(drop=True)
    lines["rest_frequency_THz"] = C_UM_THz / lines["centroid_rest_um"]
    lines["observed_peak_frequency_THz"] = C_UM_THz / lines["raw_local_peak_um_exploratory"]
    lines["published_expected_frequency_THz"] = C_UM_THz / lines["centroid_expected_observed_um"]
    lines["stretch_factor"] = lines["rest_frequency_THz"] / lines["observed_peak_frequency_THz"]
    lines["z_pair"] = lines["stretch_factor"] - 1.0

    raw = pd.read_csv(raw_csv)
    wave_col = find_column(raw, "wavelength")
    flux_col = find_column(raw, "flux_raw_")
    if wave_col is None or flux_col is None:
        raise RuntimeError(f"Could not find wavelength/flux columns in {raw_csv.name}")

    wave = raw[wave_col].to_numpy(float)
    flux = raw[flux_col].to_numpy(float)
    finite = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[finite]
    flux = flux[finite]
    order = np.argsort(wave)
    return lines, wave[order], flux[order], str(flux_col)


def narrow_samples(wave, flux, center_um, minimum_points=8):
    import numpy as np

    half_widths = [0.004, 0.006, 0.009, 0.013, 0.018, 0.026, 0.038]
    selected = None
    used_half_width = half_widths[-1]

    for half_width in half_widths:
        mask = (
            np.isfinite(wave)
            & np.isfinite(flux)
            & (wave >= center_um - half_width)
            & (wave <= center_um + half_width)
        )
        selected = mask
        used_half_width = half_width
        if int(mask.sum()) >= minimum_points:
            break

    return wave[selected], flux[selected], used_half_width


def style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.50)
    ax.tick_params(colors=TEXT, labelsize=7.1)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.80)


def dark_legend(ax):
    legend = ax.legend(
        loc="best",
        fontsize=6.0,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.96,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)


def plot_frequency_samples(ax, frequency, flux, cyan_ref, orange_ref, title, xlabel, overlap_note=False):
    import numpy as np

    style_axis(ax)
    order = np.argsort(frequency)
    x = np.asarray(frequency)[order]
    y = np.asarray(flux)[order]

    ax.plot(x, y, color=CURVE, linewidth=0.85, alpha=0.92)
    ax.scatter(
        x,
        y,
        s=20,
        color=POINTS,
        edgecolor=BG,
        linewidth=0.45,
        zorder=5,
        label=f"real JWST samples: {len(x)}",
    )

    ax.axvline(
        cyan_ref,
        color=CYAN,
        linestyle="--",
        linewidth=2.2 if overlap_note else 1.45,
        label="rest-reference position",
        zorder=3,
    )
    ax.axvline(
        orange_ref,
        color=ORANGE,
        linewidth=1.15 if overlap_note else 1.55,
        label="measured-peak position",
        zorder=4,
    )

    ax.set_title(title, fontsize=9.2, pad=6)
    ax.set_xlabel(xlabel, fontsize=7.8)
    ax.set_ylabel("Raw flux", fontsize=7.8)

    if len(x) >= 2:
        xmin = float(np.nanmin(x))
        xmax = float(np.nanmax(x))
        xpad = 0.06 * (xmax - xmin) if xmax > xmin else 0.02
        ax.set_xlim(xmin - xpad, xmax + xpad)

        ymin = float(np.nanmin(y))
        ymax = float(np.nanmax(y))
        ypad = 0.10 * (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.10, 1.0e-8)
        ax.set_ylim(ymin - ypad, ymax + ypad)

    if overlap_note:
        ax.text(
            0.018,
            0.955,
            "The two lines overlap by construction",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color=TEXT,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#07111f",
                edgecolor=CYAN,
                alpha=0.94,
            ),
        )

    dark_legend(ax)


def build_dashboard(lines, wave, flux):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(18.5, 20.5), facecolor=BG)
    sample_frames = []

    for row_index, row in enumerate(lines.itertuples()):
        local_wave, local_flux, half_width = narrow_samples(
            wave,
            flux,
            row.raw_local_peak_um_exploratory,
            minimum_points=8,
        )
        observed_frequency = C_UM_THz / local_wave
        rest_frequency = observed_frequency * row.stretch_factor

        shifted_peak_frequency = row.observed_peak_frequency_THz * row.stretch_factor

        plot_frequency_samples(
            axes[row_index, 0],
            observed_frequency,
            local_flux,
            row.published_expected_frequency_THz,
            row.observed_peak_frequency_THz,
            f"{int(row.n)} {row.line_complex} — observed frame",
            "Observed frequency, THz",
            overlap_note=False,
        )

        plot_frequency_samples(
            axes[row_index, 1],
            rest_frequency,
            local_flux,
            row.rest_frequency_THz,
            shifted_peak_frequency,
            f"{int(row.n)} {row.line_complex} — shifted to rest frame",
            "Rest-frame frequency, THz",
            overlap_note=True,
        )

        axes[row_index, 0].text(
            0.985,
            0.955,
            (
                f"νrest / νobs = {row.stretch_factor:.6f}×\n"
                f"z = {row.z_pair:.6f}\n"
                f"zoom = ±{half_width:.3f} µm"
            ),
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="top",
            fontsize=6.9,
            color=TEXT,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="#07111f",
                edgecolor=ORANGE,
                alpha=0.94,
            ),
        )

        frame = pd.DataFrame(
            {
                "line_number": int(row.n),
                "line_complex": row.line_complex,
                "observed_wavelength_um": local_wave,
                "observed_frequency_THz": observed_frequency,
                "rest_frame_frequency_THz": rest_frequency,
                "raw_flux": local_flux,
                "rest_reference_frequency_THz": row.rest_frequency_THz,
                "observed_peak_frequency_THz": row.observed_peak_frequency_THz,
                "stretch_factor": row.stretch_factor,
                "z_pair": row.z_pair,
                "zoom_half_width_um": half_width,
            }
        )
        sample_frames.append(frame)

    fig.suptitle(
        f"{VERSION} — {GALAXY}: real JWST samples, observed versus shifted-to-rest frequency",
        color=TEXT,
        fontsize=16.0,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.974,
        (
            "Left: measured archive samples in the observed frame.  "
            "Right: the identical samples remapped by νrest = νobserved × (1+z)."
        ),
        color=MUTED,
        fontsize=9.5,
        ha="center",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.955,
        bottom=0.035,
        hspace=0.34,
        wspace=0.16,
    )

    dashboard_path = PNG / f"{VERSION}_{GALAXY}_OBSERVED_VS_REST_FREQUENCY_ZOOM.png"
    fig.savefig(dashboard_path, dpi=235, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    samples = pd.concat(sample_frames, ignore_index=True, sort=False)
    samples_csv = CSV / f"{VERSION}_{GALAXY}_NARROW_FREQUENCY_SAMPLES.csv"
    samples.to_csv(samples_csv, index=False)
    return dashboard_path, samples_csv, samples


def print_table(rows, headers):
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(widths[index], len(str(row[index]))) for index in range(len(headers))]
    print(" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[index]).ljust(widths[index]) for index in range(len(headers))))


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    line_csv, raw_csv = locate_inputs()
    lines, wave, flux, flux_column = load_inputs(line_csv, raw_csv)
    dashboard_path, samples_csv, samples = build_dashboard(lines, wave, flux)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (
                int(row.n),
                row.line_complex,
                f"{row.rest_frequency_THz:.6f}",
                f"{row.observed_peak_frequency_THz:.6f}",
                f"{row.stretch_factor:.6f}",
                f"{row.z_pair:.6f}",
                int((samples["line_number"] == int(row.n)).sum()),
            )
            for row in lines.itertuples()
        ],
        ["#", "Line complex", "Rest THz", "Observed THz", "Ratio x", "z", "Samples"],
    )
    print()
    print_table(
        [
            ("Galaxy", GALAXY),
            ("Dashboard PNG", str(dashboard_path)),
            ("Narrow samples CSV", str(samples_csv)),
            ("Line reference CSV", str(line_csv)),
            ("Real spectrum CSV", str(raw_csv)),
            ("Flux column", flux_column),
            ("Layout", "5 rows x 2 columns; observed left, rest frame right"),
        ],
        ["Field", "Value"],
    )
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
