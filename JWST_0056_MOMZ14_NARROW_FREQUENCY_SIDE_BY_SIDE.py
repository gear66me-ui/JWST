# JWST_0056
# Narrow side-by-side frequency views for MoM-z14.
# Left panel: observed-frame frequency samples from real JWST/MAST spectrum.
# Right panel: same measured flux samples shifted to the rest frame using each pair-derived z.
# No AI images. Matplotlib only. No new MAST query.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0056"
GALAXY = "MoM-z14"
C_UM_THz = 299.792458
N_SAMPLES = 25
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
POINT_COLORS = ["#43b9ff", "#ff6b76", "#a78bfa", "#34d399", "#f6c453"]


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
        raise FileNotFoundError("No cached real JWST/MAST spectrum CSV was found.")
    return line_csv, raw_csv


def find_column(frame, prefix):
    for column in frame.columns:
        if str(column).lower().startswith(prefix.lower()):
            return column
    return None


def load_data(line_csv, raw_csv):
    import numpy as np
    import pandas as pd

    lines = pd.read_csv(line_csv)
    required = [
        "n",
        "line_complex",
        "centroid_rest_um",
        "raw_local_peak_um_exploratory",
        "raw_local_peak_flux",
        "window_low_um",
        "window_high_um",
    ]
    missing = [column for column in required if column not in lines.columns]
    if missing:
        raise RuntimeError(f"Line CSV missing columns: {missing}")

    lines = lines[
        np.isfinite(lines["centroid_rest_um"])
        & np.isfinite(lines["raw_local_peak_um_exploratory"])
    ].copy()
    lines = lines.sort_values("n").reset_index(drop=True)
    lines["stretch_factor"] = lines["raw_local_peak_um_exploratory"] / lines["centroid_rest_um"]
    lines["z_pair"] = lines["stretch_factor"] - 1.0
    lines["observed_peak_frequency_THz"] = C_UM_THz / lines["raw_local_peak_um_exploratory"]
    lines["rest_frequency_THz"] = C_UM_THz / lines["centroid_rest_um"]

    raw = pd.read_csv(raw_csv)
    wave_col = find_column(raw, "wavelength")
    flux_col = find_column(raw, "flux_raw_")
    if wave_col is None or flux_col is None:
        raise RuntimeError(f"Could not find wavelength and flux columns in {raw_csv.name}")

    wave = raw[wave_col].to_numpy(float)
    flux = raw[flux_col].to_numpy(float)
    finite = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[finite]
    flux = flux[finite]
    order = np.argsort(wave)
    return lines, wave[order], flux[order], str(flux_col)


def narrow_samples(row, wave, flux):
    import numpy as np
    import pandas as pd

    valid = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave >= row.window_low_um)
        & (wave <= row.window_high_um)
    )
    indices = np.flatnonzero(valid)
    if indices.size < 3:
        raise RuntimeError(f"Too few finite samples for {row.line_complex}")

    local_wave = wave[indices]
    center_local = int(np.argmin(np.abs(local_wave - row.raw_local_peak_um_exploratory)))
    half = N_SAMPLES // 2
    start = max(0, center_local - half)
    stop = min(indices.size, start + N_SAMPLES)
    start = max(0, stop - N_SAMPLES)
    chosen = indices[start:stop]

    observed_frequency = C_UM_THz / wave[chosen]
    rest_frequency = observed_frequency * row.stretch_factor
    order = np.argsort(observed_frequency)

    return pd.DataFrame(
        {
            "sample_index": np.arange(len(chosen)),
            "observed_wavelength_um": wave[chosen][order],
            "observed_frequency_THz": observed_frequency[order],
            "rest_frame_frequency_THz": rest_frequency[order],
            "raw_flux": flux[chosen][order],
        }
    )


def style_axis(ax, small=False):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.42 if small else 0.55, alpha=0.48)
    ax.tick_params(colors=TEXT, labelsize=7.0 if small else 8.8)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.80)


def draw_pair(ax_obs, ax_rest, row, samples, color, small=False):
    point_size = 20 if small else 32
    line_width = 0.72 if small else 0.88
    title_size = 8.5 if small else 10.8
    label_size = 7.2 if small else 9.2

    for ax in [ax_obs, ax_rest]:
        style_axis(ax, small=small)

    ax_obs.plot(
        samples["observed_frequency_THz"],
        samples["raw_flux"],
        color=TEXT,
        linewidth=line_width,
        alpha=0.82,
    )
    ax_obs.scatter(
        samples["observed_frequency_THz"],
        samples["raw_flux"],
        s=point_size,
        color=color,
        edgecolor=BG,
        linewidth=0.45,
        zorder=5,
    )
    ax_obs.axvline(
        row.observed_peak_frequency_THz,
        color=ORANGE,
        linewidth=1.45,
        label="observed peak",
    )
    ax_obs.set_title(f"OBSERVED FRAME — {GALAXY} — {int(row.n)} {row.line_complex}", fontsize=title_size, pad=7)
    ax_obs.set_xlabel("Observed frequency, THz", fontsize=label_size)
    ax_obs.set_ylabel("Raw flux", fontsize=label_size)

    ax_rest.plot(
        samples["rest_frame_frequency_THz"],
        samples["raw_flux"],
        color=TEXT,
        linewidth=line_width,
        alpha=0.82,
    )
    ax_rest.scatter(
        samples["rest_frame_frequency_THz"],
        samples["raw_flux"],
        s=point_size,
        color=color,
        edgecolor=BG,
        linewidth=0.45,
        zorder=5,
    )
    ax_rest.axvline(
        row.rest_frequency_THz,
        color=CYAN,
        linewidth=1.45,
        label="rest frequency",
    )
    ax_rest.set_title(f"REST FRAME — same {len(samples)} JWST samples", fontsize=title_size, pad=7)
    ax_rest.set_xlabel("Rest-frame frequency, THz", fontsize=label_size)
    ax_rest.set_ylabel("Raw flux", fontsize=label_size)

    info = (
        f"νrest = {row.rest_frequency_THz:.3f} THz\n"
        f"νobs = {row.observed_peak_frequency_THz:.3f} THz\n"
        f"νrest / νobs = {row.stretch_factor:.6f}\n"
        f"z = {row.z_pair:.6f}"
    )
    ax_rest.text(
        0.018,
        0.965,
        info,
        transform=ax_rest.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=6.7 if small else 8.4,
        linespacing=1.34,
        bbox=dict(
            boxstyle="round,pad=0.34",
            facecolor="#07111f",
            edgecolor=color,
            alpha=0.96,
        ),
    )

    for ax in [ax_obs, ax_rest]:
        legend = ax.legend(
            loc="lower right",
            fontsize=5.8 if small else 7.0,
            facecolor="#07111f",
            edgecolor=GRID,
            framealpha=0.96,
        )
        for text in legend.get_texts():
            text.set_color(TEXT)


def create_individual_plots(lines, wave, flux):
    import matplotlib.pyplot as plt

    paths = []
    all_samples = []
    for index, row in enumerate(lines.itertuples()):
        samples = narrow_samples(row, wave, flux)
        samples.insert(0, "n", int(row.n))
        samples.insert(1, "line_complex", row.line_complex)
        samples["rest_frequency_reference_THz"] = row.rest_frequency_THz
        samples["observed_peak_frequency_reference_THz"] = row.observed_peak_frequency_THz
        samples["stretch_factor"] = row.stretch_factor
        samples["z_pair"] = row.z_pair
        all_samples.append(samples)

        fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.0), facecolor=BG)
        draw_pair(axes[0], axes[1], row, samples, POINT_COLORS[index], small=False)
        fig.suptitle(
            f"{VERSION} — narrow frequency comparison | real JWST/MAST samples | no model",
            color=TEXT,
            fontsize=13.3,
            fontweight="bold",
            y=0.975,
        )
        fig.subplots_adjust(left=0.065, right=0.985, top=0.89, bottom=0.12, wspace=0.18)
        path = PNG / f"{VERSION}_{GALAXY}_NARROW_FREQUENCY_PAIR_{int(row.n)}.png"
        fig.savefig(path, dpi=245, facecolor=BG, edgecolor=BG)
        plt.show()
        plt.close(fig)
        paths.append(path)

    return paths, all_samples


def create_dashboard(lines, wave, flux):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(18.0, 20.0), facecolor=BG)
    for index, row in enumerate(lines.itertuples()):
        samples = narrow_samples(row, wave, flux)
        draw_pair(axes[index, 0], axes[index, 1], row, samples, POINT_COLORS[index], small=True)

    fig.suptitle(
        f"{VERSION} — {GALAXY}: observed versus rest-frame frequency | same real samples side by side",
        color=TEXT,
        fontsize=15.5,
        fontweight="bold",
        y=0.992,
    )
    fig.text(
        0.5,
        0.012,
        "Right-hand panels are coordinate transformations only: νrest = νobserved × (1+zpair). Flux values are unchanged.",
        color=MUTED,
        fontsize=9.0,
        ha="center",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.035, hspace=0.38, wspace=0.18)
    path = PNG / f"{VERSION}_{GALAXY}_NARROW_FREQUENCY_SIDE_BY_SIDE_DASHBOARD.png"
    fig.savefig(path, dpi=225, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return path


def create_summary_table(lines):
    import matplotlib.pyplot as plt

    rows = [
        [
            int(row.n),
            row.line_complex,
            f"{row.observed_peak_frequency_THz:.3f}",
            f"{row.rest_frequency_THz:.3f}",
            f"{row.stretch_factor:.6f}",
            f"{row.z_pair:.6f}",
            N_SAMPLES,
        ]
        for row in lines.itertuples()
    ]

    fig = plt.figure(figsize=(15.5, 5.2), facecolor=BG)
    ax = fig.add_axes([0.035, 0.14, 0.93, 0.72])
    ax.axis("off")
    ax.set_title(
        f"{VERSION} — {GALAXY}: narrow-frequency comparison summary",
        color=TEXT,
        fontsize=13.8,
        fontweight="bold",
        pad=14,
    )
    table = ax.table(
        cellText=rows,
        colLabels=["#", "Line complex", "Observed ν THz", "Rest ν THz", "νrest/νobs", "z", "Samples"],
        loc="center",
        cellLoc="center",
        colWidths=[0.05, 0.34, 0.14, 0.14, 0.12, 0.10, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.50)
        cell.get_text().set_color(TEXT)
        if r == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#081523")
            if c == 0:
                cell.get_text().set_color(POINT_COLORS[r - 1])
                cell.get_text().set_fontweight("bold")

    fig.text(
        0.5,
        0.055,
        "Same measured flux samples on both sides; only the frequency coordinate changes.",
        color=MUTED,
        fontsize=9.0,
        ha="center",
    )
    path = PNG / f"{VERSION}_{GALAXY}_NARROW_FREQUENCY_SUMMARY_TABLE.png"
    fig.savefig(path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return path


def print_table(rows, headers):
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    line_csv, raw_csv = locate_inputs()
    lines, wave, flux, flux_column = load_data(line_csv, raw_csv)

    individual_paths, sample_frames = create_individual_plots(lines, wave, flux)
    dashboard_path = create_dashboard(lines, wave, flux)
    table_path = create_summary_table(lines)

    import pandas as pd
    sample_csv = CSV / f"{VERSION}_{GALAXY}_NARROW_FREQUENCY_SAMPLES.csv"
    pd.concat(sample_frames, ignore_index=True).to_csv(sample_csv, index=False)

    summary_csv = CSV / f"{VERSION}_{GALAXY}_NARROW_FREQUENCY_SUMMARY.csv"
    lines[[
        "n",
        "line_complex",
        "centroid_rest_um",
        "raw_local_peak_um_exploratory",
        "observed_peak_frequency_THz",
        "rest_frequency_THz",
        "stretch_factor",
        "z_pair",
    ]].to_csv(summary_csv, index=False)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (
                int(row.n),
                row.line_complex,
                f"{row.observed_peak_frequency_THz:.3f}",
                f"{row.rest_frequency_THz:.3f}",
                f"{row.stretch_factor:.6f}",
                f"{row.z_pair:.6f}",
            )
            for row in lines.itertuples()
        ],
        ["#", "Line complex", "Observed THz", "Rest THz", "Factor", "z"],
    )
    print()
    print_table(
        [
            ("Galaxy", GALAXY),
            ("Samples per line", N_SAMPLES),
            ("Transformation", "rest frequency = observed frequency x (1+z_pair)"),
            ("Flux transformation", "none; same measured values"),
            ("Dashboard PNG", str(dashboard_path)),
            ("Summary table PNG", str(table_path)),
            ("Sample CSV", str(sample_csv)),
            ("Summary CSV", str(summary_csv)),
            ("Line source CSV", str(line_csv)),
            ("Spectrum source CSV", str(raw_csv)),
            ("Flux column", flux_column),
        ],
        ["Field", "Value"],
    )
    print("\nINDIVIDUAL SIDE-BY-SIDE PLOTS")
    print_table([(index + 1, str(path)) for index, path in enumerate(individual_paths)], ["#", "Path"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
