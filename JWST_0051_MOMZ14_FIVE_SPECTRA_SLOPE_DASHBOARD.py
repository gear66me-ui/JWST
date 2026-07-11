# JWST_0051
# Five real UV spectral-line panels + labeled slope panel + styled summary table.
# No AI images. Matplotlib only. Reuses saved real JWST/MAST CSV data.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0051"
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
GRAY = "#aeb7c2"
POINT_COLORS = ["#43b9ff", "#ff6b76", "#a78bfa", "#34d399", "#f6c453"]


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def locate_file(preferred_names, pattern):
    for name in preferred_names:
        path = CSV / name
        if path.exists() and path.stat().st_size > 100:
            return path
    candidates = sorted(CSV.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No file found for pattern: {pattern}")


def locate_inputs():
    measurements = locate_file(
        [
            "JWST_0048_UV_LINE_MEASUREMENTS.csv",
            "JWST_0047_UV_LINE_MEASUREMENTS.csv",
            "JWST_0046_UV_LINE_MEASUREMENTS.csv",
        ],
        "JWST_*_UV_LINE_MEASUREMENTS.csv",
    )
    spectrum = locate_file(
        [
            "JWST_0048_REAL_RAW_SPECTRUM.csv",
            "JWST_0047_REAL_RAW_SPECTRUM.csv",
            "JWST_0046_REAL_RAW_SPECTRUM.csv",
        ],
        "JWST_*_REAL_RAW_SPECTRUM.csv",
    )
    return measurements, spectrum


def find_column(frame, preferred, startswith=None):
    lower = {str(c).lower(): c for c in frame.columns}
    for name in preferred:
        if name.lower() in lower:
            return lower[name.lower()]
    if startswith:
        for column in frame.columns:
            if str(column).lower().startswith(startswith.lower()):
                return column
    return None


def load_data(measurement_path, spectrum_path):
    import numpy as np
    import pandas as pd

    meas = pd.read_csv(measurement_path)
    required = [
        "n",
        "line",
        "rest_um",
        "expected_at_z14p44_um",
        "raw_peak_sample_um",
        "raw_peak_flux_native",
        "sample_count_in_window",
    ]
    missing = [name for name in required if name not in meas.columns]
    if missing:
        raise RuntimeError(f"Measurement CSV missing columns: {missing}")

    meas = meas[
        (meas["sample_count_in_window"] > 0)
        & np.isfinite(meas["rest_um"])
        & np.isfinite(meas["raw_peak_sample_um"])
    ].copy()
    meas = meas.sort_values("n").reset_index(drop=True)
    if len(meas) < 5:
        raise RuntimeError(f"Expected five measured UV lines; found {len(meas)}")
    meas = meas.head(5).copy()

    raw = pd.read_csv(spectrum_path)
    wave_col = find_column(raw, ["wavelength_um_raw", "wavelength_um"], "wavelength")
    flux_col = find_column(raw, [], "flux_raw_")
    if wave_col is None or flux_col is None:
        raise RuntimeError(
            f"Could not identify wavelength/flux columns in {spectrum_path.name}: {list(raw.columns)}"
        )

    wave = raw[wave_col].to_numpy(float)
    flux = raw[flux_col].to_numpy(float)
    finite = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[finite]
    flux = flux[finite]
    order = np.argsort(wave)
    wave = wave[order]
    flux = flux[order]

    meas["slope_m"] = meas["raw_peak_sample_um"] / meas["rest_um"]
    meas["z_from_raw_peak"] = meas["slope_m"] - 1.0

    return meas, wave, flux, str(flux_col)


def style_axis(ax, small=False):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.48 if small else 0.58, alpha=0.48)
    ax.tick_params(colors=TEXT, labelsize=7.2 if small else 9.0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.75)


def dark_legend(ax, loc="best", fontsize=7.5):
    legend = ax.legend(
        loc=loc,
        fontsize=fontsize,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.96,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)
    return legend


def spectrum_window(wave, flux, center, half_width=0.060):
    import numpy as np

    mask = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave >= center - half_width)
        & (wave <= center + half_width)
    )
    return mask


def draw_spectrum_panel(ax, row, wave, flux, color, small=False):
    import numpy as np

    center = float(row.expected_at_z14p44_um)
    mask = spectrum_window(wave, flux, center, 0.060)
    count = int(mask.sum())
    style_axis(ax, small=small)

    if count < 2:
        ax.text(
            0.5,
            0.5,
            f"NO FINITE DATA\n{row.line}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=TEXT,
            fontsize=11,
            fontweight="bold",
        )
        ax.set_title(f"{int(row.n)} — {row.line}")
        return count

    local_wave = wave[mask]
    local_flux = flux[mask]

    ax.plot(local_wave, local_flux, color=CYAN, linewidth=0.75, label="raw MAST spectrum")
    ax.axvline(
        row.expected_at_z14p44_um,
        color=GRAY,
        linestyle=":",
        linewidth=0.90,
        label="expected at z=14.44",
    )
    ax.axvline(
        row.raw_peak_sample_um,
        color=ORANGE,
        linestyle="--",
        linewidth=1.05,
        label="raw local peak",
    )
    ax.scatter(
        [row.raw_peak_sample_um],
        [row.raw_peak_flux_native],
        s=30 if small else 42,
        color=color,
        edgecolor="#f8fbff",
        linewidth=0.70,
        zorder=6,
    )
    ax.text(
        row.raw_peak_sample_um,
        row.raw_peak_flux_native,
        f"  {int(row.n)}",
        color=TEXT,
        fontsize=8.5 if small else 10.0,
        fontweight="bold",
        va="bottom",
    )

    ymin = float(np.nanmin(local_flux))
    ymax = float(np.nanmax(local_flux))
    span = ymax - ymin
    pad = 0.08 * span if span > 0 else max(abs(ymax) * 0.10, 1.0e-6)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(center - 0.060, center + 0.060)

    ax.set_title(
        f"{int(row.n)} — {row.line} | λobs={row.raw_peak_sample_um:.6f} µm | z={row.z_from_raw_peak:.6f}",
        fontsize=8.8 if small else 11.2,
        pad=7,
    )
    ax.set_xlabel("Observed wavelength, µm", fontsize=7.6 if small else 9.4)
    ax.set_ylabel("Raw flux", fontsize=7.6 if small else 9.4)
    dark_legend(ax, "upper right", 6.2 if small else 7.4)
    return count


def draw_slope_panel(ax, meas, small=False):
    import numpy as np

    style_axis(ax, small=small)

    avg_rest = float(meas["rest_um"].mean())
    avg_obs = float(meas["raw_peak_sample_um"].mean())
    ratio_slope = avg_obs / avg_rest
    ratio_z = ratio_slope - 1.0
    mean_slope = float(meas["slope_m"].mean())
    mean_z = mean_slope - 1.0

    xmin = float(meas["rest_um"].min() - 0.007)
    xmax = float(meas["rest_um"].max() + 0.007)
    xs = np.linspace(xmin, xmax, 400)

    slopes = meas["slope_m"].to_numpy(float)
    ax.fill_between(
        xs,
        slopes.min() * xs,
        slopes.max() * xs,
        color="#35556f",
        alpha=0.18,
        label="five-slope span",
    )

    for index, row in enumerate(meas.itertuples()):
        color = POINT_COLORS[index]
        ax.plot(
            xs,
            row.slope_m * xs,
            color=color,
            linewidth=0.82 if small else 0.95,
            alpha=0.92,
        )
        ax.scatter(
            [row.rest_um],
            [row.raw_peak_sample_um],
            s=42 if small else 72,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.75,
            zorder=8,
        )
        ax.annotate(
            f"{int(row.n)} {row.line}",
            xy=(row.rest_um, row.raw_peak_sample_um),
            xytext=(6, 8 if index % 2 == 0 else -14),
            textcoords="offset points",
            color=TEXT,
            fontsize=6.8 if small else 8.5,
            fontweight="bold",
        )

    ax.plot(
        xs,
        ratio_slope * xs,
        color=ORANGE,
        linewidth=2.0 if small else 2.35,
        label=f"ratio-of-means m={ratio_slope:.6f}, z={ratio_z:.6f}",
        zorder=6,
    )

    y_values = []
    for slope in slopes:
        y_values.extend([slope * xmin, slope * xmax])
    y_values.extend([ratio_slope * xmin, ratio_slope * xmax])
    ymin = float(min(y_values))
    ymax = float(max(y_values))
    pad = max(0.010, 0.08 * (ymax - ymin))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("Rest-frame wavelength, µm", fontsize=7.6 if small else 9.6)
    ax.set_ylabel("Observed peak wavelength, µm", fontsize=7.6 if small else 9.6)
    ax.set_title(
        "Five measured UV points and normalized redshift slopes",
        fontsize=9.2 if small else 12.0,
        pad=8,
    )
    dark_legend(ax, "upper left", 6.2 if small else 7.6)

    ax.text(
        0.985,
        0.025,
        (
            "AVERAGE AUDIT\n"
            f"Σλobs / Σλrest = {ratio_slope:.6f}\n"
            f"z = {ratio_z:.6f}\n"
            f"mean(mᵢ) = {mean_slope:.6f}\n"
            f"mean(zᵢ) = {mean_z:.6f}\n"
            "Not a Gaussian average"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=TEXT,
        fontsize=6.8 if small else 8.2,
        linespacing=1.30,
        bbox=dict(
            boxstyle="round,pad=0.38",
            facecolor="#07111f",
            edgecolor=ORANGE,
            alpha=0.96,
        ),
    )

    return ratio_slope, ratio_z, mean_slope, mean_z


def create_individual_spectrum_plots(meas, wave, flux):
    import matplotlib.pyplot as plt

    paths = []
    for index, row in enumerate(meas.itertuples()):
        fig = plt.figure(figsize=(13.2, 5.8), facecolor=BG)
        ax = fig.add_axes([0.075, 0.13, 0.89, 0.77])
        draw_spectrum_panel(ax, row, wave, flux, POINT_COLORS[index], small=False)
        fig.suptitle(
            f"{VERSION} — real JWST/MAST UV spectral line {int(row.n)} of 5",
            color=TEXT,
            fontsize=13.0,
            y=0.975,
        )
        path = PNG / f"{VERSION}_SPECTRAL_LINE_{int(row.n)}_{row.line.replace(' ', '_').replace('[', '').replace(']', '')}.png"
        fig.savefig(path, dpi=240, facecolor=BG, edgecolor=BG)
        plt.show()
        plt.close(fig)
        paths.append(path)
    return paths


def create_dashboard(meas, wave, flux):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(18.0, 14.0), facecolor=BG)
    axes = axes.ravel()

    for index, row in enumerate(meas.itertuples()):
        draw_spectrum_panel(
            axes[index],
            row,
            wave,
            flux,
            POINT_COLORS[index],
            small=True,
        )

    ratio_slope, ratio_z, mean_slope, mean_z = draw_slope_panel(
        axes[5],
        meas,
        small=True,
    )

    fig.suptitle(
        f"{VERSION} — five real spectral lines + normalized five-point slope plot",
        color=TEXT,
        fontsize=16.0,
        y=0.982,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.945, bottom=0.055, hspace=0.34, wspace=0.18)

    path = PNG / f"{VERSION}_FIVE_SPECTRA_AND_SLOPE_DASHBOARD.png"
    fig.savefig(path, dpi=230, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    return path, ratio_slope, ratio_z, mean_slope, mean_z


def create_slope_plot(meas):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16.0, 9.0), facecolor=BG)
    ax = fig.add_axes([0.075, 0.105, 0.89, 0.80])
    ratio_slope, ratio_z, mean_slope, mean_z = draw_slope_panel(ax, meas, small=False)
    fig.suptitle(
        f"{VERSION} — five measured UV points, five slopes, and orange average",
        color=TEXT,
        fontsize=14.0,
        y=0.975,
    )
    path = PNG / f"{VERSION}_FIVE_POINT_NORMALIZED_SLOPE_PLOT.png"
    fig.savefig(path, dpi=240, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return path, ratio_slope, ratio_z, mean_slope, mean_z


def create_summary(meas, measurement_path, spectrum_path, ratio_slope, ratio_z, mean_slope, mean_z):
    import pandas as pd
    import matplotlib.pyplot as plt

    result = meas.copy()
    result["average_line_obs_um"] = ratio_slope * result["rest_um"]
    result["delta_from_ratio_average_um"] = (
        result["raw_peak_sample_um"] - result["average_line_obs_um"]
    )
    result["delta_from_ratio_average_percent"] = (
        100.0 * result["delta_from_ratio_average_um"] / result["average_line_obs_um"]
    )

    result["measurement_source_csv"] = str(measurement_path)
    result["spectrum_source_csv"] = str(spectrum_path)
    result["ratio_of_means_slope"] = ratio_slope
    result["ratio_of_means_z"] = ratio_z
    result["mean_individual_slope"] = mean_slope
    result["mean_individual_z"] = mean_z
    result["average_definition"] = "ratio of means = sum(observed wavelengths) / sum(rest wavelengths); not Gaussian"

    csv_path = CSV / f"{VERSION}_FIVE_LINE_RESULTS_AND_AVERAGE.csv"
    result.to_csv(csv_path, index=False)

    table_rows = []
    row_types = []
    for row in result.itertuples():
        table_rows.append(
            [
                int(row.n),
                row.line,
                f"{row.rest_um:.6f}",
                f"{row.expected_at_z14p44_um:.6f}",
                f"{row.raw_peak_sample_um:.6f}",
                f"{row.slope_m:.6f}",
                f"{row.z_from_raw_peak:.6f}",
                f"{row.delta_from_ratio_average_um:+.6f}",
                int(row.sample_count_in_window),
            ]
        )
        row_types.append("line")

    table_rows.append(
        [
            "Σ",
            "SUM",
            f"{result['rest_um'].sum():.6f}",
            "",
            f"{result['raw_peak_sample_um'].sum():.6f}",
            "",
            "",
            "",
            int(result["sample_count_in_window"].sum()),
        ]
    )
    row_types.append("sum")

    table_rows.append(
        [
            "μ",
            "ARITHMETIC MEANS",
            f"{result['rest_um'].mean():.6f}",
            "",
            f"{result['raw_peak_sample_um'].mean():.6f}",
            f"mean mᵢ={mean_slope:.6f}",
            f"mean zᵢ={mean_z:.6f}",
            "",
            "",
        ]
    )
    row_types.append("mean")

    table_rows.append(
        [
            "★",
            "ORANGE RATIO-OF-MEANS",
            "Σrest",
            "",
            "Σobserved",
            f"{ratio_slope:.6f}",
            f"{ratio_z:.6f}",
            "NOT GAUSSIAN",
            "",
        ]
    )
    row_types.append("ratio")

    fig = plt.figure(figsize=(19.0, 7.5), facecolor=BG)
    ax = fig.add_axes([0.025, 0.09, 0.95, 0.79])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        f"{VERSION} — five measured UV lines and exact average calculation",
        color=TEXT,
        fontsize=15.0,
        pad=15,
        fontweight="bold",
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "#",
            "UV line",
            "rest λ µm",
            "expected λ µm",
            "raw peak λ µm",
            "slope m",
            "z",
            "Δ from orange µm",
            "samples",
        ],
        loc="center",
        cellLoc="center",
        colWidths=[0.045, 0.16, 0.105, 0.115, 0.115, 0.11, 0.10, 0.135, 0.075],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    table.scale(1.0, 1.48)

    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.52)
        cell.get_text().set_color(TEXT)

        if row_index == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            kind = row_types[row_index - 1]
            if kind == "line":
                line_index = row_index - 1
                cell.set_facecolor("#081523")
                if col_index == 0:
                    cell.get_text().set_color(POINT_COLORS[line_index])
                    cell.get_text().set_fontweight("bold")
            elif kind == "sum":
                cell.set_facecolor("#10263a")
                cell.get_text().set_fontweight("bold")
            elif kind == "mean":
                cell.set_facecolor("#17314a")
                cell.get_text().set_fontweight("bold")
            elif kind == "ratio":
                cell.set_facecolor("#4a2e12")
                cell.get_text().set_color("#ffd39a")
                cell.get_text().set_fontweight("bold")

    fig.text(
        0.5,
        0.035,
        (
            f"Orange average: m = Σλobs / Σλrest = {ratio_slope:.6f}; "
            f"z = m − 1 = {ratio_z:.6f}.  "
            f"Arithmetic mean of individual slopes = {mean_slope:.6f}.  "
            "These are wavelength-ratio statistics, not a Gaussian fit."
        ),
        color=TEXT,
        fontsize=9.4,
        ha="center",
        va="center",
    )

    png_path = PNG / f"{VERSION}_FIVE_LINE_RESULTS_AVERAGE_TABLE.png"
    fig.savefig(png_path, dpi=250, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    return png_path, csv_path, result


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

    measurement_path, spectrum_path = locate_inputs()
    meas, wave, flux, flux_column = load_data(measurement_path, spectrum_path)

    individual_paths = create_individual_spectrum_plots(meas, wave, flux)
    dashboard_path, ratio_slope, ratio_z, mean_slope, mean_z = create_dashboard(
        meas,
        wave,
        flux,
    )
    slope_path, _, _, _, _ = create_slope_plot(meas)
    table_path, result_csv, result = create_summary(
        meas,
        measurement_path,
        spectrum_path,
        ratio_slope,
        ratio_z,
        mean_slope,
        mean_z,
    )

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (
                int(row.n),
                row.line,
                f"{row.rest_um:.6f}",
                f"{row.raw_peak_sample_um:.6f}",
                f"{row.slope_m:.6f}",
                f"{row.z_from_raw_peak:.6f}",
                int(row.sample_count_in_window),
            )
            for row in result.itertuples()
        ],
        ["#", "Line", "Rest µm", "Observed µm", "Slope m", "z", "Samples"],
    )
    print()
    print_table(
        [
            ("Measurement CSV", str(measurement_path)),
            ("Raw spectrum CSV", str(spectrum_path)),
            ("Raw flux column", flux_column),
            ("Spectral plots displayed", len(individual_paths)),
            ("Dashboard PNG", str(dashboard_path)),
            ("Normalized slope PNG", str(slope_path)),
            ("Summary table PNG", str(table_path)),
            ("Results CSV", str(result_csv)),
            ("Orange average method", "sum observed wavelengths / sum rest wavelengths"),
            ("Orange slope m", f"{ratio_slope:.6f}"),
            ("Orange z", f"{ratio_z:.6f}"),
            ("Mean individual slope", f"{mean_slope:.6f}"),
            ("Mean individual z", f"{mean_z:.6f}"),
            ("Gaussian average", "NO"),
        ],
        ["Field", "Value"],
    )
    print()
    print("INDIVIDUAL SPECTRAL PNG FILES")
    print_table(
        [(index + 1, str(path)) for index, path in enumerate(individual_paths)],
        ["#", "Path"],
    )
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
