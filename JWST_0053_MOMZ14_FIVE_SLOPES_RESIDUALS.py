# JWST_0053
# Five exploratory line-complex slopes for MoM-z14 plus published-z reference and residuals.
# No AI images. Matplotlib only. Uses cached JWST_0052 CSV output; no MAST query.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0053"
GALAXY = "MoM-z14"
Z_PUBLISHED = 14.44
M_PUBLISHED = 1.0 + Z_PUBLISHED
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
ORANGE = "#ff9d2e"
POINT_COLORS = ["#43b9ff", "#ff6b76", "#a78bfa", "#34d399", "#f6c453"]


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def locate_input_csv():
    preferred = CSV / "JWST_0052_MoM-z14_LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv"
    if preferred.exists() and preferred.stat().st_size > 100:
        return preferred
    matches = sorted(
        CSV.glob("JWST_0052_*LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "JWST_0052 line-complex CSV was not found. Run JWST_0052 first."
    )


def load_rows(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    required = [
        "n",
        "line_complex",
        "centroid_rest_um",
        "centroid_rest_frequency_THz",
        "centroid_expected_observed_um",
        "raw_local_peak_um_exploratory",
        "raw_local_peak_frequency_THz_exploratory",
        "sample_count",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Input CSV missing columns: {missing}")

    valid = frame[
        np.isfinite(frame["centroid_rest_um"])
        & np.isfinite(frame["raw_local_peak_um_exploratory"])
    ].copy()
    valid = valid.sort_values("n").reset_index(drop=True)
    if valid.empty:
        raise RuntimeError("No finite exploratory local-peak measurements were found.")

    valid["exploratory_slope_m"] = (
        valid["raw_local_peak_um_exploratory"] / valid["centroid_rest_um"]
    )
    valid["exploratory_z"] = valid["exploratory_slope_m"] - 1.0
    valid["delta_z_from_published"] = valid["exploratory_z"] - Z_PUBLISHED
    valid["delta_lambda_um_from_published"] = (
        valid["raw_local_peak_um_exploratory"]
        - valid["centroid_expected_observed_um"]
    )
    valid["delta_velocity_km_s_approx"] = (
        299792.458
        * valid["delta_z_from_published"]
        / (1.0 + Z_PUBLISHED)
    )
    return valid


def style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.50)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.85)


def dark_legend(ax, loc="best", fontsize=7.3):
    legend = ax.legend(
        loc=loc,
        fontsize=fontsize,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.97,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)
    return legend


def create_plot(rows):
    import numpy as np
    import matplotlib.pyplot as plt

    x_min = float(rows["centroid_rest_um"].min() - 0.010)
    x_max = float(rows["centroid_rest_um"].max() + 0.010)
    x_line = np.linspace(x_min, x_max, 500)

    fig = plt.figure(figsize=(17.0, 11.0), facecolor=BG)
    ax = fig.add_axes([0.075, 0.39, 0.89, 0.52])
    residual = fig.add_axes([0.075, 0.095, 0.89, 0.20])
    style_axis(ax)
    style_axis(residual)

    ax.plot(
        x_line,
        M_PUBLISHED * x_line,
        color=ORANGE,
        linewidth=2.6,
        label=f"Published joint-fit relation: m={M_PUBLISHED:.6f}, z={Z_PUBLISHED:.6f}",
        zorder=5,
    )

    for index, row in enumerate(rows.itertuples()):
        color = POINT_COLORS[index % len(POINT_COLORS)]
        ax.plot(
            x_line,
            row.exploratory_slope_m * x_line,
            color=color,
            linewidth=1.15,
            alpha=0.95,
            label=(
                f"{int(row.n)} {row.line_complex}: "
                f"m={row.exploratory_slope_m:.6f}, z={row.exploratory_z:.6f}"
            ),
        )
        ax.scatter(
            [row.centroid_rest_um],
            [row.raw_local_peak_um_exploratory],
            s=86,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.9,
            zorder=8,
        )
        ax.annotate(
            (
                f"{int(row.n)} {row.line_complex}\n"
                f"rest ν={row.centroid_rest_frequency_THz:.3f} THz\n"
                f"raw peak={row.raw_local_peak_um_exploratory:.6f} µm\n"
                f"zᵢ={row.exploratory_z:.6f}"
            ),
            xy=(row.centroid_rest_um, row.raw_local_peak_um_exploratory),
            xytext=(10, 18 if index % 2 == 0 else -52),
            textcoords="offset points",
            color=TEXT,
            fontsize=8.0,
            va="bottom" if index % 2 == 0 else "top",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="#07111f",
                edgecolor=color,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=0.8,
            ),
            zorder=10,
        )

    all_slopes = rows["exploratory_slope_m"].tolist() + [M_PUBLISHED]
    y_values = []
    for slope in all_slopes:
        y_values.extend([slope * x_min, slope * x_max])
    y_min = float(min(y_values))
    y_max = float(max(y_values))
    y_pad = max(0.012, 0.08 * (y_max - y_min))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Rest-frame centroid wavelength, µm", fontsize=10.2)
    ax.set_ylabel("Observed wavelength, µm", fontsize=10.2)
    ax.set_title(
        f"{VERSION} — {GALAXY}: five exploratory line-complex slopes versus published z=14.44",
        fontsize=13.5,
        pad=12,
    )
    dark_legend(ax, "upper left", 7.1)

    x_positions = np.arange(len(rows))
    residual.axhline(0.0, color=ORANGE, linewidth=1.6, label="published z reference")
    for index, row in enumerate(rows.itertuples()):
        color = POINT_COLORS[index % len(POINT_COLORS)]
        residual.vlines(
            index,
            0.0,
            row.delta_z_from_published,
            color=color,
            linewidth=1.25,
        )
        residual.scatter(
            [index],
            [row.delta_z_from_published],
            s=58,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.75,
            zorder=6,
        )
        residual.text(
            index,
            row.delta_z_from_published,
            f" {row.delta_z_from_published:+.4f}",
            color=TEXT,
            fontsize=8.0,
            va="bottom" if row.delta_z_from_published >= 0 else "top",
            ha="center",
        )

    residual.set_xticks(x_positions)
    residual.set_xticklabels(
        [f"{int(row.n)}\n{row.line_complex}" for row in rows.itertuples()],
        color=TEXT,
        fontsize=7.3,
    )
    residual.set_ylabel("Δz from published", fontsize=9.5)
    residual.set_title(
        "Residuals expose the differences hidden by the previous single orange relation",
        fontsize=10.5,
        pad=8,
    )

    fig.text(
        0.5,
        0.025,
        (
            "Scientific caution: each colored slope uses a raw local maximum divided by the complex centroid rest wavelength. "
            "These are exploratory visualization slopes, not independent published spectroscopic redshifts. "
            "The orange z=14.44 value comes from the paper's joint tied-redshift blend fit and Lyman-break confirmation."
        ),
        color=MUTED,
        fontsize=8.6,
        ha="center",
        va="bottom",
        wrap=True,
    )

    plot_path = PNG / f"{VERSION}_{GALAXY}_FIVE_EXPLORATORY_SLOPES_AND_RESIDUALS.png"
    fig.savefig(plot_path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return plot_path


def create_table(rows, source_csv):
    import matplotlib.pyplot as plt

    table_rows = []
    for row in rows.itertuples():
        table_rows.append(
            [
                int(row.n),
                row.line_complex,
                f"{row.centroid_rest_um:.6f}",
                f"{row.centroid_rest_frequency_THz:.3f}",
                f"{row.raw_local_peak_um_exploratory:.6f}",
                f"{row.raw_local_peak_frequency_THz_exploratory:.3f}",
                f"{row.exploratory_slope_m:.6f}",
                f"{row.exploratory_z:.6f}",
                f"{row.delta_z_from_published:+.6f}",
                f"{row.delta_lambda_um_from_published:+.6f}",
                int(row.sample_count),
            ]
        )

    fig = plt.figure(figsize=(20.0, 7.2), facecolor=BG)
    ax = fig.add_axes([0.02, 0.10, 0.96, 0.80])
    ax.axis("off")
    ax.set_title(
        f"{VERSION} — {GALAXY} five exploratory slopes and residuals",
        color=TEXT,
        fontsize=14.0,
        fontweight="bold",
        pad=14,
    )
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "#",
            "Line complex",
            "Rest centroid µm",
            "Rest ν THz",
            "Raw peak µm",
            "Raw peak ν THz",
            "Slope mᵢ",
            "zᵢ",
            "Δz",
            "Δλ µm",
            "Samples",
        ],
        loc="center",
        cellLoc="center",
        colWidths=[0.035, 0.19, 0.09, 0.09, 0.09, 0.09, 0.08, 0.075, 0.075, 0.075, 0.055],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1.0, 1.55)
    for (row_index, col_index), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.50)
        cell.get_text().set_color(TEXT)
        if row_index == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#081523")
            if col_index == 0:
                cell.get_text().set_color(POINT_COLORS[row_index - 1])
                cell.get_text().set_fontweight("bold")

    fig.text(
        0.5,
        0.035,
        f"Published reference: m={M_PUBLISHED:.6f}, z={Z_PUBLISHED:.6f}. Source: {source_csv}",
        color=MUTED,
        fontsize=8.6,
        ha="center",
    )
    table_path = PNG / f"{VERSION}_{GALAXY}_FIVE_SLOPES_RESIDUALS_TABLE.png"
    fig.savefig(table_path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return table_path


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

    source_csv = locate_input_csv()
    rows = load_rows(source_csv)
    result_csv = CSV / f"{VERSION}_{GALAXY}_FIVE_EXPLORATORY_SLOPES_RESIDUALS.csv"
    rows.to_csv(result_csv, index=False)
    plot_path = create_plot(rows)
    table_path = create_table(rows, source_csv)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (
                int(row.n),
                row.line_complex,
                f"{row.exploratory_slope_m:.6f}",
                f"{row.exploratory_z:.6f}",
                f"{row.delta_z_from_published:+.6f}",
                f"{row.delta_lambda_um_from_published:+.6f}",
            )
            for row in rows.itertuples()
        ],
        ["#", "Line complex", "Slope m", "Exploratory z", "Delta z", "Delta lambda um"],
    )
    print()
    print_table(
        [
            ("Galaxy", GALAXY),
            ("Published slope", f"{M_PUBLISHED:.6f}"),
            ("Published z", f"{Z_PUBLISHED:.6f}"),
            ("Colored curves", len(rows)),
            ("Orange curves", 1),
            ("Input CSV", str(source_csv)),
            ("Plot PNG", str(plot_path)),
            ("Table PNG", str(table_path)),
            ("Results CSV", str(result_csv)),
            ("Interpretation", "colored slopes exploratory; orange line published joint-fit redshift"),
        ],
        ["Field", "Value"],
    )
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
