# JWST_0049
# Five-point labeled UV redshift-slope plot from the real MAST measurements saved by JWST_0048.
# No AI images. Matplotlib only. No new MAST query is required.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0049"
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


def need(pkg):
    try:
        importlib.import_module(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def locate_measurement_csv():
    preferred = [
        CSV / "JWST_0048_UV_LINE_MEASUREMENTS.csv",
        CSV / "JWST_0047_UV_LINE_MEASUREMENTS.csv",
        CSV / "JWST_0046_UV_LINE_MEASUREMENTS.csv",
    ]
    for path in preferred:
        if path.exists() and path.stat().st_size > 100:
            return path

    candidates = sorted(
        CSV.glob("JWST_*_UV_LINE_MEASUREMENTS.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No UV line-measurement CSV was found in /content/JWST_OUTPUT/CSV. "
        "Run JWST_0048 first."
    )


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


def load_measurements(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    required = {
        "n",
        "line",
        "rest_um",
        "raw_peak_sample_um",
        "sample_count_in_window",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Measurement CSV is missing columns: {missing}")

    valid = frame[
        (frame["sample_count_in_window"] > 0)
        & np.isfinite(frame["rest_um"])
        & np.isfinite(frame["raw_peak_sample_um"])
    ].copy()

    valid = valid.sort_values("n").reset_index(drop=True)
    if valid.empty:
        raise RuntimeError("The measurement CSV contains no finite measured UV points.")

    valid["slope_m"] = valid["raw_peak_sample_um"] / valid["rest_um"]
    valid["z_from_raw_peak"] = valid["slope_m"] - 1.0
    valid["delta_from_average_um"] = 0.0
    return valid


def build_plot(meas, source_csv):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    avg_rest = float(meas["rest_um"].mean())
    avg_obs = float(meas["raw_peak_sample_um"].mean())
    avg_slope = avg_obs / avg_rest
    avg_z = avg_slope - 1.0

    meas = meas.copy()
    meas["average_line_obs_um"] = avg_slope * meas["rest_um"]
    meas["delta_from_average_um"] = (
        meas["raw_peak_sample_um"] - meas["average_line_obs_um"]
    )
    meas["abs_delta_from_average_um"] = meas["delta_from_average_um"].abs()
    meas["distance_rank"] = (
        meas["abs_delta_from_average_um"].rank(method="first").astype(int)
    )

    xmin = float(meas["rest_um"].min() - 0.008)
    xmax = float(meas["rest_um"].max() + 0.008)
    xs = np.linspace(xmin, xmax, 500)

    line_y = []
    for row in meas.itertuples():
        line_y.extend([row.slope_m * xmin, row.slope_m * xmax])
    line_y.extend([avg_slope * xmin, avg_slope * xmax])
    ymin = float(min(line_y))
    ymax = float(max(line_y))
    ypad = 0.055 * (ymax - ymin)

    fig = plt.figure(figsize=(18.0, 8.6), facecolor=BG)
    grid = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[4.45, 1.55],
        wspace=0.035,
    )
    ax = fig.add_subplot(grid[0, 0])
    lane = fig.add_subplot(grid[0, 1])
    style_axis(ax)
    lane.set_facecolor(BG)
    lane.axis("off")

    slopes = meas["slope_m"].to_numpy(float)
    ax.fill_between(
        xs,
        slopes.min() * xs,
        slopes.max() * xs,
        color="#34506a",
        alpha=0.18,
        label="span of the five individual slopes",
    )

    for idx, row in enumerate(meas.itertuples()):
        color = POINT_COLORS[idx % len(POINT_COLORS)]
        ax.plot(
            xs,
            row.slope_m * xs,
            color=color,
            linewidth=0.85,
            alpha=0.92,
            label=f"{int(row.n)} {row.line}: m={row.slope_m:.6f}",
        )
        ax.scatter(
            [row.rest_um],
            [row.raw_peak_sample_um],
            s=72,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.85,
            zorder=8,
        )
        ax.annotate(
            f"{int(row.n)}  {row.line}",
            xy=(row.rest_um, row.raw_peak_sample_um),
            xytext=(8, 11 if idx % 2 == 0 else -18),
            textcoords="offset points",
            color=TEXT,
            fontsize=9.2,
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=0.75,
                alpha=0.95,
            ),
        )

    ax.plot(
        xs,
        avg_slope * xs,
        color=ORANGE,
        linewidth=2.25,
        label=(
            f"orange ratio-of-means: m={avg_slope:.6f}, "
            f"z={avg_z:.6f}"
        ),
        zorder=6,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel("Rest-frame wavelength, µm", fontsize=10.5)
    ax.set_ylabel("Observed raw peak wavelength, µm", fontsize=10.5)
    ax.set_title(
        f"{VERSION} — five measured UV points, five individual slopes, and orange average",
        fontsize=13.4,
        pad=12,
    )

    legend = ax.legend(
        loc="upper left",
        fontsize=7.7,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.97,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    lane.text(
        0.03,
        0.965,
        "Five measured UV points",
        color=TEXT,
        fontsize=15,
        fontweight="bold",
        va="top",
    )
    lane.text(
        0.03,
        0.915,
        "Each point is (rest λ, observed raw peak λ).\n"
        "Each colored line passes through the origin:\n"
        "mᵢ = λobs,ᵢ / λrest,ᵢ = 1 + zᵢ",
        color=MUTED,
        fontsize=9.0,
        va="top",
        linespacing=1.45,
    )

    y_positions = np.linspace(0.79, 0.25, len(meas))
    for idx, (row, ypos) in enumerate(zip(meas.itertuples(), y_positions)):
        color = POINT_COLORS[idx % len(POINT_COLORS)]
        lane.scatter(
            [0.045],
            [ypos],
            s=250,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.9,
            transform=lane.transAxes,
            clip_on=False,
        )
        lane.text(
            0.045,
            ypos,
            str(int(row.n)),
            color="#07101f",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="center",
            transform=lane.transAxes,
        )
        lane.text(
            0.13,
            ypos + 0.025,
            f"{int(row.n)}  {row.line}",
            color=color,
            fontsize=11.5,
            fontweight="bold",
            va="center",
            transform=lane.transAxes,
        )
        lane.text(
            0.13,
            ypos - 0.003,
            f"λrest = {row.rest_um:.6f} µm",
            color=MUTED,
            fontsize=8.6,
            va="center",
            transform=lane.transAxes,
        )
        lane.text(
            0.13,
            ypos - 0.033,
            f"λobs = {row.raw_peak_sample_um:.6f} µm",
            color=MUTED,
            fontsize=8.6,
            va="center",
            transform=lane.transAxes,
        )
        lane.text(
            0.13,
            ypos - 0.063,
            (
                f"m = {row.slope_m:.6f}; z = {row.z_from_raw_peak:.6f}; "
                f"Δavg = {row.delta_from_average_um:+.6f} µm"
            ),
            color=TEXT,
            fontsize=8.25,
            va="center",
            transform=lane.transAxes,
        )

    closest = meas.sort_values("abs_delta_from_average_um").head(2)
    closest_names = " and ".join(
        f"#{int(row.n)} {row.line}" for row in closest.itertuples()
    )
    lane.text(
        0.03,
        0.135,
        "Orange line is not a Gaussian fit.",
        color=ORANGE,
        fontsize=10.5,
        fontweight="bold",
        transform=lane.transAxes,
    )
    lane.text(
        0.03,
        0.097,
        (
            f"m = Σλobs / Σλrest = {avg_slope:.6f}\n"
            f"z = m − 1 = {avg_z:.6f}\n"
            f"Closest points to orange: {closest_names}"
        ),
        color=TEXT,
        fontsize=8.8,
        va="top",
        linespacing=1.45,
        transform=lane.transAxes,
        bbox=dict(
            facecolor="#07111f",
            edgecolor=GRID,
            boxstyle="round,pad=0.45",
            alpha=0.96,
        ),
    )

    fig.subplots_adjust(left=0.065, right=0.985, top=0.925, bottom=0.105)
    plot_path = PNG / f"{VERSION}_FIVE_POINT_LABELED_UV_SLOPES.png"
    fig.savefig(
        plot_path,
        dpi=290,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

    audit = meas[
        [
            "n",
            "line",
            "rest_um",
            "raw_peak_sample_um",
            "slope_m",
            "z_from_raw_peak",
            "average_line_obs_um",
            "delta_from_average_um",
            "abs_delta_from_average_um",
            "distance_rank",
        ]
    ].copy()
    audit.insert(0, "source_csv", str(source_csv))
    audit["average_method"] = "ratio of wavelength means = sum(obs)/sum(rest); not Gaussian"
    audit["average_slope_m"] = avg_slope
    audit["average_z"] = avg_z
    audit_path = CSV / f"{VERSION}_FIVE_POINT_LABELED_UV_SLOPES.csv"
    audit.to_csv(audit_path, index=False)

    table_rows = []
    for row in audit.itertuples():
        table_rows.append(
            [
                int(row.n),
                row.line,
                f"{row.rest_um:.6f}",
                f"{row.raw_peak_sample_um:.6f}",
                f"{row.slope_m:.6f}",
                f"{row.z_from_raw_peak:.6f}",
                f"{row.delta_from_average_um:+.6f}",
            ]
        )

    table_fig, table_ax = plt.subplots(figsize=(15.8, 4.8), facecolor=BG)
    table_ax.set_facecolor(BG)
    table_ax.axis("off")
    table_ax.set_title(
        f"{VERSION} — five UV line points and distance from orange average",
        color=TEXT,
        fontsize=13,
        pad=12,
    )
    table = table_ax.table(
        cellText=table_rows,
        colLabels=["#", "line", "rest µm", "observed µm", "slope m", "z", "Δ from avg µm"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.5)
        cell.get_text().set_color(TEXT)
        if r == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#081523")
            if c == 0:
                cell.get_text().set_color(POINT_COLORS[(r - 1) % len(POINT_COLORS)])
                cell.get_text().set_fontweight("bold")

    table_fig.tight_layout()
    table_path = PNG / f"{VERSION}_FIVE_POINT_LABELED_UV_SLOPES_TABLE.png"
    table_fig.savefig(
        table_path,
        dpi=270,
        facecolor=table_fig.get_facecolor(),
        bbox_inches="tight",
    )
    plt.show()
    plt.close(table_fig)

    return plot_path, table_path, audit_path, avg_slope, avg_z, closest


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    source_csv = locate_measurement_csv()
    measurements = load_measurements(source_csv)
    plot_path, table_path, audit_path, avg_slope, avg_z, closest = build_plot(
        measurements,
        source_csv,
    )

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            ("Source measurements", str(source_csv)),
            ("Measured UV points", len(measurements)),
            ("Average method", "sum observed wavelength / sum rest wavelength"),
            ("Gaussian average", "NO"),
            ("Average slope m", f"{avg_slope:.6f}"),
            ("Average z", f"{avg_z:.6f}"),
            (
                "Closest two points",
                ", ".join(f"#{int(r.n)} {r.line}" for r in closest.itertuples()),
            ),
            ("Labeled slope PNG", str(plot_path)),
            ("Styled table PNG", str(table_path)),
            ("Audit CSV", str(audit_path)),
        ],
        ["Field", "Value"],
    )
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
