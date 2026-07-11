# JWST_0050
# Fixed-canvas five-point UV redshift-slope plot.
# No AI images. Matplotlib only. Reuses saved real MAST UV measurements.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0050"
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
LABEL_OFFSETS = [(16, 18), (16, -34), (16, 18), (16, -34), (-178, 18)]


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


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
        "No saved UV line-measurement CSV was found. Run JWST_0048 first."
    )


def load_measurements(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    required = [
        "n",
        "line",
        "rest_um",
        "raw_peak_sample_um",
        "sample_count_in_window",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing measurement columns: {missing}")

    valid = frame[
        (frame["sample_count_in_window"] > 0)
        & np.isfinite(frame["rest_um"])
        & np.isfinite(frame["raw_peak_sample_um"])
    ].copy()
    valid = valid.sort_values("n").reset_index(drop=True)

    if valid.empty:
        raise RuntimeError("No finite measured UV points are available.")

    valid["slope_m"] = valid["raw_peak_sample_um"] / valid["rest_um"]
    valid["z_from_raw_peak"] = valid["slope_m"] - 1.0
    return valid


def style_axis(ax):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.52)
    ax.tick_params(colors=TEXT, labelsize=9.5)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.85)


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def build_plot(meas, source_csv):
    import numpy as np
    import matplotlib.pyplot as plt

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
    meas["delta_from_average_percent"] = (
        100.0 * meas["delta_from_average_um"] / meas["average_line_obs_um"]
    )

    xmin = float(meas["rest_um"].min() - 0.006)
    xmax = float(meas["rest_um"].max() + 0.006)
    xs = np.linspace(xmin, xmax, 500)

    all_y = []
    for row in meas.itertuples():
        all_y.extend([row.slope_m * xmin, row.slope_m * xmax])
    all_y.extend([avg_slope * xmin, avg_slope * xmax])
    ymin = float(min(all_y))
    ymax = float(max(all_y))
    ypad = max(0.012, 0.085 * (ymax - ymin))

    fig = plt.figure(figsize=(16.0, 9.0), facecolor=BG)
    ax = fig.add_axes([0.075, 0.105, 0.89, 0.80])
    style_axis(ax)

    slopes = meas["slope_m"].to_numpy(float)
    ax.fill_between(
        xs,
        slopes.min() * xs,
        slopes.max() * xs,
        color="#35556f",
        alpha=0.18,
        label="span of the five individual slopes",
    )

    for index, row in enumerate(meas.itertuples()):
        color = POINT_COLORS[index % len(POINT_COLORS)]
        ax.plot(
            xs,
            row.slope_m * xs,
            color=color,
            linewidth=0.95,
            alpha=0.92,
            label=f"{int(row.n)} {row.line}: m={row.slope_m:.6f}, z={row.z_from_raw_peak:.6f}",
        )
        ax.scatter(
            [row.rest_um],
            [row.raw_peak_sample_um],
            s=88,
            color=color,
            edgecolor="#f8fbff",
            linewidth=0.9,
            zorder=8,
        )

        dx, dy = LABEL_OFFSETS[index % len(LABEL_OFFSETS)]
        label = (
            f"{int(row.n)}  {row.line}\n"
            f"λrest={row.rest_um:.6f} µm\n"
            f"λobs={row.raw_peak_sample_um:.6f} µm\n"
            f"z={row.z_from_raw_peak:.6f}\n"
            f"Δavg={row.delta_from_average_um:+.6f} µm"
        )
        ax.annotate(
            label,
            xy=(row.rest_um, row.raw_peak_sample_um),
            xytext=(dx, dy),
            textcoords="offset points",
            color=TEXT,
            fontsize=8.1,
            ha="left",
            va="bottom" if dy >= 0 else "top",
            bbox=dict(
                boxstyle="round,pad=0.32",
                facecolor="#07111f",
                edgecolor=color,
                alpha=0.95,
            ),
            arrowprops=dict(
                arrowstyle="-",
                color=color,
                linewidth=0.8,
                alpha=0.95,
            ),
            zorder=10,
        )

    ax.plot(
        xs,
        avg_slope * xs,
        color=ORANGE,
        linewidth=2.4,
        label=f"orange average: m={avg_slope:.6f}, z={avg_z:.6f}",
        zorder=6,
    )

    closest = meas.nsmallest(2, "abs_delta_from_average_um")
    closest_text = ", ".join(
        f"#{int(row.n)} {row.line}" for row in closest.itertuples()
    )

    explanation = (
        "ORANGE LINE\n"
        "Ratio-of-means, not Gaussian:\n"
        "m = Σλobs / Σλrest\n"
        f"m = {avg_slope:.6f}\n"
        f"z = {avg_z:.6f}\n"
        f"Closest points: {closest_text}"
    )
    ax.text(
        0.985,
        0.025,
        explanation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=TEXT,
        fontsize=8.7,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor="#07111f",
            edgecolor=ORANGE,
            alpha=0.96,
        ),
        zorder=20,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin - ypad, ymax + ypad)
    ax.set_xlabel("Rest-frame wavelength, µm", fontsize=10.8)
    ax.set_ylabel("Observed raw peak wavelength, µm", fontsize=10.8)
    ax.set_title(
        f"{VERSION} — five measured UV points, five slopes, and orange average",
        fontsize=13.8,
        pad=13,
    )

    legend = ax.legend(
        loc="upper left",
        fontsize=7.5,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.97,
        ncol=1,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    plot_path = PNG / f"{VERSION}_FIXED_FIVE_POINT_LABELED_UV_SLOPES.png"
    fig.savefig(
        plot_path,
        dpi=240,
        facecolor=BG,
        edgecolor=BG,
    )
    plt.show()
    plt.close(fig)

    audit_columns = [
        "n",
        "line",
        "rest_um",
        "raw_peak_sample_um",
        "slope_m",
        "z_from_raw_peak",
        "average_line_obs_um",
        "delta_from_average_um",
        "delta_from_average_percent",
        "abs_delta_from_average_um",
    ]
    audit = meas[audit_columns].copy()
    audit.insert(0, "source_csv", str(source_csv))
    audit["average_method"] = "ratio of means = sum observed wavelength / sum rest wavelength"
    audit["average_slope_m"] = avg_slope
    audit["average_z"] = avg_z
    audit_path = CSV / f"{VERSION}_FIXED_FIVE_POINT_LABELED_UV_SLOPES.csv"
    audit.to_csv(audit_path, index=False)

    return plot_path, audit_path, avg_slope, avg_z, closest


def main():
    for package in ["numpy", "pandas", "matplotlib"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    source_csv = locate_measurement_csv()
    measurements = load_measurements(source_csv)
    plot_path, audit_path, avg_slope, avg_z, closest = build_plot(
        measurements,
        source_csv,
    )

    print(f"CODE OUTPUT: {VERSION}\n")
    rows = []
    for row in measurements.itertuples():
        rows.append(
            (
                int(row.n),
                row.line,
                f"{row.rest_um:.6f}",
                f"{row.raw_peak_sample_um:.6f}",
                f"{row.slope_m:.6f}",
                f"{row.z_from_raw_peak:.6f}",
            )
        )
    print_table(rows, ["#", "Line", "Rest µm", "Observed µm", "Slope m", "z"])
    print()
    print_table(
        [
            ("Source CSV", str(source_csv)),
            ("Measured points", len(measurements)),
            ("Average method", "ratio of means; not Gaussian"),
            ("Average slope", f"{avg_slope:.6f}"),
            ("Average z", f"{avg_z:.6f}"),
            (
                "Closest points",
                ", ".join(
                    f"#{int(row.n)} {row.line}" for row in closest.itertuples()
                ),
            ),
            ("Plot PNG", str(plot_path)),
            ("Audit CSV", str(audit_path)),
        ],
        ["Field", "Value"],
    )
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
