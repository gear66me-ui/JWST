# JWST_0013
# Audit: Full JWST filter and wavelength-channel dashboard.
# Matplotlib only. No AI images. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0013"
PROJECT = "FULL JWST FILTER AND CHANNEL MAP"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OUTPUT_CSV = OUTPUT_DIR / "CSV"

# Curated Webb channel/filter roster for engineering visualization.
# Central wavelengths are inferred from filter names where applicable.
# Passband bars are class-width approximations, not official throughput curves.

NIRCAM_FILTERS = [
    ("NIRCam", "SW", "F070W", 0.70, "W", "wide"),
    ("NIRCam", "SW", "F090W", 0.90, "W", "wide"),
    ("NIRCam", "SW", "F115W", 1.15, "W", "wide"),
    ("NIRCam", "SW", "F140M", 1.40, "M", "medium"),
    ("NIRCam", "SW", "F150W", 1.50, "W", "wide"),
    ("NIRCam", "SW", "F150W2", 1.50, "W2", "wide-2 / blocking"),
    ("NIRCam", "SW", "F162M", 1.62, "M", "medium"),
    ("NIRCam", "SW", "F164N", 1.64, "N", "narrow"),
    ("NIRCam", "SW", "F182M", 1.82, "M", "medium"),
    ("NIRCam", "SW", "F187N", 1.87, "N", "narrow"),
    ("NIRCam", "SW", "F200W", 2.00, "W", "wide"),
    ("NIRCam", "SW", "F210M", 2.10, "M", "medium"),
    ("NIRCam", "SW", "F212N", 2.12, "N", "narrow"),
    ("NIRCam", "LW", "F250M", 2.50, "M", "medium"),
    ("NIRCam", "LW", "F277W", 2.77, "W", "wide"),
    ("NIRCam", "LW", "F300M", 3.00, "M", "medium"),
    ("NIRCam", "LW", "F322W2", 3.22, "W2", "wide-2 / grism blocking"),
    ("NIRCam", "LW", "F323N", 3.23, "N", "narrow"),
    ("NIRCam", "LW", "F335M", 3.35, "M", "medium"),
    ("NIRCam", "LW", "F356W", 3.56, "W", "wide"),
    ("NIRCam", "LW", "F360M", 3.60, "M", "medium"),
    ("NIRCam", "LW", "F405N", 4.05, "N", "narrow"),
    ("NIRCam", "LW", "F410M", 4.10, "M", "medium"),
    ("NIRCam", "LW", "F430M", 4.30, "M", "medium"),
    ("NIRCam", "LW", "F444W", 4.44, "W", "wide"),
    ("NIRCam", "LW", "F460M", 4.60, "M", "medium"),
    ("NIRCam", "LW", "F466N", 4.66, "N", "narrow"),
    ("NIRCam", "LW", "F470N", 4.70, "N", "narrow"),
    ("NIRCam", "LW", "F480M", 4.80, "M", "medium"),
]

NIRISS_FILTERS = [
    ("NIRISS", "Imaging", "F090W", 0.90, "W", "wide"),
    ("NIRISS", "Imaging", "F115W", 1.15, "W", "wide"),
    ("NIRISS", "Imaging", "F140M", 1.40, "M", "medium"),
    ("NIRISS", "Imaging", "F150W", 1.50, "W", "wide"),
    ("NIRISS", "Imaging", "F158M", 1.58, "M", "medium"),
    ("NIRISS", "Imaging", "F200W", 2.00, "W", "wide"),
    ("NIRISS", "Imaging", "F277W", 2.77, "W", "wide"),
    ("NIRISS", "Imaging", "F356W", 3.56, "W", "wide"),
    ("NIRISS", "Imaging", "F380M", 3.80, "M", "medium"),
    ("NIRISS", "Imaging", "F430M", 4.30, "M", "medium"),
    ("NIRISS", "Imaging", "F444W", 4.44, "W", "wide"),
    ("NIRISS", "Imaging", "F480M", 4.80, "M", "medium"),
]

MIRI_FILTERS = [
    ("MIRI", "Imaging", "F560W", 5.60, "W", "wide"),
    ("MIRI", "Imaging", "F770W", 7.70, "W", "wide / PAH"),
    ("MIRI", "Imaging", "F1000W", 10.00, "W", "wide / silicate"),
    ("MIRI", "Imaging", "F1130W", 11.30, "W", "wide / PAH"),
    ("MIRI", "Imaging", "F1280W", 12.80, "W", "wide"),
    ("MIRI", "Imaging", "F1500W", 15.00, "W", "wide"),
    ("MIRI", "Imaging", "F1800W", 18.00, "W", "wide / silicate"),
    ("MIRI", "Imaging", "F2100W", 21.00, "W", "wide"),
    ("MIRI", "Imaging", "F2550W", 25.50, "W", "wide"),
    ("MIRI", "Imaging", "F2550WR", 25.50, "W", "wide redundant"),
    ("MIRI", "Coronagraph", "F1065C", 10.65, "C", "coronagraphic"),
    ("MIRI", "Coronagraph", "F1140C", 11.40, "C", "coronagraphic"),
    ("MIRI", "Coronagraph", "F1550C", 15.50, "C", "coronagraphic"),
    ("MIRI", "Coronagraph", "F2300C", 23.00, "C", "coronagraphic"),
]

PHANGS_SELECTED = {"F200W", "F300M", "F335M", "F360M", "F770W", "F1000W", "F1130W", "F2100W"}

SPECTROSCOPY_MODES = [
    ("NIRCam", "Grism time-series / wide-field", 2.40, 5.00, "near-IR grism spectroscopy"),
    ("NIRISS", "WFSS", 0.80, 2.30, "wide-field slitless spectroscopy"),
    ("NIRISS", "SOSS", 0.60, 2.80, "single-object slitless spectroscopy"),
    ("NIRISS", "AMI", 2.80, 4.80, "aperture masking interferometry"),
    ("NIRSpec", "PRISM", 0.60, 5.30, "low-resolution spectroscopy"),
    ("NIRSpec", "Gratings", 0.70, 5.20, "medium/high-resolution spectroscopy"),
    ("MIRI", "LRS", 5.00, 14.00, "low-resolution spectroscopy"),
    ("MIRI", "MRS", 4.90, 27.90, "medium-resolution IFU spectroscopy"),
]

INSTRUMENT_RANGES = [
    ("NIRCam imaging", 0.60, 5.00, "NIRCam"),
    ("NIRISS imaging", 0.90, 4.80, "NIRISS"),
    ("NIRSpec spectroscopy", 0.60, 5.30, "NIRSpec"),
    ("MIRI imaging", 5.60, 25.50, "MIRI"),
    ("MIRI spectroscopy", 4.90, 27.90, "MIRI"),
]

CLASS_FRACTIONAL_WIDTH = {
    "N": 0.012,
    "M": 0.075,
    "W": 0.24,
    "W2": 0.45,
    "C": 0.055,
}


def ensure_package(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


def setup():
    ensure_package("numpy")
    ensure_package("pandas")
    ensure_package("matplotlib")
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)


def print_table(rows, headers):
    widths = []
    for i, header in enumerate(headers):
        width = max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        widths.append(min(width, 62))
    line = " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            text = str(value)
            if len(text) > widths[i]:
                text = text[:widths[i] - 1] + "…"
            cells.append(text.ljust(widths[i]))
        print(" | ".join(cells))


def build_filter_table():
    import pandas as pd
    rows = NIRCAM_FILTERS + NIRISS_FILTERS + MIRI_FILTERS
    df = pd.DataFrame(rows, columns=["instrument", "channel", "filter", "lambda_um", "class", "notes"])
    df["fractional_width_assumed"] = df["class"].map(CLASS_FRACTIONAL_WIDTH).fillna(0.10)
    df["bandwidth_um_approx"] = df["lambda_um"] * df["fractional_width_assumed"]
    df["lambda_min_um_approx"] = df["lambda_um"] - 0.5 * df["bandwidth_um_approx"]
    df["lambda_max_um_approx"] = df["lambda_um"] + 0.5 * df["bandwidth_um_approx"]
    df["phangs_selected"] = df["filter"].isin(PHANGS_SELECTED) & df["instrument"].isin(["NIRCam", "MIRI"])
    df["frequency_thz"] = 299.792458 / df["lambda_um"]
    return df


def build_spectroscopy_table():
    import pandas as pd
    df = pd.DataFrame(SPECTROSCOPY_MODES, columns=["instrument", "mode", "lambda_min_um", "lambda_max_um", "notes"])
    df["lambda_center_um"] = 0.5 * (df["lambda_min_um"] + df["lambda_max_um"])
    df["coverage_um"] = df["lambda_max_um"] - df["lambda_min_um"]
    return df


def style_dark(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.tick_params(colors="#dbeafe")
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def color_for_instrument(inst):
    return {
        "NIRCam": "#38bdf8",
        "NIRISS": "#a78bfa",
        "NIRSpec": "#22c55e",
        "MIRI": "#fb923c",
    }.get(inst, "#e5e7eb")


def plot_full_filter_map(filters):
    import matplotlib.pyplot as plt

    lane_order = [
        ("NIRCam", "SW", "NIRCam short wave"),
        ("NIRCam", "LW", "NIRCam long wave"),
        ("NIRISS", "Imaging", "NIRISS imaging"),
        ("MIRI", "Imaging", "MIRI imaging"),
        ("MIRI", "Coronagraph", "MIRI coronagraph"),
    ]
    lane_y = {label: i for i, (_, _, label) in enumerate(lane_order[::-1])}

    fig, ax = plt.subplots(figsize=(16.4, 9.2))
    style_dark(fig, ax)

    for inst, channel, label in lane_order:
        y = lane_y[label]
        sub = filters[(filters["instrument"] == inst) & (filters["channel"] == channel)].sort_values("lambda_um")
        for j, (_, row) in enumerate(sub.iterrows()):
            color = color_for_instrument(row["instrument"])
            lw = 5.0 if row["phangs_selected"] else 2.3
            alpha = 0.98 if row["phangs_selected"] else 0.68
            ax.hlines(y, row["lambda_min_um_approx"], row["lambda_max_um_approx"], color=color, linewidth=lw, alpha=alpha, zorder=3)
            ax.scatter(row["lambda_um"], y, s=70 if row["phangs_selected"] else 35,
                       color=color, edgecolor="#f8fafc", linewidth=0.55, zorder=4)
            if row["phangs_selected"] or j % 2 == 0:
                yoff = 0.20 if (j % 4 in [0, 1]) else -0.28
                va = "bottom" if yoff > 0 else "top"
                ax.text(row["lambda_um"], y + yoff, row["filter"], fontsize=7.6, color="#f8fafc",
                        ha="center", va=va,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="#020617", edgecolor="#475569", alpha=0.68))

    for start, end, label in [(0.6, 5.0, "near infrared"), (5.0, 28.0, "mid infrared")]:
        ax.axvspan(start, end, color="#0e7490" if start < 1 else "#7f1d1d", alpha=0.06)

    ax.set_xscale("log")
    ax.set_xlim(0.55, 31.0)
    ax.set_ylim(-0.75, len(lane_y) - 0.25)
    ax.set_yticks(list(lane_y.values()))
    ax.set_yticklabels(list(lane_y.keys()), color="#f8fafc")
    ax.set_xlabel("Wavelength, μm, log scale")
    ax.set_ylabel("JWST instrument / channel")
    ax.set_title("JWST imaging filter and coronagraph channel map\nhorizontal bars show approximate filter passband class; dots mark central wavelength")
    ax.text(0.012, 0.02,
            "Wide/medium/narrow passband widths are approximate engineering bars, not official throughput curves. Thick bars mark PHANGS-selected filters.",
            transform=ax.transAxes, color="#cbd5e1", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#020617", edgecolor="#475569", alpha=0.82))
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_FULL_JWST_FILTER_CHANNEL_MAP.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_spectroscopy_coverage(spec):
    import matplotlib.pyplot as plt

    spec = spec.copy()
    spec["label"] = spec["instrument"] + " " + spec["mode"]
    spec = spec.sort_values(["lambda_min_um", "lambda_max_um"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(15.4, 8.4))
    style_dark(fig, ax)

    for i, row in spec.iterrows():
        y = len(spec) - 1 - i
        color = color_for_instrument(row["instrument"])
        ax.hlines(y, row["lambda_min_um"], row["lambda_max_um"], color=color, linewidth=8.5, alpha=0.86)
        ax.scatter([row["lambda_min_um"], row["lambda_max_um"]], [y, y], s=44, color=color, edgecolor="#f8fafc", linewidth=0.55)
        ax.text(row["lambda_max_um"] * 1.03, y, f"{row['lambda_min_um']:.1f}-{row['lambda_max_um']:.1f} μm",
                color="#f8fafc", va="center", fontsize=8.2)

    ax.axvline(3.60, color="#60a5fa", linestyle="--", linewidth=1.0, alpha=0.70)
    ax.axvline(7.70, color="#fb923c", linestyle="--", linewidth=1.0, alpha=0.70)
    ax.text(3.60, len(spec) - 0.15, "PHANGS F360M", color="#bfdbfe", fontsize=8, ha="center", va="top", rotation=90)
    ax.text(7.70, len(spec) - 0.15, "PHANGS F770W", color="#fed7aa", fontsize=8, ha="center", va="top", rotation=90)

    ax.set_xscale("log")
    ax.set_xlim(0.52, 32.0)
    ax.set_ylim(-0.8, len(spec) - 0.1)
    ax.set_yticks(list(range(len(spec))))
    ax.set_yticklabels(spec["label"].iloc[::-1], color="#f8fafc")
    ax.set_xlabel("Wavelength, μm, log scale")
    ax.set_ylabel("Spectroscopic / interferometric mode")
    ax.set_title("JWST wavelength coverage by observing mode\nWebb is not blind between 3.6 and 7.7 μm; PHANGS simply did not select every intermediate channel")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_JWST_SPECTROSCOPY_COVERAGE.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_filter_counts(filters):
    import matplotlib.pyplot as plt
    import numpy as np

    bins = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
    labels = ["0.5-1", "1-2", "2-3", "3-5", "5-8", "8-12", "12-20", "20-30"]
    rows = []
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        sub = filters[(filters["lambda_um"] >= lo) & (filters["lambda_um"] < hi)]
        rows.append((labels[i], len(sub)))

    fig, ax = plt.subplots(figsize=(12.4, 6.8))
    style_dark(fig, ax)
    x = np.arange(len(rows))
    counts = [r[1] for r in rows]
    ax.bar(x, counts, edgecolor="#cbd5e1", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], color="#f8fafc")
    ax.set_xlabel("Wavelength bin, μm")
    ax.set_ylabel("Number of listed imaging / coronagraph filters")
    ax.set_title("JWST filter density by wavelength bin\nfilter roster concentration across near-IR and mid-IR")
    for xi, count in zip(x, counts):
        ax.text(xi, count + 0.25, str(count), color="#f8fafc", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.70))
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_JWST_FILTER_DENSITY_COUNTS.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_summary_table(filters, spec):
    import matplotlib.pyplot as plt

    rows = []
    for instrument in ["NIRCam", "NIRISS", "MIRI"]:
        sub = filters[filters["instrument"] == instrument]
        rows.append([
            instrument,
            f"{len(sub)} filters/channels",
            f"{sub['lambda_um'].min():.2f}-{sub['lambda_um'].max():.2f}",
            ", ".join(sorted(sub["channel"].unique())),
        ])
    rows.append(["NIRSpec", "spectroscopy", "0.60-5.30", "PRISM, gratings, MOS, fixed slit, IFU"])
    rows.append(["MIRI spectroscopy", "LRS / MRS", "4.90-27.90", "low and medium resolution"])

    fig, ax = plt.subplots(figsize=(13.2, 4.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    labels = ["Instrument", "What is plotted", "Coverage (μm)", "Channels / modes"]
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.9)
    table.scale(1.0, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.65)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            inst = rows[r-1][0]
            base = {"NIRCam": "#082f49", "NIRISS": "#312e81", "MIRI": "#3b1114", "NIRSpec": "#052e16", "MIRI spectroscopy": "#3b1114"}.get(inst, "#172554")
            cell.set_facecolor(base)
            cell.get_text().set_color("#e5e7eb")
    ax.set_title("JWST filter/channel summary\ncentral wavelengths and approximate filter-class passbands; no image data downloaded",
                 color="#f8fafc", fontsize=13.2, pad=16)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_JWST_CHANNEL_SUMMARY_TABLE.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    import pandas as pd

    filters = build_filter_table()
    spec = build_spectroscopy_table()
    ranges = pd.DataFrame(INSTRUMENT_RANGES, columns=["label", "lambda_min_um", "lambda_max_um", "instrument"])

    filters_csv = OUTPUT_CSV / f"{VERSION}_JWST_FILTER_ROSTER.csv"
    spec_csv = OUTPUT_CSV / f"{VERSION}_JWST_SPECTROSCOPY_MODES.csv"
    ranges_csv = OUTPUT_CSV / f"{VERSION}_JWST_INSTRUMENT_RANGES.csv"
    filters.to_csv(filters_csv, index=False)
    spec.to_csv(spec_csv, index=False)
    ranges.to_csv(ranges_csv, index=False)

    filter_png = plot_full_filter_map(filters)
    spec_png = plot_spectroscopy_coverage(spec)
    density_png = plot_filter_counts(filters)
    table_png = styled_summary_table(filters, spec)

    gap_filters = filters[(filters["lambda_um"] > 3.60) & (filters["lambda_um"] < 7.70)][["instrument", "channel", "filter", "lambda_um"]].sort_values("lambda_um")

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Imaging/coronagraph filters plotted", f"{len(filters):,}"),
        ("Spectroscopy modes plotted", f"{len(spec):,}"),
        ("Passband bars", "approximate by filter class, not official throughput"),
        ("Image/FITS downloads", "none"),
        ("Plot engine", "matplotlib only"),
    ], ["Field", "Value"])
    print("")
    print("RESULTS")
    print_table([
        ("Shortest listed filter", f"{filters.loc[filters['lambda_um'].idxmin(), 'filter']} | {filters['lambda_um'].min():.2f} μm"),
        ("Longest listed filter", f"{filters.loc[filters['lambda_um'].idxmax(), 'filter']} | {filters['lambda_um'].max():.2f} μm"),
        ("Full spectroscopy maximum", f"{spec['lambda_min_um'].min():.2f}-{spec['lambda_max_um'].max():.2f} μm"),
        ("Filters between 3.60 and 7.70 μm", f"{len(gap_filters)} listed channels"),
        ("PHANGS gap explanation", "PHANGS selected F360M then F770W; JWST has intermediate channels"),
    ], ["Metric", "Value"])
    print("")
    print("INTERMEDIATE CHANNELS BETWEEN 3.60 AND 7.70 μm")
    print_table([
        (r.instrument, r.channel, r.filter, f"{r.lambda_um:.2f}") for r in gap_filters.itertuples(index=False)
    ], ["Instrument", "Channel", "Filter", "λ μm"])
    print("")
    print("OUTPUT SUMMARY")
    print_table([
        ("png", str(filter_png)),
        ("png", str(spec_png)),
        ("png", str(density_png)),
        ("png", str(table_png)),
        ("csv", str(filters_csv)),
        ("csv", str(spec_csv)),
        ("csv", str(ranges_csv)),
    ], ["Type", "Path"])
    print("")
    print("COMMENTS")
    print("Dots mark central wavelength. Horizontal bars mark approximate filter passband class.")
    print("A filter passband is not a detection-error bar; it is the wavelength window admitted by the filter.")
    print("This script is a channel map and metadata dashboard, not a throughput calibration product.")
    print("Matplotlib only. No AI images.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
