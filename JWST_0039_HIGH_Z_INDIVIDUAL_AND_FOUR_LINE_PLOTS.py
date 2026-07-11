# JWST_0039
# Audit: High-z triplet: individual spectral-marker plots, combined spectral marker plot, and four-line z plot. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0039"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

TARGET_NAME = "MoM-z14"
TARGET_Z = 14.44
TARGET_STATUS = "reported JWST/NIRSpec frontier galaxy; H-alpha/[N II] positions here are predicted from z, not measured H-alpha detections"

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen"),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha"),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen"),
]

REDSHIFT_CASES = [
    ("z=13 comparison", 13.0000, "comparison"),
    ("JADES-GS-z14-1", 13.9000, "reported high-z JWST galaxy"),
    ("JADES-GS-z14-0", 14.1793, "reported high-z JWST galaxy / ALMA precision redshift"),
    ("MoM-z14", 14.4400, "reported JWST/NIRSpec frontier galaxy"),
    ("z=15 comparison", 15.0000, "comparison"),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def obs(rest_um, z):
    return rest_um * (1.0 + z)


def build_tables():
    import pandas as pd
    rows = []
    for n, line, rest_um, species in TRIPLET:
        observed_um = obs(rest_um, TARGET_Z)
        rows.append({
            "target": TARGET_NAME,
            "z": TARGET_Z,
            "status": TARGET_STATUS,
            "line_number": n,
            "line": line,
            "species": species,
            "rest_um": rest_um,
            "observed_um": observed_um,
            "slope_1_plus_z": 1.0 + TARGET_Z,
            "z_from_line": observed_um / rest_um - 1.0,
        })
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "target": TARGET_NAME,
        "z": TARGET_Z,
        "line_count": len(TRIPLET),
        "rest_sum_um": df["rest_um"].sum(),
        "rest_average_um": df["rest_um"].mean(),
        "observed_sum_um": df["observed_um"].sum(),
        "observed_average_um": df["observed_um"].mean(),
        "stretch_from_averages": df["observed_um"].mean() / df["rest_um"].mean(),
        "z_from_averages": df["observed_um"].mean() / df["rest_um"].mean() - 1.0,
        "note": "Average line is a mathematical average of the three predicted observed wavelengths; it is not a physical spectral feature.",
    }])
    return df, summary


def add_dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
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


def plot_individual_spectral_markers(df):
    import matplotlib.pyplot as plt
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    paths = []
    for idx, row in enumerate(df.itertuples()):
        fig, ax = plt.subplots(figsize=(14.4, 4.8))
        add_dark(ax)
        center = row.observed_um
        ax.set_xlim(center - 0.085, center + 0.085)
        ax.set_ylim(0, 1)
        ax.axvline(center, color=colors[idx], linewidth=2.0, label=f"{row.line}: {center:.6f} µm")
        ax.scatter([center], [0.56], s=120, color=colors[idx], edgecolor="#f8fafc", zorder=5)
        ax.text(center, 0.78, f"{int(row.line_number)}\n{row.line}\nλrest={row.rest_um:.6f} µm\nλobs={row.observed_um:.6f} µm",
                color="#f8fafc", fontsize=10, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.42", facecolor="#020617", edgecolor=colors[idx], alpha=0.92))
        ax.set_yticks([])
        ax.set_xlabel("Predicted observed wavelength, micron")
        ax.set_title(f"{VERSION} — individual spectral-marker plot {int(row.line_number)} for {TARGET_NAME}, z={TARGET_Z:.4f}")
        legend(ax, "upper right")
        fig.tight_layout()
        safe_line = row.line.replace("[", "").replace("]", "").replace(" ", "_").replace("-", "_")
        path = PNG / f"{VERSION}_INDIVIDUAL_SPECTRAL_MARKER_{int(row.line_number)}_{safe_line}.png"
        fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
        plt.show()
        paths.append(path)
    return paths


def plot_combined_spectral_markers(df, summary):
    import matplotlib.pyplot as plt
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    avg = float(summary.iloc[0]["observed_average_um"])
    xmin = min(df["observed_um"].min(), avg) - 0.055
    xmax = max(df["observed_um"].max(), avg) + 0.055
    fig, ax = plt.subplots(figsize=(15.8, 5.8))
    add_dark(ax)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    for idx, row in enumerate(df.itertuples()):
        ax.axvline(row.observed_um, color=colors[idx], linewidth=1.65, alpha=0.92, label=f"{int(row.line_number)} {row.line}: {row.observed_um:.6f} µm")
        ax.scatter([row.observed_um], [0.58], s=96, color=colors[idx], edgecolor="#f8fafc", zorder=6)
        ax.text(row.observed_um, 0.90, f"{int(row.line_number)}\n{row.line}", color=colors[idx], fontsize=9.5, ha="center", va="top")
    ax.axvline(avg, color="#f97316", linewidth=2.50, alpha=0.96, label=f"orange average marker: {avg:.6f} µm")
    ax.scatter([avg], [0.36], s=132, marker="D", color="#f97316", edgecolor="#f8fafc", zorder=7)
    ax.text(avg, 0.24, f"AVG\n{avg:.6f} µm", color="#fed7aa", fontsize=10.5, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#020617", edgecolor="#f97316", alpha=0.92))
    ax.set_yticks([])
    ax.set_xlabel("Predicted observed wavelength, micron")
    ax.set_title(f"{VERSION} — spectral-marker plot: three predicted lines plus one average marker")
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMBINED_SPECTRAL_MARKERS_THREE_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_four_lines_vs_z(df, summary):
    import numpy as np
    import matplotlib.pyplot as plt
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    zgrid = np.linspace(12.8, 15.2, 400)
    fig, ax = plt.subplots(figsize=(16.2, 8.8))
    add_dark(ax)
    for idx, row in enumerate(df.itertuples()):
        ax.plot(zgrid, row.rest_um * (1.0 + zgrid), color=colors[idx], linewidth=1.35,
                label=f"{int(row.line_number)} {row.line}")
    rest_avg = float(summary.iloc[0]["rest_average_um"])
    ax.plot(zgrid, rest_avg * (1.0 + zgrid), color="#f97316", linewidth=2.65,
            label=f"orange average curve: mean rest λ={rest_avg:.6f} µm")
    for name, z, status in REDSHIFT_CASES:
        if 12.8 <= z <= 15.2:
            yavg = rest_avg * (1.0 + z)
            ax.scatter([z], [yavg], s=62, color="#f97316", edgecolor="#f8fafc", zorder=7)
            ax.text(z, yavg + 0.055, name, color="#fed7aa", fontsize=8.0, rotation=90, ha="center", va="bottom")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Predicted observed wavelength, micron")
    ax.set_title(f"{VERSION} — four-line plot: 3 triplet lines plus 1 arithmetic-average line")
    ax.set_xlim(12.8, 15.2)
    ymin = min([r.rest_um * (1 + 12.8) for r in df.itertuples()] + [rest_avg * (1 + 12.8)]) - 0.05
    ymax = max([r.rest_um * (1 + 15.2) for r in df.itertuples()] + [rest_avg * (1 + 15.2)]) + 0.10
    ax.set_ylim(ymin, ymax)
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FOUR_LINES_VS_Z_THREE_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_sum_average_table(df, summary):
    import matplotlib.pyplot as plt
    s = summary.iloc[0]
    rows = []
    for row in df.itertuples():
        rows.append([int(row.line_number), row.line, f"{row.rest_um:.6f}", f"{row.observed_um:.6f}", f"{row.z_from_line:.6f}"])
    rows.append(["Σ", "SUM", f"{s.rest_sum_um:.6f}", f"{s.observed_sum_um:.6f}", ""])
    rows.append(["μ", "AVERAGE", f"{s.rest_average_um:.6f}", f"{s.observed_average_um:.6f}", f"{s.z_from_averages:.6f}"])
    fig, ax = plt.subplots(figsize=(16.4, 5.7))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — table behind the average marker and average curve", color="#f8fafc", fontsize=14.0, pad=14)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "observed λ µm", "z check"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#020617" if r % 2 else "#0f172a")
            cell.get_text().set_color("#dbeafe")
            if r >= 4:
                cell.set_facecolor("#431407")
                cell.get_text().set_color("#fed7aa")
                cell.get_text().set_weight("bold")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SUM_AVERAGE_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


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
    df, summary = build_tables()
    line_csv = CSV / f"{VERSION}_LINE_VALUES.csv"
    summary_csv = CSV / f"{VERSION}_SUM_AVERAGE_SUMMARY.csv"
    df.to_csv(line_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    individual_paths = plot_individual_spectral_markers(df)
    combined_path = plot_combined_spectral_markers(df, summary)
    four_line_path = plot_four_lines_vs_z(df, summary)
    table_path = plot_sum_average_table(df, summary)

    s = summary.iloc[0]
    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Previous issue", "plot mixed the one-average-line request with the three-individual-line request"),
        ("Plot set A", "three separate individual spectral-marker plots"),
        ("Plot set B", "one combined spectral-marker plot with 3 lines + average marker"),
        ("Plot set C", "four-line z plot: 3 triplet curves + 1 average curve"),
        ("Target", TARGET_NAME),
        ("z", f"{TARGET_Z:.6f}"),
        ("Observed sum um", f"{s.observed_sum_um:.6f}"),
        ("Observed average um", f"{s.observed_average_um:.6f}"),
        ("z from averages", f"{s.z_from_averages:.6f}"),
        ("Line CSV", str(line_csv)),
        ("Summary CSV", str(summary_csv)),
        ("Combined spectral plot", str(combined_path)),
        ("Four-line z plot", str(four_line_path)),
        ("Table PNG", str(table_path)),
    ], ["Field", "Value"])

    print("\nINDIVIDUAL SPECTRAL PLOTS")
    print_table([(i + 1, str(p)) for i, p in enumerate(individual_paths)], ["#", "Path"])

    print("\nLINE VALUES, SUM, AVERAGE")
    rows = []
    for row in df.itertuples():
        rows.append((int(row.line_number), row.line, f"{row.rest_um:.6f}", f"{row.observed_um:.6f}", f"{row.z_from_line:.6f}"))
    rows.append(("Σ", "SUM", f"{s.rest_sum_um:.6f}", f"{s.observed_sum_um:.6f}", ""))
    rows.append(("μ", "AVERAGE", f"{s.rest_average_um:.6f}", f"{s.observed_average_um:.6f}", f"{s.z_from_averages:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Observed um", "z check"])

    print("\nCOMMENTS")
    print("No actual high-z H-alpha spectrum is downloaded here; these are predicted line positions from the reported redshift.")
    print("The average marker/curve is a mathematical arithmetic average, not a physical fourth spectral line.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
