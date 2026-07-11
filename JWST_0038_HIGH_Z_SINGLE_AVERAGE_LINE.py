# JWST_0038
# Audit: single orange average line only for high-z H-alpha/[N II] triplet. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0038"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen"),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha"),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen"),
]

REDSHIFTS = [
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
    for pkg in ["numpy", "pandas", "matplotlib", "astropy"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def observed(rest_um, z):
    return rest_um * (1.0 + z)


def build_tables():
    import pandas as pd
    line_rows = []
    summary_rows = []
    rest_values = [r[2] for r in TRIPLET]
    rest_sum = sum(rest_values)
    rest_avg = rest_sum / 3.0
    for name, z, status in REDSHIFTS:
        obs_values = []
        for n, line, rest_um, species in TRIPLET:
            obs_um = observed(rest_um, z)
            obs_values.append(obs_um)
            line_rows.append({
                "object_or_case": name,
                "z": z,
                "status": status,
                "line_number": n,
                "line": line,
                "species": species,
                "rest_um": rest_um,
                "observed_um": obs_um,
                "stretch_factor_1_plus_z": 1.0 + z,
            })
        obs_sum = sum(obs_values)
        obs_avg = obs_sum / 3.0
        summary_rows.append({
            "object_or_case": name,
            "z": z,
            "status": status,
            "rest_sum_um": rest_sum,
            "rest_average_um": rest_avg,
            "observed_sum_um": obs_sum,
            "observed_average_um": obs_avg,
            "stretch_from_averages": obs_avg / rest_avg,
            "z_from_averages": obs_avg / rest_avg - 1.0,
            "line_count": 3,
        })
    return pd.DataFrame(line_rows), pd.DataFrame(summary_rows)


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
    leg = ax.legend(loc=loc, fontsize=8.8, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_single_average_line(summary):
    import numpy as np
    import matplotlib.pyplot as plt
    rest_avg = sum(r[2] for r in TRIPLET) / 3.0
    zgrid = np.linspace(0.0, 15.5, 500)
    yavg = rest_avg * (1.0 + zgrid)

    fig, ax = plt.subplots(figsize=(16.4, 8.6))
    add_dark(ax)
    ax.plot(zgrid, yavg, color="#f97316", linewidth=2.55,
            label=f"one plotted line: triplet average λobs = {rest_avg:.6f} × (1+z)")

    for _, row in summary.iterrows():
        ax.scatter([row["z"]], [row["observed_average_um"]], s=82, color="#f97316", edgecolor="#f8fafc", zorder=7)
        ax.text(row["z"], row["observed_average_um"] + 0.18,
                f"{row['object_or_case']}\navg={row['observed_average_um']:.6f} µm",
                color="#fed7aa", fontsize=8.2, ha="center", va="bottom")

    ax.axvspan(0.6, 5.3, color="#64748b", alpha=0.07)
    ax.axvspan(5.0, 12.0, color="#64748b", alpha=0.10)
    ax.text(2.9, 10.65, "NIRSpec range context", color="#94a3b8", fontsize=8.4, ha="center")
    ax.text(8.5, 10.65, "MIRI/MRS range context", color="#94a3b8", fontsize=8.4, ha="center")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 11.2)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Average observed wavelength of the 3-line triplet, micron")
    ax.set_title(f"{VERSION} — single average line only: arithmetic mean of [N II], H-alpha, [N II]")
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SINGLE_AVERAGE_LINE_ONLY.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_single_average_positions(summary):
    import matplotlib.pyplot as plt
    order = list(summary["object_or_case"])
    y = list(range(len(order)))
    fig, ax = plt.subplots(figsize=(15.4, 7.6))
    add_dark(ax)
    ax.plot(summary["observed_average_um"], y, color="#f97316", marker="D", linewidth=1.8, markersize=7.2,
            label="single average point per object/case")
    for i, row in enumerate(summary.itertuples()):
        ax.text(row.observed_average_um, i + 0.12,
                f"Σ={row.observed_sum_um:.6f}\nμ={row.observed_average_um:.6f}",
                color="#fed7aa", fontsize=8.3, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(order, color="#dbeafe")
    ax.set_xlabel("Average observed wavelength of triplet, micron")
    ax.set_ylabel("Object / comparison case")
    ax.set_title(f"{VERSION} — one average marker per high-redshift case")
    ax.set_xlim(9.15, 10.45)
    legend(ax, "lower right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SINGLE_AVERAGE_POSITIONS.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_table(line_df, summary):
    import matplotlib.pyplot as plt
    mom_lines = line_df[line_df["object_or_case"] == "MoM-z14"].sort_values("line_number")
    mom = summary[summary["object_or_case"] == "MoM-z14"].iloc[0]
    rows = []
    for row in mom_lines.itertuples():
        rows.append([int(row.line_number), row.line, f"{row.rest_um:.6f}", f"{row.observed_um:.6f}"])
    rows.append(["Σ", "SUM", f"{mom.rest_sum_um:.6f}", f"{mom.observed_sum_um:.6f}"])
    rows.append(["μ", "AVERAGE", f"{mom.rest_average_um:.6f}", f"{mom.observed_average_um:.6f}"])
    rows.append(["z", "FROM AVERAGES", "", f"{mom.z_from_averages:.6f}"])

    fig, ax = plt.subplots(figsize=(15.8, 5.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — MoM-z14 sum and average used for the one orange line", color="#f8fafc", fontsize=14.0, pad=14)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "observed λ µm"], loc="center", cellLoc="center")
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
    path = PNG / f"{VERSION}_MOM_Z14_SINGLE_LINE_AVERAGE_TABLE.png"
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
    line_df, summary = build_tables()
    line_csv = CSV / f"{VERSION}_TRIPLET_LINE_VALUES.csv"
    summary_csv = CSV / f"{VERSION}_AVERAGE_LINE_SUMMARY.csv"
    line_df.to_csv(line_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    p1 = plot_single_average_line(summary)
    p2 = plot_single_average_positions(summary)
    p3 = plot_table(line_df, summary)

    mom_lines = line_df[line_df["object_or_case"] == "MoM-z14"].sort_values("line_number")
    mom = summary[summary["object_or_case"] == "MoM-z14"].iloc[0]

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("What happened before", "three separate spectral-line curves were plotted"),
        ("Corrected plot", "one orange line only: arithmetic average of the three triplet wavelengths"),
        ("Formula", "mean(lambda_observed) = mean(lambda_rest) * (1 + z)"),
        ("Mean rest wavelength um", f"{mom.rest_average_um:.6f}"),
        ("MoM-z14 z", f"{mom.z:.6f}"),
        ("MoM-z14 observed sum um", f"{mom.observed_sum_um:.6f}"),
        ("MoM-z14 observed average um", f"{mom.observed_average_um:.6f}"),
        ("z from averages", f"{mom.z_from_averages:.6f}"),
        ("Line CSV", str(line_csv)),
        ("Summary CSV", str(summary_csv)),
        ("Plot 1", str(p1)),
        ("Plot 2", str(p2)),
        ("Table PNG", str(p3)),
    ], ["Field", "Value"])

    print("\nMoM-z14 THREE VALUES, SUM, AVERAGE")
    rows = []
    for row in mom_lines.itertuples():
        rows.append((int(row.line_number), row.line, f"{row.rest_um:.6f}", f"{row.observed_um:.6f}"))
    rows.append(("Σ", "SUM", f"{mom.rest_sum_um:.6f}", f"{mom.observed_sum_um:.6f}"))
    rows.append(("μ", "AVERAGE", f"{mom.rest_average_um:.6f}", f"{mom.observed_average_um:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Observed um"])

    print("\nCOMMENTS")
    print("This version intentionally does not plot the three individual line curves.")
    print("The individual [N II], H-alpha, [N II] values appear only in the table and CSV.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
