# JWST_0012
# Audit: PHANGS-JWST metadata dashboard.
# Matplotlib only. No AI images. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0012"
PROJECT = "PHANGS-JWST FILTER DASHBOARD"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OUTPUT_CSV = OUTPUT_DIR / "CSV"

# PHANGS-JWST Cycle 1 Treasury used eight-band 2-21 micron imaging for 19 nearby galaxies.
# This dashboard uses a curated metadata table only; it does not download FITS or image products.
FILTERS = [
    ("F200W", "NIRCam", 2.00, "stellar continuum / old stars"),
    ("F300M", "NIRCam", 3.00, "3 micron continuum"),
    ("F335M", "NIRCam", 3.35, "3.3 micron PAH feature"),
    ("F360M", "NIRCam", 3.60, "PAH continuum removal"),
    ("F770W", "MIRI", 7.70, "aromatic dust / PAH"),
    ("F1000W", "MIRI", 10.00, "warm dust continuum"),
    ("F1130W", "MIRI", 11.30, "aromatic dust / PAH"),
    ("F2100W", "MIRI", 21.00, "warm dust / embedded SF"),
]

# Approximate PHANGS-JWST Cycle 1 galaxy sample distances, used only for dashboard scale/resolution estimates.
# Distances are rounded metadata values from common PHANGS literature usage; exact values vary by calibration/source.
GALAXIES = [
    ("NGC 5068", 5.20),
    ("IC 5332", 9.01),
    ("NGC 0628", 9.84),
    ("NGC 3351", 9.96),
    ("NGC 3627", 11.32),
    ("NGC 2835", 12.22),
    ("NGC 4254", 13.10),
    ("NGC 4321", 15.21),
    ("NGC 4535", 15.77),
    ("NGC 1087", 15.85),
    ("NGC 4303", 16.99),
    ("NGC 1385", 17.22),
    ("NGC 1566", 17.69),
    ("NGC 1433", 18.63),
    ("NGC 7496", 18.72),
    ("NGC 1512", 18.83),
    ("NGC 1300", 19.00),
    ("NGC 1672", 19.40),
    ("NGC 1365", 19.57),
]

JWST_DIAMETER_M = 6.5
ARCSEC_PER_RAD = 206264.80624709636
PC_PER_ARCSEC_PER_MPC = 4.84813681109536


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
        widths.append(min(width, 58))
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


def build_tables():
    import pandas as pd
    filters = pd.DataFrame(FILTERS, columns=["filter", "instrument", "lambda_um", "science_use"])
    filters["diffraction_fwhm_arcsec"] = 1.22 * filters["lambda_um"] * 1e-6 / JWST_DIAMETER_M * ARCSEC_PER_RAD
    galaxies = pd.DataFrame(GALAXIES, columns=["galaxy", "distance_mpc"])
    galaxies["pc_per_arcsec"] = galaxies["distance_mpc"] * PC_PER_ARCSEC_PER_MPC
    galaxies["rank_distance"] = galaxies["distance_mpc"].rank(method="first").astype(int)

    rows = []
    for _, g in galaxies.iterrows():
        for _, f in filters.iterrows():
            rows.append({
                "galaxy": g["galaxy"],
                "distance_mpc": g["distance_mpc"],
                "filter": f["filter"],
                "instrument": f["instrument"],
                "lambda_um": f["lambda_um"],
                "diffraction_fwhm_arcsec": f["diffraction_fwhm_arcsec"],
                "physical_resolution_pc": f["diffraction_fwhm_arcsec"] * g["pc_per_arcsec"],
            })
    grid = pd.DataFrame(rows)
    return filters, galaxies, grid


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


def instrument_color(instrument):
    return "#38bdf8" if instrument == "NIRCam" else "#fb923c"


def plot_filter_ladder(filters):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14.2, 7.4))
    style_dark(fig, ax)

    y_map = {"NIRCam": 1.0, "MIRI": 0.0}
    for _, row in filters.iterrows():
        y = y_map[row["instrument"]]
        color = instrument_color(row["instrument"])
        ax.scatter(row["lambda_um"], y, s=310, color=color, edgecolor="#f8fafc", linewidth=0.9, zorder=5)
        ax.vlines(row["lambda_um"], y - 0.20, y + 0.20, color=color, linewidth=2.0, alpha=0.86)
        label_y = y + (0.34 if row["instrument"] == "NIRCam" else -0.36)
        va = "bottom" if row["instrument"] == "NIRCam" else "top"
        ax.text(row["lambda_um"], label_y, f"{row['filter']}\n{row['lambda_um']:.2f} μm",
                color="#f8fafc", fontsize=9, ha="center", va=va,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#020617", edgecolor="#475569", alpha=0.78))

    ax.axvspan(2.0, 5.0, color="#0e7490", alpha=0.13, label="NIRCam PHANGS window")
    ax.axvspan(7.0, 22.0, color="#7f1d1d", alpha=0.15, label="MIRI PHANGS window")
    ax.set_xscale("log")
    ax.set_xlim(1.65, 25)
    ax.set_ylim(-0.85, 1.85)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["MIRI", "NIRCam"], color="#f8fafc")
    ax.set_xlabel("Filter central wavelength, μm, log scale")
    ax.set_ylabel("JWST instrument")
    ax.set_title("PHANGS-JWST eight-band filter ladder\nnear-IR stars/PAH continuum to mid-IR dust emission")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", loc="upper right")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_PHANGS_FILTER_LADDER.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_resolution_curves(filters, galaxies, grid):
    import matplotlib.pyplot as plt

    nearest = galaxies.loc[galaxies["distance_mpc"].idxmin()]
    median_distance = galaxies["distance_mpc"].median()
    farthest = galaxies.loc[galaxies["distance_mpc"].idxmax()]

    def res_for(distance_mpc):
        return filters["diffraction_fwhm_arcsec"] * distance_mpc * PC_PER_ARCSEC_PER_MPC

    fig, ax = plt.subplots(figsize=(13.8, 8.0))
    style_dark(fig, ax)

    ax.plot(filters["lambda_um"], res_for(nearest["distance_mpc"]), marker="o", linewidth=2.2, label=f"nearest: {nearest['galaxy']} ({nearest['distance_mpc']:.1f} Mpc)")
    ax.plot(filters["lambda_um"], res_for(median_distance), marker="o", linewidth=2.2, label=f"median sample distance ({median_distance:.1f} Mpc)")
    ax.plot(filters["lambda_um"], res_for(farthest["distance_mpc"]), marker="o", linewidth=2.2, label=f"farthest: {farthest['galaxy']} ({farthest['distance_mpc']:.1f} Mpc)")

    for _, row in filters.iterrows():
        ax.text(row["lambda_um"], float(res_for(median_distance)[filters.index[filters["filter"] == row["filter"]][0]]) * 1.08,
                row["filter"], color="#f8fafc", fontsize=8, ha="center",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.65))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.7, 24)
    ax.set_xlabel("Filter central wavelength, μm, log scale")
    ax.set_ylabel("Approximate diffraction-limited physical scale, pc")
    ax.set_title("PHANGS-JWST physical resolution dashboard\ncomputed from JWST diffraction scale and galaxy distance metadata")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9, loc="upper left")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_PHANGS_RESOLUTION_CURVES.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_galaxy_distance_bars(galaxies):
    import matplotlib.pyplot as plt

    g = galaxies.sort_values("distance_mpc", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11.8, 9.2))
    style_dark(fig, ax)
    bars = ax.barh(g["galaxy"], g["distance_mpc"], edgecolor="#cbd5e1", linewidth=0.55)
    ax.set_xlabel("Distance, Mpc")
    ax.set_ylabel("PHANGS-JWST Cycle 1 galaxy")
    ax.set_title("PHANGS-JWST 19-galaxy distance ladder\nnearby-galaxy metadata used for dashboard resolution estimates")
    ax.invert_yaxis()
    for bar, (_, row) in zip(bars, g.iterrows()):
        ax.text(row["distance_mpc"] + 0.25, bar.get_y() + bar.get_height()/2,
                f"{row['distance_mpc']:.2f} Mpc | {row['pc_per_arcsec']:.1f} pc/arcsec",
                color="#f8fafc", va="center", fontsize=8.2)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_PHANGS_GALAXY_DISTANCE_LADDER.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_filter_table(filters):
    import matplotlib.pyplot as plt

    rows = []
    for _, row in filters.iterrows():
        rows.append([
            row["filter"],
            row["instrument"],
            f"{row['lambda_um']:.2f}",
            f"{row['diffraction_fwhm_arcsec']:.3f}",
            row["science_use"],
        ])

    fig, ax = plt.subplots(figsize=(13.6, 4.6))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    labels = ["Filter", "Instrument", "λ (μm)", "JWST FWHM (arcsec)", "Science use"]
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.65)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            inst = rows[r - 1][1]
            cell.set_facecolor("#082f49" if inst == "NIRCam" else "#3b1114")
            cell.get_text().set_color("#e5e7eb")
    ax.set_title("PHANGS-JWST eight-band filter table\nmetadata dashboard; no image products downloaded",
                 color="#f8fafc", fontsize=13.2, pad=16)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_PHANGS_FILTER_TABLE.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_galaxy_table(galaxies):
    import matplotlib.pyplot as plt

    g = galaxies.sort_values("distance_mpc").reset_index(drop=True)
    rows = []
    for idx, row in g.iterrows():
        rows.append([
            f"{idx + 1:02d}",
            row["galaxy"],
            f"{row['distance_mpc']:.2f}",
            f"{row['pc_per_arcsec']:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(9.4, 9.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    labels = ["Rank", "Galaxy", "Distance (Mpc)", "pc / arcsec"]
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.38)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.60)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            distance = float(rows[r - 1][2])
            if distance < 10:
                cell.set_facecolor("#082f49")
            elif distance < 16:
                cell.set_facecolor("#172554")
            else:
                cell.set_facecolor("#3b1114")
            cell.get_text().set_color("#e5e7eb")
    ax.set_title("PHANGS-JWST galaxy sample scale table\ncurated rounded distances for engineering plot context",
                 color="#f8fafc", fontsize=13.2, pad=16)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_PHANGS_GALAXY_TABLE.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    filters, galaxies, grid = build_tables()

    filters_csv = OUTPUT_CSV / f"{VERSION}_PHANGS_FILTERS.csv"
    galaxies_csv = OUTPUT_CSV / f"{VERSION}_PHANGS_GALAXIES.csv"
    grid_csv = OUTPUT_CSV / f"{VERSION}_PHANGS_RESOLUTION_GRID.csv"
    filters.to_csv(filters_csv, index=False)
    galaxies.to_csv(galaxies_csv, index=False)
    grid.to_csv(grid_csv, index=False)

    ladder_png = plot_filter_ladder(filters)
    resolution_png = plot_resolution_curves(filters, galaxies, grid)
    distance_png = plot_galaxy_distance_bars(galaxies)
    filter_table_png = styled_filter_table(filters)
    galaxy_table_png = styled_galaxy_table(galaxies)

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Galaxy sample", f"{len(galaxies)} PHANGS-JWST Cycle 1 galaxies"),
        ("Filter set", f"{len(filters)} filters from 2-21 micron"),
        ("Image/FITS downloads", "none"),
        ("Plot engine", "matplotlib only"),
    ], ["Field", "Value"])
    print("")
    print("RESULTS")
    print_table([
        ("Nearest galaxy", f"{galaxies.loc[galaxies['distance_mpc'].idxmin(), 'galaxy']} | {galaxies['distance_mpc'].min():.2f} Mpc"),
        ("Farthest galaxy", f"{galaxies.loc[galaxies['distance_mpc'].idxmax(), 'galaxy']} | {galaxies['distance_mpc'].max():.2f} Mpc"),
        ("Shortest wavelength", f"{filters['lambda_um'].min():.2f} micron"),
        ("Longest wavelength", f"{filters['lambda_um'].max():.2f} micron"),
        ("Resolution grid rows", f"{len(grid):,}"),
    ], ["Metric", "Value"])
    print("")
    print("OUTPUT SUMMARY")
    print_table([
        ("png", str(ladder_png)),
        ("png", str(resolution_png)),
        ("png", str(distance_png)),
        ("png", str(filter_table_png)),
        ("png", str(galaxy_table_png)),
        ("csv", str(filters_csv)),
        ("csv", str(galaxies_csv)),
        ("csv", str(grid_csv)),
    ], ["Type", "Path"])
    print("")
    print("COMMENTS")
    print("This dashboard uses curated PHANGS-JWST metadata and computed JWST diffraction scales.")
    print("It does not download or render FITS/image products.")
    print("Matplotlib only. No AI images.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
