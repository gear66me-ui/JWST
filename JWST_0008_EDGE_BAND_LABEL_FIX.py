# JWST_0008
# Audit: label-fixed version of the crowded JWST edge-band distance plot.
# Same data as JWST_0006, focused on readable engineering plots.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0008"
PROJECT = "JWST EDGE BAND LABEL FIX"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
EDGE_INNER_GLY = 36.5
EDGE_OUTER_GLY = 46.5
OBSERVABLE_RADIUS_GLY = 46.5

GALAXIES = [
    ("MoM-z14", 14.4400),
    ("JADES-GS-z14-0", 14.1793),
    ("JADES-GS-z14-1", 13.9000),
    ("PAN-z14-1", 13.5300),
    ("JADES-GS-z13-0", 13.2000),
    ("UNCOVER-z13", 13.0790),
    ("JADES-GS-z13-1-LA", 13.0000),
    ("JADES-GS-z12-0", 12.6300),
    ("UNCOVER-z12", 12.3930),
    ("CAPERS-EGS-65480", 12.3440),
    ("GLASS-z12 / GHZ2", 12.3400),
    ("Maisie's Galaxy", 11.4400),
    ("GS-z11-1", 11.2750),
    ("GN-z11", 10.6034),
    ("MACS0647-JD", 10.1700),
]

LABEL_OFFSETS = {
    "MACS0647-JD": (10, -18),
    "GN-z11": (14, -4),
    "GS-z11-1": (18, 16),
    "Maisie's Galaxy": (18, -16),
    "GLASS-z12 / GHZ2": (22, 10),
    "CAPERS-EGS-65480": (24, -8),
    "UNCOVER-z12": (26, 18),
    "JADES-GS-z12-0": (28, 2),
    "JADES-GS-z13-1-LA": (32, 18),
    "UNCOVER-z13": (32, -18),
    "JADES-GS-z13-0": (38, 6),
    "PAN-z14-1": (44, -8),
    "JADES-GS-z14-1": (48, 16),
    "JADES-GS-z14-0": (56, 2),
    "MoM-z14": (64, 18),
}

ZOOM_LABEL_X = 15.15


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
    ensure_package("astropy")
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)


def z_at_distance(distance_gly):
    from astropy.cosmology import Planck18 as cosmo
    lo, hi = 0.001, 5000.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if cosmo.comoving_distance(mid).to("Glyr").value < distance_gly:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def build_df():
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo
    df = pd.DataFrame(GALAXIES, columns=["name", "z"]).sort_values("z", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["comoving_distance_gly"] = [cosmo.comoving_distance(z).to("Glyr").value for z in df["z"]]
    df["universe_age_myr"] = [cosmo.age(z).value * 1000.0 for z in df["z"]]
    df["edge_gap_gly"] = EDGE_INNER_GLY - df["comoving_distance_gly"]
    return df


def make_curve():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo
    z_low = np.linspace(0.001, 30, 500)
    z_high = np.geomspace(30.1, 1200, 360)
    z = np.concatenate([z_low, z_high])
    curve = pd.DataFrame({"z": z})
    curve["comoving_distance_gly"] = [cosmo.comoving_distance(v).to("Glyr").value for v in z]
    return curve


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


def plot_full_edge_band(df, curve, z_inner):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    style_dark(fig, ax)

    ax.axhspan(EDGE_INNER_GLY, EDGE_OUTER_GLY, color="#7f1d1d", alpha=0.27, label="Requested 36.5-46.5 Gly band")
    ax.plot(curve["z"], curve["comoving_distance_gly"], color="#f8fafc", linewidth=2.0, alpha=0.92, label="Planck18 comoving distance curve")
    sc = ax.scatter(df["z"], df["comoving_distance_gly"], c=df["z"], cmap="coolwarm", s=145,
                    edgecolors="#f8fafc", linewidths=0.8, alpha=0.96, zorder=5)

    for _, row in df.iterrows():
        dx, dy = LABEL_OFFSETS.get(row["name"], (18, 10))
        ax.annotate(
            f"{row['name']}\n{row['comoving_distance_gly']:.2f} Gly",
            xy=(row["z"], row["comoving_distance_gly"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.4,
            color="#f8fafc",
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.72),
            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=0.85, alpha=0.9, shrinkA=0, shrinkB=5),
            zorder=7,
        )

    ax.axvline(z_inner, color="#f97316", linestyle="--", linewidth=1.3, alpha=0.90)
    ax.text(z_inner * 1.05, EDGE_INNER_GLY + 0.35, f"36.5 Gly begins near z≈{z_inner:.1f}", color="#f8fafc", fontsize=9)
    ax.text(70, 44.8, "46.5 Gly is near the observable limit, not a galaxy map", color="#f8fafc", fontsize=9)

    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlim(8.5, 1300)
    ax.set_ylim(29, 47.2)
    ax.set_xlabel("Redshift z, symlog scale")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("Known JWST high-z galaxies do not occupy the 36.5-46.5 Gly shell\nlabel-fixed full view")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9, loc="lower right")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")
    fig.tight_layout()

    path = OUTPUT_PNG / f"{VERSION}_FULL_EDGE_BAND_LABEL_FIX.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_zoom_cluster(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    style_dark(fig, ax)
    ax.scatter(df["z"], df["comoving_distance_gly"], c=df["z"], cmap="coolwarm", s=170,
               edgecolors="#f8fafc", linewidths=0.9, alpha=0.97, zorder=5)

    label_y_values = list(reversed([31.35, 31.55, 31.75, 31.95, 32.15, 32.35, 32.55, 32.75, 32.95, 33.15, 33.35, 33.55, 33.75, 33.95, 34.15]))
    for (_, row), label_y in zip(df.iterrows(), label_y_values):
        ax.plot([row["z"], ZOOM_LABEL_X - 0.12], [row["comoving_distance_gly"], label_y], color="#64748b", linewidth=0.75, alpha=0.88, zorder=3)
        ax.text(ZOOM_LABEL_X, label_y,
                f"{int(row['rank']):02d}  {row['name']}  |  z={row['z']:.4f}  |  D={row['comoving_distance_gly']:.3f} Gly  |  age={row['universe_age_myr']:.1f} Myr",
                fontsize=8.2, color="#f8fafc", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#020617", edgecolor="#475569", alpha=0.78))
        ax.text(row["z"], row["comoving_distance_gly"] + 0.025, f"{int(row['rank'])}", color="#020617", fontsize=7.0,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.12", facecolor="#f8fafc", edgecolor="#f8fafc", alpha=0.92), zorder=8)

    ax.axhline(EDGE_INNER_GLY, color="#f97316", linestyle="--", linewidth=1.1, alpha=0.85)
    ax.text(10.2, EDGE_INNER_GLY - 0.08, "36.5 Gly requested-band threshold", color="#f8fafc", fontsize=9)
    ax.set_xlim(9.8, 19.2)
    ax.set_ylim(31.1, 36.8)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("Readable zoom of the JWST high-z galaxy cluster\nnumbered points, right-side legend, leader lines")
    fig.tight_layout()

    path = OUTPUT_PNG / f"{VERSION}_ZOOM_CLUSTER_NUMBERED_LABELS.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    df = build_df()
    curve = make_curve()
    z_inner = z_at_distance(EDGE_INNER_GLY)

    full_path = plot_full_edge_band(df, curve, z_inner)
    zoom_path = plot_zoom_cluster(df)

    csv_path = OUTPUT_CSV / f"{VERSION}_LABEL_FIX_GALAXY_TABLE.csv"
    df.to_csv(csv_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("RESULTS")
    print(f"Most distant object : {df.iloc[0]['name']}")
    print(f"Max distance        : {df['comoving_distance_gly'].max():.6f} Gly")
    print(f"Gap to 36.5 Gly     : {EDGE_INNER_GLY - df['comoving_distance_gly'].max():.6f} Gly")
    print(f"36.5 Gly redshift   : z ≈ {z_inner:.6f}")
    print("")
    print("OUTPUT SUMMARY")
    print(f"PNG full : {full_path}")
    print(f"PNG zoom : {zoom_path}")
    print(f"CSV      : {csv_path}")
    print("")
    print("COMMENTS")
    print("Labels were moved off the crowded cluster and connected with leader lines.")
    print("The zoom plot uses numbered point markers plus a right-side engineering legend.")
    print("The data and cosmology are unchanged from the edge-band analysis.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
