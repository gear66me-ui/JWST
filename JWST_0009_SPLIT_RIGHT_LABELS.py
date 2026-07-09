# JWST_0009
# Audit: JWST high-z edge-band plot with right-side split label lanes.
# Matplotlib only. No AI images. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0009"
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


def split_label_y_positions(df):
    import numpy as np
    n = len(df)
    upper_count = 8
    upper_y = np.linspace(36.95, 34.95, upper_count)
    lower_y = np.linspace(33.95, 31.45, n - upper_count)
    return list(upper_y) + list(lower_y)


def plot_split_right_labels(df, curve, z_inner):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16.0, 9.2))
    style_dark(fig, ax)

    ax.axhspan(EDGE_INNER_GLY, EDGE_OUTER_GLY, color="#7f1d1d", alpha=0.25, label="Requested 36.5-46.5 Gly band")
    ax.plot(curve["z"], curve["comoving_distance_gly"], color="#f8fafc", linewidth=2.0, alpha=0.92, label="Planck18 comoving distance curve")
    sc = ax.scatter(df["z"], df["comoving_distance_gly"], c=df["z"], cmap="coolwarm", s=160,
                    edgecolors="#f8fafc", linewidths=0.9, alpha=0.96, zorder=5)

    label_x = 17.4
    elbow_x = 16.55
    label_ys = split_label_y_positions(df)

    for (_, row), label_y in zip(df.iterrows(), label_ys):
        point_x = row["z"]
        point_y = row["comoving_distance_gly"]
        ax.plot([point_x, elbow_x, label_x - 0.12], [point_y, label_y, label_y], color="#94a3b8", linewidth=0.75, alpha=0.86, zorder=3)
        ax.text(point_x, point_y + 0.045, f"{int(row['rank'])}", color="#020617", fontsize=7.3,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.13", facecolor="#f8fafc", edgecolor="#f8fafc", alpha=0.94), zorder=8)
        ax.text(label_x, label_y,
                f"{int(row['rank']):02d}  {row['name']}  |  z={row['z']:.4f}  |  D={row['comoving_distance_gly']:.3f} Gly  |  age={row['universe_age_myr']:.1f} Myr",
                fontsize=8.4, color="#f8fafc", ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#020617", edgecolor="#475569", alpha=0.80), zorder=7)

    ax.axvline(z_inner, color="#f97316", linestyle="--", linewidth=1.25, alpha=0.90)
    ax.axhline(EDGE_INNER_GLY, color="#f97316", linestyle="--", linewidth=1.05, alpha=0.90)
    ax.axhline(EDGE_OUTER_GLY, color="#f8fafc", linestyle="-", linewidth=0.95, alpha=0.55)
    ax.text(10.1, EDGE_INNER_GLY + 0.12, f"36.5 Gly boundary, z≈{z_inner:.2f}", color="#f8fafc", fontsize=9)
    ax.text(16.8, 36.62, "upper label lane", color="#f97316", fontsize=8.6, alpha=0.95)
    ax.text(16.8, 33.72, "lower label lane", color="#38bdf8", fontsize=8.6, alpha=0.95)
    ax.text(10.1, 46.02, "46.5 Gly observable-radius marker / horizon territory", color="#f8fafc", fontsize=9)

    ax.set_xlim(9.7, 23.8)
    ax.set_ylim(30.9, 47.2)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("JWST high-z galaxy cluster with split right-side labels\nleader lines keep the data points clear; redder points are higher redshift")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9, loc="upper left")

    cbar = fig.colorbar(sc, ax=ax, pad=0.015)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")
    fig.tight_layout()

    path = OUTPUT_PNG / f"{VERSION}_SPLIT_RIGHT_LABELS.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_clean_points_only(df, curve, z_inner):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.2, 7.8))
    style_dark(fig, ax)
    ax.axhspan(EDGE_INNER_GLY, EDGE_OUTER_GLY, color="#7f1d1d", alpha=0.24)
    ax.plot(curve["z"], curve["comoving_distance_gly"], color="#f8fafc", linewidth=2.0, alpha=0.92)
    sc = ax.scatter(df["z"], df["comoving_distance_gly"], c=df["z"], cmap="coolwarm", s=175,
                    edgecolors="#f8fafc", linewidths=0.9, alpha=0.97, zorder=5)
    for _, row in df.iterrows():
        ax.text(row["z"], row["comoving_distance_gly"] + 0.065, f"{int(row['rank'])}", color="#020617", fontsize=7.6,
                ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="circle,pad=0.13", facecolor="#f8fafc", edgecolor="#f8fafc", alpha=0.95), zorder=8)
    ax.axvline(z_inner, color="#f97316", linestyle="--", linewidth=1.2, alpha=0.90)
    ax.axhline(EDGE_INNER_GLY, color="#f97316", linestyle="--", linewidth=1.0, alpha=0.90)
    ax.set_xlim(9.7, 23.8)
    ax.set_ylim(30.9, 47.2)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("Clean point-only panel for JWST high-z objects\nuse the numbered split-label plot for object names")
    cbar = fig.colorbar(sc, ax=ax, pad=0.018)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_CLEAN_POINTS_ONLY.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    df = build_df()
    curve = make_curve()
    z_inner = z_at_distance(EDGE_INNER_GLY)

    split_path = plot_split_right_labels(df, curve, z_inner)
    clean_path = plot_clean_points_only(df, curve, z_inner)
    csv_path = OUTPUT_CSV / f"{VERSION}_SPLIT_RIGHT_LABELS_TABLE.csv"
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
    print(f"PNG split labels : {split_path}")
    print(f"PNG clean points : {clean_path}")
    print(f"CSV              : {csv_path}")
    print("")
    print("COMMENTS")
    print("Labels are now moved into right-side lanes, split above and below the cluster.")
    print("Leader lines connect each object to its label; the data cluster is no longer covered by text.")
    print("Matplotlib only. No AI images.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
