# JWST_0004
# Audit: z >= 13 high-redshift galaxy distance map. Curated metadata only; no FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0004"
PROJECT = "JWST Z>=13 DISTANCE EDGE MAP"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OBSERVABLE_RADIUS_GLY = 46.5

# These are galaxies, not resolved individual stars. At z >= 13, JWST is seeing early galaxies.
# Distance values are computed below with astropy Planck18 cosmology.
# Some RA/Dec entries remain field-level visual placements until exact catalog coordinates are patched.
Z13_OBJECTS = [
    {
        "name": "MoM-z14",
        "redshift_z": 14.44,
        "ra_deg": 150.093333,
        "dec_deg": 2.273108,
        "field": "COSMOS / Mirage or Miracle",
        "confirmation": "spectroscopic JWST/NIRSpec",
        "coordinate_status": "visual placement; patch exact table coordinate later",
        "note": "Reported z_spec=14.44; about 280 Myr after Big Bang.",
    },
    {
        "name": "JADES-GS-z14-0",
        "redshift_z": 14.1793,
        "ra_deg": 53.08294,
        "dec_deg": -27.85563,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic JWST + ALMA [OIII]",
        "coordinate_status": "object-name coordinate",
        "note": "ALMA [OIII] 88 micron line gives very precise redshift.",
    },
    {
        "name": "JADES-GS-z14-1",
        "redshift_z": 13.86,
        "ra_deg": 53.1600,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic JWST/NIRSpec",
        "coordinate_status": "field-level visual placement",
        "note": "Second z~14 JADES object from the same early-galaxy campaign.",
    },
    {
        "name": "PAN-z14-1",
        "redshift_z": 13.53,
        "ra_deg": 334.25035598,
        "dec_deg": 0.3792145611,
        "field": "PANORAMIC pure-parallel",
        "confirmation": "spectroscopic JWST/NIRSpec",
        "coordinate_status": "published coordinate",
        "note": "Large luminous galaxy with weak emission lines at z_spec=13.53.",
    },
    {
        "name": "JADES-GS-z13-0",
        "redshift_z": 13.20,
        "ra_deg": 53.14988,
        "dec_deg": -27.77650,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic JWST/NIRSpec",
        "coordinate_status": "object-name coordinate",
        "note": "Former earliest-galaxy record holder after JWST/NIRSpec confirmation.",
    },
    {
        "name": "UNCOVER-z13",
        "redshift_z": 13.079,
        "ra_deg": 3.5860,
        "dec_deg": -30.4000,
        "field": "Abell 2744 / UNCOVER",
        "confirmation": "spectroscopic/plausible high-confidence",
        "coordinate_status": "cluster-field visual placement",
        "note": "Lensed Abell 2744 source; useful z>=13 comparison point.",
    },
    {
        "name": "JADES-GS-z13-1-LA",
        "redshift_z": 13.00,
        "ra_deg": 53.1500,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic Lyman-alpha",
        "coordinate_status": "field-level visual placement",
        "note": "Included at the z>=13 cutoff.",
    },
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
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)

def print_table(rows, headers):
    widths = []
    for i, header in enumerate(headers):
        width = max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        widths.append(min(width, 38))
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

def build_dataframe():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    df = pd.DataFrame(Z13_OBJECTS).copy()
    df = df[df["redshift_z"] >= 13.0].sort_values("redshift_z", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["theta_ra_rad"] = np.deg2rad(df["ra_deg"] % 360.0)
    df["comoving_distance_gly"] = [cosmo.comoving_distance(z).to("Glyr").value for z in df["redshift_z"]]
    df["lookback_time_gyr"] = [cosmo.lookback_time(z).value for z in df["redshift_z"]]
    df["universe_age_gyr"] = [cosmo.age(z).value for z in df["redshift_z"]]
    df["universe_age_myr"] = df["universe_age_gyr"] * 1000.0
    df["proper_distance_then_gly"] = df["comoving_distance_gly"] / (1.0 + df["redshift_z"])
    df["distance_to_observable_edge_gly"] = OBSERVABLE_RADIUS_GLY - df["comoving_distance_gly"]
    df["observable_radius_fraction"] = df["comoving_distance_gly"] / OBSERVABLE_RADIUS_GLY
    df["distance_label"] = df.apply(
        lambda r: f"{r['name']}\nz={r['redshift_z']:.3f}\nD={r['comoving_distance_gly']:.2f} Gly",
        axis=1,
    )
    return df

def configure_dark_polar(ax):
    ax.set_facecolor("#050712")
    ax.figure.set_facecolor("#050712")
    ax.tick_params(colors="#dbeafe", labelsize=8)
    ax.xaxis.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.yaxis.grid(True, color="#475569", linewidth=0.55, alpha=0.70)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

def plot_distance_edge(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 11))
    ax = fig.add_subplot(111, projection="polar")
    configure_dark_polar(ax)

    theta = df["theta_ra_rad"].to_numpy()
    radius = df["comoving_distance_gly"].to_numpy()
    z = df["redshift_z"].to_numpy()

    ax.set_rlim(0, OBSERVABLE_RADIUS_GLY)
    ax.set_rticks([10, 20, 30, 33, 34, 35, 40, 46.5])
    ax.set_rlabel_position(140)

    horizon = np.linspace(0, 2 * np.pi, 720)
    ax.plot(horizon, np.full_like(horizon, OBSERVABLE_RADIUS_GLY), color="#f8fafc", linewidth=1.4, alpha=0.90)
    ax.fill_between(horizon, 0, OBSERVABLE_RADIUS_GLY, color="#0f172a", alpha=0.50)
    ax.fill_between(horizon, 33.0, OBSERVABLE_RADIUS_GLY, color="#1e1b4b", alpha=0.45)

    points = ax.scatter(
        theta,
        radius,
        c=z,
        cmap="plasma",
        s=180,
        alpha=0.92,
        edgecolors="#f8fafc",
        linewidths=0.9,
        zorder=5,
    )

    for _, row in df.iterrows():
        outward = 1.25 if row["redshift_z"] < 14.2 else 1.55
        ax.text(
            row["theta_ra_rad"],
            row["comoving_distance_gly"] + outward,
            row["distance_label"],
            fontsize=7.2,
            color="#f8fafc",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#020617", edgecolor="#475569", alpha=0.76),
            zorder=6,
        )

    ax.text(
        np.deg2rad(230),
        OBSERVABLE_RADIUS_GLY - 1.0,
        "Observable universe radius ≈ 46.5 Gly",
        color="#f8fafc",
        fontsize=9,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", edgecolor="#e5e7eb", alpha=0.82),
    )

    ax.set_title(
        "JWST galaxies with z ≥ 13 mapped by comoving distance\nangle = RA, radius = present comoving distance from Earth",
        color="#f8fafc",
        fontsize=15,
        pad=30,
    )

    cbar = fig.colorbar(points, ax=ax, shrink=0.70, pad=0.10)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_DISTANCE_EDGE_POLAR.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_redshift_vs_distance(df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")

    sc = ax.scatter(
        df["redshift_z"],
        df["comoving_distance_gly"],
        c=df["universe_age_myr"],
        cmap="viridis_r",
        s=190,
        edgecolors="#f8fafc",
        linewidths=0.9,
        alpha=0.92,
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['name']}  {row['comoving_distance_gly']:.2f} Gly",
            (row["redshift_z"], row["comoving_distance_gly"]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color="#f8fafc",
        )

    ax.axhline(OBSERVABLE_RADIUS_GLY, color="#f8fafc", linewidth=1.0, alpha=0.75)
    ax.text(13.03, OBSERVABLE_RADIUS_GLY - 0.55, "observable radius ≈ 46.5 Gly", color="#f8fafc", fontsize=9)
    ax.set_xlabel("Redshift z", color="#f8fafc")
    ax.set_ylabel("Comoving distance, billion light-years", color="#f8fafc")
    ax.set_title("z ≥ 13 galaxies: redshift versus distance", color="#f8fafc", fontsize=15)
    ax.tick_params(colors="#dbeafe")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Universe age at emission, Myr", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_REDSHIFT_DISTANCE_SCATTER.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_distance_ranking(df):
    import matplotlib.pyplot as plt

    ordered = df.sort_values("comoving_distance_gly", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")

    bars = ax.barh(ordered["name"], ordered["comoving_distance_gly"], color=plt.cm.plasma((ordered["redshift_z"] - 13.0) / (ordered["redshift_z"].max() - 13.0 + 1e-9)))
    ax.axvline(OBSERVABLE_RADIUS_GLY, color="#f8fafc", linewidth=1.2, alpha=0.85)

    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(
            bar.get_width() + 0.20,
            bar.get_y() + bar.get_height() / 2,
            f"{row['comoving_distance_gly']:.2f} Gly | z={row['redshift_z']:.3f} | age={row['universe_age_myr']:.0f} Myr",
            va="center",
            color="#f8fafc",
            fontsize=8,
        )

    ax.set_xlim(32.0, 47.5)
    ax.set_xlabel("Comoving distance, billion light-years", color="#f8fafc")
    ax.set_ylabel("Galaxy", color="#f8fafc")
    ax.set_title("Distance labels for z ≥ 13 galaxies", color="#f8fafc", fontsize=15)
    ax.tick_params(colors="#dbeafe")
    ax.grid(True, axis="x", color="#334155", linewidth=0.55, alpha=0.70)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_DISTANCE_RANKING.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def save_outputs(df, paths):
    csv_path = OUTPUT_CSV / f"{VERSION}_Z13_DISTANCE_TABLE.csv"
    df.to_csv(csv_path, index=False)

    notes_path = OUTPUT_CSV / f"{VERSION}_Z13_DISTANCE_NOTES.txt"
    notes_path.write_text(
        "JWST_0004 notes\n"
        "Objects are z >= 13 galaxies, not individual resolved stars.\n"
        "Radial distance is Planck18 comoving distance in billion light-years.\n"
        "The 46.5 Gly observable-universe ring is an approximate present-day radius marker.\n"
        "Some RA/Dec entries are field-level visual placements and are flagged in coordinate_status.\n"
        "No FITS or image products are downloaded.\n",
        encoding="utf-8",
    )
    return csv_path, notes_path

def main():
    setup()

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Output folder", str(OUTPUT_DIR)),
        ("Objects", len(Z13_OBJECTS)),
        ("Cutoff", "z >= 13"),
        ("Radial distance", "Planck18 comoving distance, Gly"),
        ("Outer ring", f"Observable radius marker {OBSERVABLE_RADIUS_GLY:.1f} Gly"),
        ("Mode", "Curated metadata only; no FITS/image downloads"),
    ], ["Field", "Value"])

    df = build_dataframe()
    paths = [
        plot_distance_edge(df),
        plot_redshift_vs_distance(df),
        plot_distance_ranking(df),
    ]
    csv_path, notes_path = save_outputs(df, paths)

    print("")
    print("RESULTS")
    rows = []
    for _, row in df.iterrows():
        rows.append((
            int(row["rank"]),
            row["name"],
            f"{row['redshift_z']:.4f}",
            f"{row['comoving_distance_gly']:.3f}",
            f"{row['distance_to_observable_edge_gly']:.3f}",
            f"{row['universe_age_myr']:.1f}",
        ))
    print_table(rows, ["Rank", "Name", "z", "Dist Gly", "Edge gap", "Age Myr"])

    print("")
    print("OUTPUT SUMMARY")
    output_rows = [("csv", str(csv_path)), ("notes", str(notes_path))]
    output_rows.extend(("png", str(path)) for path in paths)
    print_table(output_rows, ["Type", "Path"])

    print("")
    print("COMMENTS")
    print("The beautiful radial plot now uses distance as the radius.")
    print("Redshift is shown by color and also printed in labels.")
    print("The outer circle marks the approximate present observable-universe radius.")
    print("This is a clean visualization pass; exact catalog-coordinate patching can come next.")

    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
