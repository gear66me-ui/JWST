# JWST_0006
# Audit: JWST edge-bandwidth map; requested 36.5-46.5 Gly shell vs known JWST high-z galaxies.
# Curated metadata only. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0006"
PROJECT = "JWST EDGE BANDWIDTH MAP"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
OUTPUT_PNG = OUTPUT_DIR / "PNG"

OBSERVABLE_RADIUS_GLY = 46.5
EDGE_INNER_GLY = 36.5
EDGE_OUTER_GLY = 46.5

# Known/reported JWST high-redshift galaxy frontier objects.
# These are galaxies, not CMB/particle-horizon detections.
JWST_GALAXIES = [
    {"name":"MoM-z14", "z":14.4400, "ra_deg":150.093333, "dec_deg":2.273108, "status":"spectroscopic/reported", "field":"COSMOS / MoM"},
    {"name":"JADES-GS-z14-0", "z":14.1793, "ra_deg":53.08294, "dec_deg":-27.85563, "status":"spectroscopic + ALMA", "field":"GOODS-S / JADES"},
    {"name":"JADES-GS-z14-1", "z":13.9000, "ra_deg":53.1600, "dec_deg":-27.7800, "status":"spectroscopic", "field":"GOODS-S / JADES"},
    {"name":"PAN-z14-1", "z":13.5300, "ra_deg":334.250356, "dec_deg":0.379215, "status":"spectroscopic", "field":"PANORAMIC"},
    {"name":"JADES-GS-z13-0", "z":13.2000, "ra_deg":53.14988, "dec_deg":-27.77650, "status":"spectroscopic", "field":"GOODS-S / JADES"},
    {"name":"UNCOVER-z13", "z":13.0790, "ra_deg":3.5860, "dec_deg":-30.4000, "status":"spectroscopic/plausible", "field":"Abell 2744 / UNCOVER"},
    {"name":"JADES-GS-z13-1-LA", "z":13.0000, "ra_deg":53.1500, "dec_deg":-27.7800, "status":"spectroscopic", "field":"GOODS-S / JADES"},
    {"name":"JADES-GS-z12-0", "z":12.6300, "ra_deg":53.1600, "dec_deg":-27.7700, "status":"spectroscopic", "field":"GOODS-S / JADES"},
    {"name":"UNCOVER-z12", "z":12.3930, "ra_deg":3.5860, "dec_deg":-30.4000, "status":"spectroscopic", "field":"Abell 2744 / UNCOVER"},
    {"name":"CAPERS-EGS-65480", "z":12.3440, "ra_deg":214.9000, "dec_deg":52.9000, "status":"spectroscopic", "field":"EGS / CAPERS"},
    {"name":"GLASS-z12 / GHZ2", "z":12.3400, "ra_deg":3.5860, "dec_deg":-30.4000, "status":"spectroscopic/ALMA", "field":"Abell 2744 / GLASS"},
    {"name":"Maisie's Galaxy", "z":11.4400, "ra_deg":214.9432, "dec_deg":52.9424, "status":"spectroscopic", "field":"CEERS / EGS"},
    {"name":"GS-z11-1", "z":11.2750, "ra_deg":53.1600, "dec_deg":-27.7800, "status":"spectroscopic", "field":"GOODS-S / JADES"},
    {"name":"GN-z11", "z":10.6034, "ra_deg":189.1068, "dec_deg":62.2420, "status":"spectroscopic", "field":"GOODS-N"},
    {"name":"MACS0647-JD", "z":10.1700, "ra_deg":101.9830, "dec_deg":70.2480, "status":"spectroscopic/lensed", "field":"MACS J0647.7+7015"},
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
        widths.append(min(width, 42))
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

    df = pd.DataFrame(JWST_GALAXIES).sort_values("z", ascending=False).reset_index(drop=True)
    df["rank_by_z"] = range(1, len(df) + 1)
    df["theta_ra_rad"] = np.deg2rad(df["ra_deg"] % 360.0)
    df["comoving_distance_gly"] = [cosmo.comoving_distance(z).to("Glyr").value for z in df["z"]]
    df["light_travel_time_gyr"] = [cosmo.lookback_time(z).value for z in df["z"]]
    df["universe_age_gyr"] = [cosmo.age(z).value for z in df["z"]]
    df["universe_age_myr"] = df["universe_age_gyr"] * 1000.0
    df["gap_to_36p5_gly"] = EDGE_INNER_GLY - df["comoving_distance_gly"]
    df["inside_requested_band"] = (df["comoving_distance_gly"] >= EDGE_INNER_GLY) & (df["comoving_distance_gly"] <= EDGE_OUTER_GLY)
    return df

def z_at_distance(distance_gly):
    from astropy.cosmology import Planck18 as cosmo
    lo, hi = 0.001, 5000.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        d = cosmo.comoving_distance(mid).to("Glyr").value
        if d < distance_gly:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def make_curve():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    z_low = np.linspace(0.001, 30, 500)
    z_high = np.geomspace(30.1, 1200, 360)
    z = np.concatenate([z_low, z_high])
    curve = pd.DataFrame({"z": z})
    curve["comoving_distance_gly"] = [cosmo.comoving_distance(v).to("Glyr").value for v in z]
    curve["universe_age_myr"] = [cosmo.age(v).value * 1000.0 for v in z]
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

def plot_polar_band(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 11.5))
    fig.patch.set_facecolor("#050712")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#050712")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, OBSERVABLE_RADIUS_GLY)
    ax.set_rticks([10, 20, 30, 34, 36.5, 40, 46.5])
    ax.set_rlabel_position(135)
    ax.tick_params(colors="#dbeafe", labelsize=8)
    ax.xaxis.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.yaxis.grid(True, color="#475569", linewidth=0.55, alpha=0.70)
    ax.spines["polar"].set_color("#94a3b8")

    horizon = np.linspace(0, 2 * np.pi, 720)
    ax.fill_between(horizon, EDGE_INNER_GLY, EDGE_OUTER_GLY, color="#7f1d1d", alpha=0.30, zorder=0)
    ax.plot(horizon, np.full_like(horizon, EDGE_INNER_GLY), color="#f97316", linewidth=1.4, alpha=0.95)
    ax.plot(horizon, np.full_like(horizon, EDGE_OUTER_GLY), color="#f8fafc", linewidth=1.4, alpha=0.95)

    sc = ax.scatter(df["theta_ra_rad"], df["comoving_distance_gly"], c=df["z"], cmap="coolwarm",
                    vmin=df["z"].min(), vmax=df["z"].max(), s=155,
                    edgecolors="#f8fafc", linewidths=0.8, alpha=0.94, zorder=5)

    for _, row in df.iterrows():
        ax.text(row["theta_ra_rad"], row["comoving_distance_gly"] + 0.95,
                f"{row['name']}\nz={row['z']:.2f}\nD={row['comoving_distance_gly']:.2f} Gly\nage={row['universe_age_myr']:.0f} Myr",
                fontsize=6.5, color="#f8fafc", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#020617", edgecolor="#475569", alpha=0.72), zorder=6)

    n_inside = int(df["inside_requested_band"].sum())
    text = f"Requested band: {EDGE_INNER_GLY:.1f}–{EDGE_OUTER_GLY:.1f} Gly\nKnown JWST galaxy points inside: {n_inside}\nJWST galaxy frontier here: ~34 Gly\nThe 46.5 Gly edge is CMB/horizon territory."
    ax.text(np.deg2rad(230), 42.3, text, color="#f8fafc", fontsize=9.2, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.36", facecolor="#111827", edgecolor="#f97316", alpha=0.88), zorder=8)

    ax.set_title("JWST high-z galaxies vs the 36.5–46.5 Gly edge band\nred = older / higher redshift; blue = lower redshift", color="#f8fafc", fontsize=15, pad=32)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.10)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_POLAR_EDGE_BAND_MAP.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_distance_curve(df, curve, z_inner):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7.5))
    style_dark(fig, ax)

    ax.axhspan(EDGE_INNER_GLY, EDGE_OUTER_GLY, color="#7f1d1d", alpha=0.27, label="Requested 36.5–46.5 Gly band")
    ax.plot(curve["z"], curve["comoving_distance_gly"], color="#f8fafc", linewidth=2.0, alpha=0.92, label="Planck18 comoving distance curve")
    sc = ax.scatter(df["z"], df["comoving_distance_gly"], c=df["z"], cmap="coolwarm",
                    s=135, edgecolors="#f8fafc", linewidths=0.8, alpha=0.96, zorder=5)

    for _, row in df.iterrows():
        ax.annotate(f"{row['name']}\n{row['comoving_distance_gly']:.2f} Gly",
                    (row["z"], row["comoving_distance_gly"]), xytext=(6, 5), textcoords="offset points",
                    fontsize=7, color="#f8fafc",
                    bbox=dict(boxstyle="round,pad=0.16", facecolor="#020617", edgecolor="#475569", alpha=0.60))

    ax.axvline(z_inner, color="#f97316", linestyle="--", linewidth=1.3, alpha=0.90)
    ax.text(z_inner * 1.05, EDGE_INNER_GLY + 0.35, f"36.5 Gly begins near z≈{z_inner:.1f}", color="#f8fafc", fontsize=9)
    ax.text(70, 44.8, "46.5 Gly is near the observable limit, not a galaxy map", color="#f8fafc", fontsize=9)

    ax.set_xscale("symlog", linthresh=20)
    ax.set_xlim(8, 1300)
    ax.set_ylim(29, 47.2)
    ax.set_xlabel("Redshift z, symlog scale")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("Known JWST high-z galaxies do not occupy the 36.5–46.5 Gly shell")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9, loc="lower right")

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_DISTANCE_REDSHIFT_EDGE_BAND.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_age_rank(df):
    import matplotlib.pyplot as plt

    ordered = df.sort_values("universe_age_myr", ascending=False).copy()
    fig, ax = plt.subplots(figsize=(12, 7.5))
    style_dark(fig, ax)

    norm = (ordered["z"] - ordered["z"].min()) / (ordered["z"].max() - ordered["z"].min() + 1e-9)
    bars = ax.barh(ordered["name"], ordered["universe_age_myr"], color=plt.cm.coolwarm(norm))
    ax.invert_yaxis()

    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"z={row['z']:.2f} | D={row['comoving_distance_gly']:.2f} Gly",
                va="center", color="#f8fafc", fontsize=8)

    ax.set_xlabel("Universe age at emission, Myr after Big Bang")
    ax.set_ylabel("JWST high-redshift galaxy")
    ax.set_title("Age ranking: redder bars are higher redshift / earlier galaxies")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_AGE_RANK_COLOR_CODE.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path

def save_outputs(df, curve, paths, z_inner, z_outer):
    csv_path = OUTPUT_CSV / f"{VERSION}_JWST_EDGE_BAND_TABLE.csv"
    curve_path = OUTPUT_CSV / f"{VERSION}_REDSHIFT_DISTANCE_CURVE.csv"
    notes_path = OUTPUT_CSV / f"{VERSION}_EDGE_BAND_NOTES.txt"
    df.to_csv(csv_path, index=False)
    curve.to_csv(curve_path, index=False)
    notes_path.write_text(
        f"{VERSION} notes\n"
        "Question: what has JWST mapped in the 36.5-46.5 Gly edge band?\n"
        "Answer from this curated frontier sample: no named/confirmed JWST galaxy in this table lies in that comoving shell.\n"
        f"The inner edge {EDGE_INNER_GLY:.1f} Gly corresponds to roughly z={z_inner:.3f}.\n"
        f"The outer edge {EDGE_OUTER_GLY:.1f} Gly corresponds to extremely high redshift / observable-horizon territory; binary-search estimate z={z_outer:.3f} is only a marker.\n"
        "JWST maps early galaxies and cosmic-dawn fields, not the CMB/particle horizon itself.\n"
        "No FITS or image products are downloaded.\n",
        encoding="utf-8",
    )
    return csv_path, curve_path, notes_path

def main():
    setup()

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Output folder", str(OUTPUT_DIR)),
        ("Requested band", f"{EDGE_INNER_GLY:.1f} to {EDGE_OUTER_GLY:.1f} Gly"),
        ("Objects", len(JWST_GALAXIES)),
        ("Color code", "red = higher z / older light, blue = lower z"),
        ("Mode", "Curated metadata only; no FITS/image downloads"),
    ], ["Field", "Value"])

    df = build_dataframe()
    curve = make_curve()
    z_inner = z_at_distance(EDGE_INNER_GLY)
    z_outer = z_at_distance(EDGE_OUTER_GLY)

    paths = [
        plot_polar_band(df),
        plot_distance_curve(df, curve, z_inner),
        plot_age_rank(df),
    ]
    csv_path, curve_path, notes_path = save_outputs(df, curve, paths, z_inner, z_outer)

    print("")
    print("RESULTS")
    inside = df[df["inside_requested_band"]]
    print_table([
        ("Known JWST objects in 36.5-46.5 Gly band", len(inside)),
        ("Most distant object in table", df.iloc[0]["name"]),
        ("Most distant table distance", f"{df['comoving_distance_gly'].max():.3f} Gly"),
        ("Gap to 36.5 Gly", f"{EDGE_INNER_GLY - df['comoving_distance_gly'].max():.3f} Gly"),
        ("36.5 Gly approx redshift", f"z ≈ {z_inner:.3f}"),
    ], ["Metric", "Value"])

    print("")
    print("FRONTIER TABLE")
    rows = []
    for _, row in df.iterrows():
        rows.append((int(row["rank_by_z"]), row["name"], f"{row['z']:.4f}", f"{row['comoving_distance_gly']:.3f}", f"{row['universe_age_myr']:.1f}", row["status"]))
    print_table(rows, ["Rank", "Name", "z", "Dist Gly", "Age Myr", "Status"])

    print("")
    print("OUTPUT SUMMARY")
    output_rows = [("csv", str(csv_path)), ("csv", str(curve_path)), ("notes", str(notes_path))]
    output_rows.extend(("png", str(path)) for path in paths)
    print_table(output_rows, ["Type", "Path"])

    print("")
    print("COMMENTS")
    print("JWST can look toward cosmic dawn, but confirmed galaxies are not at the 46.5 Gly particle horizon.")
    print("The requested 36.5-46.5 Gly shell is shaded; the plotted JWST galaxy frontier sits inside it near ~34 Gly.")
    print("At the true edge, before recombination and before galaxies, there are no galaxies for JWST to map.")

    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
