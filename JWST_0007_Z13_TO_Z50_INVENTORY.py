# JWST_0007
# Audit: Inventory of reported galaxies in redshift range 13 <= z <= 50.
# Curated frontier table + empty-bin audit + cosmology distance curve. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0007"
PROJECT = "JWST Z13 TO Z50 GALAXY INVENTORY"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
Z_MIN = 13.0
Z_MAX = 50.0
OBSERVABLE_RADIUS_GLY = 46.5

# Curated inventory of named/reported JWST high-redshift galaxies at z >= 13.
# Current practical result: known/reported galaxy entries cluster at z ~ 13 to 14.44.
# There are no robust named JWST galaxies in this inventory above z = 14.44, and none near z = 50.
Z13_TO_Z50_OBJECTS = [
    {
        "name": "MoM-z14",
        "z": 14.4400,
        "ra_deg": 150.093333,
        "dec_deg": 2.273108,
        "field": "COSMOS / MoM",
        "class": "reported_spectroscopic",
        "confidence": "frontier_reported",
        "note": "Reported z_spec=14.44 JWST/NIRSpec record-holder sample point.",
    },
    {
        "name": "JADES-GS-z14-0",
        "z": 14.1793,
        "ra_deg": 53.08294,
        "dec_deg": -27.85563,
        "field": "GOODS-S / JADES",
        "class": "spectroscopic_plus_ALMA",
        "confidence": "robust",
        "note": "ALMA [OIII] 88 micron redshift z=14.1793 plus JWST detection.",
    },
    {
        "name": "JADES-GS-z14-1",
        "z": 13.9000,
        "ra_deg": 53.1600,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "class": "spectroscopic",
        "confidence": "reported",
        "note": "JWST/JADES z~14 companion object; coordinate visual placement.",
    },
    {
        "name": "PAN-z14-1",
        "z": 13.5300,
        "ra_deg": 334.250356,
        "dec_deg": 0.379215,
        "field": "PANORAMIC",
        "class": "spectroscopic",
        "confidence": "reported",
        "note": "Reported z_spec=13.53 galaxy in PANORAMIC pure-parallel field.",
    },
    {
        "name": "JADES-GS-z13-0",
        "z": 13.2000,
        "ra_deg": 53.14988,
        "dec_deg": -27.77650,
        "field": "GOODS-S / JADES",
        "class": "spectroscopic",
        "confidence": "robust",
        "note": "Early JWST spectroscopic record-holder at z=13.2.",
    },
    {
        "name": "UNCOVER-z13",
        "z": 13.0790,
        "ra_deg": 3.5860,
        "dec_deg": -30.4000,
        "field": "Abell 2744 / UNCOVER",
        "class": "lensed_spectroscopic_candidate",
        "confidence": "plausible_reported",
        "note": "Lensed Abell 2744 high-z source; field-level visual coordinate.",
    },
    {
        "name": "JADES-GS-z13-1-LA",
        "z": 13.0000,
        "ra_deg": 53.1500,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "class": "spectroscopic_Lyman_alpha",
        "confidence": "reported",
        "note": "Included at the z>=13 cutoff; coordinate visual placement.",
    },
]

# Historical high-z claims/candidates that are useful for audit but NOT counted as confirmed inventory.
# They help show why a z=13..50 search needs a confidence column.
CLAIM_AUDIT = [
    {
        "name": "CEERS-93316",
        "claimed_z": 16.7,
        "current_status": "reinterpreted_lower_redshift",
        "reason": "Initially discussed as a very high-z candidate; spectroscopy later favored a much lower redshift solution.",
    },
    {
        "name": "Generic z>20 photometric dropouts",
        "claimed_z": 20.0,
        "current_status": "not_confirmed_galaxy_inventory",
        "reason": "Photometric candidates at extreme z require spectroscopy and contamination checks before being inventoried as galaxies.",
    },
    {
        "name": "z=50 galaxy bin",
        "claimed_z": 50.0,
        "current_status": "empty_known_galaxy_bin",
        "reason": "This is earlier than the current confirmed/reported JWST galaxy frontier; useful as a theoretical marker only.",
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

def build_inventory():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    df = pd.DataFrame(Z13_TO_Z50_OBJECTS).copy()
    df = df[(df["z"] >= Z_MIN) & (df["z"] <= Z_MAX)].sort_values("z", ascending=False).reset_index(drop=True)
    df["rank_by_z"] = range(1, len(df) + 1)
    df["theta_ra_rad"] = np.deg2rad(df["ra_deg"] % 360.0)
    df["comoving_distance_gly"] = [cosmo.comoving_distance(z).to("Glyr").value for z in df["z"]]
    df["light_travel_time_gyr"] = [cosmo.lookback_time(z).value for z in df["z"]]
    df["universe_age_myr"] = [cosmo.age(z).value * 1000.0 for z in df["z"]]
    df["proper_distance_then_gly"] = df["comoving_distance_gly"] / (1.0 + df["z"])
    df["observable_radius_fraction"] = df["comoving_distance_gly"] / OBSERVABLE_RADIUS_GLY
    return df

def build_bins(df):
    import pandas as pd
    bins = []
    for start in range(13, 50):
        end = start + 1
        count = int(((df["z"] >= start) & (df["z"] < end)).sum())
        bins.append({"z_bin": f"{start}-{end}", "z_start": start, "z_end": end, "known_or_reported_count": count})
    bins.append({"z_bin": "50", "z_start": 50, "z_end": 50, "known_or_reported_count": int((df["z"] == 50).sum())})
    return pd.DataFrame(bins)

def build_curve():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    z = np.linspace(Z_MIN, Z_MAX, 800)
    curve = pd.DataFrame({"z": z})
    curve["comoving_distance_gly"] = [cosmo.comoving_distance(v).to("Glyr").value for v in z]
    curve["universe_age_myr"] = [cosmo.age(v).value * 1000.0 for v in z]
    curve["lookback_time_gyr"] = [cosmo.lookback_time(v).value for v in z]
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

def plot_inventory_bins(bin_df):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 6.5))
    style_dark(fig, ax)
    colors = ["#ef4444" if count > 0 else "#1e293b" for count in bin_df["known_or_reported_count"]]
    ax.bar(bin_df["z_bin"], bin_df["known_or_reported_count"], color=colors, edgecolor="#94a3b8", linewidth=0.3)
    ax.set_xlabel("Redshift bin")
    ax.set_ylabel("Known/reported galaxy count")
    ax.set_title("Inventory count for 13 <= z <= 50: the frontier is crowded only near z=13-14.44")
    ax.set_ylim(0, max(2, int(bin_df["known_or_reported_count"].max()) + 1))
    plt.xticks(rotation=70, ha="right")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_Z50_BIN_INVENTORY.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_redshift_distance(df, curve):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7.5))
    style_dark(fig, ax)
    ax.plot(curve["z"], curve["comoving_distance_gly"], color="#f8fafc", linewidth=2.1, alpha=0.92, label="Planck18 distance curve, z=13..50")
    sc = ax.scatter(
        df["z"], df["comoving_distance_gly"],
        c=df["z"], cmap="coolwarm", vmin=Z_MIN, vmax=Z_MAX,
        s=180, edgecolors="#f8fafc", linewidths=0.8, alpha=0.97, zorder=5,
        label="Known/reported z>=13 galaxies",
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['name']}\nz={row['z']:.3f}\nD={row['comoving_distance_gly']:.2f} Gly",
            (row["z"], row["comoving_distance_gly"]), xytext=(7, 6), textcoords="offset points",
            fontsize=7.5, color="#f8fafc",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.68),
        )
    ax.axhline(OBSERVABLE_RADIUS_GLY, color="#f8fafc", linewidth=1.0, alpha=0.60)
    ax.text(Z_MIN + 0.5, OBSERVABLE_RADIUS_GLY - 0.45, "observable radius marker ~46.5 Gly", color="#f8fafc", fontsize=9)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("z=13 to z=50 inventory: theoretical curve continues; galaxy inventory does not")
    ax.set_xlim(Z_MIN - 0.5, Z_MAX + 1.0)
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9, loc="lower right")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_Z50_DISTANCE_CURVE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_cosmic_age(df, curve):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7.5))
    style_dark(fig, ax)
    ax.plot(curve["z"], curve["universe_age_myr"], color="#38bdf8", linewidth=2.1, alpha=0.95, label="Universe age at redshift")
    sc = ax.scatter(
        df["z"], df["universe_age_myr"],
        c=df["z"], cmap="coolwarm", vmin=Z_MIN, vmax=Z_MAX,
        s=180, edgecolors="#f8fafc", linewidths=0.8, alpha=0.97, zorder=5,
    )
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['name']}\n{row['universe_age_myr']:.0f} Myr",
            (row["z"], row["universe_age_myr"]), xytext=(8, 5), textcoords="offset points",
            fontsize=7.5, color="#f8fafc",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.68),
        )
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Universe age, Myr after Big Bang")
    ax.set_title("Age meaning of z=13..50: z=50 is before the known galaxy inventory")
    ax.set_xlim(Z_MIN - 0.5, Z_MAX + 1.0)
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9, loc="upper right")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_Z50_COSMIC_AGE_CURVE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_polar_inventory(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11.5, 11.5))
    fig.patch.set_facecolor("#050712")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#050712")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(30.0, OBSERVABLE_RADIUS_GLY)
    ax.set_rticks([31, 32, 33, 34, 36.5, 40, 46.5])
    ax.set_rlabel_position(135)
    ax.tick_params(colors="#dbeafe", labelsize=8)
    ax.xaxis.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.yaxis.grid(True, color="#475569", linewidth=0.55, alpha=0.70)
    ax.spines["polar"].set_color("#94a3b8")
    horizon = np.linspace(0, 2 * np.pi, 720)
    ax.fill_between(horizon, 36.5, OBSERVABLE_RADIUS_GLY, color="#7f1d1d", alpha=0.25)
    ax.plot(horizon, np.full_like(horizon, OBSERVABLE_RADIUS_GLY), color="#f8fafc", linewidth=1.2, alpha=0.85)
    ax.plot(horizon, np.full_like(horizon, 36.5), color="#f97316", linewidth=1.2, alpha=0.85)
    sc = ax.scatter(
        df["theta_ra_rad"], df["comoving_distance_gly"],
        c=df["z"], cmap="coolwarm", vmin=Z_MIN, vmax=Z_MAX,
        s=180, edgecolors="#f8fafc", linewidths=0.8, alpha=0.97, zorder=5,
    )
    for _, row in df.iterrows():
        ax.text(
            row["theta_ra_rad"], row["comoving_distance_gly"] + 0.35,
            f"{row['name']}\nz={row['z']:.2f}\nD={row['comoving_distance_gly']:.2f} Gly",
            fontsize=6.7, color="#f8fafc", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.70),
            zorder=6,
        )
    ax.text(
        np.deg2rad(230), 42.0,
        "Requested edge shell shaded\n36.5-46.5 Gly\nNo z=13..50 galaxy inventory points land there yet",
        color="#f8fafc", fontsize=9, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", edgecolor="#f97316", alpha=0.86),
    )
    ax.set_title("Inventory map for 13 <= z <= 50\nred = higher redshift / older light; blue = lower redshift", color="#f8fafc", fontsize=15, pad=30)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.10)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z13_Z50_POLAR_INVENTORY.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path

def save_outputs(df, bin_df, curve, paths):
    inventory_path = OUTPUT_CSV / f"{VERSION}_Z13_Z50_INVENTORY_TABLE.csv"
    bins_path = OUTPUT_CSV / f"{VERSION}_Z13_Z50_BIN_COUNTS.csv"
    curve_path = OUTPUT_CSV / f"{VERSION}_Z13_Z50_COSMOLOGY_CURVE.csv"
    claims_path = OUTPUT_CSV / f"{VERSION}_CLAIM_AUDIT_NOT_IN_INVENTORY.csv"
    notes_path = OUTPUT_CSV / f"{VERSION}_NOTES.txt"

    df.to_csv(inventory_path, index=False)
    bin_df.to_csv(bins_path, index=False)
    curve.to_csv(curve_path, index=False)

    import pandas as pd
    pd.DataFrame(CLAIM_AUDIT).to_csv(claims_path, index=False)

    notes_path.write_text(
        f"{VERSION} notes\n"
        "Inventory requested: galaxies with 13 <= z <= 50.\n"
        "This script uses a curated named/reported JWST frontier table, not a live all-archive extraction.\n"
        "Result: all included named galaxy entries lie at z=13.0 to z=14.44.\n"
        "Bins above z=15 are intentionally visible and empty in the inventory chart.\n"
        "Extreme photometric claims are not counted without robust confirmation; see claim audit CSV.\n"
        "Distances and ages are computed from redshift using astropy Planck18 cosmology.\n"
        "No FITS or image products are downloaded.\n",
        encoding="utf-8",
    )
    return inventory_path, bins_path, curve_path, claims_path, notes_path

def main():
    setup()

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Output folder", str(OUTPUT_DIR)),
        ("Redshift interval", f"{Z_MIN:.1f} <= z <= {Z_MAX:.1f}"),
        ("Known/reported entries", len(Z13_TO_Z50_OBJECTS)),
        ("Distance model", "Astropy Planck18 cosmology"),
        ("Mode", "Curated inventory; no FITS/image downloads"),
    ], ["Field", "Value"])

    df = build_inventory()
    bin_df = build_bins(df)
    curve = build_curve()

    paths = [
        plot_inventory_bins(bin_df),
        plot_redshift_distance(df, curve),
        plot_cosmic_age(df, curve),
        plot_polar_inventory(df),
    ]
    inventory_path, bins_path, curve_path, claims_path, notes_path = save_outputs(df, bin_df, curve, paths)

    print("")
    print("RESULTS")
    rows = [
        ("Inventory rows in 13<=z<=50", len(df)),
        ("Highest z in inventory", f"{df['z'].max():.4f}" if len(df) else "NONE"),
        ("Lowest z in inventory", f"{df['z'].min():.4f}" if len(df) else "NONE"),
        ("Non-empty 1-z bins", int((bin_df["known_or_reported_count"] > 0).sum())),
        ("Empty bins above z=15", int(((bin_df["z_start"] >= 15) & (bin_df["known_or_reported_count"] == 0)).sum())),
        ("Interpretation", "inventory stops near z=14.44; z=15..50 is empty here"),
    ]
    print_table(rows, ["Metric", "Value"])

    print("")
    print("INVENTORY TABLE")
    inv_rows = []
    for _, row in df.iterrows():
        inv_rows.append((
            int(row["rank_by_z"]), row["name"], f"{row['z']:.4f}",
            f"{row['comoving_distance_gly']:.3f}", f"{row['universe_age_myr']:.1f}", row["confidence"]
        ))
    print_table(inv_rows, ["Rank", "Name", "z", "Dist Gly", "Age Myr", "Confidence"])

    print("")
    print("OUTPUT SUMMARY")
    out_rows = [
        ("csv", str(inventory_path)),
        ("csv", str(bins_path)),
        ("csv", str(curve_path)),
        ("csv", str(claims_path)),
        ("notes", str(notes_path)),
    ]
    out_rows.extend(("png", str(path)) for path in paths)
    print_table(out_rows, ["Type", "Path"])

    print("")
    print("COMMENTS")
    print("This answers the inventory question directly: z=13..50 returns known/reported objects near z=13..14.44 only.")
    print("The higher-z bins are shown explicitly as empty instead of being hidden.")
    print("Red means higher redshift / older light; blue means lower redshift within the requested band.")

    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
