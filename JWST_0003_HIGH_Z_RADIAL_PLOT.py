# JWST_0003
# Audit: High-redshift galaxy radial plot. Curated metadata only; no FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0003"
PROJECT = "JWST HIGH-REDSHIFT GALAXY RADIAL MAP"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
OUTPUT_PNG = OUTPUT_DIR / "PNG"

# Curated spectroscopic/high-confidence high-redshift sample.
# Coordinates are used only as angular placement in the polar plot.
# If a source is in a known survey field but exact coordinates are not carried here,
# the field center / published object-name coordinates are used and the coordinate_status notes it.
HIGH_Z_GALAXIES = [
    {
        "rank": 1,
        "name": "MoM-z14",
        "redshift_z": 14.44,
        "ra_deg": 150.093333,
        "dec_deg": 2.273108,
        "field": "COSMOS / Mirage or Miracle",
        "confirmation": "spectroscopic",
        "coordinate_status": "published wiki infobox / field object coordinate",
        "notes": "Currently reported record-holder in this curated list.",
    },
    {
        "rank": 2,
        "name": "JADES-GS-z14-0",
        "redshift_z": 14.1796,
        "ra_deg": 53.08294,
        "dec_deg": -27.85563,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic JWST + ALMA line",
        "coordinate_status": "object-name coordinate JADES-GS-53.08294-27.85563",
        "notes": "ALMA [OIII] 88 micron redshift used for precision.",
    },
    {
        "rank": 3,
        "name": "JADES-GS-z14-1",
        "redshift_z": 13.86,
        "ra_deg": 53.1600,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic",
        "coordinate_status": "GOODS-S approximate placement",
        "notes": "Very faint z≈14 JADES galaxy; exact coordinate can be patched later.",
    },
    {
        "rank": 4,
        "name": "PAN-z14-1",
        "redshift_z": 13.53,
        "ra_deg": 334.25035598,
        "dec_deg": 0.3792145611,
        "field": "PANORAMIC pure-parallel",
        "confirmation": "spectroscopic",
        "coordinate_status": "published coordinate",
        "notes": "Large luminous galaxy with weak emission lines.",
    },
    {
        "rank": 5,
        "name": "JADES-GS-z13-0",
        "redshift_z": 13.20,
        "ra_deg": 53.14988,
        "dec_deg": -27.77650,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic",
        "coordinate_status": "object-name coordinate JADES-GS+53.14988-27.77650",
        "notes": "Former record-holder after early JWST confirmation.",
    },
    {
        "rank": 6,
        "name": "UNCOVER-z13",
        "redshift_z": 13.079,
        "ra_deg": 3.5860,
        "dec_deg": -30.4000,
        "field": "Abell 2744 / UNCOVER",
        "confirmation": "spectroscopic plausible/tentative",
        "coordinate_status": "Abell 2744 approximate placement",
        "notes": "Lensed source; redshift reported as plausible in UNCOVER NIRSpec work.",
    },
    {
        "rank": 7,
        "name": "JADES-GS-z13-1-LA",
        "redshift_z": 13.00,
        "ra_deg": 53.1500,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic",
        "coordinate_status": "GOODS-S approximate placement",
        "notes": "Included as z≈13 confirmed Lyman-alpha object; coordinate patch later.",
    },
    {
        "rank": 8,
        "name": "JADES-GS-z12-0",
        "redshift_z": 12.48,
        "ra_deg": 53.1600,
        "dec_deg": -27.7700,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic",
        "coordinate_status": "GOODS-S approximate placement",
        "notes": "High-redshift JADES comparison object.",
    },
    {
        "rank": 9,
        "name": "UNCOVER-z12",
        "redshift_z": 12.393,
        "ra_deg": 3.5860,
        "dec_deg": -30.4000,
        "field": "Abell 2744 / UNCOVER",
        "confirmation": "spectroscopic",
        "coordinate_status": "Abell 2744 approximate placement",
        "notes": "Lensed source in A2744, spectroscopically confirmed.",
    },
    {
        "rank": 10,
        "name": "CAPERS-EGS-65480",
        "redshift_z": 12.344,
        "ra_deg": 214.9000,
        "dec_deg": 52.9000,
        "field": "EGS / CAPERS",
        "confirmation": "spectroscopic",
        "coordinate_status": "EGS approximate placement",
        "notes": "Included as z≈12.34 comparison object.",
    },
    {
        "rank": 11,
        "name": "GLASS-z12 / GHZ2",
        "redshift_z": 12.34,
        "ra_deg": 3.5860,
        "dec_deg": -30.4000,
        "field": "Abell 2744 / GLASS",
        "confirmation": "spectroscopic",
        "coordinate_status": "Abell 2744 approximate placement",
        "notes": "ALMA/JWST high-redshift comparison source.",
    },
    {
        "rank": 12,
        "name": "Maisie's Galaxy",
        "redshift_z": 11.40,
        "ra_deg": 214.9432,
        "dec_deg": 52.9424,
        "field": "CEERS / EGS",
        "confirmation": "spectroscopic",
        "coordinate_status": "object-name coordinate CEERS J141946.36+525632.8",
        "notes": "Bright CEERS source, useful lower-z anchor.",
    },
    {
        "rank": 13,
        "name": "GS-z11-1",
        "redshift_z": 11.275,
        "ra_deg": 53.1600,
        "dec_deg": -27.7800,
        "field": "GOODS-S / JADES",
        "confirmation": "spectroscopic",
        "coordinate_status": "GOODS-S approximate placement",
        "notes": "JADES z≈11 comparison object.",
    },
    {
        "rank": 14,
        "name": "GN-z11",
        "redshift_z": 10.6034,
        "ra_deg": 189.1068,
        "dec_deg": 62.2420,
        "field": "GOODS-N",
        "confirmation": "spectroscopic",
        "coordinate_status": "published object coordinate",
        "notes": "Hubble-era record-holder later confirmed/revised with spectroscopy.",
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
        widths.append(min(width, 36))
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

    df = pd.DataFrame(HIGH_Z_GALAXIES).sort_values("redshift_z", ascending=False).reset_index(drop=True)
    df["plot_rank"] = range(1, len(df) + 1)
    df["theta_ra_rad"] = np.deg2rad(df["ra_deg"] % 360.0)
    df["lookback_time_gyr"] = [cosmo.lookback_time(z).value for z in df["redshift_z"]]
    df["universe_age_gyr"] = [cosmo.age(z).value for z in df["redshift_z"]]
    df["universe_age_myr"] = df["universe_age_gyr"] * 1000.0
    df["comoving_distance_gly"] = [cosmo.comoving_distance(z).to("Glyr").value for z in df["redshift_z"]]
    df["luminosity_distance_gly"] = [cosmo.luminosity_distance(z).to("Glyr").value for z in df["redshift_z"]]
    df["scale_factor"] = 1.0 / (1.0 + df["redshift_z"])
    df["observed_wavelength_for_1216um"] = 0.1216 * (1.0 + df["redshift_z"])
    return df

def plot_redshift_polar(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    theta = df["theta_ra_rad"].to_numpy()
    radius = df["redshift_z"].to_numpy()

    sizes = 45 + 9 * (df["redshift_z"].to_numpy() - df["redshift_z"].min())
    ax.scatter(theta, radius, s=sizes, alpha=0.82)

    for _, row in df.iterrows():
        label = row["name"]
        angle = row["theta_ra_rad"]
        r = row["redshift_z"]
        ax.text(angle, r + 0.10, label, fontsize=7, ha="center", va="center")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(10.0, 15.0)
    ax.set_rticks([10, 11, 12, 13, 14, 15])
    ax.set_rlabel_position(135)
    ax.grid(True, linewidth=0.45, alpha=0.45)
    ax.set_title("Highest-redshift galaxies — radial plot by RA and redshift z", pad=24)

    path = OUTPUT_PNG / f"{VERSION}_HIGH_Z_GALAXIES_RADIAL_REDSHIFT.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.show()
    return path

def plot_cosmic_age_polar(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    theta = df["theta_ra_rad"].to_numpy()
    radius = df["universe_age_myr"].to_numpy()

    ax.scatter(theta, radius, s=70, alpha=0.82)

    for _, row in df.iterrows():
        label = f"{row['name']}\n{row['universe_age_myr']:.0f} Myr"
        ax.text(row["theta_ra_rad"], row["universe_age_myr"] + 8, label, fontsize=6.5, ha="center", va="center")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(250, 500)
    ax.set_rticks([275, 300, 325, 350, 400, 450, 500])
    ax.set_rlabel_position(135)
    ax.grid(True, linewidth=0.45, alpha=0.45)
    ax.set_title("Cosmic age when the light was emitted — lower radius is earlier", pad=24)

    path = OUTPUT_PNG / f"{VERSION}_HIGH_Z_GALAXIES_RADIAL_COSMIC_AGE.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.show()
    return path

def plot_redshift_bar(df):
    import matplotlib.pyplot as plt

    ordered = df.sort_values("redshift_z", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(ordered["name"], ordered["redshift_z"])
    ax.set_xlabel("Spectroscopic redshift z")
    ax.set_ylabel("Galaxy")
    ax.set_title("Highest-redshift galaxy sample")
    ax.grid(True, axis="x", linewidth=0.45, alpha=0.45)
    fig.tight_layout()

    path = OUTPUT_PNG / f"{VERSION}_HIGH_Z_GALAXIES_REDSHIFT_RANKING.png"
    fig.savefig(path, dpi=220)
    plt.show()
    return path

def save_outputs(df, plot_paths):
    csv_path = OUTPUT_CSV / f"{VERSION}_HIGH_Z_GALAXIES_CURATED.csv"
    df.to_csv(csv_path, index=False)

    source_path = OUTPUT_CSV / f"{VERSION}_HIGH_Z_GALAXIES_SOURCE_NOTES.txt"
    source_path.write_text(
        "JWST_0003 curated high-redshift galaxy plot notes\n"
        "This table is intentionally embedded for fast Colab plotting.\n"
        "No FITS/image products are downloaded.\n"
        "Several coordinate entries are approximate survey-field placements and are flagged in coordinate_status.\n"
        "Patch exact RA/Dec values later from the primary tables when the project moves from visualization to catalog-grade astrometry.\n",
        encoding="utf-8",
    )

    return csv_path, source_path

def main():
    setup()

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Output folder", str(OUTPUT_DIR)),
        ("Objects", len(HIGH_Z_GALAXIES)),
        ("Mode", "Curated metadata only; no FITS/image downloads"),
        ("Radial axis", "redshift z and cosmic age"),
        ("Angular axis", "RA degrees converted to polar angle"),
    ], ["Field", "Value"])

    df = build_dataframe()

    redshift_polar = plot_redshift_polar(df)
    age_polar = plot_cosmic_age_polar(df)
    ranking_bar = plot_redshift_bar(df)
    csv_path, source_path = save_outputs(df, [redshift_polar, age_polar, ranking_bar])

    print("")
    print("RESULTS")
    rows = []
    for _, row in df.iterrows():
        rows.append((
            int(row["plot_rank"]),
            row["name"],
            f"{row['redshift_z']:.4f}",
            f"{row['universe_age_myr']:.1f}",
            f"{row['comoving_distance_gly']:.3f}",
            row["confirmation"],
        ))
    print_table(rows, ["Rank", "Name", "z", "Age Myr", "Comov Gly", "Confirmation"])

    print("")
    print("OUTPUT SUMMARY")
    print_table([
        ("csv", str(csv_path)),
        ("notes", str(source_path)),
        ("png", str(redshift_polar)),
        ("png", str(age_polar)),
        ("png", str(ranking_bar)),
    ], ["Type", "Path"])

    print("")
    print("COMMENTS")
    print("This is a visual science map, not a final astrometric catalog.")
    print("Objects with approximate coordinates are explicitly flagged in coordinate_status.")
    print("Next logical step: patch exact RA/Dec from the primary paper tables and add uncertainty bars.")

    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
