# JWST_0005
# Audit: Oldest galaxy redshift-distance relationship. Curated named objects; no FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0005"
PROJECT = "OLDEST GALAXIES DISTANCE LINEARITY TEST"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OBSERVABLE_RADIUS_GLY = 46.5

# Named high-redshift galaxy sample, mixing Hubble-era discoveries/candidates and JWST-era confirmations.
# This is not every galaxy in the Hubble Ultra Deep Field/XDF; those fields contain thousands of galaxies.
# It is a practical oldest-known / historically important sample for plotting the distance-redshift relationship.
ANCIENT_GALAXIES = [
    {"name":"MoM-z14", "z":14.440, "ra_deg":150.0933, "dec_deg":2.2731, "era":"JWST", "status":"spectroscopic", "field":"COSMOS/MoM", "note":"record-holder sample point"},
    {"name":"JADES-GS-z14-0", "z":14.1793, "ra_deg":53.08294, "dec_deg":-27.85563, "era":"JWST", "status":"spectroscopic+ALMA", "field":"GOODS-S/JADES", "note":"ALMA [OIII] precision redshift"},
    {"name":"JADES-GS-z14-1", "z":13.900, "ra_deg":53.1600, "dec_deg":-27.7800, "era":"JWST", "status":"spectroscopic", "field":"GOODS-S/JADES", "note":"approximate field placement"},
    {"name":"PAN-z14-1", "z":13.530, "ra_deg":334.2504, "dec_deg":0.3792, "era":"JWST", "status":"spectroscopic", "field":"PANORAMIC", "note":"published coordinate"},
    {"name":"JADES-GS-z13-0", "z":13.200, "ra_deg":53.14988, "dec_deg":-27.77650, "era":"JWST", "status":"spectroscopic", "field":"GOODS-S/JADES", "note":"early JWST record holder"},
    {"name":"UNCOVER-z13", "z":13.079, "ra_deg":3.5860, "dec_deg":-30.4000, "era":"JWST", "status":"spectroscopic/plausible", "field":"Abell 2744", "note":"lensed source"},
    {"name":"JADES-GS-z13-1-LA", "z":13.000, "ra_deg":53.1500, "dec_deg":-27.7800, "era":"JWST", "status":"spectroscopic", "field":"GOODS-S/JADES", "note":"z=13 cutoff object"},
    {"name":"JADES-GS-z12-0", "z":12.630, "ra_deg":53.1600, "dec_deg":-27.7700, "era":"JWST", "status":"spectroscopic", "field":"GOODS-S/JADES", "note":"high-z JADES comparison"},
    {"name":"UNCOVER-z12", "z":12.393, "ra_deg":3.5860, "dec_deg":-30.4000, "era":"JWST", "status":"spectroscopic", "field":"Abell 2744", "note":"lensed source"},
    {"name":"CAPERS-EGS-65480", "z":12.344, "ra_deg":214.9000, "dec_deg":52.9000, "era":"JWST", "status":"spectroscopic", "field":"EGS/CAPERS", "note":"EGS approximate placement"},
    {"name":"GLASS-z12/GHZ2", "z":12.340, "ra_deg":3.5860, "dec_deg":-30.4000, "era":"JWST", "status":"spectroscopic/ALMA", "field":"Abell 2744/GLASS", "note":"high-z GLASS comparison"},
    {"name":"UDFj-39546284", "z":11.900, "ra_deg":53.1624, "dec_deg":-27.7910, "era":"Hubble", "status":"photometric/candidate", "field":"HUDF", "note":"classic Hubble extreme candidate; disputed/interloper possible"},
    {"name":"Maisie's Galaxy", "z":11.440, "ra_deg":214.9432, "dec_deg":52.9424, "era":"JWST", "status":"spectroscopic", "field":"CEERS/EGS", "note":"CEERS source"},
    {"name":"GS-z11-1", "z":11.275, "ra_deg":53.1600, "dec_deg":-27.7800, "era":"JWST", "status":"spectroscopic", "field":"GOODS-S/JADES", "note":"JADES z~11 object"},
    {"name":"GN-z11", "z":10.6034, "ra_deg":189.1068, "dec_deg":62.2420, "era":"Hubble+JWST", "status":"spectroscopic", "field":"GOODS-N", "note":"Hubble-era record; JWST follow-up"},
    {"name":"MACS0647-JD", "z":10.170, "ra_deg":101.9830, "dec_deg":70.2480, "era":"Hubble+JWST", "status":"photometric/lensed", "field":"MACS J0647.7+7015", "note":"lensed Hubble/JWST source"},
    {"name":"UDF12-3954-6284", "z":10.000, "ra_deg":53.1600, "dec_deg":-27.7900, "era":"Hubble", "status":"photometric/candidate", "field":"HUDF12", "note":"Hubble WFC3/IR candidate class"},
    {"name":"MACS1149-JD1", "z":9.110, "ra_deg":177.3970, "dec_deg":22.3980, "era":"Hubble+ALMA", "status":"spectroscopic/ALMA", "field":"MACS J1149", "note":"oxygen line; old stellar population evidence"},
    {"name":"EGS-zs8-1", "z":7.730, "ra_deg":215.0250, "dec_deg":53.0070, "era":"Hubble/Spitzer", "status":"spectroscopic", "field":"EGS", "note":"bright Lyman-break galaxy"},
    {"name":"EGS-zs8-2", "z":7.480, "ra_deg":215.1300, "dec_deg":53.0500, "era":"Hubble/Spitzer", "status":"spectroscopic", "field":"EGS", "note":"EGS high-z comparison"},
    {"name":"z8_GND_5296", "z":7.508, "ra_deg":189.0660, "dec_deg":62.2380, "era":"Hubble/Keck", "status":"spectroscopic", "field":"GOODS-N", "note":"high star-formation source"},
    {"name":"A1689-zD1", "z":7.130, "ra_deg":197.8729, "dec_deg":-1.3346, "era":"Hubble/Spitzer/ALMA", "status":"photometric+ALMA", "field":"Abell 1689", "note":"dusty lensed young galaxy"},
    {"name":"SXDF-NB1006-2", "z":7.213, "ra_deg":34.7370, "dec_deg":-5.1780, "era":"Subaru/Hubble", "status":"spectroscopic", "field":"SXDF", "note":"Lyman-alpha emitter"},
    {"name":"IOK-1", "z":6.960, "ra_deg":201.4910, "dec_deg":27.4130, "era":"Subaru/Hubble", "status":"spectroscopic", "field":"Coma Berenices", "note":"historic z~7 Lyman-alpha galaxy"},
    {"name":"LAE J095950+021219", "z":6.944, "ra_deg":149.9625, "dec_deg":2.2053, "era":"Ground+Hubble", "status":"spectroscopic", "field":"COSMOS", "note":"Lyman-alpha emitter"},
    {"name":"Himiko", "z":6.595, "ra_deg":34.5940, "dec_deg":-5.1510, "era":"Subaru/Hubble", "status":"spectroscopic", "field":"SXDS", "note":"giant Lyman-alpha blob"},
    {"name":"CR7", "z":6.604, "ra_deg":150.2410, "dec_deg":1.8040, "era":"VLT/Hubble", "status":"spectroscopic", "field":"COSMOS", "note":"bright Lyman-alpha emitter"},
    {"name":"HCM 6A", "z":6.560, "ra_deg":36.2450, "dec_deg":-4.6830, "era":"Keck/Hubble", "status":"spectroscopic", "field":"Abell 370", "note":"lensed galaxy"},
    {"name":"HFLS3", "z":6.340, "ra_deg":261.4250, "dec_deg":58.7740, "era":"Herschel/Hubble", "status":"spectroscopic", "field":"Lockman", "note":"extreme starburst"},
    {"name":"HDF 4-473.0", "z":5.600, "ra_deg":189.2100, "dec_deg":62.2300, "era":"Hubble", "status":"spectroscopic", "field":"HDF-N", "note":"classic Hubble Deep Field high-z source"},
    {"name":"HDF850.1", "z":5.183, "ra_deg":189.2000, "dec_deg":62.2070, "era":"Hubble/SCUBA", "status":"spectroscopic", "field":"HDF-N", "note":"submillimeter galaxy in HDF"},
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

    df = pd.DataFrame(ANCIENT_GALAXIES).sort_values("z", ascending=False).reset_index(drop=True)
    df["rank_by_z"] = range(1, len(df) + 1)
    df["theta_ra_rad"] = np.deg2rad(df["ra_deg"] % 360.0)
    df["comoving_distance_gly"] = [cosmo.comoving_distance(z).to("Glyr").value for z in df["z"]]
    df["light_travel_time_gyr"] = [cosmo.lookback_time(z).value for z in df["z"]]
    df["universe_age_gyr"] = [cosmo.age(z).value for z in df["z"]]
    df["universe_age_myr"] = df["universe_age_gyr"] * 1000.0
    df["proper_distance_then_gly"] = df["comoving_distance_gly"] / (1.0 + df["z"])
    df["scale_factor_then"] = 1.0 / (1.0 + df["z"])
    df["edge_gap_gly"] = OBSERVABLE_RADIUS_GLY - df["comoving_distance_gly"]
    return df

def cosmology_curve():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    z = np.linspace(0.001, 15.0, 600)
    out = pd.DataFrame({"z": z})
    out["comoving_distance_gly"] = [cosmo.comoving_distance(v).to("Glyr").value for v in z]
    out["light_travel_time_gyr"] = [cosmo.lookback_time(v).value for v in z]
    out["universe_age_gyr"] = [cosmo.age(v).value for v in z]
    return out

def regression_metrics(df):
    import numpy as np

    x = df["z"].to_numpy()
    y = df["comoving_distance_gly"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rms = float(np.sqrt(np.mean((y - pred) ** 2)))
    return slope, intercept, r2, rms

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

def plot_distance_curve(df, curve, slope, intercept, r2, rms):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7.5))
    style_dark(fig, ax)

    ax.plot(curve["z"], curve["comoving_distance_gly"], linewidth=2.2, alpha=0.95, label="Planck18 cosmology curve")
    fit_x = np.linspace(df["z"].min(), df["z"].max(), 200)
    ax.plot(fit_x, slope * fit_x + intercept, linestyle="--", linewidth=1.4, alpha=0.90, label="Linear fit to plotted galaxies")

    colors = df["universe_age_myr"]
    scatter = ax.scatter(
        df["z"], df["comoving_distance_gly"],
        c=colors, cmap="viridis_r", s=90,
        edgecolors="#f8fafc", linewidths=0.7, alpha=0.95, zorder=5,
    )

    for _, row in df.head(14).iterrows():
        ax.annotate(
            f"{row['name']}\n{row['comoving_distance_gly']:.2f} Gly",
            (row["z"], row["comoving_distance_gly"]),
            xytext=(6, 5), textcoords="offset points",
            fontsize=7, color="#f8fafc",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.65),
        )

    ax.axhline(OBSERVABLE_RADIUS_GLY, color="#f8fafc", linewidth=1.0, alpha=0.55)
    ax.text(0.4, OBSERVABLE_RADIUS_GLY - 1.0, "observable radius marker ≈ 46.5 Gly", color="#f8fafc", fontsize=9)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("Oldest/high-redshift galaxies: distance is monotonic, but not linear")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    text = f"Linear fit over sample:\nD = {slope:.3f} z + {intercept:.3f} Gly\nR² = {r2:.5f}\nRMS residual = {rms:.3f} Gly"
    ax.text(0.03, 0.05, text, transform=ax.transAxes, color="#f8fafc", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", edgecolor="#94a3b8", alpha=0.82))

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Universe age at emission, Myr", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_DISTANCE_REDSHIFT_NONLINEAR_CURVE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_age_distance(df, curve):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7.5))
    style_dark(fig, ax)

    ax.plot(curve["comoving_distance_gly"], curve["universe_age_gyr"] * 1000.0, linewidth=2.2, alpha=0.95, label="Planck18 cosmic age curve")
    sc = ax.scatter(df["comoving_distance_gly"], df["universe_age_myr"], c=df["z"], cmap="plasma", s=95,
                    edgecolors="#f8fafc", linewidths=0.7, alpha=0.95, zorder=5)

    for _, row in df.head(18).iterrows():
        ax.annotate(row["name"], (row["comoving_distance_gly"], row["universe_age_myr"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7, color="#f8fafc")

    ax.set_xlabel("Comoving distance, billion light-years")
    ax.set_ylabel("Age of universe when light was emitted, Myr")
    ax.set_title("The farther ancient galaxies are, the younger the universe was — strongly nonlinear")
    ax.legend(facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=9)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_COSMIC_AGE_DISTANCE_CURVE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_polar_distance(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 11))
    fig.patch.set_facecolor("#050712")
    ax = fig.add_subplot(111, projection="polar")
    ax.set_facecolor("#050712")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.tick_params(colors="#dbeafe", labelsize=8)
    ax.xaxis.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.yaxis.grid(True, color="#475569", linewidth=0.55, alpha=0.70)
    ax.spines["polar"].set_color("#94a3b8")
    ax.set_rlim(0, OBSERVABLE_RADIUS_GLY)
    ax.set_rticks([10, 20, 30, 34, 40, 46.5])
    ax.set_rlabel_position(135)

    horizon = np.linspace(0, 2 * np.pi, 720)
    ax.plot(horizon, np.full_like(horizon, OBSERVABLE_RADIUS_GLY), color="#f8fafc", linewidth=1.3, alpha=0.85)
    ax.fill_between(horizon, 33.0, OBSERVABLE_RADIUS_GLY, color="#1e1b4b", alpha=0.35)

    sc = ax.scatter(df["theta_ra_rad"], df["comoving_distance_gly"], c=df["z"], cmap="plasma", s=115,
                    edgecolors="#f8fafc", linewidths=0.7, alpha=0.93, zorder=5)

    for _, row in df.head(16).iterrows():
        ax.text(row["theta_ra_rad"], row["comoving_distance_gly"] + 1.0,
                f"{row['name']}\nz={row['z']:.2f}\n{row['comoving_distance_gly']:.1f} Gly",
                fontsize=6.6, color="#f8fafc", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#020617", edgecolor="#475569", alpha=0.70), zorder=6)

    ax.set_title("Oldest named galaxies mapped by RA and present comoving distance", color="#f8fafc", fontsize=15, pad=30)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.70, pad=0.10)
    cbar.set_label("Redshift z", color="#f8fafc")
    cbar.ax.yaxis.set_tick_params(color="#f8fafc")
    plt.setp(cbar.ax.get_yticklabels(), color="#f8fafc")
    cbar.outline.set_edgecolor("#94a3b8")

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_OLDEST_GALAXIES_POLAR_DISTANCE_MAP.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def plot_hubble_vs_jwst(df):
    import matplotlib.pyplot as plt

    ordered = df.sort_values("z", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    style_dark(fig, ax)

    colors = ordered["era"].map(lambda x: "#f97316" if "JWST" in x else "#38bdf8")
    bars = ax.barh(ordered["name"], ordered["z"], color=colors, alpha=0.88)

    for bar, (_, row) in zip(bars, ordered.iterrows()):
        ax.text(bar.get_width() + 0.10, bar.get_y() + bar.get_height()/2,
                f"z={row['z']:.3f} | D={row['comoving_distance_gly']:.2f} Gly | age={row['universe_age_myr']:.0f} Myr",
                va="center", color="#f8fafc", fontsize=7.2)

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Named galaxy/sample")
    ax.set_title("Named ancient galaxies: Hubble-era discoveries versus JWST-era confirmations")
    ax.set_xlim(0, max(df["z"]) + 1.2)
    ax.text(0.02, 0.03, "Blue = Hubble/Hubble-era; orange = JWST involved", transform=ax.transAxes,
            color="#f8fafc", fontsize=9, bbox=dict(boxstyle="round,pad=0.30", facecolor="#111827", edgecolor="#94a3b8", alpha=0.82))

    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_OLDEST_GALAXIES_RANKED_REDSHIFT.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path

def save_outputs(df, curve, paths, slope, intercept, r2, rms):
    csv_path = OUTPUT_CSV / f"{VERSION}_OLDEST_GALAXIES_DISTANCE_TABLE.csv"
    curve_path = OUTPUT_CSV / f"{VERSION}_PLANCK18_REDSHIFT_DISTANCE_CURVE.csv"
    notes_path = OUTPUT_CSV / f"{VERSION}_NOTES.txt"

    df.to_csv(csv_path, index=False)
    curve.to_csv(curve_path, index=False)
    notes_path.write_text(
        "JWST_0005 notes\n"
        "Question: are oldest/highest-redshift galaxies farther away, and is the relationship linear?\n"
        "Answer: redshift, lookback time, and comoving distance are monotonic for these objects, but not linear.\n"
        "Hubble-Lemaitre law is approximately linear for nearby low-redshift galaxies, not for z~5-15 galaxies.\n"
        "This script plots a named ancient-galaxy sample, not every galaxy in the Hubble Ultra Deep Field/XDF.\n"
        "The HUDF/XDF contain thousands of galaxies; catalog-grade all-object extraction requires a dedicated source catalog.\n"
        f"Linear fit over sample: D = {slope:.6f} z + {intercept:.6f} Gly, R2={r2:.8f}, RMS={rms:.6f} Gly.\n"
        "No FITS/image products are downloaded.\n",
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
        ("Named objects", len(ANCIENT_GALAXIES)),
        ("Distance model", "Astropy Planck18 cosmology"),
        ("Question", "distance/redshift linearity"),
        ("Mode", "Curated metadata only; no FITS/image downloads"),
    ], ["Field", "Value"])

    df = build_dataframe()
    curve = cosmology_curve()
    slope, intercept, r2, rms = regression_metrics(df)

    paths = [
        plot_distance_curve(df, curve, slope, intercept, r2, rms),
        plot_age_distance(df, curve),
        plot_polar_distance(df),
        plot_hubble_vs_jwst(df),
    ]

    csv_path, curve_path, notes_path = save_outputs(df, curve, paths, slope, intercept, r2, rms)

    print("")
    print("RESULTS")
    print_table([
        ("Linear fit slope", f"{slope:.6f} Gly per redshift"),
        ("Linear fit intercept", f"{intercept:.6f} Gly"),
        ("Linear fit R2", f"{r2:.8f}"),
        ("Linear fit RMS residual", f"{rms:.6f} Gly"),
        ("Interpretation", "monotonic but nonlinear at high z"),
    ], ["Metric", "Value"])

    print("")
    print("OLDEST OBJECTS")
    rows = []
    for _, row in df.head(18).iterrows():
        rows.append((
            int(row["rank_by_z"]), row["name"], f"{row['z']:.4f}",
            f"{row['comoving_distance_gly']:.3f}", f"{row['universe_age_myr']:.1f}", row["era"]
        ))
    print_table(rows, ["Rank", "Name", "z", "Dist Gly", "Age Myr", "Era"])

    print("")
    print("OUTPUT SUMMARY")
    output_rows = [("csv", str(csv_path)), ("csv", str(curve_path)), ("notes", str(notes_path))]
    output_rows.extend(("png", str(path)) for path in paths)
    print_table(output_rows, ["Type", "Path"])

    print("")
    print("COMMENTS")
    print("For nearby galaxies, Hubble's law is approximately linear.")
    print("For the oldest galaxies, the redshift-distance curve bends and approaches the observable-universe scale.")
    print("This is why the plot uses the full cosmological distance calculation instead of a simple straight line.")

    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
