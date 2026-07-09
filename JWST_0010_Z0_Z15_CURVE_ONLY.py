# JWST_0010
# Audit: curve-only z=0..15 cosmology plots for JWST project.
# Matplotlib only. No AI images. No individual galaxy markers. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0010"
PROJECT = "JWST Z0 TO Z15 CURVE ONLY"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
Z_MIN = 0.0
Z_MAX = 15.0
Z_REFERENCE = 14.44
OBSERVABLE_RADIUS_GLY = 46.5


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


def print_table(rows, headers):
    widths = []
    for i, header in enumerate(headers):
        width = max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        widths.append(min(width, 48))
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


def build_curve():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    z = np.linspace(Z_MIN, Z_MAX, 3001)
    curve = pd.DataFrame({"redshift_z": z})
    curve["comoving_distance_gly"] = [cosmo.comoving_distance(v).to("Glyr").value for v in z]
    curve["lookback_time_gyr"] = [cosmo.lookback_time(v).value for v in z]
    curve["universe_age_gyr"] = [cosmo.age(v).value for v in z]
    curve["universe_age_myr"] = curve["universe_age_gyr"] * 1000.0
    curve["proper_distance_then_gly"] = curve["comoving_distance_gly"] / (1.0 + curve["redshift_z"])
    curve["observable_radius_fraction"] = curve["comoving_distance_gly"] / OBSERVABLE_RADIUS_GLY
    return curve


def sample_curve(curve):
    import pandas as pd
    samples = [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 14.44, 15.0]
    rows = []
    for z in samples:
        idx = (curve["redshift_z"] - z).abs().idxmin()
        row = curve.loc[idx].copy()
        row["sample_note"] = "reference max z" if abs(z - Z_REFERENCE) < 1e-9 else "curve sample"
        row["redshift_z"] = z
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


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


def plot_distance_curve(curve):
    import matplotlib.pyplot as plt

    ref = curve.iloc[(curve["redshift_z"] - Z_REFERENCE).abs().idxmin()]
    fig, ax = plt.subplots(figsize=(13.8, 8.0))
    style_dark(fig, ax)

    ax.plot(curve["redshift_z"], curve["comoving_distance_gly"], linewidth=2.4, color="#f8fafc", alpha=0.96)
    ax.fill_between(curve["redshift_z"], curve["comoving_distance_gly"], color="#172554", alpha=0.30)
    ax.axvspan(0.0, 1.0, color="#0e7490", alpha=0.12)
    ax.axvspan(1.0, 6.0, color="#1e3a8a", alpha=0.10)
    ax.axvspan(6.0, 15.0, color="#7f1d1d", alpha=0.11)
    ax.axvline(Z_REFERENCE, color="#f97316", linestyle="--", linewidth=1.25, alpha=0.92)
    ax.axhline(ref["comoving_distance_gly"], color="#f97316", linestyle=":", linewidth=1.0, alpha=0.74)

    ax.text(0.18, 2.1, "low-z region\nnear-linear", color="#cffafe", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#020617", edgecolor="#155e75", alpha=0.72))
    ax.text(6.15, 21.0, "curve bends:\neach added redshift\nadds less distance", color="#f8fafc", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#020617", edgecolor="#475569", alpha=0.72))
    ax.text(10.7, ref["comoving_distance_gly"] + 0.55,
            f"z={Z_REFERENCE:.2f} reference\nD={ref['comoving_distance_gly']:.3f} Gly\nage={ref['universe_age_myr']:.1f} Myr",
            color="#f8fafc", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#111827", edgecolor="#f97316", alpha=0.86))

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 36)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, billion light-years")
    ax.set_title("Cosmology curve from z=0 to z=15\ncurve only: no individual galaxies plotted")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z0_Z15_COMOVING_DISTANCE_CURVE.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_age_curve(curve):
    import matplotlib.pyplot as plt

    ref = curve.iloc[(curve["redshift_z"] - Z_REFERENCE).abs().idxmin()]
    fig, ax = plt.subplots(figsize=(13.8, 8.0))
    style_dark(fig, ax)

    ax.plot(curve["redshift_z"], curve["universe_age_myr"], linewidth=2.4, color="#38bdf8", alpha=0.96, label="Universe age")
    ax.fill_between(curve["redshift_z"], curve["universe_age_myr"], color="#0e7490", alpha=0.20)
    ax.axvline(Z_REFERENCE, color="#f97316", linestyle="--", linewidth=1.25, alpha=0.92)
    ax.axhline(ref["universe_age_myr"], color="#f97316", linestyle=":", linewidth=1.0, alpha=0.74)
    ax.text(8.4, 2600,
            "As redshift rises,\nthe universe age at emission\ndrops rapidly toward cosmic dawn.",
            color="#f8fafc", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#020617", edgecolor="#475569", alpha=0.75))
    ax.text(10.7, ref["universe_age_myr"] + 520,
            f"z={Z_REFERENCE:.2f} reference\nage={ref['universe_age_myr']:.1f} Myr",
            color="#f8fafc", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#111827", edgecolor="#f97316", alpha=0.86))

    ax.set_xlim(0, 15)
    ax.set_ylim(0, curve["universe_age_myr"].max() * 1.02)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Universe age at emission, Myr after Big Bang")
    ax.set_title("Universe age curve from z=0 to z=15\ncurve only: no individual galaxies plotted")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z0_Z15_UNIVERSE_AGE_CURVE.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_distance_derivative(curve):
    import numpy as np
    import matplotlib.pyplot as plt

    d_dz = np.gradient(curve["comoving_distance_gly"], curve["redshift_z"])
    fig, ax = plt.subplots(figsize=(13.8, 7.7))
    style_dark(fig, ax)
    ax.plot(curve["redshift_z"], d_dz, linewidth=2.35, color="#f97316", alpha=0.95)
    ax.fill_between(curve["redshift_z"], d_dz, color="#7f1d1d", alpha=0.20)
    ax.axvline(Z_REFERENCE, color="#f8fafc", linestyle="--", linewidth=1.15, alpha=0.75)
    ax.text(5.6, float(d_dz.max()) * 0.58,
            "This is why the distance curve flattens:\nΔdistance per unit redshift decreases\nat high redshift.",
            color="#f8fafc", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="#020617", edgecolor="#475569", alpha=0.76))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, float(d_dz.max()) * 1.05)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("d(comoving distance)/dz, Gly per redshift")
    ax.set_title("Slope of the z=0 to z=15 distance curve\nthe curve is smooth, but the slope continuously falls")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_Z0_Z15_DISTANCE_SLOPE_CURVE.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_table(sample_df):
    import matplotlib.pyplot as plt

    display = sample_df[[
        "redshift_z",
        "comoving_distance_gly",
        "lookback_time_gyr",
        "universe_age_myr",
        "proper_distance_then_gly",
        "sample_note",
    ]].copy()
    display.columns = ["z", "Comoving D (Gly)", "Lookback (Gyr)", "Age (Myr)", "Proper then (Gly)", "Status"]

    rows = []
    for _, row in display.iterrows():
        rows.append([
            f"{row['z']:.2f}",
            f"{row['Comoving D (Gly)']:.3f}",
            f"{row['Lookback (Gyr)']:.3f}",
            f"{row['Age (Myr)']:.1f}",
            f"{row['Proper then (Gly)']:.3f}",
            row["Status"],
        ])

    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=list(display.columns), loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1.0, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            z = float(rows[r - 1][0])
            if z >= 10:
                cell.set_facecolor("#3b1114")
            elif z >= 3:
                cell.set_facecolor("#172554")
            else:
                cell.set_facecolor("#082f49")
            cell.get_text().set_color("#e5e7eb")

    ax.set_title("JWST_0010 sampled z=0 to z=15 cosmology curve table\nPlanck18 theoretical curve; not individual galaxy observations",
                 color="#f8fafc", fontsize=13.5, pad=18)
    path = OUTPUT_PNG / f"{VERSION}_Z0_Z15_STYLED_CURVE_TABLE.png"
    fig.tight_layout()
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    curve = build_curve()
    sample = sample_curve(curve)

    curve_csv = OUTPUT_CSV / f"{VERSION}_Z0_Z15_CURVE_DATA.csv"
    sample_csv = OUTPUT_CSV / f"{VERSION}_Z0_Z15_SAMPLED_TABLE.csv"
    curve.to_csv(curve_csv, index=False)
    sample.to_csv(sample_csv, index=False)

    distance_png = plot_distance_curve(curve)
    age_png = plot_age_curve(curve)
    slope_png = plot_distance_derivative(curve)
    table_png = styled_table(sample)

    ref = curve.iloc[(curve["redshift_z"] - Z_REFERENCE).abs().idxmin()]

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("z range", f"{Z_MIN:.1f} to {Z_MAX:.1f}"),
        ("Curve points", len(curve)),
        ("Cosmology", "Astropy Planck18"),
        ("Mode", "theoretical curve only; no individual galaxies"),
    ], ["Field", "Value"])
    print("")
    print("RESULTS")
    print_table([
        ("z=0 distance", f"{curve.iloc[0]['comoving_distance_gly']:.6f} Gly"),
        ("z=15 distance", f"{curve.iloc[-1]['comoving_distance_gly']:.6f} Gly"),
        ("z=15 age", f"{curve.iloc[-1]['universe_age_myr']:.3f} Myr"),
        ("z=14.44 reference distance", f"{ref['comoving_distance_gly']:.6f} Gly"),
        ("z=14.44 reference age", f"{ref['universe_age_myr']:.3f} Myr"),
        ("Interpretation", "smooth curve; slope decreases with redshift"),
    ], ["Metric", "Value"])
    print("")
    print("OUTPUT SUMMARY")
    print_table([
        ("png", str(distance_png)),
        ("png", str(age_png)),
        ("png", str(slope_png)),
        ("png", str(table_png)),
        ("csv", str(curve_csv)),
        ("csv", str(sample_csv)),
    ], ["Type", "Path"])
    print("")
    print("COMMENTS")
    print("This is a continuous cosmology curve, not a catalog of individual JWST galaxies.")
    print("The curve includes low redshift through z=15 so the flattening can be seen clearly.")
    print("Matplotlib only. No AI images.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
