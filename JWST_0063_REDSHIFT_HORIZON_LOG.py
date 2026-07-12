#!/usr/bin/env python3
"""
JWST_0063_REDSHIFT_HORIZON_LOG.py

Planck18 comoving-distance asymptote audit from z=1e-3 to z=1e7.

The script separates:
- the observed/reported MoM-z14 marker at z_spec = 14.44,
- theoretical Planck18 cosmology curve points,
- the present-day comoving particle-horizon limit.

No image products are downloaded. All plots are generated numerically with
Astropy, NumPy, SciPy, Pandas, and Matplotlib.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "astropy": "astropy",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.integrate import cumulative_trapezoid, quad
from astropy.cosmology import Planck18
import astropy.units as u

VERSION = "JWST_0063"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)

Z_MIN = 1.0e-3
Z_MAX = 1.0e7
MOM_Z = 14.44
CMB_Z = 1089.92

# At very high z, E(z) approaches sqrt(Omega_r) * (1+z)^2.
# Estimate the radiation coefficient directly from Planck18's own efunc.
Z_RADIATION_REFERENCE = 1.0e8
OMEGA_R_EFFECTIVE = (
    Planck18.efunc(Z_RADIATION_REFERENCE) / (1.0 + Z_RADIATION_REFERENCE) ** 2
) ** 2
SQRT_OMEGA_R = math.sqrt(float(OMEGA_R_EFFECTIVE))
HUBBLE_DISTANCE_GLY = Planck18.hubble_distance.to_value(u.Glyr)
HUBBLE_TIME_GYR = Planck18.hubble_time.to_value(u.Gyr)


def e_of_a(a: float) -> float:
    """Dimensionless expansion rate E(a), stable as a approaches zero."""
    if a <= 1.0e-10:
        return SQRT_OMEGA_R / (a * a)
    return float(Planck18.efunc(1.0 / a - 1.0))


def distance_integrand_a(a: float) -> float:
    """Integrand for present-day line-of-sight comoving distance."""
    if a <= 1.0e-10:
        return 1.0 / SQRT_OMEGA_R
    return 1.0 / (a * a * e_of_a(a))


def age_integrand_a(a: float) -> float:
    """Integrand for cosmic age t(a)."""
    if a <= 0.0:
        return 0.0
    if a <= 1.0e-10:
        return a / SQRT_OMEGA_R
    return 1.0 / (a * e_of_a(a))


def comoving_horizon_gly() -> float:
    value, _ = quad(
        distance_integrand_a,
        0.0,
        1.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=1000,
    )
    return HUBBLE_DISTANCE_GLY * value


def exact_distance_gly(z: float) -> float:
    a_emit = 1.0 / (1.0 + z)
    value, _ = quad(
        distance_integrand_a,
        a_emit,
        1.0,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
        limit=1000,
    )
    return HUBBLE_DISTANCE_GLY * value


def exact_age_gyr(z: float) -> float:
    a_emit = 1.0 / (1.0 + z)
    value, _ = quad(
        age_integrand_a,
        0.0,
        a_emit,
        epsabs=1.0e-14,
        epsrel=1.0e-11,
        limit=1000,
    )
    return HUBBLE_TIME_GYR * value


def compute_curves() -> dict[str, np.ndarray]:
    # Uniform spacing in y = ln(1+z) gives dense sampling across eight decades.
    y = np.linspace(0.0, np.log1p(Z_MAX), 24000)
    z_full = np.expm1(y)
    expansion = np.asarray(Planck18.efunc(z_full), dtype=float)

    distance_integrand_y = np.exp(y) / expansion
    distance_dimensionless = cumulative_trapezoid(
        distance_integrand_y, y, initial=0.0
    )
    distance_gly_full = HUBBLE_DISTANCE_GLY * distance_dimensionless

    age_integrand_y = 1.0 / expansion
    age_cumulative = cumulative_trapezoid(age_integrand_y, y, initial=0.0)
    high_z_age_tail = 1.0 / (
        2.0 * SQRT_OMEGA_R * (1.0 + Z_MAX) ** 2
    )
    age_gyr_full = HUBBLE_TIME_GYR * (
        age_cumulative[-1] - age_cumulative + high_z_age_tail
    )

    mask = z_full >= Z_MIN
    return {
        "z": z_full[mask],
        "distance_gly": distance_gly_full[mask],
        "age_gyr": age_gyr_full[mask],
    }


def make_table(horizon_gly: float) -> pd.DataFrame:
    sample_rows = [
        (1.0, "theoretical curve point"),
        (5.0, "theoretical curve point"),
        (10.0, "theoretical curve point"),
        (MOM_Z, "MoM-z14; reported spectroscopic redshift"),
        (20.0, "theoretical curve point"),
        (50.0, "theoretical curve point"),
        (100.0, "theoretical curve point"),
        (CMB_Z, "CMB/recombination context; not a galaxy"),
        (1.0e4, "theoretical early-Universe point"),
        (1.0e5, "theoretical early-Universe point"),
        (1.0e6, "theoretical early-Universe point"),
        (1.0e7, "theoretical early-Universe point"),
    ]

    records = []
    for z, status in sample_rows:
        distance = exact_distance_gly(z)
        age_gyr = exact_age_gyr(z)
        deficit = horizon_gly - distance
        records.append({
            "redshift_z": z,
            "scale_factor_a": 1.0 / (1.0 + z),
            "comoving_distance_Gly": distance,
            "horizon_deficit_Gly": deficit,
            "fraction_of_horizon_percent": 100.0 * distance / horizon_gly,
            "universe_age_Myr": age_gyr * 1000.0,
            "universe_age_days": age_gyr * 1.0e9 * 365.25,
            "status": status,
        })
    return pd.DataFrame(records)


def make_dashboard(curves: dict[str, np.ndarray], table: pd.DataFrame,
                   horizon_gly: float) -> Path:
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0])
    ax_distance = fig.add_subplot(grid[0, :])
    ax_deficit = fig.add_subplot(grid[1, 0])
    ax_age = fig.add_subplot(grid[1, 1])

    z = curves["z"]
    distance = curves["distance_gly"]
    age_gyr = curves["age_gyr"]
    deficit = np.maximum(horizon_gly - distance, np.finfo(float).tiny)

    ax_distance.plot(z, distance, linewidth=2.4,
                     label="Planck18 comoving distance")
    ax_distance.axhline(horizon_gly, linestyle="--", linewidth=1.5,
                        label=f"Particle-horizon asymptote = {horizon_gly:.6f} Gly")
    ax_distance.set_xscale("log")
    ax_distance.set_xlim(Z_MIN, Z_MAX)
    ax_distance.set_ylim(0.0, horizon_gly * 1.055)
    ax_distance.set_xlabel("Redshift z — logarithmic scale")
    ax_distance.set_ylabel("Present-day comoving distance [Gly]")
    ax_distance.set_title(
        "Planck18 comoving distance approaches a finite particle horizon\n"
        "Redshift keeps increasing; comoving distance does not",
        fontsize=17,
    )
    ax_distance.grid(alpha=0.20, which="both")

    marker_specs = [
        (MOM_Z, "MoM-z14\nz=14.44", (12, -38), "left"),
        (CMB_Z, "recombination\nz≈1089.92", (-92, -48), "left"),
        (1.0e7, "theoretical\nz=10,000,000", (-118, 16), "left"),
    ]
    for z_marker, label, offset, align in marker_specs:
        d_marker = exact_distance_gly(z_marker)
        ax_distance.scatter([z_marker], [d_marker], s=70, zorder=5)
        ax_distance.annotate(
            f"{label}\n{d_marker:.6f} Gly",
            xy=(z_marker, d_marker),
            xytext=offset,
            textcoords="offset points",
            ha=align,
            arrowprops=dict(arrowstyle="-", linewidth=0.9),
            fontsize=9,
        )
    ax_distance.legend(loc="lower right")

    ax_deficit.plot(z, deficit, linewidth=2.1)
    ax_deficit.set_xscale("log")
    ax_deficit.set_yscale("log")
    ax_deficit.set_xlim(Z_MIN, Z_MAX)
    ax_deficit.set_xlabel("Redshift z")
    ax_deficit.set_ylabel("Remaining distance to asymptote [Gly]")
    ax_deficit.set_title(
        "How much comoving distance remains before the horizon limit",
        fontsize=13,
    )
    ax_deficit.grid(alpha=0.20, which="both")

    ax_age.plot(z, age_gyr * 1000.0, linewidth=2.1)
    ax_age.set_xscale("log")
    ax_age.set_yscale("log")
    ax_age.set_xlim(Z_MIN, Z_MAX)
    ax_age.set_xlabel("Redshift z")
    ax_age.set_ylabel("Age of the Universe [Myr]")
    ax_age.set_title(
        "Cosmic age tends toward zero as redshift tends toward infinity",
        fontsize=13,
    )
    ax_age.grid(alpha=0.20, which="both")

    mom = table[np.isclose(table["redshift_z"], MOM_Z)].iloc[0]
    zmax = table[np.isclose(table["redshift_z"], Z_MAX)].iloc[0]
    ax_age.scatter([MOM_Z], [mom["universe_age_Myr"]], s=65, zorder=5)
    ax_age.annotate(
        f"MoM-z14\n{mom['universe_age_Myr']:.3f} Myr",
        xy=(MOM_Z, mom["universe_age_Myr"]),
        xytext=(18, 18), textcoords="offset points",
        arrowprops=dict(arrowstyle="-", linewidth=0.9), fontsize=9,
    )
    ax_age.scatter([Z_MAX], [zmax["universe_age_Myr"]], s=65, zorder=5)
    ax_age.annotate(
        f"z=10⁷\n{zmax['universe_age_days']:.3f} days",
        xy=(Z_MAX, zmax["universe_age_Myr"]),
        xytext=(-105, 20), textcoords="offset points",
        arrowprops=dict(arrowstyle="-", linewidth=0.9), fontsize=9,
    )

    fig.suptitle(
        "Redshift-to-horizon asymptote audit — Planck18 ΛCDM",
        fontsize=20,
    )
    output = PNG_DIR / f"{VERSION}_REDSHIFT_HORIZON_LOG_DASHBOARD.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def make_styled_table(table: pd.DataFrame, horizon_gly: float) -> Path:
    shown = table.copy()
    shown["z"] = shown["redshift_z"].map(lambda value: f"{value:,.2f}")
    shown["Distance [Gly]"] = shown["comoving_distance_Gly"].map(lambda value: f"{value:.6f}")
    shown["To horizon [Gly]"] = shown["horizon_deficit_Gly"].map(lambda value: f"{value:.6f}")
    shown["Horizon [%]"] = shown["fraction_of_horizon_percent"].map(lambda value: f"{value:.6f}")
    def format_age(row: pd.Series) -> str:
        age_years = row["universe_age_Myr"] * 1.0e6
        if age_years >= 1.0e9:
            return f"{age_years / 1.0e9:.6f} Gyr"
        if age_years >= 1.0e6:
            return f"{age_years / 1.0e6:.6f} Myr"
        if age_years >= 1.0e3:
            return f"{age_years / 1.0e3:.6f} kyr"
        if age_years >= 1.0:
            return f"{age_years:.6f} yr"
        if row["universe_age_days"] >= 1.0:
            return f"{row['universe_age_days']:.6f} days"
        return f"{row['universe_age_days'] * 86400.0:.6f} s"

    shown["Age"] = shown.apply(format_age, axis=1)
    shown = shown[["z", "Distance [Gly]", "To horizon [Gly]", "Horizon [%]", "Age", "status"]]
    shown.columns = ["z", "Distance [Gly]", "Remaining [Gly]", "Of horizon [%]", "Universe age", "Status"]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 8.2), constrained_layout=True)
    ax.axis("off")
    table_artist = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.09, 0.13, 0.13, 0.13, 0.15, 0.37],
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8.5)
    table_artist.scale(1.0, 1.55)

    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#536879")
        cell.set_linewidth(0.65)
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")

    ax.set_title(
        f"Planck18 redshift audit — particle-horizon asymptote {horizon_gly:.6f} Gly\n"
        "Observed/reported galaxy data are explicitly separated from theoretical curve points",
        fontsize=16,
        pad=18,
    )
    output = PNG_DIR / f"{VERSION}_REDSHIFT_HORIZON_TABLE.png"
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def print_summary(table: pd.DataFrame, horizon_gly: float,
                  dashboard: Path, table_png: Path, csv_path: Path) -> None:
    mom = table[np.isclose(table["redshift_z"], MOM_Z)].iloc[0]
    z20 = table[np.isclose(table["redshift_z"], 20.0)].iloc[0]
    zmax = table[np.isclose(table["redshift_z"], Z_MAX)].iloc[0]

    print(f"CODE OUTPUT: {VERSION}")
    print("MODEL          Astropy Planck18 flat Lambda-CDM with radiation and neutrinos")
    print(f"HORIZON        {horizon_gly:12.6f} Gly  theoretical particle-horizon limit")
    print(f"MOM-z14        z={MOM_Z:8.2f}  d={mom['comoving_distance_Gly']:10.6f} Gly  age={mom['universe_age_Myr']:10.3f} Myr")
    print(f"z=20           d={z20['comoving_distance_Gly']:10.6f} Gly  horizon={z20['fraction_of_horizon_percent']:10.6f}%")
    print(f"z=10,000,000   d={zmax['comoving_distance_Gly']:10.6f} Gly  remaining={zmax['horizon_deficit_Gly']:.9f} Gly")
    print(f"                age={zmax['universe_age_days']:.6f} days  theoretical, not a galaxy epoch")
    print(f"PLOT PNG       {dashboard}")
    print(f"TABLE PNG      {table_png}")
    print(f"CSV            {csv_path}")
    print(f"Timestamp      {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


def main() -> None:
    horizon_gly = comoving_horizon_gly()
    curves = compute_curves()
    table = make_table(horizon_gly)

    csv_path = CSV_DIR / f"{VERSION}_REDSHIFT_HORIZON_AUDIT.csv"
    table.to_csv(csv_path, index=False)

    dashboard = make_dashboard(curves, table, horizon_gly)
    table_png = make_styled_table(table, horizon_gly)
    print_summary(table, horizon_gly, dashboard, table_png, csv_path)


if __name__ == "__main__":
    main()
