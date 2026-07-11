# JWST_0022
# Audit: Plot visual consequences of Einstein equations. Matplotlib only. No AI images. No FITS/image downloads.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0022"
PROJECT = "EINSTEIN EQUATION VISUALIZATION DASHBOARD"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

C_KM_S = 299792.458
MPC_TO_GLY = 3.261563777e-3
H0_KM_S_MPC = 67.4
OMEGA_M = 0.315
OMEGA_L = 0.685
OMEGA_R = 9.2e-5
OMEGA_K = 1.0 - OMEGA_M - OMEGA_L - OMEGA_R


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def print_table(rows, heads):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(heads)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(heads)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(heads))))


def dark(fig, axes):
    import numpy as np
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    axes = np.array(axes).ravel()
    fig.patch.set_facecolor("#050712")
    for ax in axes:
        ax.set_facecolor("#050712")
        ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
        ax.tick_params(colors="#dbeafe", labelsize=8.5)
        ax.xaxis.label.set_color("#f8fafc")
        ax.yaxis.label.set_color("#f8fafc")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.2, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")
    return leg


def E_of_a(a):
    import numpy as np
    return np.sqrt(OMEGA_R/a**4 + OMEGA_M/a**3 + OMEGA_K/a**2 + OMEGA_L)


def E_of_z(z):
    import numpy as np
    zp1 = 1.0 + z
    return np.sqrt(OMEGA_R*zp1**4 + OMEGA_M*zp1**3 + OMEGA_K*zp1**2 + OMEGA_L)


def build_data():
    import numpy as np
    import pandas as pd

    # Schwarzschild exterior geometry in units of Schwarzschild radius r_s.
    x = np.linspace(1.01, 40.0, 1400)
    f = 1.0 - 1.0/x
    time_dilation = np.sqrt(f)
    tidal_norm = 1.0 / x**3
    schwarz = pd.DataFrame({
        "r_over_rs": x,
        "schwarzschild_factor_1_minus_rs_over_r": f,
        "clock_rate_far_observer_fraction": time_dilation,
        "tidal_curvature_normalized": tidal_norm,
    })

    # Friedmann equation terms from Einstein equations for a homogeneous universe.
    a = np.logspace(-4, np.log10(2.5), 1400)
    z = 1.0/a - 1.0
    rad = OMEGA_R/a**4
    mat = OMEGA_M/a**3
    curv = OMEGA_K/a**2
    lam = OMEGA_L + 0.0*a
    total = rad + mat + curv + lam
    friedmann = pd.DataFrame({
        "scale_factor_a": a,
        "redshift_z": z,
        "radiation_term": rad,
        "matter_term": mat,
        "curvature_term": curv,
        "lambda_term": lam,
        "H_over_H0_squared": total,
        "H_over_H0": np.sqrt(total),
    })

    # Horizons from the same expansion function.
    hubble_radius_gly = (C_KM_S / H0_KM_S_MPC) * MPC_TO_GLY
    z_grid = np.concatenate([[0.0], np.logspace(-6, 8, 160000)])
    particle_integrand = 1.0 / E_of_z(z_grid)
    particle_radius_gly = hubble_radius_gly * np.trapezoid(particle_integrand, z_grid)

    a_future = np.logspace(0, 6, 160000)
    event_integrand = 1.0 / (a_future**2 * E_of_a(a_future))
    event_radius_gly = hubble_radius_gly * np.trapezoid(event_integrand, a_future)

    horizons = pd.DataFrame({
        "quantity": ["Hubble radius", "Cosmic event horizon", "Particle horizon / observable radius"],
        "radius_Gly": [hubble_radius_gly, event_radius_gly, particle_radius_gly],
        "meaning": [
            "where recession speed equals c today",
            "light emitted today beyond this limit never reaches us",
            "present-day comoving radius of light/signals that could have reached us",
        ],
    })
    return schwarz, friedmann, horizons


def plot_dashboard(schwarz, friedmann, horizons):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.6))
    dark(fig, axes)
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.axis("off")
    ax0.set_title("Einstein field equation: not one curve, but a geometry-to-matter rule")
    eq = r"$G_{\mu\nu}+\Lambda g_{\mu\nu}=\frac{8\pi G}{c^4}T_{\mu\nu}$"
    ax0.text(0.5, 0.72, eq, color="#f8fafc", ha="center", va="center", fontsize=24)
    ax0.text(0.5, 0.50, "left side: spacetime curvature", color="#67e8f9", ha="center", fontsize=13)
    ax0.text(0.5, 0.38, "right side: mass-energy, pressure, radiation", color="#fbbf24", ha="center", fontsize=13)
    ax0.text(0.5, 0.22, "So we plot solutions: Schwarzschild geometry, Friedmann expansion, horizons.", color="#cbd5e1", ha="center", fontsize=11)

    ax1.plot(schwarz["r_over_rs"], schwarz["clock_rate_far_observer_fraction"], color="#38bdf8", linewidth=2.4, label="clock rate vs far observer")
    ax1.plot(schwarz["r_over_rs"], schwarz["schwarzschild_factor_1_minus_rs_over_r"], color="#f97316", linewidth=2.0, label=r"metric factor $1-r_s/r$")
    ax1.plot(schwarz["r_over_rs"], schwarz["tidal_curvature_normalized"], color="#f43f5e", linewidth=1.8, label=r"tidal scale $(r_s/r)^3$")
    ax1.axvline(1.0, color="#f8fafc", linewidth=1.2, linestyle="--", label="event horizon r = rs")
    ax1.set_xlim(1.0, 15.0)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"radius $r/r_s$")
    ax1.set_ylabel("normalized quantity")
    ax1.set_title("Schwarzschild solution: clocks slow and curvature rises near the horizon")
    legend(ax1, "lower right")

    a = friedmann["scale_factor_a"].to_numpy()
    ax2.loglog(a, friedmann["radiation_term"], color="#a78bfa", linewidth=1.8, label=r"radiation $\Omega_r/a^4$")
    ax2.loglog(a, friedmann["matter_term"], color="#38bdf8", linewidth=2.0, label=r"matter $\Omega_m/a^3$")
    ax2.loglog(a, friedmann["lambda_term"], color="#f97316", linewidth=2.2, label=r"dark energy $\Omega_\Lambda$")
    ax2.loglog(a, friedmann["H_over_H0_squared"], color="#f8fafc", linewidth=2.6, label=r"total $H^2/H_0^2$")
    ax2.axvline(1.0, color="#facc15", linewidth=1.2, linestyle="--", label="today")
    ax2.set_xlim(1e-4, 2.5)
    ax2.set_ylim(1e-1, 1e14)
    ax2.set_xlabel("scale factor a")
    ax2.set_ylabel(r"Friedmann terms in $H^2/H_0^2$")
    ax2.set_title("Friedmann equation: Einstein equations reduced to cosmic expansion")
    legend(ax2, "upper right")

    names = horizons["quantity"].to_list()
    vals = horizons["radius_Gly"].to_numpy()
    ypos = np.arange(len(names))
    ax3.barh(ypos, vals, color=["#38bdf8", "#f97316", "#f43f5e"], alpha=0.92)
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(names, color="#f8fafc")
    ax3.set_xlabel("radius today, billion light-years")
    ax3.set_title("Horizon scales computed from the same Friedmann expansion curve")
    for y, v in zip(ypos, vals):
        ax3.text(v + 0.8, y, f"{v:.2f} Gly", color="#f8fafc", va="center", fontsize=10)
    ax3.set_xlim(0, max(vals)*1.22)

    fig.suptitle("JWST_0022 — plotting Einstein's equations by plotting their solutions", color="#f8fafc", fontsize=16, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    path = PNG / f"{VERSION}_EINSTEIN_EQUATION_DASHBOARD.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_friedmann_distance_ladder(horizons):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    z = np.concatenate([[0.0], np.logspace(-3, 5, 1600)])
    hubble_radius_gly = float(horizons.loc[horizons["quantity"] == "Hubble radius", "radius_Gly"].iloc[0])
    integrand = 1.0 / E_of_z(z)
    cumulative = np.zeros_like(z)
    cumulative[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z))
    comoving_gly = hubble_radius_gly * cumulative
    lookback_like = comoving_gly * 0.0
    df = pd.DataFrame({"redshift_z": z, "comoving_distance_Gly": comoving_gly, "E_z": E_of_z(z)})

    fig, ax = plt.subplots(figsize=(13.8, 7.7))
    dark(fig, ax)
    ax.plot(z, comoving_gly, color="#67e8f9", linewidth=2.8, label=r"$D_C(z)=c/H_0\int_0^z dz/E(z)$")
    for _, row in horizons.iterrows():
        ax.axhline(row["radius_Gly"], linewidth=1.4, linestyle="--", alpha=0.9, label=f"{row['quantity']} = {row['radius_Gly']:.2f} Gly")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e5)
    ax.set_ylim(0, float(horizons["radius_Gly"].max()) * 1.08)
    ax.set_xlabel("redshift z")
    ax.set_ylabel("model-inferred comoving distance, Gly")
    ax.set_title("Einstein + FLRW + Friedmann gives the redshift-to-distance transfer function")
    legend(ax, "lower right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FRIEDMANN_REDSHIFT_DISTANCE_HORIZONS.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    csv = CSV / f"{VERSION}_FRIEDMANN_DISTANCE_CURVE.csv"
    df.to_csv(csv, index=False)
    return path, csv


def main():
    setup()
    schwarz, friedmann, horizons = build_data()
    schwarz_csv = CSV / f"{VERSION}_SCHWARZSCHILD_CURVES.csv"
    friedmann_csv = CSV / f"{VERSION}_FRIEDMANN_TERMS.csv"
    horizons_csv = CSV / f"{VERSION}_HORIZON_SCALES.csv"
    schwarz.to_csv(schwarz_csv, index=False)
    friedmann.to_csv(friedmann_csv, index=False)
    horizons.to_csv(horizons_csv, index=False)
    dash_png = plot_dashboard(schwarz, friedmann, horizons)
    dist_png, dist_csv = plot_friedmann_distance_ladder(horizons)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Equation plotted", "G_mu_nu + Lambda g_mu_nu = 8piG/c^4 T_mu_nu"),
        ("Visualization rule", "plot solutions and scalar reductions, not the raw tensor equation"),
        ("H0 km/s/Mpc", f"{H0_KM_S_MPC:.3f}"),
        ("Omega_m", f"{OMEGA_M:.6f}"),
        ("Omega_lambda", f"{OMEGA_L:.6f}"),
    ], ["Field", "Value"])
    print("\nHORIZON SCALES")
    print_table([(r["quantity"], f"{r['radius_Gly']:.3f}", r["meaning"]) for _, r in horizons.iterrows()], ["Boundary", "Gly", "Meaning"])
    print("\nOUTPUTS")
    print_table([
        ("png", str(dash_png)),
        ("png", str(dist_png)),
        ("csv", str(schwarz_csv)),
        ("csv", str(friedmann_csv)),
        ("csv", str(horizons_csv)),
        ("csv", str(dist_csv)),
    ], ["Type", "Path"])
    print("\nCOMMENTS")
    print("Matplotlib only. No AI images.")
    print("Raw Einstein field equations are tensor equations; this plots interpretable solutions/reductions.")
    print("The horizon values are model outputs from the chosen Friedmann parameters.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
