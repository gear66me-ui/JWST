# JWST_0024
# Audit: Clarify Schwarzschild radius axis from center to exterior. Matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0024"
PROJECT = "SCHWARZSCHILD ZERO-AXIS CLARIFICATION"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"


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


def dark(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def legend(ax):
    leg = ax.legend(loc="lower right", fontsize=8.0, facecolor="#020617", edgecolor="#475569")
    for text in leg.get_texts():
        text.set_color("#f8fafc")


def build_curves():
    import numpy as np
    import pandas as pd

    r = np.linspace(0.0, 15.0, 2500)
    metric = np.full_like(r, np.nan, dtype=float)
    clock = np.full_like(r, np.nan, dtype=float)
    curvature = np.full_like(r, np.nan, dtype=float)
    outside = r > 1.0
    metric[outside] = 1.0 - 1.0 / r[outside]
    clock[outside] = np.sqrt(metric[outside])
    curvature[outside] = 1.0 / r[outside]**3

    region = []
    for x in r:
        if x == 0:
            region.append("center singularity")
        elif x < 1:
            region.append("inside event horizon / no stationary exterior observer")
        elif abs(x - 1) < 1e-9:
            region.append("event horizon")
        else:
            region.append("outside event horizon")

    df = pd.DataFrame({
        "r_over_rs": r,
        "region": region,
        "metric_factor_1_minus_rs_over_r_outside_only": metric,
        "clock_rate_fraction_outside_only": clock,
        "tidal_curvature_scale_outside_only": curvature,
    })
    return df


def plot(df):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(14.8, 7.8))
    dark(fig, ax)

    r = df["r_over_rs"].to_numpy()
    ax.axvspan(0.0, 1.0, color="#7f1d1d", alpha=0.35, label="inside horizon: 0 < r/rs < 1")
    ax.axvline(0.0, color="#f8fafc", linewidth=1.4, linestyle=":", label="center r/rs = 0")
    ax.axvline(1.0, color="#facc15", linewidth=2.0, linestyle="--", label="event horizon r/rs = 1")

    ax.plot(r, df["clock_rate_fraction_outside_only"], color="#38bdf8", linewidth=2.5, label="stationary clock rate outside horizon")
    ax.plot(r, df["metric_factor_1_minus_rs_over_r_outside_only"], color="#f97316", linewidth=2.2, label=r"Schwarzschild metric factor $1-r_s/r$")
    ax.plot(r, df["tidal_curvature_scale_outside_only"], color="#f43f5e", linewidth=2.0, label=r"tidal curvature scale $(r_s/r)^3$")

    ax.annotate("center\nsingularity\nr/rs = 0", xy=(0.0, 0.72), xytext=(1.55, 0.82),
                color="#f8fafc", fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#f8fafc", "lw": 1.0})
    ax.annotate("event horizon\nr/rs = 1", xy=(1.0, 0.05), xytext=(2.45, 0.18),
                color="#facc15", fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#facc15", "lw": 1.2})
    ax.annotate("plotted physics curves are exterior-only\nbecause a stationary observer cannot hover inside", xy=(0.55, 0.45), xytext=(4.8, 0.46),
                color="#fecaca", fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#fecaca", "lw": 1.0})

    ax.set_xlim(0, 15)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"normalized Schwarzschild radius, $r/r_s$")
    ax.set_ylabel("normalized exterior quantity")
    ax.set_title("JWST_0024 — Schwarzschild map with center, horizon, and exterior all on one axis")
    legend(ax)
    fig.tight_layout()
    path = PNG / f"{VERSION}_SCHWARZSCHILD_ZERO_AXIS_MAP.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    df = build_curves()
    csv = CSV / f"{VERSION}_SCHWARZSCHILD_ZERO_AXIS_CURVES.csv"
    df.to_csv(csv, index=False)
    png = plot(df)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("x-axis", "r / r_s"),
        ("Center", "r/r_s = 0"),
        ("Black-hole event horizon", "r/r_s = 1"),
        ("Exterior region", "r/r_s > 1"),
        ("Cosmic event horizon", "not plotted here; different cosmology problem"),
    ], ["Field", "Value"])
    print("\nOUTPUTS")
    print_table([
        ("png", str(png)),
        ("csv", str(csv)),
    ], ["Type", "Path"])
    print("\nCOMMENTS")
    print("This plot shifts the axis to show r/r_s = 0, the center, and r/r_s = 1, the event horizon.")
    print("The curves are only meaningful outside the horizon for stationary Schwarzschild observers.")
    print("No AI images. Matplotlib only.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
