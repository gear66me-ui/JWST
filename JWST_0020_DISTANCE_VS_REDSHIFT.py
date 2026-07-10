# JWST_0020
# Audit: Distance versus redshift clarification plot. Matplotlib only. No AI images. No FITS/image downloads.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0020"
PROJECT = "JWST DISTANCE VERSUS REDSHIFT CLARIFICATION"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
HORIZON_GLY = 46.5


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def print_table(rows, heads):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(heads)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(heads)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(heads))))


def build_curve():
    import numpy as np
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo

    z = np.r_[
        np.linspace(0.0, 2.0, 201),
        np.linspace(2.05, 10.0, 200),
        np.linspace(10.1, 50.0, 260),
    ]
    dc_gly = cosmo.comoving_distance(z).to("Glyr").value
    lookback_gyr = cosmo.lookback_time(z).value
    age_gyr = cosmo.age(z).value
    ddc_dz = np.gradient(dc_gly, z)

    # A straight-line expectation anchored to z=1. This is deliberately not correct at high z.
    d_at_1 = float(cosmo.comoving_distance(1.0).to("Glyr").value)
    linear_z1_anchor_gly = d_at_1 * z

    return pd.DataFrame({
        "redshift_z": z,
        "comoving_distance_Gly": dc_gly,
        "linear_z1_anchor_Gly": linear_z1_anchor_gly,
        "extra_distance_per_redshift_Gly_per_z": ddc_dz,
        "lookback_time_Gyr": lookback_gyr,
        "universe_age_Gyr": age_gyr,
        "universe_age_Myr": age_gyr * 1000.0,
    })


def dark(fig, axes):
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    fig.patch.set_facecolor("#050712")
    for ax in axes:
        ax.set_facecolor("#050712")
        ax.grid(True, color="#334155", linewidth=0.55, alpha=0.65)
        ax.tick_params(colors="#dbeafe", labelsize=8.8)
        ax.xaxis.label.set_color("#f8fafc")
        ax.yaxis.label.set_color("#f8fafc")
        ax.title.set_color("#f8fafc")
        for spine in ax.spines.values():
            spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.5, facecolor="#020617", edgecolor="#475569")
    for text in leg.get_texts():
        text.set_color("#f8fafc")


def plot_distance_vs_redshift(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.6, 7.7))
    dark(fig, ax)

    ax.plot(df.redshift_z, df.comoving_distance_Gly, linewidth=2.8, color="#67e8f9", label="Planck18 comoving distance")
    ax.plot(df.redshift_z, df.linear_z1_anchor_Gly, linewidth=1.7, color="#fb923c", linestyle="--", label="straight-line expectation anchored at z=1")
    ax.axhline(HORIZON_GLY, color="#f43f5e", linewidth=1.6, linestyle=":", label="46.5 Gly observable-radius context")

    anchors = [0.5, 1, 2, 5, 10, 15, 20, 50]
    for z0 in anchors:
        d0 = np.interp(z0, df.redshift_z, df.comoving_distance_Gly)
        ax.scatter(z0, d0, s=54, color="#facc15", edgecolor="#f8fafc", linewidth=0.7, zorder=5)
        yoff = 1.2 if z0 < 12 else -1.6
        ax.text(z0, d0 + yoff, f"z={z0:g}\n{d0:.1f} Gly", color="#f8fafc", fontsize=8.1, ha="center")

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance, Gly")
    ax.set_title("Distance versus redshift is not a straight line\nredshift keeps increasing while comoving distance approaches the horizon scale")
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMOVING_DISTANCE_VS_REDSHIFT.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_incremental_distance(df):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13.6, 7.3))
    dark(fig, ax)

    ax.plot(df.redshift_z, df.extra_distance_per_redshift_Gly_per_z, linewidth=2.8, color="#a7f3d0", label="extra comoving distance per +1 redshift")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, max(df.extra_distance_per_redshift_Gly_per_z) * 1.06)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("dD / dz, Gly per redshift")
    ax.set_title("Why equal-redshift shells thin out at high z\neach extra unit of redshift adds less comoving distance")

    for z0 in [1, 2, 5, 10, 20, 50]:
        y0 = np.interp(z0, df.redshift_z, df.extra_distance_per_redshift_Gly_per_z)
        ax.scatter(z0, y0, s=52, color="#fb923c", edgecolor="#f8fafc", linewidth=0.7, zorder=5)
        ax.text(z0, y0 + 0.25, f"z={z0:g}\n{y0:.2f}", color="#f8fafc", fontsize=8.1, ha="center")

    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_INCREMENTAL_DISTANCE_PER_REDSHIFT.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_anchor_table(anchor_df):
    import matplotlib.pyplot as plt

    rows = []
    for _, r in anchor_df.iterrows():
        rows.append([
            f"{r.redshift_z:.2f}",
            f"{r.comoving_distance_Gly:.3f}",
            f"{r.extra_distance_per_redshift_Gly_per_z:.3f}",
            f"{r.lookback_time_Gyr:.3f}",
            f"{r.universe_age_Myr:.1f}",
        ])

    fig, ax = plt.subplots(figsize=(12.7, 4.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    labels = ["z", "Comoving Gly", "dD/dz Gly per z", "Lookback Gyr", "Age Myr"]
    tab = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.6)
    tab.scale(1.0, 1.55)
    for (rr, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.65)
        if rr == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            zval = float(rows[rr - 1][0])
            cell.set_facecolor("#082f49" if zval < 3 else "#172554" if zval < 15 else "#3b1114")
            cell.get_text().set_color("#e5e7eb")
    ax.set_title("JWST_0020 anchor table — redshift is not a linear distance ruler", color="#f8fafc", fontsize=13, pad=14)
    fig.tight_layout()
    path = PNG / f"{VERSION}_ANCHOR_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    df = build_curve()
    curve_csv = CSV / f"{VERSION}_DISTANCE_REDSHIFT_CURVE.csv"
    df.to_csv(curve_csv, index=False)

    anchors = [0.5, 1, 2, 5, 10, 15, 20, 50]
    anchor_df = df.iloc[[abs(df.redshift_z - z).idxmin() for z in anchors]].copy()
    anchor_csv = CSV / f"{VERSION}_ANCHOR_VALUES.csv"
    anchor_df.to_csv(anchor_csv, index=False)

    plot1 = plot_distance_vs_redshift(df)
    plot2 = plot_incremental_distance(df)
    table_png = styled_anchor_table(anchor_df)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Cosmology", "Astropy Planck18"),
        ("Main result", "distance vs redshift bends and flattens"),
        ("Engineering reason", "redshift is not a linear distance ruler"),
        ("Horizon context", f"{HORIZON_GLY:.1f} Gly"),
    ], ["Field", "Value"])
    print("\nANCHORS")
    print_table([
        (f"z={r.redshift_z:.2f}", f"{r.comoving_distance_Gly:.3f} Gly", f"dD/dz={r.extra_distance_per_redshift_Gly_per_z:.3f} Gly/z")
        for _, r in anchor_df.iterrows()
    ], ["z", "distance", "extra distance per redshift"])
    print("\nOUTPUTS")
    print_table([
        ("png", str(plot1)),
        ("png", str(plot2)),
        ("png", str(table_png)),
        ("csv", str(curve_csv)),
        ("csv", str(anchor_csv)),
    ], ["Type", "Path"])
    print("\nCOMMENTS")
    print("No AI images. Matplotlib only.")
    print("The straight-line curve is intentionally wrong at high z; it shows the intuition that fails.")
    print("The incremental-distance plot explains why equal-redshift shell volumes can peak and decline.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
