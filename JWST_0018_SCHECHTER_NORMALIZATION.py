# JWST_0018
# Audit: Schechter-style detection normalization toy model. Matplotlib only. No FITS/image downloads.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib

VERSION = "JWST_0018"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
HORIZON_GLY = 46.5
ALPHA = -1.35
L_FLOOR = 1.0e-4
L_CEIL = 1.0e3
LIM_AT_Z1 = 1.0e-3


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


def table(rows, heads):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(heads)]
    print(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(heads)))
    print("-" * (sum(w) + 3 * (len(w)-1)))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(w[i]) for i in range(len(heads))))


def schechter_fraction(z, dl_gly):
    import numpy as np
    x = np.logspace(np.log10(L_FLOOR), np.log10(L_CEIL), 12000)
    phi = x**ALPHA * np.exp(-x)
    total = np.trapz(phi, x)
    dl1 = float(np.interp(1.0, z, dl_gly))
    xmin = LIM_AT_Z1 * (np.maximum(dl_gly, 1e-12) / dl1)**2 * ((1+z)/2.0)**-1.0
    xmin = np.clip(xmin, L_FLOOR, L_CEIL)
    frac = []
    for cut in xmin:
        m = x >= cut
        frac.append(np.trapz(phi[m], x[m]) / total if m.any() else 0.0)
    return np.array(frac), xmin


def build():
    import numpy as np
    import pandas as pd
    import astropy.units as u
    from astropy.cosmology import Planck18 as cosmo

    z = np.r_[np.linspace(0.001, 2, 180), np.linspace(2.05, 15, 260), np.linspace(15.1, 50, 220)]
    dc = cosmo.comoving_distance(z).to("Glyr").value
    dl = cosmo.luminosity_distance(z).to("Glyr").value
    age = cosmo.age(z).value * 1000
    dvol = cosmo.differential_comoving_volume(z).to(u.Gpc**3 / u.sr).value * 4 * np.pi
    shell = dvol / dvol.max()
    r3 = np.clip((dc / HORIZON_GLY)**3, 0, 1)

    sch, xmin = schechter_fraction(z, dl)
    z_conf = 1 / (1 + np.exp((z - 14.0) / 2.4))
    early_build = 1 / (1 + np.exp((z - 18.0) / 2.8))
    surface = np.clip(((1 + z) / 2)**-0.35, 0, 1)
    detect = np.clip(sch * z_conf * early_build * surface, 0, 1)

    raw = shell * detect
    raw_norm = raw / raw.max()
    volcorr = np.divide(raw, shell, out=np.zeros_like(raw), where=shell > 0)
    volcorr_norm = volcorr / volcorr.max()
    dz = np.gradient(z)
    cum_detect = np.cumsum(raw * dz)
    cum_detect = cum_detect / cum_detect[-1]

    return pd.DataFrame({
        "redshift_z": z,
        "comoving_distance_Gly": dc,
        "luminosity_distance_Gly": dl,
        "universe_age_Myr": age,
        "shell_volume_Gpc3_per_z_all_sky": dvol,
        "shell_volume_norm": shell,
        "cumulative_R3_volume_fraction_to_46p5Gly": r3,
        "Lmin_over_Lstar_toy": xmin,
        "schechter_visible_fraction": sch,
        "toy_detection_fraction": detect,
        "raw_detected_proxy_norm": raw_norm,
        "volume_corrected_proxy_norm": volcorr_norm,
        "cumulative_detected_proxy": cum_detect,
    })


def dark(fig, axes):
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    fig.patch.set_facecolor("#050712")
    for ax in axes:
        ax.set_facecolor("#050712")
        ax.grid(True, color="#334155", lw=0.55, alpha=0.65)
        ax.tick_params(colors="#dbeafe", labelsize=8.5)
        ax.xaxis.label.set_color("#f8fafc")
        ax.yaxis.label.set_color("#f8fafc")
        ax.title.set_color("#f8fafc")
        for s in ax.spines.values():
            s.set_color("#94a3b8")


def leg(ax, loc="best"):
    lg = ax.legend(loc=loc, fontsize=8, facecolor="#020617", edgecolor="#475569")
    for t in lg.get_texts():
        t.set_color("#f8fafc")


def dashboard(df):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15.5, 10.2))
    dark(fig, axs)
    ax = axs.ravel()

    ax[0].plot(df.redshift_z, df.shell_volume_norm, lw=2.2, color="#38bdf8", label="shell volume")
    ax[0].plot(df.redshift_z, df.raw_detected_proxy_norm, lw=2.2, color="#fb923c", label="raw detected proxy")
    ax[0].plot(df.redshift_z, df.volume_corrected_proxy_norm, lw=2.0, color="#a7f3d0", label="volume-corrected proxy")
    ax[0].set(xlim=(0, 20), ylim=(0, 1.06), xlabel="redshift z", ylabel="normalized value",
              title="Raw counts = shell volume × detectability")
    leg(ax[0], "upper right")

    ax[1].plot(df.redshift_z, df.schechter_visible_fraction, lw=2.2, color="#facc15", label="Schechter visible fraction")
    ax[1].plot(df.redshift_z, df.toy_detection_fraction, lw=2.2, color="#fb7185", label="toy detection fraction")
    ax[1].set(xlim=(0, 25), ylim=(0, 1.06), xlabel="redshift z", ylabel="fraction",
              title="Toy selection function, not an official JWST completeness curve")
    leg(ax[1], "upper right")

    ax[2].plot(df.redshift_z, df.comoving_distance_Gly, lw=2.4, color="#67e8f9", label="comoving distance")
    ax[2].axhline(HORIZON_GLY, color="#fb923c", ls="--", lw=1.8, label="46.5 Gly horizon context")
    ax[2].set(xlim=(0, 50), ylim=(0, HORIZON_GLY*1.05), xlabel="redshift z", ylabel="Gly",
              title="46.5 Gly is horizon scale, not a galaxy location")
    leg(ax[2], "lower right")

    ax[3].plot(df.redshift_z, df.cumulative_R3_volume_fraction_to_46p5Gly, lw=2.4, color="#38bdf8", label="R³ volume fraction")
    ax[3].plot(df.redshift_z, df.cumulative_detected_proxy, lw=2.4, color="#fb923c", label="cumulative detected proxy")
    ax[3].set(xlim=(0, 50), ylim=(0, 1.03), xlabel="redshift z", ylabel="cumulative fraction",
              title="Cumulative curves rise and flatten by definition")
    leg(ax[3], "lower right")

    fig.suptitle("JWST_0018 — Schechter-style normalization vs detection toy dashboard", color="#f8fafc", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = PNG / f"{VERSION}_SCHECHTER_NORMALIZATION_DASHBOARD.png"
    fig.savefig(p, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return p


def horizon_plot(df):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    dark(fig, ax)
    ax.plot(df.comoving_distance_Gly, df.cumulative_R3_volume_fraction_to_46p5Gly, lw=2.8, color="#38bdf8")
    ax.axvline(HORIZON_GLY, color="#fb923c", ls="--", lw=1.8, label="46.5 Gly observable-radius context")
    for z in [1, 2, 5, 10, 15, 20, 50]:
        d = np.interp(z, df.redshift_z, df.comoving_distance_Gly)
        f = np.interp(z, df.redshift_z, df.cumulative_R3_volume_fraction_to_46p5Gly)
        ax.scatter(d, f, s=58, color="#fb923c", edgecolor="#f8fafc", lw=0.8, zorder=5)
        ax.text(d + 0.45, f, f"z={z}", color="#f8fafc", fontsize=8.5, va="center")
    ax.set(xlim=(0, HORIZON_GLY*1.04), ylim=(0, 1.03), xlabel="comoving distance, Gly",
           ylabel="cumulative volume fraction", title="R³ growth to the observable-radius scale")
    leg(ax, "lower right")
    fig.tight_layout()
    p = PNG / f"{VERSION}_HORIZON_R3_VOLUME_GROWTH.png"
    fig.savefig(p, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return p


def main():
    setup()
    df = build()
    curve_csv = CSV / f"{VERSION}_NORMALIZATION_CURVE.csv"
    df.to_csv(curve_csv, index=False)
    anchors = df.iloc[[abs(df.redshift_z-z).idxmin() for z in [0.5, 1, 2, 5, 10, 13, 15, 20, 50]]]
    anchor_csv = CSV / f"{VERSION}_ANCHOR_VALUES.csv"
    anchors.to_csv(anchor_csv, index=False)
    notes_csv = CSV / f"{VERSION}_MODEL_NOTES.csv"
    notes_csv.write_text("field,value\nmode,toy model / teaching dashboard\nnormalization,raw proxy divided by shell volume proxy\nhorizon,46.5 Gly is context not a galaxy location\n", encoding="utf-8")
    dash = dashboard(df)
    horiz = horizon_plot(df)
    peak = df.loc[df.raw_detected_proxy_norm.idxmax()]

    print(f"CODE OUTPUT: {VERSION}\n")
    table([
        ("Cosmology", "Astropy Planck18"),
        ("Method", "Schechter-style toy completeness + shell-volume normalization"),
        ("Raw proxy peak z", f"{peak.redshift_z:.3f}"),
        ("Peak comoving distance", f"{peak.comoving_distance_Gly:.3f} Gly"),
        ("Horizon context", f"{HORIZON_GLY:.1f} Gly, not a galaxy location"),
    ], ["Metric", "Value"])
    print("\nOUTPUTS")
    table([
        ("png", dash),
        ("png", horiz),
        ("csv", curve_csv),
        ("csv", anchor_csv),
        ("csv", notes_csv),
    ], ["Type", "Path"])
    print("\nThis is a teaching model, not an official JWST selection/completeness product.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
