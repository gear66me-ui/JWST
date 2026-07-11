# JWST_0040
# Audit: Computed high-z H-alpha/[N II] spectrum curve from redshifted rest wavelengths. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0040"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

TARGET_NAME = "MoM-z14"
TARGET_Z = 14.44
TARGET_STATUS = "computed wavelength positions from reported redshift; not an observed H-alpha flux spectrum"
SPEED_OF_LIGHT_M_PER_S = 299_792_458.0

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen", 0.45),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha", 1.00),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen", 0.70),
]


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


def lambda_obs(rest_um, z):
    return rest_um * (1.0 + z)


def freq_thz_from_um(wavelength_um):
    return SPEED_OF_LIGHT_M_PER_S / (wavelength_um * 1.0e-6) / 1.0e12


def gaussian(x, mu, sigma, amp):
    import numpy as np
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def build_line_table():
    import pandas as pd
    rows = []
    for n, label, rest_um, species, amp in TRIPLET:
        obs_um = lambda_obs(rest_um, TARGET_Z)
        rows.append({
            "target": TARGET_NAME,
            "z": TARGET_Z,
            "status": TARGET_STATUS,
            "line_number": n,
            "line": label,
            "species": species,
            "relative_amplitude": amp,
            "rest_um": rest_um,
            "observed_um": obs_um,
            "observed_frequency_THz": freq_thz_from_um(obs_um),
            "stretch_factor_1_plus_z": 1.0 + TARGET_Z,
            "z_check": obs_um / rest_um - 1.0,
        })
    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "target": TARGET_NAME,
        "z": TARGET_Z,
        "status": TARGET_STATUS,
        "line_count": len(df),
        "sum_rest_um": df["rest_um"].sum(),
        "avg_rest_um": df["rest_um"].mean(),
        "sum_observed_um": df["observed_um"].sum(),
        "avg_observed_um": df["observed_um"].mean(),
        "avg_observed_frequency_THz": df["observed_frequency_THz"].mean(),
        "stretch_from_avg_wavelengths": df["observed_um"].mean() / df["rest_um"].mean(),
        "z_from_avg_wavelengths": df["observed_um"].mean() / df["rest_um"].mean() - 1.0,
        "note": "Average wavelength is an arithmetic average of three line positions, not a fourth spectral emission line.",
    }])
    return df, summary


def computed_spectrum(line_df):
    import numpy as np
    xmin = float(line_df["observed_um"].min()) - 0.075
    xmax = float(line_df["observed_um"].max()) + 0.075
    wavelength = np.linspace(xmin, xmax, 2600)

    # Resolution-like width for a clean visual line spread, not an instrument extraction.
    # Around 10 microns, R~2500 gives FWHM~0.004 microns; sigma = FWHM/2.355.
    resolving_power_visual = 2500.0
    sigma = float(line_df["observed_um"].mean()) / resolving_power_visual / 2.355

    continuum = 0.045 + 0.010 * (wavelength - wavelength.mean())
    flux = continuum.copy()
    for row in line_df.itertuples():
        flux += gaussian(wavelength, row.observed_um, sigma, row.relative_amplitude)
    flux = flux / flux.max()
    residual = flux - continuum

    frequency = freq_thz_from_um(wavelength)
    return wavelength, frequency, flux, continuum, residual, sigma


def dark_axis(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.4, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_wavelength_spectrum(line_df, summary, wavelength, flux, continuum, sigma):
    import matplotlib.pyplot as plt
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    avg = float(summary.iloc[0]["avg_observed_um"])
    fig, ax = plt.subplots(figsize=(16.8, 8.6))
    dark_axis(ax)
    ax.plot(wavelength, flux, color="#e0f2fe", linewidth=1.45, label="computed spectrum curve: normalized flux proxy")
    ax.plot(wavelength, continuum / max(flux), color="#94a3b8", linewidth=1.0, alpha=0.75, label="low continuum baseline")
    for idx, row in enumerate(line_df.itertuples()):
        ax.axvline(row.observed_um, color=colors[idx], linewidth=1.25, alpha=0.80)
        ax.scatter([row.observed_um], [float(max(flux)) * 0.96], s=66, color=colors[idx], edgecolor="#f8fafc", zorder=7)
        ax.text(row.observed_um, 0.91, f"{int(row.line_number)}\n{row.line}\n{row.observed_um:.6f} µm",
                color=colors[idx], fontsize=9.0, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#020617", edgecolor=colors[idx], alpha=0.90))
    ax.axvline(avg, color="#f97316", linewidth=2.10, linestyle="--", alpha=0.90,
               label=f"orange dashed average wavelength = {avg:.6f} µm")
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel("Normalized flux proxy")
    ax.set_title(f"{VERSION} — computed spectral curve for {TARGET_NAME} H-alpha/[N II] at z={TARGET_Z:.4f}")
    ax.set_ylim(0, 1.12)
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMPUTED_WAVELENGTH_SPECTRUM_CURVE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_frequency_spectrum(line_df, summary, wavelength, frequency, flux):
    import matplotlib.pyplot as plt
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    avg_freq = float(summary.iloc[0]["avg_observed_frequency_THz"])
    order = frequency.argsort()
    fig, ax = plt.subplots(figsize=(16.8, 8.6))
    dark_axis(ax)
    ax.plot(frequency[order], flux[order], color="#e0f2fe", linewidth=1.45, label="computed spectrum curve: normalized flux proxy")
    for idx, row in enumerate(line_df.itertuples()):
        f = row.observed_frequency_THz
        ax.axvline(f, color=colors[idx], linewidth=1.25, alpha=0.80)
        ax.scatter([f], [0.96], s=66, color=colors[idx], edgecolor="#f8fafc", zorder=7)
        ax.text(f, 0.91, f"{int(row.line_number)}\n{row.line}\n{f:.6f} THz",
                color=colors[idx], fontsize=9.0, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#020617", edgecolor=colors[idx], alpha=0.90))
    ax.axvline(avg_freq, color="#f97316", linewidth=2.10, linestyle="--", alpha=0.90,
               label=f"orange dashed average frequency = {avg_freq:.6f} THz")
    ax.set_xlabel("Observed frequency, THz")
    ax.set_ylabel("Normalized flux proxy")
    ax.set_title(f"{VERSION} — same computed spectrum on frequency axis for {TARGET_NAME}")
    ax.set_ylim(0, 1.12)
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_COMPUTED_FREQUENCY_SPECTRUM_CURVE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_table(line_df, summary):
    import matplotlib.pyplot as plt
    s = summary.iloc[0]
    rows = []
    for row in line_df.itertuples():
        rows.append([int(row.line_number), row.line, f"{row.rest_um:.6f}", f"{row.observed_um:.6f}", f"{row.observed_frequency_THz:.6f}"])
    rows.append(["Σ", "SUM", f"{s.sum_rest_um:.6f}", f"{s.sum_observed_um:.6f}", ""])
    rows.append(["μ", "AVERAGE", f"{s.avg_rest_um:.6f}", f"{s.avg_observed_um:.6f}", f"{s.avg_observed_frequency_THz:.6f}"])
    fig, ax = plt.subplots(figsize=(16.4, 5.9))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — line positions used to compute the spectrum curve", color="#f8fafc", fontsize=14.0, pad=14)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "observed λ µm", "observed ν THz"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#020617" if r % 2 else "#0f172a")
            cell.get_text().set_color("#dbeafe")
            if r >= 4:
                cell.set_facecolor("#431407")
                cell.get_text().set_color("#fed7aa")
                cell.get_text().set_weight("bold")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SPECTRUM_CURVE_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    line_df, summary = build_line_table()
    wavelength, frequency, flux, continuum, residual, sigma = computed_spectrum(line_df)

    import pandas as pd
    line_csv = CSV / f"{VERSION}_LINE_POSITIONS.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"
    spectrum_csv = CSV / f"{VERSION}_COMPUTED_SPECTRUM_CURVE.csv"
    line_df.to_csv(line_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    pd.DataFrame({
        "observed_wavelength_um": wavelength,
        "observed_frequency_THz": frequency,
        "normalized_flux_proxy": flux,
        "continuum_proxy": continuum,
        "residual_proxy": residual,
    }).to_csv(spectrum_csv, index=False)

    p1 = plot_wavelength_spectrum(line_df, summary, wavelength, flux, continuum, sigma)
    p2 = plot_frequency_spectrum(line_df, summary, wavelength, frequency, flux)
    p3 = plot_table(line_df, summary)
    s = summary.iloc[0]

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Plot type", "computed spectral curve, not just vertical lines"),
        ("Target", TARGET_NAME),
        ("z", f"{TARGET_Z:.6f}"),
        ("Curve status", TARGET_STATUS),
        ("Gaussian sigma um", f"{sigma:.6f}"),
        ("Observed wavelength sum um", f"{s.sum_observed_um:.6f}"),
        ("Observed wavelength average um", f"{s.avg_observed_um:.6f}"),
        ("Observed frequency average THz", f"{s.avg_observed_frequency_THz:.6f}"),
        ("z from average wavelengths", f"{s.z_from_avg_wavelengths:.6f}"),
        ("Wavelength plot", str(p1)),
        ("Frequency plot", str(p2)),
        ("Table PNG", str(p3)),
        ("Spectrum CSV", str(spectrum_csv)),
        ("Line CSV", str(line_csv)),
        ("Summary CSV", str(summary_csv)),
    ], ["Field", "Value"])

    print("\nLINE VALUES, SUM, AVERAGE")
    rows = []
    for row in line_df.itertuples():
        rows.append((int(row.line_number), row.line, f"{row.rest_um:.6f}", f"{row.observed_um:.6f}", f"{row.observed_frequency_THz:.6f}"))
    rows.append(("Σ", "SUM", f"{s.sum_rest_um:.6f}", f"{s.sum_observed_um:.6f}", ""))
    rows.append(("μ", "AVERAGE", f"{s.avg_rest_um:.6f}", f"{s.avg_observed_um:.6f}", f"{s.avg_observed_frequency_THz:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Observed um", "Observed THz"])

    print("\nCOMMENTS")
    print("This is a computed spectral signature made from redshifted rest wavelengths and Gaussian line profiles.")
    print("It is not a downloaded observed H-alpha/[N II] spectrum for MoM-z14.")
    print("The orange dashed average is a mathematical average of the line positions, not a physical fourth emission line.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
