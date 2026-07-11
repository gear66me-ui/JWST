# JWST_0058
# Independent NIST laboratory rest-line references versus real JWST observed samples.
# LEFT is not a reconstructed MoM-z14 spectrum. RIGHT is cached JWST/MAST data.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import math
import subprocess
import sys

VERSION = "JWST_0058"
GALAXY = "MoM-z14"
Z = 14.44
S = 1.0 + Z
C_UM_THz = 299.792458
C_AA_THz = 2997924.58
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG, CSV = OUT / "PNG", OUT / "CSV"
BG, AXBG, GRID = "#050712", "#07101f", "#1f6f8b"
TEXT, MUTED, CYAN, ORANGE, BLUE = "#e6f4ff", "#8fb3c7", "#18c7d8", "#ff9d2e", "#43b9ff"

LINES = [
    (1, "N IV] 1487", ["N IV"], [1483.32, 1486.50], (1480, 1490)),
    (2, "C IV 1548,1551", ["C IV"], [1548.20, 1550.77], (1545, 1554)),
    (3, "He II 1640 + O III] 1661,1666", ["He II", "O III"], [1640.42, 1660.81, 1666.15], (1637, 1669)),
    (4, "N III] 1747-1754", ["N III"], [1746.82, 1748.65, 1749.67, 1752.16, 1753.99], (1744, 1757)),
    (5, "C III] 1907,1909", ["C III"], [1906.68, 1908.73], (1904, 1912)),
]


def need(name):
    try:
        importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", name])


def latest(pattern):
    files = sorted(CSV.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def locate():
    line_csv = CSV / "JWST_0052_MoM-z14_LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv"
    if not line_csv.exists():
        line_csv = latest("JWST_0052_*LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv")
    raw_csv = next((CSV / n for n in [
        "JWST_0048_REAL_RAW_SPECTRUM.csv",
        "JWST_0047_REAL_RAW_SPECTRUM.csv",
        "JWST_0046_REAL_RAW_SPECTRUM.csv",
    ] if (CSV / n).exists()), None) or latest("JWST_*_REAL_RAW_SPECTRUM.csv")
    if line_csv is None or raw_csv is None:
        raise FileNotFoundError("Run JWST_0052 first; cached line and spectrum CSV files are required.")
    return line_csv, raw_csv


def col_start(df, prefix):
    return next((c for c in df.columns if str(c).lower().startswith(prefix.lower())), None)


def fnum(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def load_jwst(line_csv, raw_csv):
    import numpy as np
    import pandas as pd
    lines = pd.read_csv(line_csv)
    raw = pd.read_csv(raw_csv)
    wc, fc = col_start(raw, "wavelength"), col_start(raw, "flux_raw_")
    if wc is None or fc is None:
        raise RuntimeError("Could not identify wavelength and raw-flux columns.")
    wave, flux = raw[wc].to_numpy(float), raw[fc].to_numpy(float)
    ok = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    order = np.argsort(wave[ok])
    return lines, wave[ok][order], flux[ok][order], str(fc)


def nist_reference(ions, anchors, bounds):
    import astropy.units as u
    from astroquery.nist import Nist
    candidates, audit = [], []
    for ion in ions:
        try:
            tab = Nist.query(bounds[0] * u.AA, bounds[1] * u.AA, linename=ion,
                             output_order="wavelength", wavelength_type="vacuum")
            for row in tab:
                value, source = None, None
                for column in ["Observed", "Ritz"]:
                    if column in tab.colnames:
                        value = fnum(row[column])
                        if value is not None:
                            source = column
                            break
                if value is not None:
                    candidates.append((ion, value, source))
                    audit.append((ion, value, source, "OK"))
        except Exception as exc:
            audit.append((ion, math.nan, "", f"ERROR:{type(exc).__name__}"))

    chosen = []
    for anchor in anchors:
        near = min(candidates, key=lambda r: abs(r[1] - anchor), default=None)
        if near and abs(near[1] - anchor) <= 1.5:
            chosen.append((*near, anchor, "NIST"))
        else:
            chosen.append((ions[0], anchor, "CURATED_FALLBACK", anchor, "FALLBACK"))
    unique = []
    for row in chosen:
        if not any(abs(row[1] - old[1]) < 0.01 for old in unique):
            unique.append(row)
    return unique, audit


def style(ax):
    ax.set_facecolor(AXBG)
    ax.grid(True, color=GRID, linewidth=0.45, alpha=0.48)
    ax.tick_params(colors=TEXT, labelsize=7.2)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.82)


def rest_panel(ax, number, name, refs):
    import numpy as np
    style(ax)
    freq = np.array([C_AA_THz / r[1] for r in refs])
    for f, row in zip(freq, refs):
        ax.vlines(f, 0, 1, color=CYAN, linewidth=2)
        ax.scatter([f], [1], s=30, color=TEXT, edgecolor=CYAN, linewidth=0.8)
        ax.text(f, 1.03, f"{row[0]}\n{row[1]:.3f} Å", color=TEXT, fontsize=6.2,
                ha="center", va="bottom")
    span = float(np.ptp(freq)) if len(freq) > 1 else max(2.0, freq[0] * 0.0015)
    pad = max(1.0, span * 0.35)
    ax.set_xlim(float(freq.min() - pad), float(freq.max() + pad))
    ax.set_ylim(0, 1.28)
    ax.set_yticks([])
    ax.set_xlabel("Laboratory rest frequency, THz", fontsize=7.8)
    ax.set_title(f"{number} {name} — NIST laboratory reference", fontsize=9.1)
    ax.text(0.018, 0.95, "Independent atomic positions\nNOT a MoM-z14 spectrum\nStick height is not intensity",
            transform=ax.transAxes, color=TEXT, fontsize=6.6, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#07111f", edgecolor=CYAN, alpha=0.95))
    return freq


def observed_panel(ax, number, name, jwst_row, wave, flux, rest_freq):
    import numpy as np
    style(ax)
    expected = rest_freq / S
    expected_um = C_UM_THz / expected
    lo, hi = float(expected_um.min() - 0.018), float(expected_um.max() + 0.018)
    mask = (wave >= lo) & (wave <= hi)
    if int(mask.sum()) < 3:
        mask = (wave >= lo - 0.025) & (wave <= hi + 0.025)
    lw, lf = wave[mask], flux[mask]
    of = C_UM_THz / lw
    order = np.argsort(of)
    of, lf = of[order], lf[order]
    ax.plot(of, lf, color=TEXT, linewidth=0.82, alpha=0.92)
    ax.scatter(of, lf, s=20, color=BLUE, edgecolor=BG, linewidth=0.4,
               label=f"real JWST samples: {len(of)}")
    for f in expected:
        ax.axvline(f, color=CYAN, linestyle="--", linewidth=1.15)
    peak_um = fnum(jwst_row.get("raw_local_peak_um_exploratory")) or float(np.median(expected_um))
    peak_f = C_UM_THz / peak_um
    ax.axvline(peak_f, color=ORANGE, linewidth=1.65, label=f"observed peak: {peak_f:.6f} THz")
    if len(of) > 1:
        xp = 0.05 * float(np.ptp(of))
        yp = 0.10 * float(np.ptp(lf)) if float(np.ptp(lf)) > 0 else 1e-8
        ax.set_xlim(float(of.min() - xp), float(of.max() + xp))
        ax.set_ylim(float(lf.min() - yp), float(lf.max() + yp))
    ax.set_xlabel("Observed frequency, THz", fontsize=7.8)
    ax.set_ylabel("Raw JWST flux", fontsize=7.8)
    ax.set_title(f"{number} {name} — JWST observed MoM-z14", fontsize=9.1)
    ax.text(0.018, 0.95, f"cyan = NIST/(1+z)\norange = raw local peak\npublished z = {Z:.2f}",
            transform=ax.transAxes, color=TEXT, fontsize=6.6, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#07111f", edgecolor=ORANGE, alpha=0.95))
    leg = ax.legend(loc="best", fontsize=6.0, facecolor="#07111f", edgecolor=GRID, framealpha=0.96)
    for text in leg.get_texts():
        text.set_color(TEXT)


def main():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "astroquery"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    import matplotlib.pyplot as plt
    line_csv, raw_csv = locate()
    lines, wave, flux, flux_col = load_jwst(line_csv, raw_csv)
    fig, axes = plt.subplots(5, 2, figsize=(18.5, 20.5), facecolor=BG)
    plotted, audit_rows = [], []

    for i, (number, name, ions, anchors, bounds) in enumerate(LINES):
        refs, audit = nist_reference(ions, anchors, bounds)
        rest_freq = rest_panel(axes[i, 0], number, name, refs)
        match = lines[lines["n"].astype(int) == number]
        if match.empty:
            raise RuntimeError(f"Missing JWST line row {number}")
        observed_panel(axes[i, 1], number, name, match.iloc[0], wave, flux, rest_freq)
        for ion, wavelength, source, anchor, status in refs:
            plotted.append({"line_number": number, "line_complex": name, "ion": ion,
                            "rest_wavelength_A": wavelength, "rest_frequency_THz": C_AA_THz / wavelength,
                            "expected_observed_frequency_THz": (C_AA_THz / wavelength) / S,
                            "source_column": source, "anchor_A": anchor, "status": status})
        for ion, wavelength, source, status in audit:
            audit_rows.append({"line_number": number, "line_complex": name, "ion": ion,
                               "wavelength_A": wavelength, "source_column": source, "status": status})

    fig.suptitle(f"{VERSION} — {GALAXY}: NIST REST REFERENCE versus REAL JWST OBSERVED",
                 color=TEXT, fontsize=16, fontweight="bold", y=0.992)
    fig.text(0.5, 0.974,
             "LEFT: independent NIST atomic transition positions.  RIGHT: actual JWST/MAST samples.  These are not the same dataset.",
             color=MUTED, fontsize=9.5, ha="center")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.955, bottom=0.035, hspace=0.35, wspace=0.16)
    png = PNG / f"{VERSION}_{GALAXY}_NIST_REST_VS_JWST_OBSERVED.png"
    fig.savefig(png, dpi=235, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    plotted_csv = CSV / f"{VERSION}_{GALAXY}_NIST_REFERENCE_LINES.csv"
    audit_csv = CSV / f"{VERSION}_{GALAXY}_NIST_QUERY_AUDIT.csv"
    pd.DataFrame(plotted).to_csv(plotted_csv, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"LEFT SOURCE  : NIST Atomic Spectra Database laboratory transition positions")
    print(f"RIGHT SOURCE : real cached JWST/MAST spectrum")
    print(f"PLOT PNG     : {png}")
    print(f"NIST CSV     : {plotted_csv}")
    print(f"AUDIT CSV    : {audit_csv}")
    print(f"JWST CSV     : {raw_csv}")
    print(f"FLUX COLUMN  : {flux_col}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
