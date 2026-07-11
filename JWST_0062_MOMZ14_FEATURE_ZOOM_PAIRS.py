# JWST_0062
# Fourteen narrow-band rest-versus-observed plots from real JWST samples.
# No AI images. Matplotlib only. No network queries.

from pathlib import Path
from datetime import datetime, timezone
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0062"
GALAXY = "MoM-z14"
Z = 14.44
S = 1.0 + Z
C_NM_THz = 299792.458
HALF_WIDTH_NM = 2.5

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG, AXBG, GRID = "#050712", "#07101f", "#1f6f8b"
TEXT, MUTED = "#e6f4ff", "#8fb3c7"
CYAN, ORANGE, POINT, FAINT = "#18c7d8", "#ff9d2e", "#d9edf7", "#6f8090"

LINES = [
    (1, "N IV] 1483", 148.332),
    (2, "N IV] 1487", 148.650),
    (3, "C IV 1548", 154.820),
    (4, "C IV 1551", 155.077),
    (5, "He II 1640", 164.042),
    (6, "O III] 1661", 166.081),
    (7, "O III] 1666", 166.615),
    (8, "N III] 1747", 174.682),
    (9, "N III] 1749", 174.865),
    (10, "N III] 1750", 174.967),
    (11, "N III] 1752", 175.216),
    (12, "N III] 1754", 175.399),
    (13, "C III] 1907", 190.668),
    (14, "C III] 1909", 190.873),
]


def newest(pattern):
    files = sorted(CSV.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def locate_spectrum():
    for name in [
        "JWST_0060_MoM-z14_EXACT_JWST.csv",
        "JWST_0059_MoM-z14_EXACT_JWST.csv",
        "JWST_0048_REAL_RAW_SPECTRUM.csv",
        "JWST_0047_REAL_RAW_SPECTRUM.csv",
        "JWST_0046_REAL_RAW_SPECTRUM.csv",
    ]:
        p = CSV / name
        if p.exists() and p.stat().st_size > 100:
            return p
    return newest("JWST_*_MoM-z14_EXACT_JWST.csv") or newest("JWST_*_REAL_RAW_SPECTRUM.csv")


def pick_col(df, exact, prefixes):
    low = {str(c).lower(): c for c in df.columns}
    for name in exact:
        if name.lower() in low:
            return low[name.lower()]
    return next((c for c in df.columns if any(str(c).lower().startswith(p) for p in prefixes)), None)


def load_data(path):
    if path is None:
        raise FileNotFoundError("No cached JWST spectrum CSV found in /content/JWST_OUTPUT/CSV")
    df = pd.read_csv(path)
    wc = pick_col(df, ["wavelength_um", "wavelength_nm", "wavelength", "wave"], ["wavelength", "wave"])
    fc = pick_col(df, ["flux", "raw_flux", "jwst_flux"], ["flux_raw_", "flux", "raw_flux"])
    if wc is None or fc is None:
        raise RuntimeError(f"Could not identify wavelength/flux columns in {path.name}")
    w = df[wc].to_numpy(float)
    f = df[fc].to_numpy(float)
    ok = np.isfinite(w) & np.isfinite(f) & (w > 0)
    w, f = w[ok], f[ok]
    name = str(wc).lower()
    med = float(np.nanmedian(w))
    if "_um" in name or med < 20:
        w_nm = w * 1000.0
    elif "_nm" in name or med < 10000:
        w_nm = w
    else:
        w_nm = w / 10.0
    order = np.argsort(w_nm)
    return w_nm[order], f[order], wc, fc


def nu(w_nm):
    return C_NM_THz / np.asarray(w_nm, float)


def lam(freq_thz):
    return C_NM_THz / np.asarray(freq_thz, float)


def style(ax):
    ax.set_facecolor(AXBG)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")


def y_limits(values):
    lo, hi = np.nanpercentile(values, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(values), np.nanmax(values)
    pad = 0.1 * (hi - lo) if hi > lo else 1e-8
    return lo - pad, hi + pad


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def plot_one(index, label, rest_center, obs_nm, flux, source):
    rest_all = obs_nm / S
    lo, hi = rest_center - HALF_WIDTH_NM, rest_center + HALF_WIDTH_NM
    mask = (rest_all >= lo) & (rest_all <= hi) & np.isfinite(flux)
    if mask.sum() < 5:
        raise RuntimeError(f"{label}: only {mask.sum()} samples in the narrow band")

    rw, ow, ff = rest_all[mask], obs_nm[mask], flux[mask]
    order = np.argsort(rw)
    rw, ow, ff = rw[order], ow[order], ff[order]

    fig, (left, right) = plt.subplots(1, 2, figsize=(15.5, 6.4), sharey=True, facecolor=BG)
    style(left)
    style(right)

    left.plot(rw, ff, color=CYAN, lw=0.95)
    left.scatter(rw, ff, s=22, color=POINT, edgecolor=BG, lw=0.35, zorder=4)
    right.plot(ow, ff, color=ORANGE, lw=0.95)
    right.scatter(ow, ff, s=22, color=POINT, edgecolor=BG, lw=0.35, zorder=4)

    for other_i, other_label, other_rest in LINES:
        if abs(other_rest - rest_center) <= HALF_WIDTH_NM:
            target = other_i == index
            left.axvline(other_rest, color=CYAN if target else FAINT, ls="-" if target else "--",
                         lw=1.8 if target else 0.8, alpha=0.95 if target else 0.7)
            right.axvline(other_rest * S, color=ORANGE if target else FAINT, ls="-" if target else "--",
                          lw=1.8 if target else 0.8, alpha=0.95 if target else 0.7)

    left.set_xlim(lo, hi)
    right.set_xlim(lo * S, hi * S)
    left.set_ylim(*y_limits(ff))
    left.set_title(f"REST FRAME — {label}", fontsize=12, pad=10)
    right.set_title(f"OBSERVED FRAME — {label}", fontsize=12, pad=10)
    left.set_xlabel("Rest wavelength, nm")
    right.set_xlabel("Observed wavelength, nm")
    left.set_ylabel("JWST flux samples")

    top_l = left.secondary_xaxis("top", functions=(nu, lam))
    top_l.set_xlabel("Rest frequency, THz", color=TEXT)
    top_l.tick_params(colors=TEXT, labelsize=8)
    top_r = right.secondary_xaxis("top", functions=(nu, lam))
    top_r.set_xlabel("Observed frequency, THz", color=TEXT)
    top_r.tick_params(colors=TEXT, labelsize=8)

    obs_center = rest_center * S
    left.text(0.02, 0.05,
              f"λrest = {rest_center:.3f} nm\nνrest = {nu(rest_center):.3f} THz\nwindow = {lo:.3f}–{hi:.3f} nm",
              transform=left.transAxes, color=TEXT, fontsize=8, va="bottom",
              bbox=dict(boxstyle="round,pad=.3", fc="#07111f", ec=CYAN, alpha=.95))
    right.text(0.02, 0.05,
               f"λobs = {obs_center:.3f} nm\nνobs = {nu(obs_center):.3f} THz\nstretch = {S:.2f}×",
               transform=right.transAxes, color=TEXT, fontsize=8, va="bottom",
               bbox=dict(boxstyle="round,pad=.3", fc="#07111f", ec=ORANGE, alpha=.95))

    fig.suptitle(f"{VERSION} — {GALAXY} — transition {index:02d}/14",
                 color=TEXT, fontsize=15, fontweight="bold", y=.975)
    fig.text(.5, .915,
             "Same real JWST samples in corresponding narrow rest and observed bandpasses. "
             "Solid line = target; dashed lines = nearby known components.",
             ha="center", color=MUTED, fontsize=9)
    fig.text(.5, .018,
             f"Source: {source.name}. Left is a rest-frame remapping of the observed JWST spectrum.",
             ha="center", color=MUTED, fontsize=8)
    fig.subplots_adjust(left=.07, right=.985, top=.85, bottom=.12, wspace=.12)

    stem = f"{VERSION}_{index:02d}_{safe_name(label)}"
    png_path = PNG / f"{stem}_REST_VS_OBSERVED.png"
    csv_path = CSV / f"{stem}_SAMPLES.csv"
    fig.savefig(png_path, dpi=245, facecolor=BG)
    plt.show()
    plt.close(fig)

    pd.DataFrame({
        "transition_number": index,
        "transition_label": label,
        "rest_wavelength_nm": rw,
        "rest_frequency_THz": nu(rw),
        "observed_wavelength_nm": ow,
        "observed_frequency_THz": nu(ow),
        "jwst_flux": ff,
        "target_rest_nm": rest_center,
        "target_observed_nm": obs_center,
        "stretch_factor": S,
        "redshift_z": Z,
    }).to_csv(csv_path, index=False)

    return {
        "number": index,
        "label": label,
        "rest_nm": rest_center,
        "observed_nm": obs_center,
        "rest_THz": float(nu(rest_center)),
        "observed_THz": float(nu(obs_center)),
        "samples": int(mask.sum()),
        "png": str(png_path),
        "csv": str(csv_path),
    }


def main():
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    source = locate_spectrum()
    obs_nm, flux, wave_col, flux_col = load_data(source)

    print(f"CODE OUTPUT: {VERSION}")
    print("MODE       : 14 independent narrow-band plot pairs")
    print(f"SOURCE CSV : {source}")
    print()

    rows = []
    failures = []
    for index, label, rest_center in LINES:
        try:
            print(f"PLOT {index:02d}/14 | {label}")
            rows.append(plot_one(index, label, rest_center, obs_nm, flux, source))
        except Exception as exc:
            print(f"FAILED {index:02d}/14 | {label} | {type(exc).__name__}: {exc}")
            failures.append((index, label, type(exc).__name__, str(exc)))

    if not rows:
        raise RuntimeError("No narrow-band transition plots were generated")

    summary = CSV / f"{VERSION}_{GALAXY}_TRANSITION_INDEX.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    if failures:
        pd.DataFrame(failures, columns=["number", "label", "error_type", "message"]).to_csv(
            CSV / f"{VERSION}_{GALAXY}_FAILURES.csv", index=False
        )

    print()
    for row in rows:
        print(f"{row['number']:02d} | {row['label']:<15} | "
              f"{row['rest_nm']:8.3f} nm -> {row['observed_nm']:9.3f} nm | "
              f"samples={row['samples']:4d}")
    print()
    print(f"WAVELENGTH COLUMN : {wave_col}")
    print(f"FLUX COLUMN       : {flux_col}")
    print(f"REST HALF-WIDTH   : {HALF_WIDTH_NM:.3f} nm")
    print(f"GENERATED         : {len(rows)} PNG + {len(rows)} CSV")
    print(f"SUMMARY CSV       : {summary}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
