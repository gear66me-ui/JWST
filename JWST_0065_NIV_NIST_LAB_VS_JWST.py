# JWST_0065
# N IV only: actual NIST ASD atomic line records versus real JWST observed data.
# LEFT is NOT a remapped JWST curve.
# RIGHT is the cached JWST spectrum in the corresponding observed window.
# No synthetic line profile. No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import math
import re
import subprocess
import sys

VERSION = "JWST_0065"
GALAXY = "MoM-z14"
ION = "N IV"
Z = 14.44
STRETCH = 1.0 + Z
C_NM_THz = 299792.458
REST_LOW_NM = 147.5
REST_HIGH_NM = 149.5
TARGET_REST_A = [1483.32, 1486.50]

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
LAB_COLOR = "#ffd84d"
JWST_COLOR = "#ff9d2e"
POINT_COLOR = "#d9edf7"
REF_COLOR = "#ff5a66"


def need(package):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def parse_number(value):
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text in {"--", "nan", "None"}:
        return math.nan
    match = re.search(r"[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?", text)
    return float(match.group(0)) if match else math.nan


def newest(pattern):
    files = sorted(CSV.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def locate_jwst_csv():
    preferred = [
        "JWST_0060_MoM-z14_EXACT_JWST.csv",
        "JWST_0059_MoM-z14_EXACT_JWST.csv",
        "JWST_0048_REAL_RAW_SPECTRUM.csv",
        "JWST_0047_REAL_RAW_SPECTRUM.csv",
        "JWST_0046_REAL_RAW_SPECTRUM.csv",
    ]
    for filename in preferred:
        path = CSV / filename
        if path.exists() and path.stat().st_size > 100:
            status = "coordinate-matched JWST extraction" if "EXACT_JWST" in filename else "cached JWST/MAST spectrum"
            return path, status

    path = newest("JWST_*_MoM-z14_EXACT_JWST.csv")
    if path is not None:
        return path, "coordinate-matched JWST extraction"

    path = newest("JWST_*_REAL_RAW_SPECTRUM.csv")
    if path is not None:
        return path, "cached JWST/MAST spectrum"

    raise FileNotFoundError("No cached JWST spectrum CSV found in /content/JWST_OUTPUT/CSV")


def find_column(frame, exact_names, prefixes):
    lookup = {str(column).lower(): column for column in frame.columns}
    for name in exact_names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    for column in frame.columns:
        text = str(column).lower()
        if any(text.startswith(prefix.lower()) for prefix in prefixes):
            return column
    return None


def load_jwst(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    wave_column = find_column(
        frame,
        ["wavelength_um", "wavelength_nm", "wavelength", "wave"],
        ["wavelength", "wave"],
    )
    flux_column = find_column(
        frame,
        ["flux", "raw_flux", "jwst_flux"],
        ["flux_raw_", "flux", "raw_flux"],
    )
    if wave_column is None or flux_column is None:
        raise RuntimeError(f"Could not identify wavelength/flux columns in {path.name}")

    wavelength = frame[wave_column].to_numpy(float)
    flux = frame[flux_column].to_numpy(float)
    finite = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
    wavelength = wavelength[finite]
    flux = flux[finite]

    median = float(np.nanmedian(wavelength))
    wave_name = str(wave_column).lower()
    if "_um" in wave_name or median < 20:
        observed_nm = wavelength * 1000.0
    elif "_nm" in wave_name or median < 10000:
        observed_nm = wavelength
    else:
        observed_nm = wavelength / 10.0

    order = np.argsort(observed_nm)
    return observed_nm[order], flux[order], str(wave_column), str(flux_column)


def query_nist_niv():
    import numpy as np
    import pandas as pd
    import astropy.units as u
    from astroquery.nist import Nist

    print("NIST QUERY : N IV, 1470-1500 Angstrom, vacuum wavelengths")
    table = Nist.query(
        1470.0 * u.AA,
        1500.0 * u.AA,
        linename=ION,
        output_order="wavelength",
        wavelength_type="vacuum",
    )
    if table is None or len(table) == 0:
        raise RuntimeError("NIST ASD returned no N IV records in the requested range")

    raw = table.to_pandas()
    raw_path = CSV / f"{VERSION}_NIST_NIV_RAW.csv"
    raw.to_csv(raw_path, index=False)

    rows = []
    for _, row in raw.iterrows():
        observed_a = parse_number(row.get("Observed"))
        ritz_a = parse_number(row.get("Ritz"))
        wavelength_a = observed_a if np.isfinite(observed_a) else ritz_a
        if not np.isfinite(wavelength_a):
            continue

        relative = parse_number(row.get("Rel."))
        aki = parse_number(row.get("Aki"))
        if np.isfinite(relative) and relative > 0:
            strength = relative
            strength_source = "NIST relative intensity"
        elif np.isfinite(aki) and aki > 0:
            strength = aki
            strength_source = "NIST Aki transition probability"
        else:
            strength = math.nan
            strength_source = "unavailable"

        rows.append(
            {
                "wavelength_A": wavelength_a,
                "wavelength_nm": wavelength_a / 10.0,
                "observed_A": observed_a,
                "ritz_A": ritz_a,
                "relative_intensity": relative,
                "Aki_s-1": aki,
                "strength_raw": strength,
                "strength_source": strength_source,
            }
        )

    lines = pd.DataFrame(rows)
    lines = lines[
        (lines["wavelength_nm"] >= REST_LOW_NM)
        & (lines["wavelength_nm"] <= REST_HIGH_NM)
    ].copy()
    if lines.empty:
        raise RuntimeError("NIST ASD returned records, but none lie inside 147.5-149.5 nm")

    usable = np.isfinite(lines["strength_raw"].to_numpy(float)) & (lines["strength_raw"].to_numpy(float) > 0)
    if not usable.any():
        raise RuntimeError(
            "NIST returned line positions but no usable relative intensity or Aki values; "
            "refusing to fabricate a laboratory profile"
        )

    minimum_positive = float(np.nanmin(lines.loc[usable, "strength_raw"]))
    lines.loc[~usable, "strength_raw"] = minimum_positive
    maximum = float(np.nanmax(lines["strength_raw"]))
    lines["strength_normalized"] = lines["strength_raw"] / maximum

    selected_rows = []
    for target_a in TARGET_REST_A:
        index = (lines["wavelength_A"] - target_a).abs().idxmin()
        selected = lines.loc[index]
        delta_a = abs(float(selected["wavelength_A"]) - target_a)
        if delta_a > 2.0:
            raise RuntimeError(
                f"Nearest NIST N IV record to {target_a:.2f} A is {delta_a:.3f} A away; "
                "refusing an uncertain match"
            )
        selected_rows.append(selected)

    selected = pd.DataFrame(selected_rows).drop_duplicates(subset=["wavelength_A"])
    selected_path = CSV / f"{VERSION}_NIST_NIV_SELECTED.csv"
    lines.to_csv(CSV / f"{VERSION}_NIST_NIV_LINE_DISTRIBUTION.csv", index=False)
    selected.to_csv(selected_path, index=False)
    return lines.sort_values("wavelength_nm"), selected.sort_values("wavelength_nm"), raw_path, selected_path


def frequency_thz(wavelength_nm):
    import numpy as np
    return C_NM_THz / np.asarray(wavelength_nm, dtype=float)


def wavelength_nm(frequency_thz_value):
    import numpy as np
    return C_NM_THz / np.asarray(frequency_thz_value, dtype=float)


def style(axis):
    axis.set_facecolor(AX_BG)
    axis.grid(True, color=GRID, linewidth=0.50, alpha=0.48)
    axis.tick_params(colors=TEXT, labelsize=8.5)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.85)


def full_limits(values):
    import numpy as np
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    padding = 0.06 * (high - low) if high > low else max(abs(high) * 0.08, 1.0e-8)
    return low - padding, high + padding


def build_plot(nist_lines, selected_lines, observed_nm, flux, source_path, source_status):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    observed_low = REST_LOW_NM * STRETCH
    observed_high = REST_HIGH_NM * STRETCH
    mask = (
        np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (observed_nm >= observed_low)
        & (observed_nm <= observed_high)
    )
    if int(mask.sum()) < 5:
        raise RuntimeError(
            f"Only {int(mask.sum())} JWST samples fall inside "
            f"{observed_low:.3f}-{observed_high:.3f} nm"
        )

    x_obs = observed_nm[mask]
    y_obs = flux[mask]
    order = np.argsort(x_obs)
    x_obs = x_obs[order]
    y_obs = y_obs[order]

    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(15.8, 6.7),
        facecolor=BG,
    )
    style(left)
    style(right)

    left.vlines(
        nist_lines["wavelength_nm"],
        0.0,
        nist_lines["strength_normalized"],
        color=LAB_COLOR,
        linewidth=1.0,
        alpha=0.82,
    )
    left.scatter(
        nist_lines["wavelength_nm"],
        nist_lines["strength_normalized"],
        s=24,
        color=LAB_COLOR,
        edgecolor=BG,
        linewidth=0.35,
        zorder=5,
    )

    for _, line in selected_lines.iterrows():
        left.axvline(
            float(line["wavelength_nm"]),
            color=REF_COLOR,
            linestyle=(0, (3, 4)),
            linewidth=0.75,
            alpha=0.85,
        )
        right.axvline(
            float(line["wavelength_nm"]) * STRETCH,
            color=REF_COLOR,
            linestyle=(0, (3, 4)),
            linewidth=0.75,
            alpha=0.85,
        )

    right.plot(x_obs, y_obs, color=JWST_COLOR, linewidth=0.80, alpha=0.90)
    right.scatter(
        x_obs,
        y_obs,
        s=20,
        color=POINT_COLOR,
        edgecolor=BG,
        linewidth=0.30,
        zorder=4,
    )

    left.set_xlim(REST_LOW_NM, REST_HIGH_NM)
    left.set_ylim(0.0, 1.08)
    right.set_xlim(observed_low, observed_high)
    right.set_ylim(*full_limits(y_obs))

    left.set_title("N IV LABORATORY / NIST ASD LINE DATA", fontsize=11.7, pad=13)
    right.set_title("N IV OBSERVED BAND / JWST", fontsize=11.7, pad=13)
    left.set_xlabel("Laboratory rest wavelength, nm")
    right.set_xlabel("Observed wavelength, nm")
    left.set_ylabel("NIST tabulated strength, normalized")
    right.set_ylabel("JWST flux samples")

    top_left = left.secondary_xaxis("top", functions=(frequency_thz, wavelength_nm))
    top_left.set_xlabel("Laboratory rest frequency, THz", color=TEXT, labelpad=7)
    top_left.tick_params(colors=TEXT, labelsize=8)
    top_right = right.secondary_xaxis("top", functions=(frequency_thz, wavelength_nm))
    top_right.set_xlabel("Observed frequency, THz", color=TEXT, labelpad=7)
    top_right.tick_params(colors=TEXT, labelsize=8)

    line_text = "\n".join(
        f"{row.wavelength_nm:.6f} nm  /  {frequency_thz(row.wavelength_nm):.3f} THz"
        for row in selected_lines.itertuples()
    )
    left.text(
        0.018,
        0.955,
        "NIST N IV records\n" + line_text + "\nNo synthetic broadening",
        transform=left.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=.30",
            facecolor="#07111f",
            edgecolor=LAB_COLOR,
            linewidth=0.70,
            alpha=0.95,
        ),
    )
    right.text(
        0.018,
        0.955,
        (
            f"real JWST samples = {len(x_obs)}\n"
            f"published z = {Z:.2f}\n"
            "thin red markers = NIST positions × (1+z)"
        ),
        transform=right.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=.30",
            facecolor="#07111f",
            edgecolor=JWST_COLOR,
            linewidth=0.70,
            alpha=0.95,
        ),
    )

    figure.suptitle(
        f"{VERSION} — {GALAXY}: N IV LAB DATA versus JWST OBSERVATION",
        color=TEXT,
        fontsize=15.2,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.925,
        "The left and right panels are different datasets: NIST atomic line records versus JWST flux measurements.",
        ha="center",
        color=MUTED,
        fontsize=8.9,
    )
    figure.text(
        0.5,
        0.018,
        (
            f"JWST source: {source_path.name} ({source_status}). "
            "NIST ASD supplies discrete line data, not a universal continuous laboratory profile."
        ),
        ha="center",
        color=MUTED,
        fontsize=8.0,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.84, bottom=0.12, wspace=0.16)

    png_path = PNG / f"{VERSION}_{GALAXY}_NIV_NIST_VS_JWST.png"
    figure.savefig(png_path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(figure)

    jwst_path = CSV / f"{VERSION}_{GALAXY}_NIV_JWST_WINDOW.csv"
    pd.DataFrame(
        {
            "observed_wavelength_nm": x_obs,
            "observed_frequency_THz": frequency_thz(x_obs),
            "jwst_flux": y_obs,
            "rest_equivalent_wavelength_nm": x_obs / STRETCH,
            "rest_equivalent_frequency_THz": frequency_thz(x_obs / STRETCH),
        }
    ).to_csv(jwst_path, index=False)
    return png_path, jwst_path, len(x_obs)


def main():
    for package in ["numpy", "pandas", "matplotlib", "astropy", "astroquery"]:
        need(package)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)

    source_path, source_status = locate_jwst_csv()
    observed_nm, flux, wave_column, flux_column = load_jwst(source_path)
    nist_lines, selected_lines, nist_raw_path, nist_selected_path = query_nist_niv()
    plot_path, jwst_window_path, sample_count = build_plot(
        nist_lines,
        selected_lines,
        observed_nm,
        flux,
        source_path,
        source_status,
    )

    print(f"CODE OUTPUT: {VERSION}")
    print(f"LEFT DATA            : NIST ASD N IV line records")
    print(f"RIGHT DATA           : {source_path}")
    print(f"JWST SOURCE STATUS   : {source_status}")
    print(f"REST WINDOW          : {REST_LOW_NM:.3f} to {REST_HIGH_NM:.3f} nm")
    print(f"OBSERVED WINDOW      : {REST_LOW_NM * STRETCH:.3f} to {REST_HIGH_NM * STRETCH:.3f} nm")
    print(f"JWST SAMPLES         : {sample_count}")
    print(f"WAVELENGTH COLUMN    : {wave_column}")
    print(f"FLUX COLUMN          : {flux_column}")
    print(f"PLOT PNG             : {plot_path}")
    print(f"NIST RAW CSV         : {nist_raw_path}")
    print(f"NIST SELECTED CSV    : {nist_selected_path}")
    print(f"JWST WINDOW CSV      : {jwst_window_path}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
