# JWST_0067
# Hydrogen Ly-alpha: published laboratory VUV scan versus real JWST MoM-z14 data.
# LEFT: a measured hydrogen-discharge VUV spectrum from Komppula et al. (2015),
#       cropped from the published Figure 2(a). No synthetic profile.
# RIGHT: cached JWST spectrum around the MoM-z14 Lyman-alpha break.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import importlib
import math
import subprocess
import sys

VERSION = "JWST_0067"
GALAXY = "MoM-z14"
Z = 14.44
STRETCH = 1.0 + Z
C_NM_THz = 299792.458
LYA_REST_NM = 121.567
EXPECTED_OBS_NM = LYA_REST_NM * STRETCH

LAB_ARTICLE = "Komppula et al. 2015, VUV irradiance measurement of a 2.45 GHz microwave-driven hydrogen discharge"
LAB_ARXIV = "1510.02246"
LAB_FIGURE_URLS = [
    "https://ar5iv.labs.arxiv.org/html/1510.02246/assets/x2.png",
    "https://arxiv.org/html/1510.02246/assets/x2.png",
]
LAB_FIGURE_X_MIN_NM = 80.0
LAB_FIGURE_X_MAX_NM = 250.0
LAB_ZOOM_LOW_NM = 115.0
LAB_ZOOM_HIGH_NM = 130.0

JWST_WINDOW_LOW_NM = 1780.0
JWST_WINDOW_HIGH_NM = 1980.0
BREAK_SEARCH_HALF_WIDTH_NM = 55.0
BREAK_SIDE_WIDTH_NM = 20.0

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
YELLOW = "#ffd84d"
ORANGE = "#ff9d2e"
RED = "#ff5a66"
POINT = "#d9edf7"


def need(package, pip_name=None):
    try:
        importlib.import_module(package)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or package]
        )


def frequency_thz(wavelength_nm):
    import numpy as np
    return C_NM_THz / np.asarray(wavelength_nm, dtype=float)


def wavelength_nm(frequency_thz_value):
    import numpy as np
    return C_NM_THz / np.asarray(frequency_thz_value, dtype=float)


def newest(pattern):
    matches = sorted(CSV.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


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
            status = (
                "coordinate-matched JWST extraction"
                if "EXACT_JWST" in filename
                else "cached JWST/MAST spectrum"
            )
            return path, status

    exact = newest("JWST_*_MoM-z14_EXACT_JWST.csv")
    if exact is not None:
        return exact, "coordinate-matched JWST extraction"

    fallback = newest("JWST_*_REAL_RAW_SPECTRUM.csv")
    if fallback is not None:
        return fallback, "cached JWST/MAST spectrum"

    raise FileNotFoundError(
        "No cached JWST spectrum CSV found in /content/JWST_OUTPUT/CSV."
    )


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
        raise RuntimeError(
            f"Could not identify wavelength/flux columns in {path.name}. "
            f"Columns={list(frame.columns)}"
        )

    wavelength = frame[wave_column].to_numpy(float)
    flux = frame[flux_column].to_numpy(float)
    finite = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
    wavelength = wavelength[finite]
    flux = flux[finite]

    median = float(np.nanmedian(wavelength))
    wave_name = str(wave_column).lower()
    if "_um" in wave_name or median < 20.0:
        observed_nm = wavelength * 1000.0
    elif "_nm" in wave_name or median < 10000.0:
        observed_nm = wavelength
    else:
        observed_nm = wavelength / 10.0

    order = np.argsort(observed_nm)
    return observed_nm[order], flux[order], str(wave_column), str(flux_column)


def download_lab_figure():
    import requests
    from PIL import Image

    DATA.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in LAB_FIGURE_URLS:
        try:
            response = requests.get(
                url,
                timeout=(15, 90),
                headers={"User-Agent": "Mozilla/5.0 JWST-science-workflow"},
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            if image.width < 300 or image.height < 200:
                raise RuntimeError(f"Downloaded image is unexpectedly small: {image.size}")
            raw_path = DATA / f"{VERSION}_KOMPPULA_FIGURE2.png"
            image.save(raw_path)
            return image, raw_path, url
        except Exception as exc:
            errors.append(f"{url} -> {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not download the published laboratory spectrum figure.\n"
        + "\n".join(errors)
    )


def longest_dark_run(binary_row):
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 0
    for index, value in enumerate(binary_row):
        if value:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
        else:
            current_length = 0
    return best_start, best_length


def detect_left_plot_box(image):
    import numpy as np

    rgb = np.asarray(image)
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    height, width = gray.shape
    panel = gray[:, : int(width * 0.58)]
    ph, pw = panel.shape

    dark = panel < 105
    row_scores = []
    for row in range(int(ph * 0.10), int(ph * 0.93)):
        start, run = longest_dark_run(dark[row, :])
        row_scores.append((run, row, start))
    horizontal = [
        item for item in row_scores
        if item[0] >= int(pw * 0.33)
    ]

    col_scores = []
    for col in range(int(pw * 0.03), int(pw * 0.60)):
        start, run = longest_dark_run(dark[:, col])
        col_scores.append((run, col, start))
    vertical = [
        item for item in col_scores
        if item[0] >= int(ph * 0.30)
    ]

    if horizontal and vertical:
        bottom_run, bottom_y, bottom_x = max(horizontal, key=lambda item: item[1])
        left_run, left_x, top_y = min(
            sorted(vertical, key=lambda item: (-item[0], item[1]))[:8],
            key=lambda item: item[1],
        )
        x0 = max(0, left_x)
        x1 = min(pw, bottom_x + bottom_run)
        y0 = max(0, top_y)
        y1 = min(ph, bottom_y)
        if (
            x1 - x0 >= int(pw * 0.25)
            and y1 - y0 >= int(ph * 0.25)
        ):
            return (x0, y0, x1, y1)

    return (
        int(pw * 0.12),
        int(ph * 0.10),
        int(pw * 0.96),
        int(ph * 0.88),
    )


def crop_lab_scan(image):
    x0, y0, x1, y1 = detect_left_plot_box(image)
    plot = image.crop((x0, y0, x1, y1))

    width, height = plot.size
    fraction_low = (
        (LAB_ZOOM_LOW_NM - LAB_FIGURE_X_MIN_NM)
        / (LAB_FIGURE_X_MAX_NM - LAB_FIGURE_X_MIN_NM)
    )
    fraction_high = (
        (LAB_ZOOM_HIGH_NM - LAB_FIGURE_X_MIN_NM)
        / (LAB_FIGURE_X_MAX_NM - LAB_FIGURE_X_MIN_NM)
    )
    px0 = max(0, int(round(width * fraction_low)))
    px1 = min(width, int(round(width * fraction_high)))
    if px1 - px0 < 20:
        raise RuntimeError("Laboratory Ly-alpha crop is too narrow after calibration.")

    zoom = plot.crop((px0, 0, px1, height))
    crop_path = PNG / f"{VERSION}_PUBLISHED_LAB_LYA_SCAN_CROP.png"
    zoom.save(crop_path)
    return zoom, crop_path, (x0, y0, x1, y1)


def rolling_median(values, window):
    import pandas as pd

    series = pd.Series(values)
    return series.rolling(
        window=max(3, int(window)),
        center=True,
        min_periods=1,
    ).median().to_numpy(float)


def measure_break(observed_nm, flux):
    import numpy as np

    window = (
        np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (observed_nm >= JWST_WINDOW_LOW_NM)
        & (observed_nm <= JWST_WINDOW_HIGH_NM)
    )
    x = observed_nm[window]
    y = flux[window]
    if len(x) < 20:
        raise RuntimeError(
            f"Only {len(x)} JWST samples lie in the Lyman-alpha break window."
        )

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    smooth = rolling_median(y, max(5, len(y) // 35))

    candidates = x[
        (x >= EXPECTED_OBS_NM - BREAK_SEARCH_HALF_WIDTH_NM)
        & (x <= EXPECTED_OBS_NM + BREAK_SEARCH_HALF_WIDTH_NM)
    ]
    if len(candidates) == 0:
        raise RuntimeError("No JWST samples near the expected Lyman-alpha break.")

    best = None
    for center in candidates:
        blue = (
            (x >= center - BREAK_SIDE_WIDTH_NM)
            & (x < center - 2.0)
        )
        red = (
            (x > center + 2.0)
            & (x <= center + BREAK_SIDE_WIDTH_NM)
        )
        if int(blue.sum()) < 4 or int(red.sum()) < 4:
            continue

        blue_level = float(np.nanmedian(smooth[blue]))
        red_level = float(np.nanmedian(smooth[red]))
        difference = red_level - blue_level
        local = smooth[blue | red]
        scale = float(np.nanmedian(np.abs(local - np.nanmedian(local)))) * 1.4826
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(local))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        score = difference / scale

        proximity_penalty = abs(center - EXPECTED_OBS_NM) / BREAK_SEARCH_HALF_WIDTH_NM
        objective = score - 0.20 * proximity_penalty
        item = {
            "center_nm": float(center),
            "score": float(score),
            "objective": float(objective),
            "blue_level": blue_level,
            "red_level": red_level,
        }
        if best is None or item["objective"] > best["objective"]:
            best = item

    if best is None:
        raise RuntimeError("Could not determine a stable Lyman-alpha break position.")

    return x, y, smooth, best


def style_axis(axis):
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
    padding = 0.055 * (high - low) if high > low else max(abs(high) * 0.08, 1.0e-8)
    return low - padding, high + padding


def make_plot(
    lab_zoom,
    lab_crop_path,
    lab_figure_url,
    source_path,
    source_status,
    observed_nm,
    flux,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    x, y, smooth, break_result = measure_break(observed_nm, flux)
    measured_break_nm = break_result["center_nm"]

    figure, (left, right) = plt.subplots(
        1,
        2,
        figsize=(16.0, 6.8),
        facecolor=BG,
    )

    style_axis(left)
    style_axis(right)

    left.imshow(
        np.asarray(lab_zoom),
        extent=[LAB_ZOOM_LOW_NM, LAB_ZOOM_HIGH_NM, 0.0, 1.0],
        origin="upper",
        aspect="auto",
        interpolation="nearest",
    )
    left.axvline(
        LYA_REST_NM,
        color=YELLOW,
        linestyle=(0, (3, 5)),
        linewidth=0.65,
        alpha=0.72,
        zorder=6,
    )
    left.set_xlim(LAB_ZOOM_LOW_NM, LAB_ZOOM_HIGH_NM)
    left.set_ylim(0.0, 1.0)
    left.set_yticks([])
    left.set_title("PUBLISHED LAB HYDROGEN VUV SCAN — Lyα", fontsize=11.5, pad=13)
    left.set_xlabel("Laboratory wavelength, nm")
    left.set_ylabel("Published spectrometer signal (figure scale)")

    top_left = left.secondary_xaxis(
        "top",
        functions=(frequency_thz, wavelength_nm),
    )
    top_left.set_xlabel("Laboratory frequency, THz", color=TEXT, labelpad=7)
    top_left.tick_params(colors=TEXT, labelsize=8)

    right.plot(x, y, color=ORANGE, linewidth=0.72, alpha=0.68, label="raw JWST samples")
    right.scatter(
        x,
        y,
        s=17,
        color=POINT,
        edgecolor=BG,
        linewidth=0.25,
        alpha=0.92,
        zorder=4,
    )
    right.plot(
        x,
        smooth,
        color="#ffffff",
        linewidth=1.15,
        alpha=0.82,
        label="rolling median",
        zorder=5,
    )
    right.axvline(
        measured_break_nm,
        color=RED,
        linestyle=(0, (3, 5)),
        linewidth=0.65,
        alpha=0.72,
        zorder=6,
    )
    right.set_xlim(JWST_WINDOW_LOW_NM, JWST_WINDOW_HIGH_NM)
    right.set_ylim(*full_limits(y))
    right.set_title("MoM-z14 JWST HYDROGEN SIGNATURE — Lyα BREAK", fontsize=11.5, pad=13)
    right.set_xlabel("Observed wavelength, nm")
    right.set_ylabel("JWST flux samples")

    top_right = right.secondary_xaxis(
        "top",
        functions=(frequency_thz, wavelength_nm),
    )
    top_right.set_xlabel("Observed frequency, THz", color=TEXT, labelpad=7)
    top_right.tick_params(colors=TEXT, labelsize=8)

    left.text(
        0.018,
        0.955,
        (
            "published hydrogen-discharge scan\n"
            f"rest marker = {LYA_REST_NM:.6f} nm\n"
            f"rest frequency = {frequency_thz(LYA_REST_NM):.3f} THz"
        ),
        transform=left.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=.30",
            facecolor="#07111f",
            edgecolor=YELLOW,
            linewidth=0.65,
            alpha=0.94,
        ),
    )
    right.text(
        0.018,
        0.955,
        (
            "measured change-point in JWST spectrum\n"
            f"break marker = {measured_break_nm:.3f} nm\n"
            f"observed frequency = {frequency_thz(measured_break_nm):.3f} THz\n"
            f"z from break = {measured_break_nm / LYA_REST_NM - 1.0:.5f}"
        ),
        transform=right.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=7.7,
        bbox=dict(
            boxstyle="round,pad=.30",
            facecolor="#07111f",
            edgecolor=RED,
            linewidth=0.65,
            alpha=0.94,
        ),
    )

    legend = right.legend(
        loc="lower right",
        fontsize=7.2,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.94,
    )
    for text in legend.get_texts():
        text.set_color(TEXT)

    figure.suptitle(
        f"{VERSION} — HYDROGEN Lyα: LABORATORY SCAN versus JWST OBSERVATION",
        color=TEXT,
        fontsize=15.1,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.925,
        (
            "Different datasets. Left: measured microwave-hydrogen-discharge VUV spectrum. "
            "Right: MoM-z14 JWST spectrum. MoM-z14 is confirmed by a sharp Lyα break, not a clean hydrogen emission peak."
        ),
        ha="center",
        color=MUTED,
        fontsize=8.7,
    )
    figure.text(
        0.5,
        0.018,
        (
            f"Lab source: {LAB_ARTICLE}, arXiv:{LAB_ARXIV}, Figure 2(a). "
            f"JWST source: {source_path.name} ({source_status})."
        ),
        ha="center",
        color=MUTED,
        fontsize=7.8,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.84,
        bottom=0.12,
        wspace=0.16,
    )

    plot_path = PNG / f"{VERSION}_{GALAXY}_HYDROGEN_LYA_LAB_VS_JWST.png"
    figure.savefig(plot_path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(figure)

    jwst_csv = CSV / f"{VERSION}_{GALAXY}_LYA_BREAK_WINDOW.csv"
    pd.DataFrame(
        {
            "observed_wavelength_nm": x,
            "observed_frequency_THz": frequency_thz(x),
            "jwst_flux": y,
            "rolling_median_flux": smooth,
            "measured_break_nm": measured_break_nm,
            "expected_from_published_z_nm": EXPECTED_OBS_NM,
            "rest_lya_nm": LYA_REST_NM,
        }
    ).to_csv(jwst_csv, index=False)

    audit_csv = CSV / f"{VERSION}_{GALAXY}_HYDROGEN_AUDIT.csv"
    pd.DataFrame(
        [
            {
                "lab_article": LAB_ARTICLE,
                "lab_arxiv": LAB_ARXIV,
                "lab_figure_url": lab_figure_url,
                "lab_crop_png": str(lab_crop_path),
                "rest_lya_nm": LYA_REST_NM,
                "rest_frequency_THz": float(frequency_thz(LYA_REST_NM)),
                "published_redshift_z": Z,
                "expected_observed_nm": EXPECTED_OBS_NM,
                "measured_break_nm": measured_break_nm,
                "measured_break_frequency_THz": float(frequency_thz(measured_break_nm)),
                "redshift_from_measured_break": measured_break_nm / LYA_REST_NM - 1.0,
                "break_score": break_result["score"],
                "blue_level": break_result["blue_level"],
                "red_level": break_result["red_level"],
                "jwst_source": str(source_path),
                "jwst_source_status": source_status,
            }
        ]
    ).to_csv(audit_csv, index=False)

    return plot_path, jwst_csv, audit_csv, break_result


def main():
    for package, pip_name in [
        ("numpy", None),
        ("pandas", None),
        ("matplotlib", None),
        ("requests", None),
        ("PIL", "Pillow"),
    ]:
        need(package, pip_name)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"CODE OUTPUT: {VERSION}")
    print("STEP 1/4 | Locate cached JWST spectrum")
    source_path, source_status = locate_jwst_csv()
    observed_nm, flux, wave_column, flux_column = load_jwst(source_path)

    print("STEP 2/4 | Download published laboratory hydrogen VUV spectrum")
    lab_image, lab_raw_path, lab_figure_url = download_lab_figure()

    print("STEP 3/4 | Crop the published Ly-alpha laboratory scan")
    lab_zoom, lab_crop_path, plot_box = crop_lab_scan(lab_image)

    print("STEP 4/4 | Measure JWST Ly-alpha break and render comparison")
    plot_path, jwst_csv, audit_csv, break_result = make_plot(
        lab_zoom,
        lab_crop_path,
        lab_figure_url,
        source_path,
        source_status,
        observed_nm,
        flux,
    )

    print()
    print(f"LAB DATA             : published measured VUV scan, Figure 2(a)")
    print(f"LAB ARTICLE          : {LAB_ARTICLE}")
    print(f"LAB RAW FIGURE       : {lab_raw_path}")
    print(f"LAB CROP PNG         : {lab_crop_path}")
    print(f"LAB PLOT BOX PX      : {plot_box}")
    print(f"REST Ly-alpha        : {LYA_REST_NM:.6f} nm")
    print(f"REST FREQUENCY       : {frequency_thz(LYA_REST_NM):.6f} THz")
    print(f"JWST SOURCE          : {source_path}")
    print(f"JWST SOURCE STATUS   : {source_status}")
    print(f"WAVELENGTH COLUMN    : {wave_column}")
    print(f"FLUX COLUMN          : {flux_column}")
    print(f"EXPECTED BREAK       : {EXPECTED_OBS_NM:.6f} nm")
    print(f"MEASURED BREAK       : {break_result['center_nm']:.6f} nm")
    print(f"BREAK REDSHIFT       : {break_result['center_nm'] / LYA_REST_NM - 1.0:.6f}")
    print(f"BREAK SCORE          : {break_result['score']:.6f}")
    print(f"PLOT PNG             : {plot_path}")
    print(f"JWST WINDOW CSV      : {jwst_csv}")
    print(f"AUDIT CSV            : {audit_csv}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
