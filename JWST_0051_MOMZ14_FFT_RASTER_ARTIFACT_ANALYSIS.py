#!/usr/bin/env python3
"""
JWST_0051_MOMZ14_FFT_RASTER_ARTIFACT_ANALYSIS.py

Analyze calibrated JWST/NIRCam MoM-z14 FITS cutouts for directional periodic
structure that could be consistent with rastering, resampling, striping, or
other image-processing artifacts.

The analysis uses native science mosaics, robust plane removal, a broad
high-pass residual, a two-dimensional Hann window, 2-D FFT power spectra,
angular power profiles, and repeated-frequency peak extraction.

A directional FFT peak is evidence of periodic structure, not by itself proof
of an instrumental artifact. Real astronomical structure, PSF diffraction,
mosaic edges, and interpolation can also create anisotropic power.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "scipy": "scipy",
        "astropy": "astropy",
        "requests": "requests",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter, maximum_filter

VERSION = "JWST_0051"
BASE_NAME = "JWST_0048_MOMZ14_4FILTER_FITS.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
BASE_PATH = Path("/content") / BASE_NAME
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
FITS_DIR = ROOT / "FITS"
for directory in (PNG_DIR, CSV_DIR, FITS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOW_CUTOFF_CYC_PER_PIX = 0.025
HIGH_CUTOFF_CYC_PER_PIX = 0.46
ANGLE_BIN_DEG = 1.0
PEAK_NEIGHBORHOOD_PIX = 9
TOP_PEAKS_PER_FILTER = 12
HIGHPASS_SIGMA_PIX = 6.0


def load_base_module():
    urllib.request.urlretrieve(BASE_URL, BASE_PATH)
    spec = importlib.util.spec_from_file_location("jwst0048", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load JWST_0048 base module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    module.FITS_DIR = FITS_DIR
    return module


def robust_fill(data: np.ndarray) -> np.ndarray:
    image = np.asarray(data, dtype=float).copy()
    finite = np.isfinite(image)
    if not finite.any():
        raise ValueError("Image contains no finite pixels")
    median = float(np.nanmedian(image[finite]))
    image[~finite] = median
    lo, hi = np.nanpercentile(image, [0.2, 99.8])
    return np.clip(image, lo, hi)


def remove_plane(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ny, nx = image.shape
    yy, xx = np.indices(image.shape, dtype=float)
    x = (xx - (nx - 1) / 2.0) / max(nx, 1)
    y = (yy - (ny - 1) / 2.0) / max(ny, 1)
    design = np.column_stack((np.ones(image.size), x.ravel(), y.ravel()))
    coeff, *_ = np.linalg.lstsq(design, image.ravel(), rcond=None)
    plane = (design @ coeff).reshape(image.shape)
    return image - plane, plane


def preprocess(data: np.ndarray) -> dict:
    image = robust_fill(data)
    detrended, plane = remove_plane(image)
    smooth = gaussian_filter(detrended, sigma=HIGHPASS_SIGMA_PIX, mode="reflect")
    residual = detrended - smooth
    ny, nx = residual.shape
    window = np.outer(np.hanning(ny), np.hanning(nx))
    windowed = residual * window
    scale = float(np.sqrt(np.mean(windowed ** 2)))
    if scale > 0:
        windowed /= scale
    return {
        "image": image,
        "plane": plane,
        "detrended": detrended,
        "residual": residual,
        "windowed": windowed,
    }


def circular_mean_180(angle_deg: np.ndarray, weight: np.ndarray) -> float:
    theta = np.deg2rad(2.0 * angle_deg)
    z = np.sum(weight * np.exp(1j * theta))
    if abs(z) == 0:
        return float("nan")
    return float((np.rad2deg(np.angle(z)) / 2.0) % 180.0)


def angular_distance_180(a: np.ndarray, b: float) -> np.ndarray:
    return np.abs((a - b + 90.0) % 180.0 - 90.0)


def fft_analysis(windowed: np.ndarray, pixel_scale: float) -> dict:
    ny, nx = windowed.shape
    fft = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(fft) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    fxx, fyy = np.meshgrid(fx, fy)
    radius = np.hypot(fxx, fyy)
    angle = np.rad2deg(np.arctan2(fyy, fxx)) % 180.0

    annulus = (
        np.isfinite(power)
        & (radius >= LOW_CUTOFF_CYC_PER_PIX)
        & (radius <= HIGH_CUTOFF_CYC_PER_PIX)
    )
    edges = np.arange(0.0, 180.0 + ANGLE_BIN_DEG, ANGLE_BIN_DEG)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_index = np.digitize(angle[annulus], edges) - 1
    weighted = np.bincount(bin_index, weights=power[annulus], minlength=len(centers))
    counts = np.bincount(bin_index, minlength=len(centers))
    angular_power = weighted / np.maximum(counts, 1)
    positive = angular_power[angular_power > 0]
    median_power = float(np.median(positive))
    normalized_angular = angular_power / max(median_power, np.finfo(float).tiny)
    dominant_index = int(np.argmax(normalized_angular))
    dominant_fft_angle = float(centers[dominant_index])
    inferred_stripe_angle = (dominant_fft_angle + 90.0) % 180.0
    anisotropy = float(normalized_angular[dominant_index])

    horizontal_band_mask = angular_distance_180(centers, 90.0) <= 3.0
    vertical_band_mask = angular_distance_180(centers, 0.0) <= 3.0
    horizontal_score = float(np.mean(normalized_angular[horizontal_band_mask]))
    vertical_score = float(np.mean(normalized_angular[vertical_band_mask]))

    local_max = power == maximum_filter(power, size=PEAK_NEIGHBORHOOD_PIX, mode="nearest")
    candidate = annulus & local_max
    threshold = float(np.percentile(power[annulus], 99.2))
    candidate &= power >= threshold
    yy, xx = np.where(candidate)
    if yy.size:
        order = np.argsort(power[yy, xx])[::-1][:TOP_PEAKS_PER_FILTER]
        yy, xx = yy[order], xx[order]
    peaks = []
    annulus_median = max(float(np.median(power[annulus])), np.finfo(float).tiny)
    for rank, (iy, ix) in enumerate(zip(yy, xx), start=1):
        frequency = float(radius[iy, ix])
        fft_angle = float(angle[iy, ix])
        period_pix = float(1.0 / frequency) if frequency > 0 else float("inf")
        peaks.append({
            "peak_rank": rank,
            "fx_cycles_per_pix": float(fxx[iy, ix]),
            "fy_cycles_per_pix": float(fyy[iy, ix]),
            "frequency_cycles_per_pix": frequency,
            "period_pix": period_pix,
            "period_arcsec": period_pix * pixel_scale,
            "fft_angle_deg": fft_angle,
            "inferred_stripe_angle_deg": (fft_angle + 90.0) % 180.0,
            "power": float(power[iy, ix]),
            "power_over_annulus_median": float(power[iy, ix] / annulus_median),
        })

    strongest_period_pix = peaks[0]["period_pix"] if peaks else float("nan")
    strongest_period_arcsec = peaks[0]["period_arcsec"] if peaks else float("nan")
    return {
        "fft": fft,
        "power": power,
        "fx": fx,
        "fy": fy,
        "radius": radius,
        "angle_centers": centers,
        "angular_power_norm": normalized_angular,
        "dominant_fft_angle_deg": dominant_fft_angle,
        "inferred_stripe_angle_deg": inferred_stripe_angle,
        "anisotropy_score": anisotropy,
        "horizontal_band_score": horizontal_score,
        "vertical_band_score": vertical_score,
        "strongest_period_pix": strongest_period_pix,
        "strongest_period_arcsec": strongest_period_arcsec,
        "peaks": peaks,
    }


def classification(score: float) -> str:
    if score >= 5.0:
        return "strong directional periodicity"
    if score >= 3.0:
        return "moderate directional periodicity"
    if score >= 2.0:
        return "weak directional periodicity"
    return "no strong directional periodicity"


def robust_limits(data: np.ndarray, low: float = 1.0, high: float = 99.0) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [low, high])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def make_dashboard(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 4, figsize=(21, 14), constrained_layout=True)

    for col, record in enumerate(records):
        entry = record["entry"]
        prep = record["prep"]
        result = record["fft_result"]
        scale = record["pixel_scale_arcsec"]
        image = prep["image"]
        residual = prep["residual"]
        power = result["power"]

        vmin, vmax = robust_limits(image, 1.0, 99.5)
        ax = axes[0, col]
        ax.imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(f"{entry['name']}  |  {entry['lambda_um']:.2f} µm", fontsize=13)
        ax.set_xlabel(f"calibrated mosaic | {scale:.4f} arcsec/pix", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        rv = max(abs(np.percentile(residual, 1.0)), abs(np.percentile(residual, 99.0)))
        ax = axes[1, col]
        ax.imshow(residual, origin="lower", cmap="coolwarm", vmin=-rv, vmax=rv,
                  interpolation="nearest")
        ax.set_title("plane-removed high-pass residual", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[2, col]
        positive = power[power > 0]
        lo = max(float(np.percentile(positive, 30.0)), np.finfo(float).tiny)
        hi = float(np.percentile(positive, 99.95))
        extent = [result["fx"][0], result["fx"][-1], result["fy"][0], result["fy"][-1]]
        ax.imshow(power, origin="lower", cmap="magma", norm=LogNorm(vmin=lo, vmax=hi),
                  extent=extent, interpolation="nearest", aspect="equal")
        ax.axhline(0, linewidth=0.5, alpha=0.45)
        ax.axvline(0, linewidth=0.5, alpha=0.45)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel("fx [cycles/pixel]", fontsize=9)
        ax.set_ylabel("fy [cycles/pixel]", fontsize=9)
        ax.set_title(
            f"FFT power | stripe ≈ {result['inferred_stripe_angle_deg']:.1f}°\n"
            f"anisotropy {result['anisotropy_score']:.2f}× median",
            fontsize=11,
        )

    fig.suptitle(
        "MoM-z14 calibrated JWST/NIRCam mosaics — FFT raster/striping diagnostic\n"
        "A narrow off-center FFT peak indicates periodic structure perpendicular to the inferred stripe direction",
        fontsize=18,
    )
    path = PNG_DIR / f"{VERSION}_MOMZ14_FFT_DASHBOARD.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return path


def make_angular_plot(records: list[dict]) -> Path:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(13, 7), constrained_layout=True)
    for record in records:
        result = record["fft_result"]
        name = record["entry"]["name"]
        ax.plot(result["angle_centers"], result["angular_power_norm"], linewidth=1.7,
                label=f"{name}: stripe {result['inferred_stripe_angle_deg']:.1f}°")
    ax.axhline(1.0, linewidth=0.8, linestyle="--", alpha=0.7, label="angular median")
    ax.set_xlim(0, 180)
    ax.set_xlabel("FFT direction [deg] — inferred image stripe is +90°", fontsize=11)
    ax.set_ylabel("Mean annular FFT power / angular median", fontsize=11)
    ax.set_title("Directional periodicity comparison across four independent NIRCam filters", fontsize=15)
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    path = PNG_DIR / f"{VERSION}_MOMZ14_FFT_ANGULAR_POWER.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return path


def make_summary_table(summary: pd.DataFrame) -> Path:
    display_cols = [
        "filter", "lambda_um", "pixel_scale_arcsec_per_pix", "anisotropy_score",
        "inferred_stripe_angle_deg", "strongest_period_pix", "strongest_period_arcsec",
        "classification",
    ]
    shown = summary[display_cols].copy()
    shown["lambda_um"] = shown["lambda_um"].map(lambda x: f"{x:.2f}")
    shown["pixel_scale_arcsec_per_pix"] = shown["pixel_scale_arcsec_per_pix"].map(lambda x: f"{x:.5f}")
    for col in ("anisotropy_score", "inferred_stripe_angle_deg", "strongest_period_pix",
                "strongest_period_arcsec"):
        shown[col] = shown[col].map(lambda x: "—" if not np.isfinite(x) else f"{x:.3f}")
    shown.columns = [
        "Filter", "λ [µm]", "Scale [arcsec/pix]", "Anisotropy [×]",
        "Stripe angle [deg]", "Period [pix]", "Period [arcsec]", "FFT assessment",
    ]

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(18, 4.8), constrained_layout=True)
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns,
                     cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#526577")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#18344f")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#111923" if row % 2 else "#17212c")
            cell.set_text_props(color="#edf3f8")
    ax.set_title(
        "MoM-z14 FFT artifact-screening summary\n"
        "Directional periodicity is diagnostic evidence, not automatic proof of detector rastering",
        fontsize=16, pad=18,
    )
    path = PNG_DIR / f"{VERSION}_MOMZ14_FFT_SUMMARY_TABLE.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return path


def main() -> None:
    base = load_base_module()
    session = base.build_session()
    records = []
    summary_rows = []
    peak_rows = []

    for entry in base.FILTERS:
        path, source_url, byte_count = base.download_channel(session, entry)
        data, header = base.load_image(path)
        pixel_scale = base.estimate_pixel_scale_arcsec(header)
        prep = preprocess(data)
        result = fft_analysis(prep["windowed"], pixel_scale)
        assessment = classification(result["anisotropy_score"])

        record = {
            "entry": entry,
            "path": path,
            "source_url": source_url,
            "download_bytes": byte_count,
            "pixel_scale_arcsec": pixel_scale,
            "prep": prep,
            "fft_result": result,
        }
        records.append(record)
        summary_rows.append({
            "filter": entry["name"],
            "lambda_um": entry["lambda_um"],
            "image_nx_pix": data.shape[1],
            "image_ny_pix": data.shape[0],
            "pixel_scale_arcsec_per_pix": pixel_scale,
            "anisotropy_score": result["anisotropy_score"],
            "dominant_fft_angle_deg": result["dominant_fft_angle_deg"],
            "inferred_stripe_angle_deg": result["inferred_stripe_angle_deg"],
            "horizontal_band_score": result["horizontal_band_score"],
            "vertical_band_score": result["vertical_band_score"],
            "strongest_period_pix": result["strongest_period_pix"],
            "strongest_period_arcsec": result["strongest_period_arcsec"],
            "classification": assessment,
            "fits_path": str(path),
            "source_url": source_url,
        })
        for peak in result["peaks"]:
            peak_rows.append({"filter": entry["name"], **peak})

    summary = pd.DataFrame(summary_rows)
    peaks = pd.DataFrame(peak_rows)
    angle_weights = np.maximum(summary["anisotropy_score"].to_numpy() - 1.0, 0.01)
    consensus_angle = circular_mean_180(
        summary["inferred_stripe_angle_deg"].to_numpy(), angle_weights
    )
    deviations = angular_distance_180(
        summary["inferred_stripe_angle_deg"].to_numpy(), consensus_angle
    )
    angle_spread = float(np.sqrt(np.average(deviations ** 2, weights=angle_weights)))
    repeated_direction = bool(angle_spread <= 12.0 and summary["anisotropy_score"].median() >= 2.0)

    dashboard = make_dashboard(records)
    angular_plot = make_angular_plot(records)
    table_png = make_summary_table(summary)
    summary_csv = CSV_DIR / f"{VERSION}_MOMZ14_FFT_SUMMARY.csv"
    peaks_csv = CSV_DIR / f"{VERSION}_MOMZ14_FFT_PEAKS.csv"
    summary.to_csv(summary_csv, index=False)
    peaks.to_csv(peaks_csv, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print("Target       MoM-z14 calibrated NIRCam FITS mosaics")
    print("Method       plane removal + broad high-pass + Hann window + 2-D FFT")
    print("Filter   Anisotropy   Stripe angle   Period pix   Assessment")
    for row in summary.itertuples(index=False):
        print(f"{row.filter:<7} {row.anisotropy_score:>9.3f}x   "
              f"{row.inferred_stripe_angle_deg:>9.3f} deg   "
              f"{row.strongest_period_pix:>9.3f}   {row.classification}")
    print(f"Consensus    {consensus_angle:.3f} deg inferred stripe direction")
    print(f"Angle spread {angle_spread:.3f} deg RMS across filters")
    print(f"Repeated     {'YES — same directional signature recurs' if repeated_direction else 'NO — not consistently repeated across filters'}")
    print("Interpretation FFT periodicity is evidence only; inspect FITS provenance, PSF, drizzle, and detector orientation before attribution.")
    print(f"Plot PNG     {dashboard}")
    print(f"Angular PNG  {angular_plot}")
    print(f"Table PNG    {table_png}")
    print(f"Summary CSV  {summary_csv}")
    print(f"Peaks CSV    {peaks_csv}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
