# JWST_0014
# Audit: JWST detector array, field-of-view, and diffraction sampling dashboard.
# Matplotlib only. No AI images. No FITS/image downloads.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0014"
PROJECT = "JWST DETECTOR FOV AND SAMPLING DASHBOARD"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OUTPUT_CSV = OUTPUT_DIR / "CSV"

PRIMARY_MIRROR_DIAMETER_M = 6.5
ARCSEC_PER_RAD = 206265.0

# Curated engineering-level instrument values for visualization.
# Pixel scales and field sizes are rounded/representative, not calibration products.
DETECTOR_GROUPS = [
    {
        "group": "NIRCam SW",
        "instrument": "NIRCam",
        "role": "science imaging",
        "detector_type": "HgCdTe",
        "arrays": 8,
        "nx": 2048,
        "ny": 2048,
        "pixel_scale_arcsec": 0.031,
        "lambda_min_um": 0.60,
        "lambda_max_um": 2.30,
        "notes": "Short-wave imaging; fine sampling."
    },
    {
        "group": "NIRCam LW",
        "instrument": "NIRCam",
        "role": "science imaging / grism",
        "detector_type": "HgCdTe",
        "arrays": 2,
        "nx": 2048,
        "ny": 2048,
        "pixel_scale_arcsec": 0.063,
        "lambda_min_um": 2.40,
        "lambda_max_um": 5.00,
        "notes": "Long-wave imaging and grism spectroscopy."
    },
    {
        "group": "NIRISS",
        "instrument": "NIRISS",
        "role": "science imaging / slitless spectroscopy",
        "detector_type": "HgCdTe",
        "arrays": 1,
        "nx": 2048,
        "ny": 2048,
        "pixel_scale_arcsec": 0.065,
        "lambda_min_um": 0.80,
        "lambda_max_um": 5.00,
        "notes": "Imaging, WFSS, SOSS, AMI modes."
    },
    {
        "group": "NIRSpec",
        "instrument": "NIRSpec",
        "role": "science spectroscopy",
        "detector_type": "HgCdTe",
        "arrays": 2,
        "nx": 2048,
        "ny": 2048,
        "pixel_scale_arcsec": None,
        "lambda_min_um": 0.60,
        "lambda_max_um": 5.30,
        "notes": "Two 2D detector arrays for spectra; not a direct imager."
    },
    {
        "group": "MIRI imager/LRS",
        "instrument": "MIRI",
        "role": "science imaging / low-resolution spectroscopy",
        "detector_type": "Si:As IBC",
        "arrays": 1,
        "nx": 1024,
        "ny": 1024,
        "pixel_scale_arcsec": 0.110,
        "lambda_min_um": 5.60,
        "lambda_max_um": 25.50,
        "notes": "Mid-infrared imager plus LRS detector."
    },
    {
        "group": "MIRI MRS",
        "instrument": "MIRI",
        "role": "science IFU spectroscopy",
        "detector_type": "Si:As IBC",
        "arrays": 2,
        "nx": 1024,
        "ny": 1024,
        "pixel_scale_arcsec": None,
        "lambda_min_um": 4.90,
        "lambda_max_um": 27.90,
        "notes": "Medium-resolution IFU spectroscopy."
    },
    {
        "group": "FGS",
        "instrument": "FGS",
        "role": "guidance / acquisition",
        "detector_type": "HgCdTe",
        "arrays": 2,
        "nx": 2048,
        "ny": 2048,
        "pixel_scale_arcsec": 0.069,
        "lambda_min_um": 0.80,
        "lambda_max_um": 5.00,
        "notes": "Guide-camera system, not the main science camera."
    },
]

FOV_MODES = [
    {"mode": "NIRSpec MSA", "instrument": "NIRSpec", "area_arcmin2": 12.24, "width_arcmin": 3.60, "height_arcmin": 3.40, "notes": "Approximate microshutter field."},
    {"mode": "NIRCam imaging", "instrument": "NIRCam", "area_arcmin2": 9.68, "width_arcmin": 4.40, "height_arcmin": 2.20, "notes": "Two-module footprint estimate; detector gap ignored."},
    {"mode": "FGS guide fields", "instrument": "FGS", "area_arcmin2": 10.58, "width_arcmin": 4.60, "height_arcmin": 2.30, "notes": "Guide fields, not primary science image."},
    {"mode": "NIRISS imaging", "instrument": "NIRISS", "area_arcmin2": 4.84, "width_arcmin": 2.20, "height_arcmin": 2.20, "notes": "Single 2D near-IR field."},
    {"mode": "MIRI imager", "instrument": "MIRI", "area_arcmin2": 2.32, "width_arcmin": 1.88, "height_arcmin": 1.23, "notes": "Approximate 113 x 74 arcsec imaging field."},
    {"mode": "NIRSpec IFU", "instrument": "NIRSpec", "area_arcmin2": 0.0025, "width_arcmin": 0.050, "height_arcmin": 0.050, "notes": "Approximate 3 x 3 arcsec IFU field."},
    {"mode": "MIRI MRS IFU small", "instrument": "MIRI", "area_arcmin2": 0.0033, "width_arcmin": 0.053, "height_arcmin": 0.062, "notes": "Representative shortest-channel IFU field."},
    {"mode": "MIRI MRS IFU large", "instrument": "MIRI", "area_arcmin2": 0.0165, "width_arcmin": 0.128, "height_arcmin": 0.128, "notes": "Representative longest-channel IFU field."},
]


def ensure_package(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


def setup():
    ensure_package("numpy")
    ensure_package("pandas")
    ensure_package("matplotlib")
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)


def print_table(rows, headers):
    widths = []
    for i, header in enumerate(headers):
        values = [str(row[i]) for row in rows] if rows else []
        widths.append(min(max([len(str(header))] + [len(v) for v in values]), 70))
    header_line = " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            text = str(value)
            if len(text) > widths[i]:
                text = text[:widths[i] - 1] + "…"
            cells.append(text.ljust(widths[i]))
        print(" | ".join(cells))


def color_for_instrument(inst):
    return {
        "NIRCam": "#38bdf8",
        "NIRISS": "#a78bfa",
        "NIRSpec": "#22c55e",
        "MIRI": "#fb923c",
        "FGS": "#94a3b8",
    }.get(inst, "#e5e7eb")


def style_dark(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.tick_params(colors="#dbeafe")
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def build_tables():
    import pandas as pd
    det = pd.DataFrame(DETECTOR_GROUPS)
    fov = pd.DataFrame(FOV_MODES)
    det["pixels_per_array"] = det["nx"] * det["ny"]
    det["total_pixels"] = det["arrays"] * det["pixels_per_array"]
    det["total_megapixels"] = det["total_pixels"] / 1e6
    det["science_flag"] = det["instrument"].ne("FGS")
    det["lambda_center_um"] = 0.5 * (det["lambda_min_um"] + det["lambda_max_um"])
    return det, fov


def plot_detector_megapixels(det):
    import matplotlib.pyplot as plt
    import numpy as np

    df = det.sort_values("total_megapixels", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.4, 7.2))
    style_dark(fig, ax)
    y = np.arange(len(df))
    colors = [color_for_instrument(i) for i in df["instrument"]]
    ax.barh(y, df["total_megapixels"], color=colors, edgecolor="#e5e7eb", linewidth=0.65, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(df["group"], color="#f8fafc")
    ax.set_xlabel("Raw detector pixels, megapixels")
    ax.set_title("JWST detector-array pixel inventory\nsmall megapixel counts, huge photon-collecting optics, cryogenic low-noise sensors")
    for yi, row in df.iterrows():
        ax.text(row["total_megapixels"] + 0.55, yi,
                f"{row['arrays']} x {row['nx']}x{row['ny']} = {row['total_megapixels']:.2f} MP",
                color="#f8fafc", va="center", fontsize=8.6)
    ax.set_xlim(0, max(df["total_megapixels"]) * 1.36)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_DETECTOR_MEGAPIXEL_INVENTORY.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def diffraction_fwhm_arcsec(lambda_um):
    return 1.22 * lambda_um * 1e-6 / PRIMARY_MIRROR_DIAMETER_M * ARCSEC_PER_RAD


def plot_sampling_vs_diffraction(det):
    import matplotlib.pyplot as plt
    import numpy as np

    lam = np.linspace(0.6, 28.0, 800)
    fwhm = diffraction_fwhm_arcsec(lam)

    fig, ax = plt.subplots(figsize=(14.2, 8.0))
    style_dark(fig, ax)
    ax.plot(lam, fwhm, linewidth=2.2, label="JWST 1.22 λ / D diffraction scale", color="#e5e7eb")
    ax.plot(lam, 0.5 * fwhm, linewidth=1.1, linestyle="--", label="Half diffraction scale", color="#94a3b8")

    rows = det[det["pixel_scale_arcsec"].notna()].copy()
    for _, row in rows.iterrows():
        color = color_for_instrument(row["instrument"])
        ax.hlines(row["pixel_scale_arcsec"], row["lambda_min_um"], row["lambda_max_um"],
                  color=color, linewidth=5.2, alpha=0.78)
        ax.scatter([row["lambda_center_um"]], [row["pixel_scale_arcsec"]], s=62,
                   color=color, edgecolor="#f8fafc", linewidth=0.55, zorder=5)
        ax.text(row["lambda_center_um"], row["pixel_scale_arcsec"] * 1.17,
                f"{row['group']}\n{row['pixel_scale_arcsec']:.3f}\"/px",
                ha="center", va="bottom", color="#f8fafc", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#020617", edgecolor="#475569", alpha=0.78))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.55, 31.0)
    ax.set_ylim(0.012, 1.35)
    ax.set_xlabel("Wavelength, μm, log scale")
    ax.set_ylabel("Angular scale, arcsec")
    ax.set_title("Pixel scale versus diffraction scale\nwhy a few-megapixel cryogenic array can still be an extremely sharp astronomical camera")
    ax.legend(loc="lower right", facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc", fontsize=8.6)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_PIXEL_SCALE_VS_DIFFRACTION.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_fov_area(fov):
    import matplotlib.pyplot as plt
    import numpy as np

    df = fov.sort_values("area_arcmin2", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13.8, 7.6))
    style_dark(fig, ax)
    y = np.arange(len(df))
    colors = [color_for_instrument(i) for i in df["instrument"]]
    ax.barh(y, df["area_arcmin2"], color=colors, edgecolor="#e5e7eb", linewidth=0.65, alpha=0.86)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(df["mode"], color="#f8fafc")
    ax.set_xlabel("Approximate field area, square arcminutes, log scale")
    ax.set_title("JWST field-of-view scale by mode\nwide imaging fields versus tiny IFU data-cube fields")
    for yi, row in df.iterrows():
        ax.text(row["area_arcmin2"] * 1.09, yi,
                f"{row['area_arcmin2']:.4g} arcmin²",
                color="#f8fafc", va="center", fontsize=8.5)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_FIELD_OF_VIEW_AREA_LADDER.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_wavelength_coverage(det):
    import matplotlib.pyplot as plt
    import numpy as np

    df = det.sort_values(["lambda_min_um", "lambda_max_um"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(14.4, 7.7))
    style_dark(fig, ax)
    for i, row in df.iterrows():
        y = len(df) - 1 - i
        color = color_for_instrument(row["instrument"])
        ax.hlines(y, row["lambda_min_um"], row["lambda_max_um"], color=color, linewidth=7.5, alpha=0.84)
        ax.scatter([row["lambda_min_um"], row["lambda_max_um"]], [y, y], s=42,
                   color=color, edgecolor="#f8fafc", linewidth=0.55)
        ax.text(row["lambda_max_um"] * 1.035, y,
                f"{row['lambda_min_um']:.1f}-{row['lambda_max_um']:.1f} μm",
                color="#f8fafc", va="center", fontsize=8.2)
    ax.axvline(5.0, color="#f8fafc", linestyle="--", linewidth=0.9, alpha=0.60)
    ax.text(5.0, len(df) - 0.40, "near / mid IR transition", color="#f8fafc",
            fontsize=8.1, rotation=90, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(0.52, 32.0)
    ax.set_ylim(-0.7, len(df) - 0.15)
    ax.set_yticks(list(range(len(df))))
    ax.set_yticklabels(df["group"].iloc[::-1], color="#f8fafc")
    ax.set_xlabel("Wavelength coverage, μm, log scale")
    ax.set_ylabel("Detector group / instrument")
    ax.set_title("JWST detector groups by wavelength coverage\nnear-IR HgCdTe arrays versus mid-IR Si:As arrays")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_DETECTOR_WAVELENGTH_COVERAGE.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_detector_table(det):
    import matplotlib.pyplot as plt

    display = det[["group", "instrument", "arrays", "nx", "ny", "total_megapixels", "pixel_scale_arcsec", "detector_type", "role"]].copy()
    display["format"] = display.apply(lambda r: f"{int(r['arrays'])} x {int(r['nx'])}x{int(r['ny'])}", axis=1)
    display["mp"] = display["total_megapixels"].map(lambda x: f"{x:.2f}")
    display["scale"] = display["pixel_scale_arcsec"].map(lambda x: "spectral/IFU" if x != x else f"{x:.3f}")
    rows = display[["group", "format", "mp", "scale", "detector_type", "role"]].values.tolist()
    labels = ["Detector group", "Arrays", "MP", "arcsec/px", "Detector", "Role"]

    fig, ax = plt.subplots(figsize=(16.2, 5.4))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1.0, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.65)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            group = rows[r-1][0]
            inst = det.loc[det["group"].eq(group), "instrument"].iloc[0]
            base = {"NIRCam": "#082f49", "NIRISS": "#312e81", "NIRSpec": "#052e16", "MIRI": "#3b1114", "FGS": "#1e293b"}.get(inst, "#172554")
            cell.set_facecolor(base)
            cell.get_text().set_color("#e5e7eb")
    ax.set_title("JWST detector-array summary\n2D infrared arrays: wavelength is selected/dispersed by optics, pixels record calibrated charge",
                 color="#f8fafc", fontsize=13.2, pad=16)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_DETECTOR_SUMMARY_TABLE.png"
    fig.savefig(path, dpi=285, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    det, fov = build_tables()

    det_csv = OUTPUT_CSV / f"{VERSION}_DETECTOR_ARRAY_TABLE.csv"
    fov_csv = OUTPUT_CSV / f"{VERSION}_FIELD_OF_VIEW_TABLE.csv"
    det.to_csv(det_csv, index=False)
    fov.to_csv(fov_csv, index=False)

    p1 = plot_detector_megapixels(det)
    p2 = plot_sampling_vs_diffraction(det)
    p3 = plot_fov_area(fov)
    p4 = plot_wavelength_coverage(det)
    p5 = styled_detector_table(det)

    science_mp = det.loc[det["science_flag"], "total_megapixels"].sum()
    total_mp = det["total_megapixels"].sum()
    science_arrays = det.loc[det["science_flag"], "arrays"].sum()
    total_arrays = det["arrays"].sum()

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Mirror diameter", f"{PRIMARY_MIRROR_DIAMETER_M:.2f} m"),
        ("Detector groups", f"{len(det):,}"),
        ("FOV modes", f"{len(fov):,}"),
        ("Image/FITS downloads", "none"),
        ("Plot engine", "matplotlib only"),
    ], ["Field", "Value"])
    print("")
    print("RESULTS")
    print_table([
        ("Science detector arrays", f"{science_arrays:.0f}"),
        ("Science detector pixels", f"{science_mp:.2f} MP"),
        ("Science + guide arrays", f"{total_arrays:.0f}"),
        ("Science + guide pixels", f"{total_mp:.2f} MP"),
        ("Largest single plotted raw group", f"NIRCam SW | {det.loc[det['group'].eq('NIRCam SW'), 'total_megapixels'].iloc[0]:.2f} MP"),
        ("Key lesson", "megapixels are modest; photon collection, sampling, cooling, and calibration dominate"),
    ], ["Metric", "Value"])
    print("")
    print("DETECTOR GROUP SUMMARY")
    print_table([
        (r.group, r.instrument, f"{int(r.arrays)}", f"{r.total_megapixels:.2f}", "spectral/IFU" if r.pixel_scale_arcsec != r.pixel_scale_arcsec else f"{r.pixel_scale_arcsec:.3f}", f"{r.lambda_min_um:.1f}-{r.lambda_max_um:.1f}")
        for r in det.itertuples(index=False)
    ], ["Group", "Instrument", "Arrays", "MP", "arcsec/px", "λ μm"])
    print("")
    print("OUTPUT SUMMARY")
    print_table([
        ("png", str(p1)),
        ("png", str(p2)),
        ("png", str(p3)),
        ("png", str(p4)),
        ("png", str(p5)),
        ("csv", str(det_csv)),
        ("csv", str(fov_csv)),
    ], ["Type", "Path"])
    print("")
    print("COMMENTS")
    print("This dashboard explains why JWST is not a gigapixel camera: it uses low-noise cryogenic IR arrays behind a 6.5 m mirror.")
    print("Normal imaging uses a 2D detector through a filter. Spectroscopy uses dispersers and calibrates detector position to wavelength.")
    print("Values are rounded engineering references for learning/visualization, not calibration reference files.")
    print("Matplotlib only. No AI images.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
