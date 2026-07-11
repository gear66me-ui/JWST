# JWST_0048
# Dark engineering-style MoM-z14 plots built from the JWST_0047 bounded MAST workflow.
# No AI images. Matplotlib only. Real MAST spectrum data for measured UV plots.
# Optical plot is explicitly theoretical/reference because the real spectrum is out of band.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import subprocess
import sys

VERSION = "JWST_0048"
HELPER_FILENAME = "JWST_0047_MOMZ14_UV_OPTICAL_REALDATA_TIMEOUT.py"
HELPER_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    + HELPER_FILENAME
)
HELPER_PATH = Path("/content") / HELPER_FILENAME if Path("/content").exists() else Path.cwd() / HELPER_FILENAME

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
CYAN = "#18c7d8"
ORANGE = "#ff9d2e"
GRAY = "#aeb7c2"
RED = "#ff6b5f"
BLUE = "#43b9ff"


def _ensure_helper():
    if HELPER_PATH.exists() and HELPER_PATH.stat().st_size > 5000:
        return HELPER_PATH
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "--connect-timeout",
            "15",
            "--max-time",
            "60",
            "-o",
            str(HELPER_PATH),
            HELPER_URL,
        ],
        check=True,
        timeout=70,
    )
    return HELPER_PATH


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _style_dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.55, alpha=0.50)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color(CYAN)
        spine.set_linewidth(0.85)


def _legend_dark(ax, loc="best", fontsize=8.0):
    leg = ax.legend(
        loc=loc,
        fontsize=fontsize,
        facecolor="#07111f",
        edgecolor=GRID,
        framealpha=0.96,
    )
    for txt in leg.get_texts():
        txt.set_color(TEXT)
    return leg


def _finite_window(wave, flux, center, half_width):
    import numpy as np

    mask = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & (wave >= center - half_width)
        & (wave <= center + half_width)
    )
    return mask


def plot_combined_uv_dark(base, wave, flux, unit, uv_df, avg_uv_obs):
    import numpy as np
    import matplotlib.pyplot as plt

    valid = uv_df[uv_df["sample_count_in_window"] > 0].copy()
    if valid.empty:
        print("PLOT SKIPPED | combined UV | no measured UV samples")
        return None

    xmin = max(float(np.nanmin(wave)), float(valid["expected_at_z14p44_um"].min() - 0.16))
    xmax = min(float(np.nanmax(wave)), float(valid["expected_at_z14p44_um"].max() + 0.16))
    mask = _finite_window(wave, flux, (xmin + xmax) / 2.0, (xmax - xmin) / 2.0)
    if int(mask.sum()) < 2:
        print("PLOT SKIPPED | combined UV | selected range contains fewer than 2 finite samples")
        return None

    fig, ax = plt.subplots(figsize=(15.8, 7.8))
    _style_dark(ax)

    ax.plot(
        wave[mask],
        flux[mask],
        color=CYAN,
        linewidth=0.70,
        alpha=0.92,
        label="raw JWST/MAST spectrum samples",
    )

    for row in valid.itertuples():
        ax.axvline(
            row.expected_at_z14p44_um,
            color=GRAY,
            linestyle=":",
            linewidth=0.80,
            alpha=0.80,
        )
        ax.axvline(
            row.raw_peak_sample_um,
            color=TEXT,
            linestyle="--",
            linewidth=0.75,
            alpha=0.95,
        )
        ax.scatter(
            [row.raw_peak_sample_um],
            [row.raw_peak_flux_native],
            s=28,
            color=TEXT,
            edgecolor=BG,
            linewidth=0.45,
            zorder=5,
        )
        ax.text(
            row.raw_peak_sample_um,
            row.raw_peak_flux_native,
            f"  {int(row.n)}",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    ax.axvline(
        avg_uv_obs,
        color=ORANGE,
        linewidth=1.55,
        label=f"average raw UV peak λ = {avg_uv_obs:.6f} µm",
    )

    top = ax.secondary_xaxis(
        "top",
        functions=(base.freq_thz_from_um, base.um_from_freq_thz),
    )
    top.set_xlabel("Frequency, THz", color=TEXT)
    top.tick_params(colors=TEXT, labelsize=8.5)

    ax.set_xlabel("Observed wavelength, µm")
    ax.set_ylabel(f"Raw flux, FITS unit: {unit}")
    ax.set_title(
        f"{VERSION} — {base.TARGET}: combined real UV spectrum windows",
        fontsize=13.5,
        pad=12,
    )
    _legend_dark(ax, "upper right", 8.2)

    fig.tight_layout()
    path = base.PNG / f"{VERSION}_COMBINED_REAL_UV_LINES_DARK.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def plot_uv_line_windows_dark(base, wave, flux, unit, uv_df):
    import numpy as np
    import matplotlib.pyplot as plt

    paths = []
    for row in uv_df.itertuples():
        center = float(row.expected_at_z14p44_um)
        mask = _finite_window(wave, flux, center, 0.095)

        if int(mask.sum()) < 2:
            print(
                f"PLOT SKIPPED | UV line {int(row.n)} {row.line} | "
                f"finite samples={int(mask.sum())}"
            )
            continue

        local_wave = wave[mask]
        local_flux = flux[mask]
        if not np.isfinite(local_flux).any():
            print(f"PLOT SKIPPED | UV line {int(row.n)} {row.line} | no finite flux")
            continue

        fig, ax = plt.subplots(figsize=(12.8, 5.8))
        _style_dark(ax)

        ax.plot(
            local_wave,
            local_flux,
            color=CYAN,
            linewidth=0.75,
            label="raw spectrum",
        )
        ax.axvline(
            center,
            color=GRAY,
            linestyle=":",
            linewidth=0.90,
            label="expected at z=14.44",
        )

        peak_ok = (
            row.raw_peak_sample_um == row.raw_peak_sample_um
            and row.raw_peak_flux_native == row.raw_peak_flux_native
        )
        if peak_ok:
            ax.axvline(
                row.raw_peak_sample_um,
                color=ORANGE,
                linestyle="--",
                linewidth=1.05,
                label="raw local peak sample",
            )
            ax.scatter(
                [row.raw_peak_sample_um],
                [row.raw_peak_flux_native],
                s=30,
                color=ORANGE,
                edgecolor=BG,
                linewidth=0.50,
                zorder=5,
            )
            ax.text(
                row.raw_peak_sample_um,
                row.raw_peak_flux_native,
                f"  {int(row.n)}",
                color=TEXT,
                fontsize=10,
                fontweight="bold",
                va="bottom",
            )

        note = (
            f"{int(row.n)}  {row.line}\n"
            f"rest λ = {row.rest_um:.6f} µm\n"
            f"expected λ = {row.expected_at_z14p44_um:.6f} µm\n"
            f"finite samples = {int(mask.sum())}"
        )
        if peak_ok:
            note += (
                f"\nraw peak λ = {row.raw_peak_sample_um:.6f} µm"
                f"\nz = {row.z_from_raw_peak:.6f}"
            )

        ax.text(
            0.018,
            0.96,
            note,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.7,
            color=TEXT,
            bbox=dict(
                facecolor="#07111f",
                edgecolor=GRID,
                boxstyle="round,pad=0.35",
                alpha=0.95,
            ),
        )

        top = ax.secondary_xaxis(
            "top",
            functions=(base.freq_thz_from_um, base.um_from_freq_thz),
        )
        top.set_xlabel("Frequency, THz", color=TEXT)
        top.tick_params(colors=TEXT, labelsize=8)

        ax.set_xlabel("Observed wavelength, µm")
        ax.set_ylabel(f"Raw flux, FITS unit: {unit}")
        ax.set_title(
            f"{VERSION} — real raw UV line window: {row.line}",
            fontsize=12.5,
            pad=10,
        )
        _legend_dark(ax, "best", 7.8)

        fig.tight_layout()
        safe = (
            row.line.replace("[", "")
            .replace("]", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        path = base.PNG / f"{VERSION}_UV_LINE_{int(row.n)}_{safe}_DARK.png"
        fig.savefig(path, dpi=270, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.show()
        plt.close(fig)
        paths.append(path)

    return paths


def plot_uv_slopes_dark(base, meas_df, avg_slope, avg_z):
    import numpy as np
    import matplotlib.pyplot as plt

    valid = meas_df[
        (meas_df["sample_count_in_window"] > 0)
        & meas_df["rest_um"].notna()
        & meas_df["raw_peak_sample_um"].notna()
    ].copy()

    if len(valid) < 1:
        print("PLOT SKIPPED | UV slopes | no measured UV lines")
        return None

    xmin = float(valid["rest_um"].min() - 0.010)
    xmax = float(valid["rest_um"].max() + 0.010)
    xs = np.linspace(xmin, xmax, 420)

    fig, ax = plt.subplots(figsize=(14.4, 7.2))
    _style_dark(ax)

    for row in valid.itertuples():
        ax.plot(
            xs,
            row.slope_m * xs,
            color=GRAY,
            linewidth=0.85,
            linestyle=":",
            alpha=0.95,
            label=f"{int(row.n)} {row.line}: m={row.slope_m:.6f}",
        )
        ax.scatter(
            [row.rest_um],
            [row.raw_peak_sample_um],
            s=32,
            color=TEXT,
            edgecolor=BG,
            linewidth=0.5,
            zorder=6,
        )
        ax.text(
            row.rest_um,
            row.raw_peak_sample_um,
            f" {int(row.n)}",
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
        )

    ax.plot(
        xs,
        avg_slope * xs,
        color=ORANGE,
        linewidth=1.90,
        label=f"average: m={avg_slope:.6f}, z={avg_z:.6f}",
    )

    ax.set_xlabel("Rest-frame wavelength, µm")
    ax.set_ylabel("Observed raw peak wavelength, µm")
    ax.set_title(
        f"{VERSION} — real UV-line slopes plus orange average",
        fontsize=13.5,
        pad=12,
    )
    _legend_dark(ax, "upper left", 8.0)

    fig.tight_layout()
    path = base.PNG / f"{VERSION}_UV_SLOPES_DARK.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def plot_optical_reference_dark(base, opt_df, wave):
    import numpy as np
    import matplotlib.pyplot as plt

    valid = opt_df[
        opt_df["rest_um"].notna()
        & opt_df["expected_at_z14p44_um"].notna()
    ].copy()

    if valid.empty:
        print("PLOT SKIPPED | optical reference | no optical line definitions")
        return None

    slope_ref = 1.0 + float(base.Z_REF)
    xmin = float(valid["rest_um"].min() - 0.020)
    xmax = float(valid["rest_um"].max() + 0.020)
    xs = np.linspace(xmin, xmax, 420)

    fig, ax = plt.subplots(figsize=(14.4, 7.2))
    _style_dark(ax)

    for row in valid.itertuples():
        ax.plot(
            xs,
            slope_ref * xs,
            color=GRAY,
            linewidth=0.85,
            linestyle=":",
            alpha=0.85,
            label=f"{int(row.n)} {row.line}: reference m={slope_ref:.6f}",
        )
        ax.scatter(
            [row.rest_um],
            [row.expected_at_z14p44_um],
            s=36,
            color=BLUE,
            edgecolor=BG,
            linewidth=0.55,
            zorder=6,
        )
        ax.text(
            row.rest_um,
            row.expected_at_z14p44_um,
            f" {int(row.n)}",
            color=TEXT,
            fontsize=9.5,
            fontweight="bold",
        )

    ax.plot(
        xs,
        slope_ref * xs,
        color=ORANGE,
        linewidth=1.90,
        label=f"reference average: m={slope_ref:.6f}, z={base.Z_REF:.6f}",
    )

    coverage_min = float(np.nanmin(wave))
    coverage_max = float(np.nanmax(wave))
    predicted_min = float(valid["expected_at_z14p44_um"].min())
    predicted_max = float(valid["expected_at_z14p44_um"].max())

    ax.text(
        0.018,
        0.965,
        (
            "THEORETICAL / REFERENCE ONLY\n"
            "No real optical spectrum samples were measured here.\n"
            f"Downloaded real spectrum coverage: {coverage_min:.3f}–{coverage_max:.3f} µm\n"
            f"Predicted Hα/[N II] range: {predicted_min:.3f}–{predicted_max:.3f} µm"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        color=TEXT,
        bbox=dict(
            facecolor="#07111f",
            edgecolor=GRID,
            boxstyle="round,pad=0.35",
            alpha=0.96,
        ),
    )

    ax.set_xlabel("Rest-frame wavelength, µm")
    ax.set_ylabel("Predicted observed wavelength, µm")
    ax.set_title(
        f"{VERSION} — optical Hα/[N II] reference slopes at z={base.Z_REF:.2f}",
        fontsize=13.5,
        pad=12,
    )
    _legend_dark(ax, "lower right", 7.9)

    fig.tight_layout()
    path = base.PNG / f"{VERSION}_OPTICAL_REFERENCE_SLOPES_DARK.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def plot_summary_table_dark(base, uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z):
    import matplotlib.pyplot as plt

    rows = []
    row_types = []

    for row in uv_df.itertuples():
        raw_peak = (
            f"{row.raw_peak_sample_um:.6f}"
            if row.raw_peak_sample_um == row.raw_peak_sample_um
            else "OUT"
        )
        slope = f"{row.slope_m:.6f}" if row.slope_m == row.slope_m else ""
        z_value = (
            f"{row.z_from_raw_peak:.6f}"
            if row.z_from_raw_peak == row.z_from_raw_peak
            else ""
        )
        rows.append(
            [
                int(row.n),
                row.line,
                f"{row.rest_um:.6f}",
                raw_peak,
                slope,
                z_value,
                "OBSERVED" if row.sample_count_in_window > 0 else "NO DATA",
            ]
        )
        row_types.append("uv")

    rows.append(
        [
            "AVG",
            "UV average",
            f"{avg_rest:.6f}",
            f"{avg_obs:.6f}",
            f"{avg_slope:.6f}",
            f"{avg_z:.6f}",
            "OBSERVED",
        ]
    )
    row_types.append("avg")

    for row in opt_df.itertuples():
        rows.append(
            [
                int(row.n),
                row.line,
                f"{row.rest_um:.6f}",
                f"{row.expected_at_z14p44_um:.6f}",
                f"{1.0 + base.Z_REF:.6f}",
                f"{base.Z_REF:.6f}",
                "THEORETICAL",
            ]
        )
        row_types.append("opt")

    fig, ax = plt.subplots(figsize=(15.8, 6.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(
        f"{VERSION} — UV observed measurements and optical reference positions",
        color=TEXT,
        fontsize=13.0,
        pad=12,
    )

    table = ax.table(
        cellText=rows,
        colLabels=[
            "#",
            "line",
            "rest λ µm",
            "obs/pred λ µm",
            "slope m",
            "z",
            "status",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.35)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.48)
        cell.get_text().set_color(TEXT)

        if r == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            row_type = row_types[r - 1]
            if row_type == "avg":
                cell.set_facecolor("#4a2e12")
                cell.get_text().set_fontweight("bold")
            elif row_type == "opt":
                cell.set_facecolor("#0b2740")
            else:
                cell.set_facecolor("#081523")

    fig.tight_layout()
    path = base.PNG / f"{VERSION}_SUMMARY_TABLE_DARK.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return path


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print("PLOT MODE  : dark navy / cyan / orange engineering style")
    print("DATA RULE  : measured UV from real MAST data; optical explicitly theoretical/reference")
    print()

    helper_path = _ensure_helper()
    helper = _load_module(helper_path, "jwst_0047_helper")
    base_path = helper._ensure_base_script()
    base = helper._load_base_module(base_path)

    base.VERSION = VERSION
    base.query_mast_products = lambda: helper.query_mast_products_safe(base)
    base.download_product = helper.download_product_safe

    base.style_light = _style_dark
    base.style_dark = _style_dark
    base.legend_dark = _legend_dark

    base.plot_combined_uv = lambda wave, flux, unit, uv_df, avg_uv_obs: (
        plot_combined_uv_dark(base, wave, flux, unit, uv_df, avg_uv_obs)
    )
    base.plot_uv_line_windows = lambda wave, flux, unit, uv_df: (
        plot_uv_line_windows_dark(base, wave, flux, unit, uv_df)
    )
    base.plot_slopes = lambda meas_df, avg_slope, avg_z: (
        plot_uv_slopes_dark(base, meas_df, avg_slope, avg_z)
    )
    base.plot_optical_audit = lambda opt_df, wave: (
        plot_optical_reference_dark(base, opt_df, wave)
    )
    base.plot_summary_table = (
        lambda uv_df, opt_df, avg_rest, avg_obs, avg_slope, avg_z: (
            plot_summary_table_dark(
                base,
                uv_df,
                opt_df,
                avg_rest,
                avg_obs,
                avg_slope,
                avg_z,
            )
        )
    )

    base.main()

    print()
    print("DARK PLOT UPDATE COMPLETE")
    print(f"PNG folder : {base.PNG}")
    print(f"CSV folder : {base.CSV}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
