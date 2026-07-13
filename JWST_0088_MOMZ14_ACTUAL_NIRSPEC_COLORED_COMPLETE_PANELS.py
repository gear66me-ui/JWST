#!/usr/bin/env python3
"""
JWST_0088_MOMZ14_ACTUAL_NIRSPEC_COLORED_COMPLETE_PANELS.py

Uses the coordinate-verified public MoM-z14 JWST/NIRSpec prism X1D extraction.
Plots only actual unsmoothed detector samples. Reference transitions receive
unique colors. Narrow panels automatically expand to include enough real X1D
samples, preventing blank plots without interpolation or synthetic profiles.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_NAME = "JWST_0087_MOMZ14_ACTUAL_NIRSPEC_RAW_SPECTROGRAPH.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
BASE_PATH = Path("/content") / BASE_NAME
VERSION = "JWST_0088"
MOM_Z = 14.44
OUT = Path("/content/JWST_OUTPUT")
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA_CACHE = OUT / "DATA" / "JWST_0087"
for directory in (PNG, CSV, DATA_CACHE):
    directory.mkdir(parents=True, exist_ok=True)


def load_base():
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size < 12000:
        urllib.request.urlretrieve(BASE_URL, BASE_PATH)
    spec = importlib.util.spec_from_file_location("jwst_0087_base", str(BASE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load JWST_0087 helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    module.PNG = PNG
    module.CSV = CSV
    module.DATA = DATA_CACHE
    return module


base = load_base()
np = base.np
pd = base.pd
plt = base.plt
matplotlib = base.matplotlib

LINE_COLORS = {
    "N IV] 1483": "#ff4d6d",
    "N IV] 1487": "#ff9f1c",
    "C IV 1548": "#00d4ff",
    "C IV 1551": "#4361ee",
    "He II 1640": "#c77dff",
    "O III] 1661": "#06d6a0",
    "O III] 1666": "#2ec4b6",
    "N III] 1747": "#f72585",
    "N III] 1749": "#ff6b6b",
    "N III] 1750": "#ffd166",
    "N III] 1752": "#e76f51",
    "N III] 1754": "#ff8fab",
    "C III] 1907": "#4cc9f0",
    "C III] 1909": "#90e0ef",
}

COMPLEXES = base.COMPLEXES
MIN_PANEL_SAMPLES = 24


def reset_style() -> None:
    plt.close("all")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "text.usetex": False,
        "figure.facecolor": "#050712",
        "axes.facecolor": "#07101f",
        "axes.edgecolor": "#8ca3b8",
        "axes.labelcolor": "#f1f5f9",
        "xtick.color": "#dbeafe",
        "ytick.color": "#dbeafe",
        "text.color": "#f8fafc",
        "font.size": 10,
    })


def obs_um(rest_A):
    return np.asarray(rest_A, dtype=float) * (1.0 + MOM_Z) * 1.0e-4


def rest_A(obs_um_values):
    return np.asarray(obs_um_values, dtype=float) * 1.0e4 / (1.0 + MOM_Z)


def all_reference_rows():
    rows = []
    index = 1
    for complex_def in COMPLEXES:
        for label, rest_wave in complex_def["lines"]:
            rows.append({
                "id": index,
                "complex": complex_def["key"],
                "line": label,
                "rest_A": float(rest_wave),
                "observed_um": float(obs_um(rest_wave)),
                "color": LINE_COLORS[label],
            })
            index += 1
    return rows


def adaptive_indices(wave_um, complex_def, minimum=MIN_PANEL_SAMPLES):
    nominal_low, nominal_high = obs_um(complex_def["window_A"])
    nominal = np.flatnonzero((wave_um >= nominal_low) & (wave_um <= nominal_high))
    center_rest = np.mean([line[1] for line in complex_def["lines"]])
    center_um = float(obs_um(center_rest))
    center_index = int(np.argmin(np.abs(wave_um - center_um)))

    half = max(minimum // 2, 1)
    start = max(0, center_index - half)
    stop = min(len(wave_um), start + minimum)
    start = max(0, stop - minimum)
    adaptive = np.arange(start, stop, dtype=int)

    if nominal.size:
        combined_start = min(int(nominal[0]), int(adaptive[0]))
        combined_stop = max(int(nominal[-1]) + 1, int(adaptive[-1]) + 1)
        selected = np.arange(combined_start, combined_stop, dtype=int)
    else:
        selected = adaptive

    if selected.size < 8:
        raise RuntimeError(f"Insufficient real X1D samples for {complex_def['key']}.")
    return selected


def style_axis(axis):
    axis.grid(True, color="#334155", linewidth=0.55, alpha=0.58)
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")


def local_limits(values, errors):
    return base.robust_limits(values, errors)


def draw_reference_lines(axis, complex_def, label_inside=True):
    for line_index, (label, wavelength) in enumerate(complex_def["lines"]):
        color = LINE_COLORS[label]
        axis.axvline(wavelength, color=color, linewidth=1.35, linestyle="--", alpha=0.95)
        if label_inside:
            axis.text(
                wavelength,
                0.98 - 0.12 * (line_index % 4),
                label,
                rotation=90,
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=8.0,
                color=color,
                bbox={"facecolor": "#050712", "alpha": 0.68, "edgecolor": "none", "pad": 1.5},
            )


def plot_actual_panel(axis, wave_um, flux, error, complex_def, ylabel):
    indices = adaptive_indices(wave_um, complex_def)
    x_um = wave_um[indices]
    x_rest = rest_A(x_um)
    y = flux[indices]
    e = error[indices]

    axis.plot(
        x_rest,
        y,
        color="#e2e8f0",
        linewidth=1.2,
        drawstyle="steps-mid",
        label=f"Actual X1D samples ({len(indices)})",
        zorder=3,
    )
    axis.scatter(x_rest, y, s=13, color="#e2e8f0", edgecolors="#0f172a", linewidths=0.35, zorder=4)
    good = np.isfinite(e) & (e >= 0)
    if good.any():
        axis.fill_between(
            x_rest[good], y[good] - e[good], y[good] + e[good],
            color="#38bdf8", alpha=0.16, step="mid", label="1-sigma uncertainty", zorder=1,
        )

    draw_reference_lines(axis, complex_def, label_inside=True)
    axis.set_xlim(float(x_rest.min()), float(x_rest.max()))
    axis.set_ylim(*local_limits(y, e))
    axis.set_title(
        f"{complex_def['title']} — actual unsmoothed MoM-z14 samples\n"
        f"adaptive real-data window: {x_rest.min():.1f}-{x_rest.max():.1f} A | n={len(indices)}",
        fontsize=12,
        pad=8,
    )
    axis.set_xlabel("Rest-frame vacuum wavelength [angstrom]")
    axis.set_ylabel(ylabel)
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=8, facecolor="#020617", edgecolor="#475569")

    top = axis.secondary_xaxis("top", functions=(lambda x: obs_um(x), lambda x: rest_A(x)))
    top.set_xlabel("Observed wavelength [micrometers]")
    return indices


def make_full_spectrum(wave_um, flux, error, ylabel):
    rows = all_reference_rows()
    figure = plt.figure(figsize=(18, 11.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=[5.5, 2.25])
    axis = figure.add_subplot(grid[0, 0])
    key_axis = figure.add_subplot(grid[1, 0])

    axis.plot(wave_um, flux, color="#e2e8f0", linewidth=0.88, drawstyle="steps-mid",
              label="Actual NIRSpec X1D flux samples", zorder=3)
    good = np.isfinite(error) & (error >= 0)
    if good.any():
        axis.fill_between(wave_um[good], flux[good] - error[good], flux[good] + error[good],
                          color="#38bdf8", alpha=0.12, step="mid", label="1-sigma uncertainty")

    for row in rows:
        axis.axvline(row["observed_um"], color=row["color"], linewidth=1.15,
                     linestyle="--", alpha=0.92)

    axis.set_xlabel("Observed wavelength [micrometers]")
    axis.set_ylabel(ylabel)
    axis.set_title(
        "MoM-z14 — actual public JWST/NIRSpec prism X1D spectrum\n"
        "Raw jagged detector samples; no smoothing, interpolation, or synthetic line profiles",
        fontsize=17,
        pad=14,
    )
    axis.set_ylim(*local_limits(flux, error))
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=9, facecolor="#020617", edgecolor="#475569")
    top = axis.secondary_xaxis("top", functions=(lambda x: rest_A(x), lambda x: obs_um(x)))
    top.set_xlabel("Rest-frame wavelength [angstrom] at z = 14.44")

    key_axis.axis("off")
    key_axis.set_title("Color key for the reference transitions", loc="left", fontsize=13, pad=6)
    table_rows = [[r["id"], r["line"], f"{r['rest_A']:.2f}", f"{r['observed_um']:.5f}"] for r in rows]
    table = key_axis.table(
        cellText=table_rows,
        colLabels=["ID", "Transition", "Rest A", "Observed um at z=14.44"],
        cellLoc="left",
        colLoc="left",
        bbox=[0.00, 0.00, 0.67, 0.92],
        colWidths=[0.06, 0.27, 0.13, 0.21],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.4)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("#223047")
            cell.set_text_props(color="#ffffff", weight="bold")
        else:
            cell.set_facecolor("#111827" if r % 2 else "#172033")
            cell.set_text_props(color="#e5edf5")
            if c == 0:
                cell.set_facecolor(rows[r - 1]["color"])
                cell.set_text_props(color="#050712", weight="bold")

    key_axis.text(
        0.715, 0.82,
        "DATA STATUS\n\n"
        "White trace: actual X1D flux samples\n"
        "Blue band: measured 1-sigma uncertainty\n"
        "Colored dashed lines: laboratory reference positions\n"
        "Spectral trace: unsmoothed and uninterpolated\n"
        "Panel windows: expanded only by selecting more real samples",
        transform=key_axis.transAxes,
        ha="left", va="top", fontsize=10.2, linespacing=1.55,
        bbox={"facecolor": "#0b1220", "edgecolor": "#475569", "boxstyle": "round,pad=0.7"},
    )

    output = PNG / f"{VERSION}_MOMZ14_ACTUAL_FULL_NIRSPEC_COLORED_KEY.png"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)
    return output


def make_group(wave_um, flux, error, keys, title, filename, ylabel):
    lookup = {item["key"]: item for item in COMPLEXES}
    figure, axes = plt.subplots(len(keys), 1, figsize=(17, 6.3 * len(keys)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    selections = []
    for axis, key in zip(axes, keys):
        indices = plot_actual_panel(axis, wave_um, flux, error, lookup[key], ylabel)
        selections.append((key, indices))
    figure.suptitle(title + "\nActual coordinate-matched JWST/NIRSpec prism data", fontsize=17)
    output = PNG / filename
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)
    return output, selections


def save_panel_samples(wave_um, flux, error, selections):
    rows = []
    for key, indices in selections:
        for index in indices:
            rows.append({
                "panel": key,
                "sample_index": int(index),
                "observed_wavelength_um": float(wave_um[index]),
                "rest_wavelength_angstrom_z14p44": float(rest_A(wave_um[index])),
                "flux_display_units": float(flux[index]),
                "flux_error_display_units": float(error[index]) if np.isfinite(error[index]) else np.nan,
                "actual_x1d_sample": True,
                "interpolated": False,
                "smoothed": False,
            })
    path = CSV / f"{VERSION}_MOMZ14_ACTUAL_PANEL_SAMPLES.csv"
    pd.DataFrame(rows).drop_duplicates(["panel", "sample_index"]).to_csv(path, index=False)
    return path


def main():
    reset_style()
    spectrum, audit_path = base.find_exact_spectrum()
    wave_um = spectrum["wavelength_um"]
    raw_flux = spectrum["flux"]
    raw_error = spectrum["error"]
    flux, error, ylabel = base.convert_flux(raw_flux, raw_error, spectrum["flux_unit"])

    full_png = make_full_spectrum(wave_um, flux, error, ylabel)
    nitrogen_png, nitrogen_sel = make_group(
        wave_um, flux, error, ("N_IV", "N_III"),
        "MoM-z14 nitrogen complexes — complete raw spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_NITROGEN_COMPLETE_RAW.png", ylabel,
    )
    carbon_png, carbon_sel = make_group(
        wave_um, flux, error, ("C_IV", "C_III"),
        "MoM-z14 carbon complexes — complete raw spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_CARBON_COMPLETE_RAW.png", ylabel,
    )
    blend_png, blend_sel = make_group(
        wave_um, flux, error, ("HE_O",),
        "MoM-z14 He II + O III] blend — complete raw spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_HEII_OIII_COMPLETE_RAW.png", ylabel,
    )

    reference_csv = CSV / f"{VERSION}_MOMZ14_REFERENCE_LINE_COLOR_KEY.csv"
    pd.DataFrame(all_reference_rows()).to_csv(reference_csv, index=False)
    sample_csv = save_panel_samples(wave_um, flux, error, nitrogen_sel + carbon_sel + blend_sel)

    print(f"CODE OUTPUT: {VERSION}")
    print("DATA            actual public coordinate-matched JWST/NIRSpec X1D samples")
    print("SMOOTHING       none")
    print("INTERPOLATION   none")
    print("SYNTHETIC DATA  none")
    print("PANEL POLICY    adaptive windows select at least 24 real detector samples")
    for key, indices in nitrogen_sel + carbon_sel + blend_sel:
        rr = rest_A(wave_um[indices])
        print(f"{key:<8}        n={len(indices):>3}  rest={rr.min():.2f}-{rr.max():.2f} A")
    print(f"FULL PNG        {full_png}")
    print(f"NITROGEN PNG    {nitrogen_png}")
    print(f"CARBON PNG      {carbon_png}")
    print(f"BLEND PNG       {blend_png}")
    print(f"PANEL CSV       {sample_csv}")
    print(f"LINE KEY CSV    {reference_csv}")
    print(f"AUDIT CSV       {audit_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
