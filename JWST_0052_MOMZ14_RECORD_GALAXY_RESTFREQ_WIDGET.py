# JWST_0052
# MoM-z14 record-galaxy archive widget: five published UV line-complex windows,
# rest/observed wavelength-frequency audit, relation plot, and styled tables.
# No AI images. Matplotlib only. Uses cached real JWST/MAST spectrum CSV data.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import subprocess
import sys

VERSION = "JWST_0052"
GALAXY = "MoM-z14"
FIELD = "COSMOS"
PROGRAM_ID = "5224"
PROGRAM_NAME = "Mirage or Miracle"
INSTRUMENT = "JWST/NIRSpec PRISM"
CONFIRMATION = "spectroscopically confirmed"
Z_SPEC = 14.44
Z_MINUS = 0.02
Z_PLUS = 0.02
RA_DEG = 150.0933255
DEC_DEG = 2.2731627
PAPER_REFERENCE = "Naidu et al. 2026, A Cosmic Miracle, arXiv:2505.11263v2"
PAPER_METHOD = "joint UV-line/blend fit with tied redshift and NIRSpec LSF + independent Lyman-alpha break"

OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
CYAN = "#18c7d8"
ORANGE = "#ff9d2e"
GRAY = "#aeb7c2"
POINT_COLORS = ["#43b9ff", "#ff6b76", "#a78bfa", "#34d399", "#f6c453"]
C_UM_THz = 299.792458

LINE_COMPLEXES = [
    {"n": 1, "name": "N IV] λ1487", "species": "nitrogen semi-forbidden", "rest_A": [1487.0]},
    {"n": 2, "name": "C IV λλ1548,1551", "species": "carbon resonant doublet", "rest_A": [1548.0, 1551.0]},
    {"n": 3, "name": "He II λ1640 + O III] λλ1661,1666", "species": "helium + oxygen blend", "rest_A": [1640.0, 1661.0, 1666.0]},
    {"n": 4, "name": "N III] λλ1747,1749,1750,1752,1754", "species": "nitrogen quintuplet", "rest_A": [1747.0, 1749.0, 1750.0, 1752.0, 1754.0]},
    {"n": 5, "name": "C III] λλ1907,1909", "species": "carbon semi-forbidden doublet", "rest_A": [1907.0, 1909.0]},
]


def need(package, import_name=None):
    name = import_name or package
    try:
        importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def latest_file(preferred, pattern):
    for filename in preferred:
        path = CSV / filename
        if path.exists() and path.stat().st_size > 100:
            return path
    matches = sorted(CSV.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    return None


def locate_inputs():
    raw_spectrum = latest_file(
        ["JWST_0048_REAL_RAW_SPECTRUM.csv", "JWST_0047_REAL_RAW_SPECTRUM.csv", "JWST_0046_REAL_RAW_SPECTRUM.csv"],
        "JWST_*_REAL_RAW_SPECTRUM.csv",
    )
    summary = latest_file(
        ["JWST_0048_SUMMARY.csv", "JWST_0047_SUMMARY.csv", "JWST_0046_SUMMARY.csv"],
        "JWST_*_SUMMARY.csv",
    )
    observations = latest_file(
        ["JWST_0048_PROGRAM_5224_OBSERVATIONS.csv", "JWST_0047_PROGRAM_5224_OBSERVATIONS.csv", "JWST_0046_PROGRAM_5224_OBSERVATIONS.csv"],
        "JWST_*_PROGRAM_5224_OBSERVATIONS.csv",
    )
    candidates = latest_file(
        ["JWST_0048_PROGRAM_5224_SPECTRUM_CANDIDATES.csv", "JWST_0047_PROGRAM_5224_SPECTRUM_CANDIDATES.csv", "JWST_0046_PROGRAM_5224_SPECTRUM_CANDIDATES.csv"],
        "JWST_*_PROGRAM_5224_SPECTRUM_CANDIDATES.csv",
    )
    if raw_spectrum is None:
        raise FileNotFoundError(
            "No cached real raw spectrum CSV was found in /content/JWST_OUTPUT/CSV. Run JWST_0048 or JWST_0051 first."
        )
    return raw_spectrum, summary, observations, candidates


def find_column(frame, exact=None, startswith=None):
    exact = exact or []
    lower = {str(column).lower(): column for column in frame.columns}
    for name in exact:
        if name.lower() in lower:
            return lower[name.lower()]
    if startswith:
        for column in frame.columns:
            if str(column).lower().startswith(startswith.lower()):
                return column
    return None


def load_raw_spectrum(path):
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(path)
    wave_col = find_column(frame, ["wavelength_um_raw", "wavelength_um"], "wavelength")
    flux_col = find_column(frame, [], "flux_raw_")
    if wave_col is None or flux_col is None:
        raise RuntimeError(f"Could not find wavelength and flux columns in {path.name}: {list(frame.columns)}")
    wave = frame[wave_col].to_numpy(float)
    flux = frame[flux_col].to_numpy(float)
    finite = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[finite]
    flux = flux[finite]
    order = np.argsort(wave)
    return wave[order], flux[order], str(flux_col)


def load_archive_metadata(summary_path, observations_path, candidates_path):
    import pandas as pd

    metadata = {
        "product": "not resolved from cached summary",
        "download_status": "cached CSV",
        "used_hdus": "not listed",
        "observation_rows": 0,
        "candidate_rows": 0,
        "summary_csv": str(summary_path) if summary_path else "not found",
        "observations_csv": str(observations_path) if observations_path else "not found",
        "candidates_csv": str(candidates_path) if candidates_path else "not found",
    }
    if summary_path and summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not summary.empty:
            row = summary.iloc[0]
            for key in ["product", "download_status", "used_hdus"]:
                if key in summary.columns and str(row[key]) not in ["", "nan", "None"]:
                    metadata[key] = str(row[key])
    if observations_path and observations_path.exists():
        try:
            metadata["observation_rows"] = int(len(pd.read_csv(observations_path)))
        except Exception:
            pass
    if candidates_path and candidates_path.exists():
        try:
            metadata["candidate_rows"] = int(len(pd.read_csv(candidates_path)))
        except Exception:
            pass
    return metadata


def line_rows(wave, flux):
    import numpy as np
    import pandas as pd

    rows = []
    for item in LINE_COMPLEXES:
        rest_A = np.asarray(item["rest_A"], dtype=float)
        rest_um = rest_A * 1.0e-4
        rest_thz = C_UM_THz / rest_um
        expected_um = rest_um * (1.0 + Z_SPEC)
        expected_thz = rest_thz / (1.0 + Z_SPEC)
        centroid_rest_um = float(rest_um.mean())
        centroid_rest_thz = float(C_UM_THz / centroid_rest_um)
        centroid_expected_um = float(centroid_rest_um * (1.0 + Z_SPEC))
        centroid_expected_thz = float(centroid_rest_thz / (1.0 + Z_SPEC))
        low = float(expected_um.min() - 0.060)
        high = float(expected_um.max() + 0.060)
        mask = np.isfinite(wave) & np.isfinite(flux) & (wave >= low) & (wave <= high)
        sample_count = int(mask.sum())
        peak_um = float("nan")
        peak_flux = float("nan")
        peak_thz = float("nan")
        exploratory_z = float("nan")
        continuum = float("nan")
        if sample_count > 0:
            local_wave = wave[mask]
            local_flux = flux[mask]
            continuum = float(np.nanmedian(local_flux))
            index = int(np.nanargmax(local_flux - continuum))
            peak_um = float(local_wave[index])
            peak_flux = float(local_flux[index])
            peak_thz = float(C_UM_THz / peak_um)
            exploratory_z = float(peak_um / centroid_rest_um - 1.0)
        rows.append(
            {
                "n": item["n"],
                "galaxy": GALAXY,
                "published_z_spec": Z_SPEC,
                "line_complex": item["name"],
                "species": item["species"],
                "rest_wavelengths_A": ", ".join(f"{value:.0f}" for value in rest_A),
                "rest_wavelengths_um": ", ".join(f"{value:.6f}" for value in rest_um),
                "rest_frequencies_THz": ", ".join(f"{value:.3f}" for value in rest_thz),
                "expected_observed_wavelengths_um": ", ".join(f"{value:.6f}" for value in expected_um),
                "expected_observed_frequencies_THz": ", ".join(f"{value:.3f}" for value in expected_thz),
                "centroid_rest_um": centroid_rest_um,
                "centroid_rest_frequency_THz": centroid_rest_thz,
                "centroid_expected_observed_um": centroid_expected_um,
                "centroid_expected_observed_frequency_THz": centroid_expected_thz,
                "raw_local_peak_um_exploratory": peak_um,
                "raw_local_peak_frequency_THz_exploratory": peak_thz,
                "raw_local_peak_flux": peak_flux,
                "raw_local_continuum_median": continuum,
                "exploratory_z_from_local_peak": exploratory_z,
                "sample_count": sample_count,
                "window_low_um": low,
                "window_high_um": high,
                "method_status": "raw local maximum for visualization only; published redshift comes from joint line/blend fit" if sample_count > 0 else "no finite cached spectrum samples in window",
            }
        )
    return pd.DataFrame(rows)


def cosmology_metadata():
    from astropy.cosmology import Planck18

    age_gyr = float(Planck18.age(Z_SPEC).value)
    lookback_gyr = float(Planck18.lookback_time(Z_SPEC).value)
    comoving_gpc = float(Planck18.comoving_distance(Z_SPEC).value / 1000.0)
    luminosity_gpc = float(Planck18.luminosity_distance(Z_SPEC).value / 1000.0)
    return {
        "universe_age_Myr": age_gyr * 1000.0,
        "lookback_time_Gyr": lookback_gyr,
        "comoving_distance_Gpc": comoving_gpc,
        "comoving_distance_Gly": comoving_gpc * 3.26156,
        "luminosity_distance_Gpc": luminosity_gpc,
    }


def style_axis(ax, small=False):
    ax.set_facecolor(AX_BG)
    ax.grid(True, color=GRID, linewidth=0.45 if small else 0.58, alpha=0.48)
    ax.tick_params(colors=TEXT, labelsize=7.0 if small else 9.0)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.75)


def dark_legend(ax, loc="best", fontsize=7.2):
    legend = ax.legend(loc=loc, fontsize=fontsize, facecolor="#07111f", edgecolor=GRID, framealpha=0.96)
    for text in legend.get_texts():
        text.set_color(TEXT)
    return legend


def top_frequency_axis(ax):
    def um_to_thz(value):
        import numpy as np
        value = np.asarray(value, dtype=float)
        return C_UM_THz / value

    def thz_to_um(value):
        import numpy as np
        value = np.asarray(value, dtype=float)
        safe = np.where(value == 0, np.nan, value)
        return C_UM_THz / safe

    top = ax.secondary_xaxis("top", functions=(um_to_thz, thz_to_um))
    top.set_xlabel("Observed frequency, THz", color=TEXT)
    top.tick_params(colors=TEXT, labelsize=7.0)
    return top


def draw_spectrum_panel(ax, row, wave, flux, color, small=False):
    import numpy as np

    style_axis(ax, small=small)
    mask = np.isfinite(wave) & np.isfinite(flux) & (wave >= row.window_low_um) & (wave <= row.window_high_um)
    if int(mask.sum()) < 2:
        ax.text(0.5, 0.5, "NO FINITE CACHED SPECTRUM DATA", transform=ax.transAxes, ha="center", va="center", color=TEXT, fontsize=9.0, fontweight="bold")
    else:
        local_wave = wave[mask]
        local_flux = flux[mask]
        ax.plot(local_wave, local_flux, color=CYAN, linewidth=0.72, label="real cached MAST spectrum samples")
        rest_values = [float(value) for value in row.rest_wavelengths_A.split(", ")]
        for component_index, rest_A in enumerate(rest_values):
            expected_um = rest_A * 1.0e-4 * (1.0 + Z_SPEC)
            ax.axvline(expected_um, color=color, linestyle=":" if component_index else "--", linewidth=0.95, alpha=0.92, label="published-z component positions" if component_index == 0 else None)
        if row.raw_local_peak_um_exploratory == row.raw_local_peak_um_exploratory:
            ax.axvline(row.raw_local_peak_um_exploratory, color=ORANGE, linewidth=1.05, label="exploratory raw local maximum")
            ax.scatter([row.raw_local_peak_um_exploratory], [row.raw_local_peak_flux], s=28 if small else 42, color=ORANGE, edgecolor="#f8fbff", linewidth=0.65, zorder=6)
        ymin = float(np.nanmin(local_flux))
        ymax = float(np.nanmax(local_flux))
        span = ymax - ymin
        pad = 0.08 * span if span > 0 else max(abs(ymax) * 0.10, 1.0e-6)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(row.window_low_um, row.window_high_um)
    ax.set_xlabel("Observed wavelength, µm", fontsize=7.4 if small else 9.3)
    ax.set_ylabel("Raw flux", fontsize=7.4 if small else 9.3)
    ax.set_title(f"{GALAXY} | zspec={Z_SPEC:.2f} | {int(row.n)} {row.line_complex}", fontsize=8.3 if small else 11.0, pad=7)
    top_frequency_axis(ax)
    info = (
        f"Rest λ [Å]: {row.rest_wavelengths_A}\n"
        f"Rest ν [THz]: {row.rest_frequencies_THz}\n"
        f"Expected obs λ [µm]: {row.expected_observed_wavelengths_um}\n"
        f"Expected obs ν [THz]: {row.expected_observed_frequencies_THz}\n"
        f"Raw local peak: {row.raw_local_peak_um_exploratory:.6f} µm | {row.raw_local_peak_frequency_THz_exploratory:.3f} THz"
    )
    ax.text(0.014, 0.965, info, transform=ax.transAxes, ha="left", va="top", color=TEXT, fontsize=5.9 if small else 7.7, linespacing=1.30, bbox=dict(boxstyle="round,pad=0.32", facecolor="#07111f", edgecolor=color, alpha=0.94))
    dark_legend(ax, "lower right", 5.7 if small else 7.0)


def create_individual_plots(rows, wave, flux):
    import matplotlib.pyplot as plt

    paths = []
    for index, row in enumerate(rows.itertuples()):
        fig = plt.figure(figsize=(14.2, 6.6), facecolor=BG)
        ax = fig.add_axes([0.075, 0.13, 0.89, 0.76])
        draw_spectrum_panel(ax, row, wave, flux, POINT_COLORS[index], small=False)
        fig.suptitle(f"{VERSION} — {GALAXY}, highest-redshift spectroscopically confirmed galaxy | GO-{PROGRAM_ID} | {INSTRUMENT}", color=TEXT, fontsize=12.4, y=0.975)
        path = PNG / f"{VERSION}_{GALAXY}_LINE_COMPLEX_{int(row.n)}_RESTFREQ.png"
        fig.savefig(path, dpi=245, facecolor=BG, edgecolor=BG)
        plt.show()
        plt.close(fig)
        paths.append(path)
    return paths


def draw_relation_plot(ax, rows, small=False):
    import numpy as np

    style_axis(ax, small=small)
    all_component_x = []
    all_component_y = []
    for index, row in enumerate(rows.itertuples()):
        color = POINT_COLORS[index]
        rest_A = np.asarray([float(value) for value in row.rest_wavelengths_A.split(", ")])
        rest_um = rest_A * 1.0e-4
        expected_um = rest_um * (1.0 + Z_SPEC)
        all_component_x.extend(rest_um.tolist())
        all_component_y.extend(expected_um.tolist())
        ax.scatter(rest_um, expected_um, s=28 if small else 48, color=color, edgecolor="#f8fbff", linewidth=0.65, label=f"{int(row.n)} {row.line_complex}", zorder=7)
        ax.scatter([row.centroid_rest_um], [row.raw_local_peak_um_exploratory], marker="x", s=54 if small else 78, color=ORANGE, linewidth=1.35, zorder=9)
        ax.annotate(str(int(row.n)), (row.centroid_rest_um, row.raw_local_peak_um_exploratory), xytext=(5, 5), textcoords="offset points", color=TEXT, fontsize=7.0 if small else 9.0, fontweight="bold")
    xmin = min(all_component_x) - 0.004
    xmax = max(all_component_x) + 0.004
    xline = np.linspace(xmin, xmax, 400)
    ax.plot(xline, (1.0 + Z_SPEC) * xline, color=ORANGE, linewidth=2.0 if small else 2.35, label=f"published relation: λobs=(1+z)λrest, z={Z_SPEC:.2f}", zorder=5)
    valid = rows[rows["raw_local_peak_um_exploratory"].notna()].copy()
    exploratory_slope = float(valid["raw_local_peak_um_exploratory"].sum() / valid["centroid_rest_um"].sum())
    exploratory_z = exploratory_slope - 1.0
    ax.set_xlabel("Rest-frame wavelength, µm", fontsize=7.6 if small else 9.6)
    ax.set_ylabel("Observed wavelength, µm", fontsize=7.6 if small else 9.6)
    ax.set_title(f"{GALAXY}: rest versus observed wavelength | colored circles=published-z components | orange X=raw local maxima", fontsize=8.6 if small else 11.5, pad=8)
    dark_legend(ax, "upper left", 5.7 if small else 7.0)
    ax.text(
        0.985,
        0.025,
        f"GALAXY: {GALAXY}\nPublished zspec: {Z_SPEC:.2f} ± 0.02\nExploratory local-peak ratio: z={exploratory_z:.6f}\nPaper method: joint tied-redshift line/blend fit\nRaw local maxima are visualization aids, not published line centers",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=TEXT,
        fontsize=6.2 if small else 7.9,
        linespacing=1.28,
        bbox=dict(boxstyle="round,pad=0.38", facecolor="#07111f", edgecolor=ORANGE, alpha=0.96),
    )
    return exploratory_slope, exploratory_z


def create_dashboard(rows, wave, flux):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(18.5, 14.4), facecolor=BG)
    axes = axes.ravel()
    for index, row in enumerate(rows.itertuples()):
        draw_spectrum_panel(axes[index], row, wave, flux, POINT_COLORS[index], small=True)
    exploratory_slope, exploratory_z = draw_relation_plot(axes[5], rows, small=True)
    fig.suptitle(f"{VERSION} — {GALAXY} | zspec={Z_SPEC:.2f}±0.02 | COSMOS | JWST GO-{PROGRAM_ID} | five published UV line complexes", color=TEXT, fontsize=15.3, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.947, bottom=0.055, hspace=0.38, wspace=0.18)
    path = PNG / f"{VERSION}_{GALAXY}_FIVE_SPECTRA_RESTFREQ_DASHBOARD.png"
    fig.savefig(path, dpi=230, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return path, exploratory_slope, exploratory_z


def create_relation_plot(rows):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.8, 8.8), facecolor=BG)
    ax = fig.add_axes([0.075, 0.11, 0.89, 0.79])
    exploratory_slope, exploratory_z = draw_relation_plot(ax, rows, small=False)
    fig.suptitle(f"{VERSION} — {GALAXY}: published spectroscopic redshift and archive-spectrum audit", color=TEXT, fontsize=13.8, y=0.975)
    path = PNG / f"{VERSION}_{GALAXY}_REST_VS_OBSERVED_RELATION.png"
    fig.savefig(path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)
    return path, exploratory_slope, exploratory_z


def create_tables(rows, archive, cosmology, raw_path, flux_column, exploratory_slope, exploratory_z):
    import pandas as pd
    import matplotlib.pyplot as plt

    line_csv = CSV / f"{VERSION}_{GALAXY}_LINE_COMPLEX_REST_OBSERVED_FREQUENCY.csv"
    rows.to_csv(line_csv, index=False)
    metadata = pd.DataFrame(
        [
            {
                "galaxy": GALAXY,
                "status": CONFIRMATION,
                "published_z_spec": Z_SPEC,
                "z_minus": Z_MINUS,
                "z_plus": Z_PLUS,
                "field": FIELD,
                "ra_deg": RA_DEG,
                "dec_deg": DEC_DEG,
                "jwst_program": PROGRAM_ID,
                "program_name": PROGRAM_NAME,
                "instrument": INSTRUMENT,
                "paper_reference": PAPER_REFERENCE,
                "paper_redshift_method": PAPER_METHOD,
                "archive_product_from_cached_summary": archive["product"],
                "archive_download_status": archive["download_status"],
                "archive_used_hdus": archive["used_hdus"],
                "raw_spectrum_csv": str(raw_path),
                "raw_flux_column": flux_column,
                "universe_age_Myr_Planck18": cosmology["universe_age_Myr"],
                "lookback_time_Gyr_Planck18": cosmology["lookback_time_Gyr"],
                "comoving_distance_Gpc_Planck18": cosmology["comoving_distance_Gpc"],
                "comoving_distance_Gly_Planck18": cosmology["comoving_distance_Gly"],
                "luminosity_distance_Gpc_Planck18": cosmology["luminosity_distance_Gpc"],
                "exploratory_local_peak_ratio_slope": exploratory_slope,
                "exploratory_local_peak_ratio_z": exploratory_z,
            }
        ]
    )
    metadata_csv = CSV / f"{VERSION}_{GALAXY}_GALAXY_ARCHIVE_METADATA.csv"
    metadata.to_csv(metadata_csv, index=False)

    table_rows = []
    for row in rows.itertuples():
        table_rows.append(
            [
                int(row.n),
                row.line_complex,
                row.rest_wavelengths_A,
                row.rest_frequencies_THz,
                row.expected_observed_wavelengths_um,
                row.expected_observed_frequencies_THz,
                f"{row.raw_local_peak_um_exploratory:.6f}",
                f"{row.raw_local_peak_frequency_THz_exploratory:.3f}",
                f"{row.exploratory_z_from_local_peak:.6f}",
                int(row.sample_count),
            ]
        )
    fig = plt.figure(figsize=(21.0, 9.0), facecolor=BG)
    ax = fig.add_axes([0.018, 0.12, 0.964, 0.76])
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.set_title(f"{VERSION} — {GALAXY} line-complex rest/observed wavelength-frequency audit", color=TEXT, fontsize=14.5, fontweight="bold", pad=16)
    table = ax.table(
        cellText=table_rows,
        colLabels=["#", "Published line complex", "Rest λ [Å]", "Rest ν [THz]", "Expected observed λ [µm]", "Expected observed ν [THz]", "Raw local peak λ [µm]", "Raw local peak ν [THz]", "Exploratory z", "Samples"],
        loc="center",
        cellLoc="center",
        colWidths=[0.035, 0.175, 0.09, 0.13, 0.14, 0.14, 0.105, 0.105, 0.085, 0.06],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.62)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.50)
        cell.get_text().set_color(TEXT)
        if r == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#081523")
            if c == 0:
                cell.get_text().set_color(POINT_COLORS[r - 1])
                cell.get_text().set_fontweight("bold")
    fig.text(0.025, 0.064, f"GALAXY: {GALAXY} | status: {CONFIRMATION} | zspec={Z_SPEC:.2f}±0.02 | RA={RA_DEG:.7f}° Dec={DEC_DEG:.7f}° | COSMOS | JWST GO-{PROGRAM_ID} {PROGRAM_NAME} | {INSTRUMENT}", color=TEXT, fontsize=9.0, ha="left")
    fig.text(0.025, 0.034, f"Planck18: universe age={cosmology['universe_age_Myr']:.1f} Myr; lookback={cosmology['lookback_time_Gyr']:.4f} Gyr; comoving distance={cosmology['comoving_distance_Gly']:.3f} Gly. Raw local maxima are exploratory; the published redshift comes from a joint tied-redshift line/blend fit.", color=MUTED, fontsize=8.4, ha="left")
    table_png = PNG / f"{VERSION}_{GALAXY}_REST_FREQUENCY_RESULTS_TABLE.png"
    fig.savefig(table_png, dpi=250, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    meta_rows = [
        ("Galaxy", GALAXY),
        ("Status", CONFIRMATION),
        ("Published redshift", f"{Z_SPEC:.2f} -{Z_MINUS:.2f}/+{Z_PLUS:.2f}"),
        ("Field", FIELD),
        ("Coordinates", f"RA {RA_DEG:.7f} deg | Dec {DEC_DEG:.7f} deg"),
        ("JWST program", f"GO-{PROGRAM_ID} | {PROGRAM_NAME}"),
        ("Instrument", INSTRUMENT),
        ("Paper", PAPER_REFERENCE),
        ("Published method", PAPER_METHOD),
        ("Cached archive product", archive["product"]),
        ("Cached raw spectrum", str(raw_path)),
        ("Universe age", f"{cosmology['universe_age_Myr']:.3f} Myr"),
        ("Lookback time", f"{cosmology['lookback_time_Gyr']:.6f} Gyr"),
        ("Comoving distance", f"{cosmology['comoving_distance_Gly']:.6f} Gly"),
        ("Exploratory local-peak z", f"{exploratory_z:.6f}"),
    ]
    meta_fig = plt.figure(figsize=(15.5, 7.5), facecolor=BG)
    meta_ax = meta_fig.add_axes([0.04, 0.08, 0.92, 0.84])
    meta_ax.axis("off")
    meta_ax.set_title(f"{VERSION} — {GALAXY} galaxy and archive reference", color=TEXT, fontsize=14.0, fontweight="bold", pad=12)
    meta_table = meta_ax.table(cellText=meta_rows, colLabels=["Field", "Value"], loc="center", cellLoc="left", colWidths=[0.24, 0.72])
    meta_table.auto_set_font_size(False)
    meta_table.set_fontsize(8.5)
    meta_table.scale(1.0, 1.45)
    for (r, c), cell in meta_table.get_celld().items():
        cell.set_edgecolor("#29495f")
        cell.set_linewidth(0.50)
        cell.get_text().set_color(TEXT)
        if r == 0:
            cell.set_facecolor("#123149")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#081523" if r % 2 else "#0b1b2b")
            if c == 0:
                cell.get_text().set_color(CYAN)
                cell.get_text().set_fontweight("bold")
    metadata_png = PNG / f"{VERSION}_{GALAXY}_GALAXY_ARCHIVE_REFERENCE_TABLE.png"
    meta_fig.savefig(metadata_png, dpi=250, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(meta_fig)
    return line_csv, metadata_csv, table_png, metadata_png


def print_table(rows, headers):
    widths = [len(str(header)) for header in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    for package in ["numpy", "pandas", "matplotlib", "astropy"]:
        need(package)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    raw_path, summary_path, observations_path, candidates_path = locate_inputs()
    wave, flux, flux_column = load_raw_spectrum(raw_path)
    archive = load_archive_metadata(summary_path, observations_path, candidates_path)
    rows = line_rows(wave, flux)
    cosmology = cosmology_metadata()
    individual_paths = create_individual_plots(rows, wave, flux)
    dashboard_path, dashboard_slope, dashboard_z = create_dashboard(rows, wave, flux)
    relation_path, relation_slope, relation_z = create_relation_plot(rows)
    line_csv, metadata_csv, table_png, metadata_png = create_tables(rows, archive, cosmology, raw_path, flux_column, relation_slope, relation_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table(
        [
            (int(row.n), row.line_complex, row.rest_wavelengths_A, row.rest_frequencies_THz, row.expected_observed_wavelengths_um, row.expected_observed_frequencies_THz, f"{row.raw_local_peak_um_exploratory:.6f}", f"{row.exploratory_z_from_local_peak:.6f}")
            for row in rows.itertuples()
        ],
        ["#", "Line complex", "Rest A", "Rest THz", "Expected obs um", "Expected obs THz", "Raw peak um", "Exploratory z"],
    )
    print()
    print_table(
        [
            ("Galaxy", GALAXY),
            ("Record status", CONFIRMATION),
            ("Published zspec", f"{Z_SPEC:.2f} -{Z_MINUS:.2f}/+{Z_PLUS:.2f}"),
            ("JWST program", f"GO-{PROGRAM_ID} {PROGRAM_NAME}"),
            ("Instrument", INSTRUMENT),
            ("Coordinates", f"RA {RA_DEG:.7f} deg | Dec {DEC_DEG:.7f} deg"),
            ("Paper", PAPER_REFERENCE),
            ("Raw spectrum CSV", str(raw_path)),
            ("Cached product", archive["product"]),
            ("Universe age", f"{cosmology['universe_age_Myr']:.3f} Myr"),
            ("Comoving distance", f"{cosmology['comoving_distance_Gly']:.6f} Gly"),
            ("Five-line dashboard", str(dashboard_path)),
            ("Rest-vs-observed plot", str(relation_path)),
            ("Results table PNG", str(table_png)),
            ("Galaxy/archive table PNG", str(metadata_png)),
            ("Line CSV", str(line_csv)),
            ("Metadata CSV", str(metadata_csv)),
            ("Exploratory local-peak z", f"{relation_z:.6f}"),
            ("Scientific caution", "published z comes from joint fit; local maxima are exploratory"),
        ],
        ["Field", "Value"],
    )
    print("\nINDIVIDUAL SPECTRAL PLOTS")
    print_table([(index + 1, str(path)) for index, path in enumerate(individual_paths)], ["#", "Path"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
