#!/usr/bin/env python3
"""
JWST_0095_NIST_SRD78_FULL_REFERENCE_SPECTRA_FIVE_GROUPS.py

Download broad NIST SRD 78 transition tables for six rest-UV ionic families and
render five publication-style figures. Every returned NIST transition inside
its configured wavelength interval is retained. The continuous curves are an
explicit R=100,000 line-spread visualization of the NIST line list, not an
observed stellar or galaxy flux spectrum.

No AI images. Python + pandas + NumPy + matplotlib only.
"""
from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "requests": "requests",
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "lxml": "lxml",
        "PIL": "pillow",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION = "JWST_0095"
ENDPOINT = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
C_NM_PHZ = 299.792458
R_VISUAL = 100_000.0

ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
for directory in (PNG, CSV, DATA):
    directory.mkdir(parents=True, exist_ok=True)

ELEMENT_COLORS = {
    "Nitrogen": "#ff4fd8",
    "Carbon": "#32d6ff",
    "Helium": "#b98cff",
    "Oxygen": "#4f8cff",
}

SPECIES = [
    {
        "key": "N_IV", "spectrum": "N IV", "label": "N IV]", "element": "Nitrogen",
        "ion": "N³⁺", "lo": 140.0, "hi": 156.0,
        "targets": [148.3321, 148.6496],
    },
    {
        "key": "C_IV", "spectrum": "C IV", "label": "C IV", "element": "Carbon",
        "ion": "C³⁺", "lo": 140.0, "hi": 156.0,
        "targets": [154.82043, 155.0784],
    },
    {
        "key": "He_II", "spectrum": "He II", "label": "He II", "element": "Helium",
        "ion": "He⁺", "lo": 160.0, "hi": 165.0,
        "targets": [164.042],
    },
    {
        "key": "O_III", "spectrum": "O III", "label": "O III]", "element": "Oxygen",
        "ion": "O²⁺", "lo": 165.0, "hi": 168.0,
        "targets": [166.0809, 166.615],
    },
    {
        "key": "N_III", "spectrum": "N III", "label": "N III]", "element": "Nitrogen",
        "ion": "N²⁺", "lo": 170.0, "hi": 176.0,
        "targets": [174.6823, 174.8646, 174.9674, 175.216, 175.399],
    },
    {
        "key": "C_III", "spectrum": "C III", "label": "C III]", "element": "Carbon",
        "ion": "C²⁺", "lo": 188.0, "hi": 192.0,
        "targets": [190.6683, 190.8734],
    },
]

FIGURES = [
    ("N_IV",),
    ("C_IV",),
    ("He_II", "O_III"),
    ("N_III",),
    ("C_III",),
]


def reset_style() -> None:
    plt.close("all")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "text.usetex": False,
        "figure.facecolor": "#05070d",
        "axes.facecolor": "#07111f",
        "axes.edgecolor": "#8fa7c3",
        "axes.labelcolor": "#eef5ff",
        "xtick.color": "#d9e7f8",
        "ytick.color": "#d9e7f8",
        "text.color": "#f6f9ff",
        "font.size": 10,
        "axes.titleweight": "semibold",
        "savefig.facecolor": "#05070d",
    })


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": f"{VERSION} NIST-SRD78 reference-atlas client",
        "Accept": "text/plain,text/csv,text/html;q=0.7,*/*;q=0.2",
    })
    return session


def query_params(item: dict) -> dict:
    return {
        "spectra": item["spectrum"],
        "limits_type": "0",
        "low_w": f"{item['lo']:.6f}",
        "upp_w": f"{item['hi']:.6f}",
        "unit": "1",
        "submit": "Retrieve Data",
        "de": "0",
        "format": "3",
        "line_out": "0",
        "remove_js": "on",
        "en_unit": "1",
        "output": "0",
        "bibrefs": "1",
        "page_size": "5000",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "show_diff_obs_calc": "1",
        "unc_out": "1",
        "order_out": "0",
        "show_av": "2",
        "A_out": "0",
        "f_out": "on",
        "S_out": "on",
        "loggf_out": "on",
        "intens_out": "on",
        "allowed_out": "1",
        "forbid_out": "1",
        "conf_out": "on",
        "term_out": "on",
        "enrg_out": "on",
        "J_out": "on",
        "g_out": "on",
    }


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join(str(part) for part in col if str(part) != "nan")
                       for col in out.columns]
    out.columns = [re.sub(r"\s+", " ", str(col)).strip().replace("\ufeff", "")
                   for col in out.columns]
    out = out.dropna(axis=1, how="all")
    return out[[col for col in out.columns if not col.lower().startswith("unnamed")]]


def parse_table(text: str) -> tuple[pd.DataFrame, str]:
    stripped = text.lstrip().lower()
    if stripped.startswith("<!doctype") or stripped.startswith("<html"):
        tables = [table for table in pd.read_html(io.StringIO(text)) if table.shape[1] >= 2]
        if not tables:
            raise RuntimeError("NIST returned HTML without a usable transition table.")
        return normalize_columns(max(tables, key=lambda table: table.shape[0] * table.shape[1])), "HTML"

    attempts = [
        ("TAB", {"sep": "\t"}),
        ("CSV", {"sep": ","}),
        ("PIPE", {"sep": "|"}),
    ]
    for mode, kwargs in attempts:
        try:
            frame = pd.read_csv(
                io.StringIO(text), dtype=str, engine="python", keep_default_na=False,
                on_bad_lines="skip", **kwargs,
            )
            frame = normalize_columns(frame)
            if frame.shape[1] >= 2 and len(frame) > 0:
                return frame, mode
        except Exception:
            continue
    raise RuntimeError("Could not parse the NIST response as a tabular dataset.")


def number(value) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().replace('="', "").replace('"', "").replace("−", "-")
    text = re.sub(r"\[[^\]]*\]", "", text)
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", text)
    try:
        return float(match.group()) if match else np.nan
    except ValueError:
        return np.nan


def find_column(columns, required, rejected=()):
    for column in columns:
        lower = str(column).lower()
        if all(token in lower for token in required) and not any(token in lower for token in rejected):
            return column
    return None


def canonicalize(raw: pd.DataFrame, item: dict, query_url: str) -> pd.DataFrame:
    columns = list(raw.columns)
    observed = (
        find_column(columns, ("obs", "wavelength"))
        or find_column(columns, ("observed",))
        or find_column(columns, ("obs", "wl"))
    )
    ritz = (
        find_column(columns, ("ritz", "wavelength"))
        or find_column(columns, ("ritz",))
        or find_column(columns, ("calc", "wavelength"))
    )

    wavelength = np.full(len(raw), np.nan)
    source = np.full(len(raw), "", dtype=object)
    if observed:
        values = raw[observed].map(number).to_numpy(float)
        mask = np.isfinite(values)
        wavelength[mask] = values[mask]
        source[mask] = "observed"
    if ritz:
        values = raw[ritz].map(number).to_numpy(float)
        mask = ~np.isfinite(wavelength) & np.isfinite(values)
        wavelength[mask] = values[mask]
        source[mask] = "Ritz"
    if not np.isfinite(wavelength).any():
        for column in [col for col in columns if "wave" in str(col).lower()]:
            values = raw[column].map(number).to_numpy(float)
            mask = ~np.isfinite(wavelength) & np.isfinite(values)
            wavelength[mask] = values[mask]
            source[mask] = "listed"

    intensity_col = find_column(columns, ("rel", "int")) or find_column(columns, ("intens",))
    aki_col = find_column(columns, ("aki",)) or find_column(columns, ("a", "s-1"))
    fik_col = find_column(columns, ("fik",)) or find_column(columns, ("osc",))
    loggf_col = find_column(columns, ("log", "gf"))

    def values(column):
        return raw[column].map(number).to_numpy(float) if column else np.full(len(raw), np.nan)

    intensity = values(intensity_col)
    aki = values(aki_col)
    fik = values(fik_col)
    loggf = values(loggf_col)

    strength = np.full(len(raw), np.nan)
    strength_source = np.full(len(raw), "unit fallback", dtype=object)
    candidates = [
        (intensity, "NIST relative intensity", lambda array: array),
        (aki, "NIST Aki", lambda array: array),
        (fik, "NIST oscillator strength", np.abs),
        (loggf, "NIST log(gf)", lambda array: 10.0 ** np.clip(array, -100.0, 100.0)),
    ]
    for candidate, label, transform in candidates:
        transformed = transform(candidate)
        mask = ~np.isfinite(strength) & np.isfinite(transformed) & (transformed > 0)
        strength[mask] = transformed[mask]
        strength_source[mask] = label
    strength[~np.isfinite(strength) | (strength <= 0)] = 1.0

    output = pd.DataFrame({
        "selected_family": item["label"],
        "nist_spectrum": item["spectrum"],
        "element": item["element"],
        "ion_stage": item["ion"],
        "rest_wavelength_vacuum_nm": wavelength,
        "rest_wavelength_vacuum_angstrom": wavelength * 10.0,
        "rest_frequency_PHz": C_NM_PHZ / wavelength,
        "wavelength_source": source,
        "relative_intensity": intensity,
        "Aki_s^-1": aki,
        "fik": fik,
        "log_gf": loggf,
        "reference_strength_raw": strength,
        "reference_strength_source": strength_source,
        "nist_query_url": query_url,
    })
    output = output[np.isfinite(output["rest_wavelength_vacuum_nm"])].copy()
    output = output[output["rest_wavelength_vacuum_nm"].between(item["lo"], item["hi"])].copy()
    if output.empty:
        raise RuntimeError(f"No parseable {item['spectrum']} transitions in {item['lo']}-{item['hi']} nm.")

    # Retain every transition, including coincident fine-structure records.
    maximum = float(output["reference_strength_raw"].max())
    output["reference_strength_normalized"] = output["reference_strength_raw"] / max(maximum, 1e-300)
    return output.sort_values("rest_wavelength_vacuum_nm").reset_index(drop=True)


def retrieve(session: requests.Session, item: dict) -> tuple[pd.DataFrame, str]:
    response = session.get(ENDPOINT, params=query_params(item), timeout=(30, 240))
    response.raise_for_status()
    text = response.text
    raw_path = DATA / f"{VERSION}_{item['key']}_NIST_RAW.txt"
    raw_path.write_text(text, encoding="utf-8")
    raw, mode = parse_table(text)
    full_path = CSV / f"{VERSION}_{item['key']}_FULL_TABLE.csv"
    raw.to_csv(full_path, index=False)
    clean = canonicalize(raw, item, response.url)
    clean_path = CSV / f"{VERSION}_{item['key']}_CANONICAL.csv"
    clean.to_csv(clean_path, index=False)
    return clean, mode


def reference_response(frame: pd.DataFrame, lo: float, hi: float):
    x = np.linspace(lo, hi, 36_000)
    optical_depth = np.zeros_like(x)
    for wavelength, normalized_strength in zip(
        frame["rest_wavelength_vacuum_nm"].to_numpy(float),
        frame["reference_strength_normalized"].to_numpy(float),
    ):
        sigma = max(wavelength / R_VISUAL / 2.354820045, 0.00020)
        depth = 2.3 * np.sqrt(max(normalized_strength, 1e-12))
        optical_depth += depth * np.exp(-0.5 * ((x - wavelength) / sigma) ** 2)
    flux = np.exp(-optical_depth)
    return x, np.clip(flux, 0.01, 1.005)


def wavelength_to_frequency(values):
    return C_NM_PHZ / np.asarray(values, dtype=float)


def frequency_to_wavelength(values):
    return C_NM_PHZ / np.asarray(values, dtype=float)


def style_axis(axis) -> None:
    axis.grid(True, color="#4b6077", linewidth=0.45, alpha=0.34)
    axis.minorticks_on()
    axis.tick_params(which="major", direction="in", top=True, right=True, length=5, width=0.75)
    axis.tick_params(which="minor", direction="in", top=True, right=True, length=2.8, width=0.55)
    for spine in axis.spines.values():
        spine.set_color("#9bb0c8")
        spine.set_linewidth(0.75)


def annotate_strongest(axis, frame: pd.DataFrame, color: str, count: int = 8) -> None:
    strongest = frame.nlargest(min(count, len(frame)), "reference_strength_normalized")
    strongest = strongest.sort_values("rest_wavelength_vacuum_nm")
    for index, row in enumerate(strongest.itertuples(index=False)):
        wavelength = float(row.rest_wavelength_vacuum_nm)
        label = f"{row.selected_family}  {wavelength:.5f} nm"
        y_text = 0.18 + 0.12 * (index % 4)
        axis.annotate(
            label,
            xy=(wavelength, 0.96),
            xytext=(wavelength, y_text),
            textcoords="data",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=7.3,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.55, "alpha": 0.8},
            clip_on=True,
        )


def draw_species(axis, frame: pd.DataFrame, item: dict) -> dict:
    color = ELEMENT_COLORS[item["element"]]
    x, response = reference_response(frame, item["lo"], item["hi"])

    axis.plot(
        x, response,
        color=color,
        linewidth=0.82,
        antialiased=True,
        label=f"{item['label']} high-resolution reference response",
    )

    # Every NIST transition appears as a fine rug line; none are filtered out.
    wavelengths = frame["rest_wavelength_vacuum_nm"].to_numpy(float)
    strengths = frame["reference_strength_normalized"].to_numpy(float)
    rug_top = 0.025 + 0.075 * np.sqrt(np.clip(strengths, 0, 1))
    axis.vlines(wavelengths, 0.01, rug_top, color=color, linewidth=0.45, alpha=0.88)

    for target in item["targets"]:
        if item["lo"] <= target <= item["hi"]:
            axis.axvline(target, color="#ffffff", linewidth=0.62, linestyle="--", alpha=0.72)

    axis.set_xlim(item["lo"], item["hi"])
    axis.set_ylim(0.0, 1.04)
    axis.set_ylabel("Normalized reference response")
    axis.set_title(
        f"{item['label']}  ({item['ion']}, {item['element']}) — complete NIST SRD 78 interval\n"
        f"{item['lo']:.1f}–{item['hi']:.1f} nm | all {len(frame)} returned transitions retained",
        fontsize=12.2,
        pad=10,
    )
    style_axis(axis)
    annotate_strongest(axis, frame, color, count=9)

    top = axis.secondary_xaxis("top", functions=(wavelength_to_frequency, frequency_to_wavelength))
    top.set_xlabel("Rest frequency [PHz]")
    top.tick_params(colors="#d9e7f8", labelsize=8)

    axis.text(
        0.995, 0.035,
        "NIST line list + R=100,000 line-spread visualization\nNot an observed stellar or galaxy flux spectrum",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.8,
        color="#cbd8e8",
        bbox={"facecolor": "#06101d", "edgecolor": "#52677f", "alpha": 0.90, "pad": 4},
    )

    variance = float(np.var(response))
    return {
        "key": item["key"],
        "family": item["label"],
        "element": item["element"],
        "range_low_nm": item["lo"],
        "range_high_nm": item["hi"],
        "transition_count": len(frame),
        "response_variance": variance,
        "wavelength_min_returned_nm": float(frame["rest_wavelength_vacuum_nm"].min()),
        "wavelength_max_returned_nm": float(frame["rest_wavelength_vacuum_nm"].max()),
    }


def make_figure(data: dict[str, pd.DataFrame], keys: tuple[str, ...], figure_number: int):
    lookup = {item["key"]: item for item in SPECIES}
    height = 7.2 if len(keys) == 1 else 12.0
    figure, axes = plt.subplots(len(keys), 1, figsize=(17.5, height), constrained_layout=True)
    axes = np.atleast_1d(axes)
    validation_rows = []

    for axis, key in zip(axes, keys):
        validation_rows.append(draw_species(axis, data[key], lookup[key]))
        axis.set_xlabel("Rest vacuum wavelength [nm]")

    title_labels = " + ".join(lookup[key]["label"] for key in keys)
    figure.suptitle(
        f"NIST SRD 78 REST-UV REFERENCE SPECTRUM — {title_labels}\n"
        "Publication-style transition atlas with full configured wavelength coverage",
        fontsize=17.5,
        y=1.02,
    )
    filename = f"{VERSION}_{figure_number:02d}_{'_'.join(keys)}_FULL_REFERENCE_SPECTRUM.png"
    output = PNG / filename
    figure.savefig(output, dpi=420, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)

    with Image.open(output) as image:
        width_px, height_px = image.size
    size_bytes = output.stat().st_size
    for row in validation_rows:
        row.update({
            "png_path": str(output),
            "png_width_px": width_px,
            "png_height_px": height_px,
            "png_size_bytes": size_bytes,
            "validation": "PASS" if (
                row["transition_count"] > 0
                and row["response_variance"] > 1e-10
                and width_px >= 3000
                and height_px >= 1600
                and size_bytes >= 80_000
            ) else "FAIL",
        })
    return output, validation_rows


def main() -> None:
    reset_style()
    session = make_session()
    data = {}
    download_rows = []

    for index, item in enumerate(SPECIES, start=1):
        print(f"DOWNLOAD {index}/6  {item['spectrum']:<6}  {item['lo']:.1f}-{item['hi']:.1f} nm")
        frame, mode = retrieve(session, item)
        data[item["key"]] = frame
        download_rows.append({
            "key": item["key"],
            "family": item["label"],
            "element": item["element"],
            "range_low_nm": item["lo"],
            "range_high_nm": item["hi"],
            "returned_transitions": len(frame),
            "parse_mode": mode,
        })
        print(f"  transitions={len(frame):4d}  mode={mode}")
        if index < len(SPECIES):
            time.sleep(0.8)

    master = pd.concat([data[item["key"]] for item in SPECIES], ignore_index=True, sort=False)
    master_path = CSV / f"{VERSION}_NIST_SRD78_BROAD_REFERENCE_MASTER.csv"
    master.to_csv(master_path, index=False)

    download_manifest = CSV / f"{VERSION}_DOWNLOAD_MANIFEST.csv"
    pd.DataFrame(download_rows).to_csv(download_manifest, index=False)

    outputs = []
    validations = []
    for figure_number, keys in enumerate(FIGURES, start=1):
        output, rows = make_figure(data, keys, figure_number)
        outputs.append(output)
        validations.extend(rows)

    validation_frame = pd.DataFrame(validations)
    validation_path = CSV / f"{VERSION}_PLOT_VALIDATION.csv"
    validation_frame.to_csv(validation_path, index=False)

    failed = validation_frame[validation_frame["validation"] != "PASS"]
    if not failed.empty:
        raise RuntimeError(
            "Plot validation failed for: " + ", ".join(failed["family"].astype(str))
            + f". See {validation_path}"
        )

    print()
    print(f"CODE OUTPUT: {VERSION}")
    print("DATABASE        NIST Standard Reference Database 78")
    print("FIGURES         five high-resolution PNG files; six ionic panels")
    print("TRANSITIONS     every returned NIST line retained inside each full interval")
    print("CURVE           R=100,000 line-spread visualization; not observed flux")
    print()
    print(f"{'FAMILY':<10} {'LINES':>7} {'RANGE [nm]':>17} {'PNG':>6}")
    print("-" * 46)
    for row in validations:
        bounds = f"{row['range_low_nm']:.1f}-{row['range_high_nm']:.1f}"
        print(f"{row['family']:<10} {row['transition_count']:>7} {bounds:>17} {row['validation']:>6}")
    print()
    for index, output in enumerate(outputs, start=1):
        print(f"PLOT {index:02d}         {output}")
    print(f"MASTER CSV      {master_path}")
    print(f"DOWNLOAD CSV    {download_manifest}")
    print(f"VALIDATION CSV  {validation_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
