#!/usr/bin/env python3
"""
JWST_0086_NIST_MOMZ14_FULL_SYNTHETIC_SPECTROGRAPHS.py

Build continuous rest-frame spectrographs from NIST SRD 78 atomic transitions
for the MoM-z14 UV complexes: N IV], N III], C IV, C III], He II and O III].

The NIST database supplies discrete wavelengths and transition-strength fields.
A continuous spectrograph therefore requires an explicit line-profile model.
This script uses Voigt profiles, intrinsic velocity broadening, and optional
NIRSpec-like resolving powers. It is a reference synthesis, not observed flux
and not a plasma-emissivity calculation.
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
        "scipy": "scipy",
        "lxml": "lxml",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from scipy.special import wofz
from urllib3.util.retry import Retry

mpl.rcParams.update({
    "text.usetex": False,
    "mathtext.default": "regular",
    "figure.autolayout": False,
})
plt.close("all")

VERSION = "JWST_0086"
ENDPOINT = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
C_KM_S = 299792.458
C_NM_PHZ = 299.792458
SIGMA_V_KM_S = 85.0
GAMMA_V_KM_S = 12.0
R_HIGH = 2700.0
R_LOW = 1000.0
EPS = 1.0e-30

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    {"key": "N_IV", "spectrum": "N IV", "label": "N IV]", "lo": 147.5, "hi": 149.2,
     "group": "nitrogen", "fallback": [(148.3321, 0.35), (148.6496, 1.00)]},
    {"key": "N_III", "spectrum": "N III", "label": "N III]", "lo": 174.0, "hi": 176.0,
     "group": "nitrogen", "fallback": [(174.6823, .18), (174.8646, .35), (174.9674, 1.0),
                                             (175.2160, .55), (175.3995, .25)]},
    {"key": "C_IV", "spectrum": "C IV", "label": "C IV", "lo": 153.8, "hi": 155.8,
     "group": "carbon", "fallback": [(154.8204, 1.0), (155.0781, .50)]},
    {"key": "C_III", "spectrum": "C III", "label": "C III]", "lo": 189.7, "hi": 191.8,
     "group": "carbon", "fallback": [(190.6683, 1.0), (190.8734, .70)]},
    {"key": "He_II", "spectrum": "He II", "label": "He II", "lo": 163.3, "hi": 164.8,
     "group": "blend", "fallback": [(164.0420, 1.0)]},
    {"key": "O_III", "spectrum": "O III", "label": "O III]", "lo": 165.4, "hi": 167.2,
     "group": "blend", "fallback": [(166.0809, .35), (166.6150, 1.0)]},
]

COLORS = {
    "N IV]": "tab:orange",
    "N III]": "tab:red",
    "C IV": "tab:blue",
    "C III]": "tab:cyan",
    "He II": "tab:purple",
    "O III]": "tab:green",
}


def build_session() -> requests.Session:
    retry = Retry(total=5, connect=5, read=5, backoff_factor=1.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}))
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": f"{VERSION}/Colab NIST-SRD78 reference synthesis",
        "Accept": "text/csv,text/plain,text/html;q=0.8,*/*;q=0.5",
    })
    return session


def query_params(q: dict) -> dict:
    return {
        "spectra": q["spectrum"], "limits_type": "0",
        "low_w": f"{q['lo']:.6f}", "upp_w": f"{q['hi']:.6f}",
        "unit": "1", "submit": "Retrieve Data", "de": "0", "format": "3",
        "line_out": "0", "en_unit": "1", "output": "0", "bibrefs": "1",
        "page_size": "500", "show_obs_wl": "1", "show_calc_wl": "1",
        "unc_out": "1", "show_av": "2", "A_out": "0", "f_out": "on",
        "S_out": "on", "loggf_out": "on", "intens_out": "on",
        "allowed_out": "1", "forbid_out": "1", "conf_out": "on",
        "term_out": "on", "enrg_out": "on", "J_out": "on", "g_out": "on",
    }


def number(value) -> float:
    if value is None:
        return np.nan
    text = str(value).strip().replace('="', '').replace('"', '').replace("−", "-")
    text = re.sub(r"\[[^\]]*\]", "", text)
    match = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?", text)
    try:
        return float(match.group()) if match else np.nan
    except ValueError:
        return np.nan


def normalize_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join(str(part) for part in col if str(part) != "nan") for col in out.columns]
    out.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in out.columns]
    out = out.dropna(axis=1, how="all")
    return out[[col for col in out.columns if not col.lower().startswith("unnamed")]]


def parse_response(text: str) -> pd.DataFrame:
    lower = text.lstrip().lower()
    if lower.startswith("<!doctype") or lower.startswith("<html"):
        tables = [table for table in pd.read_html(io.StringIO(text)) if table.shape[1] >= 2]
        if not tables:
            raise RuntimeError("NIST returned HTML without a usable line table.")
        return normalize_table(max(tables, key=lambda table: table.shape[0] * table.shape[1]))
    try:
        return normalize_table(pd.read_csv(io.StringIO(text), dtype=str, engine="python"))
    except Exception:
        return normalize_table(pd.read_csv(io.StringIO(text), dtype=str, sep="\t", engine="python"))


def find_column(columns, required, rejected=()):
    for column in columns:
        name = str(column).lower()
        if all(token in name for token in required) and not any(token in name for token in rejected):
            return column
    return None


def fallback_frame(q: dict, reason: str) -> pd.DataFrame:
    rows = []
    for wavelength, strength in q["fallback"]:
        rows.append({
            "species": q["label"],
            "nist_spectrum": q["spectrum"],
            "rest_wavelength_vacuum_nm": wavelength,
            "rest_wavelength_vacuum_angstrom": wavelength * 10.0,
            "rest_frequency_PHz": C_NM_PHZ / wavelength,
            "relative_intensity_raw": np.nan,
            "Aki_s^-1": np.nan,
            "fik": np.nan,
            "log_gf": np.nan,
            "plot_strength_raw": strength,
            "plot_strength_source": "curated fallback",
            "wavelength_source": "fallback",
            "retrieval_status": reason,
            "nist_query_url": "",
        })
    frame = pd.DataFrame(rows)
    frame["plot_strength_normalized"] = frame["plot_strength_raw"] / frame["plot_strength_raw"].max()
    return frame


def clean_nist(raw: pd.DataFrame, q: dict, url: str) -> pd.DataFrame:
    columns = list(raw.columns)
    observed = find_column(columns, ("obs", "wavelength")) or find_column(columns, ("observed",))
    ritz = find_column(columns, ("ritz", "wavelength")) or find_column(columns, ("ritz",))
    wavelength = np.full(len(raw), np.nan)
    source = np.full(len(raw), "", dtype=object)
    if observed is not None:
        values = raw[observed].map(number).to_numpy(float)
        mask = np.isfinite(values)
        wavelength[mask] = values[mask]
        source[mask] = "observed"
    if ritz is not None:
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

    def extract(column):
        return raw[column].map(number).to_numpy(float) if column is not None else np.full(len(raw), np.nan)

    intensity, aki, fik, loggf = map(extract, (intensity_col, aki_col, fik_col, loggf_col))
    strength = np.full(len(raw), np.nan)
    basis = np.full(len(raw), "unit fallback", dtype=object)
    options = [
        (intensity, "NIST relative intensity", lambda values: values),
        (aki, "NIST Aki", lambda values: values),
        (fik, "NIST oscillator strength", np.abs),
        (loggf, "NIST log(gf)", lambda values: 10.0 ** np.clip(values, -100.0, 100.0)),
    ]
    for values, label, transform in options:
        transformed = transform(values)
        mask = ~np.isfinite(strength) & np.isfinite(transformed) & (transformed > 0.0)
        strength[mask] = transformed[mask]
        basis[mask] = label
    strength[~np.isfinite(strength) | (strength <= 0.0)] = 1.0

    frame = pd.DataFrame({
        "species": q["label"],
        "nist_spectrum": q["spectrum"],
        "rest_wavelength_vacuum_nm": wavelength,
        "rest_wavelength_vacuum_angstrom": wavelength * 10.0,
        "rest_frequency_PHz": C_NM_PHZ / wavelength,
        "relative_intensity_raw": intensity,
        "Aki_s^-1": aki,
        "fik": fik,
        "log_gf": loggf,
        "plot_strength_raw": strength,
        "plot_strength_source": basis,
        "wavelength_source": source,
        "retrieval_status": "NIST SRD 78",
        "nist_query_url": url,
    })
    frame = frame[np.isfinite(frame["rest_wavelength_vacuum_nm"])]
    frame = frame[frame["rest_wavelength_vacuum_nm"].between(q["lo"], q["hi"])].copy()
    if frame.empty:
        raise RuntimeError("No parseable NIST transitions in requested interval.")
    frame["plot_strength_normalized"] = (
        frame["plot_strength_raw"] / max(float(frame["plot_strength_raw"].max()), EPS)
    )
    return frame.sort_values("rest_wavelength_vacuum_nm").reset_index(drop=True)


def retrieve(session: requests.Session, q: dict) -> tuple[pd.DataFrame, str]:
    try:
        response = session.get(ENDPOINT, params=query_params(q), timeout=(30, 180))
        response.raise_for_status()
        raw_path = CSV_DIR / f"{VERSION}_{q['key']}_NIST_RAW.csv"
        raw_path.write_text(response.text, encoding="utf-8")
        frame = clean_nist(parse_response(response.text), q, response.url)
        status = "NIST SRD 78 retrieved"
    except Exception as exc:
        frame = fallback_frame(q, f"fallback after NIST error: {type(exc).__name__}")
        status = f"fallback after {type(exc).__name__}"
    clean_path = CSV_DIR / f"{VERSION}_{q['key']}_LINES.csv"
    frame.to_csv(clean_path, index=False)
    return frame, status


def voigt_peak_profile(x_nm: np.ndarray, center_nm: float, sigma_nm: float, gamma_nm: float) -> np.ndarray:
    z = ((x_nm - center_nm) + 1j * gamma_nm) / (sigma_nm * np.sqrt(2.0))
    profile = np.real(wofz(z)) / (sigma_nm * np.sqrt(2.0 * np.pi))
    peak = max(float(np.nanmax(profile)), EPS)
    return profile / peak


def synthesize(frame: pd.DataFrame, lo: float, hi: float, resolving_power: float | None,
               samples: int = 14000) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(lo, hi, samples)
    y = np.full_like(x, 0.025)
    strongest = frame.nlargest(min(120, len(frame)), "plot_strength_normalized")
    for row in strongest.itertuples(index=False):
        center = float(row.rest_wavelength_vacuum_nm)
        amplitude = float(row.plot_strength_normalized)
        sigma_intrinsic = center * SIGMA_V_KM_S / C_KM_S
        sigma_instrument = 0.0 if resolving_power is None else center / (2.354820045 * resolving_power)
        sigma_total = max(np.hypot(sigma_intrinsic, sigma_instrument), 1.0e-7)
        gamma = max(center * GAMMA_V_KM_S / C_KM_S, 1.0e-8)
        y += amplitude * voigt_peak_profile(x, center, sigma_total, gamma)
    y /= max(float(np.nanmax(y)), EPS)
    return x, y


def detector_sample(x: np.ndarray, y: np.ndarray, lo: float, hi: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    center = 0.5 * (lo + hi)
    resolution_width = center / R_HIGH
    step = resolution_width / 2.3
    xs = np.arange(lo, hi + step, step)
    ys = np.interp(xs, x, y)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.012, size=ys.size)
    return xs, np.clip(ys + noise, 0.0, None)


def frequency_forward(wavelength_nm):
    values = np.asarray(wavelength_nm, dtype=float)
    return C_NM_PHZ / np.maximum(values, EPS)


def frequency_inverse(frequency_phz):
    values = np.asarray(frequency_phz, dtype=float)
    return C_NM_PHZ / np.maximum(values, EPS)


def plot_species_axis(ax, q: dict, frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    lo, hi = q["lo"], q["hi"]
    x_native, y_native = synthesize(frame, lo, hi, None)
    x_r2700, y_r2700 = synthesize(frame, lo, hi, R_HIGH)
    x_r1000, y_r1000 = synthesize(frame, lo, hi, R_LOW)
    xs, ys = detector_sample(x_r2700, y_r2700, lo, hi, seed)

    color = COLORS[q["label"]]
    ax.plot(x_native, y_native, lw=1.15, alpha=0.75, label="intrinsic Voigt profile")
    ax.plot(x_r2700, y_r2700, lw=2.0, color=color, label="R=2700 reference spectrum")
    ax.plot(x_r1000, y_r1000, lw=1.5, linestyle="--", label="R=1000 reference spectrum")
    ax.plot(xs, ys, lw=0.75, alpha=0.55, label="simulated detector sampling")

    selected = frame.nlargest(min(12, len(frame)), "plot_strength_normalized")
    for row in selected.itertuples(index=False):
        wavelength = float(row.rest_wavelength_vacuum_nm)
        strength = float(row.plot_strength_normalized)
        ax.axvline(wavelength, ymin=0.0, ymax=min(0.18 + 0.15 * strength, 0.40),
                   color="0.75", lw=0.55, alpha=0.45)
    strongest = selected.nlargest(min(6, len(selected)), "plot_strength_normalized")
    for index, row in enumerate(strongest.sort_values("rest_wavelength_vacuum_nm").itertuples(index=False)):
        wavelength = float(row.rest_wavelength_vacuum_nm)
        ax.annotate(f"{wavelength:.5f} nm", xy=(wavelength, 0.88),
                    xytext=(wavelength, 1.03 + 0.075 * (index % 2)),
                    rotation=90, ha="center", va="bottom", fontsize=7.0,
                    arrowprops={"arrowstyle": "-", "lw": 0.45, "alpha": 0.7},
                    clip_on=False)

    ax.set_xlim(lo, hi)
    ax.set_ylim(0.0, 1.23)
    ax.set_ylabel("normalized reference flux")
    ax.set_title(
        f"{q['label']} continuous rest-frame spectrograph | {len(frame)} NIST transitions\n"
        f"Voigt broadening: sigma_v={SIGMA_V_KM_S:.0f} km/s, gamma_v={GAMMA_V_KM_S:.0f} km/s",
        fontsize=12,
    )
    ax.grid(alpha=0.20)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    secondary = ax.secondary_xaxis("top", functions=(frequency_forward, frequency_inverse))
    secondary.set_xlabel("rest frequency [PHz]")

    return pd.DataFrame({
        "species": q["label"],
        "rest_wavelength_nm": x_native,
        "intrinsic_normalized_flux": y_native,
        "R2700_normalized_flux": y_r2700,
        "R1000_normalized_flux": y_r1000,
    })


def group_figure(data: dict[str, pd.DataFrame], keys: tuple[str, ...], title: str, filename: str,
                 seed_offset: int) -> tuple[Path, pd.DataFrame]:
    qmap = {q["key"]: q for q in QUERIES}
    plt.style.use("dark_background")
    mpl.rcParams["text.usetex"] = False
    fig, axes = plt.subplots(len(keys), 1, figsize=(16, 6.4 * len(keys)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    spectra = []
    for index, (axis, key) in enumerate(zip(axes, keys)):
        spectra.append(plot_species_axis(axis, qmap[key], data[key], seed_offset + index))
        axis.set_xlabel("rest vacuum wavelength [nm]")
    fig.suptitle(
        title + "\nContinuous NIST-based line-profile synthesis — not an observed galaxy spectrum",
        fontsize=17,
    )
    output = PNG_DIR / filename
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output, pd.concat(spectra, ignore_index=True)


def blend_figure(data: dict[str, pd.DataFrame]) -> tuple[Path, pd.DataFrame]:
    lo, hi = 162.8, 167.4
    plt.style.use("dark_background")
    mpl.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize=(16, 8.5), constrained_layout=True)
    output_rows = []
    total_r2700 = None
    x_common = None

    for index, key in enumerate(("He_II", "O_III")):
        q = next(item for item in QUERIES if item["key"] == key)
        frame = data[key]
        x_native, y_native = synthesize(frame, lo, hi, None)
        x_r2700, y_r2700 = synthesize(frame, lo, hi, R_HIGH)
        _, y_r1000 = synthesize(frame, lo, hi, R_LOW)
        color = COLORS[q["label"]]
        ax.plot(x_native, y_native, lw=0.9, alpha=0.35, color=color)
        ax.plot(x_r2700, y_r2700, lw=2.1, color=color, label=f"{q['label']} R=2700")
        ax.plot(x_r2700, y_r1000, lw=1.25, linestyle="--", color=color, alpha=0.85,
                label=f"{q['label']} R=1000")
        contribution = np.clip(y_r2700 - 0.025, 0.0, None)
        total_r2700 = contribution if total_r2700 is None else total_r2700 + contribution
        x_common = x_r2700
        output_rows.append(pd.DataFrame({
            "species": q["label"],
            "rest_wavelength_nm": x_r2700,
            "R2700_normalized_flux": y_r2700,
            "R1000_normalized_flux": y_r1000,
        }))

    total_r2700 = 0.025 + total_r2700
    total_r2700 /= max(float(total_r2700.max()), EPS)
    sampled_x, sampled_y = detector_sample(x_common, total_r2700, lo, hi, 8606)
    ax.plot(x_common, total_r2700, color="white", lw=2.5,
            label="equal-ion He II + O III] composite")
    ax.plot(sampled_x, sampled_y, color="0.75", lw=0.75, alpha=0.55,
            label="simulated detector sampling")

    for key in ("He_II", "O_III"):
        q = next(item for item in QUERIES if item["key"] == key)
        strongest = data[key].nlargest(min(5, len(data[key])), "plot_strength_normalized")
        for index, row in enumerate(strongest.sort_values("rest_wavelength_vacuum_nm").itertuples(index=False)):
            wavelength = float(row.rest_wavelength_vacuum_nm)
            ax.annotate(f"{q['label']} {wavelength:.5f}", xy=(wavelength, 0.90),
                        xytext=(wavelength, 1.04 + 0.08 * (index % 2)), rotation=90,
                        ha="center", va="bottom", fontsize=7.0,
                        arrowprops={"arrowstyle": "-", "lw": 0.45}, clip_on=False)

    ax.set(xlim=(lo, hi), ylim=(0.0, 1.25), xlabel="rest vacuum wavelength [nm]",
           ylabel="normalized reference flux")
    ax.set_title(
        "He II + O III] continuous rest-frame blend spectrograph\n"
        "Each ion normalized independently before equal-ion combination",
        fontsize=14,
    )
    ax.grid(alpha=0.20)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.secondary_xaxis("top", functions=(frequency_forward, frequency_inverse)).set_xlabel("rest frequency [PHz]")
    fig.suptitle(
        "MoM-z14 reference study — NIST-based He II and O III] spectral blend\n"
        "Synthetic line-profile spectrum, not observed flux",
        fontsize=17,
    )
    output = PNG_DIR / f"{VERSION}_HEII_OIII_FULL_SPECTROGRAPH.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)

    total_frame = pd.DataFrame({
        "species": "He II + O III] equal-ion composite",
        "rest_wavelength_nm": x_common,
        "R2700_normalized_flux": total_r2700,
        "R1000_normalized_flux": np.nan,
    })
    return output, pd.concat(output_rows + [total_frame], ignore_index=True)


def overview_figure(data: dict[str, pd.DataFrame]) -> Path:
    lo, hi = 145.0, 193.0
    plt.style.use("dark_background")
    mpl.rcParams["text.usetex"] = False
    fig, ax = plt.subplots(figsize=(18, 8), constrained_layout=True)
    offsets = np.linspace(0.0, 4.5, len(QUERIES))
    for offset, q in zip(offsets, QUERIES):
        x, y = synthesize(data[q["key"]], lo, hi, R_HIGH, samples=22000)
        y_display = offset + 0.72 * y
        ax.plot(x, y_display, lw=1.45, color=COLORS[q["label"]], label=q["label"])
        ax.text(hi + 0.25, offset + 0.25, q["label"], color=COLORS[q["label"]], va="center")
    ax.set_xlim(lo, hi + 2.2)
    ax.set_ylim(-0.1, offsets[-1] + 1.0)
    ax.set_xlabel("rest vacuum wavelength [nm]")
    ax.set_ylabel("stacked normalized reference spectra")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.18)
    ax.set_title(
        "MoM-z14 rest-UV spectral atlas — continuous NIST-based R=2700 reference spectrographs\n"
        "Species are independently normalized and vertically offset",
        fontsize=16,
    )
    ax.secondary_xaxis("top", functions=(frequency_forward, frequency_inverse)).set_xlabel("rest frequency [PHz]")
    output = PNG_DIR / f"{VERSION}_FULL_REST_UV_SPECTRAL_ATLAS.png"
    fig.savefig(output, dpi=360, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)
    return output


def diagnostics_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for q in QUERIES:
        frame = data[q["key"]]
        strongest = frame.loc[frame["plot_strength_normalized"].idxmax()]
        wavelength = float(strongest["rest_wavelength_vacuum_nm"])
        intrinsic_fwhm_nm = 2.354820045 * wavelength * SIGMA_V_KM_S / C_KM_S
        r2700_fwhm_nm = np.hypot(intrinsic_fwhm_nm, wavelength / R_HIGH)
        r1000_fwhm_nm = np.hypot(intrinsic_fwhm_nm, wavelength / R_LOW)
        rows.append({
            "species": q["label"],
            "transition_count": len(frame),
            "strongest_rest_wavelength_nm": wavelength,
            "strongest_rest_frequency_PHz": C_NM_PHZ / wavelength,
            "intrinsic_gaussian_FWHM_nm": intrinsic_fwhm_nm,
            "R2700_effective_FWHM_nm": r2700_fwhm_nm,
            "R1000_effective_FWHM_nm": r1000_fwhm_nm,
            "strength_basis_examples": "; ".join(sorted(set(frame["plot_strength_source"].astype(str)))[:4]),
            "retrieval_status": "; ".join(sorted(set(frame["retrieval_status"].astype(str)))),
        })
    return pd.DataFrame(rows)


def main() -> None:
    session = build_session()
    data: dict[str, pd.DataFrame] = {}
    manifest = []

    for index, q in enumerate(QUERIES):
        frame, status = retrieve(session, q)
        data[q["key"]] = frame
        manifest.append({
            "species": q["label"],
            "nist_spectrum": q["spectrum"],
            "query_low_nm": q["lo"],
            "query_high_nm": q["hi"],
            "line_count": len(frame),
            "status": status,
        })
        if index < len(QUERIES) - 1:
            time.sleep(0.8)

    master_lines = pd.concat(data.values(), ignore_index=True)
    master_path = CSV_DIR / f"{VERSION}_NIST_MASTER_LINE_TABLE.csv"
    master_lines.to_csv(master_path, index=False)

    nitrogen_png, nitrogen_spectra = group_figure(
        data, ("N_IV", "N_III"),
        "MoM-z14 nitrogen full spectral analysis",
        f"{VERSION}_NITROGEN_FULL_SPECTROGRAPHS.png", 8600,
    )
    carbon_png, carbon_spectra = group_figure(
        data, ("C_IV", "C_III"),
        "MoM-z14 carbon full spectral analysis",
        f"{VERSION}_CARBON_FULL_SPECTROGRAPHS.png", 8610,
    )
    blend_png, blend_spectra = blend_figure(data)
    atlas_png = overview_figure(data)

    spectra_path = CSV_DIR / f"{VERSION}_CONTINUOUS_SPECTRA.csv"
    pd.concat([nitrogen_spectra, carbon_spectra, blend_spectra], ignore_index=True).to_csv(
        spectra_path, index=False
    )
    diagnostics = diagnostics_table(data)
    diagnostics_path = CSV_DIR / f"{VERSION}_SPECTRAL_DIAGNOSTICS.csv"
    diagnostics.to_csv(diagnostics_path, index=False)
    manifest_path = CSV_DIR / f"{VERSION}_NIST_QUERY_MANIFEST.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print("SOURCE          NIST Standard Reference Database 78")
    print("PRODUCT         continuous Voigt-profile rest-frame spectrographs")
    print("STATUS          reference synthesis; not observed MoM-z14 flux")
    print(f"BROADENING      sigma_v={SIGMA_V_KM_S:.1f} km/s  gamma_v={GAMMA_V_KM_S:.1f} km/s")
    print(f"RESOLUTION      intrinsic, R={R_HIGH:.0f}, R={R_LOW:.0f}")
    for row in manifest:
        print(f"{row['species']:<8} {row['line_count']:>4} transitions  {row['status']}")
    print(f"NITROGEN PNG    {nitrogen_png}")
    print(f"CARBON PNG      {carbon_png}")
    print(f"BLEND PNG       {blend_png}")
    print(f"ATLAS PNG       {atlas_png}")
    print(f"LINES CSV       {master_path}")
    print(f"SPECTRA CSV     {spectra_path}")
    print(f"DIAGNOSTICS CSV {diagnostics_path}")
    print(f"MANIFEST CSV    {manifest_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
