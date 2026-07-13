#!/usr/bin/env python3
"""
JWST_0087_MOMZ14_ACTUAL_NIRSPEC_RAW_SPECTROGRAPH.py

Retrieve the coordinate-verified public JWST/NIRSpec X1D extraction of MoM-z14
from MAST GO-5224 and plot the actual unsmoothed, jagged detector spectrum.
No synthetic profiles, no Gaussian toy traces, no AI images.
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "astropy": "astropy",
        "astroquery": "astroquery",
        "requests": "requests",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astroquery.mast import Observations

VERSION = "JWST_0087"
GALAXY = "MoM-z14"
MOM_RA = 150.0933255
MOM_DEC = 2.2731627
MOM_Z = 14.44
JWST_PID = "5224"
MAX_X1D = 24

OUT = Path("/content/JWST_OUTPUT")
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION
for directory in (PNG, CSV, DATA):
    directory.mkdir(parents=True, exist_ok=True)

COMPLEXES = [
    {
        "key": "N_IV",
        "title": "N IV] 1483, 1487",
        "window_A": (1478.0, 1492.0),
        "lines": [("N IV] 1483", 1483.32), ("N IV] 1487", 1486.50)],
    },
    {
        "key": "C_IV",
        "title": "C IV 1548, 1551",
        "window_A": (1542.0, 1557.0),
        "lines": [("C IV 1548", 1548.20), ("C IV 1551", 1550.77)],
    },
    {
        "key": "HE_O",
        "title": "He II 1640 + O III] 1661, 1666",
        "window_A": (1633.0, 1673.0),
        "lines": [
            ("He II 1640", 1640.42),
            ("O III] 1661", 1660.81),
            ("O III] 1666", 1666.15),
        ],
    },
    {
        "key": "N_III",
        "title": "N III] 1747-1754",
        "window_A": (1742.0, 1759.0),
        "lines": [
            ("N III] 1747", 1746.82),
            ("N III] 1749", 1748.65),
            ("N III] 1750", 1749.67),
            ("N III] 1752", 1752.16),
            ("N III] 1754", 1753.99),
        ],
    },
    {
        "key": "C_III",
        "title": "C III] 1907, 1909",
        "window_A": (1901.0, 1914.0),
        "lines": [("C III] 1907", 1906.68), ("C III] 1909", 1908.73)],
    },
]


def reset_matplotlib() -> None:
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


def col(frame: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def fnum(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def query_coordinate_matched_observations() -> pd.DataFrame:
    target = SkyCoord(MOM_RA * u.deg, MOM_DEC * u.deg)
    for radius_arcsec in (3.0, 10.0, 30.0):
        table = Observations.query_region(target, radius=radius_arcsec * u.arcsec)
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
        if frame.empty:
            continue
        proposal_col = col(frame, ["proposal_id", "proposalid"])
        collection_col = col(frame, ["obs_collection"])
        instrument_col = col(frame, ["instrument_name", "instrument"])
        mask = pd.Series(True, index=frame.index)
        if proposal_col is not None:
            proposal = frame[proposal_col].astype(str).str.replace(".0", "", regex=False)
            mask &= proposal.eq(JWST_PID)
        if collection_col is not None:
            mask &= frame[collection_col].astype(str).str.upper().eq("JWST")
        if instrument_col is not None:
            mask &= frame[instrument_col].astype(str).str.contains("NIRSPEC", case=False, na=False)
        selected = frame[mask].copy()
        if not selected.empty:
            return selected
    raise RuntimeError("No GO-5224 NIRSpec observation footprint intersects the MoM-z14 coordinates.")


def query_products(obsids: list[str]) -> pd.DataFrame:
    frames = []
    for start in range(0, len(obsids), 8):
        table = Observations.get_product_list(obsids[start:start + 8])
        frames.append(table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table))
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()


def select_x1d_products(products: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    filename_col = col(products, ["productFilename", "productfilename"])
    subgroup_col = col(products, ["productSubGroupDescription"])
    if filename_col is None:
        raise RuntimeError("MAST products have no product filename column.")
    names = products[filename_col].astype(str).str.lower()
    mask = names.str.contains("x1d", na=False) & names.str.endswith(".fits", na=False)
    if subgroup_col is not None:
        mask |= products[subgroup_col].astype(str).str.upper().eq("X1D")
    chosen = products[mask].drop_duplicates(subset=[filename_col]).copy()
    if chosen.empty:
        raise RuntimeError("No public X1D FITS products were returned for the matched observation.")
    size_col = col(chosen, ["size"])
    if size_col is not None:
        chosen[size_col] = pd.to_numeric(chosen[size_col], errors="coerce")
        chosen = chosen.sort_values(size_col, ascending=False)
    return chosen.head(MAX_X1D), filename_col


def download_product(record: dict, filename_col: str) -> Path:
    uri = record.get("dataURI") or record.get("dataUri") or record.get("uri")
    name = record.get(filename_col)
    if not uri or not name:
        raise RuntimeError("MAST product record lacks URI or filename.")
    destination = DATA / str(name)
    if destination.exists() and destination.stat().st_size > 50000:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(
                "https://mast.stsci.edu/api/v0.1/Download/file",
                params={"uri": str(uri)},
                stream=True,
                timeout=(20, 180),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size < 50000:
                raise RuntimeError("Downloaded FITS file is unexpectedly small.")
            partial.replace(destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {name}")


def table_field(data, names: list[str]):
    if data is None or not getattr(data, "names", None):
        return None, None
    lookup = {str(name).upper(): name for name in data.names}
    for name in names:
        if name.upper() in lookup:
            actual = lookup[name.upper()]
            return actual, data[actual]
    return None, None


def wavelength_to_um(values, unit) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    text = str(unit or "").lower()
    if "angstrom" in text or text.strip() in {"a", "aa"}:
        return array * 1.0e-4
    if "nm" in text:
        return array * 1.0e-3
    if "um" in text or "micron" in text:
        return array
    median = float(np.nanmedian(array))
    if median > 1000:
        return array * 1.0e-4
    if median > 10:
        return array * 1.0e-3
    return array


def source_coordinates(headers) -> tuple[float | None, float | None, str]:
    pairs = [
        ("SRCRA", "SRCDEC"),
        ("RA_OBJ", "DEC_OBJ"),
        ("OBJ_RA", "OBJ_DEC"),
        ("SLIT_RA", "SLIT_DEC"),
        ("MSA_RA", "MSA_DEC"),
        ("SHUT_RA", "SHUT_DEC"),
        ("TARG_RA", "TARG_DEC"),
        ("RA_TARG", "DEC_TARG"),
    ]
    for header in headers:
        for ra_key, dec_key in pairs:
            ra = fnum(header.get(ra_key))
            dec = fnum(header.get(dec_key))
            if ra is not None and dec is not None:
                return ra, dec, f"{ra_key}/{dec_key}"
    return None, None, "NONE"


def read_x1d_extensions(path: Path) -> list[dict]:
    target = SkyCoord(MOM_RA * u.deg, MOM_DEC * u.deg)
    extensions = []
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header
        for hdu_index, hdu in enumerate(hdul[1:], 1):
            wavelength_name, wavelength_values = table_field(hdu.data, ["WAVELENGTH", "WAVE"])
            flux_name, flux_values = table_field(hdu.data, ["FLUX"])
            if wavelength_values is None or flux_values is None:
                continue
            try:
                wavelength_unit = hdu.columns[wavelength_name].unit
            except Exception:
                wavelength_unit = None
            try:
                flux_unit = str(hdu.columns[flux_name].unit or "unknown")
            except Exception:
                flux_unit = "unknown"
            wavelength_um = wavelength_to_um(wavelength_values, wavelength_unit)
            flux = np.asarray(flux_values, dtype=float).ravel()
            error_name, error_values = table_field(hdu.data, ["FLUX_ERROR", "ERROR", "ERR"])
            error = np.asarray(error_values, dtype=float).ravel() if error_values is not None else np.full_like(flux, np.nan)
            _, quality_values = table_field(hdu.data, ["DQ", "QUALITY"])
            quality = np.asarray(quality_values).ravel() if quality_values is not None else np.zeros_like(flux, dtype=int)
            valid = np.isfinite(wavelength_um) & np.isfinite(flux) & (wavelength_um > 0)
            if quality.size == valid.size:
                valid &= quality == 0
            if valid.sum() < 20:
                continue
            ra, dec, coordinate_source = source_coordinates([hdu.header, primary])
            separation = math.inf
            if ra is not None and dec is not None:
                separation = float(SkyCoord(ra * u.deg, dec * u.deg).separation(target).arcsec)
            order = np.argsort(wavelength_um[valid])
            extensions.append({
                "path": path,
                "hdu": hdu_index,
                "source_id": hdu.header.get("SOURCEID", ""),
                "ra": ra,
                "dec": dec,
                "coord_source": coordinate_source,
                "separation_arcsec": separation,
                "wavelength_um": wavelength_um[valid][order],
                "flux": flux[valid][order],
                "error": error[valid][order],
                "flux_unit": flux_unit,
            })
    return extensions


def find_exact_spectrum() -> tuple[dict, Path]:
    observations = query_coordinate_matched_observations()
    obsid_col = col(observations, ["obsid", "obs_id"])
    if obsid_col is None:
        raise RuntimeError("Matched MAST observations have no observation ID column.")
    obsids = observations[obsid_col].dropna().astype(str).drop_duplicates().tolist()
    products = query_products(obsids)
    x1d, filename_col = select_x1d_products(products)

    candidates = []
    audit_rows = []
    for index, record in enumerate(x1d.to_dict("records"), 1):
        product_name = str(record.get(filename_col, "UNKNOWN"))
        try:
            path = download_product(record, filename_col)
            extracted = read_x1d_extensions(path)
            candidates.extend(extracted)
            nearest = min((item["separation_arcsec"] for item in extracted), default=math.inf)
            audit_rows.append({
                "product": path.name,
                "status": "OK",
                "extensions": len(extracted),
                "nearest_arcsec": nearest,
            })
            print(f"X1D {index:02d}/{len(x1d):02d}  {path.name}  nearest={nearest:.4f} arcsec")
        except Exception as exc:
            audit_rows.append({
                "product": product_name,
                "status": type(exc).__name__,
                "extensions": 0,
                "nearest_arcsec": np.nan,
            })

    audit_path = CSV / f"{VERSION}_MAST_X1D_AUDIT.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    required_low_um = min(item["window_A"][0] for item in COMPLEXES) * (1.0 + MOM_Z) * 1.0e-4
    required_high_um = max(item["window_A"][1] for item in COMPLEXES) * (1.0 + MOM_Z) * 1.0e-4
    valid = [
        item for item in candidates
        if math.isfinite(item["separation_arcsec"])
        and item["separation_arcsec"] <= 1.5
        and float(np.nanmin(item["wavelength_um"])) <= required_low_um
        and float(np.nanmax(item["wavelength_um"])) >= required_high_um
    ]
    if not valid:
        nearest = min((item["separation_arcsec"] for item in candidates), default=math.inf)
        raise RuntimeError(
            f"No coordinate-verified X1D extraction covers all five UV complexes. "
            f"Nearest inspected source={nearest:.4f} arcsec. Audit={audit_path}"
        )
    valid.sort(key=lambda item: (item["separation_arcsec"], -len(item["wavelength_um"])))
    return valid[0], audit_path


def convert_flux(flux: np.ndarray, error: np.ndarray, unit: str):
    text = str(unit).strip().lower()
    if text == "jy" or text.endswith(" jy") or "jansky" in text:
        return flux * 1.0e9, error * 1.0e9, "Flux density [nJy]"
    finite = np.abs(flux[np.isfinite(flux)])
    if finite.size and np.nanmedian(finite) < 1.0e-3:
        return flux * 1.0e9, error * 1.0e9, "Flux density [nJy; FITS unit inferred as Jy]"
    return flux, error, f"Flux [{unit}]"


def robust_limits(values: np.ndarray, errors: np.ndarray | None = None) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = np.nanpercentile(finite, [1.5, 98.5])
    if errors is not None:
        ef = errors[np.isfinite(errors) & (errors >= 0)]
        if ef.size:
            pad = min(float(np.nanmedian(ef)) * 2.0, max(float(high - low), 1.0))
            low -= pad
            high += pad
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        center = float(np.nanmedian(finite))
        spread = float(np.nanstd(finite)) or 1.0
        return center - 3.0 * spread, center + 3.0 * spread
    margin = 0.12 * (high - low)
    return float(low - margin), float(high + margin)


def style_axis(axis) -> None:
    axis.grid(True, color="#334155", linewidth=0.55, alpha=0.55)
    for spine in axis.spines.values():
        spine.set_color("#94a3b8")


def plot_raw_panel(axis, rest_A, flux, error, complex_def, ylabel: str) -> None:
    low, high = complex_def["window_A"]
    mask = (rest_A >= low) & (rest_A <= high)
    x = rest_A[mask]
    y = flux[mask]
    e = error[mask]
    axis.plot(x, y, color="#dbeafe", linewidth=1.0, drawstyle="steps-mid", label="Actual NIRSpec X1D samples")
    good_error = np.isfinite(e) & (e >= 0)
    if good_error.any():
        axis.fill_between(x[good_error], y[good_error] - e[good_error], y[good_error] + e[good_error],
                          color="#38bdf8", alpha=0.18, step="mid", label="1-sigma uncertainty")
    for line_index, (label, wavelength) in enumerate(complex_def["lines"]):
        axis.axvline(wavelength, color="#fb923c", linewidth=1.0, linestyle="--", alpha=0.92)
        axis.text(wavelength, 0.97 - 0.10 * (line_index % 3), label, rotation=90,
                  transform=axis.get_xaxis_transform(), ha="right", va="top", fontsize=8, color="#fdba74")
    axis.set_xlim(low, high)
    axis.set_ylim(*robust_limits(y, e))
    axis.set_title(complex_def["title"] + " — actual unsmoothed MoM-z14 spectrum", fontsize=12, pad=8)
    axis.set_xlabel("Rest-frame vacuum wavelength [angstrom]")
    axis.set_ylabel(ylabel)
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=8, facecolor="#020617", edgecolor="#475569")


def make_full_spectrum(wavelength_um, rest_A, flux, error, ylabel: str) -> Path:
    figure, axis = plt.subplots(figsize=(17, 8.5), constrained_layout=True)
    axis.plot(wavelength_um, flux, color="#dbeafe", linewidth=0.85, drawstyle="steps-mid", label="Actual NIRSpec X1D flux samples")
    good_error = np.isfinite(error) & (error >= 0)
    if good_error.any():
        axis.fill_between(wavelength_um[good_error], flux[good_error] - error[good_error], flux[good_error] + error[good_error],
                          color="#38bdf8", alpha=0.12, step="mid", label="1-sigma uncertainty")
    for complex_def in COMPLEXES:
        for label, rest_wave in complex_def["lines"]:
            observed_um = rest_wave * (1.0 + MOM_Z) * 1.0e-4
            axis.axvline(observed_um, color="#fb923c", linewidth=0.8, linestyle="--", alpha=0.65)
    axis.set_xlabel("Observed wavelength [micrometers]")
    axis.set_ylabel(ylabel)
    axis.set_title("MoM-z14 — actual public JWST/NIRSpec prism X1D spectrum\nRaw jagged samples; no smoothing and no synthetic line profiles", fontsize=17, pad=14)
    axis.set_ylim(*robust_limits(flux, error))
    style_axis(axis)
    axis.legend(loc="upper right", fontsize=9, facecolor="#020617", edgecolor="#475569")

    def obs_to_rest(values):
        return np.asarray(values) * 1.0e4 / (1.0 + MOM_Z)

    def rest_to_obs(values):
        return np.asarray(values) * (1.0 + MOM_Z) * 1.0e-4

    top = axis.secondary_xaxis("top", functions=(obs_to_rest, rest_to_obs))
    top.set_xlabel("Rest-frame wavelength [angstrom] at z = 14.44")
    output = PNG / f"{VERSION}_MOMZ14_ACTUAL_FULL_NIRSPEC_RAW_SPECTRUM.png"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)
    return output


def make_two_panel(rest_A, flux, error, keys: tuple[str, str], title: str, filename: str, ylabel: str) -> Path:
    lookup = {item["key"]: item for item in COMPLEXES}
    figure, axes = plt.subplots(2, 1, figsize=(16, 11), constrained_layout=True)
    for axis, key in zip(axes, keys):
        plot_raw_panel(axis, rest_A, flux, error, lookup[key], ylabel)
    figure.suptitle(title + "\nActual coordinate-matched JWST/NIRSpec data", fontsize=17)
    output = PNG / filename
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)
    return output


def make_blend(rest_A, flux, error, ylabel: str) -> Path:
    definition = next(item for item in COMPLEXES if item["key"] == "HE_O")
    figure, axis = plt.subplots(figsize=(16, 8), constrained_layout=True)
    plot_raw_panel(axis, rest_A, flux, error, definition, ylabel)
    figure.suptitle("MoM-z14 He II + O III] blend — actual jagged NIRSpec response", fontsize=17)
    output = PNG / f"{VERSION}_MOMZ14_ACTUAL_HEII_OIII_RAW_BLEND.png"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.show()
    plt.close(figure)
    return output


def main() -> None:
    reset_matplotlib()
    spectrum, audit_path = find_exact_spectrum()
    wavelength_um = spectrum["wavelength_um"]
    raw_flux = spectrum["flux"]
    raw_error = spectrum["error"]
    flux, error, ylabel = convert_flux(raw_flux, raw_error, spectrum["flux_unit"])
    rest_A = wavelength_um * 1.0e4 / (1.0 + MOM_Z)

    spectrum_csv = CSV / f"{VERSION}_MOMZ14_ACTUAL_NIRSPEC_X1D_RAW.csv"
    pd.DataFrame({
        "observed_wavelength_um": wavelength_um,
        "rest_wavelength_angstrom_z14p44": rest_A,
        "flux_display_units": flux,
        "flux_error_display_units": error,
        "original_flux": raw_flux,
        "original_flux_error": raw_error,
        "original_flux_unit": spectrum["flux_unit"],
    }).to_csv(spectrum_csv, index=False)

    metadata_csv = CSV / f"{VERSION}_MOMZ14_ACTUAL_NIRSPEC_PROVENANCE.csv"
    pd.DataFrame([{
        "galaxy": GALAXY,
        "ra_deg": MOM_RA,
        "dec_deg": MOM_DEC,
        "redshift_used_for_rest_axis": MOM_Z,
        "mast_program": JWST_PID,
        "product_file": spectrum["path"].name,
        "fits_path": str(spectrum["path"]),
        "fits_hdu": spectrum["hdu"],
        "source_id": spectrum["source_id"],
        "coordinate_source": spectrum["coord_source"],
        "coordinate_separation_arcsec": spectrum["separation_arcsec"],
        "original_flux_unit": spectrum["flux_unit"],
        "sample_count": len(wavelength_um),
        "smoothing_applied": False,
        "synthetic_profiles_applied": False,
    }]).to_csv(metadata_csv, index=False)

    full_png = make_full_spectrum(wavelength_um, rest_A, flux, error, ylabel)
    nitrogen_png = make_two_panel(
        rest_A, flux, error,
        ("N_IV", "N_III"),
        "MoM-z14 nitrogen complexes — raw observed spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_NITROGEN_RAW_SPECTRA.png",
        ylabel,
    )
    carbon_png = make_two_panel(
        rest_A, flux, error,
        ("C_IV", "C_III"),
        "MoM-z14 carbon complexes — raw observed spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_CARBON_RAW_SPECTRA.png",
        ylabel,
    )
    blend_png = make_blend(rest_A, flux, error, ylabel)

    print(f"CODE OUTPUT: {VERSION}")
    print("DATA            actual public JWST/NIRSpec X1D samples")
    print("MAST PROGRAM    GO-5224")
    print(f"PRODUCT         {spectrum['path'].name}")
    print(f"SOURCE OFFSET   {spectrum['separation_arcsec']:.6f} arcsec")
    print("SMOOTHING       none")
    print("SYNTHETIC DATA  none")
    print(f"FULL PNG        {full_png}")
    print(f"NITROGEN PNG    {nitrogen_png}")
    print(f"CARBON PNG      {carbon_png}")
    print(f"BLEND PNG       {blend_png}")
    print(f"RAW CSV         {spectrum_csv}")
    print(f"PROVENANCE CSV  {metadata_csv}")
    print(f"AUDIT CSV       {audit_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
