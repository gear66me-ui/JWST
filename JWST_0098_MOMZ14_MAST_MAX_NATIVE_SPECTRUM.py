#!/usr/bin/env python3
"""
JWST_0098_MOMZ14_MAST_MAX_NATIVE_SPECTRUM.py

Query MAST directly for every public JWST/NIRSpec X1D and S2D product whose
GO-5224 footprint intersects the published MoM-z14 coordinates. Inspect every
X1D extraction, retain coordinate-matched source extensions, identify the
highest resolving-power disperser actually present in MAST, and plot the native
measured samples without smoothing, interpolation, rebinning, line models, or
reference-line overlays.
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION = "JWST_0098"
GALAXY = "MoM-z14"
RA_DEG = 150.0933255
DEC_DEG = 2.2731627
PROGRAM_ID = "5224"
MATCH_RADIUS_ARCSEC = 1.5

ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
DATA_DIR = ROOT / "DATA" / VERSION
for directory in (PNG_DIR, CSV_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RESOLUTION_R = {
    "G140H": 2700, "G235H": 2700, "G395H": 2700,
    "G140M": 1000, "G235M": 1000, "G395M": 1000,
    "PRISM": 100,
}


def reset_style() -> None:
    plt.close("all")
    matplotlib.rcdefaults()
    matplotlib.rcParams.update({
        "text.usetex": False,
        "figure.facecolor": "#050812",
        "axes.facecolor": "#081321",
        "axes.edgecolor": "#94a3b8",
        "axes.labelcolor": "#eef6ff",
        "xtick.color": "#dce8f5",
        "ytick.color": "#dce8f5",
        "text.color": "#f8fbff",
        "font.size": 10,
        "savefig.facecolor": "#050812",
    })


def http_session() -> requests.Session:
    retry = Retry(
        total=8, connect=8, read=8, backoff_factor=1.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": f"{VERSION} public-MAST-client"})
    return session


def retry_call(label, function, attempts=7):
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except Exception as exc:
            last = exc
            print(f"RETRY {label:<28} {attempt}/{attempts}  {type(exc).__name__}: {exc}")
            if attempt < attempts:
                time.sleep(min(24, 2 * attempt))
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last}")


def find_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    lookup = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def query_observations() -> pd.DataFrame:
    target = SkyCoord(RA_DEG * u.deg, DEC_DEG * u.deg)
    for radius in (3.0, 10.0, 30.0):
        table = retry_call(
            f"MAST region query {radius:.0f} arcsec",
            lambda r=radius: Observations.query_region(target, radius=r * u.arcsec),
        )
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
        if frame.empty:
            continue
        proposal = find_column(frame, ["proposal_id", "proposalid"])
        collection = find_column(frame, ["obs_collection"])
        instrument = find_column(frame, ["instrument_name", "instrument"])
        mask = pd.Series(True, index=frame.index)
        if proposal:
            mask &= frame[proposal].astype(str).str.replace(".0", "", regex=False).eq(PROGRAM_ID)
        if collection:
            mask &= frame[collection].astype(str).str.upper().eq("JWST")
        if instrument:
            mask &= frame[instrument].astype(str).str.contains("NIRSPEC", case=False, na=False)
        selected = frame[mask].drop_duplicates().copy()
        if not selected.empty:
            return selected
    raise RuntimeError("No public GO-5224 NIRSpec observation footprint intersects MoM-z14.")


def query_products(observations: pd.DataFrame) -> pd.DataFrame:
    obsid_col = find_column(observations, ["obsid", "obs_id"])
    if obsid_col is None:
        raise RuntimeError("MAST observation table does not contain an observation ID column.")
    obsids = observations[obsid_col].dropna().astype(str).drop_duplicates().tolist()
    frames = []
    for start in range(0, len(obsids), 6):
        ids = obsids[start:start + 6]
        table = retry_call(
            f"MAST product list {start + 1}-{start + len(ids)}",
            lambda x=ids: Observations.get_product_list(x),
        )
        frames.append(table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table))
    products = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()
    products.to_csv(CSV_DIR / f"{VERSION}_MAST_ALL_PRODUCTS.csv", index=False)
    return products


def select_spectral_products(products: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    filename_col = find_column(products, ["productFilename", "productfilename"])
    subgroup_col = find_column(products, ["productSubGroupDescription"])
    rights_col = find_column(products, ["dataRights", "data_rights"])
    if filename_col is None:
        raise RuntimeError("MAST product table has no product filename column.")
    names = products[filename_col].astype(str).str.lower()
    subgroup = products[subgroup_col].astype(str).str.upper() if subgroup_col else pd.Series("", index=products.index)
    mask = (
        names.str.endswith("_x1d.fits", na=False)
        | names.str.endswith("_s2d.fits", na=False)
        | subgroup.isin(["X1D", "S2D"])
    )
    if rights_col:
        rights = products[rights_col].astype(str).str.upper()
        mask &= rights.isin(["PUBLIC", "", "NAN"])
    chosen = products[mask].drop_duplicates(subset=[filename_col]).copy()
    if chosen.empty:
        raise RuntimeError("MAST returned no public X1D or S2D FITS products.")
    return chosen, filename_col


def download_product(record: dict, filename_col: str, session: requests.Session) -> Path:
    name = str(record.get(filename_col, ""))
    uri = record.get("dataURI") or record.get("dataUri") or record.get("uri")
    if not name or not uri:
        raise RuntimeError("MAST product record is missing filename or data URI.")
    destination = DATA_DIR / name
    if destination.exists() and destination.stat().st_size > 10000:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        try:
            with session.get(
                "https://mast.stsci.edu/api/v0.1/Download/file",
                params={"uri": str(uri)}, stream=True, timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size < 10000:
                raise RuntimeError("Downloaded FITS file is unexpectedly small.")
            partial.replace(destination)
            return destination
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt < 5:
                time.sleep(2 * attempt)
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


def wavelength_um(values, unit) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    text = str(unit or "").lower()
    if "angstrom" in text or text.strip() in {"a", "aa"}:
        return array * 1e-4
    if "nm" in text:
        return array * 1e-3
    if "um" in text or "micron" in text:
        return array
    median = float(np.nanmedian(array))
    if median > 1000:
        return array * 1e-4
    if median > 10:
        return array * 1e-3
    return array


def finite_number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def source_coordinates(headers) -> tuple[float | None, float | None, str]:
    pairs = [
        ("SRCRA", "SRCDEC"), ("RA_OBJ", "DEC_OBJ"),
        ("OBJ_RA", "OBJ_DEC"), ("SLIT_RA", "SLIT_DEC"),
        ("MSA_RA", "MSA_DEC"), ("TARG_RA", "TARG_DEC"),
        ("RA_TARG", "DEC_TARG"),
    ]
    for header in headers:
        for ra_key, dec_key in pairs:
            ra = finite_number(header.get(ra_key))
            dec = finite_number(header.get(dec_key))
            if ra is not None and dec is not None:
                return ra, dec, f"{ra_key}/{dec_key}"
    return None, None, "NONE"


def header_value(headers, keys: list[str], default=""):
    for header in headers:
        for key in keys:
            value = header.get(key)
            if value not in (None, ""):
                return str(value).strip()
    return default


def convert_flux(flux, error, unit):
    text = str(unit or "").strip().lower()
    if text == "jy" or "jansky" in text or text.endswith(" jy"):
        return flux * 1e9, error * 1e9, "Flux density [nJy]"
    return flux, error, f"Flux [{unit or 'native FITS units'}]"


def inspect_x1d(path: Path) -> list[dict]:
    target = SkyCoord(RA_DEG * u.deg, DEC_DEG * u.deg)
    results = []
    with fits.open(path, memmap=False) as hdul:
        primary = hdul[0].header
        for hdu_index, hdu in enumerate(hdul[1:], 1):
            wave_name, wave_values = table_field(hdu.data, ["WAVELENGTH", "WAVE"])
            flux_name, flux_values = table_field(hdu.data, ["FLUX"])
            if wave_values is None or flux_values is None:
                continue
            try:
                wave_unit = hdu.columns[wave_name].unit
            except Exception:
                wave_unit = None
            try:
                flux_unit = str(hdu.columns[flux_name].unit or "")
            except Exception:
                flux_unit = ""
            _, error_values = table_field(hdu.data, ["FLUX_ERROR", "ERROR", "ERR"])
            _, dq_values = table_field(hdu.data, ["DQ", "QUALITY"])
            wave = wavelength_um(wave_values, wave_unit)
            flux = np.asarray(flux_values, dtype=float).ravel()
            error = np.asarray(error_values, dtype=float).ravel() if error_values is not None else np.full_like(flux, np.nan)
            dq = np.asarray(dq_values).ravel() if dq_values is not None else np.zeros_like(flux, dtype=np.int64)
            finite = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
            if finite.sum() < 20:
                continue
            order = np.argsort(wave[finite])
            wave, flux, error, dq = wave[finite][order], flux[finite][order], error[finite][order], dq[finite][order]
            ra, dec, coord_source = source_coordinates([hdu.header, primary])
            separation = math.inf
            if ra is not None and dec is not None:
                separation = float(SkyCoord(ra * u.deg, dec * u.deg).separation(target).arcsec)
            grating = header_value([hdu.header, primary], ["GRATING", "PUPIL"], "UNKNOWN").upper()
            filter_name = header_value([hdu.header, primary], ["FILTER"], "UNKNOWN").upper()
            delta = np.diff(wave)
            good_delta = np.isfinite(delta) & (delta > 0)
            sampling_r = float(np.nanmedian(wave[:-1][good_delta] / delta[good_delta])) if good_delta.any() else np.nan
            results.append({
                "path": path,
                "hdu": hdu_index,
                "source_id": header_value([hdu.header], ["SOURCEID", "SOURCE_ID"], ""),
                "ra": ra, "dec": dec, "coord_source": coord_source,
                "separation_arcsec": separation,
                "grating": grating, "filter": filter_name,
                "nominal_R": RESOLUTION_R.get(grating, 0),
                "native_sampling_R_median": sampling_r,
                "wavelength_um": wave, "flux": flux, "error": error, "dq": dq,
                "flux_unit": flux_unit,
            })
    return results


def safe_name(text: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in text)


def plot_candidate(candidate: dict, rank: int) -> tuple[Path, Path]:
    wave = candidate["wavelength_um"]
    flux, error, ylabel = convert_flux(candidate["flux"], candidate["error"], candidate["flux_unit"])
    dq = candidate["dq"]
    name = safe_name(candidate["path"].stem)

    csv_path = CSV_DIR / f"{VERSION}_{rank:02d}_{name}_HDU{candidate['hdu']}_NATIVE_X1D.csv"
    pd.DataFrame({
        "observed_wavelength_um": wave,
        "flux_native_display_units": flux,
        "flux_error_native_display_units": error,
        "dq": dq,
        "source_id": candidate["source_id"],
        "grating": candidate["grating"],
        "filter": candidate["filter"],
        "separation_arcsec": candidate["separation_arcsec"],
        "smoothed": False,
        "interpolated": False,
        "rebinned": False,
    }).to_csv(csv_path, index=False)

    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharex=True, constrained_layout=True)
    for axis in axes:
        good_error = np.isfinite(error) & (error >= 0)
        if good_error.any():
            axis.fill_between(wave[good_error], flux[good_error] - error[good_error],
                              flux[good_error] + error[good_error], alpha=0.12, linewidth=0)
        axis.plot(wave, flux, linewidth=0.48, alpha=0.97, label=f"Native X1D samples: {len(wave):,}")
        bad = np.asarray(dq) != 0
        if bad.any():
            axis.scatter(wave[bad], flux[bad], s=4, alpha=0.45, label=f"DQ != 0: {bad.sum():,}")
        axis.axhline(0.0, linewidth=0.45, alpha=0.55)
        axis.grid(True, linewidth=0.42, alpha=0.38)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.85)
    axes[0].set_title("Linear scale — complete native sample range", fontsize=12.5)
    finite_abs = np.abs(flux[np.isfinite(flux)])
    linthresh = float(np.nanmedian(finite_abs)) if finite_abs.size else 1.0
    if not np.isfinite(linthresh) or linthresh <= 0:
        linthresh = 1.0
    axes[1].set_yscale("symlog", linthresh=linthresh)
    axes[1].set_title("Symmetric-log display of the same unmodified samples", fontsize=12.5)
    axes[1].set_xlabel("Observed wavelength [µm]")
    title = (
        f"{GALAXY} — MAST native JWST/NIRSpec X1D\n"
        f"{candidate['path'].name} | HDU {candidate['hdu']} | "
        f"{candidate['grating']}/{candidate['filter']} | nominal R≈{candidate['nominal_R'] or 'unknown'} | "
        f"offset={candidate['separation_arcsec']:.6f} arcsec"
    )
    fig.suptitle(title, fontsize=16.2)
    png_path = PNG_DIR / f"{VERSION}_{rank:02d}_{name}_HDU{candidate['hdu']}_RAW_NATIVE.png"
    fig.savefig(png_path, dpi=420, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return png_path, csv_path


def main() -> None:
    reset_style()
    observations = query_observations()
    observations.to_csv(CSV_DIR / f"{VERSION}_MAST_MATCHED_OBSERVATIONS.csv", index=False)
    products = query_products(observations)
    spectral_products, filename_col = select_spectral_products(products)
    session = http_session()

    downloaded = []
    download_rows = []
    for index, record in enumerate(spectral_products.to_dict("records"), 1):
        name = str(record.get(filename_col, "UNKNOWN"))
        try:
            path = download_product(record, filename_col, session)
            downloaded.append(path)
            download_rows.append({"filename": path.name, "status": "OK", "bytes": path.stat().st_size})
            print(f"DOWNLOAD {index:03d}/{len(spectral_products):03d}  {path.name}")
        except Exception as exc:
            download_rows.append({"filename": name, "status": type(exc).__name__, "bytes": 0})
            print(f"FAILED   {index:03d}/{len(spectral_products):03d}  {name}  {exc}")
    download_manifest = CSV_DIR / f"{VERSION}_MAST_DOWNLOAD_MANIFEST.csv"
    pd.DataFrame(download_rows).to_csv(download_manifest, index=False)

    candidates = []
    audit_rows = []
    x1d_paths = [path for path in downloaded if path.name.lower().endswith("_x1d.fits")]
    for path in x1d_paths:
        try:
            extracted = inspect_x1d(path)
            candidates.extend(extracted)
            nearest = min((item["separation_arcsec"] for item in extracted), default=math.inf)
            audit_rows.append({"product": path.name, "extensions": len(extracted), "nearest_arcsec": nearest, "status": "OK"})
        except Exception as exc:
            audit_rows.append({"product": path.name, "extensions": 0, "nearest_arcsec": np.nan, "status": type(exc).__name__})
    audit_path = CSV_DIR / f"{VERSION}_X1D_EXTENSION_AUDIT.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    matched = [item for item in candidates
               if math.isfinite(item["separation_arcsec"])
               and item["separation_arcsec"] <= MATCH_RADIUS_ARCSEC]
    if not matched:
        nearest = min((item["separation_arcsec"] for item in candidates), default=math.inf)
        raise RuntimeError(f"No coordinate-matched X1D extraction found. Nearest={nearest:.6f} arcsec. Audit={audit_path}")

    matched.sort(key=lambda item: (
        -item["nominal_R"],
        -item["native_sampling_R_median"] if np.isfinite(item["native_sampling_R_median"]) else 0,
        item["separation_arcsec"],
        -len(item["wavelength_um"]),
    ))
    highest_R = matched[0]["nominal_R"]
    highest = [item for item in matched if item["nominal_R"] == highest_R]

    candidate_table = []
    for rank, item in enumerate(matched, 1):
        candidate_table.append({
            "rank": rank,
            "product": item["path"].name,
            "hdu": item["hdu"],
            "source_id": item["source_id"],
            "separation_arcsec": item["separation_arcsec"],
            "grating": item["grating"],
            "filter": item["filter"],
            "nominal_R": item["nominal_R"],
            "native_sampling_R_median": item["native_sampling_R_median"],
            "samples": len(item["wavelength_um"]),
            "wavelength_min_um": float(np.nanmin(item["wavelength_um"])),
            "wavelength_max_um": float(np.nanmax(item["wavelength_um"])),
        })
    candidate_path = CSV_DIR / f"{VERSION}_MOMZ14_MATCHED_X1D_CANDIDATES.csv"
    pd.DataFrame(candidate_table).to_csv(candidate_path, index=False)

    plot_paths = []
    sample_paths = []
    for rank, item in enumerate(highest, 1):
        png_path, csv_path = plot_candidate(item, rank)
        plot_paths.append(png_path)
        sample_paths.append(csv_path)

    best = highest[0]
    print()
    print(f"CODE OUTPUT: {VERSION}")
    print(f"TARGET          {GALAXY}")
    print(f"MAST PROGRAM    JWST GO-{PROGRAM_ID}")
    print(f"COORDINATES     {RA_DEG:.7f}, {DEC_DEG:.7f} deg")
    print(f"PRODUCTS        {len(downloaded)} public X1D/S2D FITS files downloaded")
    print(f"MATCHED X1D     {len(matched)} coordinate-matched native extractions")
    print(f"HIGHEST MODE    {best['grating']}/{best['filter']}")
    print(f"NOMINAL R       {best['nominal_R'] or 'unknown'}")
    print(f"NATIVE SAMPLES  {len(best['wavelength_um']):,}")
    print(f"WAVELENGTH      {np.nanmin(best['wavelength_um']):.6f}-{np.nanmax(best['wavelength_um']):.6f} µm")
    print("PROCESSING      no smoothing; no interpolation; no rebinning; no line model")
    for path in plot_paths:
        print(f"RAW PLOT PNG    {path}")
    for path in sample_paths:
        print(f"RAW SAMPLE CSV  {path}")
    print(f"CANDIDATE CSV   {candidate_path}")
    print(f"PRODUCT CSV     {CSV_DIR / f'{VERSION}_MAST_ALL_PRODUCTS.csv'}")
    print(f"DOWNLOAD CSV    {download_manifest}")
    print(f"AUDIT CSV       {audit_path}")
    print(f"FITS DIRECTORY  {DATA_DIR}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
