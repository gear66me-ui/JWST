#!/usr/bin/env python3
"""
JWST_0089_MOMZ14_CACHED_NIRSPEC_COLORED_COMPLETE_PANELS.py

Regenerate the colored MoM-z14 NIRSpec plots entirely from real X1D data
already cached in the active Colab runtime. No MAST query, no redownload,
no smoothing, no interpolation, no synthetic profiles, and no AI imagery.
"""
from __future__ import annotations

import importlib.util
import math
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

VERSION = "JWST_0089"
HELPER_NAME = "JWST_0088_MOMZ14_ACTUAL_NIRSPEC_COLORED_COMPLETE_PANELS.py"
HELPER_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{HELPER_NAME}"
HELPER_PATH = Path("/content") / HELPER_NAME
OUT = Path("/content/JWST_OUTPUT")
PNG = OUT / "PNG"
CSV = OUT / "CSV"
for directory in (PNG, CSV):
    directory.mkdir(parents=True, exist_ok=True)


def load_helper():
    if not HELPER_PATH.exists() or HELPER_PATH.stat().st_size < 12000:
        urllib.request.urlretrieve(HELPER_URL, HELPER_PATH)
    spec = importlib.util.spec_from_file_location("jwst_0088_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the JWST_0088 plotting helper.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    module.PNG = PNG
    module.CSV = CSV
    return module


plotter = load_helper()
np = plotter.np
pd = plotter.pd


def csv_candidates() -> list[Path]:
    preferred = [
        CSV / "JWST_0087_MOMZ14_ACTUAL_NIRSPEC_X1D_RAW.csv",
        CSV / "JWST_0060_MoM-z14_EXACT_JWST.csv",
        CSV / "JWST_0059_MoM-z14_EXACT_JWST.csv",
    ]
    discovered = sorted(
        [
            path for path in CSV.glob("*.csv")
            if any(token in path.name.upper() for token in ("MOMZ14", "MOM-Z14"))
            and "PANEL_SAMPLES" not in path.name.upper()
            and "REFERENCE_LINE" not in path.name.upper()
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    ordered = []
    for path in preferred + discovered:
        if path.exists() and path not in ordered:
            ordered.append(path)
    return ordered


def first_column(frame, names):
    lookup = {str(column).lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def read_cached_csv(path: Path):
    frame = pd.read_csv(path)
    wave_col = first_column(frame, ["observed_wavelength_um", "wavelength_um"])
    if wave_col is None:
        return None

    display_flux_col = first_column(frame, ["flux_display_units"])
    display_error_col = first_column(frame, ["flux_error_display_units"])
    raw_flux_col = first_column(frame, ["flux", "original_flux"])
    raw_error_col = first_column(frame, ["flux_error", "original_flux_error", "error", "err"])

    wave = pd.to_numeric(frame[wave_col], errors="coerce").to_numpy(float)
    if display_flux_col is not None:
        flux = pd.to_numeric(frame[display_flux_col], errors="coerce").to_numpy(float)
        error = (
            pd.to_numeric(frame[display_error_col], errors="coerce").to_numpy(float)
            if display_error_col is not None else np.full_like(flux, np.nan)
        )
        unit_col = first_column(frame, ["original_flux_unit"])
        original_unit = str(frame[unit_col].dropna().iloc[0]) if unit_col and frame[unit_col].notna().any() else "Jy"
        ylabel = "Flux density [nJy]" if "jy" in original_unit.lower() else f"Flux [{original_unit}]"
    elif raw_flux_col is not None:
        raw_flux = pd.to_numeric(frame[raw_flux_col], errors="coerce").to_numpy(float)
        raw_error = (
            pd.to_numeric(frame[raw_error_col], errors="coerce").to_numpy(float)
            if raw_error_col is not None else np.full_like(raw_flux, np.nan)
        )
        flux, error, ylabel = plotter.base.convert_flux(raw_flux, raw_error, "Jy")
    else:
        return None

    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    if valid.sum() < 40:
        return None
    order = np.argsort(wave[valid])
    wave = wave[valid][order]
    flux = flux[valid][order]
    error = error[valid][order]

    required_low = min(item["window_A"][0] for item in plotter.COMPLEXES) * (1.0 + plotter.MOM_Z) * 1.0e-4
    required_high = max(item["window_A"][1] for item in plotter.COMPLEXES) * (1.0 + plotter.MOM_Z) * 1.0e-4
    if wave.min() > required_low or wave.max() < required_high:
        return None
    return wave, flux, error, ylabel, str(path)


def read_cached_fits():
    roots = [OUT / "DATA" / "JWST_0087", OUT / "DATA" / "JWST_0088", OUT / "DATA"]
    fits_files = []
    for root in roots:
        if root.exists():
            fits_files.extend(root.rglob("*x1d*.fits"))
    candidates = []
    for path in sorted(set(fits_files)):
        try:
            candidates.extend(plotter.base.read_x1d_extensions(path))
        except Exception:
            continue
    if not candidates:
        return None

    required_low = min(item["window_A"][0] for item in plotter.COMPLEXES) * (1.0 + plotter.MOM_Z) * 1.0e-4
    required_high = max(item["window_A"][1] for item in plotter.COMPLEXES) * (1.0 + plotter.MOM_Z) * 1.0e-4
    usable = [
        item for item in candidates
        if item["wavelength_um"].min() <= required_low
        and item["wavelength_um"].max() >= required_high
    ]
    if not usable:
        return None
    usable.sort(key=lambda item: (item.get("separation_arcsec", math.inf), -len(item["wavelength_um"])))
    spectrum = usable[0]
    flux, error, ylabel = plotter.base.convert_flux(spectrum["flux"], spectrum["error"], spectrum["flux_unit"])
    return spectrum["wavelength_um"], flux, error, ylabel, str(spectrum["path"])


def load_cached_spectrum():
    for path in csv_candidates():
        result = read_cached_csv(path)
        if result is not None:
            return result
    result = read_cached_fits()
    if result is not None:
        return result
    raise FileNotFoundError(
        "No cached MoM-z14 X1D CSV or FITS data were found in /content/JWST_OUTPUT. "
        "Run JWST_0087 once in this runtime, then run this script again."
    )


def main():
    plotter.reset_style()
    wave_um, flux, error, ylabel, source = load_cached_spectrum()

    full_png = plotter.make_full_spectrum(wave_um, flux, error, ylabel)
    nitrogen_png, nitrogen_sel = plotter.make_group(
        wave_um, flux, error, ("N_IV", "N_III"),
        "MoM-z14 nitrogen complexes — complete raw spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_NITROGEN_COMPLETE_RAW.png", ylabel,
    )
    carbon_png, carbon_sel = plotter.make_group(
        wave_um, flux, error, ("C_IV", "C_III"),
        "MoM-z14 carbon complexes — complete raw spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_CARBON_COMPLETE_RAW.png", ylabel,
    )
    blend_png, blend_sel = plotter.make_group(
        wave_um, flux, error, ("HE_O",),
        "MoM-z14 He II + O III] blend — complete raw spectral response",
        f"{VERSION}_MOMZ14_ACTUAL_HEII_OIII_COMPLETE_RAW.png", ylabel,
    )

    line_key = CSV / f"{VERSION}_MOMZ14_REFERENCE_LINE_COLOR_KEY.csv"
    pd.DataFrame(plotter.all_reference_rows()).to_csv(line_key, index=False)
    panel_csv = plotter.save_panel_samples(
        wave_um, flux, error, nitrogen_sel + carbon_sel + blend_sel
    )
    raw_csv = CSV / f"{VERSION}_MOMZ14_CACHED_X1D_USED.csv"
    pd.DataFrame({
        "observed_wavelength_um": wave_um,
        "rest_wavelength_angstrom_z14p44": plotter.rest_A(wave_um),
        "flux_display_units": flux,
        "flux_error_display_units": error,
    }).to_csv(raw_csv, index=False)

    print(f"CODE OUTPUT: {VERSION}")
    print("DATA SOURCE     cached real JWST/NIRSpec X1D data")
    print(f"CACHE FILE      {source}")
    print("NETWORK         no MAST query and no FITS redownload")
    print("SMOOTHING       none")
    print("INTERPOLATION   none")
    print("SYNTHETIC DATA  none")
    for key, indices in nitrogen_sel + carbon_sel + blend_sel:
        rr = plotter.rest_A(wave_um[indices])
        print(f"{key:<8}        n={len(indices):>3}  rest={rr.min():.2f}-{rr.max():.2f} A")
    print(f"FULL PNG        {full_png}")
    print(f"NITROGEN PNG    {nitrogen_png}")
    print(f"CARBON PNG      {carbon_png}")
    print(f"BLEND PNG       {blend_png}")
    print(f"PANEL CSV       {panel_csv}")
    print(f"LINE KEY CSV    {line_key}")
    print(f"RAW CSV         {raw_csv}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
