# JWST_0068
# Self-contained hydrogen Ly-alpha comparison.
# Uses cached JWST data when available; otherwise performs a coordinate-filtered
# MAST retrieval for MoM-z14 before rendering the published laboratory VUV scan.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib
import importlib.util
import subprocess
import sys

VERSION = "JWST_0068"
REPO_RAW = "https://raw.githubusercontent.com/gear66me-ui/JWST/main"
LAB_MODULE_NAME = "JWST_0067_HYDROGEN_LYA_LAB_SCAN_VS_JWST.py"
MAST_MODULE_NAME = "JWST_0060_MOMZ14_FAST_CONE_CLASSY.py"

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
OUT = ROOT / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION
LAB_PATH = ROOT / LAB_MODULE_NAME
MAST_PATH = ROOT / MAST_MODULE_NAME


def ensure_package(import_name, pip_name=None):
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )


def ensure_repo_file(path, filename, minimum_size):
    if path.exists() and path.stat().st_size >= minimum_size:
        return path
    url = f"{REPO_RAW}/{filename}"
    subprocess.run(
        [
            "curl", "-fsSL", "--connect-timeout", "15", "--max-time", "120",
            "-o", str(path), url,
        ],
        check=True,
        timeout=130,
    )
    if not path.exists() or path.stat().st_size < minimum_size:
        raise RuntimeError(f"Could not download helper module: {filename}")
    return path


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_lab_module(module):
    module.VERSION = VERSION
    module.OUT = OUT
    module.PNG = PNG
    module.CSV = CSV
    module.DATA = DATA / "LAB"


def configure_mast_module(module):
    module.VERSION = VERSION
    module.OUT = OUT
    module.PNG = PNG
    module.CSV = CSV
    module.DATA = DATA / "MAST"
    module.MAX_JWST_X1D = 24


def acquire_jwst_csv(lab_module, mast_module):
    try:
        path, status = lab_module.locate_jwst_csv()
        print(f"  cached spectrum found | {path.name}")
        return Path(path), status, None
    except FileNotFoundError:
        print("  no cache found | starting coordinate-filtered MAST retrieval")

    base_path = mast_module.ensure_base()
    base = mast_module.load_base(base_path)
    best, metadata = mast_module.exact_momz14_cone(base)
    exact_path = Path(metadata["exact_csv"])
    if not exact_path.exists() or exact_path.stat().st_size < 100:
        raise RuntimeError("MAST retrieval completed without producing a usable spectrum CSV")

    status = (
        f"coordinate-verified GO-5224 X1D; separation="
        f"{metadata['sep']:.6f} arcsec; keys={metadata['coord_source']}"
    )
    return exact_path, status, metadata


def main():
    for import_name, pip_name in [
        ("numpy", None),
        ("pandas", None),
        ("matplotlib", None),
        ("requests", None),
        ("PIL", "Pillow"),
        ("astropy", None),
        ("astroquery", None),
    ]:
        ensure_package(import_name, pip_name)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    ensure_repo_file(LAB_PATH, LAB_MODULE_NAME, 15000)
    ensure_repo_file(MAST_PATH, MAST_MODULE_NAME, 12000)
    lab = load_module("jwst_0067_lab", LAB_PATH)
    mast = load_module("jwst_0060_mast", MAST_PATH)
    configure_lab_module(lab)
    configure_mast_module(mast)

    print(f"CODE OUTPUT: {VERSION}")
    print("STEP 1/4 | Locate or retrieve the MoM-z14 JWST spectrum")
    source_path, source_status, mast_metadata = acquire_jwst_csv(lab, mast)
    observed_nm, flux, wave_column, flux_column = lab.load_jwst(source_path)

    print("STEP 2/4 | Download the published laboratory hydrogen VUV spectrum")
    lab_image, lab_raw_path, lab_figure_url = lab.download_lab_figure()

    print("STEP 3/4 | Crop the published Ly-alpha laboratory scan")
    lab_zoom, lab_crop_path, plot_box = lab.crop_lab_scan(lab_image)

    print("STEP 4/4 | Measure the JWST Ly-alpha break and render the comparison")
    plot_path, jwst_csv, audit_csv, break_result = lab.make_plot(
        lab_zoom,
        lab_crop_path,
        lab_figure_url,
        source_path,
        source_status,
        observed_nm,
        flux,
    )

    print()
    print(f"JWST SOURCE          : {source_path}")
    print(f"JWST SOURCE STATUS   : {source_status}")
    if mast_metadata is not None:
        print(f"JWST PRODUCT         : {Path(mast_metadata['product']).name}")
        print(f"JWST SEPARATION      : {mast_metadata['sep']:.6f} arcsec")
        print(f"JWST COORDINATE KEYS : {mast_metadata['coord_source']}")
        print(f"JWST X1D AUDIT       : {mast_metadata['audit']}")
    print(f"WAVELENGTH COLUMN    : {wave_column}")
    print(f"FLUX COLUMN          : {flux_column}")
    print(f"LAB RAW FIGURE       : {lab_raw_path}")
    print(f"LAB CROP PNG         : {lab_crop_path}")
    print(f"LAB PLOT BOX PX      : {plot_box}")
    print(f"REST Ly-alpha        : {lab.LYA_REST_NM:.6f} nm")
    print(f"EXPECTED BREAK       : {lab.EXPECTED_OBS_NM:.6f} nm")
    print(f"MEASURED BREAK       : {break_result['center_nm']:.6f} nm")
    print(f"BREAK REDSHIFT       : {break_result['center_nm'] / lab.LYA_REST_NM - 1.0:.6f}")
    print(f"PLOT PNG             : {plot_path}")
    print(f"JWST WINDOW CSV      : {jwst_csv}")
    print(f"AUDIT CSV            : {audit_csv}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
