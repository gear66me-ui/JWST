# JWST_0072
# Fix the hydrogen observed marker in JWST_0071/0070.
# The right-side red marker is placed on the highest raw JWST sample
# inside the Ly-alpha feature window, not at a jump midpoint.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0072"
BASE_NAME = "JWST_0070_FIVE_MEASURED_SPECTROGRAPH_TRACES.py"
BASE_URL = f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
BASE_PATH = ROOT / BASE_NAME


def ensure_base():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 15000:
        return BASE_PATH
    subprocess.run(
        [
            "curl", "-fsSL", "--connect-timeout", "15", "--max-time", "120",
            "-o", str(BASE_PATH), BASE_URL,
        ],
        check=True,
        timeout=130,
    )
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size < 15000:
        raise RuntimeError("Could not download JWST_0070 base script")
    return BASE_PATH


def load_base(path):
    spec = importlib.util.spec_from_file_location("jwst_0070_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load JWST_0070 base script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corrected_measure_observed(feature, x, y, expected):
    if feature["key"] == "H_I_LYA":
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 1:
            raise RuntimeError("No finite JWST samples in the hydrogen window")
        x_valid = x[finite]
        y_valid = y[finite]
        index = int(np.nanargmax(y_valid))
        spacing = float(np.nanmedian(np.diff(x_valid))) if len(x_valid) > 1 else np.nan
        return {
            "marker_nm": float(x_valid[index]),
            "marker_flux": float(y_valid[index]),
            "kind": "highest raw JWST sample in Ly-alpha window",
            "uncertainty_nm": 0.5 * spacing if np.isfinite(spacing) else np.nan,
            "left_sample_nm": np.nan,
            "right_sample_nm": np.nan,
        }
    return ORIGINAL_MEASURE(feature, x, y, expected)


def main():
    global ORIGINAL_MEASURE

    base = load_base(ensure_base())
    base.np = np
    base.pd = pd
    base.plt = plt
    base.VERSION = VERSION
    base.DATA = base.OUT / "DATA" / VERSION

    ORIGINAL_MEASURE = base.measure_observed
    base.measure_observed = corrected_measure_observed

    for feature in base.FEATURES:
        if feature["key"] == "H_I_LYA":
            feature["observed_mode"] = "peak"
            feature["observed_rest_half_width_nm"] = 6.5

    print(f"CODE OUTPUT: {VERSION}")
    print("PATCH      : hydrogen marker moved to highest raw JWST sample")
    print("BASE       : JWST_0070_FIVE_MEASURED_SPECTROGRAPH_TRACES.py")
    print()

    base.main()

    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
