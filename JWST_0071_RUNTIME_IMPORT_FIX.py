# JWST_0071
# Runtime import repair for JWST_0070.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0071"
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


def main():
    base = load_base(ensure_base())

    base.np = np
    base.pd = pd
    base.plt = plt
    base.VERSION = VERSION
    base.DATA = base.OUT / "DATA" / VERSION

    print(f"CODE OUTPUT: {VERSION}")
    print("PATCH      : restored numpy, pandas, and matplotlib globals")
    print("BASE       : JWST_0070_FIVE_MEASURED_SPECTROGRAPH_TRACES.py")
    print()

    base.main()

    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
