#!/usr/bin/env python3
"""
JWST_0085_NIST_SRD78_MOMZ14_REST_UV_LINES_NO_LATEX.py

Colab-safe launcher for the JWST_0084 NIST SRD 78 rest-UV line study.
It force-disables external LaTeX rendering, closes stale failed figures,
renames the output version to JWST_0085, and executes the original analysis.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

VERSION = "JWST_0085"
SOURCE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "JWST_0084_NIST_SRD78_MOMZ14_REST_UV_LINES.py"
)
LOCAL_SOURCE = Path("/content/JWST_0084_NIST_SRD78_MOMZ14_REST_UV_LINES.py")


def prepare_matplotlib() -> None:
    plt.close("all")
    mpl.rcdefaults()
    mpl.rcParams["text.usetex"] = False
    mpl.rcParams["mathtext.fontset"] = "dejavusans"
    mpl.rcParams["font.family"] = "DejaVu Sans"


def load_and_patch_source() -> str:
    urllib.request.urlretrieve(SOURCE_URL, LOCAL_SOURCE)
    source = LOCAL_SOURCE.read_text(encoding="utf-8")
    source = source.replace('VERSION="JWST_0084"', f'VERSION="{VERSION}"', 1)

    marker = "import matplotlib.pyplot as plt\n"
    patch = (
        "import matplotlib.pyplot as plt\n"
        "plt.close('all')\n"
        "plt.rcParams['text.usetex'] = False\n"
        "plt.rcParams['mathtext.fontset'] = 'dejavusans'\n"
        "plt.rcParams['font.family'] = 'DejaVu Sans'\n"
    )
    if marker not in source:
        raise RuntimeError("Could not locate the Matplotlib import in JWST_0084.")
    return source.replace(marker, patch, 1)


def main() -> None:
    prepare_matplotlib()
    source = load_and_patch_source()
    namespace = {
        "__name__": "__main__",
        "__file__": str(LOCAL_SOURCE),
        "__package__": None,
    }
    exec(compile(source, str(LOCAL_SOURCE), "exec"), namespace, namespace)


if __name__ == "__main__":
    main()
