# JWST_0152
import io
import logging
import warnings
import contextlib
import requests

VERSION = "JWST_0152"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0151_JADES_GS_Z14_0_GOLD_STANDARD_BREAK_SIX_LINES_WIDGET.py"

warnings.filterwarnings("ignore")
for name in ["astroquery", "astroquery.mast", "astropy", "astropy.utils"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    r = requests.get(SOURCE_URL, timeout=120)
    r.raise_for_status()
    source = r.text

source = source.replace("# JWST_0151", "# JWST_0152", 1)
source = source.replace('VERSION="JWST_0151"', 'VERSION="JWST_0152"')
source = source.replace("JWST_0151_JADES_GS_Z14_0_FEATURES.csv", "JWST_0152_JADES_GS_Z14_0_FEATURES.csv")
source = source.replace("JWST_0151_JADES_GS_Z14_0_BREAK_PLUS_SIX_LINES.png", "JWST_0152_JADES_GS_Z14_0_BREAK_PLUS_SIX_LINES.png")

exec(compile(source, "JWST_0152", "exec"), globals(), globals())
