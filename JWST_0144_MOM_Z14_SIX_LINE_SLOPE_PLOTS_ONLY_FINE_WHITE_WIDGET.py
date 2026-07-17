# JWST_0144
import requests

VERSION = "JWST_0144"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0143_MOM_Z14_SIX_LINE_SLOPE_DERIVATION_WIDGET.py"

response = requests.get(SOURCE_URL, timeout=120)
response.raise_for_status()
source = response.text
source = source.replace("# JWST_0143", "# JWST_0144", 1)
source = source.replace('VERSION="JWST_0143"', 'VERSION="JWST_0144"')
source = source.replace("SIX OBSERVED UV LINES — INDIVIDUAL LIKELIHOODS + SLOPE-DERIVED REDSHIFT", "SIX OBSERVED UV LINES — PLOTS ONLY + FINE WHITE FITS")
source = source.replace('color="white",lw=3.0', 'color="white",lw=1.15')
source = source.replace('color="white",lw=1.5)', 'color="white",lw=0.70)')
source = source.replace('JWST_0143_MOM_Z14_SIX_LINE_SLOPE_DERIVATION.png', 'JWST_0144_MOM_Z14_SIX_LINE_SLOPE_PLOTS_ONLY_FINE_WHITE.png')
source = source.replace('print(', '_quiet_print(')

_quiet_print = lambda *args, **kwargs: None
exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
