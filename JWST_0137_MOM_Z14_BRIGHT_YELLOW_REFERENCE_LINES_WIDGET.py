# JWST_0137
import requests

VERSION = "JWST_0137"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0136_MOM_Z14_ORANGE_RED_REFERENCE_LINES_WIDGET.py"

response = requests.get(SOURCE_URL, timeout=120)
response.raise_for_status()
source = response.text
source = source.replace("# JWST_0136", "# JWST_0137", 1)
source = source.replace('VERSION="JWST_0136"', 'VERSION="JWST_0137"')
source = source.replace("ORANGE-RED REFERENCE LINES", "BRIGHT YELLOW REFERENCE LINES")
source = source.replace('REFERENCE_COLOR="#ff5a1f"', 'REFERENCE_COLOR="#ffff00"')
source = source.replace("BLACK_ORANGE_RED_LINES", "BLACK_BRIGHT_YELLOW_LINES")

exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
