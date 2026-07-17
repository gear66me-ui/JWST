# JWST_0138
import requests

VERSION = "JWST_0138"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0137_MOM_Z14_BRIGHT_YELLOW_REFERENCE_LINES_WIDGET.py"

response = requests.get(SOURCE_URL, timeout=120)
response.raise_for_status()
source = response.text
source = source.replace("# JWST_0137", "# JWST_0138", 1)
source = source.replace('VERSION = "JWST_0137"', 'VERSION = "JWST_0138"')
source = source.replace('VERSION="JWST_0137"', 'VERSION="JWST_0138"')
source = source.replace("BRIGHT YELLOW REFERENCE LINES", "WHITE REFERENCE LINES")
source = source.replace('REFERENCE_COLOR="#ffff00"', 'REFERENCE_COLOR="#ffffff"')
source = source.replace("BLACK_BRIGHT_YELLOW_LINES", "BLACK_WHITE_LINES")

exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
