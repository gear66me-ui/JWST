from urllib.request import urlopen

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0249_MODEL_E_APPROVED_MASK_MINUS1Y_0242_STYLE_ANALYSIS.py"

source = urlopen(SOURCE_URL, timeout=60).read().decode("utf-8")

source = source.replace('VERSION="JWST_0249"', 'VERSION="JWST_0250"')
source = source.replace(
    'MODEL="MODEL_E_APPROVED_MASK_MINUS1Y_0242_STYLE_ANALYSIS"',
    'MODEL="MODEL_E_APPROVED_MASK_TWO_PIXELS_DOWN_0242_STYLE_ANALYSIS"',
)

# JWST_0249 moved the approved JWST_0248 mask one displayed pixel upward.
# Move two displayed pixels downward from that position: net +1 pixel from JWST_0248.
source = source.replace('MASK_VERTICES[:,1]-=1.0', 'MASK_VERTICES[:,1]+=1.0')
source = source.replace(
    '# Approved 0248 mask, then moved one FITS pixel downward in negative Y.',
    '# Approved 0248 mask, then shifted +1 array row: two displayed pixels down from JWST_0249.',
)
source = source.replace('APPROVED_MASK_MINUS1Y', 'APPROVED_MASK_TWO_PIXELS_DOWN')
source = source.replace('Approved mask −1Y', 'Approved mask: two pixels down from JWST_0249')
source = source.replace('Approved mask moved one pixel down', 'Approved mask moved two displayed pixels down from JWST_0249')
source = source.replace('shifted one pixel toward −Y', 'moved two displayed pixels downward from JWST_0249')
source = source.replace('Includes requested −1 pixel shift', 'Net +1 array-row shift from JWST_0248')
source = source.replace('approved mask analysis', 'corrected approved-mask analysis')

exec(compile(source, "JWST_0250_runtime.py", "exec"), {"__name__": "__main__"})
