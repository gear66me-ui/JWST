from pathlib import Path
import urllib.request

import numpy as np

VERSION="JWST_0233"
SOURCE_NAME="JWST_0231_MODEL_E_FIXED_FITS_APERTURE_WIDGETS.py"
SOURCE_URL=(
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    + SOURCE_NAME
)
LOCAL_SOURCE=Path("/content")/SOURCE_NAME

if not LOCAL_SOURCE.exists():
    urllib.request.urlretrieve(SOURCE_URL,LOCAL_SOURCE)

source=LOCAL_SOURCE.read_text(encoding="utf-8")
source=source.replace('VERSION="JWST_0231"','VERSION="JWST_0233"')
source=source.replace(
    'MODEL="MODEL_E_FIXED_FITS_APERTURE"',
    'MODEL="MODEL_E_BLUE_FLIPPED_APERTURE"'
)
source=source.replace(
    'APERTURE_VERTICES=np.array([\n'
    '    [71.2,70.1],[68.7,72.5],[66.1,72.7],[61.3,69.9],\n'
    '    [60.0,68.3],[59.8,66.5],[63.3,65.7],[69.5,67.6]\n'
    '],dtype=float)',
    'APERTURE_VERTICES=np.array([\n'
    '    [71.2,56.9],[68.7,54.5],[66.1,54.3],[61.3,57.1],\n'
    '    [60.0,58.7],[59.8,60.5],[63.3,61.3],[69.5,59.4]\n'
    '],dtype=float)'
)
source=source.replace(
    'Model E fixed aperture from user-marked Model D moment-0',
    'Model E confirmed blue vertically flipped aperture'
)
source=source.replace(
    'JWST_0231 — Model E fixed FITS aperture',
    'JWST_0233 — Model E confirmed blue flipped FITS aperture'
)
source=source.replace(
    'Model D moment-0 with fixed Model E aperture',
    'Model D moment-0 with confirmed blue flipped aperture'
)
source=source.replace(
    'Model E fixed FITS extraction',
    'Model E blue flipped FITS extraction'
)
source=source.replace(
    'Fixed-aperture null test',
    'Blue flipped-aperture null test'
)
source=source.replace(
    '["Moment-0 + aperture","Extracted spectrum","Null controls"]',
    '["Moment-0 + blue mask","Extracted spectrum","Null controls"]'
)

namespace={"__name__":"jwst_0233_embedded"}
exec(compile(source,SOURCE_NAME,"exec"),namespace)
namespace["main"]()
