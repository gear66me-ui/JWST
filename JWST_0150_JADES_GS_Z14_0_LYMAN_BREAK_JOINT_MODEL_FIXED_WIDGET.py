# JWST_0150
import io
import warnings
import contextlib
import requests

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0149_JADES_GS_Z14_0_LYMAN_BREAK_JOINT_MODEL_WIDGET.py"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    response = requests.get(SOURCE_URL, timeout=120)
    response.raise_for_status()
    source = response.text

source = source.replace("# JWST_0149", "# JWST_0150", 1)
source = source.replace(
    'lo, hi = np.nanpercentile(fn, [2, 98])\npad = 0.18 * (hi - lo if hi > lo else 1)',
    'finite_fn = np.asarray(fn, dtype=float).reshape(-1)\nfinite_fn = finite_fn[np.isfinite(finite_fn)]\nif finite_fn.size >= 2:\n    q = np.nanpercentile(finite_fn, [2, 98])\n    lo, hi = float(q[0]), float(q[1])\nelif finite_fn.size == 1:\n    lo = hi = float(finite_fn[0])\nelse:\n    lo, hi = -1.0, 1.0\npad = 0.18 * (hi - lo if hi > lo else max(abs(lo), 1.0))'
)
source = source.replace(
    'JWST_0149_JADES_GS_Z14_0_LYMAN_BREAK_JOINT_MODEL.png',
    'JWST_0150_JADES_GS_Z14_0_LYMAN_BREAK_JOINT_MODEL_FIXED.png'
)

silent_stdout = io.StringIO()
silent_stderr = io.StringIO()
with warnings.catch_warnings(), contextlib.redirect_stdout(silent_stdout), contextlib.redirect_stderr(silent_stderr):
    warnings.simplefilter("ignore")
    exec(compile(source, "JWST_0150", "exec"), globals(), globals())
