# JWST_0167
import requests

SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0166_JADES_GS_Z11_0_LYMAN_ALPHA_REDSHIFT_DERIVATION_WIDGET.py"
source = requests.get(SOURCE_URL, timeout=120).text

replacements = {
    "JWST_0166": "JWST_0167",
    "Plot Ly-alpha redshift derivation": "Plot continuum-break redshift derivation",
    "Best Lyman-break fit, z=": "Empirical continuum-break model, z=",
    "Fitted Lyα break =": "Fitted continuum break anchored to rest-frame Lyα =",
    "Observed Lyman-α break used as the redshift anchor": "Observed continuum break anchored to rest-frame Lyα",
    "Observed Lyα-break wavelength [μm]": "Observed continuum-break wavelength [μm]",
    "JADES-GS-z11-0 — direct Lyα-break redshift derivation": "JADES-GS-z11-0 — continuum-break redshift derivation anchored to Lyα",
    "LYMAN_ALPHA_REDSHIFT_DERIVATION": "CONTINUUM_BREAK_REDSHIFT_DERIVATION",
    "LYMAN_ALPHA_REDSHIFT_CURVE": "CONTINUUM_BREAK_REDSHIFT_CURVE",
}
for old, new in replacements.items():
    source = source.replace(old, new)

exec(compile(source, "JWST_0167_runtime.py", "exec"), globals())
