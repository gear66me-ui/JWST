# JWST_0147
import io
import warnings
import contextlib
import requests

VERSION = "JWST_0147"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0143_MOM_Z14_SIX_LINE_SLOPE_DERIVATION_WIDGET.py"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    response = requests.get(SOURCE_URL, timeout=120)
    response.raise_for_status()
    source = response.text

source = source.replace("# JWST_0143", "# JWST_0147", 1)
source = source.replace('VERSION="JWST_0143"', 'VERSION="JWST_0147"')
source = source.replace(
    "MOM-z14 SIX OBSERVED UV LINES — INDIVIDUAL LIKELIHOODS + SLOPE-DERIVED REDSHIFT",
    "MOM-z14 SIX OBSERVED UV LINES — COMBINED-LIKELIHOOD SLOPE FIXED"
)
source = source.replace('color="white",lw=3.0', 'color="white",lw=1.00')
source = source.replace('color="white",lw=1.5)', 'color="white",lw=0.60)')

old = 'xx=np.linspace(x.min()*0.97,x.max()*1.03,200); ax1.plot(xx,slope*xx,color="#ffd400",lw=2.2,label=f"Best line: λobs = {slope:.6f} λrest; z = slope − 1 = {z_slope:.6f}")'
new = '''xx=np.linspace(x.min()*0.97,x.max()*1.03,200)
        combined_slope=1.0+z_joint
        published_slope=1.0+PUBLISHED_Z
        published_slope_lo=1.0+(PUBLISHED_Z-0.020)
        published_slope_hi=1.0+(PUBLISHED_Z+0.020)
        ax1.plot(xx,combined_slope*xx,color="#ffd400",lw=2.2,label=f"Combined six-line likelihood: λobs = {combined_slope:.6f} λrest; z = {z_joint:.6f}")
        ax1.fill_between(xx,published_slope_lo*xx,published_slope_hi*xx,color="#ff3333",alpha=0.16,label="Published z = 14.44 ± 0.020 band")
        ax1.plot(xx,published_slope*xx,color="#ff3333",lw=1.0,ls="--",label=f"Published relation: λobs = {published_slope:.6f} λrest")
        equation_text=(
            r"$m=1+z$"+"\\n"
            f"Combined likelihood: m={combined_slope:.6f}, z={z_joint:.6f}\\n"
            f"Published: m={published_slope:.6f}, z={PUBLISHED_Z:.6f} ± 0.020"
        )
        ax1.text(0.02,0.97,equation_text,transform=ax1.transAxes,ha="left",va="top",color="white",fontsize=10,bbox=dict(facecolor="black",edgecolor="#777777",alpha=0.82,pad=7))'''
if old not in source:
    raise RuntimeError("Expected wavelength-slope plotting line not found")
source = source.replace(old, new)
source = source.replace(
    'ax1.set_xlabel("Rest wavelength [μm]"); ax1.set_ylabel("Fitted observed wavelength [μm]")',
    'ax1.set_xlabel("Rest wavelength, λrest [μm]"); ax1.set_ylabel("Observed wavelength, λobs [μm]"); ax1.set_title("Combined-likelihood slope: λobs = (1+zjoint) λrest",fontsize=12)'
)
source = source.replace(
    'JWST_0143_MOM_Z14_SIX_LINE_SLOPE_DERIVATION.png',
    'JWST_0147_MOM_Z14_COMBINED_LIKELIHOOD_SLOPE_FIXED.png'
)

silent_stdout = io.StringIO()
silent_stderr = io.StringIO()
with warnings.catch_warnings(), contextlib.redirect_stdout(silent_stdout), contextlib.redirect_stderr(silent_stderr):
    warnings.simplefilter("ignore")
    exec(compile(source, "JWST_0147", "exec"), globals(), globals())
