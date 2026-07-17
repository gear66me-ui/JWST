# JWST_0145
import requests

VERSION = "JWST_0145"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0143_MOM_Z14_SIX_LINE_SLOPE_DERIVATION_WIDGET.py"

response = requests.get(SOURCE_URL, timeout=120)
response.raise_for_status()
source = response.text
source = source.replace("# JWST_0143", "# JWST_0145", 1)
source = source.replace('VERSION="JWST_0143"', 'VERSION="JWST_0145"')
source = source.replace(
    "MOM-z14 SIX OBSERVED UV LINES — INDIVIDUAL LIKELIHOODS + SLOPE-DERIVED REDSHIFT",
    "MOM-z14 SIX OBSERVED UV LINES — MEASURED SLOPE VS PUBLISHED z=14.44 ± 0.02"
)
source = source.replace('color="white",lw=3.0', 'color="white",lw=1.15')
source = source.replace('color="white",lw=1.5)', 'color="white",lw=0.70)')

old = 'xx=np.linspace(x.min()*0.97,x.max()*1.03,200); ax1.plot(xx,slope*xx,color="#ffd400",lw=2.2,label=f"Best line: λobs = {slope:.6f} λrest; z = slope − 1 = {z_slope:.6f}")'
new = '''xx=np.linspace(x.min()*0.97,x.max()*1.03,200)
        ax1.plot(xx,slope*xx,color="#ffd400",lw=2.2,label=f"Measured six-line fit: λobs = {slope:.6f} λrest; z = {z_slope:.6f}")
        published_slope=1.0+PUBLISHED_Z
        published_slope_lo=1.0+(PUBLISHED_Z-0.020)
        published_slope_hi=1.0+(PUBLISHED_Z+0.020)
        ax1.fill_between(xx,published_slope_lo*xx,published_slope_hi*xx,color="#ff3333",alpha=0.18,label="Published ±0.020 redshift band")
        ax1.plot(xx,published_slope*xx,color="#ff3333",lw=1.35,ls="--",label=f"Published relation: λobs = {published_slope:.6f} λrest; z = {PUBLISHED_Z:.6f}")
        ax1.text(0.02,0.97,"Redshift is the wavelength slope minus one:\n"
                 r"$m=\\lambda_{obs}/\\lambda_{rest}=1+z$"+"\n"
                 f"Measured: m={slope:.6f} → z={z_slope:.6f}\n"
                 f"Published: m={published_slope:.6f} → z={PUBLISHED_Z:.6f} ± 0.020",
                 transform=ax1.transAxes,ha="left",va="top",color="white",fontsize=10,
                 bbox=dict(facecolor="black",edgecolor="#777777",alpha=0.82,pad=7))'''
if old not in source:
    raise RuntimeError("Expected wavelength-slope plotting line not found in source script")
source = source.replace(old, new)

source = source.replace(
    'ax1.set_xlabel("Rest wavelength [μm]"); ax1.set_ylabel("Fitted observed wavelength [μm]")',
    'ax1.set_xlabel("Rest wavelength, λrest [μm]"); ax1.set_ylabel("Observed wavelength, λobs [μm]"); ax1.set_title("Slope derivation: λobs = (1+z) λrest",fontsize=12)'
)
source = source.replace(
    'JWST_0143_MOM_Z14_SIX_LINE_SLOPE_DERIVATION.png',
    'JWST_0145_MOM_Z14_PUBLISHED_SLOPE_BAND_WAVELENGTH.png'
)

exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
