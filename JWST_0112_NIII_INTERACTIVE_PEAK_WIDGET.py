# JWST_0112
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import ipywidgets as widgets
from IPython.display import display

try:
    from google.colab import output
    output.enable_custom_widget_manager()
except Exception:
    pass

VERSION = "JWST_0112"
print(f"CODE OUTPUT: {VERSION}")

Z_ADOPTED = 9.31102
LINES = {
    "N III] 1746.82 A": 1746.82,
    "N III] 1748.65 A": 1748.65,
    "N III] 1749.67 A": 1749.67,
    "N III] 1752.16 A": 1752.16,
    "N III] 1753.99 A": 1753.99,
}

name = "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits"
roots = [
    Path("/content/JWST_BACKUP_REPO/FITS_DATA/MOM_Z14_SOURCE_10003"),
    Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS"),
]
paths = []
for root in roots:
    if root.exists():
        paths.extend(root.rglob(name))
if not paths:
    raise FileNotFoundError(name)
path = paths[0]

with fits.open(path, memmap=False) as hdul:
    tab = hdul["SPEC1D"].data
    wave = np.asarray(tab["wave"], float)
    flux = np.asarray(tab["flux"], float)
    err = np.asarray(tab["err"], float)

rest = wave * 10000.0 / (1.0 + Z_ADOPTED)
valid = (np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) &
         (err > 0) & (flux != 0) & (rest >= 1700) & (rest <= 1800))
wave, flux, err, rest = wave[valid], flux[valid], err[valid], rest[valid]
order = np.argsort(wave)
wave, flux, err, rest = wave[order], flux[order], err[order], rest[order]

wmin, wmax = float(wave.min()), float(wave.max())
line = widgets.Dropdown(options=list(LINES), value="N III] 1752.16 A", description="Line")
cursor = widgets.FloatSlider(value=float(np.median(wave)), min=wmin, max=wmax,
                             step=(wmax-wmin)/2000, description="Cursor um",
                             readout_format=".6f", continuous_update=False,
                             layout=widgets.Layout(width="850px"))
window = widgets.FloatSlider(value=0.0015, min=0.0002, max=0.0100, step=0.0001,
                             description="Peak +/- um", readout_format=".4f",
                             continuous_update=False,
                             layout=widgets.Layout(width="600px"))
zoom = widgets.FloatRangeSlider(value=(wmin, wmax), min=wmin, max=wmax,
                                step=(wmax-wmin)/500, description="View um",
                                readout_format=".6f", continuous_update=False,
                                layout=widgets.Layout(width="850px"))
show_error = widgets.Checkbox(value=True, description="Show 1 sigma")
out = widgets.Output()

def draw(*_):
    out.clear_output(wait=True)
    rest_ref = LINES[line.value]
    x0, x1 = zoom.value
    m = (wave >= x0) & (wave <= x1)
    local = np.abs(wave - cursor.value) <= window.value
    nearest = int(np.argmin(np.abs(wave - cursor.value)))
    peak = int(np.where(local)[0][np.argmax(flux[local])]) if np.any(local) else nearest
    implied_z = wave[peak] * 10000.0 / rest_ref - 1.0
    expected = rest_ref * (1.0 + Z_ADOPTED) / 10000.0

    with out:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(wave[m], flux[m], lw=0.75)
        ax.scatter(wave[m], flux[m], s=10)
        if show_error.value:
            ax.fill_between(wave[m], flux[m]-err[m], flux[m]+err[m], alpha=0.18)
        ax.axvline(expected, ls="--", lw=1.0, label=f"Expected at z={Z_ADOPTED:.5f}")
        ax.axvline(cursor.value, ls=":", lw=1.0, label="Cursor")
        ax.scatter([wave[peak]], [flux[peak]], s=85, marker="x", label="Local peak")
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Observed wavelength [um]")
        ax.set_ylabel("Flux [native FITS units]")
        ax.set_title("MoM-z14 source 10003 — N III] G235H interactive peak inspector")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()
        print(f"REFERENCE             {line.value}")
        print(f"EXPECTED OBSERVED     {expected:.6f} um")
        print(f"SELECTED LOCAL PEAK   {wave[peak]:.6f} um")
        print(f"PEAK FLUX             {flux[peak]:.6e}")
        print(f"IMPLIED REDSHIFT      {implied_z:.6f}")
        print(f"DELTA Z               {implied_z-Z_ADOPTED:+.6f}")
        print("NOTE                  Peak selection is exploratory; line identification must be validated against noise and the full multiplet.")

for control in (line, cursor, window, zoom, show_error):
    control.observe(draw, names="value")

display(widgets.VBox([widgets.HTML("<b>JWST_0112 N III] interactive peak inspector</b>"),
                      widgets.HBox([line, window, show_error]), zoom, cursor, out]))
draw()
print(f"END {VERSION}")
