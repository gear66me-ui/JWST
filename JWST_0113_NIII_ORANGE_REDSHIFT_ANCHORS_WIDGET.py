# JWST_0113
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

VERSION = "JWST_0113"
print(f"CODE OUTPUT: {VERSION}")

# IMPORTANT: this verified GLASS source is at z=9.31102; it is not MoM-z14 at z=14.44.
Z_ADOPTED = 9.31102

# Rest-UV reference features commonly used to support spectroscopic redshifts.
# The MoM-z14 paper specifically describes a Ly-alpha absorption break plus UV emission lines;
# therefore the orange metal-line markers below are emission-line references, not absorption lines.
REDSHIFT_ANCHORS = {
    "Ly-alpha break 1215.67 A": 1215.67,
    "N IV] 1486.50 A": 1486.50,
    "C IV 1548.20 A": 1548.20,
    "C IV 1550.77 A": 1550.77,
    "He II 1640.42 A": 1640.42,
    "O III] 1660.81 A": 1660.81,
    "O III] 1666.15 A": 1666.15,
    "N III] 1746.82 A": 1746.82,
    "N III] 1748.65 A": 1748.65,
    "N III] 1749.67 A": 1749.67,
    "N III] 1752.16 A": 1752.16,
    "N III] 1753.99 A": 1753.99,
    "C III] 1906.68 A": 1906.68,
    "C III] 1908.73 A": 1908.73,
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

valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0) & (flux != 0)
wave, flux, err = wave[valid], flux[valid], err[valid]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]
rest = wave * 10000.0 / (1.0 + Z_ADOPTED)

# Default to the N III] region while allowing navigation across the full G235H spectrum.
niii_min = 1700.0 * (1.0 + Z_ADOPTED) / 10000.0
niii_max = 1800.0 * (1.0 + Z_ADOPTED) / 10000.0
wmin, wmax = float(wave.min()), float(wave.max())

line = widgets.Dropdown(
    options=list(REDSHIFT_ANCHORS),
    value="N III] 1752.16 A",
    description="Reference",
    layout=widgets.Layout(width="360px"),
)
cursor = widgets.FloatSlider(
    value=float((niii_min + niii_max) / 2), min=wmin, max=wmax,
    step=(wmax-wmin)/3000, description="Cursor um", readout_format=".6f",
    continuous_update=False, layout=widgets.Layout(width="900px"),
)
window = widgets.FloatSlider(
    value=0.0015, min=0.0002, max=0.0100, step=0.0001,
    description="Peak +/- um", readout_format=".4f", continuous_update=False,
    layout=widgets.Layout(width="650px"),
)
zoom = widgets.FloatRangeSlider(
    value=(max(wmin, niii_min), min(wmax, niii_max)), min=wmin, max=wmax,
    step=(wmax-wmin)/800, description="View um", readout_format=".6f",
    continuous_update=False, layout=widgets.Layout(width="900px"),
)
show_error = widgets.Checkbox(value=True, description="Show 1 sigma")
show_all = widgets.Checkbox(value=True, description="Orange redshift anchors")
out = widgets.Output()


def draw(*_):
    out.clear_output(wait=True)
    rest_ref = REDSHIFT_ANCHORS[line.value]
    x0, x1 = zoom.value
    m = (wave >= x0) & (wave <= x1)
    local = np.abs(wave - cursor.value) <= window.value
    nearest = int(np.argmin(np.abs(wave - cursor.value)))
    peak = int(np.where(local)[0][np.argmax(flux[local])]) if np.any(local) else nearest
    implied_z = wave[peak] * 10000.0 / rest_ref - 1.0
    expected = rest_ref * (1.0 + Z_ADOPTED) / 10000.0

    with out:
        fig, ax = plt.subplots(figsize=(14, 6.5))
        ax.plot(wave[m], flux[m], lw=0.75)
        ax.scatter(wave[m], flux[m], s=9)
        if show_error.value:
            ax.fill_between(wave[m], flux[m]-err[m], flux[m]+err[m], alpha=0.16)

        if show_all.value:
            y_top = np.nanpercentile(flux[m], 97) if np.any(m) else 1.0
            for label, rest_a in REDSHIFT_ANCHORS.items():
                obs_um = rest_a * (1.0 + Z_ADOPTED) / 10000.0
                if x0 <= obs_um <= x1:
                    ax.axvline(obs_um, color="orange", ls="--", lw=1.4, alpha=0.95)
                    ax.text(obs_um, y_top, label.split(" A")[0], color="orange",
                            rotation=90, va="top", ha="right", fontsize=8)

        ax.axvline(expected, color="orange", ls="-", lw=2.2,
                   label=f"Selected reference at z={Z_ADOPTED:.5f}")
        ax.axvline(cursor.value, ls=":", lw=1.0, label="Cursor")
        ax.scatter([wave[peak]], [flux[peak]], s=90, marker="x", label="Local peak")
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Observed wavelength [um]")
        ax.set_ylabel("Flux [native FITS units]")
        ax.set_title("Verified GLASS source 10003 (z=9.31102) — interactive rest-UV line inspector")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

        print(f"REFERENCE             {line.value}")
        print(f"EXPECTED OBSERVED     {expected:.6f} um")
        print(f"SELECTED LOCAL PEAK   {wave[peak]:.6f} um")
        print(f"PEAK FLUX             {flux[peak]:.6e}")
        print(f"IMPLIED REDSHIFT      {implied_z:.6f}")
        print(f"DELTA Z               {implied_z-Z_ADOPTED:+.6f}")
        print("SCIENCE NOTE          Orange Ly-alpha is an absorption break; orange metal features are emission-line references.")
        print("IDENTITY NOTE         This file is source 10003 at z=9.31102, not MoM-z14 at z=14.44.")

for control in (line, cursor, window, zoom, show_error, show_all):
    control.observe(draw, names="value")

display(widgets.VBox([
    widgets.HTML("<b>JWST_0113 — N III / rest-UV interactive redshift inspector</b>"),
    widgets.HBox([line, window, show_error, show_all]),
    zoom,
    cursor,
    out,
]))
draw()
print(f"END {VERSION}")
