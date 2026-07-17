# JWST_0114
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

VERSION = "JWST_0114"
print(f"CODE OUTPUT: {VERSION}")

Z0 = 9.31102
NIII_LINES = {
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
    wave = np.asarray(tab["wave"], float).ravel()
    flux = np.asarray(tab["flux"], float).ravel()
    err = np.asarray(tab["err"], float).ravel()

rest0 = wave * 10000.0 / (1.0 + Z0)
valid = (np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) &
         (err > 0) & (flux != 0) & (rest0 >= 1700) & (rest0 <= 1800))
wave, flux, err = wave[valid], flux[valid], err[valid]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]

wmin, wmax = float(wave.min()), float(wave.max())
median_step = float(np.median(np.diff(wave)))

z_slider = widgets.FloatSlider(
    value=Z0, min=8.8, max=9.8, step=0.0005,
    description="Trial z", readout_format=".4f",
    continuous_update=False, layout=widgets.Layout(width="850px"))
zoom = widgets.FloatRangeSlider(
    value=(wmin, wmax), min=wmin, max=wmax,
    step=max((wmax-wmin)/600, 1e-6), description="View um",
    readout_format=".6f", continuous_update=False,
    layout=widgets.Layout(width="850px"))
oversample = widgets.IntSlider(
    value=12, min=1, max=50, step=1, description="Display x",
    continuous_update=False, layout=widgets.Layout(width="500px"))
smooth = widgets.FloatSlider(
    value=0.0, min=0.0, max=3.0, step=0.25,
    description="Smooth pix", readout_format=".2f",
    continuous_update=False, layout=widgets.Layout(width="500px"))
show_error = widgets.Checkbox(value=True, description="Show 1 sigma")
show_native = widgets.Checkbox(value=True, description="Show native points")
out = widgets.Output()


def gaussian_smooth(y, sigma_pix):
    if sigma_pix <= 0:
        return y.copy()
    radius = max(1, int(np.ceil(4*sigma_pix)))
    x = np.arange(-radius, radius+1, dtype=float)
    k = np.exp(-0.5*(x/sigma_pix)**2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def draw(*_):
    out.clear_output(wait=True)
    z = z_slider.value
    x0, x1 = zoom.value
    m = (wave >= x0) & (wave <= x1)
    if not np.any(m):
        with out:
            print("No samples in selected range")
        return

    y_native = gaussian_smooth(flux, smooth.value)
    n_display = max(len(wave), len(wave)*oversample.value)
    x_dense = np.linspace(wave.min(), wave.max(), n_display)
    y_dense = np.interp(x_dense, wave, y_native)
    md = (x_dense >= x0) & (x_dense <= x1)

    with out:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(x_dense[md], y_dense[md], lw=1.0, label="Interpolated display curve")
        if show_native.value:
            ax.plot(wave[m], y_native[m], lw=0.7, alpha=0.7)
            ax.scatter(wave[m], y_native[m], s=16, label="Native FITS samples")
        if show_error.value:
            ax.fill_between(wave[m], flux[m]-err[m], flux[m]+err[m], alpha=0.18)

        ymax = np.nanmax(y_native[m])
        ymin = np.nanmin(y_native[m])
        yrange = ymax-ymin if ymax > ymin else 1.0

        rows = []
        for label, rest_a in NIII_LINES.items():
            obs_um = rest_a*(1.0+z)/10000.0
            ax.axvline(obs_um, color="orange", ls="--", lw=1.5)
            if x0 <= obs_um <= x1:
                ax.text(obs_um, ymax+0.03*yrange, label.replace(" A", ""),
                        rotation=90, color="orange", va="bottom", ha="center", fontsize=9)
            j = int(np.argmin(np.abs(wave-obs_um)))
            rows.append((label, obs_um, wave[j], (wave[j]-obs_um)*1e4, flux[j]))

        ax.set_xlim(x0, x1)
        ax.set_xlabel("Observed wavelength [um]")
        ax.set_ylabel("Flux [native FITS units]")
        ax.set_title("Source 10003 — N III] native sampling and trial-redshift alignment")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        print(f"TRIAL REDSHIFT          {z:.6f}")
        print(f"NATIVE VALID SAMPLES    {len(wave)}")
        print(f"MEDIAN NATIVE STEP      {median_step:.9f} um")
        print(f"DISPLAY OVERSAMPLING    {oversample.value}x (interpolation only; no new measured information)")
        print(f"SMOOTHING               {smooth.value:.2f} native pixels")
        print()
        print(f"{'LINE':21s} {'EXPECTED_UM':>12s} {'NEAREST_UM':>12s} {'DELTA_A':>10s} {'FLUX':>14s}")
        for label, expected, nearest, delta_a, f in rows:
            print(f"{label:21s} {expected:12.6f} {nearest:12.6f} {delta_a:10.3f} {f:14.6e}")
        print()
        print("NOTE: N III] is an emission multiplet, not an absorption feature.")
        print("NOTE: Interpolation adds visual granularity only. It cannot create additional JWST measurements.")

for control in (z_slider, zoom, oversample, smooth, show_error, show_native):
    control.observe(draw, names="value")

display(widgets.VBox([
    widgets.HTML("<b>JWST_0114 N III] native-data redshift scan</b>"),
    z_slider,
    zoom,
    widgets.HBox([oversample, smooth]),
    widgets.HBox([show_native, show_error]),
    out,
]))
draw()
print(f"END {VERSION}")
