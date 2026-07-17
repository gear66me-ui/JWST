# JWST_0118
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

VERSION = "JWST_0118"
print(f"CODE OUTPUT: {VERSION}")

Z_ADOPTED = 9.31102
TARGET_NAME = "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits"
ROOTS = [
    Path("/content/JWST_BACKUP_REPO/FITS_DATA/MOM_Z14_SOURCE_10003"),
    Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS"),
    Path("/content/JWST_OUTPUT/DATA/JWST_0116/FITS"),
]

paths = []
for root in ROOTS:
    if root.exists():
        paths.extend(root.rglob(TARGET_NAME))
if not paths:
    raise FileNotFoundError(TARGET_NAME)
path = paths[0]

with fits.open(path, memmap=False) as hdul:
    tab = hdul["SPEC1D"].data
    wave = np.asarray(tab["wave"], dtype=float).ravel()
    flux = np.asarray(tab["flux"], dtype=float).ravel()
    err = np.asarray(tab["err"], dtype=float).ravel()

valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err)
wave, flux, err = wave[valid], flux[valid], err[valid]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]
rest = wave * 10000.0 / (1.0 + Z_ADOPTED)

LINES = {
    "N III] 1746.82 A": 1746.82,
    "N III] 1748.65 A": 1748.65,
    "N III] 1749.67 A": 1749.67,
    "N III] 1752.16 A": 1752.16,
    "N III] 1753.99 A": 1753.99,
}

wmin, wmax = float(wave.min()), float(wave.max())
nii_obs = [v * (1 + Z_ADOPTED) / 10000.0 for v in LINES.values()]
nii_center = float(np.mean(nii_obs))
nii_half = 0.025

mode = widgets.ToggleButtons(
    options=["Full 3814", "N III zoom", "Custom"],
    value="Full 3814",
    description="View"
)
zoom = widgets.FloatRangeSlider(
    value=(wmin, wmax), min=wmin, max=wmax,
    step=(wmax - wmin) / 2000.0,
    description="Range um", readout_format=".6f",
    continuous_update=False,
    layout=widgets.Layout(width="900px")
)
cursor = widgets.FloatSlider(
    value=nii_center, min=wmin, max=wmax,
    step=(wmax - wmin) / 38140.0,
    description="Cursor um", readout_format=".6f",
    continuous_update=False,
    layout=widgets.Layout(width="900px")
)
window = widgets.FloatSlider(
    value=0.0015, min=0.0001, max=0.02, step=0.0001,
    description="Search +/-", readout_format=".4f",
    continuous_update=False,
    layout=widgets.Layout(width="600px")
)
show_err = widgets.Checkbox(value=True, description="Show 1 sigma")
show_orange = widgets.Checkbox(value=True, description="Show N III anchors")
show_points = widgets.Checkbox(value=True, description="Show native samples")
out = widgets.Output()


def set_mode(change=None):
    if mode.value == "Full 3814":
        zoom.value = (wmin, wmax)
    elif mode.value == "N III zoom":
        zoom.value = (max(wmin, nii_center - nii_half), min(wmax, nii_center + nii_half))


def draw(*_):
    out.clear_output(wait=True)
    x0, x1 = zoom.value
    m = (wave >= x0) & (wave <= x1)
    nearest = int(np.argmin(np.abs(wave - cursor.value)))
    local = np.abs(wave - cursor.value) <= window.value
    peak = int(np.where(local)[0][np.argmax(flux[local])]) if np.any(local) else nearest

    with out:
        fig, ax = plt.subplots(figsize=(14, 6.5))
        ax.plot(wave[m], flux[m], lw=0.65)
        if show_points.value:
            ax.scatter(wave[m], flux[m], s=7)
        if show_err.value:
            ax.fill_between(wave[m], flux[m] - err[m], flux[m] + err[m], alpha=0.16)
        if show_orange.value:
            for label, rest_a in LINES.items():
                obs = rest_a * (1.0 + Z_ADOPTED) / 10000.0
                if x0 <= obs <= x1:
                    ax.axvline(obs, color="orange", lw=1.5, alpha=0.95)
                    ax.text(obs, 0.98, label.replace("N III] ", ""), color="orange",
                            rotation=90, ha="right", va="top",
                            transform=ax.get_xaxis_transform(), fontsize=8)
        ax.axvline(cursor.value, ls=":", lw=1.0)
        ax.scatter([wave[nearest]], [flux[nearest]], s=40, marker="o", label="Nearest sample")
        ax.scatter([wave[peak]], [flux[peak]], s=80, marker="x", label="Local maximum")
        ax.set_xlim(x0, x1)
        ax.set_xlabel("Observed wavelength [um]")
        ax.set_ylabel("Flux [native FITS units]")
        ax.set_title("MoM-z14 source 10003 — verified GLASS G235H — all 3814 native samples")
        ax.grid(alpha=0.25)
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(f"FILE                  {path}")
        print(f"TOTAL NATIVE SAMPLES  {len(wave)}")
        print(f"VISIBLE SAMPLES       {int(np.count_nonzero(m))}")
        print(f"NEAREST INDEX         {nearest}")
        print(f"NEAREST OBSERVED      {wave[nearest]:.9f} um")
        print(f"NEAREST REST          {rest[nearest]:.6f} A")
        print(f"NEAREST FLUX          {flux[nearest]:.9e}")
        print(f"NEAREST ERROR         {err[nearest]:.9e}")
        print(f"LOCAL MAX INDEX       {peak}")
        print(f"LOCAL MAX OBSERVED    {wave[peak]:.9f} um")
        print(f"LOCAL MAX REST        {rest[peak]:.6f} A")
        print(f"LOCAL MAX FLUX        {flux[peak]:.9e}")

mode.observe(set_mode, names="value")
for control in (zoom, cursor, window, show_err, show_orange, show_points):
    control.observe(draw, names="value")

display(widgets.VBox([
    widgets.HTML("<b>JWST_0118 — full 3814-point G235H inspector</b>"),
    widgets.HBox([mode, window, show_err, show_orange, show_points]),
    zoom,
    cursor,
    out,
]))
set_mode()
draw()
print(f"# {VERSION}")
