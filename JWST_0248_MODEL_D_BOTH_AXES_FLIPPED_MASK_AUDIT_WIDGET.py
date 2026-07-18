from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0248"
MODEL="MODEL_D_BOTH_AXES_FLIPPED_MASK_AUDIT_WIDGET"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
OUTPUT_PIXELS=1024

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"

# Latest user contour traced from the 1024 master display in its drawn orientation.
SCREEN_VERTICES=np.array([
    [57.58,68.25],
    [58.08,67.17],
    [59.67,66.42],
    [61.83,66.17],
    [63.83,66.58],
    [64.92,67.58],
    [65.17,69.00],
    [64.50,70.42],
    [62.83,71.25],
    [60.75,71.50],
    [58.92,70.83],
    [57.67,69.58],
],dtype=float)

# User-requested correction: flip on both image axes.
MASK_VERTICES=np.column_stack((127.0-SCREEN_VERTICES[:,0],127.0-SCREEN_VERTICES[:,1]))


def spec_axis(header):
    for axis in range(1,int(header["NAXIS"])+1):
        if "FREQ" in str(header.get(f"CTYPE{axis}","")).upper():
            return axis
    raise RuntimeError("No frequency axis found")


def axis_values_ghz(header,axis):
    n=int(header[f"NAXIS{axis}"])
    pixel=np.arange(n,dtype=float)+1.0
    step=float(header.get(f"CDELT{axis}",header.get(f"CD{axis}_{axis}")))
    return (float(header[f"CRVAL{axis}"])+(pixel-float(header[f"CRPIX{axis}"]))*step)/1e9


def collapse_to_spectral_cube(array,np_spec):
    array=np.moveaxis(array,np_spec,0)
    while array.ndim>3:
        array=np.nanmean(array,axis=1)
    return array


def make_rgb(moment0):
    finite=moment0[np.isfinite(moment0)]
    vmin=float(np.nanmin(finite))
    vmax=float(np.nanmax(finite))
    normalized=np.clip((moment0-vmin)/(vmax-vmin),0.0,1.0)
    normalized=np.nan_to_num(normalized,nan=0.0,posinf=1.0,neginf=0.0)
    rgb=(plt.get_cmap("viridis")(normalized)[...,:3]*255.0).astype(np.uint8)
    return Image.fromarray(rgb,mode="RGB").resize(
        (OUTPUT_PIXELS,OUTPUT_PIXELS),Image.Resampling.NEAREST
    )


def overlay_mask(image,vertices):
    out=image.copy()
    draw=ImageDraw.Draw(out)
    scale=OUTPUT_PIXELS/128.0
    points=[(float(x)*scale,float(y)*scale) for x,y in vertices]
    points.append(points[0])
    draw.line(points,fill=(255,35,20),width=5,joint="curve")
    return out


def png_bytes(image):
    from io import BytesIO
    buf=BytesIO()
    image.save(buf,format="PNG",compress_level=0,optimize=False)
    return buf.getvalue()


def main():
    warnings.filterwarnings("ignore")
    if not CUBE.exists():
        raise FileNotFoundError(CUBE)

    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        header=hdul[0].header
        data=hdul[0].data
        fits_frequency_axis=spec_axis(header)
        frequency_ghz=axis_values_ghz(header,fits_frequency_axis)
        numpy_frequency_axis=data.ndim-fits_frequency_axis

        target_x,target_y=WCS(header).celestial.world_to_pixel_values(RA_DEG,DEC_DEG)
        target_x=int(np.rint(target_x))
        target_y=int(np.rint(target_y))
        half=64
        y_slice=slice(target_y-half,target_y+half)
        x_slice=slice(target_x-half,target_x+half)

        native_mhz=float(np.nanmedian(np.abs(np.diff(frequency_ghz)))*1000.0)
        channels_per_group=max(1,int(round(5.0/native_mhz)))
        nearest=int(np.argmin(np.abs(frequency_ghz-TARGET_GHZ)))
        lower=max(0,nearest-14*channels_per_group)
        upper=min(len(frequency_ghz),nearest+14*channels_per_group)
        groups=[np.arange(i,min(i+channels_per_group,upper)) for i in range(lower,upper,channels_per_group)]
        centers_ghz=np.array([np.mean(frequency_ghz[group]) for group in groups])
        grouped_cube=np.empty((len(groups),128,128),dtype=np.float32)

        for index,group in enumerate(groups):
            selection=[slice(None)]*data.ndim
            selection[numpy_frequency_axis]=group
            selection[-2]=y_slice
            selection[-1]=x_slice
            grouped_cube[index]=np.nanmean(
                collapse_to_spectral_cube(np.asarray(data[tuple(selection)],dtype=np.float32),numpy_frequency_axis),
                axis=0,
            )
            if (index+1)%5==0 or index==len(groups)-1:
                print(f"5 MHz cube: {index+1:3d}/{len(groups):3d}")

        del data
        gc.collect()

    velocity_kms=C_KMS*(TARGET_GHZ-centers_ghz)/TARGET_GHZ
    line_selection=(velocity_kms>=-14.0)&(velocity_kms<=29.0)
    if int(line_selection.sum())<5:
        nearest_groups=np.argsort(np.abs(centers_ghz-TARGET_GHZ))[:8]
        line_selection=np.zeros(len(centers_ghz),dtype=bool)
        line_selection[nearest_groups]=True

    moment0=np.sum(grouped_cube[line_selection],axis=0)
    raw_image=make_rgb(moment0)
    masked_image=overlay_mask(raw_image,MASK_VERTICES)
    raw_bytes=png_bytes(raw_image)
    masked_bytes=png_bytes(masked_image)

    raw_widget=widgets.Image(value=raw_bytes,format="png",layout=widgets.Layout(width="760px",max_width="100%"))
    masked_widget=widgets.Image(value=masked_bytes,format="png",layout=widgets.Layout(width="760px",max_width="100%"))

    toggle_image=widgets.Image(value=raw_bytes,format="png",layout=widgets.Layout(width="760px",max_width="100%"))
    toggle=widgets.ToggleButtons(
        options=[("Raw Model D","raw"),("Both-axes flipped mask","mask")],
        value="raw",
        description="Compare:"
    )
    def switch(change):
        toggle_image.value=raw_bytes if change["new"]=="raw" else masked_bytes
    toggle.observe(switch,names="value")

    bounds_html=widgets.HTML(
        f"<div style='font-family:monospace;background:#071019;color:#eaf2ff;padding:10px;border-radius:8px'>"
        f"<b>{VERSION} — MASK LOCATION AUDIT ONLY</b><br>"
        f"Transformation: x → 127 − x, y → 127 − y<br>"
        f"Corrected bounds: X {MASK_VERTICES[:,0].min():.2f}–{MASK_VERTICES[:,0].max():.2f}, "
        f"Y {MASK_VERTICES[:,1].min():.2f}–{MASK_VERTICES[:,1].max():.2f}<br>"
        f"No extraction, centroid, S/N, or null test is run in this audit.</div>"
    )

    display(bounds_html)
    display(widgets.HTML("<h4>Raw Model D moment-0</h4>"))
    display(raw_widget)
    display(widgets.HTML("<h4>Both-axes flipped mask overlay</h4>"))
    display(masked_widget)
    display(widgets.HTML("<h4>Toggle comparison</h4>"))
    display(toggle)
    display(toggle_image)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"MODEL: {MODEL}")
    print(f"MASK X BOUNDS: {MASK_VERTICES[:,0].min():.3f} to {MASK_VERTICES[:,0].max():.3f}")
    print(f"MASK Y BOUNDS: {MASK_VERTICES[:,1].min():.3f} to {MASK_VERTICES[:,1].max():.3f}")
    print("AUDIT ONLY: no scientific extraction performed")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__=="__main__":
    main()
