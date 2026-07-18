from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath

from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0232"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
OUT=Path("/content/JWST_OUTPUT/PNG")
DRIVE=ROOT/"PNG"
OUT.mkdir(parents=True,exist_ok=True)
DRIVE.mkdir(parents=True,exist_ok=True)

VERTICES=np.array([
    [71.2,70.1],
    [68.7,72.5],
    [66.1,72.7],
    [61.3,69.9],
    [60.0,68.3],
    [59.8,66.5],
    [63.3,65.7],
    [69.5,67.6]
],dtype=float)


def spec_axis(h):
    for ax in range(1,int(h["NAXIS"])+1):
        if "FREQ" in str(h.get(f"CTYPE{ax}","")).upper():
            return ax
    raise RuntimeError("No frequency axis")


def axis_values(h,ax):
    n=int(h[f"NAXIS{ax}"])
    p=np.arange(n)+1.0
    return (
        float(h[f"CRVAL{ax}"])
        +(p-float(h[f"CRPIX{ax}"]))
        *float(h.get(f"CDELT{ax}",h.get(f"CD{ax}_{ax}")))
    )/1e9


def collapse(a,np_spec):
    a=np.moveaxis(a,np_spec,0)
    while a.ndim>3:
        a=np.nanmean(a,axis=1)
    return a


def polygon_mask(vertices,shape):
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    pts=np.column_stack((xx.ravel(),yy.ravel()))
    return MplPath(vertices).contains_points(pts).reshape(shape)


def save(fig,path):
    fig.savefig(path,dpi=260,bbox_inches="tight",pad_inches=0.04)
    plt.close(fig)
    (DRIVE/path.name).write_bytes(path.read_bytes())


def main():
    warnings.filterwarnings("ignore")
    if not CUBE.exists():
        raise FileNotFoundError(CUBE)

    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header
        data=hdul[0].data
        fax=spec_axis(h)
        freq=axis_values(h,fax)
        np_spec=data.ndim-fax
        aw=WCS(h).celestial
        tx0,ty0=aw.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0))
        ty=int(np.rint(ty0))

        half=64
        ys=slice(ty-half,ty+half)
        xs=slice(tx-half,tx+half)
        native=np.nanmedian(np.abs(np.diff(freq)))*1000
        k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ)))
        lo=max(0,near-14*k)
        hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups])
        cube=np.empty((len(groups),128,128),np.float32)

        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim
            sl[np_spec]=g
            sl[-2]=ys
            sl[-1]=xs
            cube[j]=np.nanmean(
                collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),
                axis=0
            )
            if (j+1)%5==0 or j==len(groups)-1:
                print(f"5 MHz cube: {j+1:3d}/{len(groups):3d}")

        del data
        gc.collect()

    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]
        line_sel=np.zeros(len(centers),bool)
        line_sel[ids]=True

    moment=np.sum(cube[line_sel],axis=0)

    original_vertices=VERTICES.copy()
    flipped_vertices=VERTICES.copy()
    flipped_vertices[:,1]=127.0-flipped_vertices[:,1]

    original_mask=polygon_mask(original_vertices,moment.shape)
    flipped_mask=polygon_mask(flipped_vertices,moment.shape)

    original_sum=float(np.nansum(moment[original_mask]))
    flipped_sum=float(np.nansum(moment[flipped_mask]))

    raw_png=OUT/f"{VERSION}_MODEL_D_MOMENT0_RAW.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    ax.set_xlim(-0.5,127.5)
    ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — raw")
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    save(fig,raw_png)

    mask_png=OUT/f"{VERSION}_MODEL_D_MOMENT0_MASK_ORIENTATION_AUDIT.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    ax.contour(original_mask.astype(float),levels=[0.5],colors=["#ff3b30"],linewidths=2.5)
    ax.contour(flipped_mask.astype(float),levels=[0.5],colors=["#00e5ff"],linewidths=2.5)
    ax.plot([],[],color="#ff3b30",lw=2.5,label=f"Original mask | sum={original_sum:.6g}")
    ax.plot([],[],color="#00e5ff",lw=2.5,label=f"Vertically flipped mask | sum={flipped_sum:.6g}")
    ax.set_xlim(-0.5,127.5)
    ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title("Model D moment-0 — mask orientation audit")
    ax.legend(loc="upper right")
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    save(fig,mask_png)

    report=(
        f"<h3>{VERSION} — raw image and mask audit</h3>"
        f"<b>Line channels:</b> {int(line_sel.sum())}<br>"
        f"<b>Original mask pixels:</b> {int(original_mask.sum())}<br>"
        f"<b>Original moment sum:</b> {original_sum:.6g}<br>"
        f"<b>Flipped mask pixels:</b> {int(flipped_mask.sum())}<br>"
        f"<b>Flipped moment sum:</b> {flipped_sum:.6g}<br>"
        "Red = original mapping; cyan = vertically flipped mapping."
    )

    display(widgets.VBox([
        widgets.HTML(report),
        widgets.HTML("<h4>1. Raw Model D moment-0</h4>"),
        widgets.Image(value=raw_png.read_bytes(),format="png"),
        widgets.HTML("<h4>2. Mask orientation audit</h4>"),
        widgets.Image(value=mask_png.read_bytes(),format="png")
    ]))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"RAW PNG: {raw_png}")
    print(f"MASK PNG: {mask_png}")
    print(f"ORIGINAL MASK MOMENT SUM: {original_sum:.9g}")
    print(f"FLIPPED MASK MOMENT SUM: {flipped_sum:.9g}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__=="__main__":
    main()
