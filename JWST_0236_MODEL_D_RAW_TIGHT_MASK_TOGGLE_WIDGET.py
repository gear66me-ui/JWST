from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0236"
MODEL="MODEL_D_RAW_TIGHT_MASK_TOGGLE_WIDGET"

TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
OUT_PNG=Path("/content/JWST_OUTPUT/PNG")
OUT_CSV=Path("/content/JWST_OUTPUT/CSV")
DRIVE_PNG=ROOT/"PNG"
DRIVE_CSV=ROOT/"CSV"

for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV):
    p.mkdir(parents=True,exist_ok=True)

TIGHT_VERTICES=np.array([
    [61.0,59.6],
    [61.8,61.0],
    [63.2,61.6],
    [65.0,61.2],
    [67.8,60.2],
    [68.8,59.1],
    [68.6,57.7],
    [66.8,56.9],
    [63.5,56.8],
    [61.7,57.8]
],dtype=float)

plt.rcParams.update({
    "figure.facecolor":"#05080d",
    "axes.facecolor":"#05080d",
    "savefig.facecolor":"#05080d",
    "text.color":"#e8f1ff",
    "axes.labelcolor":"#e8f1ff",
    "axes.edgecolor":"#8aa0b8",
    "xtick.color":"#c7d4e5",
    "ytick.color":"#c7d4e5"
})


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


def save_plot(moment,path,title,vertices=None):
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    if vertices is not None:
        closed=np.vstack([vertices,vertices[0]])
        ax.plot(closed[:,0],closed[:,1],color="#ff3b30",lw=2.0)
    ax.set_xlim(-0.5,127.5)
    ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title(title)
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    (DRIVE_PNG/path.name).write_bytes(path.read_bytes())


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

    raw_png=OUT_PNG/f"{VERSION}_MODEL_D_RAW_MOMENT0.png"
    mask_png=OUT_PNG/f"{VERSION}_MODEL_D_TIGHT_MASK_OVERLAY.png"

    save_plot(moment,raw_png,"Model D line-window moment-0 — raw")
    save_plot(moment,mask_png,"Model D line-window moment-0 — tight mask audit",TIGHT_VERTICES)

    vcsv=OUT_CSV/f"{VERSION}_TIGHT_MASK_VERTICES.csv"
    pd.DataFrame(TIGHT_VERTICES,columns=["x_pixel","y_pixel"]).to_csv(vcsv,index=False)
    (DRIVE_CSV/vcsv.name).write_bytes(vcsv.read_bytes())

    raw_bytes=raw_png.read_bytes()
    mask_bytes=mask_png.read_bytes()

    toggle=widgets.ToggleButtons(
        options=[("Raw image","raw"),("Tight mask","mask")],
        value="raw",
        description="View:",
        button_style="info"
    )
    image=widgets.Image(
        value=raw_bytes,
        format="png",
        layout=widgets.Layout(width="100%",max_width="1100px")
    )
    status=widgets.HTML(
        f"<b>Showing:</b> raw FITS-derived Model D moment-0 &nbsp; | &nbsp; "
        f"Raw PNG: {len(raw_bytes)/1_000_000:.3f} MB &nbsp; | &nbsp; "
        f"Masked PNG: {len(mask_bytes)/1_000_000:.3f} MB"
    )

    def change_view(change):
        if change.get("name")!="value":
            return
        if change["new"]=="raw":
            image.value=raw_bytes
            status.value=(
                f"<b>Showing:</b> raw FITS-derived Model D moment-0 &nbsp; | &nbsp; "
                f"Raw PNG: {len(raw_bytes)/1_000_000:.3f} MB"
            )
        else:
            image.value=mask_bytes
            status.value=(
                f"<b>Showing:</b> tight-mask overlay on the same FITS-derived moment-0 &nbsp; | &nbsp; "
                f"Masked PNG: {len(mask_bytes)/1_000_000:.3f} MB"
            )

    toggle.observe(change_view,names="value")

    display(widgets.VBox([
        widgets.HTML("<h3>JWST_0236 — Raw / Tight Mask visual comparison</h3>"),
        toggle,
        status,
        image
    ]))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"LINE CHANNELS: {int(line_sel.sum())}")
    print(f"RAW PNG: {raw_png} | {raw_png.stat().st_size} bytes")
    print(f"MASK PNG: {mask_png} | {mask_png.stat().st_size} bytes")
    print(f"CSV OUTPUT: {vcsv}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__=="__main__":
    main()
