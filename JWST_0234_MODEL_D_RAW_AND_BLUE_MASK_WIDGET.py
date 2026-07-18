from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0234"
MODEL="MODEL_D_RAW_AND_BLUE_MASK_WIDGET"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
OUT_PNG=Path("/content/JWST_OUTPUT/PNG")
DRIVE_PNG=ROOT/"PNG"
for p in (OUT_PNG,DRIVE_PNG):
    p.mkdir(parents=True,exist_ok=True)

ORIGINAL=np.array([
    [71.2,70.1],[68.7,72.5],[66.1,72.7],[61.3,69.9],
    [60.0,68.3],[59.8,66.5],[63.3,65.7],[69.5,67.6]
],dtype=float)
BLUE=ORIGINAL.copy()
BLUE[:,1]=127.0-BLUE[:,1]

plt.rcParams.update({
    "figure.facecolor":"#05080d","axes.facecolor":"#05080d",
    "savefig.facecolor":"#05080d","text.color":"#e8f1ff",
    "axes.labelcolor":"#e8f1ff","axes.edgecolor":"#8aa0b8",
    "xtick.color":"#c7d4e5","ytick.color":"#c7d4e5"
})

def spec_axis(h):
    for ax in range(1,int(h["NAXIS"])+1):
        if "FREQ" in str(h.get(f"CTYPE{ax}","")).upper():
            return ax
    raise RuntimeError("No frequency axis")

def axis_values(h,ax):
    n=int(h[f"NAXIS{ax}"])
    p=np.arange(n)+1.0
    return (float(h[f"CRVAL{ax}"])+(p-float(h[f"CRPIX{ax}"]))*float(h.get(f"CDELT{ax}",h.get(f"CD{ax}_{ax}"))))/1e9

def collapse(a,np_spec):
    a=np.moveaxis(a,np_spec,0)
    while a.ndim>3:
        a=np.nanmean(a,axis=1)
    return a

def save(fig,path):
    fig.savefig(path,dpi=600,bbox_inches="tight",pad_inches=0.05)
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
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0))
        half=64
        ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half)
        native=np.nanmedian(np.abs(np.diff(freq)))*1000
        k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ)))
        lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups])
        cube=np.empty((len(groups),128,128),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim
            sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1:
                print(f"5 MHz cube: {j+1:3d}/{len(groups):3d}")
        del data
        gc.collect()

    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]
        line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    moment=np.sum(cube[line_sel],axis=0)

    raw_png=OUT_PNG/f"{VERSION}_MODEL_D_RAW_MOMENT0.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — raw")
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    save(fig,raw_png)

    mask_png=OUT_PNG/f"{VERSION}_MODEL_D_BLUE_MASK_LOCATION_CHECK.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    closed=np.vstack([BLUE,BLUE[0]])
    ax.plot(closed[:,0],closed[:,1],color="#26d9ff",lw=3,label="Corrected blue mask")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D moment-0 — corrected blue mask location")
    ax.legend(loc="upper right")
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    save(fig,mask_png)

    report=widgets.HTML(
        f"<h3>{VERSION} — location check only</h3>"
        f"<b>Raw PNG:</b> {raw_png.name} ({raw_png.stat().st_size/1_000_000:.3f} MB)<br>"
        f"<b>Mask PNG:</b> {mask_png.name} ({mask_png.stat().st_size/1_000_000:.3f} MB)<br>"
        f"<b>Blue bounds:</b> X {BLUE[:,0].min():.1f}–{BLUE[:,0].max():.1f}, "
        f"Y {BLUE[:,1].min():.1f}–{BLUE[:,1].max():.1f}<br>"
        f"<b>Extraction:</b> not run"
    )
    tabs=widgets.Tab(children=[
        widgets.VBox([widgets.Image(value=raw_png.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=mask_png.read_bytes(),format="png")])
    ])
    tabs.set_title(0,"Raw Model D moment-0")
    tabs.set_title(1,"Corrected blue mask")
    display(widgets.VBox([report,tabs]))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"RAW PNG: {raw_png}")
    print(f"MASK PNG: {mask_png}")
    print(f"BLUE MASK X RANGE: {BLUE[:,0].min():.3f} to {BLUE[:,0].max():.3f}")
    print(f"BLUE MASK Y RANGE: {BLUE[:,1].min():.3f} to {BLUE[:,1].max():.3f}")
    print("EXTRACTION: NOT RUN")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__":
    main()
