from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc, warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0235"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
OUT_PNG=Path("/content/JWST_OUTPUT/PNG")
DRIVE_PNG=ROOT/"PNG"
for p in (OUT_PNG,DRIVE_PNG): p.mkdir(parents=True,exist_ok=True)

# Tightened polygon traced from the latest user-marked Model D image.
TIGHT_VERTICES=np.array([
    [61.0,60.0],[61.8,61.2],[63.0,61.7],[65.0,61.3],
    [67.0,60.5],[68.5,60.1],[69.0,59.0],[68.2,58.0],
    [66.0,57.4],[63.5,57.5],[61.8,58.3],[61.0,59.2]
],dtype=float)

plt.rcParams.update({
    "figure.facecolor":"#05080d","axes.facecolor":"#05080d",
    "savefig.facecolor":"#05080d","text.color":"#e8f1ff",
    "axes.labelcolor":"#e8f1ff","axes.edgecolor":"#8aa0b8",
    "xtick.color":"#c7d4e5","ytick.color":"#c7d4e5"
})

def spec_axis(h):
    for ax in range(1,int(h["NAXIS"])+1):
        if "FREQ" in str(h.get(f"CTYPE{ax}","")).upper(): return ax
    raise RuntimeError("No frequency axis")

def axis_values(h,ax):
    n=int(h[f"NAXIS{ax}"]); p=np.arange(n)+1.0
    return (float(h[f"CRVAL{ax}"])+(p-float(h[f"CRPIX{ax}"]))*float(h.get(f"CDELT{ax}",h.get(f"CD{ax}_{ax}"))))/1e9

def collapse(a,np_spec):
    a=np.moveaxis(a,np_spec,0)
    while a.ndim>3: a=np.nanmean(a,axis=1)
    return a

def save(fig,path):
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    (DRIVE_PNG/path.name).write_bytes(path.read_bytes())

def main():
    warnings.filterwarnings("ignore")
    if not CUBE.exists(): raise FileNotFoundError(CUBE)

    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        tx0,ty0=WCS(h).celestial.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); half=64
        ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half)
        native=np.nanmedian(np.abs(np.diff(freq)))*1000
        k=max(1,int(round(5/native))); near=int(np.argmin(np.abs(freq-TARGET_GHZ)))
        lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups])
        cube=np.empty((len(groups),128,128),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f"5 MHz cube: {j+1:3d}/{len(groups):3d}")
        del data; gc.collect()

    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]
        line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    moment=np.sum(cube[line_sel],axis=0)

    raw_path=OUT_PNG/f"{VERSION}_MODEL_D_RAW_MOMENT0.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — raw")
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    save(fig,raw_path)

    mask_path=OUT_PNG/f"{VERSION}_MODEL_D_TIGHT_MASK_AUDIT.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    closed=np.vstack([TIGHT_VERTICES,TIGHT_VERTICES[0]])
    ax.plot(closed[:,0],closed[:,1],lw=2.2,label="Tightened user aperture")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — tightened mask audit")
    ax.legend(); fig.colorbar(im,ax=ax,label="Integrated brightness")
    save(fig,mask_path)

    tabs=widgets.Tab(children=[
        widgets.VBox([widgets.Image(value=raw_path.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=mask_path.read_bytes(),format="png")])
    ])
    tabs.set_title(0,"Raw Model D moment-0")
    tabs.set_title(1,"Tightened mask check")
    report=widgets.HTML(
        f"<h3>{VERSION} — tightened mask visual audit</h3>"
        f"<b>Mask bounds:</b> X {TIGHT_VERTICES[:,0].min():.1f}–{TIGHT_VERTICES[:,0].max():.1f}, "
        f"Y {TIGHT_VERTICES[:,1].min():.1f}–{TIGHT_VERTICES[:,1].max():.1f}<br>"
        f"<b>Line channels:</b> {int(line_sel.sum())}"
    )
    display(widgets.VBox([report,tabs]))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"RAW PNG: {raw_path}")
    print(f"MASK PNG: {mask_path}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__": main()
