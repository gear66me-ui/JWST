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

VERSION="JWST_0227"
MODEL="MODEL_D_MOMENT0_UNMARKED"
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
        ny=nx=128

        native=np.nanmedian(np.abs(np.diff(freq)))*1000
        k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ)))
        lo=max(0,near-14*k)
        hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups])
        cube=np.empty((len(groups),ny,nx),np.float32)

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

    moment0=np.sum(cube[line_sel],axis=0)

    png=OUT_PNG/f"{VERSION}_MODEL_D_MOMENT0_UNMARKED.png"
    csv=OUT_CSV/f"{VERSION}_MODEL_D_MOMENT0_PIXELS.csv"

    plt.imsave(png,moment0,origin="lower",cmap="viridis")
    np.savetxt(csv,moment0,delimiter=",",fmt="%.9e")

    (DRIVE_PNG/png.name).write_bytes(png.read_bytes())
    (DRIVE_CSV/csv.name).write_bytes(csv.read_bytes())

    image=widgets.Image(
        value=png.read_bytes(),
        format="png",
        layout=widgets.Layout(width="100%",max_width="1100px")
    )
    display(image)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"MODEL: {MODEL}")
    print(f"Grouped channels in line window: {int(line_sel.sum())}")
    print(f"Velocity window km/s: {vel[line_sel].min():.6f} to {vel[line_sel].max():.6f}")
    print(f"PNG: {png}")
    print(f"CSV: {csv}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__=="__main__":
    main()
