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
from PIL import Image

VERSION="JWST_0230"
MODEL="MODEL_D_MOMENT0_UNCOMPRESSED_WIDGET"
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

    moment=np.sum(cube[line_sel],axis=0)

    finite=moment[np.isfinite(moment)]
    vmin=float(np.nanmin(finite))
    vmax=float(np.nanmax(finite))
    norm=np.clip((moment-vmin)/(vmax-vmin),0,1)
    rgba=plt.get_cmap("viridis")(norm,bytes=True)
    rgb=rgba[:,:,:3]

    image=Image.fromarray(rgb,"RGB")
    image=image.resize((1024,1024),resample=Image.Resampling.NEAREST)

    p_png=OUT_PNG/f"{VERSION}_MODEL_D_MOMENT0_UNTOUCHED_1024PX_UNCOMPRESSED.png"
    image.save(p_png,format="PNG",compress_level=0,optimize=False,dpi=(600,600))
    (DRIVE_PNG/p_png.name).write_bytes(p_png.read_bytes())

    size_bytes=p_png.stat().st_size
    size_mb=size_bytes/1_000_000
    size_mib=size_bytes/1_048_576
    status="PASS" if size_bytes>=2_500_000 else "FAIL"

    report=widgets.HTML(
        f"<h3>Model D moment-0 baseline — untouched</h3>"
        f"<b>Dimensions:</b> 1024 × 1024 px<br>"
        f"<b>File size:</b> {size_bytes:,} bytes | {size_mb:.3f} MB | {size_mib:.3f} MiB<br>"
        f"<b>PNG compression:</b> disabled<br>"
        f"<b>2.5 MB minimum:</b> {status}<br>"
        f"<b>Source grid:</b> 128 × 128 pixels<br>"
        f"<b>Line channels:</b> {int(line_sel.sum())}"
    )

    display(widgets.VBox([
        report,
        widgets.Image(value=p_png.read_bytes(),format="png")
    ]))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"PIXEL DIMENSIONS : 1024 x 1024")
    print(f"PNG BYTES        : {size_bytes:,}")
    print(f"PNG MB           : {size_mb:.3f}")
    print(f"PNG MiB          : {size_mib:.3f}")
    print(f"2.5 MB CHECK     : {status}")
    print(f"PNG OUTPUT       : {p_png}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__=="__main__":
    main()
