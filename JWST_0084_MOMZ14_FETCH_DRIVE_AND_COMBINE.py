#!/usr/bin/env python3
from __future__ import annotations
import importlib.util, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path


def ensure():
    req={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","astropy":"astropy","scipy":"scipy","skimage":"scikit-image"}
    miss=[p for m,p in req.items() if importlib.util.find_spec(m) is None]
    if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
ensure()

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation
from google.colab import drive

VERSION="JWST_0084"
DRIVE=Path("/content/drive/MyDrive/Colab Notebooks/JWST/MoM-14")
OUT=Path("/content/JWST_OUTPUT"); FITS_OUT=OUT/"FITS"; PNG_OUT=OUT/"PNG"; CSV_OUT=OUT/"CSV"
for d in (FITS_OUT,PNG_OUT,CSV_OUT): d.mkdir(parents=True,exist_ok=True)
FILTERS=[("F115W",1.15,"MoM-14_F115W.fits"),("F150W",1.50,"MoM-14_F150W.fits"),("F277W",2.77,"MoM-14_F277W.fits"),("F444W",4.44,"MoM-14_F444W.fits")]
COLORS=np.array([[0.05,0.25,1.0],[0.0,0.95,0.75],[1.0,0.72,0.05],[1.0,0.05,0.0]])
NAMES=["blue","cyan-green","yellow-orange","red"]
EPS=1e-12


def load(path):
    with fits.open(path,memmap=False) as h:
        for u in h:
            if u.data is None: continue
            a=np.asarray(u.data,dtype=float).squeeze()
            while a.ndim>2: a=a[0]
            if a.ndim==2:
                f=np.isfinite(a)
                if not f.any(): raise ValueError(f"No finite pixels: {path.name}")
                return np.where(f,a,np.nanmedian(a[f]))
    raise ValueError(f"No 2-D image: {path.name}")


def crop(a,shape):
    h,w=shape; y=(a.shape[0]-h)//2; x=(a.shape[1]-w)//2
    return a[y:y+h,x:x+w]


def align(arrays):
    shape=(min(a.shape[0] for a in arrays),min(a.shape[1] for a in arrays))
    arr=[crop(a,shape) for a in arrays]; ref=arr[-1]-np.median(arr[-1]); out=[]; shifts=[]
    for i,a in enumerate(arr):
        if i==len(arr)-1: out.append(a); shifts.append((0.,0.)); continue
        s,_,_=phase_cross_correlation(ref,a-np.median(a),upsample_factor=20)
        out.append(ndi_shift(a,s,order=1,mode="constant",cval=float(np.median(a))))
        shifts.append((float(s[0]),float(s[1])))
    return out,shifts


def stretch(a):
    h,w=a.shape; b=max(3,int(.08*min(h,w))); m=np.zeros_like(a,bool)
    m[:b]=1; m[-b:]=1; m[:,:b]=1; m[:,-b:]=1
    bg=float(np.median(a[m])); x=a-bg; lo,hi=np.percentile(x[np.isfinite(x)],[1,99.7]); hi=max(float(hi),float(lo)+EPS)
    z=np.clip((x-lo)/(hi-lo),0,1); z=np.arcsinh(8*z)/np.arcsinh(8)
    return np.nan_to_num(z),bg,float(lo),float(hi)


def main():
    drive.mount("/content/drive",force_remount=False)
    missing=[DRIVE/f for _,_,f in FILTERS if not (DRIVE/f).exists()]
    if missing: raise FileNotFoundError("Missing Drive files: "+", ".join(str(p) for p in missing))

    paths=[]; arrays=[]
    for name,_,filename in FILTERS:
        src=DRIVE/filename; dst=FITS_OUT/f"{VERSION}_MOMZ14_{name}.fits"; shutil.copy2(src,dst)
        paths.append(dst); arrays.append(load(dst))

    aligned,shifts=align(arrays); channels=[]; stats=[]
    for a in aligned:
        s,bg,lo,hi=stretch(a); channels.append(s); stats.append((bg,lo,hi))

    rgb=np.zeros((*channels[0].shape,3))
    for c,v in zip(channels,COLORS): rgb+=c[...,None]*v
    rgb=np.clip(rgb/max(float(np.percentile(rgb,99.8)),EPS),0,1)**0.92

    plt.style.use("dark_background")
    fig,ax=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for i,((name,w,_),c) in enumerate(zip(FILTERS,channels)):
        r,k=divmod(i,2); ax[r,k].imshow(c,cmap="gray",origin="lower",vmin=0,vmax=1)
        ax[r,k].set_title(f"{name}  {w:.2f} µm → {NAMES[i]}"); ax[r,k].set_xticks([]); ax[r,k].set_yticks([])
    ax[0,2].imshow(rgb,origin="lower"); ax[0,2].set_title("MoM-z14 four-filter composite")
    lum=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
    ax[1,2].imshow(lum,cmap="gray",origin="lower",vmin=0,vmax=1); ax[1,2].set_title("Composite luminance")
    for a in (ax[0,2],ax[1,2]): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MoM-z14 — FITS loaded from Google Drive and registered by FFT phase correlation",fontsize=17)
    dash=PNG_OUT/f"{VERSION}_MOMZ14_DRIVE_COMPOSITE_DASHBOARD.png"; fig.savefig(dash,dpi=360,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    comp=PNG_OUT/f"{VERSION}_MOMZ14_DRIVE_COMPOSITE_ONLY.png"; plt.imsave(comp,rgb,origin="lower",dpi=360)

    rows=[]
    for i,((name,w,_),p,s,st) in enumerate(zip(FILTERS,paths,shifts,stats)):
        rows.append({"filter":name,"wavelength_um":w,"display_color":NAMES[i],"runtime_fits":str(p),"shift_y_pix":s[0],"shift_x_pix":s[1],"background":st[0],"stretch_low":st[1],"stretch_high":st[2]})
    csv=CSV_OUT/f"{VERSION}_MOMZ14_DRIVE_COMPOSITE_MANIFEST.csv"; pd.DataFrame(rows).to_csv(csv,index=False)
    shutil.copy2(dash,DRIVE/dash.name); shutil.copy2(comp,DRIVE/comp.name); shutil.copy2(csv,DRIVE/csv.name)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"DRIVE SOURCE    {DRIVE}")
    print("FILTERS         F115W  F150W  F277W  F444W")
    print("ALIGNMENT       FFT phase correlation to F444W")
    print(f"DASHBOARD PNG   {dash}")
    print(f"COMPOSITE PNG   {comp}")
    print(f"MANIFEST CSV    {csv}")
    print(f"Drive copies    {DRIVE}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")

if __name__=="__main__": main()
