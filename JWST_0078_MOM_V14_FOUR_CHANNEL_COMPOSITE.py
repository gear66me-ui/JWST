#!/usr/bin/env python3
"""Combine four real MoM-z14/JWST channel images into a false-color composite.
No AI imagery. Files may be FITS, PNG, JPG, or TIFF.
"""
from __future__ import annotations
import importlib.util, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages():
    req={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib",
         "PIL":"pillow","astropy":"astropy","scipy":"scipy","skimage":"scikit-image"}
    miss=[p for m,p in req.items() if importlib.util.find_spec(m) is None]
    if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
ensure_packages()

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from astropy.io import fits
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation

VERSION="JWST_0078"
INPUT_DIR=Path("/content/MOM_V14_INPUT")
ROOT=Path("/content/JWST_OUTPUT"); PNG_DIR=ROOT/"PNG"; CSV_DIR=ROOT/"CSV"
PNG_DIR.mkdir(parents=True,exist_ok=True); CSV_DIR.mkdir(parents=True,exist_ok=True)
CHANNEL_FILES=[]  # Optional exact file names; otherwise auto-discover four files.
SUPPORTED={".fits",".fit",".fts",".png",".jpg",".jpeg",".tif",".tiff"}
COLORS=np.array([[0.05,0.35,1.0],[0.0,1.0,0.65],[1.0,0.72,0.05],[1.0,0.08,0.0]])
COLOR_NAMES=["blue","cyan-green","yellow-orange","red"]
EPS=1e-12


def discover():
    if CHANNEL_FILES:
        paths=[Path(x) if Path(x).is_absolute() else INPUT_DIR/x for x in CHANNEL_FILES]
    else:
        root=INPUT_DIR if INPUT_DIR.exists() else Path("/content")
        cand=[p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED]
        tagged=[p for p in cand if any(t in p.name.lower() for t in
                ("mom","z14","v14","f090","f115","f150","f200","f277","f356","f410","f444"))]
        paths=sorted(tagged if len(tagged)>=4 else cand,key=lambda p:p.stat().st_mtime,reverse=True)[:4]
    if len(paths)!=4 or any(not p.exists() for p in paths):
        INPUT_DIR.mkdir(parents=True,exist_ok=True)
        raise RuntimeError(f"Place exactly four channel images in {INPUT_DIR}; found {len(paths)}.")
    return paths


def parse_filter(path):
    m=re.search(r"(?i)(F\d{3,4}[MWN]?)",path.name)
    if not m: return path.stem[:24],None
    label=m.group(1).upper(); wave=int(re.search(r"\d+",label).group())/100.0
    return label,wave


def load(path):
    if path.suffix.lower() in {".fits",".fit",".fts"}:
        with fits.open(path,memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is None: continue
                a=np.asarray(hdu.data,dtype=float)
                while a.ndim>2: a=a[0]
                if a.ndim==2: return a,"FITS"
        raise ValueError(f"No 2-D image in {path.name}")
    with Image.open(path) as im: a=np.asarray(im,dtype=float)
    if a.ndim==3: a=.2126*a[...,0]+.7152*a[...,1]+.0722*a[...,2]
    return a,"raster"


def fill(a):
    finite=np.isfinite(a)
    if not finite.any(): raise ValueError("Channel has no finite pixels")
    return np.where(finite,a,np.nanmedian(a[finite]))


def crop(a,shape):
    h,w=shape; y=max((a.shape[0]-h)//2,0); x=max((a.shape[1]-w)//2,0)
    return a[y:y+h,x:x+w]


def align(arrays):
    shape=(min(a.shape[0] for a in arrays),min(a.shape[1] for a in arrays))
    data=[crop(fill(a),shape) for a in arrays]; ref=data[-1]; ref0=ref-np.median(ref)
    out=[]; shifts=[]
    for i,a in enumerate(data):
        if i==len(data)-1: out.append(a); shifts.append((0.,0.)); continue
        s,_,_=phase_cross_correlation(ref0,a-np.median(a),upsample_factor=20)
        out.append(ndi_shift(a,s,order=1,mode="constant",cval=np.nan))
        shifts.append((float(s[0]),float(s[1])))
    return out,shifts


def stretch(a):
    h,w=a.shape; b=max(3,int(.08*min(h,w)))
    mask=np.zeros_like(a,dtype=bool); mask[:b]=1; mask[-b:]=1; mask[:,:b]=1; mask[:,-b:]=1
    bg=float(np.nanmedian(a[mask])); x=a-bg; finite=x[np.isfinite(x)]
    lo,hi=np.percentile(finite,[1,99.7]); hi=max(float(hi),float(lo)+EPS)
    z=np.clip((x-lo)/(hi-lo),0,1); z=np.arcsinh(8*z)/np.arcsinh(8)
    return np.nan_to_num(z),{"background":bg,"low":float(lo),"high":float(hi)}


def combine(channels):
    rgb=np.zeros((*channels[0].shape,3))
    for c,v in zip(channels,COLORS): rgb+=c[...,None]*v
    rgb/=max(float(np.percentile(rgb[np.isfinite(rgb)],99.8)),EPS)
    return np.clip(rgb,0,1)**.92


def dashboard(records,channels,rgb):
    plt.style.use("dark_background")
    fig,ax=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for i in range(4):
        r,c=divmod(i,2); ax[r,c].imshow(channels[i],cmap="gray",origin="lower",vmin=0,vmax=1)
        w=records[i]["wave"]; wt=f"{w:.2f} µm" if w is not None else "wavelength not parsed"
        ax[r,c].set_title(f"{records[i]['label']} → {COLOR_NAMES[i]}\n{wt}")
        ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
    ax[0,2].imshow(rgb,origin="lower"); ax[0,2].set_title("Four-channel composite")
    lum=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
    ax[1,2].imshow(lum,cmap="gray",origin="lower",vmin=0,vmax=1); ax[1,2].set_title("Composite luminance")
    for a in (ax[0,2],ax[1,2]): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MoM-z14 four-channel combination\nFFT phase registration; colors are representational",fontsize=18)
    out=PNG_DIR/f"{VERSION}_MOM_V14_FOUR_CHANNEL_DASHBOARD.png"
    fig.savefig(out,dpi=360,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    return out


def main():
    records=[]
    for p in discover():
        label,wave=parse_filter(p); data,kind=load(p)
        records.append({"path":p,"label":label,"wave":wave,"data":data,"kind":kind})
    records.sort(key=lambda r:(r["wave"] is None,r["wave"] if r["wave"] is not None else r["path"].name))
    aligned,shifts=align([r["data"] for r in records])
    channels=[]; stats=[]
    for a in aligned:
        s,st=stretch(a); channels.append(s); stats.append(st)
    rgb=combine(channels); dash=dashboard(records,channels,rgb)
    comp=PNG_DIR/f"{VERSION}_MOM_V14_COMPOSITE_ONLY.png"; plt.imsave(comp,rgb,origin="lower",dpi=360)
    rows=[]
    for i,(r,s,st) in enumerate(zip(records,shifts,stats)):
        rows.append({"order_short_to_long":i+1,"file":str(r["path"]),"filter":r["label"],
          "wavelength_um":r["wave"],"display_color":COLOR_NAMES[i],"source_type":r["kind"],
          "shift_y_pix":s[0],"shift_x_pix":s[1],**st})
    manifest=CSV_DIR/f"{VERSION}_MOM_V14_CHANNEL_MANIFEST.csv"; pd.DataFrame(rows).to_csv(manifest,index=False)
    print(f"CODE OUTPUT: {VERSION}")
    print("INPUTS          4 real MoM-z14 channel images")
    print("ALIGNMENT       FFT phase correlation to longest-wavelength channel")
    for i,r in enumerate(records):
        w=f"{r['wave']:.2f} um" if r['wave'] is not None else "not parsed"
        print(f"CHANNEL {i+1}       {r['label']:<10} {w:<12} -> {COLOR_NAMES[i]}")
    print(f"DASHBOARD PNG   {dash}"); print(f"COMPOSITE PNG   {comp}"); print(f"MANIFEST CSV    {manifest}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")

if __name__=="__main__": main()
