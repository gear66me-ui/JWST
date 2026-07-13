#!/usr/bin/env python3
"""Upload, align, stretch, and combine four real MoM-z14/JWST channel images.
No AI imagery. Supports FITS/FITS.GZ, PNG, JPG, and TIFF.
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

VERSION="JWST_0079"
INPUT_DIR=Path("/content/MOM_V14_INPUT")
ROOT=Path("/content/JWST_OUTPUT"); PNG_DIR=ROOT/"PNG"; CSV_DIR=ROOT/"CSV"
INPUT_DIR.mkdir(parents=True,exist_ok=True); PNG_DIR.mkdir(parents=True,exist_ok=True); CSV_DIR.mkdir(parents=True,exist_ok=True)
COLORS=np.array([[0.05,0.35,1.0],[0.0,1.0,0.65],[1.0,0.72,0.05],[1.0,0.08,0.0]])
COLOR_NAMES=["blue","cyan-green","yellow-orange","red"]
EPS=1e-12


def supported(path: Path) -> bool:
    name=path.name.lower()
    return name.endswith((".fits",".fit",".fts",".fits.gz",".fit.gz",".fts.gz",
                          ".png",".jpg",".jpeg",".tif",".tiff"))


def upload_four() -> list[Path]:
    try:
        from google.colab import files
    except Exception as exc:
        raise RuntimeError("Run this script in Google Colab so the four-channel upload dialog can open.") from exc
    print("Select the FOUR MoM-z14 channel files together in the upload dialog.")
    uploaded=files.upload()
    names=[name for name in uploaded if supported(Path(name))]
    if len(names)!=4:
        raise RuntimeError(f"Exactly four supported channel files are required; received {len(names)}.")
    paths=[]
    for name in names:
        target=INPUT_DIR/Path(name).name
        target.write_bytes(uploaded[name])
        paths.append(target)
    return paths


def discover() -> list[Path]:
    saved=sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and supported(p)])
    if len(saved)==4:
        print("Using four channel files already stored in /content/MOM_V14_INPUT.")
        return saved
    if saved:
        for p in saved: p.unlink()
    return upload_four()


def parse_filter(path: Path):
    m=re.search(r"(?i)(F\d{3,4}[MWN]?)",path.name)
    if not m: return path.stem[:28],None
    label=m.group(1).upper(); wave=int(re.search(r"\d+",label).group())/100.0
    return label,wave


def load(path: Path):
    if any(path.name.lower().endswith(x) for x in (".fits",".fit",".fts",".fits.gz",".fit.gz",".fts.gz")):
        with fits.open(path,memmap=False) as hdul:
            for hdu in hdul:
                if hdu.data is None: continue
                a=np.asarray(hdu.data,dtype=float)
                while a.ndim>2: a=a[0]
                if a.ndim==2: return a,"FITS"
        raise ValueError(f"No 2-D image found in {path.name}")
    with Image.open(path) as im: a=np.asarray(im,dtype=float)
    if a.ndim==3: a=.2126*a[...,0]+.7152*a[...,1]+.0722*a[...,2]
    return a,"raster"


def fill(a):
    finite=np.isfinite(a)
    if not finite.any(): raise ValueError("A channel contains no finite pixels.")
    return np.where(finite,a,np.nanmedian(a[finite]))


def crop(a,shape):
    h,w=shape; y=max((a.shape[0]-h)//2,0); x=max((a.shape[1]-w)//2,0)
    return a[y:y+h,x:x+w]


def align(arrays):
    shape=(min(a.shape[0] for a in arrays),min(a.shape[1] for a in arrays))
    data=[crop(fill(a),shape) for a in arrays]
    ref=data[-1]; ref0=ref-np.median(ref); out=[]; shifts=[]
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
    rgb=np.zeros((*channels[0].shape,3),dtype=float)
    for c,v in zip(channels,COLORS): rgb+=c[...,None]*v
    rgb/=max(float(np.percentile(rgb[np.isfinite(rgb)],99.8)),EPS)
    return np.clip(rgb,0,1)**.92


def dashboard(records,channels,rgb):
    plt.style.use("dark_background")
    fig,ax=plt.subplots(2,3,figsize=(16,10),constrained_layout=True)
    for i in range(4):
        r,c=divmod(i,2)
        ax[r,c].imshow(channels[i],cmap="gray",origin="lower",vmin=0,vmax=1)
        w=records[i]["wave"]; wt=f"{w:.2f} µm" if w is not None else "wavelength not parsed"
        ax[r,c].set_title(f"{records[i]['label']} → {COLOR_NAMES[i]}\n{wt}")
        ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
    ax[0,2].imshow(rgb,origin="lower"); ax[0,2].set_title("Four-channel composite")
    lum=.2126*rgb[...,0]+.7152*rgb[...,1]+.0722*rgb[...,2]
    ax[1,2].imshow(lum,cmap="gray",origin="lower",vmin=0,vmax=1); ax[1,2].set_title("Composite luminance")
    for a in (ax[0,2],ax[1,2]): a.set_xticks([]); a.set_yticks([])
    fig.suptitle("MoM-z14 four-channel combination\nFFT phase registration; display colors are representational",fontsize=18)
    out=PNG_DIR/f"{VERSION}_MOM_V14_FOUR_CHANNEL_DASHBOARD.png"
    fig.savefig(out,dpi=360,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    return out


def main():
    records=[]
    for sequence,p in enumerate(discover()):
        label,wave=parse_filter(p); data,kind=load(p)
        records.append({"sequence":sequence,"path":p,"label":label,"wave":wave,"data":data,"kind":kind})
    if all(r["wave"] is not None for r in records): records.sort(key=lambda r:r["wave"])
    else: records.sort(key=lambda r:r["sequence"])
    aligned,shifts=align([r["data"] for r in records])
    channels=[]; stats=[]
    for a in aligned:
        s,st=stretch(a); channels.append(s); stats.append(st)
    rgb=combine(channels); dash=dashboard(records,channels,rgb)
    comp=PNG_DIR/f"{VERSION}_MOM_V14_COMPOSITE_ONLY.png"; plt.imsave(comp,rgb,origin="lower",dpi=360)
    rows=[]
    for i,(r,s,st) in enumerate(zip(records,shifts,stats)):
        rows.append({"order_short_to_long_or_upload":i+1,"file":str(r["path"]),"filter":r["label"],
          "wavelength_um":r["wave"],"display_color":COLOR_NAMES[i],"source_type":r["kind"],
          "shift_y_pix":s[0],"shift_x_pix":s[1],**st})
    manifest=CSV_DIR/f"{VERSION}_MOM_V14_CHANNEL_MANIFEST.csv"; pd.DataFrame(rows).to_csv(manifest,index=False)
    print(f"CODE OUTPUT: {VERSION}")
    print("INPUTS          4 uploaded real MoM-z14 channel images")
    print("ALIGNMENT       FFT phase correlation to channel 4")
    for i,r in enumerate(records):
        w=f"{r['wave']:.2f} um" if r['wave'] is not None else "upload order"
        print(f"CHANNEL {i+1}       {r['label']:<12} {w:<12} -> {COLOR_NAMES[i]}")
    print(f"DASHBOARD PNG   {dash}"); print(f"COMPOSITE PNG   {comp}"); print(f"MANIFEST CSV    {manifest}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")

if __name__=="__main__": main()
