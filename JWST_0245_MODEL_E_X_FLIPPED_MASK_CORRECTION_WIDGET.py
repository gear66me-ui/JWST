from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO
import gc, warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image, ImageDraw
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0245"
MODEL="MODEL_E_X_FLIPPED_MASK_CORRECTION_WIDGET"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
OUTPUT_PIXELS=1024

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
OUT_PNG=Path("/content/JWST_OUTPUT/PNG")
DRIVE_PNG=ROOT/"PNG"
for p in (OUT_PNG,DRIVE_PNG): p.mkdir(parents=True,exist_ok=True)

# Previous selective mask, corrected by mirroring X only in the 128x128 FITS crop.
BASE_VERTICES=np.array([
    [62.17,59.33],[62.17,58.17],[62.50,57.42],[63.58,56.42],
    [65.75,55.75],[66.17,56.08],[68.50,56.67],[69.08,57.42],
    [69.08,58.00],[68.17,59.00],[65.50,60.17],[63.00,60.17]
],dtype=float)
MASK_VERTICES=BASE_VERTICES.copy()
MASK_VERTICES[:,0]=127.0-MASK_VERTICES[:,0]

plt.rcParams.update({
    "figure.facecolor":"#05080d","axes.facecolor":"#05080d",
    "savefig.facecolor":"#05080d","text.color":"#e8f1ff",
    "axes.labelcolor":"#e8f1ff","axes.edgecolor":"#8aa0b8",
    "xtick.color":"#c7d4e5","ytick.color":"#c7d4e5",
    "grid.color":"#33485f"
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

def polygon_mask(vertices,shape):
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    pts=np.column_stack((xx.ravel(),yy.ravel()))
    return MplPath(vertices).contains_points(pts).reshape(shape).astype(np.float32)

def robust_sigma(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    med=np.median(x); mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)

def shifted_nonwrapping(template,dx,dy):
    out=np.zeros_like(template)
    y0=max(0,dy); y1=min(template.shape[0],template.shape[0]+dy)
    x0=max(0,dx); x1=min(template.shape[1],template.shape[1]+dx)
    sy0=max(0,-dy); sy1=sy0+(y1-y0)
    sx0=max(0,-dx); sx1=sx0+(x1-x0)
    if y1>y0 and x1>x0: out[y0:y1,x0:x1]=template[sy0:sy1,sx0:sx1]
    return out

def fig_bytes(fig,dpi=230):
    b=BytesIO(); fig.savefig(b,format="png",dpi=dpi,bbox_inches="tight"); plt.close(fig)
    return b.getvalue()

def rgb_master(moment,vertices=None):
    finite=moment[np.isfinite(moment)]
    vmin=float(np.nanmin(finite)); vmax=float(np.nanmax(finite))
    scaled=np.clip((moment-vmin)/(vmax-vmin),0,1)
    rgb=(plt.get_cmap("viridis")(np.nan_to_num(scaled))[...,:3]*255).astype(np.uint8)
    image=Image.fromarray(rgb,"RGB").resize((OUTPUT_PIXELS,OUTPUT_PIXELS),Image.Resampling.NEAREST)
    if vertices is not None:
        draw=ImageDraw.Draw(image)
        pts=[(float(x)*8.0,float(127.0-y)*8.0) for x,y in vertices]
        pts.append(pts[0]); draw.line(pts,fill=(255,35,20),width=5,joint="curve")
    return image

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
    mask=polygon_mask(MASK_VERTICES,moment.shape)
    nmask=int(mask.sum())
    if nmask==0: raise RuntimeError("Corrected X-flipped mask is empty")
    template=mask/np.sqrt(np.sum(mask**2))
    amps=np.einsum('ijk,jk->i',cube,template,optimize=True)
    target_flux=float(np.sum(amps[line_sel]))

    controls=[]
    for radius in (18,24,30,36,42):
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            dx=int(round(radius*np.cos(angle))); dy=int(round(radius*np.sin(angle)))
            shifted=shifted_nonwrapping(template,dx,dy)
            controls.append(float(np.nansum(moment*shifted)))
    controls=np.asarray(controls,float)
    sigma=robust_sigma(controls); snr=target_flux/sigma
    percentile=100*np.mean(controls<=target_flux)
    exceedances=int(np.sum(controls>=target_flux)); fap=exceedances/len(controls)

    pos=np.clip(amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    peak_i=int(np.argmax(amps[line_sel])); peak_freq=float(centers[line_sel][peak_i])

    raw=rgb_master(moment); masked=rgb_master(moment,MASK_VERTICES)
    raw_path=OUT_PNG/f"{VERSION}_MODEL_D_RAW.png"
    mask_path=OUT_PNG/f"{VERSION}_MODEL_D_X_FLIPPED_MASK.png"
    raw.save(raw_path,format="PNG",compress_level=0,optimize=False)
    masked.save(mask_path,format="PNG",compress_level=0,optimize=False)
    (DRIVE_PNG/raw_path.name).write_bytes(raw_path.read_bytes())
    (DRIVE_PNG/mask_path.name).write_bytes(mask_path.read_bytes())

    fig,ax=plt.subplots(figsize=(11,7))
    ax.plot(centers,amps,lw=1.8)
    ax.axvline(TARGET_GHZ,ls="--",label="Target 279.901 GHz")
    ax.axvline(centroid,ls=":",label=f"Centroid {centroid:.6f} GHz")
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.15,label="Line window")
    ax.set_xlabel("Observed frequency (GHz)"); ax.set_ylabel("Aperture amplitude")
    ax.set_title(f"X-flipped selective mask spectrum | S/N={snr:.3f}")
    ax.grid(alpha=.25); ax.legend(); spectrum_png=fig_bytes(fig)

    table=widgets.HTML(f"""
    <div style='background:#071019;color:#eaf2ff;border:1px solid #29445f;border-radius:10px;padding:14px;font-family:Arial'>
      <h3 style='margin-top:0'>JWST_0245 — X-axis mask correction</h3>
      <table style='border-collapse:collapse;width:100%;font-size:14px'>
      <tr><th align='left'>Quantity</th><th align='right'>Value</th><th align='left'>Unit</th></tr>
      <tr><td>Mask correction</td><td align='right'>x → 127 − x</td><td>X-axis mirror only</td></tr>
      <tr><td>Mask area</td><td align='right'>{nmask}</td><td>pixels</td></tr>
      <tr><td>Integrated line statistic</td><td align='right'>{target_flux:.7f}</td><td>aperture units</td></tr>
      <tr><td>Robust null σ</td><td align='right'>{sigma:.7f}</td><td>aperture units</td></tr>
      <tr><td>Integrated S/N</td><td align='right'><b>{snr:.4f}</b></td><td>σ</td></tr>
      <tr><td>Centroid</td><td align='right'><b>{centroid:.9f}</b></td><td>GHz</td></tr>
      <tr><td>Centroid offset</td><td align='right'>{(centroid-TARGET_GHZ)*1000:+.4f}</td><td>MHz</td></tr>
      <tr><td>Peak grouped channel</td><td align='right'>{peak_freq:.9f}</td><td>GHz</td></tr>
      <tr><td>Null percentile</td><td align='right'>{percentile:.3f}</td><td>%</td></tr>
      <tr><td>Null exceedances</td><td align='right'>{exceedances}/{len(controls)}</td><td>positions</td></tr>
      <tr><td>Empirical FAP</td><td align='right'>{fap:.6f}</td><td>fraction</td></tr>
      </table></div>""")

    raw_bytes=raw_path.read_bytes(); mask_bytes=mask_path.read_bytes()
    image=widgets.Image(value=raw_bytes,format="png")
    toggle=widgets.ToggleButtons(options=[("Raw Model D","raw"),("Corrected X-flipped mask","mask")],value="raw",description="Compare:")
    toggle.observe(lambda c: setattr(image,'value',raw_bytes if c['new']=='raw' else mask_bytes),names='value')

    display(widgets.HTML("<h3>JWST_0245 — corrected X-axis mask audit</h3>"))
    display(toggle); display(image)
    display(widgets.Image(value=spectrum_png,format="png"))
    display(table)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"MODEL: {MODEL}")
    print(f"MASK PIXELS: {nmask}")
    print(f"INTEGRATED S/N: {snr:.6f}")
    print(f"CENTROID GHz: {centroid:.9f}")
    print(f"PEAK GHz: {peak_freq:.9f}")
    print(f"NULL PERCENTILE: {percentile:.3f}%")
    print(f"NULL EXCEEDANCES: {exceedances}/{len(controls)}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__": main()
