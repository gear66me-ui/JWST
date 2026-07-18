from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0239"
MODEL="MODEL_E_TIGHT_MASK_WIDGET_3"
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
for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)

# Tight Mask Widget 3: traced from the latest user-marked image.
MASK_VERTICES=np.array([
    [68.17,59.00],[65.17,59.83],[63.83,59.83],[62.75,59.25],
    [62.33,58.67],[62.42,58.17],[64.00,56.33],[65.08,55.83],
    [67.83,55.75],[68.83,56.08],[69.08,56.75],[69.08,57.58],
    [68.92,58.33],[68.58,58.75]
],dtype=float)

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

def save(fig,path):
    fig.savefig(path,dpi=300,bbox_inches="tight")
    plt.close(fig)
    (DRIVE_PNG/path.name).write_bytes(path.read_bytes())

def shifted_nonwrapping(template,dx,dy):
    out=np.zeros_like(template)
    y0=max(0,dy); y1=min(template.shape[0],template.shape[0]+dy)
    x0=max(0,dx); x1=min(template.shape[1],template.shape[1]+dx)
    sy0=max(0,-dy); sy1=sy0+(y1-y0)
    sx0=max(0,-dx); sx1=sx0+(x1-x0)
    if y1>y0 and x1>x0: out[y0:y1,x0:x1]=template[sy0:sy1,sx0:sx1]
    return out

def main():
    warnings.filterwarnings("ignore")
    if not CUBE.exists(): raise FileNotFoundError(CUBE)

    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        tx0,ty0=WCS(h).celestial.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); half=64
        ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half)
        pix=abs(float(h["CDELT2"]))*3600
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
    if nmask==0: raise RuntimeError("Tight Mask Widget 3 produced an empty mask")
    template=mask/np.sqrt(np.sum(mask**2))
    amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=float(np.sum(amps[line_sel]))

    controls=[]
    for radius in (18,24,30,36,42):
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            dx=int(round(radius*np.cos(angle))); dy=int(round(radius*np.sin(angle)))
            shifted=shifted_nonwrapping(template,dx,dy)
            ca=np.array([np.nansum(im*shifted) for im in cube])
            controls.append({"radius_pix":radius,"dx_pix":dx,"dy_pix":dy,
                             "offset_arcsec":np.hypot(dx,dy)*pix,
                             "line_flux":float(np.sum(ca[line_sel]))})
    cdf=pd.DataFrame(controls)
    sigma=robust_sigma(cdf["line_flux"].values)
    snr=target_flux/sigma
    cdf["line_SNR"]=cdf["line_flux"]/sigma
    percentile=100*np.mean(cdf["line_flux"].values<=target_flux)
    fap=np.mean(cdf["line_flux"].values>=target_flux)
    pos=np.clip(amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    peak_idx=np.argmax(amps[line_sel]); peak_freq=centers[line_sel][peak_idx]

    raw_path=OUT_PNG/f"{VERSION}_RAW_MODEL_D_MOMENT0.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — raw")
    fig.colorbar(im,ax=ax,label="Integrated brightness"); save(fig,raw_path)

    mask_path=OUT_PNG/f"{VERSION}_TIGHT_MASK_WIDGET_3_OVERLAY.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    closed=np.vstack([MASK_VERTICES,MASK_VERTICES[0]])
    ax.plot(closed[:,0],closed[:,1],lw=2.2,label=f"Tight Mask 3: {nmask} pixels")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D moment-0 — Tight Mask Widget 3")
    ax.legend(); fig.colorbar(im,ax=ax,label="Integrated brightness"); save(fig,mask_path)

    spec_path=OUT_PNG/f"{VERSION}_TIGHT_MASK_WIDGET_3_SPECTRUM.png"
    fig,ax=plt.subplots(figsize=(11,7))
    ax.plot(centers,amps,lw=1.8)
    ax.axvline(TARGET_GHZ,ls="--",label="Target 279.901 GHz")
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.15,label="Line window")
    ax.set_xlabel("Observed frequency (GHz)"); ax.set_ylabel("Aperture amplitude")
    ax.set_title(f"Tight Mask Widget 3 spectrum | S/N={snr:.3f} | centroid={centroid:.6f} GHz")
    ax.grid(alpha=.25); ax.legend(); save(fig,spec_path)

    null_path=OUT_PNG/f"{VERSION}_TIGHT_MASK_WIDGET_3_NULLS.png"
    fig,ax=plt.subplots(figsize=(10,7))
    ax.hist(cdf["line_SNR"],bins=24,alpha=.8,label=f"{len(cdf)} null positions")
    ax.axvline(snr,ls="--",lw=2,label=f"Target S/N={snr:.3f}")
    ax.set_xlabel("Integrated aperture S/N"); ax.set_ylabel("Count")
    ax.set_title(f"Null test | percentile={percentile:.1f}% | FAP={fap:.3f}")
    ax.grid(alpha=.25); ax.legend(); save(fig,null_path)

    summary=pd.DataFrame([{
        "model":"Model E Tight Mask Widget 3","target_frequency_GHz":TARGET_GHZ,
        "line_channels":int(line_sel.sum()),"mask_pixels":nmask,
        "mask_x_min":MASK_VERTICES[:,0].min(),"mask_x_max":MASK_VERTICES[:,0].max(),
        "mask_y_min":MASK_VERTICES[:,1].min(),"mask_y_max":MASK_VERTICES[:,1].max(),
        "target_line_flux":target_flux,"null_sigma":sigma,"target_integrated_SNR":snr,
        "target_percentile_vs_null":percentile,"false_alarm_fraction":fap,
        "centroid_GHz":centroid,"centroid_offset_MHz":(centroid-TARGET_GHZ)*1000,
        "peak_frequency_GHz":peak_freq,"peak_offset_MHz":(peak_freq-TARGET_GHZ)*1000,
        "control_positions":len(cdf)
    }])
    scsv=OUT_CSV/f"{VERSION}_SUMMARY.csv"; ncsv=OUT_CSV/f"{VERSION}_NULLS.csv"
    pcsv=OUT_CSV/f"{VERSION}_SPECTRUM.csv"; vcsv=OUT_CSV/f"{VERSION}_VERTICES.csv"
    summary.to_csv(scsv,index=False); cdf.to_csv(ncsv,index=False)
    pd.DataFrame({"frequency_GHz":centers,"velocity_kms":vel,"amplitude":amps,"line_channel":line_sel}).to_csv(pcsv,index=False)
    pd.DataFrame(MASK_VERTICES,columns=["x_pixel","y_pixel"]).to_csv(vcsv,index=False)
    for p in (scsv,ncsv,pcsv,vcsv): (DRIVE_CSV/p.name).write_bytes(p.read_bytes())

    image=widgets.Image(value=raw_path.read_bytes(),format="png")
    toggle=widgets.ToggleButtons(options=[("Raw image","raw"),("Tight Mask 3","mask")],value="raw",description="View:")
    def switch(change):
        image.value=(raw_path if change["new"]=="raw" else mask_path).read_bytes()
    toggle.observe(switch,names="value")
    report=widgets.HTML(
        f"<h3>{VERSION} — Tight Mask Widget 3</h3>"
        f"<b>Mask pixels:</b> {nmask}<br>"
        f"<b>Integrated S/N:</b> {snr:.3f}<br>"
        f"<b>Null percentile:</b> {percentile:.1f}%<br>"
        f"<b>False-alarm fraction:</b> {fap:.3f}<br>"
        f"<b>Centroid:</b> {centroid:.6f} GHz"
    )
    tabs=widgets.Tab(children=[
        widgets.VBox([toggle,image]),
        widgets.VBox([widgets.Image(value=spec_path.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=null_path.read_bytes(),format="png")])
    ])
    tabs.set_title(0,"Raw ↔ Mask 3")
    tabs.set_title(1,"Spectrum")
    tabs.set_title(2,"Null controls")
    display(widgets.VBox([report,tabs]))

    print(f"CODE OUTPUT: {VERSION}")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    for p in (raw_path,mask_path,spec_path,null_path): print(f"PNG OUTPUT: {p}")
    for p in (scsv,ncsv,pcsv,vcsv): print(f"CSV OUTPUT: {p}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__": main()
