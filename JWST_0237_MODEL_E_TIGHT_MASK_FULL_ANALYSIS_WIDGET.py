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

VERSION="JWST_0237"
MODEL="MODEL_E_TIGHT_MASK_FULL_ANALYSIS"
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

TIGHT_VERTICES=np.array([
    [61.0,60.0],[61.8,61.2],[63.0,61.7],[65.0,61.3],
    [67.0,60.5],[68.5,60.1],[69.0,59.0],[68.2,58.0],
    [66.0,57.4],[63.5,57.5],[61.8,58.3],[61.0,59.2]
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
    return MplPath(vertices).contains_points(np.column_stack((xx.ravel(),yy.ravel()))).reshape(shape).astype(np.float32)

def robust_sigma(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    med=np.median(x); mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x,ddof=1)

def shifted_no_wrap(a,dy,dx):
    out=np.zeros_like(a)
    y0=max(0,dy); y1=min(a.shape[0],a.shape[0]+dy)
    x0=max(0,dx); x1=min(a.shape[1],a.shape[1]+dx)
    sy0=max(0,-dy); sy1=sy0+(y1-y0)
    sx0=max(0,-dx); sx1=sx0+(x1-x0)
    if y1>y0 and x1>x0: out[y0:y1,x0:x1]=a[sy0:sy1,sx0:sx1]
    return out

def savefig(fig,path):
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
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); pix=abs(float(h["CDELT2"]))*3600
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half)
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
    mask=polygon_mask(TIGHT_VERTICES,moment.shape)
    mask_pixels=int(mask.sum())
    if mask_pixels==0: raise RuntimeError("Tight mask is empty")
    template=mask/np.sqrt(np.sum(mask**2))
    amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=float(np.sum(amps[line_sel]))

    controls=[]
    for radius in (18,24,30,36,42):
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            dx=int(round(radius*np.cos(angle))); dy=int(round(radius*np.sin(angle)))
            shifted=shifted_no_wrap(template,dy,dx)
            if np.sum(shifted>0)<mask_pixels: continue
            ca=np.array([np.nansum(im*shifted) for im in cube])
            controls.append({"radius_pix":radius,"dx_pix":dx,"dy_pix":dy,
                             "offset_arcsec":np.hypot(dx,dy)*pix,
                             "line_flux":float(np.sum(ca[line_sel]))})
    cdf=pd.DataFrame(controls)
    sigma=robust_sigma(cdf["line_flux"].values)
    target_snr=target_flux/sigma if sigma>0 else np.nan
    cdf["line_SNR"]=cdf["line_flux"]/sigma
    percentile=100*np.mean(cdf["line_flux"].values<=target_flux)
    fap=np.mean(cdf["line_flux"].values>=target_flux)

    positive=np.clip(amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*positive)/np.sum(positive) if np.sum(positive)>0 else np.nan
    peak_i=int(np.nanargmax(amps))
    peak_freq=float(centers[peak_i])
    peak_amp=float(amps[peak_i])

    summary=pd.DataFrame([{
        "model":"Model E tight user aperture",
        "target_frequency_GHz":TARGET_GHZ,
        "line_channels":int(line_sel.sum()),
        "velocity_min_kms":float(np.min(vel[line_sel])),
        "velocity_max_kms":float(np.max(vel[line_sel])),
        "mask_pixels":mask_pixels,
        "mask_x_min":float(TIGHT_VERTICES[:,0].min()),
        "mask_x_max":float(TIGHT_VERTICES[:,0].max()),
        "mask_y_min":float(TIGHT_VERTICES[:,1].min()),
        "mask_y_max":float(TIGHT_VERTICES[:,1].max()),
        "target_line_flux":target_flux,
        "null_sigma":sigma,
        "target_integrated_SNR":target_snr,
        "target_percentile_vs_null":percentile,
        "false_alarm_fraction":fap,
        "centroid_GHz":centroid,
        "centroid_offset_MHz":(centroid-TARGET_GHZ)*1000 if np.isfinite(centroid) else np.nan,
        "peak_frequency_GHz":peak_freq,
        "peak_offset_MHz":(peak_freq-TARGET_GHZ)*1000,
        "peak_amplitude":peak_amp,
        "control_positions":len(cdf)
    }])

    scsv=OUT_CSV/f"{VERSION}_TIGHT_MASK_SUMMARY.csv"
    ccsv=OUT_CSV/f"{VERSION}_TIGHT_MASK_NULLS.csv"
    spcsv=OUT_CSV/f"{VERSION}_TIGHT_MASK_SPECTRUM.csv"
    vcsv=OUT_CSV/f"{VERSION}_TIGHT_MASK_VERTICES.csv"
    summary.to_csv(scsv,index=False); cdf.to_csv(ccsv,index=False)
    pd.DataFrame({"frequency_GHz":centers,"velocity_kms":vel,"aperture_amplitude":amps,"line_window":line_sel}).to_csv(spcsv,index=False)
    pd.DataFrame(TIGHT_VERTICES,columns=["x_pixel","y_pixel"]).to_csv(vcsv,index=False)
    for p in (scsv,ccsv,spcsv,vcsv): (DRIVE_CSV/p.name).write_bytes(p.read_bytes())

    p_mask=OUT_PNG/f"{VERSION}_TIGHT_MASK_AUDIT.png"
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    closed=np.vstack([TIGHT_VERTICES,TIGHT_VERTICES[0]]); ax.plot(closed[:,0],closed[:,1],lw=2.4,label=f"Tight mask: {mask_pixels} pixels")
    ax.set(xlim=(-.5,127.5),ylim=(-.5,127.5),xlabel="X pixel",ylabel="Y pixel",title=f"Model D moment-0 + accepted tight mask | S/N={target_snr:.3f}")
    ax.legend(); fig.colorbar(im,ax=ax,label="Integrated brightness"); savefig(fig,p_mask)

    p_spec=OUT_PNG/f"{VERSION}_TIGHT_MASK_SPECTRUM.png"
    fig,ax=plt.subplots(figsize=(11,7)); ax.plot(centers,amps,lw=1.8)
    ax.axvline(TARGET_GHZ,ls="--",label="Target 279.901 GHz")
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.16,label="8-channel line window")
    if np.isfinite(centroid): ax.axvline(centroid,ls=":",label=f"Positive-weight centroid {centroid:.6f} GHz")
    ax.set(xlabel="Observed frequency (GHz)",ylabel="Tight-mask amplitude",title=f"Tight-aperture FITS spectrum | integrated S/N={target_snr:.3f}")
    ax.grid(alpha=.25); ax.legend(); savefig(fig,p_spec)

    p_null=OUT_PNG/f"{VERSION}_TIGHT_MASK_NULL_DISTRIBUTION.png"
    fig,ax=plt.subplots(figsize=(10,7)); ax.hist(cdf["line_SNR"],bins=24,alpha=.85,label=f"{len(cdf)} non-wrapping controls")
    ax.axvline(target_snr,ls="--",lw=2,label=f"Target S/N={target_snr:.3f}")
    ax.set(xlabel="Integrated aperture S/N",ylabel="Count",title=f"Null test | percentile={percentile:.1f}% | FAP={fap:.3f}")
    ax.grid(alpha=.25); ax.legend(); savefig(fig,p_null)

    math_html=(
        f"<h3>{VERSION} — accepted tight-mask FITS analysis</h3>"
        f"<b>Mask:</b> {mask_pixels} pixels; L2-normalized binary aperture<br>"
        f"<b>Integrated line statistic:</b> Σ aperture amplitudes over {int(line_sel.sum())} grouped channels<br>"
        f"<b>Target line flux:</b> {target_flux:.6g}<br>"
        f"<b>Robust null σ:</b> {sigma:.6g}<br>"
        f"<b>Integrated S/N = target flux / null σ:</b> {target_snr:.3f}<br>"
        f"<b>Null percentile:</b> {percentile:.1f}%<br>"
        f"<b>False-alarm fraction:</b> {fap:.3f}<br>"
        f"<b>Positive-weight centroid:</b> {centroid:.6f} GHz<br>"
        f"<b>Strongest grouped channel:</b> {peak_freq:.6f} GHz<br>"
        f"<b>Controls:</b> {len(cdf)} non-wrapping shifted apertures"
    )
    tabs=widgets.Tab(children=[
        widgets.VBox([widgets.Image(value=p_mask.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=p_spec.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=p_null.read_bytes(),format="png")])
    ])
    for i,t in enumerate(["Mask audit","Extracted spectrum","Null controls"]): tabs.set_title(i,t)
    display(widgets.VBox([widgets.HTML(math_html),tabs]))

    print(f"CODE OUTPUT: {VERSION}")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    for p in (p_mask,p_spec,p_null): print(f"PNG OUTPUT: {p}")
    for p in (scsv,ccsv,spcsv,vcsv): print(f"CSV OUTPUT: {p}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__": main()
