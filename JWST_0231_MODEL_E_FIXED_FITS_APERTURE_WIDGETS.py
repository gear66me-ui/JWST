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

VERSION="JWST_0231"
MODEL="MODEL_E_FIXED_FITS_APERTURE"
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

APERTURE_VERTICES=np.array([
    [71.2,70.1],[68.7,72.5],[66.1,72.7],[61.3,69.9],
    [60.0,68.3],[59.8,66.5],[63.3,65.7],[69.5,67.6]
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

def polygon_mask(vertices,shape=(128,128)):
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    pts=np.column_stack((xx.ravel(),yy.ravel()))
    return MplPath(vertices).contains_points(pts).reshape(shape).astype(np.float32)

def robust_rms(x):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    med=np.median(x)
    mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)

def savefig(fig,path):
    fig.savefig(path,dpi=300,bbox_inches="tight")
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
        pix=abs(float(h["CDELT2"]))*3600
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
    mask=polygon_mask(APERTURE_VERTICES,moment.shape)
    mask_pixels=int(mask.sum())
    if mask_pixels==0:
        raise RuntimeError("Fixed polygon produced an empty mask")
    template=mask/np.sqrt(np.sum(mask**2))

    amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=float(np.sum(amps[line_sel]))

    controls=[]
    for radius in (18,24,30,36,42):
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            dx=int(round(radius*np.cos(angle))); dy=int(round(radius*np.sin(angle)))
            shifted=np.roll(np.roll(template,dy,axis=0),dx,axis=1)
            ca=np.array([np.nansum(im*shifted) for im in cube])
            controls.append({"radius_pix":radius,"dx_pix":dx,"dy_pix":dy,
                             "offset_arcsec":np.hypot(dx,dy)*pix,
                             "line_flux":float(np.sum(ca[line_sel]))})
    cdf=pd.DataFrame(controls)
    sigma=robust_rms(cdf["line_flux"].values)
    target_snr=target_flux/sigma
    cdf["line_SNR"]=cdf["line_flux"]/sigma
    percentile=100*np.mean(cdf["line_flux"].values<=target_flux)
    fap=np.mean(cdf["line_flux"].values>=target_flux)

    pos=np.clip(amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan

    summary=pd.DataFrame([{
        "model":"Model E fixed aperture from user-marked Model D moment-0",
        "target_frequency_GHz":TARGET_GHZ,
        "line_channels":int(line_sel.sum()),
        "velocity_min_kms":float(np.min(vel[line_sel])),
        "velocity_max_kms":float(np.max(vel[line_sel])),
        "mask_pixels":mask_pixels,
        "mask_x_min":float(APERTURE_VERTICES[:,0].min()),
        "mask_x_max":float(APERTURE_VERTICES[:,0].max()),
        "mask_y_min":float(APERTURE_VERTICES[:,1].min()),
        "mask_y_max":float(APERTURE_VERTICES[:,1].max()),
        "target_line_flux":target_flux,
        "null_sigma":sigma,
        "target_integrated_SNR":target_snr,
        "target_percentile_vs_null":percentile,
        "false_alarm_fraction":fap,
        "centroid_GHz":centroid,
        "centroid_offset_MHz":(centroid-TARGET_GHZ)*1000 if np.isfinite(centroid) else np.nan,
        "control_positions":len(cdf)
    }])

    scsv=OUT_CSV/f"{VERSION}_MODEL_E_FIXED_APERTURE_SUMMARY.csv"
    ccsv=OUT_CSV/f"{VERSION}_MODEL_E_FIXED_APERTURE_NULLS.csv"
    vcsv=OUT_CSV/f"{VERSION}_MODEL_E_FIXED_APERTURE_VERTICES.csv"
    summary.to_csv(scsv,index=False); cdf.to_csv(ccsv,index=False)
    pd.DataFrame(APERTURE_VERTICES,columns=["x_pixel","y_pixel"]).to_csv(vcsv,index=False)
    for p in (scsv,ccsv,vcsv):
        (DRIVE_CSV/p.name).write_bytes(p.read_bytes())

    p_m0=OUT_PNG/f"{VERSION}_MODEL_D_MOMENT0_WITH_FIXED_APERTURE.png"
    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    closed=np.vstack([APERTURE_VERTICES,APERTURE_VERTICES[0]])
    ax.plot(closed[:,0],closed[:,1],lw=2,label=f"Fixed aperture: {mask_pixels} pixels")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title(f"Model D moment-0 with fixed Model E aperture | S/N={target_snr:.3f}")
    ax.legend(); fig.colorbar(im,ax=ax,label="Integrated brightness")
    savefig(fig,p_m0)

    p_spec=OUT_PNG/f"{VERSION}_MODEL_E_FIXED_APERTURE_SPECTRUM.png"
    fig,ax=plt.subplots(figsize=(11,7))
    ax.plot(centers,amps,lw=1.8)
    ax.axvline(TARGET_GHZ,ls="--",label="Target 279.901 GHz")
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.15,label="Model D line window")
    ax.set_xlabel("Observed frequency (GHz)"); ax.set_ylabel("Fixed-aperture amplitude")
    ax.set_title(f"Model E fixed FITS extraction | centroid={centroid:.6f} GHz")
    ax.grid(alpha=.25); ax.legend()
    savefig(fig,p_spec)

    p_null=OUT_PNG/f"{VERSION}_MODEL_E_FIXED_APERTURE_NULL_DISTRIBUTION.png"
    fig,ax=plt.subplots(figsize=(10,7))
    ax.hist(cdf["line_SNR"],bins=24,alpha=.8,label=f"{len(cdf)} null positions")
    ax.axvline(target_snr,ls="--",lw=2,label=f"Target S/N={target_snr:.3f}")
    ax.set_xlabel("Integrated aperture S/N"); ax.set_ylabel("Count")
    ax.set_title(f"Fixed-aperture null test | percentile={percentile:.1f}% | FAP={fap:.3f}")
    ax.grid(alpha=.25); ax.legend()
    savefig(fig,p_null)

    report=widgets.HTML(
        f"<h3>JWST_0231 — Model E fixed FITS aperture</h3>"
        f"<b>Mask pixels:</b> {mask_pixels}<br>"
        f"<b>Target S/N:</b> {target_snr:.3f}<br>"
        f"<b>Null percentile:</b> {percentile:.1f}%<br>"
        f"<b>False-alarm fraction:</b> {fap:.3f}<br>"
        f"<b>Centroid:</b> {centroid:.6f} GHz<br>"
        f"<b>Line channels:</b> {int(line_sel.sum())}"
    )
    tabs=widgets.Tab(children=[
        widgets.VBox([widgets.Image(value=p_m0.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=p_spec.read_bytes(),format="png")]),
        widgets.VBox([widgets.Image(value=p_null.read_bytes(),format="png")])
    ])
    for i,title in enumerate(["Moment-0 + aperture","Extracted spectrum","Null controls"]):
        tabs.set_title(i,title)
    display(widgets.VBox([report,tabs]))

    print(f"CODE OUTPUT: {VERSION}")
    print(summary.to_string(index=False,float_format=lambda x:f"{x:.6f}"))
    print(f"PNG OUTPUT: {p_m0}")
    print(f"PNG OUTPUT: {p_spec}")
    print(f"PNG OUTPUT: {p_null}")
    print(f"CSV OUTPUT: {scsv}")
    print(f"CSV OUTPUT: {ccsv}")
    print(f"CSV OUTPUT: {vcsv}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__":
    main()
