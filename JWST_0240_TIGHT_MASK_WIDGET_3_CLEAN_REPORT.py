from pathlib import Path
from datetime import datetime, timezone, timedelta
from io import BytesIO
import gc, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0240"
MODEL="MODEL_E_TIGHT_MASK_WIDGET_3_CLEAN_REPORT"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"

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

def polygon_mask(vertices,shape):
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    pts=np.column_stack((xx.ravel(),yy.ravel()))
    return MplPath(vertices).contains_points(pts).reshape(shape).astype(np.float32)

def robust_sigma(x):
    x=np.asarray(x,float)
    x=x[np.isfinite(x)]
    med=np.median(x)
    mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)

def shifted_nonwrapping(template,dx,dy):
    out=np.zeros_like(template)
    y0=max(0,dy); y1=min(template.shape[0],template.shape[0]+dy)
    x0=max(0,dx); x1=min(template.shape[1],template.shape[1]+dx)
    sy0=max(0,-dy); sy1=sy0+(y1-y0)
    sx0=max(0,-dx); sx1=sx0+(x1-x0)
    if y1>y0 and x1>x0:
        out[y0:y1,x0:x1]=template[sy0:sy1,sx0:sx1]
    return out

def fig_bytes(fig):
    buf=BytesIO()
    fig.savefig(buf,format="png",dpi=250,bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def scientific_table(rows):
    body="".join(
        f"<tr><td>{name}</td><td style='text-align:right;font-weight:700'>{value}</td><td>{unit}</td><td>{note}</td></tr>"
        for name,value,unit,note in rows
    )
    return f"""
    <div style='background:#071019;color:#eaf2ff;border:1px solid #29445f;border-radius:10px;padding:14px;font-family:Arial,sans-serif'>
      <div style='font-size:20px;font-weight:700;margin-bottom:10px'>Tight Mask Widget 3 — Scientific Summary</div>
      <table style='border-collapse:collapse;width:100%;font-size:14px'>
        <thead><tr style='background:#102235'>
          <th style='text-align:left;padding:8px'>Quantity</th>
          <th style='text-align:right;padding:8px'>Value</th>
          <th style='text-align:left;padding:8px'>Unit</th>
          <th style='text-align:left;padding:8px'>Interpretation</th>
        </tr></thead>
        <tbody>{body}</tbody>
      </table>
    </div>
    """

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
        tx0,ty0=WCS(h).celestial.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); half=64
        ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half)
        pix=abs(float(h["CDELT2"]))*3600
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
        line_sel=np.zeros(len(centers),bool)
        line_sel[ids]=True

    moment=np.sum(cube[line_sel],axis=0)
    mask=polygon_mask(MASK_VERTICES,moment.shape)
    nmask=int(mask.sum())
    if nmask==0:
        raise RuntimeError("Tight Mask Widget 3 produced an empty mask")

    template=mask/np.sqrt(np.sum(mask**2))
    amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=float(np.sum(amps[line_sel]))

    controls=[]
    for radius in (18,24,30,36,42):
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            dx=int(round(radius*np.cos(angle)))
            dy=int(round(radius*np.sin(angle)))
            shifted=shifted_nonwrapping(template,dx,dy)
            ca=np.array([np.nansum(im*shifted) for im in cube])
            controls.append(float(np.sum(ca[line_sel])))
    controls=np.asarray(controls,float)
    sigma=robust_sigma(controls)
    snr=target_flux/sigma
    percentile=100*np.mean(controls<=target_flux)
    fap=np.mean(controls>=target_flux)

    pos=np.clip(amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    peak_local=int(np.argmax(amps[line_sel]))
    peak_freq=float(centers[line_sel][peak_local])
    peak_amp=float(amps[line_sel][peak_local])

    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — raw")
    fig.colorbar(im,ax=ax,label="Integrated brightness")
    raw_png=fig_bytes(fig)

    fig,ax=plt.subplots(figsize=(9,8))
    im=ax.imshow(moment,origin="lower",cmap="viridis",interpolation="nearest")
    closed=np.vstack([MASK_VERTICES,MASK_VERTICES[0]])
    ax.plot(closed[:,0],closed[:,1],lw=2.2,label=f"Tight Mask 3: {nmask} pixels")
    ax.set_xlim(-0.5,127.5); ax.set_ylim(-0.5,127.5)
    ax.set_xlabel("X pixel"); ax.set_ylabel("Y pixel")
    ax.set_title("Model D line-window moment-0 — Tight Mask Widget 3")
    ax.legend(); fig.colorbar(im,ax=ax,label="Integrated brightness")
    masked_png=fig_bytes(fig)

    fig,ax=plt.subplots(figsize=(11,7))
    ax.plot(centers,amps,lw=1.8)
    ax.axvline(TARGET_GHZ,ls="--",label="Target 279.901 GHz")
    ax.axvline(centroid,ls=":",label=f"Centroid {centroid:.6f} GHz")
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.15,label="Line window")
    ax.set_xlabel("Observed frequency (GHz)")
    ax.set_ylabel("Aperture amplitude")
    ax.set_title(f"Tight Mask Widget 3 spectrum | S/N={snr:.3f}")
    ax.grid(alpha=.25); ax.legend()
    spectrum_png=fig_bytes(fig)

    fig,ax=plt.subplots(figsize=(10,7))
    null_snr=controls/sigma
    ax.hist(null_snr,bins=24,alpha=.8,label=f"{len(null_snr)} null positions")
    ax.axvline(snr,ls="--",lw=2,label=f"Target S/N={snr:.3f}")
    ax.set_xlabel("Integrated aperture S/N")
    ax.set_ylabel("Count")
    ax.set_title(f"Null test | percentile={percentile:.1f}% | FAP={fap:.3f}")
    ax.grid(alpha=.25); ax.legend()
    null_png=fig_bytes(fig)

    rows=[
        ("Mask area",f"{nmask:d}","pixels","Binary polygon aperture"),
        ("Line channels",f"{int(line_sel.sum()):d}","grouped channels","Approximately 5 MHz each"),
        ("Velocity window",f"{np.min(vel[line_sel]):.3f} to {np.max(vel[line_sel]):.3f}","km/s","Fixed Model D integration window"),
        ("Integrated line statistic",f"{target_flux:.7f}","aperture units","Sum across selected channels"),
        ("Robust null sigma",f"{sigma:.7f}","aperture units","MAD-based control dispersion"),
        ("Integrated S/N",f"{snr:.3f}","sigma","Target statistic divided by null sigma"),
        ("Null percentile",f"{percentile:.1f}","%","Fraction of controls below target"),
        ("False-alarm fraction",f"{fap:.4f}","fraction","Controls at or above target"),
        ("Positive-weight centroid",f"{centroid:.6f}","GHz",f"Offset {(centroid-TARGET_GHZ)*1000:+.3f} MHz"),
        ("Strongest grouped channel",f"{peak_freq:.6f}","GHz",f"Offset {(peak_freq-TARGET_GHZ)*1000:+.3f} MHz"),
        ("Peak channel amplitude",f"{peak_amp:.7f}","aperture units","Maximum grouped-channel response"),
        ("Null controls",f"{len(controls):d}","positions","Non-wrapping shifted apertures")
    ]

    table=widgets.HTML(scientific_table(rows))
    image=widgets.Image(value=raw_png,format="png")
    toggle=widgets.ToggleButtons(
        options=[("Raw image","raw"),("Mask 3 overlay","mask")],
        value="raw",description="Compare:"
    )
    def switch(change):
        image.value=raw_png if change["new"]=="raw" else masked_png
    toggle.observe(switch,names="value")

    tabs=widgets.Tab(children=[
        widgets.VBox([toggle,image]),
        widgets.VBox([widgets.Image(value=spectrum_png,format="png")]),
        widgets.VBox([widgets.Image(value=null_png,format="png")]),
        widgets.VBox([table])
    ])
    tabs.set_title(0,"Raw ↔ Mask 3")
    tabs.set_title(1,"Spectrum")
    tabs.set_title(2,"Null controls")
    tabs.set_title(3,"Scientific table")
    display(tabs)

    print(f"CODE OUTPUT: {VERSION}")
    print(f"MODEL: {MODEL}")
    print("No PNG or CSV files were written.")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__":
    main()
