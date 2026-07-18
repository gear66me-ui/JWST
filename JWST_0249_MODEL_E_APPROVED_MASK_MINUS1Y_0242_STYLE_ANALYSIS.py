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
from PIL import Image, ImageDraw
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0249"
MODEL="MODEL_E_APPROVED_MASK_MINUS1Y_0242_STYLE_ANALYSIS"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
OUTPUT_PIXELS=1024
MIN_BYTES=2_500_000

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
OUT_PNG=Path("/content/JWST_OUTPUT/PNG")
OUT_CSV=Path("/content/JWST_OUTPUT/CSV")
DRIVE_PNG=ROOT/"PNG"
DRIVE_CSV=ROOT/"CSV"
for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)

SCREEN_VERTICES=np.array([
    [57.58,68.25],[58.08,67.17],[59.67,66.42],[61.83,66.17],
    [63.83,66.58],[64.92,67.58],[65.17,69.00],[64.50,70.42],
    [62.83,71.25],[60.75,71.50],[58.92,70.83],[57.67,69.58]
],dtype=float)

# Approved 0248 mask, then moved one FITS pixel downward in negative Y.
MASK_VERTICES=np.column_stack((127.0-SCREEN_VERTICES[:,0],127.0-SCREEN_VERTICES[:,1]))
MASK_VERTICES[:,1]-=1.0

plt.rcParams.update({
    "figure.facecolor":"#05080d","axes.facecolor":"#05080d",
    "savefig.facecolor":"#05080d","text.color":"#e8f1ff",
    "axes.labelcolor":"#e8f1ff","axes.edgecolor":"#8aa0b8",
    "xtick.color":"#c7d4e5","ytick.color":"#c7d4e5","grid.color":"#33485f"
})

def spec_axis(h):
    for ax in range(1,int(h["NAXIS"])+1):
        if "FREQ" in str(h.get(f"CTYPE{ax}","")).upper(): return ax
    raise RuntimeError("No frequency axis")

def axis_values(h,ax):
    n=int(h[f"NAXIS{ax}"]); p=np.arange(n)+1.0
    step=float(h.get(f"CDELT{ax}",h.get(f"CD{ax}_{ax}")))
    return (float(h[f"CRVAL{ax}"])+(p-float(h[f"CRPIX{ax}"]))*step)/1e9

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
    return 1.4826*mad if mad>0 else np.std(x)

def shifted_nonwrapping(template,dx,dy):
    out=np.zeros_like(template)
    y0=max(0,dy); y1=min(template.shape[0],template.shape[0]+dy)
    x0=max(0,dx); x1=min(template.shape[1],template.shape[1]+dx)
    sy0=max(0,-dy); sx0=max(0,-dx)
    if y1>y0 and x1>x0: out[y0:y1,x0:x1]=template[sy0:sy0+y1-y0,sx0:sx0+x1-x0]
    return out

def rgb_master(moment,vertices=None):
    finite=moment[np.isfinite(moment)]
    vmin=float(np.nanmin(finite)); vmax=float(np.nanmax(finite))
    scaled=np.nan_to_num(np.clip((moment-vmin)/(vmax-vmin),0,1),nan=0.0)
    rgb=(plt.get_cmap("viridis")(scaled)[...,:3]*255).astype(np.uint8)
    image=Image.fromarray(rgb,"RGB").resize((OUTPUT_PIXELS,OUTPUT_PIXELS),Image.Resampling.NEAREST)
    if vertices is not None:
        draw=ImageDraw.Draw(image)
        scale=OUTPUT_PIXELS/128.0
        pts=[(float(x)*scale,float(y)*scale) for x,y in vertices]
        pts.append(pts[0]); draw.line(pts,fill=(255,35,20),width=5,joint="curve")
    return image

def fig_bytes(fig):
    buf=BytesIO(); fig.savefig(buf,format="png",dpi=250,bbox_inches="tight"); plt.close(fig); return buf.getvalue()

def table_html(rows):
    body="".join(f"<tr><td style='padding:8px'>{a}</td><td style='padding:8px;text-align:right;font-weight:700'>{b}</td><td style='padding:8px'>{c}</td><td style='padding:8px'>{d}</td></tr>" for a,b,c,d in rows)
    return f"""<div style='background:#071019;color:#eaf2ff;border:1px solid #29445f;border-radius:10px;padding:14px;font-family:Arial,sans-serif;max-width:980px'><div style='font-size:21px;font-weight:700;margin-bottom:10px'>{VERSION} — Approved Mask Scientific Summary</div><table style='border-collapse:collapse;width:100%;font-size:14px'><thead><tr style='background:#102235'><th style='text-align:left;padding:8px'>Quantity</th><th style='text-align:right;padding:8px'>Value</th><th style='text-align:left;padding:8px'>Unit</th><th style='text-align:left;padding:8px'>Interpretation</th></tr></thead><tbody>{body}</tbody></table></div>"""

def main():
    warnings.filterwarnings("ignore")
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        tx,ty=WCS(h).celestial.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx)); ty=int(np.rint(ty)); half=64
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
    if nmask==0: raise RuntimeError("Approved mask is empty")
    template=mask/np.sqrt(np.sum(mask**2))
    amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=float(np.sum(amps[line_sel]))

    controls=[]
    for radius in (18,24,30,36,42):
        for angle in np.linspace(0,2*np.pi,32,endpoint=False):
            shifted=shifted_nonwrapping(template,int(round(radius*np.cos(angle))),int(round(radius*np.sin(angle))))
            controls.append(float(np.nansum(moment*shifted)))
    controls=np.asarray(controls,float)
    sigma=robust_sigma(controls); snr=target_flux/sigma
    percentile=100*np.mean(controls<=target_flux)
    exceedances=int(np.sum(controls>=target_flux)); fap=exceedances/len(controls)

    pos=np.clip(amps[line_sel],0,None)
    centroid=float(np.sum(centers[line_sel]*pos)/np.sum(pos)) if np.sum(pos)>0 else np.nan
    peak_i=int(np.argmax(amps[line_sel])); peak_freq=float(centers[line_sel][peak_i]); peak_amp=float(amps[line_sel][peak_i])

    raw_image=rgb_master(moment)
    masked_image=rgb_master(moment,MASK_VERTICES)
    raw_path=OUT_PNG/f"{VERSION}_MODEL_D_MOMENT0_RAW.png"
    mask_path=OUT_PNG/f"{VERSION}_MODEL_E_APPROVED_MASK_MINUS1Y.png"
    raw_image.save(raw_path,"PNG",compress_level=0,optimize=False)
    masked_image.save(mask_path,"PNG",compress_level=0,optimize=False)
    if raw_path.stat().st_size<MIN_BYTES or mask_path.stat().st_size<MIN_BYTES: raise RuntimeError("PNG export below 2.5 MB")
    drive_raw=DRIVE_PNG/raw_path.name; drive_mask=DRIVE_PNG/mask_path.name
    drive_raw.write_bytes(raw_path.read_bytes()); drive_mask.write_bytes(mask_path.read_bytes())

    summary=pd.DataFrame([{
        "version":VERSION,"model":MODEL,"mask_pixels":nmask,"line_channels":int(line_sel.sum()),
        "target_line_flux":target_flux,"null_sigma":sigma,"integrated_SNR":snr,
        "null_percentile":percentile,"null_exceedances":exceedances,"false_alarm_fraction":fap,
        "centroid_GHz":centroid,"centroid_offset_MHz":(centroid-TARGET_GHZ)*1000,
        "peak_frequency_GHz":peak_freq,"peak_offset_MHz":(peak_freq-TARGET_GHZ)*1000,
        "peak_amplitude":peak_amp,"control_positions":len(controls)
    }])
    csv_path=OUT_CSV/f"{VERSION}_SUMMARY.csv"; summary.to_csv(csv_path,index=False)
    drive_csv=DRIVE_CSV/csv_path.name; drive_csv.write_bytes(csv_path.read_bytes())

    fig,ax=plt.subplots(figsize=(11,7)); ax.plot(centers,amps,lw=1.8)
    ax.axvline(TARGET_GHZ,ls="--",label="Target 279.901 GHz")
    ax.axvline(centroid,ls=":",label=f"Centroid {centroid:.6f} GHz")
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.15,label="Model D line window")
    ax.set_xlabel("Observed frequency (GHz)"); ax.set_ylabel("Aperture amplitude")
    ax.set_title(f"Approved mask spectrum | integrated S/N={snr:.3f}"); ax.grid(alpha=.25); ax.legend()
    spectrum_png=fig_bytes(fig)

    fig,ax=plt.subplots(figsize=(10,7)); null_snr=controls/sigma
    ax.hist(null_snr,bins=24,alpha=.8,label=f"{len(null_snr)} null positions")
    ax.axvline(snr,ls="--",lw=2,label=f"Target S/N={snr:.3f}")
    ax.set_xlabel("Integrated aperture S/N"); ax.set_ylabel("Count")
    ax.set_title(f"Null test | percentile={percentile:.1f}% | exceedances={exceedances}/{len(controls)}")
    ax.grid(alpha=.25); ax.legend(); null_png=fig_bytes(fig)

    rows=[
        ("Mask area",f"{nmask}","pixels","Approved 0248 mask shifted one pixel toward −Y"),
        ("Mask X bounds",f"{MASK_VERTICES[:,0].min():.3f} to {MASK_VERTICES[:,0].max():.3f}","pixels","Both-axis-flipped coordinate frame"),
        ("Mask Y bounds",f"{MASK_VERTICES[:,1].min():.3f} to {MASK_VERTICES[:,1].max():.3f}","pixels","Includes requested −1 pixel shift"),
        ("Line channels",f"{int(line_sel.sum())}","grouped channels","Approximately 5 MHz each"),
        ("Integrated line statistic",f"{target_flux:.7f}","aperture units","Selected-channel sum"),
        ("Robust null sigma",f"{sigma:.7f}","aperture units","MAD-based control dispersion"),
        ("Integrated S/N",f"{snr:.3f}","sigma","Target statistic divided by null sigma"),
        ("Null percentile",f"{percentile:.1f}","%","Controls below target"),
        ("Null exceedances",f"{exceedances}/{len(controls)}","positions",f"Empirical FAP {fap:.4f}"),
        ("Positive-weight centroid",f"{centroid:.6f}","GHz",f"Offset {(centroid-TARGET_GHZ)*1000:+.3f} MHz"),
        ("Strongest grouped channel",f"{peak_freq:.6f}","GHz",f"Offset {(peak_freq-TARGET_GHZ)*1000:+.3f} MHz"),
        ("Peak channel amplitude",f"{peak_amp:.7f}","aperture units","Maximum grouped-channel response"),
        ("Raw PNG size",f"{raw_path.stat().st_size/1_000_000:.3f}","MB","1024×1024, compression disabled"),
        ("Masked PNG size",f"{mask_path.stat().st_size/1_000_000:.3f}","MB","1024×1024, compression disabled")
    ]

    raw_bytes=raw_path.read_bytes(); mask_bytes=mask_path.read_bytes()
    toggle_img=widgets.Image(value=raw_bytes,format="png",layout=widgets.Layout(width="900px",max_width="100%"))
    toggle=widgets.ToggleButtons(options=[("Raw Model D","raw"),("Approved mask −1Y","mask")],value="raw",description="Compare:")
    toggle.observe(lambda ch:setattr(toggle_img,"value",raw_bytes if ch["new"]=="raw" else mask_bytes),names="value")

    def button(label,path,style):
        b=widgets.Button(description=label,button_style=style,icon="download")
        def go(_):
            from google.colab import files; files.download(str(path))
        b.on_click(go); return b

    display(widgets.HTML(f"<h3>{VERSION} — approved mask analysis</h3>"))
    display(widgets.HTML("<h4>Raw Model D moment-0</h4>")); display(widgets.Image(value=raw_bytes,format="png",layout=widgets.Layout(width="900px",max_width="100%")))
    display(widgets.HTML("<h4>Approved mask moved one pixel down</h4>")); display(widgets.Image(value=mask_bytes,format="png",layout=widgets.Layout(width="900px",max_width="100%")))
    display(toggle); display(toggle_img)
    display(widgets.HBox([button("Download raw PNG",raw_path,"success"),button("Download masked PNG",mask_path,"warning"),button("Download summary CSV",csv_path,"info")]))
    display(widgets.Image(value=spectrum_png,format="png")); display(widgets.Image(value=null_png,format="png"))
    display(widgets.HTML(table_html(rows)))
    display(widgets.HTML(f"<div style='font-family:monospace;background:#071019;color:#eaf2ff;padding:10px;border-radius:8px'><b>Drive raw:</b> {drive_raw}<br><b>Drive masked:</b> {drive_mask}<br><b>Drive summary:</b> {drive_csv}</div>"))

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
