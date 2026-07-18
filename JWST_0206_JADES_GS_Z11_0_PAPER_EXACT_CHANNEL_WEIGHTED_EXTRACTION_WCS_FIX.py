from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import label
from scipy.optimize import curve_fit

VERSION='JWST_0206'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
REST_GHZ=3393.006244
C_KMS=299792.458
N_MC=1000
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG=ROOT/'PNG'; DRIVE_CSV=ROOT/'CSV'
for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)

def world_axis(h,ax):
    n=int(h[f'NAXIS{ax}']); pix=np.arange(n)+1.0
    vals=float(h[f'CRVAL{ax}'])+(pix-float(h[f'CRPIX{ax}']))*float(h.get(f'CDELT{ax}',h.get(f'CD{ax}_{ax}')))
    u=str(h.get(f'CUNIT{ax}','Hz')).lower()
    if u=='hz': vals/=1e9
    elif u=='khz': vals/=1e6
    elif u=='mhz': vals/=1e3
    return vals

def spec_axis(h):
    for ax in range(1,int(h['NAXIS'])+1):
        if 'FREQ' in str(h.get(f'CTYPE{ax}','')).upper(): return ax
    raise RuntimeError('No frequency axis')

def robust_rms(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    med=np.median(x); mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)

def gauss(x,a,mu,sig,c):
    return c+a*np.exp(-0.5*((x-mu)/sig)**2)

def scalar(v):
    a=np.asarray(v,dtype=float).reshape(-1)
    if a.size==0 or not np.isfinite(a[0]): raise RuntimeError('Invalid WCS pixel coordinate')
    return float(a[0])

def main():
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data; shape=data.shape
        fax=spec_axis(h); freq=world_axis(h,fax); np_spec=data.ndim-fax
        w=WCS(h).celestial
        tx_raw,ty_raw=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(scalar(tx_raw))); ty=int(np.rint(scalar(ty_raw)))
        pixscale=abs(float(h['CDELT2']))*3600.0
        bmaj=float(h.get('BMAJ',0))*3600.0; bmin=float(h.get('BMIN',0))*3600.0
        beam_pix=np.pi*bmaj*bmin/(4*np.log(2)*pixscale**2) if bmaj>0 and bmin>0 else np.nan
        half=64; ys=slice(max(0,ty-half),min(shape[-2],ty+half)); xs=slice(max(0,tx-half),min(shape[-1],tx+half))
        ny=ys.stop-ys.start; nx=xs.stop-xs.start; txc=tx-xs.start; tyc=ty-ys.start
        native=np.nanmedian(np.diff(freq))*1000.0
        k=max(1,int(round(5.0/native)))
        nearest=int(np.argmin(np.abs(freq-TARGET_GHZ)))
        start=nearest-(nearest%k)
        lo=max(0,start-12*k); hi=min(len(freq),start+12*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k) if i<hi]
        centers=np.array([np.mean(freq[g]) for g in groups])
        cube5=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            arr=np.asarray(data[tuple(sl)],np.float32); arr=np.moveaxis(arr,np_spec,0)
            while arr.ndim>3: arr=np.nanmean(arr,axis=1)
            cube5[j]=np.nanmean(arr,axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz paper cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); rr_arc=np.hypot((xx-txc)*pixscale,(yy-tyc)*pixscale)
    central_exclude=rr_arc<1.0
    rms_chan=np.array([robust_rms(im[~central_exclude]) for im in cube5])
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14.0)&(vel<=29.0)
    if line_sel.sum()<5:
        order=np.argsort(np.abs(centers-TARGET_GHZ))[:9]; line_sel=np.zeros(len(centers),bool); line_sel[order]=True
    sub=cube5[line_sel]
    sig0=np.sqrt(np.sum(rms_chan[line_sel]**2))
    moment0=np.sum(sub,axis=0); sn0=moment0/sig0
    cand=(sn0>=2.0)&(rr_arc<=0.8)
    labs,nlab=label(cand); mask=np.zeros_like(cand)
    if nlab:
        candidates=[]
        for q in range(1,nlab+1):
            m=labs==q
            if m.any(): candidates.append((np.sum(sn0[m])/(1+np.min(rr_arc[m])),m))
        if candidates: mask=max(candidates,key=lambda z:z[0])[1]
    if not mask.any(): mask=rr_arc<=max(0.18,np.sqrt((bmaj*bmin)/4) if bmaj>0 else 0.25)
    diagnostics=[]; weights=np.ones(line_sel.sum())
    for it in range(2):
        sb=np.tensordot(weights,sub,axes=(0,0))/np.sum(np.abs(weights))
        sb_rms=robust_rms(sb[~central_exclude]); sn=sb/sb_rms
        cand=(sn>=2.0)&(rr_arc<=0.8); labs,nlab=label(cand)
        if nlab:
            cc=[]
            for q in range(1,nlab+1):
                m=labs==q
                if m.any(): cc.append((np.sum(sn[m])/(1+np.min(rr_arc[m])),m))
            if cc: mask=max(cc,key=lambda z:z[0])[1]
        spec=np.array([np.nansum(im[mask]) for im in cube5]); weights=spec[line_sel].copy()
        if not np.any(np.isfinite(weights)) or np.allclose(weights,0): weights=np.ones(line_sel.sum())
        diagnostics.append((it+1,int(mask.sum()),float(np.nanmax(sn))))
    spec=np.array([np.nansum(im[mask]) for im in cube5])
    err=rms_chan*np.sqrt(max(mask.sum(),1)/max(beam_pix,1.0))
    f=centers[line_sel]; s=spec[line_sel]; e=err[line_sel]
    centroid=np.sum(f*s)/np.sum(s)
    rng=np.random.default_rng(20260717); mc=[]
    for _ in range(N_MC):
        sm=s+rng.normal(0,e); d=np.sum(sm)
        if d!=0 and np.isfinite(d): mc.append(np.sum(f*sm)/d)
    mc=np.asarray(mc); cent_err=np.nanstd(mc)
    int_snr=np.sum(s)/np.sqrt(np.sum(e**2))
    try:
        p0=[np.nanmax(s)-np.nanmedian(s),centroid,0.012,np.nanmedian(s)]
        popt,_=curve_fit(gauss,f,s,p0=p0,sigma=e,absolute_sigma=True,maxfev=20000)
        gmu=popt[1]; gfwhm=2.35482*abs(popt[2])/TARGET_GHZ*C_KMS
    except Exception:
        gmu=np.nan; gfwhm=np.nan
    df=pd.DataFrame({'frequency_GHz':centers,'velocity_km_s':vel,'flux_sum_Jy_beam':spec,'sigma_Jy_beam':err,'line_window':line_sel})
    csv=OUT_CSV/f'{VERSION}_PAPER_EXACT_SPECTRUM.csv'; df.to_csv(csv,index=False); df.to_csv(DRIVE_CSV/csv.name,index=False)
    mdf=pd.DataFrame({'centroid_GHz':mc}); mcsv=OUT_CSV/f'{VERSION}_MONTE_CARLO_CENTROIDS.csv'; mdf.to_csv(mcsv,index=False); mdf.to_csv(DRIVE_CSV/mcsv.name,index=False)
    plt.figure(figsize=(13,7)); plt.plot(centers,spec*1000,'o-',ms=4,lw=1.2,label='Paper-style extraction')
    plt.fill_between(centers,(spec-err)*1000,(spec+err)*1000,alpha=.18,label='±1σ')
    plt.axvline(TARGET_GHZ,ls='--',label='Paper 279.901 GHz'); plt.axvline(centroid,ls=':',label=f'Centroid {centroid:.6f} GHz')
    plt.axvspan(f.min(),f.max(),alpha=.08); plt.xlabel('Observed frequency (GHz)'); plt.ylabel('Integrated flux (mJy beam⁻¹)')
    plt.title('JADES-GS-z11-0 — paper-channel iterative extraction'); plt.grid(alpha=.2); plt.legend(); plt.tight_layout()
    png=OUT_PNG/f'{VERSION}_PAPER_EXACT_SPECTRUM.png'; plt.savefig(png,dpi=180); plt.close(); (DRIVE_PNG/png.name).write_bytes(png.read_bytes())
    plt.figure(figsize=(8,7)); plt.imshow(moment0,origin='lower'); plt.contour(mask.astype(float),levels=[0.5],linewidths=1.2); plt.scatter([txc],[tyc],marker='+',s=120)
    plt.title('Paper line window moment-0 and S/N≥2 extraction mask'); plt.tight_layout()
    mpng=OUT_PNG/f'{VERSION}_MOMENT0_MASK.png'; plt.savefig(mpng,dpi=180); plt.close(); (DRIVE_PNG/mpng.name).write_bytes(mpng.read_bytes())
    print(f'CODE OUTPUT: {VERSION}')
    print(f'Target WCS pixel: x={tx}, y={ty} | cutout x={txc}, y={tyc}')
    print(f'Beam: {bmaj:.4f} x {bmin:.4f} arcsec | pixel scale={pixscale:.5f} arcsec | beam area={beam_pix:.2f} pixels')
    print(f'Native spacing: {native:.6f} MHz | grouped channels/bin={k}')
    print(f'Paper velocity window: {vel[line_sel].min():.3f} to {vel[line_sel].max():.3f} km/s | bins={line_sel.sum()}')
    for it,npix,peak in diagnostics: print(f'Iteration {it}: mask pixels={npix:4d} | peak S/N={peak:7.3f}')
    print(f'Flux-weighted centroid: {centroid:.6f} ± {cent_err:.6f} GHz')
    print(f'Paper centroid: 279.901000 ± 0.014000 GHz | offset={(centroid-TARGET_GHZ)*1000:.3f} MHz')
    print(f'Gaussian centroid: {gmu:.6f} GHz | FWHM={gfwhm:.3f} km/s')
    print(f'Integrated line-window S/N: {int_snr:.3f}')
    print(f'PNG: {png}\nPNG: {mpng}\nCSV: {csv}\nCSV: {mcsv}')
    print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')

if __name__=='__main__': main()
