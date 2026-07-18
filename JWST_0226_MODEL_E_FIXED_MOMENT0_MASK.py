from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, gc, base64, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path as MplPath

VERSION='JWST_0226'
MODEL='MODEL_E_FIXED_MOMENT0_MASK'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG=ROOT/'PNG'; DRIVE_CSV=ROOT/'CSV'
for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d','text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8','xtick.color':'#c7d4e5','ytick.color':'#c7d4e5','grid.color':'#33485f'})

FIXED_VERTICES=np.array([
    [34.2,56.7],[39.2,60.7],[51.7,61.6],[58.3,58.0],[65.0,53.6],
    [75.0,49.5],[88.3,48.2],[95.8,44.6],[101.7,36.5],[102.5,24.8],
    [98.3,20.4],[87.5,17.7],[75.0,13.2],[65.0,15.0],[58.3,20.4],
    [50.0,17.7],[41.7,24.8],[35.8,31.1],[33.3,38.3]
],dtype=float)

def robust_rms(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    if x.size==0:return np.nan
    m=np.median(x); mad=np.median(np.abs(x-m))
    return 1.4826*mad if mad>0 else np.std(x)

def spec_axis(h):
    for ax in range(1,int(h['NAXIS'])+1):
        if 'FREQ' in str(h.get(f'CTYPE{ax}','')).upper(): return ax
    raise RuntimeError('No frequency axis')

def axis_values(h,ax):
    n=int(h[f'NAXIS{ax}']); p=np.arange(n)+1.0
    return (float(h[f'CRVAL{ax}'])+(p-float(h[f'CRPIX{ax}']))*float(h.get(f'CDELT{ax}',h.get(f'CD{ax}_{ax}'))))/1e9

def collapse(a,np_spec):
    a=np.moveaxis(a,np_spec,0)
    while a.ndim>3:a=np.nanmean(a,axis=1)
    return a

def polygon_to_mask(vertices,shape):
    yy,xx=np.mgrid[:shape[0],:shape[1]]
    pts=np.column_stack((xx.ravel(),yy.ravel()))
    return MplPath(vertices).contains_points(pts).reshape(shape).astype(np.float32)

def shift_mask(mask,dy,dx):
    return np.roll(np.roll(mask,dy,axis=0),dx,axis=1)

def savefig(fig,name,figs):
    p=OUT_PNG/name; fig.savefig(p,dpi=210,bbox_inches='tight'); plt.close(fig)
    (DRIVE_PNG/p.name).write_bytes(p.read_bytes()); figs.append(p)

def upload(files):
    token=os.environ.get('GITHUB_TOKEN')
    if not token:
        try:
            from google.colab import userdata; token=userdata.get('GITHUB_TOKEN')
        except Exception: token=None
    if not token:return 'GitHub upload skipped: add Colab secret GITHUB_TOKEN.'
    import requests
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); ok=0
    for p in files:
        path=f'results/{VERSION}/{stamp}/{p.name}'
        payload={'message':f'Add {VERSION} result {p.name}','content':base64.b64encode(p.read_bytes()).decode(),'branch':'main'}
        r=requests.put(f'https://api.github.com/repos/gear66me-ui/JWST/contents/{path}',headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'},json=payload,timeout=180)
        ok+=int(r.status_code in (200,201))
    return f'GitHub uploaded: {ok}/{len(files)} files'

def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        aw=WCS(h).celestial; tx0,ty0=aw.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); pix=abs(float(h['CDELT2']))*3600
        bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half); ny=nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq)))*1000; k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups]); cube=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()

    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[ids]=True

    moment=np.sum(cube[line_sel],axis=0)
    template=polygon_to_mask(FIXED_VERTICES,moment.shape)
    if template.sum()==0: raise RuntimeError('Fixed aperture mask is empty.')
    template/=np.sqrt(np.sum(template**2))
    mask_pixels=int(np.count_nonzero(template))

    target_amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=np.sum(target_amps[line_sel])
    controls=[]
    for r in (18,24,30,36,42):
        for a in np.linspace(0,2*np.pi,32,endpoint=False):
            dxp=int(round(r*np.cos(a))); dyp=int(round(r*np.sin(a)))
            t=shift_mask(template,dyp,dxp)
            amps=np.array([np.nansum(im*t) for im in cube])
            controls.append({'radius_pix':r,'dx_pix':dxp,'dy_pix':dyp,'offset_arcsec':np.hypot(dxp,dyp)*pix,'line_flux':np.sum(amps[line_sel]),'peak_channel_flux':np.nanmax(amps[line_sel]),'mask_pixels':int(np.count_nonzero(t))})
    cdf=pd.DataFrame(controls)
    sigma=robust_rms(cdf['line_flux'].values); target_snr=target_flux/sigma
    cdf['line_SNR']=cdf['line_flux']/sigma
    percentile=100*np.mean(cdf['line_flux'].values<=target_flux)
    false_alarm=np.mean(cdf['line_flux'].values>=target_flux)
    pos=np.clip(target_amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    nearest=int(np.argmin(np.abs(centers-TARGET_GHZ)))
    peak_y,peak_x=np.unravel_index(np.nanargmax(moment),moment.shape)
    peak_sep=np.hypot(peak_x-txc,peak_y-tyc)*pix

    summary=pd.DataFrame([{'model':'Model E — fixed screenshot-derived moment-0 mask','target_frequency_GHz':TARGET_GHZ,'nearest_channel_GHz':centers[nearest],'nearest_channel_offset_MHz':(centers[nearest]-TARGET_GHZ)*1000,'line_channels':int(line_sel.sum()),'target_line_flux':target_flux,'null_sigma':sigma,'target_integrated_SNR':target_snr,'target_percentile_vs_null':percentile,'false_alarm_fraction':false_alarm,'centroid_GHz':centroid,'centroid_offset_MHz':(centroid-TARGET_GHZ)*1000 if np.isfinite(centroid) else np.nan,'moment0_peak_offset_arcsec':peak_sep,'control_positions':len(cdf),'mask_pixels':mask_pixels,'beam_major_arcsec':bmaj,'beam_minor_arcsec':bmin}])
    scsv=OUT_CSV/f'{VERSION}_MODEL_E_SUMMARY.csv'; ccsv=OUT_CSV/f'{VERSION}_MODEL_E_NULL_POSITIONS.csv'
    summary.to_csv(scsv,index=False); cdf.to_csv(ccsv,index=False)
    for p in (scsv,ccsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())

    figs=[]
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest')
    ax.contour(template/template.max(),levels=[0.5],colors=['#ff5a36'],linewidths=2.5)
    ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='JWST coordinate')
    ax.scatter([peak_x],[peak_y],marker='x',s=100,c='#fdd835',linewidths=2,label=f'Moment-0 peak offset {peak_sep:.3f} arcsec')
    ax.set_title(f'MODEL E fixed aperture on line-window moment-0\nS/N={target_snr:.3f} | FAP={false_alarm:.3f} | mask={mask_pixels} pixels')
    ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Integrated brightness')
    savefig(fig,f'{VERSION}_MODEL_E_FIXED_MOMENT0_MASK.png',figs)

    fig,ax=plt.subplots(figsize=(11,7)); ax.hist(cdf['line_SNR'],bins=24,alpha=.75,label=f'{len(cdf)} null positions')
    ax.axvline(target_snr,ls='--',lw=2,c='white',label=f'JWST target S/N={target_snr:.3f}')
    ax.set_xlabel('Integrated aperture S/N'); ax.set_ylabel('Control-position count'); ax.grid(alpha=.25); ax.legend()
    ax.set_title(f'MODEL E null distribution | percentile={percentile:.1f}% | FAP={false_alarm:.3f}')
    savefig(fig,f'{VERSION}_MODEL_E_NULL_DISTRIBUTION.png',figs)

    fig,ax=plt.subplots(figsize=(12,7)); ax.plot(centers,target_amps,lw=1.8,label='Fixed-mask amplitude')
    ax.axvline(TARGET_GHZ,ls='--',c='white',label='Target 279.901 GHz')
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.12,label='Line window')
    ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Manual-mask amplitude'); ax.grid(alpha=.25); ax.legend()
    ax.set_title(f'MODEL E spectrum | centroid={centroid:.6f} GHz')
    savefig(fig,f'{VERSION}_MODEL_E_SPECTRUM.png',figs)

    print(f'CODE OUTPUT: {VERSION}')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nOUTPUT FILES')
    for p in [scsv,ccsv]+figs: print(p)
    print(upload([scsv,ccsv]+figs))
    print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')

if __name__=='__main__': main()
