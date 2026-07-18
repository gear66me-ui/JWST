from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, json, base64, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import label
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0209'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
N_REPEAT=3
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG=ROOT/'PNG'; DRIVE_CSV=ROOT/'CSV'
for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d','text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8','xtick.color':'#c7d4e5','ytick.color':'#c7d4e5','grid.color':'#33485f'})
COLORS={'beam':'#4dd0e1','anchored':'#ffb74d','free':'#ef5350','paper':'#f5f7fa','target':'#ffffff','mask':'#ce93d8'}

def robust_rms(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    if x.size==0: return np.nan
    med=np.median(x); mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)

def world_axis(h,ax):
    n=int(h[f'NAXIS{ax}']); pix=np.arange(n)+1.0
    v=float(h[f'CRVAL{ax}'])+(pix-float(h[f'CRPIX{ax}']))*float(h.get(f'CDELT{ax}',h.get(f'CD{ax}_{ax}')))
    u=str(h.get(f'CUNIT{ax}','Hz')).lower()
    if u=='hz': v/=1e9
    elif u=='khz': v/=1e6
    elif u=='mhz': v/=1e3
    return v

def spec_axis(h):
    for ax in range(1,int(h['NAXIS'])+1):
        if 'FREQ' in str(h.get(f'CTYPE{ax}','')).upper(): return ax
    raise RuntimeError('No frequency axis')

def collapse_to_cube(arr,np_spec):
    arr=np.moveaxis(arr,np_spec,0)
    while arr.ndim>3: arr=np.nanmean(arr,axis=1)
    return arr

def connected_component(mask,seed):
    labs,n=label(mask)
    sy,sx=seed
    if n==0 or sy<0 or sx<0 or sy>=mask.shape[0] or sx>=mask.shape[1]: return np.zeros_like(mask)
    q=labs[sy,sx]
    return labs==q if q>0 else np.zeros_like(mask)

def centroid(freq,flux,sel):
    f=freq[sel]; s=flux[sel]; d=np.sum(s)
    return np.sum(f*s)/d if np.isfinite(d) and d!=0 else np.nan

def upload_outputs(files):
    token=os.environ.get('GITHUB_TOKEN')
    if not token:
        try:
            from google.colab import userdata
            token=userdata.get('GITHUB_TOKEN')
        except Exception: token=None
    if not token: return 'GitHub upload skipped: add Colab secret GITHUB_TOKEN to enable automatic result commits.'
    import requests
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    uploaded=[]
    for p in files:
        repo_path=f'results/{VERSION}/{stamp}/{p.name}'
        url=f'https://api.github.com/repos/gear66me-ui/JWST/contents/{repo_path}'
        payload={'message':f'Add {VERSION} result {p.name}','content':base64.b64encode(p.read_bytes()).decode(),'branch':'main'}
        r=requests.put(url,headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'},json=payload,timeout=120)
        if r.status_code in (200,201): uploaded.append(repo_path)
    return f'GitHub uploaded: {len(uploaded)}/{len(files)} files' if uploaded else 'GitHub upload attempted but no files were committed.'

def main():
    warnings.filterwarnings('ignore',category=Warning,module='astropy.wcs')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data; shape=data.shape
        fax=spec_axis(h); freq_native=world_axis(h,fax); np_spec=data.ndim-fax
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); w=WCS(h).celestial
        tx0,ty0=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(np.asarray(tx0).reshape(-1)[0])); ty=int(np.rint(np.asarray(ty0).reshape(-1)[0]))
        pixscale=abs(float(h['CDELT2']))*3600
        bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600; bpa=float(h.get('BPA',0))
        half=64; ys=slice(max(0,ty-half),min(shape[-2],ty+half)); xs=slice(max(0,tx-half),min(shape[-1],tx+half))
        ny=ys.stop-ys.start; nx=xs.stop-xs.start; txc=tx-xs.start; tyc=ty-ys.start
        native=np.nanmedian(np.abs(np.diff(freq_native)))*1000; k=max(1,int(round(5.0/native)))
        near=int(np.argmin(np.abs(freq_native-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq_native),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq_native[g]) for g in groups]); cube5=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            arr=collapse_to_cube(np.asarray(data[tuple(sl)],np.float32),np_spec)
            cube5[j]=np.nanmean(arr,axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pixscale; dy=(yy-tyc)*pixscale
    theta=np.deg2rad(bpa); xr=dx*np.cos(theta)+dy*np.sin(theta); yr=-dx*np.sin(theta)+dy*np.cos(theta)
    sigma_x=max(bmaj/2.35482,0.15); sigma_y=max(bmin/2.35482,0.12)
    beam_weight=np.exp(-0.5*((xr/sigma_x)**2+(yr/sigma_y)**2)); beam_mask=beam_weight>=np.exp(-0.5)
    rr=np.hypot(dx,dy); off=rr>1.2
    rms_chan=np.array([robust_rms(im[off]) for im in cube5])
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        idx=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[idx]=True
    moment0=np.sum(cube5[line_sel],axis=0); map_rms=robust_rms(moment0[off]); snmap=moment0/map_rms
    rows=[]; spectra=[]; masks={}
    rng=np.random.default_rng(20260718)
    methods=['Fixed beam','Target-seeded mask','Free optimized mask']
    for rep in range(1,N_REPEAT+1):
        jitter=rng.normal(0,1e-12,cube5.shape).astype(np.float32)
        work=cube5+jitter
        fixed_flux=np.array([np.nansum(im*beam_weight)/np.nansum(beam_weight) for im in work])
        anchored=connected_component((snmap>=2.0)&(rr<=0.8),(tyc,txc))
        if not anchored.any(): anchored=beam_mask.copy()
        anchored_flux=np.array([np.nansum(im[anchored]) for im in work])
        cand=(snmap>=2.0)&(rr<=0.8); labs,n=label(cand); free=np.zeros_like(cand)
        if n:
            choices=[]
            for q in range(1,n+1):
                m=labs==q
                if m.any(): choices.append((np.sum(snmap[m]),m))
            if choices: free=max(choices,key=lambda z:z[0])[1]
        if not free.any(): free=beam_mask.copy()
        free_flux=np.array([np.nansum(im[free]) for im in work])
        for name,flux,mask in [('Fixed beam',fixed_flux,beam_mask),('Target-seeded mask',anchored_flux,anchored),('Free optimized mask',free_flux,free)]:
            c=centroid(centers,flux,line_sel)
            base=np.where(~line_sel)[0]
            sigma=robust_rms(flux[base]) if base.size else robust_rms(flux)
            sn=np.sum(flux[line_sel])/(sigma*np.sqrt(line_sel.sum())) if sigma>0 else np.nan
            rows.append({'method':name,'repeat':rep,'centroid_GHz':c,'offset_MHz':(c-TARGET_GHZ)*1000,'integrated_SNR_proxy':sn,'mask_pixels':int(mask.sum())})
            spectra.append(pd.DataFrame({'method':name,'repeat':rep,'frequency_GHz':centers,'velocity_km_s':vel,'flux':flux,'line_window':line_sel}))
            masks[(name,rep)]=mask.copy()
    summary=pd.DataFrame(rows); specdf=pd.concat(spectra,ignore_index=True)
    stats=summary.groupby('method').agg(centroid_mean_GHz=('centroid_GHz','mean'),centroid_std_GHz=('centroid_GHz','std'),offset_mean_MHz=('offset_MHz','mean'),SNR_mean=('integrated_SNR_proxy','mean'),mask_pixels_mean=('mask_pixels','mean')).reset_index()
    summary_csv=OUT_CSV/f'{VERSION}_REPEATABILITY_RESULTS.csv'; spec_csv=OUT_CSV/f'{VERSION}_ALL_SPECTRA.csv'; stats_csv=OUT_CSV/f'{VERSION}_METHOD_SUMMARY.csv'
    summary.to_csv(summary_csv,index=False); specdf.to_csv(spec_csv,index=False); stats.to_csv(stats_csv,index=False)
    for p in (summary_csv,spec_csv,stats_csv): (DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(moment0,origin='lower',cmap='viridis')
    ax.scatter([txc],[tyc],marker='+',s=160,c=COLORS['target'],linewidths=2,label='JWST target')
    ax.contour(masks[('Target-seeded mask',1)].astype(float),levels=[.5],colors=[COLORS['anchored']],linewidths=2)
    ax.contour(masks[('Free optimized mask',1)].astype(float),levels=[.5],colors=[COLORS['free']],linewidths=2)
    ax.add_patch(Ellipse((txc,tyc),bmaj/pixscale,bmin/pixscale,angle=bpa,fill=False,ec=COLORS['beam'],lw=2))
    ax.set_title('Moment-0 map: fixed beam, target-seeded mask, free mask'); ax.set_xlabel('Cutout x pixel'); ax.set_ylabel('Cutout y pixel'); fig.colorbar(im,ax=ax,label='Integrated surface brightness')
    fig.tight_layout(); p=OUT_PNG/f'{VERSION}_MASK_COMPARISON.png'; fig.savefig(p,dpi=190); plt.close(fig); (DRIVE_PNG/p.name).write_bytes(p.read_bytes()); figs.append(p)
    for name,color in [('Fixed beam',COLORS['beam']),('Target-seeded mask',COLORS['anchored']),('Free optimized mask',COLORS['free'])]:
        fig,ax=plt.subplots(figsize=(12,7))
        for rep in range(1,N_REPEAT+1):
            d=specdf[(specdf.method==name)&(specdf.repeat==rep)]
            ax.plot(d.frequency_GHz,d.flux,lw=1.4,alpha=.82,label=f'Repeat {rep}')
        ax.axvline(TARGET_GHZ,ls='--',lw=1.4,c=COLORS['paper'],label='Paper 279.901 GHz')
        ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.10,color=color)
        ax.set_title(f'{name} — three-repeat spectrum'); ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Extracted flux'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        p=OUT_PNG/f"{VERSION}_{name.upper().replace(' ','_').replace('-','_')}_SPECTRUM.png"; fig.savefig(p,dpi=190); plt.close(fig); (DRIVE_PNG/p.name).write_bytes(p.read_bytes()); figs.append(p)
    fig,ax=plt.subplots(figsize=(12,7)); y=np.arange(len(stats))
    ax.errorbar(stats.centroid_mean_GHz,y,xerr=stats.centroid_std_GHz.fillna(0),fmt='o',capsize=5,lw=1.5)
    ax.axvline(TARGET_GHZ,ls='--',c=COLORS['paper'],label='Paper centroid')
    ax.set_yticks(y,stats.method); ax.invert_yaxis(); ax.set_xlabel('Centroid frequency (GHz)'); ax.set_title('Three-method repeatability comparison'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    p=OUT_PNG/f'{VERSION}_CENTROID_REPEATABILITY.png'; fig.savefig(p,dpi=190); plt.close(fig); (DRIVE_PNG/p.name).write_bytes(p.read_bytes()); figs.append(p)
    options=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=options,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='94%'))
    img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px'))
    cap=widgets.HTML(f'<b>{figs[0].name}</b>'); state={'i':0}
    prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        i%=len(figs); state['i']=i; p=figs[i]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{i+1}/{len(figs)} — {p.name}</b>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1))
    display(widgets.VBox([widgets.HTML('<h3>JADES-GS-z11-0 automated extraction gallery</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    upload_status=upload_outputs([summary_csv,spec_csv,stats_csv]+figs)
    print(f'CODE OUTPUT: {VERSION}')
    print(stats.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print(f'Images in gallery: {len(figs)}')
    print(f'CSV: {summary_csv}\nCSV: {spec_csv}\nCSV: {stats_csv}')
    print(upload_status)
    print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')

if __name__=='__main__': main()
