from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, base64, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import label
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0215'
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
COL={'beam':'#4dd0e1','anch':'#ffb74d','free':'#ef5350','paper':'#f5f7fa','target':'#ffffff'}

def robust_rms(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    if x.size==0:return np.nan
    m=np.median(x); mad=np.median(np.abs(x-m))
    return 1.4826*mad if mad>0 else np.std(x)

def spec_axis(h):
    for ax in range(1,int(h['NAXIS'])+1):
        if 'FREQ' in str(h.get(f'CTYPE{ax}','')).upper(): return ax
    raise RuntimeError('No frequency axis')

def world_axis(h,ax):
    n=int(h[f'NAXIS{ax}']); pix=np.arange(n)+1.0
    v=float(h[f'CRVAL{ax}'])+(pix-float(h[f'CRPIX{ax}']))*float(h.get(f'CDELT{ax}',h.get(f'CD{ax}_{ax}')))
    return v/1e9

def collapse(arr,np_spec):
    arr=np.moveaxis(arr,np_spec,0)
    while arr.ndim>3: arr=np.nanmean(arr,axis=1)
    return arr

def connected(mask,seed):
    labs,n=label(mask); sy,sx=seed
    if n==0:return np.zeros_like(mask)
    q=labs[sy,sx]
    return labs==q if q>0 else np.zeros_like(mask)

def savefig(fig,name,figs):
    p=OUT_PNG/name; fig.savefig(p,dpi=190,bbox_inches='tight'); plt.close(fig)
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
        url=f'https://api.github.com/repos/gear66me-ui/JWST/contents/{path}'
        payload={'message':f'Add {VERSION} result {p.name}','content':base64.b64encode(p.read_bytes()).decode(),'branch':'main'}
        r=requests.put(url,headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'},json=payload,timeout=120)
        ok+=int(r.status_code in (200,201))
    return f'GitHub uploaded: {ok}/{len(files)} files'

def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data; shape=data.shape
        fax=spec_axis(h); freq_native=world_axis(h,fax); np_spec=data.ndim-fax
        w=WCS(h).celestial; tx0,ty0=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(np.asarray(tx0).reshape(-1)[0])); ty=int(np.rint(np.asarray(ty0).reshape(-1)[0]))
        pix=abs(float(h['CDELT2']))*3600; bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600; bpa=float(h.get('BPA',0))
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half); ny=nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq_native)))*1000; k=max(1,int(round(5.0/native)))
        near=int(np.argmin(np.abs(freq_native-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq_native),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq_native[g]) for g in groups]); cube5=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube5[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pix; dy=(yy-tyc)*pix; rr=np.hypot(dx,dy)
    th=np.deg2rad(bpa); xr=dx*np.cos(th)+dy*np.sin(th); yr=-dx*np.sin(th)+dy*np.cos(th)
    sx=max(bmaj/2.35482,0.15); sy=max(bmin/2.35482,0.12)
    beam_w=np.exp(-0.5*((xr/sx)**2+(yr/sy)**2)); beam_mask=beam_w>=np.exp(-0.5)
    off=rr>1.2; rms_chan=np.array([robust_rms(im[off]) for im in cube5])
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ; line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    moment0=np.sum(cube5[line_sel],axis=0); map_rms=robust_rms(moment0[off]); snmap=moment0/map_rms
    anchored=connected((snmap>=2)&(rr<=0.8),(tyc,txc)); anchored=anchored if anchored.any() else beam_mask.copy()
    cand=(snmap>=2)&(rr<=0.8); labs,n=label(cand); free=np.zeros_like(cand)
    if n:
        choices=[]
        for q in range(1,n+1):
            m=labs==q
            if m.any(): choices.append((np.sum(snmap[m]),m))
        if choices: free=max(choices,key=lambda z:z[0])[1]
    if not free.any(): free=beam_mask.copy()
    figs=[]
    for j in np.where(line_sel)[0]:
        fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cube5[j],origin='lower',cmap='viridis')
        ax.scatter([txc],[tyc],marker='+',s=160,c=COL['target'],linewidths=2)
        ax.add_patch(Ellipse((txc,tyc),bmaj/pix,bmin/pix,angle=bpa,fill=False,ec=COL['beam'],lw=2))
        ax.set_title(f'Line-window 5 MHz channel | {centers[j]:.6f} GHz | {vel[j]:+.2f} km/s')
        ax.set_xlabel('Cutout x pixel'); ax.set_ylabel('Cutout y pixel'); fig.colorbar(im,ax=ax,label='Surface brightness')
        savefig(fig,f'{VERSION}_CHANNEL_{j:02d}_{centers[j]:.6f}GHz.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment0,origin='lower',cmap='viridis')
    ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2,label='JWST target')
    ax.add_patch(Ellipse((txc,tyc),bmaj/pix,bmin/pix,angle=bpa,fill=False,ec=COL['beam'],lw=2,label='Fixed beam'))
    ax.contour(anchored.astype(float),levels=[.5],colors=[COL['anch']],linewidths=2)
    ax.contour(free.astype(float),levels=[.5],colors=[COL['free']],linewidths=2)
    ax.set_title('Observed line-window moment-0 map and all extraction regions'); fig.colorbar(im,ax=ax,label='Integrated surface brightness'); ax.legend(loc='upper right')
    savefig(fig,f'{VERSION}_MOMENT0_ALL_REGIONS.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(snmap,origin='lower',cmap='magma',vmin=-3,vmax=max(5,np.nanpercentile(snmap,99)))
    ax.contour(snmap,levels=[2],colors=['#ffffff'],linewidths=1.2); ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2)
    ax.contour(anchored.astype(float),levels=[.5],colors=[COL['anch']],linewidths=2); ax.contour(free.astype(float),levels=[.5],colors=[COL['free']],linewidths=2)
    ax.set_title('S/N map: white = S/N 2 threshold, orange = target-seeded, red = free optimized'); fig.colorbar(im,ax=ax,label='Moment-0 S/N')
    savefig(fig,f'{VERSION}_SNMAP_MASK_SELECTION.png',figs)
    methods={'Fixed beam':beam_mask,'Target-seeded mask':anchored,'Free optimized mask':free}
    rows=[]; spec=[]
    for name,mask in methods.items():
        flux=np.array([np.nansum(im*beam_w)/np.nansum(beam_w) if name=='Fixed beam' else np.nansum(im[mask]) for im in cube5])
        sig=robust_rms(flux[~line_sel]); sn=np.sum(flux[line_sel])/(sig*np.sqrt(line_sel.sum())) if sig>0 else np.nan
        cen=np.sum(centers[line_sel]*flux[line_sel])/np.sum(flux[line_sel])
        rows.append({'method':name,'centroid_GHz':cen,'offset_MHz':(cen-TARGET_GHZ)*1000,'SNR_proxy':sn,'mask_pixels':int(mask.sum())})
        spec.append(pd.DataFrame({'method':name,'frequency_GHz':centers,'velocity_km_s':vel,'flux':flux,'line_window':line_sel}))
    summary=pd.DataFrame(rows); specdf=pd.concat(spec,ignore_index=True)
    scsv=OUT_CSV/f'{VERSION}_METHOD_SUMMARY.csv'; pcsv=OUT_CSV/f'{VERSION}_ALL_SPECTRA.csv'
    summary.to_csv(scsv,index=False); specdf.to_csv(pcsv,index=False)
    for p in (scsv,pcsv): (DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    fig,ax=plt.subplots(figsize=(12,7))
    for name,color in [('Fixed beam',COL['beam']),('Target-seeded mask',COL['anch']),('Free optimized mask',COL['free'])]:
        d=specdf[specdf.method==name]; ax.plot(d.frequency_GHz,d.flux,lw=1.7,label=name,color=color)
    ax.axvline(TARGET_GHZ,ls='--',c=COL['paper'],label='Paper 279.901 GHz'); ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.10,color='#ffffff')
    ax.set_title('Three extraction spectra from the same observed cube'); ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Extracted flux'); ax.grid(alpha=.25); ax.legend()
    savefig(fig,f'{VERSION}_THREE_METHOD_SPECTRA.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='92%'))
    img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(f'<b>{figs[0].name}</b>'); state={'i':0}
    prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(change):
        p=Path(change['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1))
    display(widgets.VBox([widgets.HTML('<h3>JADES-GS-z11-0 image-selection analysis gallery</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[scsv,pcsv]+figs
    print(f'CODE OUTPUT: {VERSION}'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
