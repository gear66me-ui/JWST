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

VERSION='JWST_0210'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
N_MC=300
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

def world_axis(h,ax):
    n=int(h[f'NAXIS{ax}']); pix=np.arange(n)+1.0
    v=float(h[f'CRVAL{ax}'])+(pix-float(h[f'CRPIX{ax}']))*float(h.get(f'CDELT{ax}',h.get(f'CD{ax}_{ax}')))
    u=str(h.get(f'CUNIT{ax}','Hz')).lower()
    if u=='hz':v/=1e9
    elif u=='khz':v/=1e6
    elif u=='mhz':v/=1e3
    return v

def spec_axis(h):
    for ax in range(1,int(h['NAXIS'])+1):
        if 'FREQ' in str(h.get(f'CTYPE{ax}','')).upper(): return ax
    raise RuntimeError('No frequency axis')

def collapse(arr,np_spec):
    arr=np.moveaxis(arr,np_spec,0)
    while arr.ndim>3: arr=np.nanmean(arr,axis=1)
    return arr

def connected(mask,seed):
    labs,n=label(mask); sy,sx=seed
    if n==0:return np.zeros_like(mask)
    q=labs[sy,sx]
    return labs==q if q>0 else np.zeros_like(mask)

def centroid(freq,flux,sel):
    f=freq[sel]; s=flux[sel]; d=np.nansum(s)
    return np.nansum(f*s)/d if np.isfinite(d) and d!=0 else np.nan

def upload(files):
    token=os.environ.get('GITHUB_TOKEN')
    if not token:
        try:
            from google.colab import userdata
            token=userdata.get('GITHUB_TOKEN')
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
        w=WCS(h).celestial
        tx0,ty0=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(np.asarray(tx0).reshape(-1)[0])); ty=int(np.rint(np.asarray(ty0).reshape(-1)[0]))
        pix=abs(float(h['CDELT2']))*3600; bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600; bpa=float(h.get('BPA',0))
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half); ny=128; nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq_native)))*1000; k=max(1,int(round(5.0/native)))
        near=int(np.argmin(np.abs(freq_native-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq_native),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq_native[g]) for g in groups]); cube5=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            arr=collapse(np.asarray(data[tuple(sl)],np.float32),np_spec)
            cube5[j]=np.nanmean(arr,axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pix; dy=(yy-tyc)*pix
    th=np.deg2rad(bpa); xr=dx*np.cos(th)+dy*np.sin(th); yr=-dx*np.sin(th)+dy*np.cos(th)
    sx=max(bmaj/2.35482,0.15); sy=max(bmin/2.35482,0.12)
    beam_w=np.exp(-0.5*((xr/sx)**2+(yr/sy)**2)); beam_mask=beam_w>=np.exp(-0.5)
    rr=np.hypot(dx,dy); off=rr>1.2
    rms_chan=np.array([robust_rms(im[off]) for im in cube5])
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
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
    methods={'Fixed beam':beam_mask,'Target-seeded mask':anchored,'Free optimized mask':free}
    def extract(cube,mask,name):
        if name=='Fixed beam': return np.array([np.nansum(im*beam_w)/np.nansum(beam_w) for im in cube])
        return np.array([np.nansum(im[mask]) for im in cube])
    base_flux={k:extract(cube5,m,k) for k,m in methods.items()}
    rows=[]; rng=np.random.default_rng(20260718)
    for i in range(N_MC):
        noise=np.empty_like(cube5)
        for ch in range(len(centers)):
            noise[ch]=rng.normal(0,rms_chan[ch],size=(ny,nx)).astype(np.float32)
        work=cube5+noise
        mm=np.sum(work[line_sel],axis=0); mr=robust_rms(mm[off]); sm=mm/mr
        a=connected((sm>=2)&(rr<=0.8),(tyc,txc)); a=a if a.any() else beam_mask.copy()
        c=(sm>=2)&(rr<=0.8); ll,nn=label(c); f=np.zeros_like(c)
        if nn:
            z=[]
            for q in range(1,nn+1):
                m=ll==q
                if m.any(): z.append((np.sum(sm[m]),m))
            if z:f=max(z,key=lambda t:t[0])[1]
        if not f.any():f=beam_mask.copy()
        iter_masks={'Fixed beam':beam_mask,'Target-seeded mask':a,'Free optimized mask':f}
        for name,mask in iter_masks.items():
            flux=extract(work,mask,name); sig=robust_rms(flux[~line_sel]); sn=np.sum(flux[line_sel])/(sig*np.sqrt(line_sel.sum())) if sig>0 else np.nan
            rows.append({'iteration':i+1,'method':name,'centroid_GHz':centroid(centers,flux,line_sel),'SNR_proxy':sn,'mask_pixels':int(mask.sum())})
        if (i+1)%50==0 or i==N_MC-1: print(f'Monte Carlo: {i+1:3d}/{N_MC:3d}')
    mc=pd.DataFrame(rows)
    observed=[]
    for name,flux in base_flux.items():
        sig=robust_rms(flux[~line_sel]); sn=np.sum(flux[line_sel])/(sig*np.sqrt(line_sel.sum()))
        observed.append({'method':name,'centroid_GHz':centroid(centers,flux,line_sel),'SNR_proxy':sn,'mask_pixels':int(methods[name].sum())})
    obs=pd.DataFrame(observed)
    summary=mc.groupby('method').agg(centroid_mean_GHz=('centroid_GHz','mean'),centroid_std_GHz=('centroid_GHz','std'),SNR_median=('SNR_proxy','median'),SNR_p16=('SNR_proxy',lambda x:np.nanpercentile(x,16)),SNR_p84=('SNR_proxy',lambda x:np.nanpercentile(x,84))).reset_index().merge(obs,on='method',suffixes=('_MC','_observed'))
    mcsv=OUT_CSV/f'{VERSION}_REAL_NOISE_MONTE_CARLO.csv'; ocsv=OUT_CSV/f'{VERSION}_OBSERVED_METHODS.csv'; scsv=OUT_CSV/f'{VERSION}_SUMMARY.csv'
    mc.to_csv(mcsv,index=False); obs.to_csv(ocsv,index=False); summary.to_csv(scsv,index=False)
    for p in (mcsv,ocsv,scsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(10,9)); im=ax.imshow(moment0,origin='lower',cmap='viridis'); ax.scatter([txc],[tyc],marker='+',s=170,c=COL['target'],linewidths=2)
    ax.contour(anchored.astype(float),levels=[.5],colors=[COL['anch']],linewidths=2); ax.contour(free.astype(float),levels=[.5],colors=[COL['free']],linewidths=2)
    ax.add_patch(Ellipse((txc,tyc),bmaj/pix,bmin/pix,angle=bpa,fill=False,ec=COL['beam'],lw=2)); ax.set_title('Observed moment-0 map and three extraction regions'); fig.colorbar(im,ax=ax); fig.tight_layout()
    p=OUT_PNG/f'{VERSION}_MASK_COMPARISON.png'; fig.savefig(p,dpi=190); plt.close(fig); figs.append(p); (DRIVE_PNG/p.name).write_bytes(p.read_bytes())
    for name,color in [('Fixed beam',COL['beam']),('Target-seeded mask',COL['anch']),('Free optimized mask',COL['free'])]:
        fig,ax=plt.subplots(figsize=(12,7)); d=mc[mc.method==name]
        ax.hist(d.centroid_GHz.dropna(),bins=35,alpha=.78); ax.axvline(TARGET_GHZ,ls='--',c=COL['paper'],label='Paper 279.901 GHz'); ax.axvline(obs.loc[obs.method==name,'centroid_GHz'].iloc[0],ls=':',c=color,label='Observed extraction')
        ax.set_title(f'{name} — real-noise centroid Monte Carlo'); ax.set_xlabel('Centroid frequency (GHz)'); ax.set_ylabel('Count'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
        p=OUT_PNG/f"{VERSION}_{name.upper().replace(' ','_').replace('-','_')}_CENTROID_MC.png"; fig.savefig(p,dpi=190); plt.close(fig); figs.append(p); (DRIVE_PNG/p.name).write_bytes(p.read_bytes())
    fig,ax=plt.subplots(figsize=(12,7))
    for name,color in [('Fixed beam',COL['beam']),('Target-seeded mask',COL['anch']),('Free optimized mask',COL['free'])]:
        d=mc[mc.method==name]; ax.hist(d.SNR_proxy.dropna(),bins=35,histtype='step',lw=2,label=name,color=color)
        ax.axvline(obs.loc[obs.method==name,'SNR_proxy'].iloc[0],ls=':',c=color)
    ax.axvline(4.5,ls='--',c=COL['paper'],label='Paper nominal 4.5'); ax.set_title('Real-noise Monte Carlo S/N distributions'); ax.set_xlabel('Integrated S/N proxy'); ax.set_ylabel('Count'); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    p=OUT_PNG/f'{VERSION}_SNR_MONTE_CARLO_COMPARISON.png'; fig.savefig(p,dpi=190); plt.close(fig); figs.append(p); (DRIVE_PNG/p.name).write_bytes(p.read_bytes())
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='92%')); img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(f'<b>{figs[0].name}</b>'); state={'i':0}
    prev=widgets.Button(description='Previous',icon='arrow-left'); nxt=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(lambda c: show([str(p) for p in figs].index(c['new'])) if c['name']=='value' else None,names='value'); prev.on_click(lambda b:show(state['i']-1)); nxt.on_click(lambda b:show(state['i']+1))
    display(widgets.VBox([widgets.HTML('<h3>JADES-GS-z11-0 real-noise Monte Carlo gallery</h3>'),dd,widgets.HBox([prev,nxt]),cap,img]))
    files=[mcsv,ocsv,scsv]+figs; print(f'CODE OUTPUT: {VERSION}'); print(obs.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
