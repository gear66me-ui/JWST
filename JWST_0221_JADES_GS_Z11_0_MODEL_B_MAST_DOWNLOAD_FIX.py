from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, sys, subprocess, base64, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import gaussian_filter
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0221'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
JWST_DIR=Path('/content/drive/MyDrive/JWST/JADES_GS_Z11_0_JWST_TEMPLATE')
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG=ROOT/'PNG'; DRIVE_CSV=ROOT/'CSV'
for p in (JWST_DIR,OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d','text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8','xtick.color':'#c7d4e5','ytick.color':'#c7d4e5','grid.color':'#33485f'})
COL={'jwst':'#ffb74d','target':'#ffffff','paper':'#f5f7fa'}

def ensure(pkg,mod=None):
    try: __import__(mod or pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

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

def valid_science_hdu(path):
    try:
        with fits.open(path,memmap=True) as hdul:
            for i,hdu in enumerate(hdul):
                if hdu.data is None or np.ndim(hdu.data)!=2: continue
                try:
                    w=WCS(hdu.header).celestial; x,y=w.world_to_pixel_values(RA_DEG,DEC_DEG)
                    if np.isfinite(x) and np.isfinite(y) and 2<=x<hdu.data.shape[1]-2 and 2<=y<hdu.data.shape[0]-2:return i
                except Exception: pass
    except Exception: pass
    return None

def get_jwst_fits():
    for p in sorted(JWST_DIR.rglob('*.fits')):
        ext=valid_science_hdu(p)
        if ext is not None:return p,ext
    ensure('astroquery')
    from astroquery.mast import Observations
    coord=SkyCoord(RA_DEG,DEC_DEG,unit='deg')
    obs=Observations.query_region(coord,radius=25*u.arcsec)
    obs=obs[np.array([str(x).upper()=='JWST' for x in obs['obs_collection']])]
    pref={'F444W':0,'F356W':1,'F277W':2,'F200W':3,'F150W':4}
    ranked=[]
    for row in obs:
        inst=str(row['instrument_name']).upper(); filt=str(row['filters']).upper(); cal=int(row['calib_level'] or 0)
        if 'NIRCAM' in inst and cal>=3: ranked.append((pref.get(filt,20),row))
    ranked.sort(key=lambda z:z[0])
    if not ranked: raise RuntimeError('No public JWST/NIRCam level-3 observations found')
    for _,row in ranked[:20]:
        products=Observations.get_product_list(row)
        good=[]
        for idx,pr in enumerate(products):
            fn=str(pr['productFilename']); rights=str(pr['dataRights']).upper(); typ=str(pr['productType']).upper()
            if rights=='PUBLIC' and typ=='SCIENCE' and (fn.endswith('_i2d.fits') or fn.endswith('_cal.fits')):
                size=int(pr['size']) if pr['size'] not in (None,'') else 10**18
                good.append((0 if fn.endswith('_i2d.fits') else 1,size,idx))
        good.sort()
        for _,_,idx in good[:4]:
            one=products[[idx]]
            manifest=Observations.download_products(one,download_dir=str(JWST_DIR),mrp_only=False,cache=True)
            if len(manifest)==0: continue
            local=Path(str(manifest['Local Path'][0]))
            if not local.exists(): continue
            dest=JWST_DIR/local.name
            if local!=dest and not dest.exists(): dest.write_bytes(local.read_bytes())
            ext=valid_science_hdu(dest)
            if ext is not None:return dest,ext
    raise FileNotFoundError('Could not download an astrometrically valid public JWST image from MAST')

def normalized_positive(a):
    a=np.asarray(a,float); f=np.isfinite(a); bg=np.nanmedian(a[f]); s=robust_rms(a[f]-bg)
    out=np.clip(a-bg,0,None)
    if s>0: out[out<1.5*s]=0
    sm=np.sum(out)
    return out/sm if sm>0 else np.zeros_like(out)

def main():
    warnings.filterwarnings('ignore'); ensure('reproject'); from reproject import reproject_interp
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    jwst_path,jwst_ext=get_jwst_fits()
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data; fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        aw=WCS(h).celestial; tx0,ty0=aw.world_to_pixel_values(RA_DEG,DEC_DEG); tx=int(np.rint(tx0)); ty=int(np.rint(ty0))
        pix=abs(float(h['CDELT2']))*3600; bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half); ny=nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq)))*1000; k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups]); cube=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        cut_header=aw.slice((ys,xs)).to_header(); del data; gc.collect()
    with fits.open(jwst_path,memmap=True) as jh:
        jdata=np.asarray(jh[jwst_ext].data,float); jhdr=jh[jwst_ext].header
        jwst_reproj,_=reproject_interp((jdata,jhdr),cut_header,shape_out=(ny,nx))
    raw=normalized_positive(jwst_reproj)
    sigma=np.sqrt(max((bmaj/2.35482)/pix,.5)*max((bmin/2.35482)/pix,.5))
    conv=gaussian_filter(raw,sigma=sigma,mode='constant')
    if np.sum(conv)<=0: raise RuntimeError('JWST template became empty after background subtraction')
    conv/=np.sqrt(np.sum(conv**2))
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ; line_sel=(vel>=-14)&(vel<=29)
    amps=np.array([np.nansum(im*conv) for im in cube])
    blank=[]
    for r in (18,24,30,36,42):
        for a in np.linspace(0,2*np.pi,16,endpoint=False):
            dx=int(round(r*np.cos(a))); dy=int(round(r*np.sin(a))); t=np.roll(np.roll(conv,dy,0),dx,1)
            blank.append([np.nansum(im*t) for im in cube])
    blank=np.asarray(blank); noise=np.array([robust_rms(blank[:,j]) for j in range(len(centers))])
    line_flux=np.sum(amps[line_sel]); line_noise=robust_rms(np.sum(blank[:,line_sel],axis=1)); line_snr=line_flux/line_noise
    pos=np.clip(amps[line_sel],0,None); centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    moment=np.sum(cube[line_sel],axis=0)
    summary=pd.DataFrame([{'model':'Model B — JWST morphology template','jwst_file':jwst_path.name,'jwst_hdu':jwst_ext,'line_channels':int(line_sel.sum()),'line_flux_sum':line_flux,'line_noise_sigma':line_noise,'integrated_SNR':line_snr,'centroid_GHz':centroid,'offset_MHz':(centroid-TARGET_GHZ)*1000,'beam_major_arcsec':bmaj,'beam_minor_arcsec':bmin}])
    channels=pd.DataFrame({'channel_index':np.arange(len(centers)),'frequency_GHz':centers,'velocity_km_s':vel,'template_amplitude':amps,'noise_sigma':noise,'channel_SNR':amps/noise,'line_window':line_sel})
    scsv=OUT_CSV/f'{VERSION}_MODEL_B_SUMMARY.csv'; ccsv=OUT_CSV/f'{VERSION}_MODEL_B_CHANNELS.csv'; summary.to_csv(scsv,index=False); channels.to_csv(ccsv,index=False)
    for p in (scsv,ccsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(jwst_reproj,origin='lower',cmap='magma',interpolation='nearest'); ax.scatter([txc],[tyc],marker='+',s=190,c=COL['target'],linewidths=2,label='JWST catalog coordinate'); ax.set_title(f'JWST image reprojected to ALMA 128×128 frame\n{jwst_path.name} [HDU {jwst_ext}]'); ax.legend(); fig.colorbar(im,ax=ax,label='JWST image units'); savefig(fig,f'{VERSION}_JWST_128X128_REPROJECTED.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(conv,origin='lower',cmap='magma',interpolation='nearest'); ax.scatter([txc],[tyc],marker='+',s=190,c=COL['target'],linewidths=2,label='Exact JWST coordinate'); ax.set_title('MODEL B — beam-convolved JWST morphology template'); ax.legend(); fig.colorbar(im,ax=ax,label='Normalized template weight'); savefig(fig,f'{VERSION}_MODEL_B_CONVOLVED_TEMPLATE.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest'); ax.contour(conv/np.nanmax(conv),levels=[.25,.5,.75],colors=[COL['jwst']],linewidths=[1,1.6,2]); ax.scatter([txc],[tyc],marker='+',s=190,c=COL['target'],linewidths=2,label='JWST coordinate'); ax.set_title(f'MODEL B — JWST template over ALMA moment-0 | S/N={line_snr:.3f}'); ax.legend(); fig.colorbar(im,ax=ax,label='ALMA moment-0 brightness'); savefig(fig,f'{VERSION}_MODEL_B_TEMPLATE_OVER_ALMA_MOMENT0.png',figs)
    fig,ax=plt.subplots(figsize=(12,7)); ax.plot(centers,amps,lw=1.8,label='JWST morphology-template amplitude'); ax.fill_between(centers,-noise,noise,alpha=.18,label='±1σ blank-position noise'); ax.axvline(TARGET_GHZ,ls='--',c=COL['paper'],label='Reference 279.901 GHz'); ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.10); ax.set_title(f'MODEL B spectrum | integrated S/N={line_snr:.3f}'); ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Matched-template amplitude'); ax.grid(alpha=.25); ax.legend(); savefig(fig,f'{VERSION}_MODEL_B_SPECTRUM.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='92%')); img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(); state={'i':0}; prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1)); show(0)
    display(widgets.VBox([widgets.HTML('<h3>Model B — JWST morphology template</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[scsv,ccsv]+figs
    print(f'CODE OUTPUT: {VERSION}'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print(f'JWST image: {jwst_path} [HDU {jwst_ext}]'); print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
