from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, base64, warnings, gc, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import gaussian_filter
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0219'
MODEL='MODEL_B_JWST_MORPHOLOGY_TEMPLATE'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
TEMPLATE_DIR=Path('/content/drive/MyDrive/JWST/JADES_GS_Z11_0_JWST_TEMPLATE')
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG=ROOT/'PNG'; DRIVE_CSV=ROOT/'CSV'
for p in (TEMPLATE_DIR,OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d','text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8','xtick.color':'#c7d4e5','ytick.color':'#c7d4e5','grid.color':'#33485f'})
COL={'template':'#ffb74d','target':'#ffffff','paper':'#f5f7fa','peak':'#fdd835'}

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
        r=requests.put(f'https://api.github.com/repos/gear66me-ui/JWST/contents/{path}',headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'},json=payload,timeout=120)
        ok+=int(r.status_code in (200,201))
    return f'GitHub uploaded: {ok}/{len(files)} files'

def find_jwst_fits():
    pats=['*.fits','*.fit','*.fits.gz']
    candidates=[]
    for pat in pats:
        candidates += list(TEMPLATE_DIR.rglob(pat))
    if not candidates:
        search_root=Path('/content/drive/MyDrive/JWST')
        for pat in pats:
            for p in search_root.rglob(pat):
                s=str(p).lower()
                if 'alma' not in s and any(k in s for k in ('jades','z11','nircam','jwst')):
                    candidates.append(p)
    valid=[]
    target=SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg)
    for p in candidates:
        try:
            with fits.open(p,memmap=True) as hdul:
                for k,hdu in enumerate(hdul):
                    if hdu.data is None or np.ndim(hdu.data)<2: continue
                    w=WCS(hdu.header).celestial
                    if w.pixel_n_dim!=2: continue
                    x,y=w.world_to_pixel(target)
                    ny,nx=hdu.data.shape[-2:]
                    if -5<x<nx+5 and -5<y<ny+5:
                        pix=np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix)))*3600
                        valid.append((pix,p,k))
        except Exception:
            pass
    if not valid:
        raise FileNotFoundError('No astrometrically valid JWST FITS image found. Place a JWST/NIRCam FITS cutout containing RA=53.1647632 Dec=-27.7746223 in '+str(TEMPLATE_DIR))
    valid.sort(key=lambda z:z[0])
    return valid[0]

def ensure_reproject():
    try:
        from reproject import reproject_interp
        return reproject_interp
    except Exception:
        subprocess.check_call([sys.executable,'-m','pip','install','-q','reproject'])
        from reproject import reproject_interp
        return reproject_interp

def shifted_template(template,dy,dx):
    return np.roll(np.roll(template,dy,axis=0),dx,axis=1)

def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    jwst_pix,jwst_path,jwst_ext=find_jwst_fits()
    reproject_interp=ensure_reproject()
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        w=WCS(h).celestial; tx0,ty0=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(np.asarray(tx0).reshape(-1)[0])); ty=int(np.rint(np.asarray(ty0).reshape(-1)[0]))
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
        alma_cut_wcs=w.slice((ys,xs))
        del data; gc.collect()
    with fits.open(jwst_path,memmap=True) as hdul:
        jhdu=hdul[jwst_ext]; jdata=np.asarray(jhdu.data,dtype=float)
        while jdata.ndim>2:jdata=np.nanmean(jdata,axis=0)
        jwcs=WCS(jhdu.header).celestial
        reproj,foot=reproject_interp((jdata,jwcs),alma_cut_wcs,shape_out=(ny,nx))
    good=np.isfinite(reproj)&(foot>0)
    bg=np.nanmedian(reproj[good]) if np.any(good) else 0.0
    raw=np.where(good,reproj-bg,0.0)
    raw=np.clip(raw,0,None)
    if np.sum(raw)<=0: raise RuntimeError('JWST template is empty after reprojection/background subtraction.')
    sigma_x=max((bmaj/2.35482)/pix,0.5); sigma_y=max((bmin/2.35482)/pix,0.5)
    sigma=np.sqrt(sigma_x*sigma_y)
    conv=gaussian_filter(raw,sigma=sigma,mode='constant')
    yy,xx=np.indices(conv.shape); rr=np.hypot((xx-txc)*pix,(yy-tyc)*pix)
    conv[rr>1.5]=0
    template=conv/np.sqrt(np.sum(conv**2))
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    amps=np.array([np.nansum(im*template) for im in cube])
    blank=[]
    for rad_pix in (18,24,30,36,42):
        for ang in np.linspace(0,2*np.pi,16,endpoint=False):
            dxs=int(round(rad_pix*np.cos(ang))); dys=int(round(rad_pix*np.sin(ang)))
            blank.append([np.nansum(im*shifted_template(template,dys,dxs)) for im in cube])
    blank=np.asarray(blank)
    noise_chan=np.array([robust_rms(blank[:,j]) for j in range(len(centers))])
    line_flux=np.sum(amps[line_sel]); line_noise=robust_rms(np.sum(blank[:,line_sel],axis=1)); line_snr=line_flux/line_noise
    positive=np.clip(amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*positive)/np.sum(positive) if np.sum(positive)>0 else np.nan
    moment=np.sum(cube[line_sel],axis=0)
    rows=pd.DataFrame({'channel_index':np.arange(len(centers)),'frequency_GHz':centers,'velocity_km_s':vel,'template_amplitude':amps,'noise_sigma':noise_chan,'channel_SNR':amps/noise_chan,'line_window':line_sel})
    summary=pd.DataFrame([{'model':'Model B — JWST morphology template','jwst_file':str(jwst_path),'jwst_extension':jwst_ext,'jwst_pixel_scale_arcsec':jwst_pix,'RA_deg':RA_DEG,'Dec_deg':DEC_DEG,'line_channels':int(line_sel.sum()),'line_flux_sum':line_flux,'line_noise_sigma':line_noise,'integrated_SNR':line_snr,'centroid_GHz':centroid,'offset_MHz':(centroid-TARGET_GHZ)*1000 if np.isfinite(centroid) else np.nan,'beam_major_arcsec':bmaj,'beam_minor_arcsec':bmin}])
    rcsv=OUT_CSV/f'{VERSION}_MODEL_B_CHANNELS.csv'; scsv=OUT_CSV/f'{VERSION}_MODEL_B_SUMMARY.csv'
    rows.to_csv(rcsv,index=False); summary.to_csv(scsv,index=False)
    for p in (rcsv,scsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(raw,origin='lower',cmap='magma',interpolation='nearest'); ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2,label='JWST coordinate'); ax.set_title('MODEL B — reprojected JWST morphology before ALMA convolution'); ax.legend(); fig.colorbar(im,ax=ax,label='JWST relative surface brightness'); savefig(fig,f'{VERSION}_MODEL_B_JWST_REPROJECTED.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(template,origin='lower',cmap='magma',interpolation='nearest'); ax.contour(template,levels=[0.2*template.max(),0.5*template.max(),0.8*template.max()],colors=[COL['template']]*3,linewidths=[1,1.5,2]); ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2,label='Exact JWST coordinate'); ax.set_title('MODEL B — JWST morphology convolved to ALMA resolution'); ax.legend(); fig.colorbar(im,ax=ax,label='Normalized template weight'); savefig(fig,f'{VERSION}_MODEL_B_ALMA_CONVOLVED_TEMPLATE.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest'); ax.contour(template,levels=[0.2*template.max(),0.5*template.max(),0.8*template.max()],colors=[COL['template']]*3,linewidths=[1,1.5,2]); ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2,label='JWST coordinate'); ax.set_title(f'MODEL B — ALMA moment-0 with JWST morphology template | S/N={line_snr:.3f}'); ax.legend(); fig.colorbar(im,ax=ax,label='Integrated ALMA surface brightness'); savefig(fig,f'{VERSION}_MODEL_B_TEMPLATE_ON_ALMA_MOMENT0.png',figs)
    fig,ax=plt.subplots(figsize=(12,7)); ax.plot(centers,amps,lw=1.8,label='JWST morphology-template amplitude'); ax.fill_between(centers,-noise_chan,noise_chan,alpha=.18,label='±1σ blank-position noise'); ax.axvline(TARGET_GHZ,ls='--',c=COL['paper'],label='Reference 279.901 GHz'); ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.10); ax.set_title(f'MODEL B spectrum — JWST morphology prior | integrated S/N={line_snr:.3f}'); ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Matched-template amplitude'); ax.grid(alpha=.25); ax.legend(); savefig(fig,f'{VERSION}_MODEL_B_TEMPLATE_SPECTRUM.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='92%')); img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(); state={'i':0}; prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1)); show(0)
    display(widgets.VBox([widgets.HTML('<h3>Model B — JWST morphology template matched to ALMA cube</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[rcsv,scsv]+figs
    print(f'CODE OUTPUT: {VERSION}'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print(f'Images in gallery: {len(figs)}'); print(f'JWST template: {jwst_path} [HDU {jwst_ext}]'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
