# JWST_0162
import os, io, sys, warnings, subprocess
from pathlib import Path
warnings.filterwarnings('ignore')
for pkg in ['numpy','pandas','matplotlib','scipy','astropy','requests','ipywidgets','lxml']:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, requests
from astropy.io import fits
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION='JWST_0162'; ROOT=Path('/content/JWST_OUTPUT'); PNG=ROOT/'PNG'; CSV=ROOT/'CSV'; DATA=ROOT/'DATA'/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
RA,DEC=53.1647632,-27.7746223; ZP=11.38; LYA=0.121567; HEII=0.1640
INDEX='https://s3.amazonaws.com/msaexp-nirspec/extractions/public_prelim_v4.2.html'
BASE='https://s3.amazonaws.com/msaexp-nirspec/extractions/'

def get_public_fits():
    cache=DATA/'JADES_GS_Z11_0_PUBLIC_DR3.spec.fits'
    if cache.exists() and cache.stat().st_size>50000: return cache
    tables=pd.read_html(INDEX)
    tab=max(tables,key=len)
    cols={str(c).strip().lower():c for c in tab.columns}
    if 'ra' not in cols or 'dec' not in cols: raise RuntimeError('Public DR3 index lacks RA/Dec columns')
    ra=pd.to_numeric(tab[cols['ra']],errors='coerce'); dec=pd.to_numeric(tab[cols['dec']],errors='coerce')
    sep=((ra-RA)*np.cos(np.deg2rad(DEC)))**2+(dec-DEC)**2
    order=np.argsort(np.asarray(sep,float))
    fits_col=None
    for key in ('fits','file','root'):
        if key in cols: fits_col=cols[key]; break
    if fits_col is None: raise RuntimeError('Public DR3 index lacks FITS/file column')
    url=None
    for i in order[:20]:
        val=str(tab.iloc[int(i)][fits_col])
        if '.fits' in val.lower():
            if val.startswith('http'): url=val
            else: url=BASE+val.lstrip('./')
            break
    if url is None: raise RuntimeError('No public FITS link near JADES-GS-z11-0 coordinates')
    r=requests.get(url,timeout=180); r.raise_for_status(); cache.write_bytes(r.content)
    return cache

def read_spec(path):
    with fits.open(path,memmap=False) as hdul:
        for hdu in hdul:
            d=getattr(hdu,'data',None)
            if d is None or not hasattr(d,'names') or d.names is None: continue
            nm={n.lower():n for n in d.names}
            def pick(keys):
                for k in keys:
                    if k in nm:
                        a=np.asarray(d[nm[k]],float).squeeze()
                        if a.ndim==1 and a.size>20: return a
                return None
            w=pick(['wave','wavelength','lambda']); f=pick(['flux','flux_corr','flam','fnu']); e=pick(['err','error','flux_err','flux_error','sigma'])
            if w is not None and f is not None and w.size==f.size:
                if e is None or e.size!=f.size: e=np.full_like(f,np.nan)
                return w,f,e
    raise RuntimeError('No usable 1D spectrum in public FITS')

path=get_public_fits(); wave,flux,err=read_spec(path)
med=np.nanmedian(wave)
if med>1000: wave/=1e4
elif med>10: wave/=1e3
m=np.isfinite(wave)&np.isfinite(flux)&(wave>0.8)&(wave<5.35)
wave,flux,err=wave[m],flux[m],err[m]
o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]
wave,idx=np.unique(wave,return_index=True); flux,err=flux[idx],err[idx]
if wave.size<100: raise RuntimeError(f'Public spectrum has only {wave.size} finite samples; wrong product rejected')
if np.nanmedian(np.abs(flux))<1e-8:
    conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-21; flux*=conv; err*=conv
g=np.isfinite(err)&(err>0)
if np.sum(g)<20:
    mad=np.nanmedian(np.abs(flux-np.nanmedian(flux))); err=np.full_like(flux,max(1.4826*mad,1e-6))
else: err[~g]=np.nanmedian(err[g])
fit=(wave>1.15)&(wave<2.30); x,y,s=wave[fit],flux[fit],err[fit]

def model(z,beta=-1.8,width=0.018):
    edge=LYA*(1+z); cont=(np.maximum(x,edge)/1.7)**beta; tr=.5*(1+np.tanh((x-edge)/width)); raw=cont*tr
    dl=np.median(np.diff(x)); sig=max(np.median(x)/(100*2.35482)/dl,.35); return gaussian_filter1d(raw,sig,mode='nearest')
def chi(z):
    u=model(z); w=1/s**2; a=np.sum(w*y*u)/np.sum(w*u*u); return np.sum(((y-a*u)/s)**2)
res=minimize_scalar(chi,bounds=(10.7,11.8),method='bounded',options={'xatol':1e-5}); zbest=float(res.x)
zg=np.linspace(10.9,11.7,401); ch=np.array([chi(z) for z in zg]); p=np.exp(np.clip(-.5*(ch-ch.min()),-700,0)); p/=np.trapz(p,zg)
c=np.cumsum(p); c/=c[-1]; z16,z50,z84=[np.interp(q,c,zg) for q in (.16,.5,.84)]
u=model(zbest); w=1/s**2; amp=np.sum(w*y*u)/np.sum(w*u*u); best=amp*u; resid=(y-best)/s
lya=LYA*(1+zbest); heii=HEII*(1+zbest)
plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Replot public DR3 fit',button_style='success',layout=widgets.Layout(width='230px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,14)); gs=fig.add_gridspec(3,1,height_ratios=[.8,2.3,.8],hspace=.14); a0,a1,a2=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a0,a1,a2): a.grid(color='#303944',lw=.6,alpha=.75)
        a0.plot(zg,p,color='#62dfd1',lw=2,label=f'Public-spectrum posterior z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})'); a0.axvline(zbest,color='#ffd400',lw=1.4); a0.axvline(ZP,color='#ff6b6b',ls='--',lw=1,label='Paper model z=11.38'); a0.axvspan(z16,z84,color='#62dfd1',alpha=.14); a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False)
        a1.fill_between(x,y-s,y+s,step='mid',color='gray',alpha=.32,label='1σ uncertainty'); a1.step(x,y,where='mid',color='#4f79b9',lw=1,label='Public JADES PRISM data'); a1.plot(x,best,color='#e68645',lw=2.5,label=f'Best Lyman-break fit z={zbest:.4f}'); a1.axvline(lya,color='#ffd400',ls='--',lw=1.4,label=f'Lyα anchor {lya:.5f} μm'); a1.axvline(LYA*(1+ZP),color='#ff6b6b',ls=':',lw=1.2,label=f'Paper z=11.38 → {LYA*(1+ZP):.5f} μm'); a1.axvline(heii,color='#ff4d4d',ls='--',lw=1,label=f'He II reference {heii:.5f} μm'); a1.set_xlim(1.15,2.30); q=np.nanpercentile(y,[1,99]); a1.set_ylim(q[0]-.2*(q[1]-q[0]),q[1]+.2*(q[1]-q[0])); a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=9)
        a2.axhline(0,color='white',lw=.6); a2.step(x,resid,where='mid',color='#62dfd1',lw=.9); a2.axhspan(-1,1,color='gray',alpha=.2,label='±1σ'); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_xlabel('Observed wavelength [μm]'); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False)
        fig.suptitle('JADES-GS-z11-0 — public DR3 Lyman-break redshift chain',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_PUBLIC_DR3_REDSHIFT_CHAIN.png',dpi=450,bbox_inches='tight')
        pd.DataFrame({'wavelength_um':x,'flux_1e-21':y,'sigma_1e-21':s,'model':best,'residual_sigma':resid}).to_csv(CSV/f'{VERSION}_PUBLIC_DR3_FIT.csv',index=False)
        plt.show()
run.on_click(draw); display(run,out); draw()
