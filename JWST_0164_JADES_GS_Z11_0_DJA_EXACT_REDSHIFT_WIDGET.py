# JWST_0164
import os, sys, warnings, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
warnings.filterwarnings('ignore')
for p in ['numpy','pandas','matplotlib','scipy','astropy','requests','ipywidgets']:
    try: __import__(p)
    except Exception: subprocess.run([sys.executable,'-m','pip','install','-q',p],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, requests
from astropy.io import fits
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION='JWST_0164'; ROOT=Path('/content/JWST_OUTPUT'); PNG=ROOT/'PNG'; CSV=ROOT/'CSV'; DATA=ROOT/'DATA'/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
TARGET='10014220'; ZP=11.38; LYA=0.121567; HEII=0.1640
BUCKET='https://s3.amazonaws.com/msaexp-nirspec'
PREFIXES=['extractions/gds-deep-v4/','extractions/gds-udeep-v4/']

def find_key():
    for prefix in PREFIXES:
        token=None
        for _ in range(30):
            params={'list-type':'2','prefix':prefix,'max-keys':'1000'}
            if token: params['continuation-token']=token
            r=requests.get(BUCKET,params=params,timeout=120); r.raise_for_status()
            root=ET.fromstring(r.content)
            ns={'s3':'http://s3.amazonaws.com/doc/2006-03-01/'}
            keys=[e.text for e in root.findall('.//s3:Key',ns)]
            hits=[k for k in keys if TARGET in k and 'prism-clear' in k.lower() and k.lower().endswith('.spec.fits')]
            if hits: return sorted(hits,key=lambda k:('combined' not in k.lower(),len(k)))[0]
            nxt=root.find('.//s3:NextContinuationToken',ns)
            if nxt is None or not nxt.text: break
            token=nxt.text
    raise RuntimeError('Exact DJA PRISM spectrum for JADES-GS-z11-0 (ID 10014220) was not found')

def download():
    cache=DATA/'JADES_GS_Z11_0_DJA_PRISM.spec.fits'
    if cache.exists() and cache.stat().st_size>50000: return cache
    key=find_key(); url=f'{BUCKET}/{key}'
    r=requests.get(url,timeout=180); r.raise_for_status(); cache.write_bytes(r.content)
    return cache

def read_spec(path):
    with fits.open(path,memmap=False) as hdul:
        for hdu in hdul:
            d=getattr(hdu,'data',None)
            if d is None or not hasattr(d,'names') or d.names is None: continue
            nm={n.lower():n for n in d.names}
            def pick(*ks):
                for k in ks:
                    if k in nm:
                        a=np.asarray(d[nm[k]],float).squeeze()
                        if a.ndim==1 and a.size>50: return a
            w=pick('wave','wavelength','lambda'); f=pick('flux','flux_corr','flam','fnu'); e=pick('err','error','flux_err','flux_error','sigma')
            if w is not None and f is not None and w.size==f.size:
                if e is None or e.size!=f.size: e=np.full_like(f,np.nan)
                return w,f,e
    raise RuntimeError('No usable 1D spectrum in the exact DJA FITS file')

path=download(); wave,flux,err=read_spec(path)
med=np.nanmedian(wave)
if med>1000: wave/=1e4
elif med>10: wave/=1e3
m=np.isfinite(wave)&np.isfinite(flux)&(wave>0.8)&(wave<5.35)
wave,flux,err=wave[m],flux[m],err[m]
o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]
wave,idx=np.unique(wave,return_index=True); flux,err=flux[idx],err[idx]
if np.nanmedian(np.abs(flux))<1e-8:
    conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-21; flux*=conv; err*=conv
g=np.isfinite(err)&(err>0)
if np.sum(g)<20:
    mad=np.nanmedian(np.abs(flux-np.nanmedian(flux))); err=np.full_like(flux,max(1.4826*mad,1e-6))
else: err[~g]=np.nanmedian(err[g])
fit=(wave>=1.15)&(wave<=2.30); x,y,s=wave[fit],flux[fit],err[fit]
if x.size<50: raise RuntimeError(f'Exact DJA spectrum has only {x.size} samples in the Lyman-break window')

def model(z,beta=-1.8,width=0.018):
    edge=LYA*(1+z); cont=(np.maximum(x,edge)/1.7)**beta; tr=.5*(1+np.tanh((x-edge)/width)); raw=cont*tr
    d=np.diff(x); d=d[np.isfinite(d)&(d>0)]; dl=np.median(d)
    sig=max(np.median(x)/(100*2.35482)/dl,.35)
    return gaussian_filter1d(raw,sig,mode='nearest')

def chi(z):
    u=model(z); w=1/s**2; den=np.sum(w*u*u)
    if not np.isfinite(den) or den<=0: return 1e99
    a=np.sum(w*y*u)/den; v=np.sum(((y-a*u)/s)**2)
    return float(v) if np.isfinite(v) else 1e99

res=minimize_scalar(chi,bounds=(10.7,11.8),method='bounded',options={'xatol':1e-5}); zbest=float(res.x)
zg=np.linspace(10.9,11.7,401); ch=np.array([chi(z) for z in zg]); p=np.exp(np.clip(-.5*(ch-ch.min()),-700,0)); p/=np.trapz(p,zg)
c=np.cumsum(p); c/=c[-1]; z16,z50,z84=[np.interp(q,c,zg) for q in (.16,.5,.84)]
u=model(zbest); w=1/s**2; amp=np.sum(w*y*u)/np.sum(w*u*u); best=amp*u; resid=(y-best)/s
lya=LYA*(1+zbest); heii=HEII*(1+zbest)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Replot exact DJA fit',button_style='success',layout=widgets.Layout(width='220px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,14)); gs=fig.add_gridspec(3,1,height_ratios=[.8,2.3,.8],hspace=.14); a0,a1,a2=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a0,a1,a2): a.grid(color='#303944',lw=.6,alpha=.75)
        a0.plot(zg,p,lw=2,label=f'Posterior z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})'); a0.axvline(zbest,lw=1.4); a0.axvline(ZP,ls='--',lw=1,label='Paper model z=11.38'); a0.axvspan(z16,z84,alpha=.14); a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False)
        a1.fill_between(x,y-s,y+s,step='mid',alpha=.30,label='1σ uncertainty'); a1.step(x,y,where='mid',lw=1,label='Exact DJA JADES PRISM data'); a1.plot(x,best,lw=2.5,label=f'Best Lyman-break fit z={zbest:.4f}'); a1.axvline(lya,ls='--',lw=1.4,label=f'Lyα anchor {lya:.5f} μm'); a1.axvline(LYA*(1+ZP),ls=':',lw=1.2,label=f'Paper z=11.38 → {LYA*(1+ZP):.5f} μm'); a1.axvline(heii,ls='--',lw=1,label=f'He II reference {heii:.5f} μm'); a1.set_xlim(1.15,2.30); q=np.nanpercentile(y,[1,99]); a1.set_ylim(q[0]-.2*(q[1]-q[0]),q[1]+.2*(q[1]-q[0])); a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=9)
        a2.axhline(0,lw=.6); a2.step(x,resid,where='mid',lw=.9); a2.axhspan(-1,1,alpha=.2,label='±1σ'); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_xlabel('Observed wavelength [μm]'); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False)
        fig.suptitle('JADES-GS-z11-0 — exact DJA Lyman-break redshift chain',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_DJA_EXACT_REDSHIFT_CHAIN.png',dpi=450,bbox_inches='tight')
        pd.DataFrame({'wavelength_um':x,'flux_1e-21':y,'sigma_1e-21':s,'model':best,'residual_sigma':resid}).to_csv(CSV/f'{VERSION}_DJA_EXACT_FIT.csv',index=False)
        plt.show()
run.on_click(draw); display(run,out); draw()
print(f'CODE OUTPUT: {VERSION}')
print(f'best_z={zbest:.6f}  Lyalpha_obs_um={lya:.6f}  HeII_obs_um={heii:.6f}')
print(f'plot={PNG/f"{VERSION}_DJA_EXACT_REDSHIFT_CHAIN.png"}')
print(f'# {VERSION}')
