# JWST_0163
import os, io, sys, warnings, contextlib, subprocess
from pathlib import Path
warnings.filterwarnings('ignore')
for pkg in ['numpy','pandas','matplotlib','scipy','astropy','astroquery','ipywidgets']:
    try: __import__(pkg)
    except Exception:
        subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION='JWST_0163'
ROOT=Path('/content/JWST_OUTPUT'); PNG=ROOT/'PNG'; CSV=ROOT/'CSV'; DATA=ROOT/'DATA'/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
RA,DEC=53.1647632,-27.7746223
Z_PAPER=11.38; LYA=0.121567; HEII=0.1640

def vectors_from_hdul(path):
    out=[]
    with fits.open(path,memmap=False) as hdul:
        for hdu in hdul:
            d=getattr(hdu,'data',None)
            if d is None: continue
            if hasattr(d,'names') and d.names:
                nm={n.lower():n for n in d.names}
                def get(keys):
                    for k in keys:
                        if k in nm:
                            a=np.asarray(d[nm[k]],float)
                            a=np.ravel(a)
                            if a.size>20: return a
                    return None
                w=get(['wavelength','wave','lambda','lam'])
                f=get(['flux','flux_corr','flam','fnu','data'])
                e=get(['error','err','flux_error','flux_err','sigma'])
                if w is not None and f is not None and w.size==f.size:
                    if e is None or e.size!=f.size: e=np.full_like(f,np.nan)
                    out.append((w,f,e,str(getattr(hdu,'name',''))))
            elif isinstance(d,np.ndarray) and d.ndim==2 and min(d.shape)>=3:
                arr=np.asarray(d,float)
                if arr.shape[0] in (3,4,5): arr=arr.T
                if arr.shape[1]>=3:
                    w,f,e=arr[:,0],arr[:,1],arr[:,2]
                    out.append((w,f,e,str(getattr(hdu,'name',''))))
    return out

def normalize_candidate(w,f,e):
    w=np.asarray(w,float); f=np.asarray(f,float); e=np.asarray(e,float)
    med=np.nanmedian(w)
    if med>1000: w=w/1e4
    elif med>10: w=w/1e3
    m=np.isfinite(w)&np.isfinite(f)&(w>0.55)&(w<5.6)
    w,f,e=w[m],f[m],e[m]
    if w.size<50: return None
    o=np.argsort(w); w,f,e=w[o],f[o],e[o]
    w,idx=np.unique(w,return_index=True); f,e=f[idx],e[idx]
    if w.size<50 or np.nanmax(w)-np.nanmin(w)<0.8: return None
    if np.nanmedian(np.abs(f))<1e-8:
        conv=1e-6*1e-23*2.99792458e18/(w*1e4)**2/1e-21
        f=f*conv; e=e*conv
    g=np.isfinite(e)&(e>0)
    if np.sum(g)<20:
        mad=np.nanmedian(np.abs(f-np.nanmedian(f)))
        e=np.full_like(f,max(1.4826*mad,1e-6))
    else:
        e[~g]=np.nanmedian(e[g])
    return w,f,e

def acquire_best_spectrum():
    cache=DATA/'JADES_GS_Z11_0_BEST_PUBLIC_PRISM.fits'
    if cache.exists() and cache.stat().st_size>50000:
        vv=vectors_from_hdul(cache)
        norm=[normalize_candidate(*x[:3]) for x in vv]
        norm=[x for x in norm if x is not None]
        if norm: return cache,max(norm,key=lambda q:q[0].size)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        obs=Observations.query_region(f'{RA} {DEC}',radius='1.2 arcsec')
        if len(obs)==0: raise RuntimeError('No public observations at the published coordinates')
        if 'obs_collection' in obs.colnames:
            obs=obs[np.array([str(x).upper()=='JWST' for x in obs['obs_collection']])]
        products=Observations.get_product_list(obs)
    names=np.array([str(x).lower() for x in products['productFilename']])
    score=[]
    for n in names:
        s=0
        s+=300 if ('x1d' in n or 'spec1d' in n) else 0
        s+=160 if 'prism' in n else 0
        s+=100 if 'clear' in n else 0
        s+=50 if n.endswith('.fits') else -1000
        s-=120 if any(k in n for k in ['s2d','rate','uncal','calints']) else 0
        score.append(s)
    order=np.argsort(score)[::-1]
    best=None; best_path=None; best_n=-1
    tried=0
    for j in order:
        if score[j]<0 or tried>=24: break
        row=products[int(j)]; tried+=1
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                man=Observations.download_products(Table(rows=[row]),download_dir=str(DATA),cache=True)
            p=Path(str(man['Local Path'][0]))
            if not p.exists(): continue
            for v in vectors_from_hdul(p):
                q=normalize_candidate(*v[:3])
                if q is not None and q[0].size>best_n:
                    best,best_path,best_n=q,p,q[0].size
        except Exception:
            continue
    if best is None: raise RuntimeError('No usable public 1D PRISM spectrum was found after testing 24 coordinate-matched products')
    cache.write_bytes(best_path.read_bytes())
    return cache,best

PATH,(wave,flux,err)=acquire_best_spectrum()
fit=(wave>=1.15)&(wave<=2.30)
x,y,s=wave[fit],flux[fit],err[fit]
if x.size<50: raise RuntimeError(f'Only {x.size} valid samples in the Lyman-break window')

def model(z,beta=-1.8,width=0.018):
    edge=LYA*(1+z)
    cont=(np.maximum(x,edge)/1.7)**beta
    tr=.5*(1+np.tanh((x-edge)/width))
    raw=cont*tr
    d=np.diff(x); d=d[np.isfinite(d)&(d>0)]
    if d.size==0: return raw
    dl=float(np.median(d)); sig=max(float(np.median(x)/(100*2.35482)/dl),.35)
    return gaussian_filter1d(raw,sig,mode='nearest')

def chi(z):
    u=model(z); w=1/s**2; den=np.sum(w*u*u)
    if not np.isfinite(den) or den<=0: return 1e99
    a=np.sum(w*y*u)/den
    return float(np.sum(((y-a*u)/s)**2))

res=minimize_scalar(chi,bounds=(10.7,11.8),method='bounded',options={'xatol':1e-5})
zbest=float(res.x)
zg=np.linspace(10.9,11.7,321); ch=np.array([chi(z) for z in zg])
p=np.exp(np.clip(-.5*(ch-np.nanmin(ch)),-700,0)); p/=np.trapz(p,zg)
c=np.cumsum(p); c/=c[-1]; z16,z50,z84=[np.interp(q,c,zg) for q in (.16,.5,.84)]
u=model(zbest); w=1/s**2; amp=np.sum(w*y*u)/np.sum(w*u*u); best=amp*u; resid=(y-best)/s
lya=LYA*(1+zbest); heii=HEII*(1+zbest)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Replot robust public fit',button_style='success',layout=widgets.Layout(width='240px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,14)); gs=fig.add_gridspec(3,1,height_ratios=[.8,2.3,.8],hspace=.14)
        a0,a1,a2=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a0,a1,a2): a.grid(color='#303944',lw=.6,alpha=.75)
        a0.plot(zg,p,lw=2,label=f'Posterior z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})'); a0.axvline(zbest,lw=1.4,label=f'Best fit z={zbest:.4f}'); a0.axvline(Z_PAPER,ls='--',lw=1,label='Paper model z=11.38'); a0.axvspan(z16,z84,alpha=.14); a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False,ncol=2)
        a1.fill_between(x,y-s,y+s,step='mid',alpha=.30,label='1σ uncertainty'); a1.step(x,y,where='mid',lw=1,label='Public JADES NIRSpec/PRISM data'); a1.plot(x,best,lw=2.5,label=f'Best Lyman-break fit z={zbest:.4f}'); a1.axvline(lya,ls='--',lw=1.4,label=f'Fitted Lyα anchor {lya:.5f} μm'); a1.axvline(LYA*(1+Z_PAPER),ls=':',lw=1.2,label=f'Paper z=11.38 → {LYA*(1+Z_PAPER):.5f} μm'); a1.axvline(heii,ls='--',lw=1,label=f'He II reference {heii:.5f} μm'); a1.set_xlim(1.15,2.30); q=np.nanpercentile(y,[1,99]); pad=.2*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad); a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=9)
        a2.axhline(0,lw=.6); a2.step(x,resid,where='mid',lw=.9); a2.axhspan(-1,1,alpha=.18,label='±1σ'); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_xlabel('Observed wavelength [μm]'); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False)
        fig.suptitle('JADES-GS-z11-0 — robust multi-product public-spectrum redshift chain',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_REDSHIFT_CHAIN.png',dpi=450,bbox_inches='tight')
        pd.DataFrame({'wavelength_um':x,'flux_1e-21':y,'sigma_1e-21':s,'model':best,'residual_sigma':resid}).to_csv(CSV/f'{VERSION}_FIT.csv',index=False)
        plt.show()
run.on_click(draw); display(run,out); draw()
print(f'CODE OUTPUT: {VERSION}')
print(f'fit samples: {x.size} | best z: {zbest:.6f} | Lyα observed: {lya:.6f} μm')
print(f'PNG: {PNG}/{VERSION}_REDSHIFT_CHAIN.png')
print(f'CSV: {CSV}/{VERSION}_FIT.csv')
print(f'# {VERSION}')
