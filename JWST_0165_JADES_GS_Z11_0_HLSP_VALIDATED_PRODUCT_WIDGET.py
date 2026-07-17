# JWST_0165
import io, os, sys, warnings, subprocess, contextlib
from pathlib import Path
warnings.filterwarnings('ignore')
for pkg in ['numpy','pandas','matplotlib','scipy','astropy','astroquery','ipywidgets']:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,'-m','pip','install','-q',pkg],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION='JWST_0165'; ROOT=Path('/content/JWST_OUTPUT'); PNG=ROOT/'PNG'; CSV=ROOT/'CSV'; DATA=ROOT/'DATA'/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
RA,DEC=53.1647632,-27.7746223; ZP=11.38; LYA=0.121567; HEII=0.1640

def extract_1d(path):
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
            w=pick(['wavelength','wave','lambda']); f=pick(['flux','flux_corr','flam','fnu']); e=pick(['error','err','flux_error','flux_err','sigma'])
            if w is not None and f is not None and w.size==f.size:
                if e is None or e.size!=f.size: e=np.full_like(f,np.nan)
                med=np.nanmedian(w)
                if med>1000: w=w/1e4
                elif med>10: w=w/1e3
                return w,f,e
    return None

def get_valid_product():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        obs=Observations.query_region(f'{RA} {DEC}',radius='1.5 arcsec')
        if len(obs)==0: raise RuntimeError('MAST returned no observations at the published coordinates')
        prod=Observations.get_product_list(obs)
    names=np.array([str(x).lower() for x in prod['productFilename']])
    mask=np.array([n.endswith('.fits') and ('x1d' in n or 'spec1d' in n) for n in names])
    cand=prod[mask]
    if len(cand)==0: raise RuntimeError('MAST returned no 1D FITS candidates')
    scores=[]
    for r in cand:
        n=str(r['productFilename']).lower()
        s=(100 if 'prism' in n else 0)+(80 if 'clear' in n else 0)+(40 if 'hlsp_jades' in n else 0)+(20 if 'x1d' in n else 0)
        scores.append(s)
    order=np.argsort(scores)[::-1]
    audit=[]
    for j in order[:20]:
        row=cand[int(j)]
        name=str(row['productFilename'])
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                man=Observations.download_products(Table(rows=[row]),download_dir=str(DATA),cache=True)
            p=Path(str(man['Local Path'][0]))
            spec=extract_1d(p)
            if spec is None:
                audit.append((name,'unreadable',0,np.nan,np.nan)); continue
            w,f,e=spec; finite=np.isfinite(w)&np.isfinite(f)
            nwin=int(np.sum(finite&(w>=1.15)&(w<=2.30)))
            audit.append((name,'ok',nwin,float(np.nanmin(w)),float(np.nanmax(w))))
            if nwin>=50:
                pd.DataFrame(audit,columns=['product','status','samples_1p15_2p30','wave_min_um','wave_max_um']).to_csv(CSV/f'{VERSION}_PRODUCT_AUDIT.csv',index=False)
                return p,w,f,e,name
        except Exception as ex:
            audit.append((name,type(ex).__name__,0,np.nan,np.nan))
    pd.DataFrame(audit,columns=['product','status','samples_1p15_2p30','wave_min_um','wave_max_um']).to_csv(CSV/f'{VERSION}_PRODUCT_AUDIT.csv',index=False)
    raise RuntimeError('No downloaded candidate actually covered 1.15–2.30 µm with at least 50 finite samples; see product audit CSV')

path,wave,flux,err,product=get_valid_product()
m=np.isfinite(wave)&np.isfinite(flux)&(wave>0.8)&(wave<5.35); wave,flux,err=wave[m],flux[m],err[m]
o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]
wave,idx=np.unique(wave,return_index=True); flux,err=flux[idx],err[idx]
if np.nanmedian(np.abs(flux))<1e-8:
    conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-21; flux*=conv; err*=conv
g=np.isfinite(err)&(err>0)
if np.sum(g)<20:
    mad=np.nanmedian(np.abs(flux-np.nanmedian(flux))); err=np.full_like(flux,max(1.4826*mad,1e-6))
else: err[~g]=np.nanmedian(err[g])
fit=(wave>=1.15)&(wave<=2.30); x,y,s=wave[fit],flux[fit],err[fit]

def model(z,beta=-1.8,width=0.018):
    edge=LYA*(1+z); cont=(np.maximum(x,edge)/1.7)**beta; tr=.5*(1+np.tanh((x-edge)/width)); raw=cont*tr
    d=np.diff(x); d=d[np.isfinite(d)&(d>0)]; dl=np.median(d); sig=max(np.median(x)/(100*2.35482)/dl,.35)
    return gaussian_filter1d(raw,sig,mode='nearest')
def chi(z):
    u=model(z); wt=1/s**2; a=np.sum(wt*y*u)/np.sum(wt*u*u); return np.sum(((y-a*u)/s)**2)
res=minimize_scalar(chi,bounds=(10.7,11.8),method='bounded',options={'xatol':1e-5}); zbest=float(res.x)
zg=np.linspace(10.9,11.7,401); ch=np.array([chi(z) for z in zg]); p=np.exp(np.clip(-.5*(ch-ch.min()),-700,0)); p/=np.trapz(p,zg)
c=np.cumsum(p); c/=c[-1]; z16,z50,z84=[np.interp(q,c,zg) for q in (.16,.5,.84)]
u=model(zbest); wt=1/s**2; amp=np.sum(wt*y*u)/np.sum(wt*u*u); best=amp*u; resid=(y-best)/s
lya=LYA*(1+zbest); heii=HEII*(1+zbest)
plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Replot validated fit',button_style='success',layout=widgets.Layout(width='230px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,14)); gs=fig.add_gridspec(3,1,height_ratios=[.8,2.3,.8],hspace=.14); a0,a1,a2=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a0,a1,a2): a.grid(color='#303944',lw=.6,alpha=.75)
        a0.plot(zg,p,color='#62dfd1',lw=2,label=f'Posterior z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})'); a0.axvline(zbest,color='#ffd400',lw=1.4); a0.axvline(ZP,color='#ff6b6b',ls='--',lw=1,label='Paper z=11.38'); a0.axvspan(z16,z84,color='#62dfd1',alpha=.14); a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False)
        a1.fill_between(x,y-s,y+s,step='mid',color='gray',alpha=.32,label='1σ uncertainty'); a1.step(x,y,where='mid',color='#4f79b9',lw=1,label='Validated JADES 1D spectrum'); a1.plot(x,best,color='#e68645',lw=2.5,label=f'Best Lyman-break fit z={zbest:.4f}'); a1.axvline(lya,color='#ffd400',ls='--',lw=1.4,label=f'Lyα anchor {lya:.5f} µm'); a1.axvline(LYA*(1+ZP),color='#ff6b6b',ls=':',lw=1.2,label=f'Paper z=11.38 → {LYA*(1+ZP):.5f} µm'); a1.axvline(heii,color='#ff4d4d',ls='--',lw=1,label=f'He II reference {heii:.5f} µm'); a1.set_xlim(1.15,2.30); q=np.nanpercentile(y,[1,99]); a1.set_ylim(q[0]-.2*(q[1]-q[0]),q[1]+.2*(q[1]-q[0])); a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=9)
        a2.axhline(0,color='white',lw=.6); a2.step(x,resid,where='mid',color='#62dfd1',lw=.9); a2.axhspan(-1,1,color='gray',alpha=.2,label='±1σ'); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_xlabel('Observed wavelength [µm]'); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False)
        fig.suptitle(f'JADES-GS-z11-0 — validated product redshift chain\n{product}',fontsize=14,y=.995)
        fig.savefig(PNG/f'{VERSION}_VALIDATED_REDSHIFT_CHAIN.png',dpi=450,bbox_inches='tight')
        pd.DataFrame({'wavelength_um':x,'flux_1e-21':y,'sigma_1e-21':s,'model':best,'residual_sigma':resid}).to_csv(CSV/f'{VERSION}_VALIDATED_FIT.csv',index=False)
        plt.show()
run.on_click(draw); display(run,out); draw()
