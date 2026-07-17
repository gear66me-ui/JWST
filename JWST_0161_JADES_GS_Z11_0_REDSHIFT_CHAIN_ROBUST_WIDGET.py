# JWST_0161
import io, os, sys, warnings, contextlib, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")
for pkg in ["numpy","pandas","matplotlib","astropy","astroquery","scipy","ipywidgets"]:
    try: __import__(pkg)
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION="JWST_0161"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
RA,DEC=53.1647632,-27.7746223
Z_PAPER=11.38; LYA_REST=0.121567; HEII_REST=0.1640

def download_spectrum():
    saved=DATA/"JADES_GS_Z11_0_PRISM_SPEC1D.fits"
    if saved.exists() and saved.stat().st_size>50000: return saved
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        obs=Observations.query_region(f"{RA} {DEC}",radius="1.0 arcsec")
        keep=np.array([str(x).upper()=="JWST" for x in obs["obs_collection"]]) if "obs_collection" in obs.colnames else np.ones(len(obs),bool)
        obs=obs[keep]
        prod=Observations.get_product_list(obs)
    names=np.array([str(x).lower() for x in prod["productFilename"]])
    masks=[np.array([n.endswith(".fits") and ("spec1d" in n or "x1d" in n) and ("prism" in n or "clear" in n) for n in names]),
           np.array([n.endswith(".fits") and ("spec1d" in n or "x1d" in n) for n in names])]
    cand=None
    for m in masks:
        if np.any(m): cand=prod[m]; break
    if cand is None or len(cand)==0: raise RuntimeError("No public 1D JWST spectrum found at JADES-GS-z11-0 coordinates")
    score=[]
    for r in cand:
        n=str(r["productFilename"]).lower(); score.append((100 if "spec1d" in n else 0)+(80 if "x1d" in n else 0)+(60 if "prism" in n else 0)+(40 if "clear" in n else 0))
    row=cand[int(np.argmax(score))]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        man=Observations.download_products(Table(rows=[row]),download_dir=str(DATA),cache=True)
    src=Path(str(man["Local Path"][0])); saved.write_bytes(src.read_bytes()); return saved

def read_spectrum(path):
    with fits.open(path,memmap=False) as h:
        for hd in h:
            d=getattr(hd,"data",None)
            if d is None or not hasattr(d,"names") or d.names is None: continue
            nm={n.upper():n for n in d.names}
            def pick(*ks):
                for k in ks:
                    if k in nm: return np.ravel(np.asarray(d[nm[k]],float))
                return None
            w=pick("WAVELENGTH","WAVE","LAMBDA"); f=pick("FLUX","FLUX_CORR","FLAM","FNU"); e=pick("ERROR","ERR","FLUX_ERROR","FLUX_ERR","SIGMA")
            if w is None or f is None: continue
            if e is None:
                iv=pick("IVAR","INVERSE_VARIANCE"); e=np.where(iv>0,1/np.sqrt(iv),np.nan) if iv is not None else np.full_like(f,np.nan)
            return w,f,e
    raise RuntimeError("No wavelength/flux table found")

PATH=download_spectrum(); wave,flux,err=read_spectrum(PATH)
if np.nanmedian(wave)>100: wave/=1e4
elif np.nanmedian(wave)>10: wave/=1e3
if np.nanmedian(np.abs(flux[np.isfinite(flux)]))<1e-10:
    conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-21; flux*=conv; err*=conv
m=np.isfinite(wave)&np.isfinite(flux)&(wave>0.8)&(wave<5.3)
wave,flux,err=wave[m],flux[m],err[m]
o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]
uw,idx=np.unique(wave,return_index=True); wave,flux,err=uw,flux[idx],err[idx]
g=np.isfinite(err)&(err>0)
if not np.any(g):
    mad=np.nanmedian(np.abs(flux-np.nanmedian(flux))); err=np.full_like(flux,max(1.4826*mad,1e-6))
else: err[~g]=np.nanmedian(err[g])
fitmask=(wave>=1.05)&(wave<=5.20); lam,y,s=wave[fitmask],flux[fitmask],err[fitmask]
if lam.size<20: raise RuntimeError("Insufficient finite PRISM samples")

def empirical_sed(lam_um,z,beta,width,amp):
    if not np.all(np.isfinite([z,beta,width,amp])) or width<=0: return np.full_like(lam_um,np.nan)
    edge=LYA_REST*(1+z)
    cont=amp*(np.maximum(lam_um,edge)/1.7)**beta
    trans=0.5*(1+np.tanh((lam_um-edge)/width))
    raw=cont*trans
    d=np.diff(lam_um); d=d[np.isfinite(d)&(d>0)]
    if d.size==0: return raw
    dl=float(np.median(d)); sigma_lam=float(np.median(lam_um)/(100*2.354820045)); sigma_pix=max(sigma_lam/dl,0.35)
    if not np.isfinite(sigma_pix): return raw
    return gaussian_filter1d(raw,sigma_pix,mode="nearest")

def objective(theta,yy=y):
    z,beta,width=theta
    if not (10.7<=z<=11.9 and -4.5<=beta<=0.5 and 0.002<=width<=0.080): return 1e99
    u=empirical_sed(lam,z,beta,width,1.0)
    if not np.all(np.isfinite(u)): return 1e99
    w=1/s**2; den=np.sum(w*u*u)
    if not np.isfinite(den) or den<=0: return 1e99
    amp=np.sum(w*yy*u)/den; mod=amp*u
    val=np.sum(((yy-mod)/s)**2)
    return float(val) if np.isfinite(val) else 1e99

res=minimize(objective,[Z_PAPER,-1.8,0.018],method="Nelder-Mead",options={"maxiter":700,"xatol":1e-6,"fatol":1e-4})
zbest,bbest,wbest=res.x
u=empirical_sed(lam,zbest,bbest,wbest,1.0); wt=1/s**2; abest=np.sum(wt*y*u)/np.sum(wt*u*u); best=abest*u
zg=np.linspace(11.0,11.72,241); chi=np.empty_like(zg)
for i,z in enumerate(zg):
    r=minimize(lambda q: objective([z,q[0],q[1]]),[bbest,wbest],method="Nelder-Mead",options={"maxiter":80,"xatol":5e-5,"fatol":1e-3}); chi[i]=r.fun
p=np.exp(np.clip(-0.5*(chi-np.nanmin(chi)),-700,0)); p/=np.trapz(p,zg)
c=np.cumsum(p); c/=c[-1]; z16,z50,z84=[np.interp(q,c,zg) for q in (0.16,0.5,0.84)]
lya=LYA_REST*(1+zbest); heii=HEII_REST*(1+zbest); resid=(y-best)/s

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","xtick.color":"white","ytick.color":"white","axes.edgecolor":"#b8c4d0"})
display(HTML("<style>.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"))
run=widgets.Button(description="Re-fit full spectrum",button_style="success",layout=widgets.Layout(width="220px")); run.add_class("jwst")
out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(16,15)); gs=fig.add_gridspec(3,1,height_ratios=[0.9,2.5,0.9],hspace=.12)
        ax0,ax1,ax2=fig.add_subplot(gs[0]),fig.add_subplot(gs[1]),fig.add_subplot(gs[2])
        for ax in (ax0,ax1,ax2): ax.grid(color="#303944",lw=.6,alpha=.72)
        ax0.plot(zg,p,color="#62dfd1",lw=2,label=f"Posterior z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})"); ax0.axvline(zbest,color="#ffd400",lw=1.4); ax0.axvline(Z_PAPER,color="#ff6b6b",ls="--",lw=1); ax0.axvspan(z16,z84,color="#62dfd1",alpha=.12); ax0.set_ylabel("Normalized posterior"); ax0.set_xlabel("Redshift, z"); ax0.legend(frameon=False,ncol=2)
        ax1.fill_between(lam,y-s,y+s,step="mid",color="#8f8f8f",alpha=.30,label="1σ uncertainty"); ax1.step(lam,y,where="mid",color="#4f79b9",lw=1.0,label="JADES NIRSpec/PRISM data"); ax1.plot(lam,best,color="#e68645",lw=2.4,label=f"Best full-spectrum fit z={zbest:.4f}"); ax1.axvline(lya,color="#ffd400",ls="--",lw=1.3,label=f"Lyα break anchor {lya:.4f} μm"); ax1.axvline(heii,color="#ff4d4d",ls="--",lw=1.1,label=f"He II 1640 reference {heii:.4f} μm"); ax1.axhline(0,color="#9aa5b1",lw=.5,ls=":"); ax1.set_xlim(1.0,5.25); q=np.nanpercentile(y,[1,99]); ax1.set_ylim(q[0]-.18*(q[1]-q[0]),q[1]+.18*(q[1]-q[0])); ax1.set_ylabel(r"Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]"); ax1.legend(frameon=False,ncol=2,fontsize=9)
        ax2.axhline(0,color="white",lw=.6); ax2.step(lam,resid,where="mid",color="#62dfd1",lw=.9); ax2.axhspan(-1,1,color="#8f8f8f",alpha=.18,label="±1σ"); ax2.set_xlim(1.0,5.25); ax2.set_ylim(-6,6); ax2.set_xlabel("Observed wavelength [μm]"); ax2.set_ylabel("Residual / σ"); ax2.legend(frameon=False)
        fig.suptitle("JADES-GS-z11-0 — robust Lyman-break full-spectrum redshift chain",fontsize=15,y=.995)
        fig.savefig(PNG/f"{VERSION}_JADES_GS_Z11_0_REDSHIFT_CHAIN.png",dpi=430,bbox_inches="tight")
        pd.DataFrame({"wavelength_um":lam,"flux_1e-21":y,"sigma_1e-21":s,"best_model":best,"residual_sigma":resid}).to_csv(CSV/f"{VERSION}_SPECTRUM_FIT.csv",index=False)
        plt.show()
run.on_click(draw); display(run); display(out); draw()
