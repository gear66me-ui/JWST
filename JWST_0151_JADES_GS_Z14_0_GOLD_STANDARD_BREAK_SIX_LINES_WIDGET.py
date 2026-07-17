# JWST_0151
import io, warnings, contextlib, sys, subprocess, os
from pathlib import Path
for p in ["numpy","pandas","matplotlib","astropy","astroquery","ipywidgets","scipy"]:
    try: __import__(p)
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q",p],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from scipy.special import erf
from scipy.optimize import lsq_linear
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
warnings.filterwarnings("ignore")

VERSION="JWST_0151"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
RA,DEC=53.1537083,-27.7803694
PUB=14.32; PUBLO=0.20; PUBHI=0.08
REST={"Lyα break":0.121567,"N IV]":0.1487,"C IV":0.15495,"He II":0.1640,"O III]":0.1663,"N III]":0.1750,"C III]":0.1908}
LINES=list(REST)[1:]

def download_public_spectrum():
    saved=DATA/"JADES_GS_Z14_0_PRISM_CLEAR_X1D.fits"
    if saved.exists() and saved.stat().st_size>50000: return saved
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        obs=Observations.query_region(f"{RA} {DEC}",radius="1.0 arcsec")
        if len(obs)==0: raise RuntimeError("No MAST observations found at JADES-GS-z14-0")
        keep=np.ones(len(obs),dtype=bool)
        if "obs_collection" in obs.colnames: keep &= np.array([str(x).upper()=="JWST" for x in obs["obs_collection"]])
        if "proposal_id" in obs.colnames: keep &= np.array([str(x).strip()=="1287" for x in obs["proposal_id"]])
        j=obs[keep]
        if len(j)==0: j=obs
        prod=Observations.get_product_list(j)
        names=np.array([str(x).lower() for x in prod["productFilename"]])
        mask=np.array([n.endswith(".fits") and ("x1d" in n or "spec1d" in n) for n in names])
        cand=prod[mask]
        if len(cand)==0: raise RuntimeError("No public 1D JWST spectrum found for JADES-GS-z14-0")
        score=[]
        for r in cand:
            n=str(r["productFilename"]).lower(); s=0
            s += 50 if "prism" in n else 0
            s += 30 if "clear" in n else 0
            s += 20 if "x1d" in n else 0
            s += 10 if "1287" in n else 0
            score.append(s)
        row=cand[int(np.argmax(score))]
        man=Observations.download_products(Table(rows=[row]),download_dir=str(DATA),cache=True)
    src=Path(str(man["Local Path"][0]))
    if not src.exists(): raise RuntimeError("MAST download did not produce a local FITS file")
    saved.write_bytes(src.read_bytes())
    return saved

def read_spectrum(path):
    with fits.open(path,memmap=False) as h:
        for hd in h:
            d=getattr(hd,"data",None)
            if d is None or not hasattr(d,"names") or d.names is None: continue
            nm={n.upper():n for n in d.names}
            def pick(*ks):
                for k in ks:
                    if k.upper() in nm: return np.ravel(np.asarray(d[nm[k.upper()]],float))
                return None
            w=pick("WAVELENGTH","WAVE","LAMBDA"); f=pick("FLUX","FLUX_CORR","FNU","FLAM")
            e=pick("ERROR","ERR","FLUX_ERROR","FLUX_ERR","SIGMA")
            if w is None or f is None: continue
            if e is None:
                iv=pick("IVAR","INVERSE_VARIANCE")
                if iv is not None: e=np.where(iv>0,1/np.sqrt(iv),np.nan)
            if e is None: e=np.full_like(f,np.nan)
            unit=str(getattr(hd.columns[nm.get("FLUX","FLUX")],"unit","") if "FLUX" in nm else "").lower()
            return w,f,e,unit
    raise RuntimeError("No wavelength/flux table found in downloaded FITS")

PATH=download_public_spectrum(); wave,flux,err,unit=read_spectrum(PATH)
if np.nanmedian(wave)>100: wave=wave/1e4
if "jy" in unit or np.nanmedian(np.abs(flux))<1e-10:
    conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-20
    flam=flux*conv; sig=err*conv
else:
    flam=flux/1e-20; sig=err/1e-20
m=np.isfinite(wave)&np.isfinite(flam)&(wave>0.7)&(wave<5.35)
wave,flam,sig=wave[m],flam[m],sig[m]
o=np.argsort(wave); wave,flam,sig=wave[o],flam[o],sig[o]
g=np.isfinite(sig)&(sig>0)
if not np.any(g):
    mad=np.nanmedian(np.abs(flam-np.nanmedian(flam))); sig=np.full_like(flam,max(1.4826*mad,1e-6))
else: sig[~g]=np.nanmedian(sig[g])

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white"})
display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst label,.jwst .widget-label{color:#fff!important;font-weight:600}.jwst input{background:#111!important;color:#fff!important;border:1px solid #777!important}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
zmin=widgets.FloatText(value=13.80,description="z min:",layout=widgets.Layout(width="150px")); zmax=widgets.FloatText(value=14.55,description="z max:",layout=widgets.Layout(width="150px")); dz=widgets.FloatText(value=.0005,description="Δz:",layout=widgets.Layout(width="145px"))
bw=widgets.FloatSlider(value=.022,min=.005,max=.060,step=.001,description="Break σ [μm]:",layout=widgets.Layout(width="320px"),readout_format=".3f")
fw=widgets.FloatSlider(value=.028,min=.010,max=.080,step=.001,description="Line FWHM [μm]:",layout=widgets.Layout(width="320px"),readout_format=".3f")
run=widgets.Button(description="Run break + six lines",button_style="success",layout=widgets.Layout(width="210px"))
panel=widgets.HBox([zmin,zmax,dz,bw,fw,run],layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="8px")); panel.add_class("jwst")
out=widgets.Output(layout=widgets.Layout(border="1px solid #333"))

def matrix(lam,z,bs,lf):
    pivot=2.25; lya=REST["Lyα break"]*(1+z); step=0.5*(1+erf((lam-lya)/(np.sqrt(2)*bs)))
    cols=[np.ones_like(lam),step,step*(lam-pivot)]; labels=["blue","red","slope"]
    gs=lf/2.354820045
    for n in LINES:
        c=REST[n]*(1+z); cols.append(np.exp(-.5*((lam-c)/gs)**2)); labels.append(n)
    return np.column_stack(cols),labels

def fit(lam,y,e,z,bs,lf):
    A,labels=matrix(lam,z,bs,lf); Aw=A/e[:,None]; yw=y/e
    lo=np.r_[np.full(3,-np.inf),np.zeros(len(LINES))]; hi=np.full(A.shape[1],np.inf)
    s=lsq_linear(Aw,yw,bounds=(lo,hi),method="trf",lsmr_tol="auto")
    model=A@s.x; return np.sum(((y-model)/e)**2),model,s.x,labels

def post(zg,chi):
    p=np.exp(np.clip(-.5*(chi-np.nanmin(chi)),-700,0)); a=np.trapz(p,zg); return p/a if a>0 else p

def draw(_=None):
    with out:
        clear_output(wait=True)
        use=(wave>=1.55)&(wave<=3.05); lam,y,e=wave[use],flam[use],sig[use]
        if lam.size<20: raise RuntimeError("Downloaded product does not contain enough PRISM samples in 1.55–3.05 μm")
        zg=np.arange(float(zmin.value),float(zmax.value)+.5*float(dz.value),float(dz.value))
        chi=np.array([fit(lam,y,e,z,float(bw.value),float(fw.value))[0] for z in zg]); p=post(zg,chi)
        ib=int(np.nanargmin(chi)); zb=float(zg[ib]); _,model,coef,labels=fit(lam,y,e,zb,float(bw.value),float(fw.value))
        slope=1+zb; rest=np.array([REST[n] for n in LINES]); obs=slope*rest
        pd.DataFrame({"feature":["Lyα break"]+LINES,"rest_wavelength_um":[REST["Lyα break"]]+[REST[n] for n in LINES],"observed_wavelength_um":slope*np.array([REST["Lyα break"]]+[REST[n] for n in LINES]),"fit_coefficient":[np.nan]+list(coef[3:])}).to_csv(CSV/f"{VERSION}_JADES_GS_Z14_0_FEATURES.csv",index=False)
        fig=plt.figure(figsize=(16,15),facecolor="black"); gs=fig.add_gridspec(3,1,height_ratios=[1.15,1.35,2.2],hspace=.24)
        ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); ax2=fig.add_subplot(gs[2])
        for ax in (ax0,ax1,ax2):
            ax.set_facecolor("black"); [sp.set_color("white") for sp in ax.spines.values()]; ax.tick_params(colors="white"); ax.grid(color="#303030",lw=.6,alpha=.8)
        ax0.plot(zg,p,color="#67e0d1",lw=2,label=f"Break + six-line posterior: z={zb:.4f}")
        ax0.axvspan(PUB-PUBLO,PUB+PUBHI,color="#777",alpha=.25,label="Published NIRSpec z=14.32 +0.08/-0.20")
        ax0.axvline(PUB,color="white",ls="--",lw=.8); ax0.axvline(zb,color="#ffd400",lw=1.5)
        ax0.set_xlabel("Shared redshift, z"); ax0.set_ylabel("Normalized posterior density")
        lg=ax0.legend(facecolor="black",edgecolor="#777",fontsize=9); [t.set_color("white") for t in lg.get_texts()]
        xx=np.linspace(.115,.200,300)
        ax1.fill_between(xx,(1+PUB-PUBLO)*xx,(1+PUB+PUBHI)*xx,color="#ff3333",alpha=.15,label="Published NIRSpec interval")
        ax1.plot(xx,(1+PUB)*xx,color="#ff6666",ls="--",lw=.9,label=f"Published relation: z={PUB:.4f}")
        ax1.plot(xx,slope*xx,color="#ffd400",lw=2,label=f"Combined break + six lines: z={zb:.4f}")
        ax1.scatter(rest,obs,s=62,c=np.arange(6),cmap="tab10",edgecolors="white",linewidths=.5,zorder=3)
        for n,xv,yv in zip(LINES,rest,obs): ax1.annotate(n,(xv,yv),xytext=(5,5),textcoords="offset points",fontsize=9)
        lrest=REST["Lyα break"]; lobs=slope*lrest
        ax1.scatter([lrest],[lobs],marker="D",s=85,facecolors="none",edgecolors="white",linewidths=1.2,zorder=4,label=f"Lyα break model edge: {lobs:.4f} μm")
        ax1.annotate("Lyα break",(lrest,lobs),xytext=(7,-14),textcoords="offset points",color="white",fontsize=9)
        ax1.set_xlabel("Rest wavelength [μm]"); ax1.set_ylabel("Observed wavelength [μm]"); ax1.set_title("Combined shared-z relation: λobs = (1+z) λrest")
        lg=ax1.legend(facecolor="black",edgecolor="#777",fontsize=8,ncol=2); [t.set_color("white") for t in lg.get_texts()]
        ax2.fill_between(lam,y-e,y+e,step="mid",color="#888",alpha=.28,label="1σ uncertainty")
        ax2.step(lam,y,where="mid",color="#67e0d1",lw=1.2,label="Public JWST/NIRSpec PRISM spectrum")
        ax2.plot(lam,model,color="#ffd400",lw=1.5,label=f"Best joint model, z={zb:.4f}")
        lya=REST["Lyα break"]*(1+zb); ax2.axvline(lya,color="white",ls="--",lw=.8)
        ax2.text(lya,.97,"Lyα break",rotation=90,transform=ax2.get_xaxis_transform(),ha="right",va="top",color="white",fontsize=9)
        for n in LINES:
            c=REST[n]*(1+zb); ax2.axvline(c,color="white",ls="--",lw=.45,alpha=.85); ax2.text(c,.97,n,rotation=90,transform=ax2.get_xaxis_transform(),ha="right",va="top",fontsize=8)
        ax2.set_xlim(1.55,3.05); ax2.set_xlabel("Observed wavelength [μm]"); ax2.set_ylabel(r"$f_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
        lg=ax2.legend(facecolor="black",edgecolor="#777",fontsize=9); [t.set_color("white") for t in lg.get_texts()]
        fig.suptitle("JADES-GS-z14-0 — Lyman-α break + six UV features in one shared-redshift fit",fontsize=15,y=.995)
        fig.tight_layout(rect=[.03,.02,.995,.98]); fig.savefig(PNG/f"{VERSION}_JADES_GS_Z14_0_BREAK_PLUS_SIX_LINES.png",dpi=450,bbox_inches="tight",facecolor="black"); plt.show()

run.on_click(draw); display(panel); display(out); draw()
