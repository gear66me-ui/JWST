# JWST_0148
import io, warnings, contextlib, sys, subprocess
from pathlib import Path
for p in ["numpy","pandas","matplotlib","astropy","requests","ipywidgets","scipy"]:
    try: __import__(p)
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q",p],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, requests, matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import erf
from scipy.optimize import lsq_linear
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets
warnings.filterwarnings("ignore")

VERSION="JWST_0148"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
NAME="mom-cos04-v4_prism-clear_5224_277193.spec.fits"
URL=f"https://s3.amazonaws.com/msaexp-nirspec/extractions/mom-cos04-v4/{NAME}"
PATH=DATA/NAME
if not PATH.exists() or PATH.stat().st_size<100000:
    with requests.get(URL,stream=True,timeout=120) as r:
        r.raise_for_status()
        with open(PATH,"wb") as f:
            for c in r.iter_content(1024*1024):
                if c: f.write(c)
with fits.open(PATH,memmap=False) as h:
    t=h["SPEC1D"].data; nm={n.upper():n for n in t.names}
    def pick(*ks):
        for k in ks:
            if k.upper() in nm: return np.ravel(np.asarray(t[nm[k.upper()]],float))
        return None
    wave=pick("wave","wavelength","lam"); flux=pick("flux","flux_corr","fnu"); err=pick("err","error","flux_err","sigma")
    if err is None:
        iv=pick("ivar","inverse_variance"); err=np.where(iv>0,1/np.sqrt(iv),np.nan)
conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-20
flam=flux*conv; sig=err*conv
m=np.isfinite(wave)&np.isfinite(flam)&np.isfinite(sig)&(sig>0)
wave,flam,sig=wave[m],flam[m],sig[m]; o=np.argsort(wave); wave,flam,sig=wave[o],flam[o],sig[o]
REST={"Lyα break":0.121567,"N IV]":0.1487,"C IV":0.15495,"He II":0.1640,"O III]":0.1663,"N III]":0.1750,"C III]":0.1908}
LINES=list(REST)[1:]; PUB=14.44; PUBSIG=.02

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white"})
display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst label,.jwst .widget-label{color:#fff!important;font-weight:600}.jwst input{background:#111!important;color:#fff!important;border:1px solid #777!important}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
zmin=widgets.FloatText(value=14.20,description="z min:",layout=widgets.Layout(width="150px")); zmax=widgets.FloatText(value=14.65,description="z max:",layout=widgets.Layout(width="150px")); dz=widgets.FloatText(value=.0005,description="Δz:",layout=widgets.Layout(width="145px"))
bw=widgets.FloatSlider(value=.018,min=.005,max=.050,step=.001,description="Break σ [μm]:",layout=widgets.Layout(width="320px"),readout_format=".3f")
fw=widgets.FloatSlider(value=.025,min=.010,max=.060,step=.001,description="Line FWHM [μm]:",layout=widgets.Layout(width="320px"),readout_format=".3f")
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
        zg=np.arange(float(zmin.value),float(zmax.value)+.5*float(dz.value),float(dz.value))
        chi=np.array([fit(lam,y,e,z,float(bw.value),float(fw.value))[0] for z in zg]); p=post(zg,chi)
        ib=int(np.nanargmin(chi)); zb=float(zg[ib]); _,model,coef,labels=fit(lam,y,e,zb,float(bw.value),float(fw.value))
        slope=1+zb; pub_slope=1+PUB
        rest=np.array([REST[n] for n in LINES]); obs=slope*rest
        fig=plt.figure(figsize=(16,15),facecolor="black"); gs=fig.add_gridspec(3,1,height_ratios=[1.15,1.35,2.2],hspace=.24)
        ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); ax2=fig.add_subplot(gs[2])
        for ax in (ax0,ax1,ax2):
            ax.set_facecolor("black"); [sp.set_color("white") for sp in ax.spines.values()]; ax.tick_params(colors="white"); ax.grid(color="#303030",lw=.6,alpha=.8)
        ax0.plot(zg,p,color="#67e0d1",lw=2,label=f"Break + six-line posterior: z={zb:.4f}")
        ax0.axvspan(PUB-PUBSIG,PUB+PUBSIG,color="#777",alpha=.25,label="Published z=14.44 ± 0.02")
        ax0.axvline(PUB,color="white",ls="--",lw=.8); ax0.axvline(zb,color="#ffd400",lw=1.5)
        ax0.set_xlabel("Shared redshift, z"); ax0.set_ylabel("Normalized posterior density")
        lg=ax0.legend(facecolor="black",edgecolor="#777",fontsize=9); [t.set_color("white") for t in lg.get_texts()]
        xx=np.linspace(.115,.200,300)
        ax1.fill_between(xx,(1+PUB-PUBSIG)*xx,(1+PUB+PUBSIG)*xx,color="#ff3333",alpha=.15,label="Published ±0.020 band")
        ax1.plot(xx,pub_slope*xx,color="#ff6666",ls="--",lw=.9,label=f"Published relation: z={PUB:.4f}")
        ax1.plot(xx,slope*xx,color="#ffd400",lw=2,label=f"Combined break + six lines: z={zb:.4f}")
        ax1.scatter(rest,obs,s=62,c=np.arange(6),cmap="tab10",edgecolors="white",linewidths=.5,zorder=3)
        for n,xv,yv in zip(LINES,rest,obs): ax1.annotate(n,(xv,yv),xytext=(5,5),textcoords="offset points",fontsize=9)
        lrest=REST["Lyα break"]; lobs=slope*lrest
        ax1.scatter([lrest],[lobs],marker="D",s=85,facecolors="none",edgecolors="white",linewidths=1.2,zorder=4,label=f"Lyα break model edge: {lobs:.4f} μm")
        ax1.annotate("Lyα break",(lrest,lobs),xytext=(7,-14),textcoords="offset points",color="white",fontsize=9)
        ax1.set_xlabel("Rest wavelength [μm]"); ax1.set_ylabel("Observed wavelength [μm]")
        ax1.set_title("Combined shared-z relation: λobs = (1+z) λrest")
        lg=ax1.legend(facecolor="black",edgecolor="#777",fontsize=8,ncol=2); [t.set_color("white") for t in lg.get_texts()]
        ax2.fill_between(lam,y-e,y+e,step="mid",color="#888",alpha=.28,label="1σ uncertainty")
        ax2.step(lam,y,where="mid",color="#67e0d1",lw=1.2,label="DJA spectrum")
        ax2.plot(lam,model,color="#ffd400",lw=1.5,label=f"Best joint model, z={zb:.4f}")
        lya=REST["Lyα break"]*(1+zb); ax2.axvline(lya,color="white",ls="--",lw=.8)
        ax2.text(lya,.97,"Lyα break",rotation=90,transform=ax2.get_xaxis_transform(),ha="right",va="top",color="white",fontsize=9)
        for n in LINES:
            c=REST[n]*(1+zb); ax2.axvline(c,color="white",ls="--",lw=.45,alpha=.85); ax2.text(c,.97,n,rotation=90,transform=ax2.get_xaxis_transform(),ha="right",va="top",fontsize=8)
        ax2.set_xlim(1.55,3.05); ax2.set_xlabel("Observed wavelength [μm]"); ax2.set_ylabel(r"$f_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
        lg=ax2.legend(facecolor="black",edgecolor="#777",fontsize=9); [t.set_color("white") for t in lg.get_texts()]
        fig.suptitle("MoM-z14 — Lyman-α break + six UV features in one shared-redshift fit",fontsize=15,y=.995)
        fig.tight_layout(rect=[.03,.02,.995,.98]); fig.savefig(PNG/f"{VERSION}_MOM_Z14_LYMAN_BREAK_PLUS_SIX_LINES.png",dpi=450,bbox_inches="tight",facecolor="black"); plt.show()

run.on_click(draw); display(panel); display(out); draw()
