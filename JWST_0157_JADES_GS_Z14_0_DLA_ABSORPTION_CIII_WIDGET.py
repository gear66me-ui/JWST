# JWST_0157
import os, io, sys, warnings, contextlib, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")
for p in ["numpy","pandas","matplotlib","requests","ipywidgets","scipy"]:
    try: __import__(p)
    except Exception:
        subprocess.run([sys.executable,"-m","pip","install","-q",p],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
import numpy as np, pandas as pd, matplotlib.pyplot as plt, requests
from scipy.optimize import least_squares
from scipy.special import wofz
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION="JWST_0157"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
URL="https://zenodo.org/records/12578543/files/JADES-GS-z14-0_spec1D.txt?download=1"
PATH=DATA/"JADES-GS-z14-0_spec1D.txt"
ZSYS=14.1793; SZ=0.0007; LYA=0.121567
ABS={"Lyα":0.121567,"Si II 1260":0.126042,"O I 1302":0.130217,"Si II 1304":0.130437,
     "C II 1334":0.133453,"Si IV 1393":0.139376,"Si IV 1402":0.140277}
CIII=0.1908

if not PATH.exists() or PATH.stat().st_size<10000:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r=requests.get(URL,timeout=180); r.raise_for_status(); PATH.write_bytes(r.content)
arr=np.loadtxt(PATH,comments="#")
if arr.ndim==1: arr=arr[None,:]
wave,flux,err=[np.asarray(arr[:,i],float) for i in range(3)]
if np.nanmedian(wave)>100: wave/=1e4
elif np.nanmedian(wave)>10: wave/=1e3
m=np.isfinite(wave)&np.isfinite(flux)&np.isfinite(err)&(err>0)&(wave>0.7)&(wave<5.3)
wave,flux,err=wave[m],flux[m],err[m]
o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]
med=np.nanmedian(np.abs(flux[(wave>1.95)&(wave<2.25)]))
if med<1e-12: flux/=1e-21; err/=1e-21

C=2.99792458e10; E=4.803204712e-10; ME=9.1093837e-28
FOSC=0.4164; GAMMA=6.265e8; LYA_CM=1215.67e-8; NU0=C/LYA_CM; B=20e5
DNU_D=NU0*B/C; SIG0=np.sqrt(np.pi)*E**2/(ME*C)*FOSC/DNU_D

def tau_dla(lam,logn):
    lr=(lam/(1+ZSYS))*1e-4; nu=C/lr; u=(nu-NU0)/DNU_D; a=GAMMA/(4*np.pi*DNU_D)
    return (10**logn)*SIG0*np.real(wofz(u+1j*a))

def convolve_prism(lam,model,R=100.):
    g=np.linspace(lam.min(),lam.max(),max(3000,len(lam)*5)); mg=np.interp(g,lam,model)
    dl=np.nanmedian(np.diff(g)); sig=np.nanmedian(g)/(R*2.354820045)
    return np.interp(lam,g,gaussian_filter1d(mg,max(sig/dl,.5),mode="nearest"))

def dla_model(lam,p,with_dla=True):
    amp,beta,logn,xhi=p; cont=amp*(lam/2.05)**beta; lya=LYA*(1+ZSYS)
    blue=0.5*(1+np.tanh((lam-lya)/.004)); red=np.maximum(lam-lya,1e-4)
    trans=np.where(lam<lya,1e-5,np.exp(-xhi*.018/red))
    if with_dla: trans*=np.exp(-tau_dla(lam,logn))
    return convolve_prism(lam,cont*blue*trans)

fitmask=(wave>=1.68)&(wave<=2.40)
lam,y,s=wave[fitmask],flux[fitmask],err[fitmask]
lo=[.01,-4,20,0]; hi=[20,1,24,1]; init=[3.2,-1.93,22.27,.59]
res=least_squares(lambda p:(y-dla_model(lam,p,True))/s,init,bounds=(lo,hi),max_nfev=800)
pbest=res.x; best=dla_model(lam,pbest,True)
res0=least_squares(lambda p:(y-dla_model(lam,p,False))/s,[3.2,-1.93,22.27,1],bounds=(lo,hi),max_nfev=800)
best0=dla_model(lam,res0.x,False)
chi=np.sum(((y-best)/s)**2); chi0=np.sum(((y-best0)/s)**2)

# Marginal C III] local weighted fit: linear continuum + non-negative Gaussian.
c0=CIII*(1+ZSYS); q=(wave>=c0-.11)&(wave<=c0+.11)
wc,yc,ec=wave[q],flux[q],err[q]
def ciii_model(p):
    a,b,amp,cen,sig=p
    return a+b*(wc-c0)+amp*np.exp(-.5*((wc-cen)/sig)**2)
medc=np.nanmedian(yc); p0=[medc,0,max(np.nanmax(yc)-medc,0),c0,.018]
clo=[-50,-200,0,c0-.035,.006]; chi2=[50,200,50,c0+.035,.060]
rc=least_squares(lambda p:(yc-ciii_model(p))/ec,p0,bounds=(clo,chi2),max_nfev=1000)
pc=rc.x; mc=ciii_model(pc)
A=np.column_stack([np.ones_like(wc),wc-c0,np.exp(-.5*((wc-pc[3])/pc[4])**2)])
try:
    cov=np.linalg.inv((A/ec[:,None]).T@(A/ec[:,None])); amp_err=np.sqrt(max(cov[2,2],0))
except Exception: amp_err=np.nan
snr=pc[2]/amp_err if np.isfinite(amp_err) and amp_err>0 else np.nan

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black",
"text.color":"white","axes.labelcolor":"white","axes.edgecolor":"#9fb0c0","xtick.color":"white","ytick.color":"white","font.size":10})
display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst label,.jwst .widget-label{color:#fff!important;font-weight:600}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
run=widgets.Button(description="Re-fit full public spectrum",button_style="success",layout=widgets.Layout(width="230px"))
out=widgets.Output(layout=widgets.Layout(border="1px solid #333"))

def style(ax):
    ax.set_facecolor("black"); ax.grid(color="#2b3440",lw=.55,alpha=.72)
    for sp in ax.spines.values(): sp.set_color("#9fb0c0")

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,19),facecolor="black")
        gs=fig.add_gridspec(4,1,height_ratios=[1.15,1.45,1.25,1.25],hspace=.30)
        ax0,ax1,ax2,ax3=[fig.add_subplot(gs[i]) for i in range(4)]
        for ax in (ax0,ax1,ax2,ax3): style(ax)

        ax0.fill_between(wave,flux-err,flux+err,step="mid",color="#8e8e8e",alpha=.22,label="1σ uncertainty")
        ax0.step(wave,flux,where="mid",color="#e6e6e6",lw=.85,label="Released JWST/NIRSpec PRISM spectrum")
        ax0.axvspan(1.68,2.40,color="#d94343",alpha=.06,label="DLA fitting interval")
        ax0.axvspan(c0-.11,c0+.11,color="#3c9cff",alpha=.06,label="C III] fitting interval")
        ax0.set_xlim(.8,5.2); ax0.set_ylabel(r"$F_\lambda$ [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
        ax0.set_title("Complete released PRISM spectrum — no wavelength region omitted")
        lg=ax0.legend(facecolor="black",edgecolor="#666",fontsize=8,ncol=2); [t.set_color("white") for t in lg.get_texts()]

        ax1.fill_between(lam,y-s,y+s,step="mid",color="#8f8f8f",alpha=.28,label="1σ uncertainty")
        ax1.step(lam,y,where="mid",color="#f0f0f0",lw=1.0,label="PRISM data")
        ax1.plot(lam,best,color="#d94343",lw=2.0,label=f"IGM + DLA: log NHI={pbest[2]:.2f}, β={pbest[1]:.2f}")
        ax1.plot(lam,best0,color="#63d4ee",ls="-.",lw=1.3,label=f"IGM only; Δχ²={chi0-chi:.1f}")
        for i,(n,rw) in enumerate(ABS.items()):
            x=rw*(1+ZSYS); col="#ffd400" if n=="Lyα" else "#7dd3fc"
            ax1.axvline(x,color=col,ls="--" if n=="Lyα" else ":",lw=1.05 if n=="Lyα" else .75,alpha=.95)
            ax1.text(x,.97-.055*(i%2),n,rotation=90,transform=ax1.get_xaxis_transform(),ha="right",va="top",fontsize=8,color=col)
        ax1.set_xlim(1.68,2.40); ax1.set_ylabel(r"$F_\lambda$ [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
        ax1.set_title("Damped Lyα absorption profile with low-ion metal absorption reference wavelengths")
        lg=ax1.legend(facecolor="black",edgecolor="#666",fontsize=8,ncol=2); [t.set_color("white") for t in lg.get_texts()]

        qa=(wave>=1.84)&(wave<=2.16)
        ax2.fill_between(wave[qa],flux[qa]-err[qa],flux[qa]+err[qa],step="mid",color="#8f8f8f",alpha=.30)
        ax2.step(wave[qa],flux[qa],where="mid",color="#67e0d1",lw=1.15,label="Observed spectrum")
        ax2.plot(lam,best,color="#d94343",lw=1.7,label="Best IGM + DLA continuum")
        for i,(n,rw) in enumerate(ABS.items()):
            x=rw*(1+ZSYS)
            if 1.84<=x<=2.16:
                ax2.axvline(x,color="#ffd400" if n=="Lyα" else "white",ls="--",lw=.85)
                ax2.text(x,.96-.06*(i%2),n,rotation=90,transform=ax2.get_xaxis_transform(),ha="right",va="top",fontsize=8)
        ax2.set_xlim(1.84,2.16); ax2.set_ylabel(r"$F_\lambda$ [$10^{-21}$]")
        ax2.set_title("Absorption-line atlas zoom — expected positions shown; no discrete metal detection claimed")
        lg=ax2.legend(facecolor="black",edgecolor="#666",fontsize=8); [t.set_color("white") for t in lg.get_texts()]

        ax3.fill_between(wc,yc-ec,yc+ec,step="mid",color="#8f8f8f",alpha=.30,label="1σ uncertainty")
        ax3.step(wc,yc,where="mid",color="#67e0d1",lw=1.2,label="Observed C III] window")
        ax3.plot(wc,mc,color="#ffd400",lw=2.0,label=f"Local continuum + Gaussian fit; S/N={snr:.2f}")
        ax3.axvline(c0,color="white",ls="--",lw=.8,label=f"ALMA-systemic C III] position = {c0:.4f} μm")
        ax3.axvline(pc[3],color="#ff7f50",ls=":",lw=1.0,label=f"Fitted centroid = {pc[3]:.4f} μm")
        ax3.set_xlim(c0-.11,c0+.11); ax3.set_xlabel("Observed wavelength [μm]")
        ax3.set_ylabel(r"$F_\lambda$ [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
        ax3.set_title("Marginal C III] signal — actual PRISM samples and weighted best-fit line profile")
        lg=ax3.legend(facecolor="black",edgecolor="#666",fontsize=8,ncol=2); [t.set_color("white") for t in lg.get_texts()]

        fig.suptitle("JADES-GS-z14-0 — complete PRISM spectrum, DLA absorption, metal-line references, and marginal C III] fit",fontsize=16,y=.995)
        fig.savefig(PNG/f"{VERSION}_JADES_GS_Z14_0_DLA_ABSORPTION_CIII.png",dpi=430,bbox_inches="tight",facecolor="black")
        pd.DataFrame({"wavelength_um":wave,"flux_1e-21":flux,"sigma_1e-21":err}).to_csv(CSV/f"{VERSION}_FULL_PRISM.csv",index=False)
        pd.DataFrame({"feature":list(ABS)+["C III]"],"rest_um":list(ABS.values())+[CIII],"observed_um":[v*(1+ZSYS) for v in ABS.values()]+[c0],"status":["DLA absorption" if k=="Lyα" else "reference; not detected" for k in ABS]+["marginal fitted emission"]}).to_csv(CSV/f"{VERSION}_FEATURES.csv",index=False)
        plt.show()

run.on_click(draw); display(run); display(out); draw()
