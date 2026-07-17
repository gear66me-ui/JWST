# JWST_0155
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

VERSION="JWST_0155"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)

URL="https://zenodo.org/records/12578543/files/JADES-GS-z14-0_spec1D.txt?download=1"
PATH=DATA/"JADES-GS-z14-0_spec1D.txt"
Z_SYS=14.1793
SIGMA_Z=0.0007
LYA_REST=0.121567
REST={"Lyα":LYA_REST,"N IV]":0.1487,"C IV":0.15495,"He II":0.1640,"O III]":0.1663,"N III]":0.1750,"C III]":0.1908}
LINES=[k for k in REST if k!="Lyα"]

if not PATH.exists() or PATH.stat().st_size<10000:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r=requests.get(URL,timeout=180); r.raise_for_status(); PATH.write_bytes(r.content)

# Robustly read the public three-column release: wavelength, flux density, 1σ uncertainty.
arr=np.loadtxt(PATH,comments="#")
if arr.ndim==1: arr=arr[None,:]
if arr.shape[1]<3: raise RuntimeError("Zenodo spectrum does not contain wavelength, flux, and uncertainty columns")
wave=np.asarray(arr[:,0],float); flux=np.asarray(arr[:,1],float); err=np.asarray(arr[:,2],float)
if np.nanmedian(wave)>100: wave/=1e4
elif np.nanmedian(wave)>10: wave/=1e3
m=np.isfinite(wave)&np.isfinite(flux)&np.isfinite(err)&(err>0)&(wave>0.7)&(wave<5.3)
wave,flux,err=wave[m],flux[m],err[m]
o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]

# Convert to the paper's display unit: 10^-21 erg s^-1 cm^-2 Å^-1.
med=np.nanmedian(np.abs(flux[(wave>1.95)&(wave<2.25)]))
if med<1e-12:
    flux=flux/1e-21; err=err/1e-21
# Otherwise the public table is already in the paper's scaled flux-density unit.

# Physical Lyα Voigt cross section for the local DLA; systemic redshift is fixed by ALMA [O III] 88 μm.
C=2.99792458e10; E=4.803204712e-10; ME=9.1093837e-28
FOSC=0.4164; GAMMA=6.265e8; LYA_CM=1215.67e-8; NU0=C/LYA_CM; B=20e5
DNU_D=NU0*B/C
SIGMA0=np.sqrt(np.pi)*E**2/(ME*C)*FOSC/DNU_D

def dla_tau(lam_um,logn):
    lam_rest_cm=(lam_um/(1+Z_SYS))*1e-4
    nu=C/lam_rest_cm
    u=(nu-NU0)/DNU_D; a=GAMMA/(4*np.pi*DNU_D)
    H=np.real(wofz(u+1j*a))
    return (10**logn)*SIGMA0*H

def prism_convolve(lam,model,R=100.0):
    grid=np.linspace(lam.min(),lam.max(),max(3000,len(lam)*5))
    mg=np.interp(grid,lam,model)
    dl=np.nanmedian(np.diff(grid)); sigma_l=np.nanmedian(grid)/(R*2.354820045)
    sm=gaussian_filter1d(mg,max(sigma_l/dl,0.5),mode="nearest")
    return np.interp(lam,grid,sm)

def model(lam,p,with_dla=True):
    amp,beta,logn,xhi=p
    cont=amp*(lam/2.05)**beta
    lya=LYA_REST*(1+Z_SYS)
    blue=0.5*(1+np.tanh((lam-lya)/0.004))
    # Compact IGM damping-wing approximation redward of systemic Lyα.
    red=np.maximum(lam-lya,1e-4)
    tau_igm=xhi*0.018/red
    trans=np.where(lam<lya,1e-5,np.exp(-tau_igm))
    if with_dla: trans*=np.exp(-dla_tau(lam,logn))
    return prism_convolve(lam,cont*blue*trans,R=100.0)

fitmask=(wave>=1.68)&(wave<=2.40)
lam,y,s=wave[fitmask],flux[fitmask],err[fitmask]
init=[3.2,-1.93,22.27,0.59]
lo=[0.01,-4.0,20.0,0.0]; hi=[20.0,1.0,24.0,1.0]
res=least_squares(lambda p:(y-model(lam,p,True))/s,init,bounds=(lo,hi),max_nfev=500)
pbest=res.x; best=model(lam,pbest,True)
res0=least_squares(lambda p:(y-model(lam,p,False))/s,[3.2,-1.93,22.27,1.0],bounds=(lo,hi),max_nfev=500)
best_igm=model(lam,res0.x,False)
chi=np.sum(((y-best)/s)**2); chi0=np.sum(((y-best_igm)/s)**2)

# ALMA systemic-redshift posterior, kept separate from the PRISM absorption fit.
zg=np.linspace(Z_SYS-0.004,Z_SYS+0.004,1600)
pz=np.exp(-0.5*((zg-Z_SYS)/SIGMA_Z)**2); pz/=np.trapz(pz,zg)

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"#9fb0c0","xtick.color":"white","ytick.color":"white","font.size":10})
display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst label,.jwst .widget-label{color:#fff!important;font-weight:600}.jwst input{background:#111!important;color:#fff!important;border:1px solid #777!important}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
run=widgets.Button(description="Re-fit public spectrum",button_style="success",layout=widgets.Layout(width="210px"))
out=widgets.Output(layout=widgets.Layout(border="1px solid #333"))


def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,18),facecolor="black")
        gs=fig.add_gridspec(3,1,height_ratios=[0.85,1.7,2.5],hspace=.28)
        ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1])
        sub=gs[2].subgridspec(2,3,wspace=.23,hspace=.38)
        zoomaxes=[fig.add_subplot(sub[i,j]) for i in range(2) for j in range(3)]
        for ax in [ax0,ax1]+zoomaxes:
            ax.set_facecolor("black"); ax.grid(color="#2b3440",lw=.55,alpha=.72)
            for sp in ax.spines.values(): sp.set_color("#9fb0c0")

        ax0.plot(zg,pz,color="#62dfd1",lw=2.0,label=f"ALMA [O III] systemic posterior: z={Z_SYS:.4f} ± {SIGMA_Z:.4f}")
        ax0.axvline(Z_SYS,color="#ffd400",lw=1.4); ax0.axvspan(Z_SYS-SIGMA_Z,Z_SYS+SIGMA_Z,color="#ffd400",alpha=.12)
        ax0.set_xlabel("Systemic redshift, z"); ax0.set_ylabel("Normalized posterior density")
        ax0.set_title("Systemic redshift from the ALMA [O III] 88 μm centroid")
        lg=ax0.legend(facecolor="black",edgecolor="#666"); [t.set_color("white") for t in lg.get_texts()]

        ax1.fill_between(lam,y-s,y+s,step="mid",color="#a6a6a6",alpha=.28,label="Public 1σ uncertainty")
        ax1.step(lam,y,where="mid",color="#e8e8e8",lw=1.05,label="Zenodo JADES-GS-z14-0 PRISM spectrum")
        ax1.plot(lam,best,color="#d94343",lw=2.0,label=f"IGM + DLA fit: log NHI={pbest[2]:.2f}, β={pbest[1]:.2f}")
        ax1.plot(lam,best_igm,color="#63d4ee",ls="-.",lw=1.35,label=f"IGM-only comparison  Δχ²={chi0-chi:.1f}")
        lya=LYA_REST*(1+Z_SYS)
        ax1.axvline(lya,color="#ffd400",ls="--",lw=1.0,label=f"Systemic Lyα = {lya:.4f} μm")
        ax1.axvspan(1.84,1.92,color="#ff9f43",alpha=.08,label="Observed break / damping-wing region")
        for n in LINES:
            c=REST[n]*(1+Z_SYS)
            if c<=2.40:
                ax1.axvline(c,color="#8aa0b8",ls=":",lw=.65,alpha=.75)
                ax1.text(c,.965,n,rotation=90,transform=ax1.get_xaxis_transform(),ha="right",va="top",fontsize=8,color="#b8c7d8")
        ax1.set_xlim(1.68,2.40); ax1.set_xlabel("Observed wavelength [μm]")
        ax1.set_ylabel(r"$F_\lambda$ [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
        ax1.set_title("Public JWST/NIRSpec PRISM spectrum — native samples, uncertainty, IGM-only and IGM+DLA fits")
        lg=ax1.legend(facecolor="black",edgecolor="#666",fontsize=8,ncol=2); [t.set_color("white") for t in lg.get_texts()]

        for ax,n in zip(zoomaxes,LINES):
            cen=REST[n]*(1+Z_SYS); half=.055 if n!="C III]" else .070
            q=(wave>=cen-half)&(wave<=cen+half)
            ax.fill_between(wave[q],flux[q]-err[q],flux[q]+err[q],step="mid",color="#8f8f8f",alpha=.28)
            ax.step(wave[q],flux[q],where="mid",color="#67e0d1",lw=1.15)
            if np.any(q): ax.plot(wave[q],np.interp(wave[q],lam,best),color="#ffd400",lw=1.25)
            ax.axvline(cen,color="white",ls="--",lw=.75)
            ax.set_title(f"{n} reference window — {cen:.4f} μm",fontsize=10)
            ax.set_xlabel("Observed wavelength [μm]"); ax.set_ylabel(r"$F_\lambda$ [$10^{-21}$]")
            ax.text(.02,.94,"reference only — no detection assumed",transform=ax.transAxes,ha="left",va="top",fontsize=8,color="#b8c7d8")

        fig.suptitle("JADES-GS-z14-0 — ALMA systemic redshift + public PRISM DLA analysis + UV reference windows",fontsize=16,y=.995)
        fig.savefig(PNG/f"{VERSION}_JADES_GS_Z14_0_PRISM_DLA_REFERENCE_WINDOWS.png",dpi=430,bbox_inches="tight",facecolor="black")
        pd.DataFrame({"feature":["Lyα"]+LINES,"rest_wavelength_um":[REST["Lyα"]]+[REST[n] for n in LINES],"observed_wavelength_um":[REST["Lyα"]*(1+Z_SYS)]+[REST[n]*(1+Z_SYS) for n in LINES],"status":["systemic break anchor"]+["reference only"]*len(LINES)}).to_csv(CSV/f"{VERSION}_JADES_GS_Z14_0_REFERENCE_LINES.csv",index=False)
        pd.DataFrame({"wavelength_um":wave,"flux_1e-21":flux,"sigma_1e-21":err}).to_csv(CSV/f"{VERSION}_JADES_GS_Z14_0_PUBLIC_PRISM.csv",index=False)
        plt.show()

run.on_click(draw); display(run); display(out); draw()
