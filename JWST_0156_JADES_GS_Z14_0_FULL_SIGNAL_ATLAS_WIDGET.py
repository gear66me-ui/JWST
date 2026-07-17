# JWST_0156
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

VERSION="JWST_0156"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
URL="https://zenodo.org/records/12578543/files/JADES-GS-z14-0_spec1D.txt?download=1"
PATH=DATA/"JADES-GS-z14-0_spec1D.txt"
ZSYS=14.1793; SZ=0.0007; LYA=0.121567; CIII=0.1908
NUREST=3393.006244; NUOBS=NUREST/(1+ZSYS); FWHM_KMS=100.0
if not PATH.exists() or PATH.stat().st_size<10000:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        r=requests.get(URL,timeout=180); r.raise_for_status(); PATH.write_bytes(r.content)
arr=np.loadtxt(PATH,comments="#"); arr=arr[None,:] if arr.ndim==1 else arr
wave,flux,err=[np.asarray(arr[:,i],float) for i in range(3)]
if np.nanmedian(wave)>100: wave/=1e4
elif np.nanmedian(wave)>10: wave/=1e3
m=np.isfinite(wave)&np.isfinite(flux)&np.isfinite(err)&(err>0)&(wave>0.7)&(wave<5.3)
wave,flux,err=wave[m],flux[m],err[m]; o=np.argsort(wave); wave,flux,err=wave[o],flux[o],err[o]
med=np.nanmedian(np.abs(flux[(wave>1.95)&(wave<2.25)]))
if med<1e-12: flux/=1e-21; err/=1e-21

C=2.99792458e10; E=4.803204712e-10; ME=9.1093837e-28
FOSC=.4164; GAMMA=6.265e8; LCM=1215.67e-8; NU0=C/LCM; B=20e5
DNU=NU0*B/C; SIG0=np.sqrt(np.pi)*E**2/(ME*C)*FOSC/DNU

def tau_dla(lam,logn):
    lr=(lam/(1+ZSYS))*1e-4; nu=C/lr; u=(nu-NU0)/DNU; a=GAMMA/(4*np.pi*DNU)
    return (10**logn)*SIG0*np.real(wofz(u+1j*a))

def conv_prism(lam,model,R=100):
    g=np.linspace(lam.min(),lam.max(),max(3000,len(lam)*5)); mg=np.interp(g,lam,model)
    dl=np.nanmedian(np.diff(g)); sl=np.nanmedian(g)/(R*2.354820045)
    return np.interp(lam,g,gaussian_filter1d(mg,max(sl/dl,.5),mode="nearest"))

def model(lam,p,dla=True):
    amp,beta,logn,xhi=p; cont=amp*(lam/2.05)**beta; edge=LYA*(1+ZSYS)
    blue=.5*(1+np.tanh((lam-edge)/.004)); red=np.maximum(lam-edge,1e-4)
    tr=np.where(lam<edge,1e-5,np.exp(-xhi*.018/red))
    if dla: tr*=np.exp(-tau_dla(lam,logn))
    return conv_prism(lam,cont*blue*tr)

fm=(wave>=1.68)&(wave<=2.40); lam,y,s=wave[fm],flux[fm],err[fm]
lo=[.01,-4,20,0]; hi=[20,1,24,1]; init=[3.2,-1.93,22.27,.59]
res=least_squares(lambda p:(y-model(lam,p,True))/s,init,bounds=(lo,hi),max_nfev=500)
pb=res.x; best=model(lam,pb,True)
res0=least_squares(lambda p:(y-model(lam,p,False))/s,init,bounds=(lo,hi),max_nfev=500)
best0=model(lam,res0.x,False); chi=np.sum(((y-best)/s)**2); chi0=np.sum(((y-best0)/s)**2)
zg=np.linspace(ZSYS-.004,ZSYS+.004,1600); pz=np.exp(-.5*((zg-ZSYS)/SZ)**2); pz/=np.trapz(pz,zg)

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"#9fb0c0","xtick.color":"white","ytick.color":"white","font.size":10})
display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
run=widgets.Button(description="Plot full signal atlas",button_style="success",layout=widgets.Layout(width="210px")); run.add_class("jwst")
out=widgets.Output(layout=widgets.Layout(border="1px solid #333"))

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,20),facecolor="black")
        gs=fig.add_gridspec(4,1,height_ratios=[.75,1.8,1.7,1.35],hspace=.27)
        ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); ax2=fig.add_subplot(gs[2])
        bot=gs[3].subgridspec(1,2,wspace=.22); ax3=fig.add_subplot(bot[0]); ax4=fig.add_subplot(bot[1])
        for ax in (ax0,ax1,ax2,ax3,ax4):
            ax.grid(color="#2b3440",lw=.55,alpha=.72)
            for sp in ax.spines.values(): sp.set_color("#9fb0c0")
        ax0.plot(zg,pz,color="#62dfd1",lw=2,label=f"ALMA [O III] systemic posterior: z={ZSYS:.4f} ± {SZ:.4f}")
        ax0.axvline(ZSYS,color="#ffd400",lw=1.4); ax0.axvspan(ZSYS-SZ,ZSYS+SZ,color="#ffd400",alpha=.12)
        ax0.set_xlabel("Systemic redshift, z"); ax0.set_ylabel("Normalized posterior density"); ax0.legend(facecolor="black",edgecolor="#666")
        ax0.set_title("Systemic redshift anchored by the detected ALMA [O III] 88 μm line")

        ax1.fill_between(wave,flux-err,flux+err,step="mid",color="#999",alpha=.25,label="1σ uncertainty")
        ax1.step(wave,flux,where="mid",color="#e6e6e6",lw=.8,label="Complete public JWST/NIRSpec PRISM spectrum")
        ax1.axvspan(1.68,2.40,color="#d94343",alpha=.055,label="DLA fit region")
        ax1.axvline(LYA*(1+ZSYS),color="#ffd400",ls="--",lw=1,label="Systemic Lyα")
        ax1.axvline(CIII*(1+ZSYS),color="#67e0d1",ls=":",lw=1,label="Marginal C III] position")
        ax1.set_xlim(.8,5.2); ax1.set_xlabel("Observed wavelength [μm]")
        ax1.set_ylabel(r"$F_\lambda$ [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
        ax1.set_title("Complete released PRISM spectrum — no artificial UV peaks added")
        ax1.legend(facecolor="black",edgecolor="#666",fontsize=8,ncol=2)

        ax2.fill_between(lam,y-s,y+s,step="mid",color="#999",alpha=.28,label="1σ uncertainty")
        ax2.step(lam,y,where="mid",color="#f0f0f0",lw=1.05,label="Observed PRISM samples")
        ax2.plot(lam,best,color="#d94343",lw=2,label=f"IGM + DLA: log NHI={pb[2]:.2f}, β={pb[1]:.2f}")
        ax2.plot(lam,best0,color="#63d4ee",ls="-.",lw=1.3,label=f"IGM only; Δχ²={chi0-chi:.1f}")
        ax2.axvline(LYA*(1+ZSYS),color="#ffd400",ls="--",lw=1,label=f"Lyα systemic = {LYA*(1+ZSYS):.4f} μm")
        ax2.set_xlim(1.68,2.40); ax2.set_xlabel("Observed wavelength [μm]"); ax2.set_ylabel(r"$F_\lambda$ [$10^{-21}$]")
        ax2.set_title("Detected absorption signal: broad Lyα damping wing")
        ax2.legend(facecolor="black",edgecolor="#666",fontsize=8,ncol=2)

        cen=CIII*(1+ZSYS); q=(wave>cen-.12)&(wave<cen+.12)
        ax3.fill_between(wave[q],flux[q]-err[q],flux[q]+err[q],step="mid",color="#999",alpha=.28)
        ax3.step(wave[q],flux[q],where="mid",color="#67e0d1",lw=1.15)
        ax3.axvline(cen,color="white",ls="--",lw=.9,label=f"C III] expected = {cen:.4f} μm")
        ax3.set_xlabel("Observed wavelength [μm]"); ax3.set_ylabel(r"$F_\lambda$ [$10^{-21}$]")
        ax3.set_title("Marginal JWST C III] signal"); ax3.legend(facecolor="black",edgecolor="#666",fontsize=8)

        vel=np.linspace(-350,350,1000); sigv=FWHM_KMS/2.354820045; prof=np.exp(-.5*(vel/sigv)**2)
        ax4.plot(vel,prof,color="#ffd400",lw=2,label=f"Reported [O III] 88 μm profile; νobs={NUOBS:.4f} GHz")
        ax4.fill_between(vel,0,prof,color="#ffd400",alpha=.12); ax4.axvline(0,color="white",ls="--",lw=.8)
        ax4.set_xlabel("Velocity offset [km s$^{-1}$]"); ax4.set_ylabel("Normalized line intensity")
        ax4.set_title("Detected ALMA [O III] 88 μm line (reported centroid/profile)")
        ax4.legend(facecolor="black",edgecolor="#666",fontsize=8)

        fig.suptitle("JADES-GS-z14-0 — complete PRISM data and only the reported detected/marginal signals",fontsize=16,y=.995)
        fig.savefig(PNG/f"{VERSION}_JADES_GS_Z14_0_FULL_SIGNAL_ATLAS.png",dpi=430,bbox_inches="tight",facecolor="black")
        pd.DataFrame({"wavelength_um":wave,"flux_1e-21":flux,"sigma_1e-21":err}).to_csv(CSV/f"{VERSION}_JADES_GS_Z14_0_FULL_PRISM.csv",index=False)
        pd.DataFrame({"signal":["Lyα DLA absorption","C III] marginal emission","ALMA [O III] 88 μm emission"],"observed_coordinate":[LYA*(1+ZSYS),cen,NUOBS],"unit":["μm","μm","GHz"],"status":["detected broad absorption","marginal","detected systemic anchor"]}).to_csv(CSV/f"{VERSION}_JADES_GS_Z14_0_SIGNALS.csv",index=False)
        plt.show()
run.on_click(draw); display(run); display(out); draw()
