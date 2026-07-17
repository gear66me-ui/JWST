# JWST_0141
import sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
for pkg in ["numpy","pandas","matplotlib","astropy","requests","ipywidgets","scipy"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True)
import numpy as np, pandas as pd, requests, matplotlib.pyplot as plt
from astropy.io import fits
from scipy.special import erf
from scipy.optimize import lsq_linear
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION="JWST_0141"
ROOT=Path("/content/JWST_OUTPUT"); PNG_DIR=ROOT/"PNG"; CSV_DIR=ROOT/"CSV"; DATA_DIR=ROOT/"DATA"/VERSION
for d in (PNG_DIR,CSV_DIR,DATA_DIR): d.mkdir(parents=True,exist_ok=True)
print(f"CODE OUTPUT: {VERSION}")
print("MOM-z14 SHARED-z LIKELIHOOD + INDIVIDUAL LINE COMPONENT SLOPES")
print("-"*112)
SPEC_NAME="mom-cos04-v4_prism-clear_5224_277193.spec.fits"
SPEC_URL=f"https://s3.amazonaws.com/msaexp-nirspec/extractions/mom-cos04-v4/{SPEC_NAME}"
SPEC_PATH=DATA_DIR/SPEC_NAME
if not SPEC_PATH.exists() or SPEC_PATH.stat().st_size<100000:
    with requests.get(SPEC_URL,stream=True,timeout=120) as r:
        r.raise_for_status()
        with open(SPEC_PATH,"wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
with fits.open(SPEC_PATH,memmap=False) as hdul:
    tab=hdul["SPEC1D"].data; names={n.upper():n for n in tab.names}
    def pick(*keys):
        for k in keys:
            if k.upper() in names: return np.ravel(np.asarray(tab[names[k.upper()]],dtype=float))
        return None
    wave=pick("wave","wavelength","lam"); flux=pick("flux","flux_corr","fnu"); err=pick("err","error","flux_err","sigma")
    if err is None:
        ivar=pick("ivar","inverse_variance"); err=np.where(ivar>0,1/np.sqrt(ivar),np.nan)
if wave is None or flux is None or err is None: raise RuntimeError(f"Required columns unavailable: {tab.names}")
c_ang_s=2.99792458e18; lam_ang=wave*1e4; conv=1e-6*1e-23*c_ang_s/lam_ang**2/1e-20
flam=flux*conv; sigma=err*conv
m=np.isfinite(wave)&np.isfinite(flam)&np.isfinite(sigma)&(sigma>0)
wave,flam,sigma=wave[m],flam[m],sigma[m]; o=np.argsort(wave); wave,flam,sigma=wave[o],flam[o],sigma[o]
REST={"Lyα break":0.121567,"N IV]":0.1487,"C IV":0.15495,"He II":0.1640,"O III]":0.1663,"N III]":0.1750,"C III]":0.1908}
LINE_NAMES=list(REST)[1:]; PUBLISHED_Z=14.44; PUBLISHED_SIGMA=0.02

display(HTML("""<style>.jwst-dark{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst-dark label,.jwst-dark .widget-label{color:#fff!important;font-weight:600}.jwst-dark select,.jwst-dark input{background:#111!important;color:#fff!important;border:1px solid #777!important}.jwst-dark button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
zmin_w=widgets.FloatText(value=14.20,description="z min:",layout=widgets.Layout(width="150px")); zmax_w=widgets.FloatText(value=14.65,description="z max:",layout=widgets.Layout(width="150px")); step_w=widgets.FloatText(value=0.0005,description="Δz:",layout=widgets.Layout(width="145px"))
break_w=widgets.FloatSlider(value=0.018,min=0.005,max=0.050,step=0.001,description="Break σ [μm]:",layout=widgets.Layout(width="320px"),readout_format=".3f")
fwhm_w=widgets.FloatSlider(value=0.025,min=0.010,max=0.060,step=0.001,description="Line FWHM [μm]:",layout=widgets.Layout(width="320px"),readout_format=".3f")
range_w=widgets.FloatRangeSlider(value=(1.55,3.05),min=1.0,max=3.2,step=0.01,description="Fit λ [μm]:",layout=widgets.Layout(width="420px"),readout_format=".2f")
line_checks={n:widgets.Checkbox(value=True,description=n,indent=False,layout=widgets.Layout(width="105px")) for n in LINE_NAMES}
run_btn=widgets.Button(description="Run shared-z scan",button_style="success",layout=widgets.Layout(width="200px")); save_btn=widgets.Button(description="Save current PNG",layout=widgets.Layout(width="180px"))
panel=widgets.VBox([widgets.HBox([zmin_w,zmax_w,step_w,break_w,fwhm_w],layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="7px")),widgets.HBox([range_w]+list(line_checks.values())+[run_btn,save_btn],layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="7px"))]); panel.add_class("jwst-dark")
out=widgets.Output(layout=widgets.Layout(border="1px solid #333")); last={"fig":None,"png":None}
plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white"})

def design_matrix(lam,z,break_sigma,line_fwhm,selected):
    pivot=2.25; lya=REST["Lyα break"]*(1+z); red_step=0.5*(1+erf((lam-lya)/(np.sqrt(2)*break_sigma)))
    cols=[np.ones_like(lam),red_step,red_step*(lam-pivot)]; labels=["blue pedestal","red continuum","red slope"]
    gsig=line_fwhm/2.354820045
    for name in selected:
        center=REST[name]*(1+z); cols.append(np.exp(-0.5*((lam-center)/gsig)**2)); labels.append(name)
    return np.column_stack(cols),labels

def weighted_fit(lam,y,e,z,break_sigma,line_fwhm,selected):
    A,labels=design_matrix(lam,z,break_sigma,line_fwhm,selected); Aw=A/e[:,None]; yw=y/e
    lo=np.r_[np.full(3,-np.inf),np.zeros(len(selected))]; hi=np.full(A.shape[1],np.inf)
    sol=lsq_linear(Aw,yw,bounds=(lo,hi),method="trf",lsmr_tol="auto"); model=A@sol.x
    return np.sum(((y-model)/e)**2),model,sol.x,labels,A

def posterior_interval(zgrid,chi2):
    p=np.exp(np.clip(-0.5*(chi2-np.nanmin(chi2)),-700,0)); area=np.trapz(p,zgrid); p=p/area if area>0 else p
    cdf=np.zeros_like(p); cdf[1:]=np.cumsum(0.5*(p[1:]+p[:-1])*np.diff(zgrid)); cdf/=cdf[-1] if cdf[-1]>0 else 1
    return p,np.interp(.158655,cdf,zgrid),np.interp(.5,cdf,zgrid),np.interp(.841345,cdf,zgrid)

def run_scan(_=None):
    with out:
        clear_output(wait=True)
        zmin,zmax,dz=float(zmin_w.value),float(zmax_w.value),float(step_w.value); lo_lam,hi_lam=map(float,range_w.value)
        use=(wave>=lo_lam)&(wave<=hi_lam); lam,y,e=wave[use],flam[use],sigma[use]; selected=[n for n,c in line_checks.items() if c.value]
        zgrid=np.arange(zmin,zmax+0.5*dz,dz); chi2=np.empty_like(zgrid)
        for i,z in enumerate(zgrid): chi2[i]=weighted_fit(lam,y,e,z,break_w.value,fwhm_w.value,selected)[0]
        bi=int(np.nanargmin(chi2)); z_best=float(zgrid[bi]); p,q16,q50,q84=posterior_interval(zgrid,chi2)
        _,best_model,coeff,labels,A=weighted_fit(lam,y,e,z_best,break_w.value,fwhm_w.value,selected)
        components={labels[i]:A[:,i]*coeff[i] for i in range(len(labels))}; continuum=components["blue pedestal"]+components["red continuum"]+components["red slope"]
        slope_coeff=float(coeff[2]); gsig=fwhm_w.value/2.354820045
        rows=[]
        for name in selected:
            center=REST[name]*(1+z_best); amp=float(coeff[labels.index(name)]); peak_slope=amp*np.exp(-0.5)/(gsig)
            rows.append({"feature":name,"rest_um":REST[name],"observed_center_um":center,"amplitude_flam20":amp,"max_abs_profile_slope_flam20_per_um":peak_slope,"continuum_slope_flam20_per_um":slope_coeff})
        slope_df=pd.DataFrame(rows); slope_csv=CSV_DIR/f"{VERSION}_INDIVIDUAL_LINE_SLOPES.csv"; slope_df.to_csv(slope_csv,index=False)
        scan_csv=CSV_DIR/f"{VERSION}_SHARED_Z_LIKELIHOOD_SCAN.csv"; pd.DataFrame({"z":zgrid,"chi2":chi2,"delta_chi2":chi2-chi2[bi],"posterior_density":p}).to_csv(scan_csv,index=False)
        fig=plt.figure(figsize=(15,14),facecolor="black"); gs=fig.add_gridspec(3,1,height_ratios=[1.15,2.0,2.0],hspace=.23)
        axp=fig.add_subplot(gs[0]); axs=fig.add_subplot(gs[1]); axc=fig.add_subplot(gs[2],sharex=axs)
        for ax in (axp,axs,axc):
            ax.set_facecolor("black"); [sp.set_color("white") for sp in ax.spines.values()]; ax.tick_params(colors="white"); ax.grid(color="#303030",lw=.6,alpha=.8)
        axp.plot(zgrid,p,color="#67e0d1",lw=2,label="Shared-z posterior"); axp.axvspan(PUBLISHED_Z-PUBLISHED_SIGMA,PUBLISHED_Z+PUBLISHED_SIGMA,color="#777",alpha=.28,label="Published z=14.44±0.02"); axp.axvline(PUBLISHED_Z,color="white",ls="--",lw=1.3); axp.axvline(z_best,color="#ffd400",lw=1.8,label=f"Best scan z={z_best:.4f}"); axp.axvspan(q16,q84,color="#67e0d1",alpha=.16,label=f"68%: {q16:.4f}–{q84:.4f}"); axp.set_ylabel("Posterior density [dimensionless]"); axp.set_xlabel("Shared redshift, z [dimensionless]")
        axs.fill_between(lam,y-e,y+e,step="mid",color="#9a9a9a",alpha=.35,label="1σ uncertainty"); axs.step(lam,y,where="mid",color="#67e0d1",lw=1.6,label="DJA spectrum"); axs.plot(lam,best_model,color="#ffd400",lw=2,label=f"Best joint model, z={z_best:.4f}"); axs.plot(lam,continuum,color="white",ls=":",lw=1.5,label=f"Continuum slope={slope_coeff:.4f} per μm")
        colors=plt.cm.tab10(np.linspace(0,1,max(len(selected),1)))
        for color,name in zip(colors,selected):
            comp=components[name]; center=REST[name]*(1+z_best); axs.plot(lam,continuum+comp,color=color,lw=1.2,alpha=.95,label=f"{name} component")
            axc.plot(lam,comp,color=color,lw=1.8,label=f"{name} | max |slope|={slope_df.loc[slope_df.feature==name,'max_abs_profile_slope_flam20_per_um'].iloc[0]:.3f}")
            axc.axvline(center,color=color,ls="--",lw=.8,alpha=.8)
        axs.set_xlim(lo_lam,hi_lam); axs.set_ylabel(r"$f_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]"); axs.set_xlabel(r"Observed wavelength, $\lambda_{obs}$ [$\mu$m]")
        axc.axhline(0,color="#777",lw=.8); axc.set_ylabel(r"Individual fitted line component [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]"); axc.set_xlabel(r"Observed wavelength, $\lambda_{obs}$ [$\mu$m]")
        for ax in (axp,axs,axc):
            leg=ax.legend(loc="best",facecolor="black",edgecolor="#777",fontsize=8,ncol=2 if ax is not axp else 1); [t.set_color("white") for t in leg.get_texts()]
        fig.suptitle("MoM-z14 — shared-redshift likelihood, fitted continuum slope, and individual UV-line profiles",fontsize=15,y=.995); fig.tight_layout(rect=[.03,.02,.995,.98])
        png=PNG_DIR/f"{VERSION}_MOM_Z14_SHARED_Z_COMPONENT_SLOPES.png"; fig.savefig(png,dpi=450,bbox_inches="tight",facecolor="black"); plt.show()
        print("\nSHARED-z + SLOPE RESULT"); print(f"Best grid z                 {z_best:.6f}"); print(f"Posterior median z          {q50:.6f}"); print(f"68% credible interval       -{q50-q16:.6f} / +{q84-q50:.6f}"); print(f"Published reference         {PUBLISHED_Z:.6f} ± {PUBLISHED_SIGMA:.6f}"); print(f"Fitted red-continuum slope  {slope_coeff:.6f} [paper flux units per μm]")
        if not slope_df.empty: print(slope_df.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
        print(f"Likelihood CSV              {scan_csv}"); print(f"Line-slope CSV              {slope_csv}"); print(f"Plot PNG                    {png}"); print("NOTE: Individual colored curves are fitted Gaussian line templates at the common shared redshift; their quoted slope is the maximum absolute derivative of that template."); print(f"Timestamp UTC               {datetime.now(timezone.utc).isoformat()}"); print(f"# {VERSION}")
        last["fig"],last["png"]=fig,png

def save_current(_):
    if last["fig"] is not None: last["fig"].savefig(last["png"],dpi=500,bbox_inches="tight",facecolor="black")
run_btn.on_click(run_scan); save_btn.on_click(save_current); display(panel); display(out); run_scan()
