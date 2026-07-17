# JWST_0143
import sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
for p in ["numpy","pandas","matplotlib","astropy","requests","ipywidgets","scipy"]:
    try: __import__(p)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",p],check=True)
import numpy as np, pandas as pd, requests, matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import lsq_linear
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION="JWST_0143"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
print(f"CODE OUTPUT: {VERSION}")
print("MOM-z14 SIX OBSERVED UV LINES — INDIVIDUAL LIKELIHOODS + SLOPE-DERIVED REDSHIFT")
print("-"*112)

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
    wave=pick("wave","wavelength","lam"); flux=pick("flux","flux_corr","fnu"); err=pick("err","error","flux_err","sigma")
    if err is None:
        iv=pick("ivar","inverse_variance"); err=np.where(iv>0,1/np.sqrt(iv),np.nan)
if wave is None or flux is None or err is None: raise RuntimeError("Required SPEC1D columns not found")
conv=1e-6*1e-23*2.99792458e18/(wave*1e4)**2/1e-20
flam=flux*conv; sig=err*conv
m=np.isfinite(wave)&np.isfinite(flam)&np.isfinite(sig)&(sig>0)
wave,flam,sig=wave[m],flam[m],sig[m]; o=np.argsort(wave); wave,flam,sig=wave[o],flam[o],sig[o]

REST={"N IV]":0.1487,"C IV":0.15495,"He II":0.1640,"O III]":0.1663,"N III]":0.1750,"C III]":0.1908}
COLORS=plt.cm.tab10(np.linspace(0,1,6))
PUBLISHED_Z=14.44

display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst label,.jwst .widget-label{color:#fff!important;font-weight:600}.jwst input{background:#111!important;color:#fff!important;border:1px solid #777!important}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
zmin=widgets.FloatText(value=14.20,description="z min:",layout=widgets.Layout(width="155px")); zmax=widgets.FloatText(value=14.65,description="z max:",layout=widgets.Layout(width="155px")); dz=widgets.FloatText(value=0.0005,description="Δz:",layout=widgets.Layout(width="145px"))
fwhm=widgets.FloatSlider(value=0.025,min=0.010,max=0.060,step=0.001,description="Line FWHM [μm]:",layout=widgets.Layout(width="340px"),readout_format=".3f")
halfwin=widgets.FloatSlider(value=0.080,min=0.040,max=0.160,step=0.005,description="Half-window [μm]:",layout=widgets.Layout(width="340px"),readout_format=".3f")
run=widgets.Button(description="Run six-line derivation",button_style="success",layout=widgets.Layout(width="215px")); save=widgets.Button(description="Save current PNG",layout=widgets.Layout(width="180px"))
panel=widgets.HBox([zmin,zmax,dz,fwhm,halfwin,run,save],layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="8px")); panel.add_class("jwst")
out=widgets.Output(layout=widgets.Layout(border="1px solid #333")); last={"fig":None,"png":None}
plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white"})

def local_fit(rest,z,fw,hw):
    c=rest*(1+z); q=(wave>=c-hw)&(wave<=c+hw)
    x,y,e=wave[q],flam[q],sig[q]
    if len(x)<5: return np.inf,None,None,None,None
    gs=fw/2.354820045; g=np.exp(-0.5*((x-c)/gs)**2); A=np.column_stack([np.ones_like(x),x-c,g])
    Aw=A/e[:,None]; yw=y/e; sol=lsq_linear(Aw,yw,bounds=([-np.inf,-np.inf,0],[np.inf,np.inf,np.inf]))
    model=A@sol.x; chi=np.sum(((y-model)/e)**2)
    return chi,x,y,e,model

def posterior(zg,chi):
    p=np.exp(np.clip(-0.5*(chi-np.nanmin(chi)),-700,0)); a=np.trapz(p,zg); return p/a if a>0 else p

def run_plot(_=None):
    with out:
        clear_output(wait=True)
        zg=np.arange(float(zmin.value),float(zmax.value)+0.5*float(dz.value),float(dz.value)); fw=float(fwhm.value); hw=float(halfwin.value)
        curves={}; best_rows=[]; joint=np.zeros_like(zg)
        for name,rest in REST.items():
            chi=np.array([local_fit(rest,z,fw,hw)[0] for z in zg]); curves[name]=chi; joint+=chi-np.nanmin(chi)
            ib=int(np.nanargmin(chi)); zb=float(zg[ib]); best_rows.append({"feature":name,"rest_wavelength_um":rest,"best_z":zb,"observed_center_um":rest*(1+zb),"delta_z_from_14p44":zb-PUBLISHED_Z})
        df=pd.DataFrame(best_rows)
        # Weighted straight-line fit through origin: lambda_obs = slope * lambda_rest; slope = 1+z.
        x=df.rest_wavelength_um.to_numpy(); y=df.observed_center_um.to_numpy(); slope=float(np.sum(x*y)/np.sum(x*x)); z_slope=slope-1
        p_joint=posterior(zg,joint); z_joint=float(zg[np.argmax(p_joint)])
        df["slope_fit_predicted_um"]=slope*x; df["slope_residual_nm"]=(y-slope*x)*1000
        table_csv=CSV/f"{VERSION}_SIX_LINE_SLOPE_TABLE.csv"; df.to_csv(table_csv,index=False)
        scan=pd.DataFrame({"z":zg,"joint_delta_chi2":joint,"joint_posterior":p_joint})
        for name,chi in curves.items(): scan[f"{name}_delta_chi2"]=chi-np.nanmin(chi); scan[f"{name}_posterior"]=posterior(zg,chi)
        scan_csv=CSV/f"{VERSION}_SIX_LINE_LIKELIHOODS.csv"; scan.to_csv(scan_csv,index=False)

        fig=plt.figure(figsize=(16,15),facecolor="black"); gs=fig.add_gridspec(3,1,height_ratios=[1.35,1.25,2.25],hspace=.25)
        ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); ax2=fig.add_subplot(gs[2])
        for ax in (ax0,ax1,ax2):
            ax.set_facecolor("black"); [sp.set_color("white") for sp in ax.spines.values()]; ax.tick_params(colors="white"); ax.grid(color="#303030",lw=.6,alpha=.8)
        for color,(name,chi) in zip(COLORS,curves.items()): ax0.plot(zg,posterior(zg,chi),color=color,lw=1.5,label=f"{name}: z={df.loc[df.feature==name,'best_z'].iloc[0]:.4f}")
        ax0.plot(zg,p_joint,color="white",lw=3.0,label=f"Combined six-line likelihood: z={z_joint:.4f}")
        ax0.axvline(z_joint,color="#ffd400",ls="--",lw=1.5); ax0.axvline(PUBLISHED_Z,color="#ff6666",ls=":",lw=1.5,label="Published z=14.44")
        ax0.set_xlabel("Redshift, z [dimensionless]"); ax0.set_ylabel("Normalized likelihood density [dimensionless]")
        leg=ax0.legend(loc="upper left",ncol=2,fontsize=9,facecolor="black",edgecolor="#777"); [t.set_color("white") for t in leg.get_texts()]

        ax1.scatter(x,y,s=70,c=COLORS,edgecolors="white",linewidths=.6,zorder=3)
        xx=np.linspace(x.min()*0.97,x.max()*1.03,200); ax1.plot(xx,slope*xx,color="#ffd400",lw=2.2,label=f"Best line: λobs = {slope:.6f} λrest; z = slope − 1 = {z_slope:.6f}")
        for color,row in zip(COLORS,df.itertuples()): ax1.annotate(row.feature,(row.rest_wavelength_um,row.observed_center_um),xytext=(5,6),textcoords="offset points",color=color,fontsize=10)
        ax1.set_xlabel("Rest wavelength [μm]"); ax1.set_ylabel("Fitted observed wavelength [μm]")
        lg=ax1.legend(loc="upper left",facecolor="black",edgecolor="#777",fontsize=10); [t.set_color("white") for t in lg.get_texts()]

        offsets=np.arange(6)[::-1]*0.18
        for color,(name,rest),off in zip(COLORS,REST.items(),offsets):
            zb=float(df.loc[df.feature==name,"best_z"].iloc[0]); chi,xw,yw,ew,model=local_fit(rest,zb,fw,hw); center=rest*(1+zb)
            ax2.fill_between(xw,yw-ew+off,yw+ew+off,step="mid",color="#777",alpha=.23)
            ax2.step(xw,yw+off,where="mid",color=color,lw=1.2)
            ax2.plot(xw,model+off,color="white",lw=1.5)
            ax2.axvline(center,color=color,ls="--",lw=1.0)
            ax2.text(center,off+0.145,f"{name}  z={zb:.4f}",rotation=90,ha="right",va="top",color=color,fontsize=9)
        ax2.set_xlabel("Observed wavelength [μm]"); ax2.set_ylabel(r"Offset $f_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]")
        ax2.set_title("Six observed-line windows: spectrum, 1σ band, fitted Gaussian, and dashed fitted center",fontsize=12)
        fig.suptitle(f"MoM-z14 — how the six UV lines produce the redshift | combined likelihood z={z_joint:.4f} | slope z={z_slope:.4f}",fontsize=15,y=.995)
        fig.tight_layout(rect=[.03,.02,.995,.98]); png=PNG/f"{VERSION}_MOM_Z14_SIX_LINE_SLOPE_DERIVATION.png"; fig.savefig(png,dpi=450,bbox_inches="tight",facecolor="black"); plt.show()
        print("\nSIX-LINE REDSHIFT DERIVATION")
        print(df.to_string(index=False,float_format=lambda v:f"{v:.6f}"))
        print(f"\nCombined six-line likelihood z   {z_joint:.6f}")
        print(f"Observed-vs-rest slope            {slope:.6f}")
        print(f"Slope-derived z = slope - 1       {z_slope:.6f}")
        print(f"Published reference               {PUBLISHED_Z:.6f}")
        print(f"Line table CSV                    {table_csv}")
        print(f"Likelihood CSV                    {scan_csv}")
        print(f"Plot PNG                          {png}")
        print("NOTE: Each colored likelihood comes from one local line-window fit. The white curve combines all six; the middle panel independently converts the six fitted centers into z through the slope λobs=(1+z)λrest.")
        print(f"Timestamp UTC                     {datetime.now(timezone.utc).isoformat()}")
        print(f"# {VERSION}")
        last["fig"],last["png"]=fig,png

def save_plot(_):
    if last["fig"] is not None: last["fig"].savefig(last["png"],dpi=500,bbox_inches="tight",facecolor="black")
run.on_click(run_plot); save.on_click(save_plot); display(panel); display(out); run_plot()
