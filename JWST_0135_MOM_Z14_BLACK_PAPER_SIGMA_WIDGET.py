# JWST_0135
import sys, subprocess
from pathlib import Path
from datetime import datetime, timezone
for pkg in ["astropy","ipywidgets","requests","pandas","matplotlib","numpy"]:
    try: __import__(pkg)
    except Exception: subprocess.run([sys.executable,"-m","pip","install","-q",pkg],check=True)
import numpy as np, pandas as pd, requests, matplotlib.pyplot as plt
from astropy.io import fits
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION="JWST_0135"
ROOT=Path("/content/JWST_OUTPUT"); PNG_DIR=ROOT/"PNG"; CSV_DIR=ROOT/"CSV"; DATA_DIR=ROOT/"DATA"/VERSION
for d in (PNG_DIR,CSV_DIR,DATA_DIR): d.mkdir(parents=True,exist_ok=True)
print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 BLACK PAPER-STYLE SPECTRUM WITH 1σ / 2σ UNCERTAINTY BAND")
print("-"*112)

SPEC_NAME="mom-cos04-v4_prism-clear_5224_277193.spec.fits"
SPEC_URL=f"https://s3.amazonaws.com/msaexp-nirspec/extractions/mom-cos04-v4/{SPEC_NAME}"
SPEC_PATH=DATA_DIR/SPEC_NAME
if not SPEC_PATH.exists() or SPEC_PATH.stat().st_size<100000:
    print(f"DOWNLOADING: {SPEC_NAME}")
    with requests.get(SPEC_URL,stream=True,timeout=120) as r:
        r.raise_for_status()
        with open(SPEC_PATH,"wb") as f:
            for chunk in r.iter_content(1024*1024):
                if chunk: f.write(chunk)
else: print(f"USING CACHED FILE: {SPEC_PATH}")

with fits.open(SPEC_PATH,memmap=False) as hdul:
    tab=hdul["SPEC1D"].data; names={n.upper():n for n in tab.names}
    def pick(*opts):
        for key in opts:
            if key.upper() in names: return np.ravel(np.asarray(tab[names[key.upper()]],dtype=float))
        return None
    wave_all=pick("wave","wavelength","lam")
    flux_all=pick("flux","flux_corr","fnu")
    err_all=pick("err","error","flux_err","sigma")
    if err_all is None:
        ivar=pick("ivar","inverse_variance")
        err_all=np.where(ivar>0,1/np.sqrt(ivar),np.nan) if ivar is not None else np.full_like(flux_all,np.nan)
    sci=np.asarray(hdul["SCI"].data,dtype=float)
    wht=np.asarray(hdul["WHT"].data,dtype=float)
if wave_all is None or flux_all is None: raise RuntimeError(f"Required columns missing: {tab.names}")

c_ang_s=2.99792458e18
lam_ang=wave_all*1e4
conv=1e-6*1e-23*c_ang_s/lam_ang**2/1e-20
flam_all=flux_all*conv; err20_all=err_all*conv
valid=np.isfinite(wave_all)&np.isfinite(flam_all)&np.isfinite(err20_all)
wave=wave_all[valid]; flam20=flam_all[valid]; err20=err20_all[valid]
order=np.argsort(wave); wave,flam20,err20=wave[order],flam20[order],err20[order]

sn2d=sci*np.sqrt(np.clip(wht,0,None)); finite_sn=sn2d[np.isfinite(sn2d)]
lim=np.nanpercentile(np.abs(finite_sn),98.5) if finite_sn.size else 1.0
if not np.isfinite(lim) or lim<=0: lim=1.0
full_wave=wave_all.copy()
if len(full_wave)!=sn2d.shape[1]: full_wave=np.linspace(np.nanmin(wave),np.nanmax(wave),sn2d.shape[1])

csv_path=CSV_DIR/f"{VERSION}_MOM_Z14_DJA_PAPER_UNITS.csv"
pd.DataFrame({"wavelength_um":wave,"flux_flam_1e-20_erg_s_cm2_A":flam20,"sigma_flam_1e-20_erg_s_cm2_A":err20}).to_csv(csv_path,index=False)

LINES={"Lyα":0.121567*15.44,"N IV]":0.1487*15.44,"C IV":0.15495*15.44,"He II":0.1640*15.44,"O III]":0.1663*15.44,"N III]":0.1750*15.44,"C III]":0.1908*15.44}
WINDOWS={
"Full paper region 1.0–3.0 μm":(1.0,3.0),"1.0–1.5 μm":(1.0,1.5),"1.5–2.0 μm":(1.5,2.0),
"Lyα break 1.75–2.05 μm":(1.75,2.05),"N IV] 2.20–2.36 μm":(2.20,2.36),"C IV 2.32–2.46 μm":(2.32,2.46),
"He II / O III] 2.46–2.64 μm":(2.46,2.64),"N III] 2.62–2.78 μm":(2.62,2.78),"C III] 2.86–3.00 μm":(2.86,3.00),
"2.0–2.5 μm":(2.0,2.5),"2.5–3.0 μm":(2.5,3.0)}

display(HTML("""<style>
.jwst-dark-panel{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst-dark-panel label,.jwst-dark-panel .widget-label{color:#fff!important;font-weight:600}.jwst-dark-panel select,.jwst-dark-panel input{background:#111!important;color:#fff!important;border:1px solid #777!important}.jwst-dark-panel button{background:#202020!important;color:#fff!important;border:1px solid #888!important}
</style>"""))
window_dd=widgets.Dropdown(options=list(WINDOWS),value="Full paper region 1.0–3.0 μm",description="Window:",layout=widgets.Layout(width="520px"),style={"description_width":"80px"})
sigma_dd=widgets.Dropdown(options=[("1 sigma",1.0),("2 sigma",2.0)],value=1.0,description="Band:",layout=widgets.Layout(width="190px"),style={"description_width":"55px"})
ymin_box=widgets.FloatText(value=-0.10,description="Y min:",layout=widgets.Layout(width="170px"),style={"description_width":"55px"})
ymax_box=widgets.FloatText(value=0.50,description="Y max:",layout=widgets.Layout(width="170px"),style={"description_width":"55px"})
save_btn=widgets.Button(description="Save current PNG",layout=widgets.Layout(width="175px"))
controls=widgets.HBox([window_dd,sigma_dd,ymin_box,ymax_box,save_btn],layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="8px",width="100%")); controls.add_class("jwst-dark-panel")
out=widgets.Output(layout=widgets.Layout(border="1px solid #333")); last={"fig":None,"name":None}
plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white"})

def draw(*_):
    x0,x1=WINDOWS[window_dd.value]; nsig=float(sigma_dd.value)
    m=(wave>=x0)&(wave<=x1); cols=(full_wave>=x0)&(full_wave<=x1)
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,9),facecolor="black")
        gs=fig.add_gridspec(2,1,height_ratios=[1.0,4.3],hspace=0.04)
        ax2=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1],sharex=ax2)
        for ax in (ax1,ax2):
            ax.set_facecolor("black")
            for sp in ax.spines.values(): sp.set_color("white"); sp.set_linewidth(1.1)
            ax.tick_params(colors="white",labelcolor="white")
        if np.any(cols):
            sub=sn2d[:,cols]
            ax2.imshow(sub,origin="lower",aspect="auto",extent=[x0,x1,-0.5,sub.shape[0]-0.5],cmap="RdBu_r",vmin=-lim,vmax=lim,interpolation="nearest")
        ax2.set_ylabel("2-D signal-to-noise\n[dimensionless]",fontsize=12,labelpad=12)
        ax2.set_yticks([]); ax2.tick_params(axis="x",labelbottom=False)
        if np.any(m):
            lo=flam20[m]-nsig*err20[m]; hi=flam20[m]+nsig*err20[m]
            ax1.fill_between(wave[m],lo,hi,step="mid",color="#9a9a9a",alpha=0.38,zorder=1,label=f"{nsig:.0f}σ uncertainty")
            ax1.step(wave[m],flam20[m],where="mid",color="#67e0d1",linewidth=1.9,zorder=3,label="DJA flux")
        for label,xpos in LINES.items():
            if x0<=xpos<=x1:
                ax1.axvline(xpos,color="#aaaaaa",ls="--",lw=0.9,zorder=0)
                ax1.text(xpos,ymax_box.value*0.965,label,rotation=90,ha="right",va="top",fontsize=10,color="white")
        ax1.axhline(0,color="#777",lw=0.8)
        ax1.set_xlim(x0,x1); ax1.set_ylim(ymin_box.value,ymax_box.value)
        ax1.set_xlabel(r"Observed wavelength, $\lambda_{obs}$ [$\mu$m]",fontsize=15,labelpad=12)
        ax1.set_ylabel(r"Spectral flux density, $f_\lambda$ [$10^{-20}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]",fontsize=14,labelpad=14)
        ax1.grid(color="#333",lw=0.55,alpha=0.7); ax1.tick_params(labelsize=12,direction="out",length=5,width=1.0)
        leg=ax1.legend(loc="upper right",frameon=True,facecolor="black",edgecolor="#777",fontsize=10)
        for t in leg.get_texts(): t.set_color("white")
        npts=int(np.count_nonzero(m))
        fig.suptitle(f"MoM-z14 — DJA/msaexp PRISM/CLEAR | {x0:.2f}–{x1:.2f} μm observed | {npts} native samples | {nsig:.0f}σ band",fontsize=15,y=0.985)
        fig.tight_layout(rect=[0.03,0.025,0.995,0.965]); plt.show()
        last["fig"]=fig; last["name"]=f"{VERSION}_MOM_Z14_{x0:.2f}_{x1:.2f}UM_{int(nsig)}SIGMA_BLACK.png".replace(".","p")

def save_current(_):
    if last["fig"] is not None:
        path=PNG_DIR/last["name"]; last["fig"].savefig(path,dpi=500,bbox_inches="tight",facecolor="black"); print(f"SAVED: {path}")
for w in (window_dd,sigma_dd,ymin_box,ymax_box): w.observe(draw,names="value")
save_btn.on_click(save_current)
display(controls); display(out); draw()
default_png=PNG_DIR/f"{VERSION}_MOM_Z14_1p0_TO_3p0UM_1SIGMA_BLACK.png"
if last["fig"] is not None: last["fig"].savefig(default_png,dpi=500,bbox_inches="tight",facecolor="black")
print("\nOUTPUT SUMMARY")
print(f"DJA spectrum:       {SPEC_PATH}")
print(f"Paper-units CSV:    {csv_path}")
print(f"Default black PNG:  {default_png}")
print(f"Native samples:     {len(wave)}")
print(f"Timestamp UTC:      {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")