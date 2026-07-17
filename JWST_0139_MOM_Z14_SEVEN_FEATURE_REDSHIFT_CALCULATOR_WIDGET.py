# JWST_0139
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

VERSION="JWST_0139"
ROOT=Path("/content/JWST_OUTPUT"); PNG_DIR=ROOT/"PNG"; CSV_DIR=ROOT/"CSV"; DATA_DIR=ROOT/"DATA"/VERSION
for d in (PNG_DIR,CSV_DIR,DATA_DIR): d.mkdir(parents=True,exist_ok=True)
print(f"CODE OUTPUT: {VERSION}")
print("MOM-Z14 SEVEN-FEATURE REDSHIFT CALCULATOR — DJA/msaexp PRISM/CLEAR")
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
    wave=pick("wave","wavelength","lam")
    flux=pick("flux","flux_corr","fnu")
    err=pick("err","error","flux_err","sigma")
    if err is None:
        ivar=pick("ivar","inverse_variance")
        err=np.where(ivar>0,1/np.sqrt(ivar),np.nan) if ivar is not None else np.full_like(flux,np.nan)
if wave is None or flux is None: raise RuntimeError(f"Required columns missing: {tab.names}")
valid=np.isfinite(wave)&np.isfinite(flux)&np.isfinite(err)&(err>0)
wave,flux,err=wave[valid],flux[valid],err[valid]
order=np.argsort(wave); wave,flux,err=wave[order],flux[order],err[order]

FEATURES=[
    ("Lyα break",0.121567,"break"),
    ("N IV]",0.14870,"emission"),
    ("C IV",0.15495,"emission"),
    ("He II",0.16400,"emission"),
    ("O III]",0.16630,"emission"),
    ("N III]",0.17500,"emission"),
    ("C III]",0.19080,"emission"),
]
Z_REF=14.44; Z_REF_ERR=0.02

def local_step_unc(x):
    i=int(np.argmin(np.abs(wave-x)))
    ds=[]
    if i>0: ds.append(abs(wave[i]-wave[i-1]))
    if i<len(wave)-1: ds.append(abs(wave[i+1]-wave[i]))
    return 0.5*np.nanmedian(ds) if ds else 0.01

def auto_break():
    m=(wave>=1.72)&(wave<=2.04)
    x,y=wave[m],flux[m]
    if len(x)<5: return 0.121567*(1+Z_REF)
    ys=pd.Series(y).rolling(3,center=True,min_periods=1).median().to_numpy()
    grad=np.gradient(ys,x)
    return float(x[np.nanargmax(grad)])

def auto_emission(rest,half_width):
    expected=rest*(1+Z_REF)
    m=(wave>=expected-half_width)&(wave<=expected+half_width)
    if not np.any(m): return expected
    x,y,e=wave[m],flux[m],err[m]
    baseline=np.nanmedian(y)
    sn=(y-baseline)/e
    return float(x[np.nanargmax(sn)])

half_width=widgets.FloatSlider(value=0.055,min=0.020,max=0.120,step=0.005,description="Peak ± μm:",continuous_update=False,layout=widgets.Layout(width="420px"),style={"description_width":"90px"})
obs_boxes={}
for name,rest,kind in FEATURES:
    val=auto_break() if kind=="break" else auto_emission(rest,half_width.value)
    obs_boxes[name]=widgets.FloatText(value=val,description=name+":",layout=widgets.Layout(width="245px"),style={"description_width":"85px"})
auto_btn=widgets.Button(description="Auto-detect all",button_style="info")
calc_btn=widgets.Button(description="Recalculate",button_style="success")
save_btn=widgets.Button(description="Save PNG + CSV")
out=widgets.Output(); last={"fig":None,"df":None}

display(HTML("""<style>
.jwst-dark{background:#000;padding:12px;border:1px solid #555;border-radius:8px}.jwst-dark label,.jwst-dark .widget-label{color:white!important;font-weight:600}.jwst-dark input,.jwst-dark select{background:#111!important;color:white!important;border:1px solid #777!important}.jwst-dark button{color:white!important}
</style>"""))
controls=widgets.VBox([
    widgets.HBox([half_width,auto_btn,calc_btn,save_btn],layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="8px")),
    widgets.HBox(list(obs_boxes.values()),layout=widgets.Layout(display="flex",flex_flow="row wrap",gap="6px"))
]); controls.add_class("jwst-dark")

plt.rcParams.update({"figure.facecolor":"black","axes.facecolor":"black","savefig.facecolor":"black","text.color":"white","axes.labelcolor":"white","axes.edgecolor":"white","xtick.color":"white","ytick.color":"white"})

def detect_all(_=None):
    for name,rest,kind in FEATURES:
        obs_boxes[name].value=auto_break() if kind=="break" else auto_emission(rest,half_width.value)
    calculate()

def calculate(_=None):
    rows=[]
    for idx,(name,rest,kind) in enumerate(FEATURES,1):
        obs=float(obs_boxes[name].value)
        z=obs/rest-1.0
        sig_lam=local_step_unc(obs)
        sig_z=sig_lam/rest
        rows.append({"rank":idx,"feature":name,"type":kind,"rest_um":rest,"observed_um":obs,"z":z,"sigma_z_sampling":sig_z,"delta_z_vs_14p44":z-Z_REF})
    df=pd.DataFrame(rows)
    emission=df[df["type"]=="emission"].copy()
    z_mean=float(emission["z"].mean())
    z_std=float(emission["z"].std(ddof=1))
    z_sem=z_std/np.sqrt(len(emission))
    w=1/np.square(emission["sigma_z_sampling"].to_numpy())
    z_wmean=float(np.sum(w*emission["z"])/np.sum(w))
    z_werr=float(np.sqrt(1/np.sum(w)))
    z_break=float(df.loc[df["type"]=="break","z"].iloc[0])
    z_break_err=float(df.loc[df["type"]=="break","sigma_z_sampling"].iloc[0])
    wc=np.array([1/max(z_werr,1e-9)**2,1/max(z_break_err,1e-9)**2])
    combined=float(np.sum(wc*np.array([z_wmean,z_break]))/np.sum(wc))
    combined_err=float(np.sqrt(1/np.sum(wc)))
    summary=pd.DataFrame([
        ["Emission-line arithmetic mean",z_mean,z_sem,"6 emission complexes"],
        ["Emission-line weighted mean",z_wmean,z_werr,"sampling-weighted"],
        ["Lyα break",z_break,z_break_err,"maximum positive continuum gradient"],
        ["Combined weighted estimate",combined,combined_err,"weighted line mean + Lyα break"],
        ["Published reference",Z_REF,Z_REF_ERR,"MoM-z14 paper: z_spec = 14.44 ± 0.02"],
    ],columns=["estimator","z","plus_minus","notes"])
    with out:
        clear_output(wait=True)
        print("\nINDIVIDUAL FEATURE REDSHIFTS")
        print(df[["rank","feature","rest_um","observed_um","z","sigma_z_sampling","delta_z_vs_14p44"]].to_string(index=False,formatters={c:(lambda v:f"{v:.6f}") for c in ["rest_um","observed_um","z","sigma_z_sampling","delta_z_vs_14p44"]}))
        print("\nCOMBINED RESULTS")
        print(summary.to_string(index=False,formatters={"z":lambda v:f"{v:.6f}","plus_minus":lambda v:f"{v:.6f}"}))
        print("\nNOTE: the paper used a joint spectral model; this widget's simple/weighted averages are an independent engineering audit, not a reproduction of the paper posterior.")
        fig,ax=plt.subplots(figsize=(13,7),facecolor="black")
        y=np.arange(len(df))
        ax.errorbar(df["z"],y,xerr=df["sigma_z_sampling"],fmt="o",color="cyan",ecolor="gray",capsize=3,label="Feature-derived z")
        ax.axvspan(Z_REF-Z_REF_ERR,Z_REF+Z_REF_ERR,color="white",alpha=0.15,label="Published ±0.02")
        ax.axvline(Z_REF,color="white",ls="--",lw=1.3,label="Published z=14.44")
        ax.axvline(combined,color="gold",ls="-",lw=1.5,label=f"Combined z={combined:.4f}")
        ax.set_yticks(y); ax.set_yticklabels(df["feature"]); ax.invert_yaxis()
        ax.set_xlabel("Redshift, z [dimensionless]"); ax.set_ylabel("Observed spectral feature")
        ax.set_title("MoM-z14 — redshift from seven observed features")
        ax.grid(color="#333",alpha=0.7); leg=ax.legend(facecolor="black",edgecolor="#777"); [t.set_color("white") for t in leg.get_texts()]
        fig.tight_layout(); plt.show()
        last["fig"]=fig; last["df"]=(df,summary)

def save_outputs(_=None):
    if last["fig"] is None: calculate()
    png=PNG_DIR/f"{VERSION}_MOM_Z14_SEVEN_FEATURE_REDSHIFT.png"
    csv1=CSV_DIR/f"{VERSION}_MOM_Z14_FEATURE_REDSHIFTS.csv"
    csv2=CSV_DIR/f"{VERSION}_MOM_Z14_REDSHIFT_SUMMARY.csv"
    last["fig"].savefig(png,dpi=500,bbox_inches="tight",facecolor="black")
    last["df"][0].to_csv(csv1,index=False); last["df"][1].to_csv(csv2,index=False)
    print(f"SAVED: {png}\nSAVED: {csv1}\nSAVED: {csv2}")

auto_btn.on_click(detect_all); calc_btn.on_click(calculate); save_btn.on_click(save_outputs)
display(controls); display(out); calculate()
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
