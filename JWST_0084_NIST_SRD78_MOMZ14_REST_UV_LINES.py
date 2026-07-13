#!/usr/bin/env python3
"""NIST SRD 78 rest-UV reference-line study for MoM-z14.

Queries N IV], N III], C IV, C III], He II and O III] from the NIST Atomic
Spectra Database, saves the returned tables, and plots discrete lines plus a
narrow Gaussian visualization. The traces are reference visualizations, not
an observed MoM-z14 spectrum or a physical plasma model.
"""
from __future__ import annotations
import importlib.util, io, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path


def packages():
    req={"requests":"requests","numpy":"numpy","pandas":"pandas",
         "matplotlib":"matplotlib","lxml":"lxml"}
    miss=[p for m,p in req.items() if importlib.util.find_spec(m) is None]
    if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
packages()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION="JWST_0084"
ENDPOINT="https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
C_NM_PHZ=299.792458
EPS=1e-30
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"
PNG.mkdir(parents=True,exist_ok=True); CSV.mkdir(parents=True,exist_ok=True)

Q=[
 {"key":"N_IV","spectrum":"N IV","label":"N IV]","note":"N³⁺ intercombination complex","lo":147.5,"hi":149.2,"group":"N"},
 {"key":"N_III","spectrum":"N III","label":"N III]","note":"N²⁺ ultraviolet multiplet","lo":174.0,"hi":176.0,"group":"N"},
 {"key":"C_IV","spectrum":"C IV","label":"C IV","note":"C³⁺ resonance-doublet region","lo":153.8,"hi":155.8,"group":"C"},
 {"key":"C_III","spectrum":"C III","label":"C III]","note":"C²⁺ intercombination-doublet region","lo":189.7,"hi":191.8,"group":"C"},
 {"key":"He_II","spectrum":"He II","label":"He II","note":"He⁺ recombination-line region","lo":163.3,"hi":164.7,"group":"blend"},
 {"key":"O_III","spectrum":"O III","label":"O III]","note":"O²⁺ intercombination region","lo":165.4,"hi":167.2,"group":"blend"},
]
COL={"N IV]":"tab:orange","N III]":"tab:red","C IV":"tab:blue",
     "C III]":"tab:cyan","He II":"tab:purple","O III]":"tab:green"}


def session():
    retry=Retry(total=5,connect=5,read=5,backoff_factor=1.5,
                status_forcelist=(429,500,502,503,504),allowed_methods=frozenset({"GET"}))
    s=requests.Session(); s.mount("https://",HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent":f"{VERSION}/Colab educational NIST-SRD78 client",
                      "Accept":"text/csv,text/plain,text/html;q=0.8,*/*;q=0.5"})
    return s


def params(q):
    return {"spectra":q["spectrum"],"limits_type":"0","low_w":f"{q['lo']:.6f}",
      "upp_w":f"{q['hi']:.6f}","unit":"1","submit":"Retrieve Data","de":"0",
      "format":"3","line_out":"0","en_unit":"1","output":"0","bibrefs":"1",
      "page_size":"500","show_obs_wl":"1","show_calc_wl":"1","unc_out":"1",
      "order_out":"0","show_av":"2","A_out":"0","f_out":"on","S_out":"on",
      "loggf_out":"on","intens_out":"on","allowed_out":"1","forbid_out":"1",
      "conf_out":"on","term_out":"on","enrg_out":"on","J_out":"on","g_out":"on"}


def num(v):
    if v is None: return np.nan
    t=str(v).strip().replace('="','').replace('"','').replace("−","-")
    t=re.sub(r"\[[^\]]*\]","",t)
    m=re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?",t)
    try: return float(m.group()) if m else np.nan
    except ValueError: return np.nan


def normalize(df):
    x=df.copy()
    if isinstance(x.columns,pd.MultiIndex):
        x.columns=[" ".join(str(p) for p in c if str(p)!="nan") for c in x.columns]
    x.columns=[re.sub(r"\s+"," ",str(c)).strip() for c in x.columns]
    x=x.dropna(axis=1,how="all")
    return x[[c for c in x.columns if not c.lower().startswith("unnamed")]]


def parse(text):
    low=text.lstrip().lower()
    if low.startswith("<!doctype") or low.startswith("<html"):
        tables=[t for t in pd.read_html(io.StringIO(text)) if t.shape[1]>=2]
        if not tables: raise RuntimeError("NIST returned HTML without a usable line table.")
        return normalize(max(tables,key=lambda t:t.shape[0]*t.shape[1]))
    try: return normalize(pd.read_csv(io.StringIO(text),dtype=str,engine="python"))
    except Exception: return normalize(pd.read_csv(io.StringIO(text),dtype=str,sep="\t",engine="python"))


def findcol(cols,need,reject=()):
    for c in cols:
        s=c.lower()
        if all(w in s for w in need) and not any(w in s for w in reject): return c
    return None


def clean(raw,q,url):
    cols=list(map(str,raw.columns))
    obs=findcol(cols,("obs","wavelength")) or findcol(cols,("observed",)) or findcol(cols,("obs","wl"))
    ritz=findcol(cols,("ritz","wavelength")) or findcol(cols,("ritz",)) or findcol(cols,("calc","wavelength"))
    wave=np.full(len(raw),np.nan); source=np.full(len(raw),"",object)
    if obs:
        a=raw[obs].map(num).to_numpy(float); m=np.isfinite(a); wave[m]=a[m]; source[m]="observed"
    if ritz:
        a=raw[ritz].map(num).to_numpy(float); m=~np.isfinite(wave)&np.isfinite(a); wave[m]=a[m]; source[m]="Ritz"
    if not np.isfinite(wave).any():
        for c in [c for c in cols if "wave" in c.lower()]:
            a=raw[c].map(num).to_numpy(float); m=~np.isfinite(wave)&np.isfinite(a); wave[m]=a[m]; source[m]="listed"
    ic=findcol(cols,("rel","int")) or findcol(cols,("intens",))
    ac=findcol(cols,("aki",)) or findcol(cols,("a","s-1"))
    fc=findcol(cols,("fik",)) or findcol(cols,("osc",))
    gc=findcol(cols,("log","gf"))
    get=lambda c: raw[c].map(num).to_numpy(float) if c else np.full(len(raw),np.nan)
    intensity,aki,fik,loggf=map(get,(ic,ac,fc,gc))
    strength=np.full(len(raw),np.nan); basis=np.full(len(raw),"unit fallback",object)
    for vals,name,fn in [(intensity,"NIST relative intensity",lambda x:x),
                         (aki,"NIST Aki",lambda x:x),(fik,"NIST oscillator strength",np.abs),
                         (loggf,"NIST log(gf)",lambda x:10**np.clip(x,-100,100))]:
        z=fn(vals); m=~np.isfinite(strength)&np.isfinite(z)&(z>0); strength[m]=z[m]; basis[m]=name
    strength[~np.isfinite(strength)|(strength<=0)]=1
    out=pd.DataFrame({"species":q["label"],"nist_spectrum":q["spectrum"],"ion_note":q["note"],
      "rest_wavelength_vacuum_nm":wave,"rest_wavelength_vacuum_angstrom":wave*10,
      "rest_frequency_PHz":C_NM_PHZ/wave,"wavelength_source":source,
      "relative_intensity_raw":intensity,"Aki_s^-1":aki,"fik":fik,"log_gf":loggf,
      "plot_strength_raw":strength,"plot_strength_source":basis,"nist_query_url":url})
    out=out[np.isfinite(out.rest_wavelength_vacuum_nm)]
    out=out[out.rest_wavelength_vacuum_nm.between(q["lo"],q["hi"])].copy()
    if out.empty: raise RuntimeError(f"No parseable {q['spectrum']} lines in {q['lo']}-{q['hi']} nm.")
    out["plot_strength_normalized"]=out.plot_strength_raw/max(float(out.plot_strength_raw.max()),EPS)
    return out.sort_values("rest_wavelength_vacuum_nm").reset_index(drop=True)


def retrieve(s,q):
    r=s.get(ENDPOINT,params=params(q),timeout=(30,180)); r.raise_for_status()
    raw_path=CSV/f"{VERSION}_NIST_SRD78_{q['key']}_RAW.csv"; raw_path.write_text(r.text,encoding="utf-8")
    out=clean(parse(r.text),q,r.url)
    path=CSV/f"{VERSION}_NIST_SRD78_{q['key']}_CLEAN.csv"; out.to_csv(path,index=False)
    return out,path,r.url


def trace(df,lo,hi):
    x=np.linspace(lo,hi,8000); y=np.zeros_like(x); sig=max((hi-lo)/1800,0.0008)
    for w,a in zip(df.rest_wavelength_vacuum_nm,df.plot_strength_normalized):
        y+=a*np.exp(-0.5*((x-w)/sig)**2)
    return x,y/max(float(y.max()),EPS)


def w2f(x): return C_NM_PHZ/np.asarray(x)
def f2w(x): return C_NM_PHZ/np.asarray(x)


def labels(ax,df,n=7,prefix=True):
    top=df.nlargest(min(n,len(df)),"plot_strength_normalized").sort_values("rest_wavelength_vacuum_nm")
    for i,row in enumerate(top.itertuples(index=False)):
        w=float(row.rest_wavelength_vacuum_nm); y=float(row.plot_strength_normalized)
        txt=(f"{row.species}\n" if prefix else "")+f"{w:.5f} nm"
        ax.annotate(txt,xy=(w,max(.04,y)),xytext=(w,1.12+.11*(i%3)),rotation=90,
          ha="center",va="bottom",fontsize=7.4,arrowprops={"arrowstyle":"-","lw":.55,"alpha":.75},clip_on=False)


def species_axis(ax,df,q):
    x,y=trace(df,q["lo"],q["hi"]); color=COL[q["label"]]
    ax.plot(x,y,color=color,lw=1.7,label="narrow Gaussian-convolved trace")
    ax.vlines(df.rest_wavelength_vacuum_nm,0,df.plot_strength_normalized,color=".85",lw=.75,alpha=.65,label="NIST discrete transitions")
    ax.scatter(df.rest_wavelength_vacuum_nm,df.plot_strength_normalized,s=12,color=color,zorder=4)
    ax.set(xlim=(q["lo"],q["hi"]),ylim=(0,1.52),ylabel="normalized reference strength")
    ax.set_title(f"{q['label']} — {q['note']}\nNIST SRD 78 rest-vacuum lines | {len(df)} transitions",fontsize=12)
    ax.grid(alpha=.2); ax.legend(loc="upper right",fontsize=8); labels(ax,df,prefix=False)
    sec=ax.secondary_xaxis("top",functions=(w2f,f2w)); sec.set_xlabel("rest frequency [PHz]")


def two_panel(data,keys,title,name):
    qmap={q["key"]:q for q in Q}; plt.style.use("dark_background")
    fig,axes=plt.subplots(2,1,figsize=(15,11),constrained_layout=True)
    for ax,key in zip(axes,keys): species_axis(ax,data[key],qmap[key]); ax.set_xlabel("rest vacuum wavelength [nm]")
    fig.suptitle(title+"\nNIST line positions plus narrow-kernel visualization",fontsize=17)
    out=PNG/name; fig.savefig(out,dpi=360,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    return out


def blend(data):
    lo,hi=162.8,167.4; plt.style.use("dark_background"); fig,ax=plt.subplots(figsize=(15,8),constrained_layout=True)
    combined=[]
    for key in ("He_II","O_III"):
        q=next(z for z in Q if z["key"]==key); df=data[key]; x,y=trace(df,lo,hi); c=COL[q["label"]]
        ax.plot(x,y,color=c,lw=1.8,label=f"{q['label']} convolved trace")
        ax.vlines(df.rest_wavelength_vacuum_nm,0,df.plot_strength_normalized,color=c,lw=.8,alpha=.55); combined.append(df)
    all_df=pd.concat(combined,ignore_index=True); labels(ax,all_df,n=10,prefix=True)
    ax.set(xlim=(lo,hi),ylim=(0,1.52),xlabel="rest vacuum wavelength [nm]",ylabel="normalized reference strength")
    ax.set_title("He II + O III] rest-UV blend region\nSpecies normalized independently; NIST SRD 78 discrete transitions",fontsize=14)
    ax.grid(alpha=.2); ax.legend(loc="upper right")
    ax.secondary_xaxis("top",functions=(w2f,f2w)).set_xlabel("rest frequency [PHz]")
    fig.suptitle("MoM-z14 reference study — He II and O III] native rest-frame structure",fontsize=17)
    out=PNG/f"{VERSION}_NIST_HEII_OIII_BLEND_REST_UV.png"; fig.savefig(out,dpi=360,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    return out


def table_figure(master):
    t=(master.sort_values(["species","plot_strength_normalized"],ascending=[True,False])
       .groupby("species",sort=False).head(8).sort_values(["species","rest_wavelength_vacuum_nm"]))
    t=t[["species","rest_wavelength_vacuum_nm","rest_frequency_PHz","wavelength_source","plot_strength_source"]].copy()
    t.rest_wavelength_vacuum_nm=t.rest_wavelength_vacuum_nm.map(lambda v:f"{v:.6f}")
    t.rest_frequency_PHz=t.rest_frequency_PHz.map(lambda v:f"{v:.6f}")
    t.columns=["Species","λvac [nm]","νrest [PHz]","λ type","strength basis"]
    plt.style.use("dark_background"); fig,ax=plt.subplots(figsize=(15,max(7.5,.30*len(t)+2.4)),constrained_layout=True); ax.axis("off")
    tab=ax.table(cellText=t.values,colLabels=t.columns,loc="center",cellLoc="left",colLoc="left",colWidths=[.12,.16,.16,.12,.36])
    tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1,1.32)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor("#536273"); cell.set_linewidth(.45); cell.set_facecolor("#233142" if r==0 else ("#111821" if r%2 else "#17212c")); cell.set_text_props(color="white" if r==0 else "#e8edf2",weight="bold" if r==0 else "normal")
    ax.set_title("NIST SRD 78 — strongest listed rest-UV reference transitions\nStrength basis follows available NIST fields; each species is normalized only for plotting",fontsize=15,pad=18)
    out=PNG/f"{VERSION}_NIST_REST_UV_REFERENCE_TABLE.png"; fig.savefig(out,dpi=320,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    return out


def main():
    s=session(); data={}; manifest=[]
    for i,q in enumerate(Q):
        df,path,url=retrieve(s,q); data[q["key"]]=df
        manifest.append({"species":q["label"],"nist_spectrum":q["spectrum"],"query_low_nm":q["lo"],"query_high_nm":q["hi"],"returned_line_count":len(df),"clean_csv":str(path),"nist_query_url":url})
        if i<len(Q)-1: time.sleep(1)
    master=pd.concat(data.values(),ignore_index=True); master_path=CSV/f"{VERSION}_NIST_SRD78_MOMZ14_REST_UV_MASTER.csv"; master.to_csv(master_path,index=False)
    manifest_path=CSV/f"{VERSION}_NIST_SRD78_QUERY_MANIFEST.csv"; pd.DataFrame(manifest).to_csv(manifest_path,index=False)
    n=two_panel(data,("N_IV","N_III"),"MoM-z14 reference study — nitrogen rest-frame UV structure",f"{VERSION}_NIST_NITROGEN_REST_UV.png")
    c=two_panel(data,("C_IV","C_III"),"MoM-z14 reference study — carbon rest-frame UV structure",f"{VERSION}_NIST_CARBON_REST_UV.png")
    b=blend(data); t=table_figure(master)
    print(f"CODE OUTPUT: {VERSION}"); print("SOURCE          NIST Standard Reference Database 78")
    print("WAVELENGTHS     rest-frame vacuum wavelengths"); print("TRACE           narrow Gaussian convolution of discrete NIST lines")
    print("STATUS          reference visualization; not observed MoM-z14 flux")
    for r in manifest: print(f"{r['species']:<8} {r['returned_line_count']:>4} lines  {r['query_low_nm']:.1f}-{r['query_high_nm']:.1f} nm")
    print(f"NITROGEN PNG    {n}"); print(f"CARBON PNG      {c}"); print(f"BLEND PNG       {b}"); print(f"TABLE PNG       {t}")
    print(f"MASTER CSV      {master_path}"); print(f"MANIFEST CSV    {manifest_path}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}"); print(f"# {VERSION}")

if __name__=="__main__": main()
