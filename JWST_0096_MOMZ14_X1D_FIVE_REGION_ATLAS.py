#!/usr/bin/env python3
"""Actual MoM-z14 NIRSpec X1D spectrum: one overview + four rest-UV regions.
NIST SRD 78 markers come from the JWST_0093 repository dataset.
No smoothing, interpolation, synthetic profiles, step plotting, or AI images.
"""
from __future__ import annotations
import importlib.util, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path


def packages():
    req={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib",
         "astropy":"astropy","astroquery":"astroquery","requests":"requests"}
    miss=[p for m,p in req.items() if importlib.util.find_spec(m) is None]
    if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
packages()

import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION="JWST_0096"; Z=14.44
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
BASE_NAME="JWST_0087_MOMZ14_ACTUAL_NIRSPEC_RAW_SPECTROGRAPH.py"
BASE_URL=f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{BASE_NAME}"
BASE_PATH=Path("/content")/BASE_NAME
NIST_URL=("https://raw.githubusercontent.com/gear66me-ui/JWST/main/data/NIST_SRD78/"
          "JWST_0093/JWST_0093_NIST_SRD78_SIX_REFERENCE_FAMILIES_MASTER.csv")
COLORS={"Nitrogen":"#ff4fa3","Carbon":"#25c7f7","Helium":"#b978ff","Oxygen":"#4d8dff"}
REGIONS=[
 {"key":"OVERVIEW","title":"Observed-frame overview","kind":"obs","lo":2.0,"hi":3.0,
  "note":"All six NIST SRD 78 reference families"},
 {"key":"NIV_CIV","title":"N IV] + C IV","kind":"rest","lo":1400.,"hi":1560.,
  "note":"Nitrogen and carbon high-ionization region"},
 {"key":"HEII_OIII","title":"He II + O III]","kind":"rest","lo":1560.,"hi":1700.,
  "note":"Helium recombination and oxygen intercombination region"},
 {"key":"NIII","title":"N III] multiplet","kind":"rest","lo":1700.,"hi":1800.,
  "note":"Nitrogen rest-UV multiplet region"},
 {"key":"CIII","title":"C III] doublet","kind":"rest","lo":1880.,"hi":1930.,
  "note":"Carbon intercombination-doublet region"},
]


def session():
    r=Retry(total=8,connect=8,read=8,backoff_factor=1.4,
            status_forcelist=(429,500,502,503,504),allowed_methods=frozenset({"GET","POST"}))
    s=requests.Session(); a=HTTPAdapter(max_retries=r)
    s.mount("https://",a); s.mount("http://",a)
    s.headers.update({"User-Agent":f"{VERSION} public JWST spectrum client"})
    return s


def download(url,path,minimum=1000):
    s=session(); last=None
    for k in range(1,7):
        try:
            q=s.get(url,timeout=(30,240)); q.raise_for_status(); path.write_bytes(q.content)
            if path.stat().st_size<minimum: raise RuntimeError("downloaded file is too small")
            return path
        except Exception as exc:
            last=exc; path.unlink(missing_ok=True)
            if k<6: time.sleep(2*k)
    raise RuntimeError(f"Failed to download {url}: {last}")


def load_base():
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size<12000: download(BASE_URL,BASE_PATH,12000)
    spec=importlib.util.spec_from_file_location("jwst0087",BASE_PATH)
    if spec is None or spec.loader is None: raise RuntimeError("Cannot load JWST_0087 helper")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.VERSION=VERSION; mod.PNG=PNG; mod.CSV=CSV; mod.DATA=DATA; mod.MAX_X1D=60
    return mod
base=load_base()


def retry(label,fn,n=7):
    last=None
    for k in range(1,n+1):
        try: return fn()
        except Exception as exc:
            last=exc; print(f"RETRY {label:<24} {k}/{n} {type(exc).__name__}: {exc}")
            if k<n: time.sleep(min(20,2*k))
    raise RuntimeError(f"{label} failed: {last}")


def query_observations():
    target=SkyCoord(base.MOM_RA*u.deg,base.MOM_DEC*u.deg)
    for radius in (3.,10.,30.):
        table=retry(f"MAST query {radius:.0f} arcsec",lambda r=radius: Observations.query_region(target,radius=r*u.arcsec))
        f=table.to_pandas() if hasattr(table,"to_pandas") else pd.DataFrame(table)
        if f.empty: continue
        pc=base.col(f,["proposal_id","proposalid"]); cc=base.col(f,["obs_collection"])
        ic=base.col(f,["instrument_name","instrument"]); mask=pd.Series(True,index=f.index)
        if pc: mask &= f[pc].astype(str).str.replace(".0","",regex=False).eq(base.JWST_PID)
        if cc: mask &= f[cc].astype(str).str.upper().eq("JWST")
        if ic: mask &= f[ic].astype(str).str.contains("NIRSPEC",case=False,na=False)
        out=f[mask].copy()
        if not out.empty: return out
    raise RuntimeError("No GO-5224 NIRSpec observation intersects MoM-z14")


def query_products(obsids):
    frames=[]
    for i in range(0,len(obsids),6):
        ids=obsids[i:i+6]
        t=retry(f"MAST products {i+1}-{i+len(ids)}",lambda x=ids: Observations.get_product_list(x))
        frames.append(t.to_pandas() if hasattr(t,"to_pandas") else pd.DataFrame(t))
    return pd.concat(frames,ignore_index=True,sort=False).drop_duplicates()
base.query_coordinate_matched_observations=query_observations
base.query_products=query_products


def load_nist():
    p=CSV/f"{VERSION}_NIST_SRD78_REFERENCE_LINES.csv"; download(NIST_URL,p,500)
    f=pd.read_csv(p)
    for c in ("rest_wavelength_vacuum_angstrom","Aki_s^-1","relative_intensity"):
        f[c]=pd.to_numeric(f[c],errors="coerce")
    f=f[np.isfinite(f.rest_wavelength_vacuum_angstrom)].copy()
    f["observed_um"]=f.rest_wavelength_vacuum_angstrom*(1+Z)*1e-4
    f["strength"]=f.relative_intensity
    m=~np.isfinite(f.strength)|(f.strength<=0); f.loc[m,"strength"]=f.loc[m,"Aki_s^-1"]
    f["strength_norm"]=0.45
    for _,g in f.groupby("selected_family"):
        v=g.strength.to_numpy(float); ok=np.isfinite(v)&(v>0)
        if ok.any():
            q=np.log10(v[ok]); lo,hi=q.min(),q.max(); z=np.full(len(g),.45)
            z[ok]=.45+.55*((q-lo)/(hi-lo) if hi>lo else 1.); f.loc[g.index,"strength_norm"]=z
    return f


def rest_to_obs(x): return np.asarray(x,dtype=float)*(1+Z)*1e-4
def obs_to_rest(x): return np.asarray(x,dtype=float)*1e4/(1+Z)


def style():
    plt.close("all"); matplotlib.rcdefaults()
    matplotlib.rcParams.update({"text.usetex":False,"figure.facecolor":"#050812",
      "axes.facecolor":"#081323","axes.edgecolor":"#8aa0b8","axes.labelcolor":"#eef5ff",
      "xtick.color":"#dce9f7","ytick.color":"#dce9f7","text.color":"#f7fbff","font.size":10})


def limits(y,e):
    v=y[np.isfinite(y)]
    if not len(v): return -1.,1.
    lo,hi=np.nanpercentile(v,[1,99])
    q=e[np.isfinite(e)&(e>=0)]
    if len(q):
        pad=min(float(np.nanmedian(q))*1.5,max(float(hi-lo),1.)); lo-=pad; hi+=pad
    if not np.isfinite(lo+hi) or hi<=lo:
        c=float(np.nanmedian(v)); s=float(np.nanstd(v)) or 1.; lo,hi=c-3*s,c+3*s
    p=.10*(hi-lo); return float(lo-p),float(hi+p)


def region_mask(w,region):
    lo,hi=(region["lo"],region["hi"]) if region["kind"]=="obs" else rest_to_obs([region["lo"],region["hi"]])
    return (w>=lo)&(w<=hi)


def marker_rows(nist,region):
    if region["kind"]=="obs": m=nist.observed_um.between(region["lo"],region["hi"])
    else: m=nist.rest_wavelength_vacuum_angstrom.between(region["lo"],region["hi"])
    return nist[m].sort_values("rest_wavelength_vacuum_angstrom")


def draw(ax,region,w,flux,error,ylabel,nist):
    m=region_mask(w,region); x=w[m]; y=flux[m]; e=error[m]
    if len(x)<5: raise RuntimeError(f"{region['key']} contains only {len(x)} real samples")
    good=np.isfinite(e)&(e>=0)
    if good.any(): ax.fill_between(x[good],y[good]-e[good],y[good]+e[good],color="#2b7aa8",alpha=.15,lw=0,label="1σ uncertainty")
    ax.plot(x,y,color="#e8f4ff",lw=.72,marker=".",ms=2.8,mfc="#95d9ff",mew=0,
            alpha=.97,label=f"Actual X1D samples (n={len(x)})",zorder=3)
    ymin,ymax=limits(y,e); ax.set_ylim(ymin,ymax); ax.set_xlim(float(x.min()),float(x.max()))
    lines=marker_rows(nist,region); span=ymax-ymin
    for i,r in enumerate(lines.itertuples(index=False)):
        color=COLORS.get(str(r.element),"#f6c453"); xx=float(r.observed_um)
        ax.axvline(xx,color=color,lw=.48+.50*float(r.strength_norm),ls="--",alpha=.90,zorder=2)
        yy=ymax-(.07+.12*(i%4))*span
        ax.annotate(f"{r.selected_family}\n{float(r.rest_wavelength_vacuum_angstrom):.2f} Å",
          xy=(xx,yy-.025*span),xytext=(xx,yy),ha="center",va="bottom",rotation=90,
          color=color,fontsize=7.1,arrowprops={"arrowstyle":"-","color":color,"lw":.42},
          bbox={"facecolor":"#050812","edgecolor":"none","alpha":.67,"pad":.7},clip_on=True)
    ax.grid(True,color="#33465e",lw=.48,alpha=.45)
    for s in ax.spines.values(): s.set_color("#8ca3b8"); s.set_linewidth(.75)
    ax.set_title(f"{region['title']}\n{region['note']}",fontsize=13,pad=10)
    ax.set_xlabel("Observed wavelength [µm]"); ax.set_ylabel(ylabel)
    ax.secondary_xaxis("top",functions=(obs_to_rest,rest_to_obs)).set_xlabel("Rest-frame vacuum wavelength [Å]")
    ax.legend(loc="lower left",fontsize=7.8,facecolor="#050812",edgecolor="#42556c",framealpha=.88)
    summary={"region":region["key"],"samples":len(x),"observed_low_um":x.min(),"observed_high_um":x.max(),
             "rest_low_A":obs_to_rest(x.min()),"rest_high_A":obs_to_rest(x.max()),"nist_markers":len(lines)}
    return summary,x,y,e


def individual(w,flux,error,ylabel,nist):
    paths=[]; summaries=[]; rows=[]
    for j,r in enumerate(REGIONS,1):
        fig,ax=plt.subplots(figsize=(17,8.2),constrained_layout=True)
        s,x,y,e=draw(ax,r,w,flux,error,ylabel,nist)
        fig.suptitle("MoM-z14 — actual public JWST/NIRSpec prism X1D spectrum\nMeasured jagged samples + NIST SRD 78 markers",fontsize=16.5)
        p=PNG/f"{VERSION}_{j:02d}_{r['key']}_ACTUAL_SPECTRUM.png"
        fig.savefig(p,dpi=420,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
        paths.append(p); summaries.append(s)
        rows.extend({"region":r["key"],"observed_wavelength_um":x[k],"rest_wavelength_A_z14p44":obs_to_rest(x[k]),
                     "flux_display_units":y[k],"flux_error_display_units":e[k],"actual_x1d_sample":True,
                     "smoothed":False,"interpolated":False} for k in range(len(x)))
    p=CSV/f"{VERSION}_FIVE_REGION_ACTUAL_X1D_SAMPLES.csv"; pd.DataFrame(rows).to_csv(p,index=False)
    return paths,pd.DataFrame(summaries),p


def atlas(w,flux,error,ylabel,nist):
    fig=plt.figure(figsize=(19,24),constrained_layout=True)
    gs=fig.add_gridspec(6,1,height_ratios=[1.55,1,1,1,1,.54])
    for i,r in enumerate(REGIONS): draw(fig.add_subplot(gs[i,0]),r,w,flux,error,ylabel,nist)
    ax=fig.add_subplot(gs[5,0]); ax.axis("off"); ax.set_title("Reference-line color key and data status",loc="left",fontsize=12.5)
    rows=[]
    for el in ("Nitrogen","Carbon","Helium","Oxygen"):
        fam=", ".join(sorted(nist.loc[nist.element==el,"selected_family"].dropna().unique())); rows.append([el,fam,"NIST SRD 78 laboratory positions"])
    t=ax.table(cellText=rows,colLabels=["Element","Ion families","Overlay meaning"],cellLoc="left",colLoc="left",
               bbox=[0,.05,.72,.84],colWidths=[.16,.22,.34]); t.auto_set_font_size(False); t.set_fontsize(8.7)
    for (rr,cc),cell in t.get_celld().items():
        cell.set_edgecolor("#42556c"); cell.set_linewidth(.45)
        if rr==0: cell.set_facecolor("#24344a"); cell.set_text_props(color="white",weight="bold")
        else: cell.set_facecolor("#0d192a" if rr%2 else "#122136"); cell.set_text_props(color=COLORS[rows[rr-1][0]] if cc==0 else "#e7f0fa")
    ax.text(.755,.78,"WHITE/CYAN CURVE\nActual coordinate-verified X1D samples\n\nBLUE BAND\nMeasured 1σ uncertainty\n\nCOLORED DASHES\nNIST SRD 78 reference wavelengths\n\nPROCESSING\nNo smoothing, interpolation, or synthetic profiles",
            transform=ax.transAxes,ha="left",va="top",fontsize=9.1,linespacing=1.35,
            bbox={"facecolor":"#091421","edgecolor":"#42556c","boxstyle":"round,pad=.65"})
    fig.suptitle("MoM-z14 full NIRSpec X1D spectrum divided into four rest-UV study regions\nActual public data + NIST Standard Reference Database 78 overlays",fontsize=19)
    p=PNG/f"{VERSION}_MOMZ14_FIVE_REGION_PUBLICATION_ATLAS.png"
    fig.savefig(p,dpi=360,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig); return p


def main():
    style(); spectrum,audit=base.find_exact_spectrum(); nist=load_nist()
    w=spectrum["wavelength_um"]; raw=spectrum["flux"]; rawerr=spectrum["error"]
    flux,error,ylabel=base.convert_flux(raw,rawerr,spectrum["flux_unit"])
    full=CSV/f"{VERSION}_MOMZ14_ACTUAL_NIRSPEC_X1D_FULL.csv"
    pd.DataFrame({"observed_wavelength_um":w,"rest_wavelength_A_z14p44":obs_to_rest(w),
      "flux_display_units":flux,"flux_error_display_units":error,"original_flux":raw,
      "original_flux_error":rawerr,"original_flux_unit":spectrum["flux_unit"],"actual_x1d_sample":True}).to_csv(full,index=False)
    paths,summary,samples=individual(w,flux,error,ylabel,nist); atlas_path=atlas(w,flux,error,ylabel,nist)
    summary_path=CSV/f"{VERSION}_FIVE_REGION_SUMMARY.csv"; summary.to_csv(summary_path,index=False)
    print(f"CODE OUTPUT: {VERSION}"); print("DATA            actual coordinate-verified JWST/NIRSpec X1D samples")
    print("REFERENCE       NIST SRD 78 dataset from JWST_0093"); print("SMOOTHING       none")
    print("INTERPOLATION   none"); print("SYNTHETIC DATA  none"); print(f"PRODUCT         {spectrum['path'].name}")
    print(f"SOURCE OFFSET   {spectrum['separation_arcsec']:.6f} arcsec"); print(f"ATLAS PNG       {atlas_path}")
    for p in paths: print(f"REGION PNG      {p}")
    print(f"FULL X1D CSV    {full}"); print(f"REGION CSV      {samples}"); print(f"SUMMARY CSV     {summary_path}")
    print(f"AUDIT CSV       {audit}"); print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")

if __name__=="__main__": main()
