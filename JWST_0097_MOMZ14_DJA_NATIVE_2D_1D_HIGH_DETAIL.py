#!/usr/bin/env python3
"""MoM-z14 public DJA/msaexp native 2-D + 1-D NIRSpec/PRISM atlas.
Real measured data only: no AI, templates, smoothing, or artificial upsampling.
"""
from __future__ import annotations
import importlib.util, io, math, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

def packages():
    req={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","astropy":"astropy","requests":"requests"}
    miss=[p for m,p in req.items() if importlib.util.find_spec(m) is None]
    if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])
packages()

import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION="JWST_0097"; TARGET="MoM-z14"; RA=150.0933255; DEC=2.2731627; Z=14.44; PID="5224"
API="https://grizli-cutout.herokuapp.com/nirspec_extractions"
S3="https://s3.amazonaws.com/msaexp-nirspec/extractions/{root}/{file}"
ROOT=Path("/content/JWST_OUTPUT"); PNG=ROOT/"PNG"; CSV=ROOT/"CSV"; DATA=ROOT/"DATA"/VERSION
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
COL={"Nitrogen":"#ff4fa3","Carbon":"#27d6ff","Helium":"#b978ff","Oxygen":"#4f8cff"}
LINES=[("N IV]","Nitrogen",1483.321),("N IV]","Nitrogen",1486.496),("C IV","Carbon",1548.2043),
 ("C IV","Carbon",1550.784),("He II","Helium",1640.420),("O III]","Oxygen",1660.809),
 ("O III]","Oxygen",1666.150),("N III]","Nitrogen",1746.823),("N III]","Nitrogen",1748.646),
 ("N III]","Nitrogen",1749.674),("N III]","Nitrogen",1752.160),("N III]","Nitrogen",1753.990),
 ("C III]","Carbon",1906.683),("C III]","Carbon",1908.734)]
REGIONS=[("OVERVIEW","Complete observed-frame overview","obs",2.0,3.0),
 ("NIV_CIV","N IV] and C IV high-ionization region","rest",1400.,1560.),
 ("HEII_OIII","He II and O III] blend region","rest",1560.,1700.),
 ("NIII","N III] multiplet region","rest",1700.,1800.),
 ("CIII","C III] doublet region","rest",1880.,1930.)]

def session():
    r=Retry(total=8,connect=8,read=8,backoff_factor=1.5,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset({"GET"}))
    s=requests.Session(); a=HTTPAdapter(max_retries=r); s.mount("https://",a); s.mount("http://",a)
    s.headers.update({"User-Agent":f"{VERSION} public DJA spectrum client"}); return s

def get(s,url,params=None,stream=False,n=6):
    last=None
    for k in range(1,n+1):
        try:
            q=s.get(url,params=params,stream=stream,timeout=(30,300)); q.raise_for_status(); return q
        except Exception as e:
            last=e
            if k<n: time.sleep(min(30,2.5*k))
    raise RuntimeError(f"Request failed: {url}: {last}")

def offset(ra,dec):
    return np.hypot((np.asarray(ra)-RA)*np.cos(np.deg2rad(DEC)),np.asarray(dec)-DEC)*3600

def query(s):
    for radius in (.4,.8,1.5,3.0):
        q=get(s,API,{"coords":f"{RA},{DEC}","size":radius,"output":"csv"})
        try: f=pd.read_csv(io.StringIO(q.text))
        except Exception: continue
        if not f.empty: f["query_radius_arcsec"]=radius; return f,q.url
    raise RuntimeError("No DJA extraction table returned at the MoM-z14 coordinates")

def select(f):
    for c in ("root","file","ra","dec"):
        if c not in f: raise RuntimeError(f"DJA table missing {c}")
    x=f.copy(); x["ra"]=pd.to_numeric(x.ra,errors="coerce"); x["dec"]=pd.to_numeric(x.dec,errors="coerce")
    x["offset_arcsec"]=offset(x.ra,x.dec); x["exptime_num"]=pd.to_numeric(x.get("exptime",np.nan),errors="coerce")
    x["z_num"]=pd.to_numeric(x.get("z",np.nan),errors="coerce"); g=x.get("grating",pd.Series("",index=x.index)).astype(str)
    prism=g.str.contains("PRISM",case=False,na=False)|x.file.astype(str).str.contains("prism",case=False,na=False)
    program=x.file.astype(str).str.contains(PID,regex=False); y=x[prism&(x.offset_arcsec<=1.5)&program].copy()
    if y.empty: y=x[prism&(x.offset_arcsec<=1.5)].copy()
    if y.empty: raise RuntimeError(f"No close DJA PRISM extraction; nearest={x.offset_arcsec.min():.3f} arcsec")
    y["zd"]=np.abs(y.z_num-Z).fillna(99.); y["v4"]=y.root.astype(str).str.contains("v4",case=False).astype(int)
    return y.sort_values(["offset_arcsec","zd","v4","exptime_num"],ascending=[True,True,False,False]).iloc[0]

def download(s,row):
    url=S3.format(root=row.root,file=row.file); p=DATA/Path(str(row.file)).name
    if not p.exists() or p.stat().st_size<100000:
        q=get(s,url,stream=True); part=p.with_suffix(p.suffix+".part")
        with part.open("wb") as h:
            for c in q.iter_content(1024*1024):
                if c: h.write(c)
        if part.stat().st_size<100000: part.unlink(missing_ok=True); raise RuntimeError("DJA FITS download too small")
        part.replace(p)
    return p,url

def col(names,*wanted):
    low={str(n).lower():str(n) for n in names}
    for w in wanted:
        if w.lower() in low: return low[w.lower()]
    for w in wanted:
        for k,v in low.items():
            if w.lower() in k: return v
    return None

def read(p):
    with fits.open(p,memmap=False) as h:
        ext=[z.name for z in h]
        if "SPEC1D" not in ext: raise RuntimeError(f"No SPEC1D extension: {ext}")
        t=h["SPEC1D"]; n=list(t.data.names or []); wc=col(n,"wave","wavelength"); fc=col(n,"flux","aper_flux"); ec=col(n,"err","full_err","aper_err")
        if not all((wc,fc,ec)): raise RuntimeError(f"Unrecognized SPEC1D columns: {n}")
        w=np.asarray(t.data[wc],float).ravel(); f=np.asarray(t.data[fc],float).ravel(); e=np.asarray(t.data[ec],float).ravel()
        vc=col(n,"valid"); v=np.asarray(t.data[vc],bool).ravel() if vc else np.ones(len(w),bool)
        v&=np.isfinite(w)&np.isfinite(f)&np.isfinite(e)&(e>0)&(w>0); o=np.argsort(w[v]); w,f,e=w[v][o],f[v][o],e[v][o]
        unit="microJy"
        try: unit=str(t.columns[fc].unit or unit)
        except Exception: pass
        arr=lambda name: np.asarray(h[name].data,float) if name in ext else None
        return {"wave":w,"flux":f,"err":e,"unit":unit,"sci":arr("SCI"),"wht":arr("WHT"),"err2d":arr("ERR"),"profile":arr("PROFILE"),"ext":ext,"cols":n}

def r2o(x): return np.asarray(x,float)*(1+Z)*1e-4
def o2r(x): return np.asarray(x,float)*1e4/(1+Z)
def bounds(r): return (r[3],r[4]) if r[2]=="obs" else tuple(r2o([r[3],r[4]]))
def marks(r):
    lo,hi=bounds(r); return [(a,b,c,float(r2o(c))) for a,b,c in LINES if lo<=float(r2o(c))<=hi]

def style():
    plt.close("all"); matplotlib.rcdefaults(); matplotlib.rcParams.update({"text.usetex":False,"figure.facecolor":"#030712","axes.facecolor":"#071426","axes.edgecolor":"#8aa2bd","axes.labelcolor":"#eef6ff","xtick.color":"#dcecff","ytick.color":"#dcecff","text.color":"#f7fbff","font.size":10})
def axes_style(ax):
    ax.grid(True,color="#344a63",lw=.45,alpha=.42)
    for s in ax.spines.values(): s.set_color("#8da6c0"); s.set_linewidth(.7)
def orient(a,n):
    if a is None or getattr(a,"ndim",0)!=2: return None
    if a.shape[1]==n: return a
    if a.shape[0]==n: return a.T
    return None

def sn2d(d):
    sci=orient(d["sci"],len(d["wave"])); wht=orient(d["wht"],len(d["wave"])); er=orient(d["err2d"],len(d["wave"]))
    if sci is None: return None
    if wht is not None: return np.where(np.isfinite(wht)&(wht>0),sci*np.sqrt(wht),np.nan)
    if er is not None: return np.where(np.isfinite(er)&(er>0),sci/er,np.nan)
    med=np.nanmedian(sci,axis=0,keepdims=True); scale=np.nanmedian(np.abs(sci-med),axis=0); return (sci-med)/np.where(scale>0,scale,np.nan)
def center_row(d,sn):
    p=orient(d["profile"],len(d["wave"]));
    if p is not None:
        q=np.nansum(np.maximum(p,0),axis=1)
        if np.isfinite(q).any(): return int(np.nanargmax(q))
    m=(d["wave"]>=2)&(d["wave"]<=3); q=np.nanmedian(np.abs(sn[:,m]),axis=1)
    return int(np.nanargmax(q)) if np.isfinite(q).any() else sn.shape[0]//2

def ylim(y,e):
    lo,hi=np.nanpercentile(y[np.isfinite(y)],[1,99]); q=e[np.isfinite(e)&(e>0)]
    if len(q): p=min(2*np.nanmedian(q),max(hi-lo,1)); lo-=p; hi+=p
    if not np.isfinite(lo+hi) or hi<=lo: c=np.nanmedian(y); s=np.nanstd(y) or 1; lo,hi=c-3*s,c+3*s
    p=.1*(hi-lo); return float(lo-p),float(hi+p)

def plot(d,r,i):
    w,f,e=d["wave"],d["flux"],d["err"]; lo,hi=bounds(r); m=(w>=lo)&(w<=hi)
    if m.sum()<6: raise RuntimeError(f"{r[0]} has only {m.sum()} native samples")
    x,y,q=w[m],f[m],e[m]; sn=sn2d(d); fig=plt.figure(figsize=(18,10.5),constrained_layout=True); gs=fig.add_gridspec(2,1,height_ratios=[1.12,1.65])
    a2=fig.add_subplot(gs[0]); a1=fig.add_subplot(gs[1],sharex=a2)
    if sn is not None:
        c=center_row(d,sn); y0=max(0,c-7); y1=min(sn.shape[0],c+8); im=sn[y0:y1,:][:,m]; z=im[np.isfinite(im)]; s=max(float(np.nanpercentile(np.abs(z),98.5)) if len(z) else 1,1)
        a2.imshow(im,origin="lower",aspect="auto",interpolation="nearest",cmap="coolwarm",vmin=-s,vmax=s,extent=[x.min(),x.max(),y0-c,y1-c]); a2.axhline(0,color="white",lw=.55,alpha=.75); a2.set_ylabel("Spatial offset [native pixels]"); a2.set_title("Actual rectified 2-D measured signal-to-noise",fontsize=11.5)
    else: a2.text(.5,.5,"No rectified 2-D extension",transform=a2.transAxes,ha="center",va="center")
    axes_style(a2); good=np.isfinite(q)&(q>0)
    if good.any(): a1.fill_between(x[good],y[good]-q[good],y[good]+q[good],color="#2b8cc4",alpha=.13,lw=0,label="Measured 1σ uncertainty")
    a1.plot(x,y,color="#f2f7fb",lw=.78,marker="o",ms=2.25,mfc="#96ddff",mew=0,alpha=.96,label=f"DJA msaexp native optimal extraction (n={len(x)})")
    yl,yh=ylim(y,q); a1.set_ylim(yl,yh); a1.set_xlim(lo,hi); span=yh-yl
    for j,(fam,el,rest,obs) in enumerate(marks(r)):
        cc=COL[el]; a1.axvline(obs,color=cc,lw=.85,ls="--",alpha=.95); a2.axvline(obs,color=cc,lw=.70,ls="--",alpha=.88)
        yy=yh-(.08+.12*(j%4))*span; a1.annotate(f"{fam}\n{rest:.2f} Å",xy=(obs,yy-.02*span),xytext=(obs,yy),rotation=90,ha="center",va="bottom",fontsize=7.4,color=cc,arrowprops={"arrowstyle":"-","color":cc,"lw":.45},bbox={"facecolor":"#030712","edgecolor":"none","alpha":.68,"pad":.8},clip_on=True)
    a1.set_xlabel("Observed wavelength [µm]"); a1.set_ylabel(f"Flux density [{d['unit']}]"); a1.set_title(f"{r[1]}\nActual DJA/msaexp data — no template, smoothing, or upsampling",fontsize=13.5); axes_style(a1)
    a1.legend(loc="lower left",fontsize=8.2,facecolor="#030712",edgecolor="#445a72"); a2.secondary_xaxis("top",functions=(o2r,r2o)).set_xlabel(f"Rest-frame vacuum wavelength [Å] at z={Z:.2f}")
    fig.suptitle(f"{TARGET} — public DAWN JWST Archive NIRSpec/PRISM spectrum\nNative rectified 2-D data + native optimal 1-D extraction",fontsize=17)
    pp=PNG/f"{VERSION}_{i:02d}_{r[0]}_DJA_NATIVE_2D_1D.png"; fig.savefig(pp,dpi=420,bbox_inches="tight",facecolor=fig.get_facecolor()); plt.show(); plt.close(fig)
    cp=CSV/f"{VERSION}_{i:02d}_{r[0]}_NATIVE_SPEC1D.csv"; pd.DataFrame({"observed_wavelength_um":x,"rest_wavelength_angstrom_z14p44":o2r(x),"flux":y,"flux_error_1sigma":q,"native_measured_sample":True,"smoothed":False,"interpolated":False,"synthetic":False}).to_csv(cp,index=False)
    return pp,cp,{"region":r[0],"native_samples":len(x),"observed_low_um":x.min(),"observed_high_um":x.max(),"rest_low_A":o2r(x.min()),"rest_high_A":o2r(x.max()),"canonical_markers":len(marks(r)),"has_2d_data":sn is not None}

def main():
    style(); s=session(); table,qurl=query(s); audit=CSV/f"{VERSION}_DJA_QUERY_RESULTS.csv"; table.to_csv(audit,index=False)
    row=select(table); fp,furl=download(s,row); d=read(fp)
    full=CSV/f"{VERSION}_MOMZ14_DJA_NATIVE_SPEC1D_FULL.csv"; pd.DataFrame({"observed_wavelength_um":d["wave"],"rest_wavelength_angstrom_z14p44":o2r(d["wave"]),"flux":d["flux"],"flux_error_1sigma":d["err"],"native_measured_sample":True,"smoothed":False,"interpolated":False,"synthetic":False}).to_csv(full,index=False)
    linecsv=CSV/f"{VERSION}_CANONICAL_REFERENCE_LINES.csv"; pd.DataFrame([{"family":a,"element":b,"rest_vacuum_angstrom":c,"observed_um_at_z14p44":float(r2o(c)),"color":COL[b]} for a,b,c in LINES]).to_csv(linecsv,index=False)
    pngs=[]; regioncsv=[]; sums=[]
    for i,r in enumerate(REGIONS,1): p,c,z=plot(d,r,i); pngs.append(p); regioncsv.append(c); sums.append(z)
    summary=CSV/f"{VERSION}_REGION_SUMMARY.csv"; pd.DataFrame(sums).to_csv(summary,index=False)
    prov=CSV/f"{VERSION}_PROVENANCE.csv"; pd.DataFrame([{"target":TARGET,"ra_deg":RA,"dec_deg":DEC,"redshift_axis":Z,"dja_query_url":qurl,"dja_root":row.root,"dja_file":row.file,"coordinate_offset_arcsec":row.offset_arcsec,"dja_grating":row.get("grating",""),"dja_exptime_s":row.exptime_num,"dja_redshift_metadata":row.z_num,"download_url":furl,"local_fits":str(fp),"fits_extensions":";".join(d["ext"]),"spec1d_columns":";".join(d["cols"]),"native_samples":len(d["wave"]),"smoothing":False,"interpolation":False,"synthetic_model":False}]).to_csv(prov,index=False)
    print(f"CODE OUTPUT: {VERSION}"); print("DATA SOURCE      DAWN JWST Archive msaexp public extraction"); print("DATA PRODUCT     Native SPEC1D + rectified SCI/WHT data"); print("DISPLAY POLICY   Real measured samples; no smoothing or artificial upsampling")
    print(f"DJA FILE         {row.file}"); print(f"SOURCE OFFSET    {row.offset_arcsec:.6f} arcsec"); print(f"DJA REDSHIFT     {row.z_num}"); print(f"EXPOSURE         {row.exptime_num} s"); print(f"NATIVE SAMPLES   {len(d['wave'])}"); print(f"FITS             {fp}")
    for p in pngs: print(f"PLOT PNG         {p}")
    print(f"FULL CSV         {full}"); print(f"LINE CSV         {linecsv}"); print(f"SUMMARY CSV      {summary}"); print(f"PROVENANCE CSV   {prov}"); print(f"QUERY AUDIT CSV  {audit}"); print(f"Timestamp        {datetime.now(timezone.utc).isoformat(timespec='seconds')}"); print(f"# {VERSION}")
if __name__=="__main__": main()
