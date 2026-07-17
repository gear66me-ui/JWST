#!/usr/bin/env python3
# JWST_0094
# Scientific NIST SRD 78 rest-frequency atlas. Real archived line data only.
from pathlib import Path
from datetime import datetime, timezone
import importlib.util, io, math, os, subprocess, sys, time

REQ={"numpy":"numpy","pandas":"pandas","matplotlib":"matplotlib","requests":"requests"}
miss=[p for m,p in REQ.items() if importlib.util.find_spec(m) is None]
if miss: subprocess.check_call([sys.executable,"-m","pip","install","-q",*miss])

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION="JWST_0094"
C=299.792458
REPO="gear66me-ui/JWST"
MASTER="data/NIST_SRD78/JWST_0093/JWST_0093_NIST_SRD78_SIX_REFERENCE_FAMILIES_MASTER.csv"
URL=f"https://raw.githubusercontent.com/{REPO}/main/{MASTER}"
ROOT=Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd()/"JWST_OUTPUT"
PNG,CSV,DATA=ROOT/"PNG",ROOT/"CSV",ROOT/"DATA"/VERSION
for p in (PNG,CSV,DATA): p.mkdir(parents=True,exist_ok=True)
ORDER=["N IV]","C IV","He II","O III]","N III]","C III]"]
META={
 "N IV]":("Nitrogen","N³⁺","#ff4fd8"),"N III]":("Nitrogen","N²⁺","#ff4fd8"),
 "C IV":("Carbon","C³⁺","#42d9ff"),"C III]":("Carbon","C²⁺","#42d9ff"),
 "He II":("Helium","He⁺","#8cff66"),"O III]":("Oxygen","O²⁺","#4f8cff")}


def style():
 plt.close("all"); mpl.rcdefaults()
 mpl.rcParams.update({"text.usetex":False,"font.family":"DejaVu Sans","mathtext.fontset":"dejavusans",
 "figure.facecolor":"#030711","savefig.facecolor":"#030711","axes.facecolor":"#07101e",
 "axes.edgecolor":"#8ea3bc","axes.labelcolor":"#edf6ff","xtick.color":"#dbeafe",
 "ytick.color":"#dbeafe","text.color":"#f8fbff","grid.color":"#28405c","grid.alpha":.34,
 "grid.linewidth":.55,"font.size":10,"axes.titleweight":"semibold"})


def session():
 r=Retry(total=6,connect=6,read=6,backoff_factor=1.2,status_forcelist=(429,500,502,503,504),allowed_methods=frozenset({"GET"}))
 s=requests.Session(); s.mount("https://",HTTPAdapter(max_retries=r)); s.headers["User-Agent"]=f"{VERSION} NIST atlas"; return s


def load_master():
 override=os.environ.get("JWST_NIST_MASTER_CSV","").strip(); mode="local_override"
 if override: raw=Path(override).read_bytes()
 else:
  mode="github_archived_master"; last=None
  for k in range(4):
   try:
    q=session().get(URL,timeout=(20,120)); q.raise_for_status(); raw=q.content
    if len(raw)<500: raise RuntimeError("master CSV too small")
    last=None; break
   except Exception as e: last=e; time.sleep(1.5*(k+1))
  if last: raise RuntimeError(f"Cannot download archived NIST table: {last}")
 local=DATA/f"{VERSION}_NIST_SRD78_MASTER_INPUT.csv"; local.write_bytes(raw)
 d=pd.read_csv(io.BytesIO(raw))
 if d.empty: raise RuntimeError("Archived NIST table is empty")
 return d,local,mode


def targets(v):
 out=[]
 for x in str(v).replace(",",";").split(";"):
  try: out.append(float(x.strip()))
  except: pass
 return out


def basis(g):
 choices=[("NIST relative intensity","relative_intensity","plain"),("NIST Aki [s⁻¹]","Aki_s^-1","plain"),
          ("NIST oscillator strength |fik|","fik","abs"),("NIST 10^log(gf)","log_gf","pow")]
 for label,col,mode in choices:
  if col not in g: continue
  a=pd.to_numeric(g[col],errors="coerce").to_numpy(float)
  if mode=="abs": a=np.abs(a)
  if mode=="pow": a=np.where(np.isfinite(a),10**np.clip(a,-100,100),np.nan)
  ok=np.isfinite(a)&(a>0)
  if ok.sum()>=max(1,math.ceil(len(g)*.5)):
   a=np.where(ok,a,np.nanmin(a[ok])*1e-6); return label,a
 return "unit line-presence fallback",np.ones(len(g))


def prepare(d):
 need={"selected_family","rest_wavelength_vacuum_nm","rest_wavelength_vacuum_angstrom","study_target_rest_angstrom"}
 if need-set(d): raise RuntimeError(f"Missing columns: {sorted(need-set(d))}")
 d=d.copy(); d["rest_wavelength_vacuum_nm"]=pd.to_numeric(d["rest_wavelength_vacuum_nm"],errors="coerce")
 d["rest_wavelength_vacuum_angstrom"]=pd.to_numeric(d["rest_wavelength_vacuum_angstrom"],errors="coerce")
 d=d[np.isfinite(d.rest_wavelength_vacuum_nm)&(d.rest_wavelength_vacuum_nm>0)].copy(); out=[]; audit=[]; tr=[]
 for fam in ORDER:
  g=d[d.selected_family==fam].copy()
  if g.empty: raise RuntimeError(f"No NIST rows for {fam}")
  label,a=basis(g); a=np.clip(a/np.nanmax(a),1e-12,1); g["strength_basis"]=label; g["strength_normalized_family"]=a
  g["rest_frequency_PHz"]=C/g.rest_wavelength_vacuum_nm; g["element_color"]=META[fam][2]; out.append(g)
  audit.append({"family":fam,"element":META[fam][0],"ion":META[fam][1],"rows":len(g),"strength_basis":label,
                "wavelength_min_nm":g.rest_wavelength_vacuum_nm.min(),"wavelength_max_nm":g.rest_wavelength_vacuum_nm.max()})
  ta=targets(g.study_target_rest_angstrom.dropna().iloc[0]); w=g.rest_wavelength_vacuum_angstrom.to_numpy(float)
  for t in ta:
   j=int(np.nanargmin(abs(w-t))); n=g.iloc[j]
   tr.append({"family":fam,"element":META[fam][0],"target_rest_angstrom":t,"target_rest_nm":t/10,
              "target_frequency_PHz":C/(t/10),"nearest_nist_angstrom":n.rest_wavelength_vacuum_angstrom,
              "nearest_offset_angstrom":n.rest_wavelength_vacuum_angstrom-t,"wavelength_source":n.get("wavelength_source","")})
 return pd.concat(out,ignore_index=True),pd.DataFrame(audit),pd.DataFrame(tr)


def f2w(x):
 with np.errstate(divide="ignore",invalid="ignore"): return C/np.asarray(x,float)
def w2f(x):
 with np.errstate(divide="ignore",invalid="ignore"): return C/np.asarray(x,float)
def axes_style(ax):
 ax.grid(True,which="major"); ax.grid(True,which="minor",alpha=.13,linewidth=.35)
 for s in ax.spines.values(): s.set_color("#8ea3bc"); s.set_linewidth(.75)


def overview(d,t):
 fig,ax=plt.subplots(figsize=(18,10.5),constrained_layout=True); gap=1.; h=.72; yt=[]; yl=[]
 for lane,fam in enumerate(ORDER[::-1]):
  b=lane*gap; g=d[d.selected_family==fam].sort_values("rest_frequency_PHz"); col=META[fam][2]
  y=np.clip((np.log10(g.strength_normalized_family.to_numpy())+12)/12,.04,1); x=g.rest_frequency_PHz.to_numpy(); top=b+h*y
  ax.hlines(b,x.min(),x.max(),color="#66809c",lw=.55,alpha=.75); ax.vlines(x,b,top,color=col,lw=.78,alpha=.92)
  ax.scatter(x,top,s=11,color=col,edgecolors="#eaf6ff",linewidths=.25,zorder=3)
  for k,r in enumerate(t[t.family==fam].itertuples(index=False)):
   ax.vlines(r.target_frequency_PHz,b+.02,b+h+.08,color="white",lw=.62,linestyle=(0,(2,2)),alpha=.72)
   ax.text(r.target_frequency_PHz,b+h+.07+.1*(k%2),f"{r.target_rest_angstrom:.2f} Å",rotation=90,ha="center",va="bottom",fontsize=7.1)
  yt.append(b+h*.45); yl.append(f"{fam}   {META[fam][1]}   n={len(g)}")
 ax.set(yticks=yt,yticklabels=yl,xlabel="Rest frequency [PHz]",ylabel="Ionic family lanes")
 ax.set_ylim(-.18,(len(ORDER)-1)*gap+1.1); ax.set_xlim(d.rest_frequency_PHz.min()-.012,d.rest_frequency_PHz.max()+.012)
 ax.set_title("NIST SRD 78 ultraviolet rest-frequency atlas\nAll archived transitions for six MoM-z14 reference families",fontsize=17,pad=14)
 axes_style(ax); top=ax.secondary_xaxis("top",functions=(f2w,w2f)); top.set_xlabel("Rest vacuum wavelength [nm]")
 ax.text(.995,.02,"Colored fine lines: NIST SRD 78 transitions\nWhite dashed guides: selected redshift-study wavelengths\nHeights: log-scaled, family-normalized",transform=ax.transAxes,ha="right",va="bottom",fontsize=9.2,bbox={"facecolor":"#07101e","edgecolor":"#49617b","boxstyle":"round,pad=.55","alpha":.92})
 p=PNG/f"{VERSION}_NIST_SRD78_SIX_FAMILY_REST_FREQUENCY_ATLAS.png"; fig.savefig(p,dpi=420,bbox_inches="tight"); plt.show(); plt.close(fig); return p


def family_axis(ax,d,t,fam):
 g=d[d.selected_family==fam].sort_values("rest_wavelength_vacuum_nm"); col=META[fam][2]; x=g.rest_wavelength_vacuum_nm.to_numpy(); y=g.strength_normalized_family.to_numpy(); floor=1e-12
 ax.vlines(x,floor,y,color=col,lw=.72,alpha=.92); ax.scatter(x,y,s=19,color=col,edgecolors="#f2fbff",linewidths=.38,zorder=4)
 ax.set_yscale("log"); ax.set_ylim(floor,1.8); pad=max((x.max()-x.min())*.1,.025); ax.set_xlim(x.min()-pad,x.max()+pad)
 ax.set(xlabel="Rest vacuum wavelength [nm]",ylabel="Family-normalized NIST strength")
 ax.set_title(f"{fam} | {META[fam][0]} {META[fam][1]}\n{len(g)} NIST transitions | strength basis: {g.strength_basis.iloc[0]}",fontsize=12,pad=8)
 axes_style(ax); span=x.max()-x.min(); ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f" if span<.2 else "%.3f"))
 top=ax.secondary_xaxis("top",functions=(w2f,f2w)); top.set_xlabel("Rest frequency [PHz]"); fs=(C/x.min())-(C/x.max()); top.xaxis.set_major_formatter(FormatStrFormatter("%.6f" if fs<.01 else "%.4f"))
 for k,r in enumerate(t[t.family==fam].itertuples(index=False)):
  ax.axvline(r.target_rest_nm,color="white",lw=.72,linestyle=(0,(2,2)),alpha=.78)
  ax.text(r.target_rest_nm,.025+.055*(k%2),f"target {r.target_rest_angstrom:.2f} Å",transform=ax.get_xaxis_transform(),rotation=90,ha="left",va="bottom",fontsize=7.1,bbox={"facecolor":"#07101e","edgecolor":"none","alpha":.65,"pad":1})
 strong=g[g.strength_normalized_family>=.03] if len(g)>3 else g
 for k,r in enumerate(strong.itertuples(index=False)):
  ax.text(r.rest_wavelength_vacuum_nm,.96-.075*(k%4),f"{r.rest_wavelength_vacuum_angstrom:.3f} Å",transform=ax.get_xaxis_transform(),rotation=90,ha="center",va="top",fontsize=6.8,color="#dff6ff")


def pair(d,t,fams,title,name):
 fig,axs=plt.subplots(2,1,figsize=(16,11.5),constrained_layout=True)
 for ax,f in zip(axs,fams): family_axis(ax,d,t,f)
 fig.suptitle(title+"\nNIST SRD 78 discrete transitions — no synthetic spectral profile",fontsize=17)
 p=PNG/name; fig.savefig(p,dpi=420,bbox_inches="tight"); plt.show(); plt.close(fig); return p


def table_png(t):
 q=t.copy(); q["Target λ [Å]"]=q.target_rest_angstrom.map(lambda v:f"{v:.3f}"); q["Target ν [PHz]"]=q.target_frequency_PHz.map(lambda v:f"{v:.6f}"); q["Nearest NIST λ [Å]"]=q.nearest_nist_angstrom.map(lambda v:f"{v:.3f}"); q["Offset [Å]"]=q.nearest_offset_angstrom.map(lambda v:f"{v:+.4f}")
 q=q[["family","element","Target λ [Å]","Target ν [PHz]","Nearest NIST λ [Å]","Offset [Å]"]]
 fig,ax=plt.subplots(figsize=(16.5,7.8),constrained_layout=True); ax.axis("off"); ax.set_title("Selected rest-UV references and nearest archived NIST SRD 78 transitions",loc="left",fontsize=16,pad=14)
 tb=ax.table(cellText=q.values,colLabels=["Family","Element","Target λ [Å]","Target ν [PHz]","Nearest NIST λ [Å]","Offset [Å]"],cellLoc="center",colLoc="center",bbox=[.01,.04,.98,.84],colWidths=[.12,.13,.16,.16,.2,.13]); tb.auto_set_font_size(False); tb.set_fontsize(9.2); tb.scale(1,1.35)
 for (r,c),cell in tb.get_celld().items():
  cell.set_edgecolor("#3f5874"); cell.set_linewidth(.5)
  if r==0: cell.set_facecolor("#22344b"); cell.set_text_props(color="white",weight="bold")
  else:
   fam=q.iloc[r-1,0]; cell.set_facecolor("#0b1728" if r%2 else "#102039"); cell.set_text_props(color="#e8f3ff")
   if c in (0,1): cell.set_text_props(color=META[fam][2],weight="bold")
 p=PNG/f"{VERSION}_NIST_SRD78_SELECTED_REFERENCE_TABLE.png"; fig.savefig(p,dpi=420,bbox_inches="tight"); plt.show(); plt.close(fig); return p


def main():
 style(); master,copy,mode=load_master(); d,a,t=prepare(master)
 dp=CSV/f"{VERSION}_NIST_SRD78_PLOT_DATA.csv"; ap=CSV/f"{VERSION}_NIST_SRD78_FAMILY_STRENGTH_BASIS.csv"; tp=CSV/f"{VERSION}_NIST_SRD78_SELECTED_TARGETS.csv"
 d.to_csv(dp,index=False); a.to_csv(ap,index=False); t.to_csv(tp,index=False)
 ov=overview(d,t); ni=pair(d,t,("N IV]","N III]"),"Nitrogen ultraviolet reference atlas",f"{VERSION}_NIST_SRD78_NITROGEN_HIGH_DETAIL.png"); ca=pair(d,t,("C IV","C III]"),"Carbon ultraviolet reference atlas",f"{VERSION}_NIST_SRD78_CARBON_HIGH_DETAIL.png"); ho=pair(d,t,("He II","O III]"),"Helium and oxygen ultraviolet reference atlas",f"{VERSION}_NIST_SRD78_HELIUM_OXYGEN_HIGH_DETAIL.png"); tb=table_png(t)
 man=pd.DataFrame([("PNG","overview",ov),("PNG","nitrogen",ni),("PNG","carbon",ca),("PNG","helium oxygen",ho),("PNG","table",tb),("CSV","plot data",dp),("CSV","targets",tp)],columns=["type","description","path"]); mp=CSV/f"{VERSION}_OUTPUT_MANIFEST.csv"; man.to_csv(mp,index=False)
 print(f"CODE OUTPUT: {VERSION}"); print("DATABASE        NIST Standard Reference Database 78 v5.12"); print(f"INPUT MODE      {mode}"); print(f"INPUT COPY      {copy}"); print(f"TRANSITIONS     {len(d)}"); print(f"FAMILIES        {len(ORDER)}"); print(f"TARGETS         {len(t)}"); print(f"OVERVIEW PNG    {ov}"); print(f"NITROGEN PNG    {ni}"); print(f"CARBON PNG      {ca}"); print(f"HE/O PNG        {ho}"); print(f"TABLE PNG       {tb}"); print(f"PLOT CSV        {dp}"); print(f"TARGET CSV      {tp}"); print(f"MANIFEST CSV    {mp}"); print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}"); print(f"# {VERSION}")

if __name__=="__main__": main()
