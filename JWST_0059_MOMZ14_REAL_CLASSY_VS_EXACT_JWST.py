# JWST_0059
# Real HST/COS CLASSY analog curves versus a coordinate-verified MoM-z14 JWST spectrum.
# No synthetic profiles. No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib, math, subprocess, sys, time

VERSION = "JWST_0059"
GALAXY = "MoM-z14"
MOM_RA, MOM_DEC, MOM_Z = 150.0933255, 2.2731627, 14.44
JWST_PID, HST_PID = "5224", "15840"
C_UM_THz, C_AA_THz = 299.792458, 2997924.58
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd()/"JWST_OUTPUT"
PNG, CSV, DATA = OUT/"PNG", OUT/"CSV", OUT/"DATA"/VERSION
BG, AXBG, GRID = "#050712", "#07101f", "#1f6f8b"
TEXT, MUTED, CYAN, ORANGE, BLUE, GRAY = "#e6f4ff", "#8fb3c7", "#18c7d8", "#ff9d2e", "#43b9ff", "#aeb7c2"

COMPLEXES = [
    (1, "N IV] 1483,1487", [("N IV] 1483",1483.32),("N IV] 1487",1486.50)], (1479.5,1490.0)),
    (2, "C IV 1548,1551", [("C IV 1548",1548.20),("C IV 1551",1550.77)], (1544.5,1554.5)),
    (3, "He II 1640 + O III] 1661,1666", [("He II 1640",1640.42),("O III] 1661",1660.81),("O III] 1666",1666.15)], (1636.0,1670.5)),
    (4, "N III] 1747-1754", [("N III] 1747",1746.82),("N III] 1749",1748.65),("N III] 1750",1749.67),("N III] 1752",1752.16),("N III] 1754",1753.99)], (1743.5,1757.5)),
    (5, "C III] 1907,1909", [("C III] 1907",1906.68),("C III] 1909",1908.73)], (1903.5,1912.0)),
]


def need(name):
    try: importlib.import_module(name)
    except Exception: subprocess.check_call([sys.executable,"-m","pip","install","-q",name])


def col(df, names):
    lookup={str(c).lower():c for c in df.columns}
    return next((lookup[n.lower()] for n in names if n.lower() in lookup),None)


def fnum(x):
    try:
        y=float(x); return y if math.isfinite(y) else None
    except Exception: return None


def query_obs(collection,pid):
    import pandas as pd
    from astroquery.mast import Observations
    t=Observations.query_criteria(obs_collection=collection,proposal_id=str(pid))
    return t.to_pandas() if hasattr(t,"to_pandas") else pd.DataFrame(t)


def query_products(obsids):
    import pandas as pd
    from astroquery.mast import Observations
    frames=[]
    for i in range(0,len(obsids),12):
        t=Observations.get_product_list([str(x) for x in obsids[i:i+12]])
        frames.append(t.to_pandas() if hasattr(t,"to_pandas") else pd.DataFrame(t))
    return pd.concat(frames,ignore_index=True,sort=False).drop_duplicates()


def x1d_products(products):
    fn=col(products,["productFilename","productfilename"])
    sg=col(products,["productSubGroupDescription"])
    if fn is None: raise RuntimeError("No MAST product filename column")
    names=products[fn].astype(str).str.lower()
    mask=names.str.contains("x1d",na=False)&names.str.endswith(".fits",na=False)
    if sg: mask|=products[sg].astype(str).str.upper().eq("X1D")
    return products[mask].drop_duplicates(subset=[fn]).copy(),fn


def download(record,folder):
    import requests
    uri=record.get("dataURI") or record.get("dataUri") or record.get("uri")
    name=record.get("productFilename") or record.get("productfilename")
    if not uri or not name: raise RuntimeError("Product lacks URI or filename")
    dest=DATA/folder/str(name); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size>50000: return dest
    part=dest.with_suffix(dest.suffix+".part")
    for attempt in range(3):
        try:
            with requests.get("https://mast.stsci.edu/api/v0.1/Download/file",params={"uri":str(uri)},stream=True,timeout=(20,180)) as r:
                r.raise_for_status()
                with part.open("wb") as h:
                    for chunk in r.iter_content(1024*1024):
                        if chunk: h.write(chunk)
            if part.stat().st_size<50000: raise RuntimeError("Downloaded FITS is too small")
            part.replace(dest); return dest
        except Exception:
            part.unlink(missing_ok=True); time.sleep(2*(attempt+1))
    raise RuntimeError(f"Download failed: {name}")


def table_field(data,names):
    if data is None or not getattr(data,"names",None): return None,None
    lookup={str(n).upper():n for n in data.names}
    for n in names:
        if n.upper() in lookup: return lookup[n.upper()],data[lookup[n.upper()]]
    return None,None


def to_um(values,unit):
    import numpy as np
    a=np.asarray(values,float).ravel(); u=(unit or "").lower()
    if "angstrom" in u or u.strip() in {"aa","a"}: return a*1e-4
    if "nm" in u: return a*1e-3
    if "um" in u or "micron" in u: return a
    m=float(np.nanmedian(a)); return a*1e-4 if m>1000 else (a*1e-3 if m>10 else a)


def to_A(values,unit):
    import numpy as np
    a=np.asarray(values,float).ravel(); u=(unit or "").lower()
    if "angstrom" in u or u.strip() in {"aa","a"}: return a
    if "nm" in u: return a*10
    if "um" in u or "micron" in u: return a*1e4
    m=float(np.nanmedian(a)); return a*1e4 if m<10 else (a*10 if m<1000 else a)


def coords(headers):
    for h in headers:
        for rk,dk in [("SRCRA","SRCDEC"),("TARG_RA","TARG_DEC"),("RA_TARG","DEC_TARG")]:
            ra,dec=fnum(h.get(rk)),fnum(h.get(dk))
            if ra is not None and dec is not None: return ra,dec,f"{rk}/{dk}"
    return None,None,"NONE"


def read_jwst(path):
    import numpy as np, astropy.units as u
    from astropy.io import fits
    from astropy.coordinates import SkyCoord
    out=[]
    with fits.open(path,memmap=False) as hdul:
        p=hdul[0].header
        for i,hdu in enumerate(hdul[1:],1):
            wn,w=table_field(hdu.data,["WAVELENGTH","WAVE"]); fn,f=table_field(hdu.data,["FLUX"])
            if w is None or f is None: continue
            try: unit=hdu.columns[wn].unit
            except Exception: unit=None
            wave=to_um(w,unit); flux=np.asarray(f,float).ravel()
            en,e=table_field(hdu.data,["FLUX_ERROR","ERROR","ERR"])
            err=np.asarray(e,float).ravel() if e is not None else np.full_like(flux,np.nan)
            dn,dq=table_field(hdu.data,["DQ","QUALITY"])
            q=np.asarray(dq).ravel() if dq is not None else np.zeros_like(flux,int)
            ok=np.isfinite(wave)&np.isfinite(flux)&(wave>0)
            if q.size==ok.size: ok&=q==0
            if ok.sum()<5: continue
            ra,dec,source=coords([hdu.header,p]); sep=math.inf
            if ra is not None:
                sep=float(SkyCoord(ra*u.deg,dec*u.deg).separation(SkyCoord(MOM_RA*u.deg,MOM_DEC*u.deg)).arcsec)
            order=np.argsort(wave[ok])
            out.append(dict(path=path,hdu=i,ra=ra,dec=dec,coord_source=source,sep=sep,
                            source_id=hdu.header.get("SOURCEID",""),wave=wave[ok][order],flux=flux[ok][order],err=err[ok][order]))
    return out


def exact_momz14():
    import pandas as pd
    obs=query_obs("JWST",JWST_PID); oid=col(obs,["obsid","obs_id"])
    if oid is None: raise RuntimeError("JWST query has no obsid")
    products=query_products(obs[oid].dropna().astype(str).drop_duplicates().tolist())
    x1d,fn=x1d_products(products)
    size=col(x1d,["size"])
    if size: x1d=x1d.sort_values(size,ascending=False)
    candidates=[]; audit=[]
    for rec in x1d.head(40).to_dict("records"):
        try:
            path=download(rec,"JWST_GO5224"); exts=read_jwst(path); candidates+=exts
            audit.append((path.name,"OK",len(exts)))
        except Exception as exc: audit.append((str(rec.get(fn,"")),type(exc).__name__,0))
    pd.DataFrame(audit,columns=["file","status","extensions"]).to_csv(CSV/f"{VERSION}_JWST_DOWNLOAD_AUDIT.csv",index=False)
    valid=[x for x in candidates if x["coord_source"]=="SRCRA/SRCDEC" and math.isfinite(x["sep"])]
    if not valid: raise RuntimeError("No JWST X1D extension has SRCRA/SRCDEC; refusing unverified data")
    valid.sort(key=lambda x:(x["sep"],-len(x["wave"])))
    best=valid[0]
    if best["sep"]>1.0: raise RuntimeError(f"Nearest exact extraction is {best['sep']:.3f} arcsec away")
    req=(min(x[3][0] for x in COMPLEXES)*(1+MOM_Z)*1e-4,max(x[3][1] for x in COMPLEXES)*(1+MOM_Z)*1e-4)
    if best["wave"].min()>req[0] or best["wave"].max()<req[1]: raise RuntimeError("Exact JWST extraction lacks required wavelength coverage")
    out=CSV/f"{VERSION}_{GALAXY}_EXACT_JWST.csv"
    pd.DataFrame({"wavelength_um":best["wave"],"flux":best["flux"],"flux_error":best["err"]}).to_csv(out,index=False)
    return best,dict(exact_csv=str(out),product=str(best["path"]),hdu=best["hdu"],sep=best["sep"],source_id=best["source_id"])


def redshift(name,ra,dec):
    import numpy as np, astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.simbad import Simbad
    s=Simbad()
    try: s.add_votable_fields("rvz_redshift")
    except Exception: pass
    for result in [lambda:s.query_object(str(name)),lambda:s.query_region(SkyCoord(ra*u.deg,dec*u.deg),radius=8*u.arcsec)]:
        try:
            t=result()
            if t is not None:
                for c in t.colnames:
                    if "redshift" in c.lower():
                        v=np.asarray(t[c],float); v=v[np.isfinite(v)]
                        if v.size and 0<=v[0]<0.3: return float(v[0]),f"SIMBAD:{c}"
        except Exception: pass
    return None,"UNRESOLVED"


def bounds_A(row):
    lo,hi=fnum(row.get("em_min")),fnum(row.get("em_max"))
    if lo is None or hi is None: return None,None
    if hi<0.01: return lo*1e10,hi*1e10
    if hi<1000: return lo*10,hi*10
    return lo,hi


def read_hst(path):
    import numpy as np
    from astropy.io import fits
    waves=[]; fluxes=[]
    with fits.open(path,memmap=False) as hdul:
        target=str(hdul[0].header.get("TARGNAME",""))
        for hdu in hdul[1:]:
            wn,w=table_field(hdu.data,["WAVELENGTH","WAVE"]); fn,f=table_field(hdu.data,["FLUX"])
            if w is None or f is None: continue
            try: unit=hdu.columns[wn].unit
            except Exception: unit=None
            wa=to_A(w,unit); fl=np.asarray(f,float).ravel()
            dn,dq=table_field(hdu.data,["DQ","QUALITY"]); q=np.asarray(dq).ravel() if dq is not None else np.zeros_like(fl,int)
            ok=np.isfinite(wa)&np.isfinite(fl)&(wa>0)
            if q.size==ok.size: ok&=q==0
            waves.append(wa[ok]); fluxes.append(fl[ok])
    if not waves: return None
    w=np.concatenate(waves); f=np.concatenate(fluxes); o=np.argsort(w)
    return dict(path=path,target=target,wave=w[o],flux=f[o])


def normalize(f):
    import numpy as np
    med=float(np.nanmedian(f)); mad=float(np.nanmedian(np.abs(f-med))); scale=1.4826*mad
    if not math.isfinite(scale) or scale<=0: scale=float(np.nanstd(f)) or 1.0
    return (f-med)/scale


def score(spec,z,components,window):
    import numpy as np
    rw=spec["wave"]/(1+z); m=(rw>=window[0])&(rw<=window[1])
    if m.sum()<20: return -math.inf,[]
    w=rw[m]; f=normalize(spec["flux"][m]); peaks=[]
    for _,c in components:
        q=np.abs(w-c)<=0.75; peaks.append(float(np.nanmax(f[q])) if q.sum()>=2 else -5.0)
    return sum(max(-1,min(12,p)) for p in peaks)+1.5*sum(p>0.5 for p in peaks),peaks


def classy_curves():
    import pandas as pd
    obs=query_obs("HST",HST_PID)
    oid,target,ra,dec=col(obs,["obsid","obs_id"]),col(obs,["target_name","target"]),col(obs,["s_ra"]),col(obs,["s_dec"])
    exp,inst=col(obs,["t_exptime"]),col(obs,["instrument_name"])
    if None in [oid,target,ra,dec]: raise RuntimeError("HST query lacks required columns")
    if inst:
        mask=obs[inst].astype(str).str.contains("COS",case=False,na=False)
        if mask.any(): obs=obs[mask].copy()
    obs["_exp"]=pd.to_numeric(obs[exp],errors="coerce").fillna(0) if exp else 0
    top=set(obs.groupby(target)["_exp"].sum().sort_values(ascending=False).head(30).index.astype(str))
    obs=obs[obs[target].astype(str).isin(top)].copy()
    zmap={}; zaudit=[]
    for name,g in obs.groupby(target):
        r,d=fnum(g.iloc[0][ra]),fnum(g.iloc[0][dec])
        if r is None: continue
        z,method=redshift(str(name),r,d); zaudit.append((name,r,d,z,method))
        if z is not None: zmap[str(name)]=(z,method)
    pd.DataFrame(zaudit,columns=["target","ra","dec","z","method"]).to_csv(CSV/f"{VERSION}_CLASSY_REDSHIFTS.csv",index=False)
    if not zmap: raise RuntimeError("No CLASSY redshifts resolved")
    wanted=set(); mapping={}
    for _,_,components,window in COMPLEXES:
        center=sum(x[1] for x in components)/len(components); ranked=[]
        for _,row in obs.iterrows():
            name=str(row[target]); info=zmap.get(name)
            if not info: continue
            lo,hi=bounds_A(row); expected=center*(1+info[0])
            if lo is not None and not(lo<=expected<=hi): continue
            ranked.append((float(row["_exp"]),str(row[oid]),name,info))
        used=set()
        for _,ob,name,info in sorted(ranked,reverse=True):
            if name in used: continue
            wanted.add(ob); mapping[ob]=(name,info[0],info[1]); used.add(name)
            if len(used)>=12: break
    products=query_products(sorted(wanted)); x1d,fn=x1d_products(products); poid=col(x1d,["obsID","obsid","obs_id"])
    if poid is None: raise RuntimeError("CLASSY products have no observation ID column")
    spectra=[]; audit=[]
    for ob,g in x1d.groupby(poid):
        key=str(ob); meta=mapping.get(key) or next((v for k,v in mapping.items() if k.rstrip(".0")==key.rstrip(".0")),None)
        if not meta: continue
        size=col(g,["size"])
        if size: g=g.sort_values(size,ascending=False)
        for rec in g.head(2).to_dict("records"):
            try:
                p=download(rec,"HST_CLASSY"); s=read_hst(p)
                if s: s.update(name=meta[0],z=meta[1],z_method=meta[2]); spectra.append(s); audit.append((p.name,meta[0],"OK",len(s["wave"])))
            except Exception as exc: audit.append((str(rec.get(fn,"")),meta[0],type(exc).__name__,0))
    pd.DataFrame(audit,columns=["file","target","status","samples"]).to_csv(CSV/f"{VERSION}_CLASSY_DOWNLOADS.csv",index=False)
    if not spectra: raise RuntimeError("No real CLASSY spectra downloaded")
    best={}; rows=[]
    for n,name,components,window in COMPLEXES:
        candidates=[]
        for s in spectra:
            sc,peaks=score(s,s["z"],components,window); candidates.append((sc,peaks,s)); rows.append((n,name,s["name"],s["z"],s["path"].name,sc,str(peaks)))
        candidates.sort(key=lambda x:x[0],reverse=True); chosen=candidates[0]
        if not math.isfinite(chosen[0]): raise RuntimeError(f"No CLASSY curve covers {name}")
        best[n]=dict(score=chosen[0],peaks=chosen[1],spec=chosen[2])
    pd.DataFrame(rows,columns=["n","complex","target","z","product","score","component_peak_sigma"]).to_csv(CSV/f"{VERSION}_CLASSY_SELECTION.csv",index=False)
    return best


def style(ax):
    ax.set_facecolor(AXBG); ax.grid(True,color=GRID,lw=.45,alpha=.48); ax.tick_params(colors=TEXT,labelsize=7.2)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT); ax.title.set_color(TEXT)
    for s in ax.spines.values(): s.set_color("#4fa8c8"); s.set_linewidth(.82)


def local(w,f,lo,hi):
    import numpy as np
    m=np.isfinite(w)&np.isfinite(f)&(w>=lo)&(w<=hi)
    if m.sum()<3: return np.array([]),np.array([])
    o=np.argsort(w[m]); return w[m][o],normalize(f[m][o])


def plot(jw,jmeta,refs):
    import numpy as np, pandas as pd, matplotlib.pyplot as plt
    fig,axs=plt.subplots(5,2,figsize=(18.5,20.5),facecolor=BG); audits=[]; component_rows=[]
    for i,(n,name,components,window) in enumerate(COMPLEXES):
        left,right=axs[i]; style(left); style(right); ref=refs[n]; s=ref["spec"]
        rw=s["wave"]/(1+s["z"]); x1,y1=local(rw,s["flux"],*window)
        olo,ohi=window[0]*(1+MOM_Z)*1e-4,window[1]*(1+MOM_Z)*1e-4
        x2,y2=local(jw["wave"],jw["flux"],olo,ohi)
        left.plot(x1,y1,color=CYAN,lw=.9); left.scatter(x1,y1,s=9,color=BLUE,edgecolor=BG,lw=.25)
        right.plot(x2,y2,color=ORANGE,lw=.9); right.scatter(x2,y2,s=18,color=TEXT,edgecolor=BG,lw=.3)
        for k,(label,rest) in enumerate(components,1):
            observed=rest*(1+MOM_Z)*1e-4
            left.axvline(rest,color=GRAY,ls="--",lw=1); right.axvline(observed,color=GRAY,ls="--",lw=1)
            left.text(rest,.98,str(k),transform=left.get_xaxis_transform(),ha="center",va="top",color=TEXT,fontsize=7,fontweight="bold")
            right.text(observed,.98,str(k),transform=right.get_xaxis_transform(),ha="center",va="top",color=TEXT,fontsize=7,fontweight="bold")
            component_rows.append((n,name,k,label,rest,C_AA_THz/rest,observed,C_UM_THz/observed,s["name"],s["path"].name,Path(jmeta["product"]).name))
        left.set_xlim(*window); right.set_xlim(olo,ohi)
        vals=np.r_[y1,y2]; ymin,ymax=float(np.nanmin(vals)),float(np.nanmax(vals)); pad=.08*(ymax-ymin if ymax>ymin else 1)
        left.set_ylim(ymin-pad,ymax+pad); right.set_ylim(ymin-pad,ymax+pad)
        left.set_title(f"{n} {name} — REAL HST/COS analog {s['name']}",fontsize=9.1)
        right.set_title(f"{n} {name} — VERIFIED JWST {GALAXY}",fontsize=9.1)
        left.set_xlabel("Reference rest wavelength, Å",fontsize=7.8); right.set_xlabel("Observed wavelength, µm",fontsize=7.8)
        left.set_ylabel("Local normalized flux",fontsize=7.8); right.set_ylabel("Local normalized flux",fontsize=7.8)
        top=left.secondary_xaxis("top",functions=(lambda a:C_AA_THz/np.asarray(a),lambda v:C_AA_THz/np.asarray(v)))
        top.set_xlabel("Rest frequency, THz",color=TEXT,fontsize=7.4); top.tick_params(colors=TEXT,labelsize=6.7)
        top=right.secondary_xaxis("top",functions=(lambda a:C_UM_THz/np.asarray(a),lambda v:C_UM_THz/np.asarray(v)))
        top.set_xlabel("Observed frequency, THz",color=TEXT,fontsize=7.4); top.tick_params(colors=TEXT,labelsize=6.7)
        labels="   ".join(f"{k}:{v[0]}" for k,v in enumerate(components,1))
        left.text(.015,.955,f"Independent real galaxy spectrum\nz={s['z']:.6f}\n{labels}",transform=left.transAxes,ha="left",va="top",color=TEXT,fontsize=6.3,bbox=dict(boxstyle="round",fc="#07111f",ec=CYAN,alpha=.95))
        right.text(.015,.955,f"Exact coordinate-matched extraction\nseparation={jmeta['sep']:.3f} arcsec\nmarkers are expected components; PRISM may blend them",transform=right.transAxes,ha="left",va="top",color=TEXT,fontsize=6.3,bbox=dict(boxstyle="round",fc="#07111f",ec=ORANGE,alpha=.95))
        passed=(Path(s["path"]).resolve()!=Path(jw["path"]).resolve() and len(x1)>=20 and len(x2)>=3 and len(components)>0)
        audits.append((n,name,passed,len(x1),len(x2),len(components),s["name"],s["path"].name,Path(jmeta["product"]).name))
    audit=CSV/f"{VERSION}_PREFLIGHT_AUDIT.csv"; pd.DataFrame(audits,columns=["n","complex","passed","rest_samples","observed_samples","component_markers_each_side","analog","hst_file","jwst_file"]).to_csv(audit,index=False)
    if not all(x[2] for x in audits): plt.close(fig); raise RuntimeError(f"Preflight failed; no plot rendered: {audit}")
    fig.suptitle(f"{VERSION} — {GALAXY}: REAL HST/COS REST-ANALOG CURVES vs EXACT JWST OBSERVED CURVES",color=TEXT,fontsize=15.5,fontweight="bold",y=.993)
    fig.text(.5,.977,"Every row uses an independent real HST spectrum on the left and the coordinate-matched MoM-z14 extraction on the right. The wavelength windows are identical after applying (1+z).",ha="center",color=MUTED,fontsize=9)
    fig.text(.5,.018,"No synthetic peaks. A real analog is not MoM-z14's unknowable emitted spectrum. Multiple components can be unresolved by NIRSpec/PRISM.",ha="center",color=MUTED,fontsize=8.4)
    fig.subplots_adjust(left=.055,right=.985,top=.955,bottom=.04,hspace=.36,wspace=.16)
    out=PNG/f"{VERSION}_{GALAXY}_REAL_CLASSY_REST_VS_EXACT_JWST.png"; fig.savefig(out,dpi=235,facecolor=BG); plt.show(); plt.close(fig)
    comp=CSV/f"{VERSION}_{GALAXY}_COMPONENTS.csv"; pd.DataFrame(component_rows,columns=["n","complex","component","label","rest_A","rest_THz","observed_um","observed_THz","analog","hst_product","jwst_product"]).to_csv(comp,index=False)
    return out,comp,audit


def main():
    for p in ["numpy","pandas","matplotlib","astropy","astroquery","requests"]: need(p)
    for p in [PNG,CSV,DATA]: p.mkdir(parents=True,exist_ok=True)
    print(f"CODE OUTPUT: {VERSION}")
    print("PRECHECK: independent HST curves + exact SRCRA/SRCDEC JWST extraction required")
    jw,meta=exact_momz14(); refs=classy_curves(); out,comp,audit=plot(jw,meta,refs)
    print(f"JWST separation arcsec : {meta['sep']:.6f}")
    print(f"JWST product           : {Path(meta['product']).name} HDU={meta['hdu']}")
    print(f"Plot PNG               : {out}")
    print(f"Components CSV          : {comp}")
    print(f"Preflight audit         : {audit}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__=="__main__": main()
