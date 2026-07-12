# JWST_0073
# Real numerical rest-frame H I Ly-alpha spectrum (LASD) versus raw MoM-z14 JWST/PRISM.
# No AI images. No Gaussian profiles. No smoothing. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin
from io import BytesIO
import importlib, importlib.util, subprocess, sys, zipfile, math

VERSION="JWST_0073"; GALAXY="MoM-z14"; Z=14.44
LYA_NM=121.56701; OBS_LYA_NM=LYA_NM*(1+Z); C=299792.458
ROOT=Path('/content') if Path('/content').exists() else Path.cwd()
OUT=ROOT/'JWST_OUTPUT'; PNG=OUT/'PNG'; CSV=OUT/'CSV'; DATA=OUT/'DATA'/VERSION
BG='#050712'; AX='#07101f'; GRID='#1f6f8b'; TEXT='#e6f4ff'; MUTED='#8fb3c7'
LAB='#ffd84d'; OBS='#ff9d2e'; RESTMARK='#6ee7ff'; OBSMARK='#ff5a66'; POINT='#d9edf7'


def need(name,pip=None):
    try: importlib.import_module(name)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pip or name])


def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,str(path)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def f_thz(nm): return C/np.asarray(nm,float)
def w_nm(thz): return C/np.asarray(thz,float)


def jwst_csv():
    hits=sorted(CSV.glob('JWST_*_MoM-z14_EXACT_JWST.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
    if hits: return hits[0],'coordinate-matched cached JWST X1D',None
    helper=ROOT/'JWST_0060_MOMZ14_FAST_CONE_CLASSY.py'
    subprocess.run(['curl','-fsSL','-o',str(helper),'https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0060_MOMZ14_FAST_CONE_CLASSY.py'],check=True)
    m=load_module('m0060',helper); m.VERSION=VERSION; m.OUT=OUT; m.PNG=PNG; m.CSV=CSV; m.DATA=DATA/'MAST'; m.MAX_JWST_X1D=24
    b=m.load_base(m.ensure_base()); _,meta=m.exact_momz14_cone(b); p=Path(meta['exact_csv'])
    return p,f"coordinate-verified GO-5224 X1D; separation={meta['sep']:.6f} arcsec",meta


def load_jwst(path):
    d=pd.read_csv(path); cols={str(c).lower():c for c in d.columns}
    wc=next((cols[k] for k in ['wavelength_um','wavelength_nm','wavelength','wave'] if k in cols),None)
    fc=next((cols[k] for k in ['flux','raw_flux','jwst_flux'] if k in cols),None)
    ec=next((cols[k] for k in ['flux_error','error','err'] if k in cols),None)
    if wc is None or fc is None: raise RuntimeError(f'No wavelength/flux columns in {path.name}')
    w=pd.to_numeric(d[wc],errors='coerce').to_numpy(float); y=pd.to_numeric(d[fc],errors='coerce').to_numpy(float)
    e=pd.to_numeric(d[ec],errors='coerce').to_numpy(float) if ec else np.full_like(y,np.nan)
    ok=np.isfinite(w)&np.isfinite(y)&(w>0); w,y,e=w[ok],y[ok],e[ok]
    med=float(np.nanmedian(w)); name=str(wc).lower(); wn=w*1000 if '_um' in name or med<20 else (w if '_nm' in name or med<10000 else w/10)
    o=np.argsort(wn); return wn[o],y[o],e[o],str(wc),str(fc)


def is_zip(r): return r.content[:4]==b'PK\x03\x04' or 'zip' in r.headers.get('content-type','').lower()


def download_lasd():
    from bs4 import BeautifulSoup
    base='https://lasd.lyman-alpha.com'; page=requests.get(base+'/download',timeout=90); page.raise_for_status(); soup=BeautifulSoup(page.text,'html.parser')
    tries=[]
    for a in soup.find_all('a',href=True):
        if any(k in (a.get_text(' ',strip=True)+' '+a['href']).lower() for k in ['spectra','measure','download']): tries.append(('GET',urljoin(page.url,a['href']),{}))
    for form in soup.find_all('form'):
        action=urljoin(page.url,form.get('action') or page.url); method=(form.get('method') or 'GET').upper(); common={}
        for x in form.find_all('input'):
            if x.get('name'): common[x['name']]=x.get('value','')
        buttons=form.find_all(['button','input'])
        for b in buttons or [None]:
            p=dict(common)
            if b is not None and b.get('name'): p[b['name']]=b.get('value') or b.get_text(' ',strip=True)
            tries.append((method,action,p))
    for u in ['/download/spectra','/download/all_spectra','/download/measurements','/download/all_measurements','/download_spectra','/download_measurements']:
        tries.append(('GET',base+u,{}))
    raw=DATA/'LASD'; raw.mkdir(parents=True,exist_ok=True); archives=[]; audit=[]; seen=set()
    for i,(method,url,payload) in enumerate(tries,1):
        try:
            r=requests.post(url,data=payload,timeout=180) if method=='POST' else requests.get(url,params=payload,timeout=180)
            audit.append((i,method,url,r.status_code,len(r.content),is_zip(r)))
            if r.status_code==200 and is_zip(r) and len(r.content)>1000:
                key=hash(r.content)
                if key not in seen:
                    seen.add(key); q=raw/f'lasd_{len(archives)+1:02d}.zip'; q.write_bytes(r.content); archives.append(q)
        except Exception as ex: audit.append((i,method,url,type(ex).__name__,0,False))
    audit_path=CSV/f'{VERSION}_LASD_DOWNLOAD_AUDIT.csv'; pd.DataFrame(audit,columns=['attempt','method','url','status','bytes','zip']).to_csv(audit_path,index=False)
    if not archives: raise RuntimeError(f'No LASD numerical ZIP downloaded; audit={audit_path}')
    ex=raw/'EXTRACTED'; ex.mkdir(parents=True,exist_ok=True)
    for q in archives:
        try:
            with zipfile.ZipFile(q) as zf: zf.extractall(ex/q.stem)
        except zipfile.BadZipFile: pass
    return ex,audit_path


def read_spec(path):
    try: d=pd.read_csv(path,comment='#',sep=r'\s+|,',engine='python',header=None)
    except Exception: return None
    if d.shape[1]<2:return None
    w=pd.to_numeric(d.iloc[:,0],errors='coerce').to_numpy(float); y=pd.to_numeric(d.iloc[:,1],errors='coerce').to_numpy(float)
    e=pd.to_numeric(d.iloc[:,2],errors='coerce').to_numpy(float) if d.shape[1]>2 else np.full_like(y,np.nan)
    ok=np.isfinite(w)&np.isfinite(y)&(w>0)
    return (w[ok],y[ok],e[ok]) if ok.sum()>10 else None


def choose_lasd(root):
    tables=list(root.rglob('zsysdf.ascii'))
    if not tables: raise RuntimeError('LASD ZIP lacks zsysdf.ascii systemic-redshift table')
    tab=pd.read_csv(tables[0],sep=r'\s+',engine='python'); names={str(c).lower():c for c in tab.columns}
    filecol=names.get('spectral_file'); zcol=names.get('z'); ewcol=names.get('ew'); rcol=names.get('r')
    if filecol is None or zcol is None: raise RuntimeError('LASD table lacks Spectral_file or z')
    files={p.name:p for p in root.rglob('*') if p.is_file() and p.name not in ['zsysdf.ascii','zautodf.ascii']}
    scored=[]
    for _,row in tab.iterrows():
        p=files.get(Path(str(row[filecol])).name); z=pd.to_numeric(row[zcol],errors='coerce')
        if p is None or not np.isfinite(z): continue
        s=read_spec(p)
        if s is None: continue
        w,y,e=s; rest=w/(1+float(z))/10; m=(rest>=119)&(rest<=124.2)
        if m.sum()<25: continue
        yy=y[m]; noise=1.4826*np.nanmedian(np.abs(yy-np.nanmedian(yy))); noise=noise if np.isfinite(noise) and noise>0 else 1
        score=(np.nanmax(yy)-np.nanmedian(yy))/noise+0.001*m.sum(); ew=pd.to_numeric(row[ewcol],errors='coerce') if ewcol else np.nan
        if np.isfinite(ew): score+=0.01*float(ew)
        scored.append((score,p,float(z),float(ew) if np.isfinite(ew) else np.nan,float(pd.to_numeric(row[rcol],errors='coerce')) if rcol and np.isfinite(pd.to_numeric(row[rcol],errors='coerce')) else np.nan,w,y,e,rest,m))
    if not scored: raise RuntimeError('No well-sampled downloadable LASD Ly-alpha spectrum found')
    scored.sort(key=lambda x:x[0],reverse=True); score,p,z,ew,res,w,y,e,rest,m=scored[0]; o=np.argsort(rest[m])
    return dict(path=p,z=z,ew=ew,res=res,score=score,nm=rest[m][o],flux=y[m][o],err=e[m][o],table=tables[0])


def style(a):
    a.set_facecolor(AX); a.grid(True,color=GRID,lw=.48,alpha=.45); a.tick_params(colors=TEXT,labelsize=8.5)
    a.xaxis.label.set_color(TEXT); a.yaxis.label.set_color(TEXT); a.title.set_color(TEXT)
    for s in a.spines.values(): s.set_color('#4fa8c8'); s.set_linewidth(.8)


def limits(y):
    lo,hi=float(np.nanmin(y)),float(np.nanmax(y)); p=.06*(hi-lo) if hi>lo else 1e-8; return lo-p,hi+p


def plot(ref,jx,jy,je,jpath,jstatus):
    rf=f_thz(ref['nm']); rn=(ref['flux']-np.nanpercentile(ref['flux'],5))/(np.nanpercentile(ref['flux'],95)-np.nanpercentile(ref['flux'],5)); ro=np.argsort(rf)
    of=f_thz(jx); oo=np.argsort(of); rf,rn,rnm,rraw,rerr=rf[ro],rn[ro],ref['nm'][ro],ref['flux'][ro],ref['err'][ro]; of,onm,oy,oe=of[oo],jx[oo],jy[oo],je[oo]
    fig,(l,r)=plt.subplots(1,2,figsize=(15.8,6.5),facecolor=BG); style(l);style(r)
    l.step(rf,rn,where='mid',color=LAB,lw=.7); l.scatter(rf,rn,s=10,color=LAB,alpha=.6); l.axvline(float(f_thz(LYA_NM)),color=RESTMARK,ls=(0,(3,5)),lw=.58,alpha=.62)
    r.step(of,oy,where='mid',color=OBS,lw=.7); r.scatter(of,oy,s=30,color=POINT,edgecolor=BG,lw=.3); r.axvline(float(f_thz(OBS_LYA_NM)),color=OBSMARK,ls=(0,(3,5)),lw=.58,alpha=.62)
    l.set(xlim=(rf.min(),rf.max()),ylim=limits(rn),title='REAL NUMERICAL REST-FRAME Lyα — LASD',xlabel='Rest frequency, THz',ylabel='Raw LASD flux, independently normalized')
    r.set(xlim=(of.min(),of.max()),ylim=limits(oy),title='RAW MoM-z14 JWST/PRISM — Lyα BREAK REGION',xlabel='Observed frequency, THz',ylabel='JWST flux samples')
    lt=l.secondary_xaxis('top',functions=(w_nm,f_thz)); lt.set_xlabel('Rest wavelength, nm',color=TEXT); lt.tick_params(colors=TEXT,labelsize=8)
    rt=r.secondary_xaxis('top',functions=(w_nm,f_thz)); rt.set_xlabel('Observed wavelength, nm',color=TEXT); rt.tick_params(colors=TEXT,labelsize=8)
    l.text(.02,.05,f"raw LASD file: {ref['path'].name}\nsource z={ref['z']:.6f}\nrest Lyα={LYA_NM:.6f} nm",transform=l.transAxes,color=TEXT,fontsize=7.3,bbox=dict(boxstyle='round',fc=AX,ec=LAB,alpha=.94))
    r.text(.02,.05,f"all raw PRISM bins={len(onm)}\npublished z={Z:.2f}\nshifted Lyα={OBS_LYA_NM:.3f} nm\nline marks published-z position",transform=r.transAxes,color=TEXT,fontsize=7.3,bbox=dict(boxstyle='round',fc=AX,ec=OBSMARK,alpha=.94))
    fig.suptitle(f'{VERSION} — H I Lyα: REAL REST-FRAME DATA versus MoM-z14',color=TEXT,fontsize=14.7,fontweight='bold',y=.982)
    fig.text(.5,.914,'Independent measured numerical datasets. Left is an empirical LASD rest-frame galaxy spectrum, not a laboratory discharge. No smoothing or synthetic profile.',ha='center',color=MUTED,fontsize=8.2)
    fig.text(.5,.017,f"LASD: {ref['path']} | JWST: {jpath.name} ({jstatus})",ha='center',color=MUTED,fontsize=7.2); fig.subplots_adjust(left=.075,right=.985,top=.825,bottom=.12,wspace=.16)
    pp=PNG/f'{VERSION}_{GALAXY}_HI_LYA_LASD_VS_JWST.png'; fig.savefig(pp,dpi=245,facecolor=BG); plt.show(); plt.close(fig)
    rc=CSV/f'{VERSION}_LASD_RAW_REST_LYA.csv'; oc=CSV/f'{VERSION}_{GALAXY}_RAW_LYA_WINDOW.csv'
    pd.DataFrame(dict(rest_wavelength_nm=rnm,rest_frequency_THz=rf,raw_flux=rraw,raw_flux_error=rerr,display_normalized_flux=rn)).to_csv(rc,index=False)
    pd.DataFrame(dict(observed_wavelength_nm=onm,observed_frequency_THz=of,jwst_flux=oy,jwst_flux_error=oe,published_z_shifted_lya_nm=OBS_LYA_NM)).to_csv(oc,index=False)
    return pp,rc,oc


def main():
    global np,pd,plt,requests
    for n,p in [('numpy',None),('pandas',None),('matplotlib',None),('requests',None),('bs4','beautifulsoup4'),('astropy',None),('astroquery',None)]: need(n,p)
    import numpy as np, pandas as pd, matplotlib.pyplot as plt, requests
    PNG.mkdir(parents=True,exist_ok=True); CSV.mkdir(parents=True,exist_ok=True); DATA.mkdir(parents=True,exist_ok=True)
    print(f'CODE OUTPUT: {VERSION}')
    print('STEP 1/4 | Load coordinate-verified MoM-z14 JWST spectrum'); jp,js,meta=jwst_csv(); w,y,e,wc,fc=load_jwst(jp); m=(w>=1760)&(w<=1995); jx,jy,je=w[m],y[m],e[m]
    if len(jx)<6: raise RuntimeError(f'Only {len(jx)} JWST bins in Ly-alpha window')
    print('STEP 2/4 | Download LASD raw numerical spectra'); root,audit=download_lasd()
    print('STEP 3/4 | Select well-sampled systemic-redshift Ly-alpha spectrum'); ref=choose_lasd(root)
    print('STEP 4/4 | Plot raw rest and observed spectra'); pp,rc,oc=plot(ref,jx,jy,je,jp,js)
    ac=CSV/f'{VERSION}_SOURCE_AUDIT.csv'; pd.DataFrame([dict(reference_class='empirical rest-frame Ly-alpha galaxy spectrum; not laboratory',reference_file=str(ref['path']),reference_redshift=ref['z'],reference_EW_A=ref['ew'],reference_resolution=ref['res'],jwst_source=str(jp),jwst_status=js,rest_lya_nm=LYA_NM,observed_lya_nm=OBS_LYA_NM)]).to_csv(ac,index=False)
    print(f'REST H I Ly-alpha   : {LYA_NM:.6f} nm'); print(f'OBSERVED z=14.44    : {OBS_LYA_NM:.6f} nm'); print(f'LASD RAW FILE       : {ref["path"]}'); print(f'LASD SAMPLES        : {len(ref["nm"])}'); print(f'JWST RAW BINS       : {len(jx)}'); print(f'PLOT PNG            : {pp}'); print(f'REFERENCE CSV       : {rc}'); print(f'JWST CSV            : {oc}'); print(f'SOURCE AUDIT CSV    : {ac}'); print(f'DOWNLOAD AUDIT CSV  : {audit}'); print(datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')); print(f'# {VERSION}')

if __name__=='__main__': main()
