# JWST_0188
import sys, subprocess, warnings, io, os, re, tarfile, zipfile, shutil
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

for pkg in ['requests','astropy','pandas','numpy','matplotlib','scipy']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from astropy.io import votable, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from IPython.display import display, clear_output, Image
import ipywidgets as widgets
from google.colab import drive

VERSION='JWST_0188'
PROJECT='2023.1.00336.S'
TARGET_GHZ=279.901
NU_REST_GHZ=3393.006244
RA_DEG=53.1647632
DEC_DEG=-27.7746223
DRIVE_ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S')
ARCHIVE_DIR=DRIVE_ROOT/'ARCHIVE'
EXTRACT_DIR=DRIVE_ROOT/'EXTRACTED'
PNG_DIR=DRIVE_ROOT/'PNG'
CSV_DIR=DRIVE_ROOT/'CSV'
LOCAL_OUT=Path('/content/JWST_OUTPUT')
for d in [ARCHIVE_DIR,EXTRACT_DIR,PNG_DIR,CSV_DIR,LOCAL_OUT/'PNG',LOCAL_OUT/'CSV']:
    d.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})

ESO_TAP='https://almascience.eso.org/tap'
ESO_DATALINK='https://almascience.eso.org/datalink/sync'
MAX_ARCHIVE_GB=60.0

run_btn=widgets.Button(description='Mount Drive + fetch ALMA archive',button_style='success',layout=widgets.Layout(width='360px'))
gallery_btn=widgets.Button(description='Open output gallery',button_style='info',layout=widgets.Layout(width='220px'))
out=widgets.Output()


def fmt_bytes(n):
    n=float(n)
    for unit in ['B','KB','MB','GB','TB']:
        if n<1024: return f'{n:.1f} {unit}'
        n/=1024
    return f'{n:.1f} PB'


def tap_query():
    adql=f"SELECT * FROM ivoa.obscore WHERE proposal_id='{PROJECT}'"
    r=requests.get(ESO_TAP+'/sync',params={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'votable','QUERY':adql},timeout=180)
    r.raise_for_status()
    return votable.parse_single_table(io.BytesIO(r.content)).to_table().to_pandas()


def datalink(uid):
    r=requests.get(ESO_DATALINK,params={'ID':uid},timeout=180)
    r.raise_for_status()
    return votable.parse_single_table(io.BytesIO(r.content)).to_table().to_pandas()


def col(row,*names):
    for n in names:
        if n in row.index and pd.notna(row[n]): return str(row[n])
    return ''


def choose_mous(rows):
    candidates=[]
    for _,r in rows.iterrows():
        uid=col(r,'member_ous_uid','obs_publisher_did','obs_id')
        target=(col(r,'target_name','obs_title','proposal_title')+' '+col(r,'source_name')).lower()
        score=0
        if 'jades-gs-z11-0' in target or 'jades' in target: score+=100
        if '279.901' in target: score+=30
        if uid.startswith('uid://'): score+=20
        if uid: candidates.append((score,uid))
    candidates=sorted(set(candidates),reverse=True)
    return candidates


def choose_archive(products):
    ranked=[]
    for _,r in products.iterrows():
        url=col(r,'access_url','accessURL')
        desc=col(r,'description','content_type','semantics','content_qualifier')
        size_s=col(r,'content_length','contentLength')
        try: size=float(size_s)
        except: size=np.nan
        text=(url+' '+desc).lower()
        score=0
        if 'readme' in text or (np.isfinite(size) and size<10000): score-=200
        if any(k in text for k in ['dataset','tar','tgz','zip']): score+=20
        if np.isfinite(size):
            gb=size/1024**3
            if 1<gb<=MAX_ARCHIVE_GB: score+=40
            if gb>MAX_ARCHIVE_GB: score-=500
        ranked.append((score,size,url,desc))
    ranked.sort(key=lambda x:(-x[0],x[1] if np.isfinite(x[1]) else 1e99))
    return ranked


def filename_from_response(url,resp):
    cd=resp.headers.get('content-disposition','')
    m=re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)',cd,re.I)
    if m: return Path(m.group(1)).name
    name=url.split('?')[0].rstrip('/').split('/')[-1]
    return name or f'{VERSION}_ALMA_ARCHIVE.tar'


def resumable_download(url):
    probe=requests.get(url,stream=True,timeout=180)
    probe.raise_for_status()
    total=int(probe.headers.get('content-length',0))
    name=filename_from_response(url,probe)
    probe.close()
    path=ARCHIVE_DIR/name
    existing=path.stat().st_size if path.exists() else 0
    headers={}
    mode='wb'
    if existing>0 and total>existing:
        headers['Range']=f'bytes={existing}-'; mode='ab'
        print(f'Resuming existing file at {fmt_bytes(existing)}')
    elif total and existing==total:
        print(f'Archive already complete: {path}')
        return path
    with requests.get(url,headers=headers,stream=True,timeout=300) as r:
        r.raise_for_status()
        if headers and r.status_code!=206:
            print('Server did not honor resume; restarting archive download.')
            existing=0; mode='wb'
        expected=total or int(r.headers.get('content-length',0))+existing
        with open(path,mode) as f:
            done=existing
            for chunk in r.iter_content(8*1024*1024):
                if chunk:
                    f.write(chunk); done+=len(chunk)
                    if expected:
                        print(f'\rDownloading to Drive {100*done/expected:6.2f}% ({fmt_bytes(done)}/{fmt_bytes(expected)})',end='')
        print()
    return path


def list_archive(path):
    members=[]
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t:
            for m in t.getmembers():
                n=m.name.lower()
                if n.endswith(('.fits','.fits.gz','.fit','.fz')):
                    score=0
                    if 'cube' in n or 'image' in n: score+=50
                    if 'spw22' in n or 'spw.22' in n: score+=40
                    if 'jades-gs-z11-0' in n: score+=40
                    if any(k in n for k in ['pbcor','science','contsub']): score+=20
                    if any(k in n for k in ['mom0','moment','continuum']): score-=40
                    members.append((score,m.size,m.name))
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                n=m.filename.lower()
                if n.endswith(('.fits','.fits.gz','.fit','.fz')):
                    score=(50 if 'cube' in n or 'image' in n else 0)+(40 if 'spw22' in n or 'spw.22' in n else 0)+(40 if 'jades-gs-z11-0' in n else 0)
                    members.append((score,m.file_size,m.filename))
    members.sort(key=lambda x:(-x[0],x[1]))
    return members


def extract_member(path,name):
    outpath=EXTRACT_DIR/Path(name).name
    if outpath.exists() and outpath.stat().st_size>0: return outpath
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t, t.extractfile(name) as src, open(outpath,'wb') as dst:
            shutil.copyfileobj(src,dst,length=8*1024*1024)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z, z.open(name) as src, open(outpath,'wb') as dst:
            shutil.copyfileobj(src,dst,length=8*1024*1024)
    return outpath


def spectral_axis(h,n):
    spec=None
    for i in range(1,int(h.get('NAXIS',0))+1):
        if any(k in str(h.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec=i; break
    if spec is None: raise ValueError('No spectral axis')
    pix=np.arange(n,dtype=float)
    vals=float(h[f'CRVAL{spec}'])+(pix+1-float(h[f'CRPIX{spec}']))*float(h[f'CDELT{spec}'])
    ctype=str(h[f'CTYPE{spec}']).upper(); cunit=str(h.get(f'CUNIT{spec}','Hz')).lower()
    if 'FREQ' in ctype: return vals*(1e-9 if 'hz' in cunit else 1.0),spec
    rest=float(h.get('RESTFRQ',h.get('RESTFREQ',NU_REST_GHZ*1e9)))
    vel=vals*(1e-3 if 'm/s' in cunit else 1.0)
    return rest*(1-vel/299792.458)*1e-9,spec


def extract_spectrum(fpath):
    with fits.open(fpath,memmap=True) as hdul:
        hdu=next((h for h in hdul if getattr(h,'data',None) is not None and np.asarray(h.data).ndim>=3),None)
        if hdu is None: raise ValueError('No 3-D cube HDU')
        hdr=hdu.header.copy(); arr=np.squeeze(np.asarray(hdu.data,dtype=float))
    if arr.ndim!=3: raise ValueError(f'Cube shape after squeeze is {arr.shape}')
    naxis=int(hdr['NAXIS']); spec_fits=None
    for i in range(1,naxis+1):
        if any(k in str(hdr.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec_fits=i
    spec_np=naxis-spec_fits
    while spec_np>=arr.ndim: spec_np-=1
    cube=np.moveaxis(arr,spec_np,0)
    freq,_=spectral_axis(hdr,cube.shape[0])
    if not (np.nanmin(freq)<=TARGET_GHZ<=np.nanmax(freq)): raise ValueError(f'Cube covers {np.nanmin(freq):.3f}-{np.nanmax(freq):.3f} GHz')
    wc=WCS(hdr).celestial
    x0,y0=wc.world_to_pixel(SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg))
    ny,nx=cube.shape[1:]; yy,xx=np.indices((ny,nx)); rr=np.hypot(xx-x0,yy-y0)
    pixscale=np.mean(np.abs(wc.proj_plane_pixel_scales()))*3600
    rap=max(1.0,0.15/pixscale); src=rr<=rap; ann=(rr>=max(rap*3,4))&(rr<=max(rap*6,8))
    flux=np.nansum(cube[:,src],axis=1)
    annvals=cube[:,ann]; med=np.nanmedian(annvals,axis=1); mad=np.nanmedian(np.abs(annvals-med[:,None]),axis=1)
    unc=1.4826*mad*np.sqrt(np.count_nonzero(src))
    return freq,flux,unc,float(np.nanmedian(np.abs(np.diff(freq)))*1000)


def gaussian(x,a,mu,sig,c): return c+a*np.exp(-0.5*((x-mu)/sig)**2)


def analyse(freq,flux,unc,source,dw):
    o=np.argsort(freq); freq=freq[o]; flux=flux[o]; unc=unc[o]
    broad=np.abs(freq-TARGET_GHZ)<0.12; line=np.abs(freq-TARGET_GHZ)<0.045; base=broad&~line
    p=np.polyfit(freq[base]-TARGET_GHZ,flux[base],1,w=1/np.maximum(unc[base],np.nanmedian(unc[base])))
    cont=np.polyval(p,freq-TARGET_GHZ); f=flux-cont
    centroid=np.nansum(freq[line]*f[line])/np.nansum(f[line]); z=NU_REST_GHZ/centroid-1
    x=freq[line]; y=f[line]; e=np.where((unc[line]>0)&np.isfinite(unc[line]),unc[line],np.nanmedian(unc[line]))
    popt,_=curve_fit(gaussian,x,y,p0=[np.nanmax(y),centroid,.012,0],sigma=e,absolute_sigma=True,bounds=([0,x.min(),.001,-np.inf],[np.inf,x.max(),.08,np.inf]),maxfev=30000)
    gcent=popt[1]
    df=pd.DataFrame({'frequency_GHz':freq,'channel_spacing_MHz':np.r_[np.nan,np.diff(freq)*1000],'flux_native_units':flux,'continuum_native_units':cont,'line_flux_native_units':f,'uncertainty_native_units':unc,'in_centroid_window':line})
    csv=CSV_DIR/f'{VERSION}_ALMA_5MHZ_CHANNELS.csv'; df.to_csv(csv,index=False)
    shutil.copy2(csv,LOCAL_OUT/'CSV'/csv.name)
    fig,ax=plt.subplots(figsize=(16,7))
    ax.errorbar(freq[broad],f[broad],yerr=unc[broad],fmt='o',ms=4,lw=.8,capsize=2,label='Extracted channels')
    gx=np.linspace(x.min(),x.max(),1000); ax.plot(gx,gaussian(gx,*popt),lw=2,label='Gaussian cross-check')
    ax.axvline(centroid,ls='--',lw=1.8,label=f'Flux-weighted {centroid:.9f} GHz')
    ax.axvline(gcent,ls='-.',lw=1.6,label=f'Gaussian {gcent:.9f} GHz')
    ax.set_xlabel('Observed frequency [GHz]'); ax.set_ylabel('Continuum-subtracted flux [native units]')
    ax.set_title(f'JADES-GS-z11-0 ALMA [O III] 88 µm channel extraction\nspacing ≈ {dw:.6f} MHz | {Path(source).name}')
    ax.grid(color='#303944',lw=.6,alpha=.75); ax.legend(frameon=False,loc='upper left')
    png=PNG_DIR/f'{VERSION}_ALMA_CHANNEL_SPECTRUM.png'; fig.savefig(png,dpi=500,bbox_inches='tight'); plt.close(fig)
    shutil.copy2(png,LOCAL_OUT/'PNG'/png.name)
    return centroid,z,gcent,csv,png


def gallery(_=None):
    with out:
        clear_output(wait=True)
        imgs=sorted(PNG_DIR.glob('*.png'))
        if not imgs: print('No PNG outputs yet. Run the archive workflow first.'); return
        dd=widgets.Dropdown(options=[(p.name,str(p)) for p in imgs],description='Image:',layout=widgets.Layout(width='95%'))
        box=widgets.Output()
        def show(change=None):
            with box:
                clear_output(wait=True); display(Image(filename=dd.value))
        dd.observe(show,names='value'); display(dd,box); show()


def run(_=None):
    with out:
        clear_output(wait=True)
        print(f'CODE OUTPUT: {VERSION}')
        print('Google Drive persistent mode: large ALMA archive downloads are saved under MyDrive/JWST/ALMA.')
        drive.mount('/content/drive',force_remount=False)
        print(f'Drive folder: {DRIVE_ROOT}')
        rows=tap_query(); print(f'TAP rows returned: {len(rows)}')
        mous=choose_mous(rows); print(f'Candidate MOUS: {len(mous)}')
        archive_path=None
        for score,uid in mous[:4]:
            print(f'Checking DataLink: {uid} | score={score}')
            try: products=datalink(uid)
            except Exception as e:
                print('DataLink warning:',e); continue
            ranked=choose_archive(products)
            for pscore,size,url,desc in ranked[:12]:
                if not url or pscore<0: continue
                print(f'Archive candidate: score={pscore} | {fmt_bytes(size) if np.isfinite(size) else "unknown"} | {desc[:90]}')
                try:
                    archive_path=resumable_download(url)
                    break
                except Exception as e: print('Download warning:',e)
            if archive_path: break
        if archive_path is None:
            print('ERROR: no downloadable archive succeeded.'); return
        print(f'Archive saved: {archive_path}')
        members=list_archive(archive_path)
        print(f'FITS members found: {len(members)}')
        if not members:
            print('ERROR: archive contains no FITS products.'); return
        diagnostics=[]
        for score,size,name in members[:40]:
            print(f'Trying FITS member: score={score} | {fmt_bytes(size)} | {name}')
            try:
                fpath=extract_member(archive_path,name)
                freq,flux,unc,dw=extract_spectrum(fpath)
                centroid,z,gcent,csv,png=analyse(freq,flux,unc,str(fpath),dw)
                print(f'Flux-weighted centroid: {centroid:.9f} GHz')
                print(f'Flux-weighted redshift: {z:.9f}')
                print(f'Gaussian centroid: {gcent:.9f} GHz')
                print(f'Channel spacing: {dw:.6f} MHz')
                print(f'CSV: {csv}')
                print(f'PNG: {png}')
                print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
                print(f'# {VERSION}')
                return
            except Exception as e:
                diagnostics.append(f'{name}: {e}')
                try:
                    if 'fpath' in locals() and fpath.exists(): fpath.unlink()
                except: pass
        print('ERROR: no extracted FITS cube covered 279.901 GHz.')
        for d in diagnostics[-12:]: print(' -',d)
        print(f'# {VERSION}')

run_btn.on_click(run); gallery_btn.on_click(gallery)
display(widgets.HBox([run_btn,gallery_btn]),out)
