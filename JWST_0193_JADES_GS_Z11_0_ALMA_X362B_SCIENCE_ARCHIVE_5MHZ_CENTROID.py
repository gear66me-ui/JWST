# JWST_0193
import io, os, re, sys, shutil, tarfile, subprocess, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

for pkg in ['requests','astropy','numpy','pandas','matplotlib','scipy']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import votable, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.optimize import curve_fit
from google.colab import drive

VERSION='JWST_0193'
PROJECT='2023.1.00336.S'
UID='uid://A001/X362b/Xae6'
TARGET_GHZ=279.901
NU_REST_GHZ=3393.006244
RA_DEG=53.1647632
DEC_DEG=-27.7746223
ESO_DATALINK='https://almascience.eso.org/datalink/sync'
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X362B_SCIENCE')
ARCHIVE_DIR=ROOT/'ARCHIVE'
EXTRACT_DIR=ROOT/'EXTRACTED'
PNG_DIR=ROOT/'PNG'
CSV_DIR=ROOT/'CSV'
for d in [ARCHIVE_DIR,EXTRACT_DIR,PNG_DIR,CSV_DIR]: d.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})

def parse_votable(content):
    return votable.parse_single_table(io.BytesIO(content)).to_table().to_pandas()

def sval(row,*names):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            v=row[n]
            if isinstance(v,bytes): v=v.decode(errors='ignore')
            return str(v)
    return ''

def fmt_bytes(n):
    n=float(n)
    for unit in ['B','KB','MB','GB','TB']:
        if n<1024: return f'{n:.1f} {unit}'
        n/=1024
    return f'{n:.1f} PB'

def get_science_product():
    r=requests.get(ESO_DATALINK,params={'ID':UID},timeout=180)
    r.raise_for_status()
    df=parse_votable(r.content)
    candidates=[]
    for _,row in df.iterrows():
        url=sval(row,'access_url','accessURL')
        name=url.split('?')[0].rstrip('/').split('/')[-1]
        desc=sval(row,'description')
        size_s=sval(row,'content_length','contentLength')
        try: size=float(size_s)
        except: size=np.nan
        text=(name+' '+desc).lower()
        if UID.split('/')[-1].lower() not in text and 'x362b' not in text: continue
        if 'auxiliary' in text or '.asdm.sdm.tar' in text or 'readme' in text: continue
        if name.endswith('_001_of_001.tar'):
            candidates.append((size,url,name,desc))
    if not candidates: raise RuntimeError('No X362b science archive candidate found.')
    candidates.sort(key=lambda x:x[0] if np.isfinite(x[0]) else 1e99)
    return candidates[0]

def resumable_download(url,name,expected_size):
    path=ARCHIVE_DIR/name
    existing=path.stat().st_size if path.exists() else 0
    if np.isfinite(expected_size) and existing==int(expected_size):
        print(f'Archive already complete: {path}')
        return path
    headers={}
    mode='wb'
    if existing>0:
        headers['Range']=f'bytes={existing}-'
        mode='ab'
        print(f'Resuming at {fmt_bytes(existing)}')
    with requests.get(url,headers=headers,stream=True,timeout=300) as r:
        r.raise_for_status()
        if headers and r.status_code!=206:
            print('Server did not honor resume; restarting.')
            existing=0; mode='wb'
        total=int(expected_size) if np.isfinite(expected_size) else existing+int(r.headers.get('content-length',0))
        done=existing
        with open(path,mode) as f:
            for chunk in r.iter_content(8*1024*1024):
                if not chunk: continue
                f.write(chunk); done+=len(chunk)
                if total:
                    print(f'\rDownloading to Drive {100*done/total:6.2f}% ({fmt_bytes(done)}/{fmt_bytes(total)})',end='')
        print()
    if np.isfinite(expected_size) and path.stat().st_size!=int(expected_size):
        raise IOError(f'Size mismatch: got {path.stat().st_size}, expected {int(expected_size)}')
    return path

def rank_members(path):
    ranked=[]
    with tarfile.open(path) as t:
        for m in t.getmembers():
            n=m.name.lower()
            if not n.endswith(('.fits','.fits.gz','.fit','.fz')): continue
            score=0
            if 'jades-gs-z11-0' in n: score+=120
            if any(k in n for k in ['spw22','spw.22']): score+=100
            if any(k in n for k in ['cube','image']): score+=80
            if any(k in n for k in ['pbcor','contsub','science']): score+=40
            if any(k in n for k in ['moment','mom0','continuum','pb.fits','psf']): score-=100
            ranked.append((score,m.size,m.name))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    return ranked

def extract_member(path,name):
    out=EXTRACT_DIR/Path(name).name
    if out.exists() and out.stat().st_size>0: return out
    with tarfile.open(path) as t:
        src=t.extractfile(name)
        if src is None: raise IOError('Could not extract member')
        with src, open(out,'wb') as dst: shutil.copyfileobj(src,dst,length=8*1024*1024)
    return out

def spectral_axis(h,n):
    spec=None
    for i in range(1,int(h.get('NAXIS',0))+1):
        if any(k in str(h.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec=i; break
    if spec is None: raise ValueError('No spectral axis')
    pix=np.arange(n,dtype=float)
    vals=float(h[f'CRVAL{spec}'])+(pix+1-float(h[f'CRPIX{spec}']))*float(h[f'CDELT{spec}'])
    ctype=str(h[f'CTYPE{spec}']).upper(); cunit=str(h.get(f'CUNIT{spec}','Hz')).lower()
    if 'FREQ' in ctype:
        if 'ghz' in cunit: return vals,spec
        if 'mhz' in cunit: return vals*1e-3,spec
        return vals*1e-9,spec
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
    if spec_fits is None: raise ValueError('No spectral FITS axis')
    spec_np=naxis-spec_fits
    while spec_np>=arr.ndim: spec_np-=1
    cube=np.moveaxis(arr,spec_np,0)
    freq,_=spectral_axis(hdr,cube.shape[0])
    fmin,fmax=np.nanmin(freq),np.nanmax(freq)
    if not (fmin<=TARGET_GHZ<=fmax): raise ValueError(f'Cube covers {fmin:.6f}-{fmax:.6f} GHz')
    wc=WCS(hdr).celestial
    x0,y0=wc.world_to_pixel(SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg))
    ny,nx=cube.shape[1:]
    yy,xx=np.indices((ny,nx)); rr=np.hypot(xx-x0,yy-y0)
    pixscale=np.mean(np.abs(wc.proj_plane_pixel_scales()))*3600
    rap=max(1.0,0.15/pixscale)
    src=rr<=rap
    ann=(rr>=max(rap*3,4))&(rr<=max(rap*6,8))
    flux=np.nansum(cube[:,src],axis=1)
    annvals=cube[:,ann]
    med=np.nanmedian(annvals,axis=1)
    mad=np.nanmedian(np.abs(annvals-med[:,None]),axis=1)
    unc=1.4826*mad*np.sqrt(np.count_nonzero(src))
    dw=float(np.nanmedian(np.abs(np.diff(freq)))*1000)
    return freq,flux,unc,dw

def gaussian(x,a,mu,sig,c): return c+a*np.exp(-0.5*((x-mu)/sig)**2)

def analyse(freq,flux,unc,source,dw):
    o=np.argsort(freq); freq=freq[o]; flux=flux[o]; unc=unc[o]
    broad=np.abs(freq-TARGET_GHZ)<0.12
    line=np.abs(freq-TARGET_GHZ)<0.045
    base=broad&~line
    if base.sum()<3 or line.sum()<3: raise ValueError('Insufficient channels around line')
    w=1/np.maximum(unc[base],np.nanmedian(unc[base]))
    p=np.polyfit(freq[base]-TARGET_GHZ,flux[base],1,w=w)
    cont=np.polyval(p,freq-TARGET_GHZ)
    f=flux-cont
    centroid=np.nansum(freq[line]*f[line])/np.nansum(f[line])
    z=NU_REST_GHZ/centroid-1
    x=freq[line]; y=f[line]
    e=np.where((unc[line]>0)&np.isfinite(unc[line]),unc[line],np.nanmedian(unc[line]))
    popt,_=curve_fit(gaussian,x,y,p0=[np.nanmax(y),centroid,.012,0],sigma=e,absolute_sigma=True,bounds=([0,x.min(),.001,-np.inf],[np.inf,x.max(),.08,np.inf]),maxfev=30000)
    gcent=float(popt[1])
    df=pd.DataFrame({'frequency_GHz':freq,'channel_spacing_MHz':np.r_[np.nan,np.diff(freq)*1000],'flux_native_units':flux,'continuum_native_units':cont,'line_flux_native_units':f,'uncertainty_native_units':unc,'in_centroid_window':line})
    csv=CSV_DIR/f'{VERSION}_ALMA_5MHZ_CHANNELS.csv'; df.to_csv(csv,index=False)
    fig,ax=plt.subplots(figsize=(16,7))
    ax.errorbar(freq[broad],f[broad],yerr=unc[broad],fmt='o',ms=4,lw=.8,capsize=2,label='Extracted 5 MHz channels')
    gx=np.linspace(x.min(),x.max(),1000); ax.plot(gx,gaussian(gx,*popt),lw=2,label='Gaussian cross-check')
    ax.axvline(centroid,ls='--',lw=1.8,label=f'Flux centroid {centroid:.9f} GHz')
    ax.axvline(gcent,ls='-.',lw=1.5,label=f'Gaussian {gcent:.9f} GHz')
    ax.set_xlabel('Observed frequency [GHz]'); ax.set_ylabel('Continuum-subtracted flux [native units]')
    ax.set_title(f'JADES-GS-z11-0 ALMA [O III] 88 µm\nchannel spacing ≈ {dw:.6f} MHz | {Path(source).name}')
    ax.grid(color='#303944',lw=.6,alpha=.75); ax.legend(frameon=False)
    png=PNG_DIR/f'{VERSION}_ALMA_CHANNEL_SPECTRUM.png'; fig.savefig(png,dpi=500,bbox_inches='tight'); plt.close(fig)
    return centroid,z,gcent,csv,png

def main():
    print(f'CODE OUTPUT: {VERSION}')
    print('Automatic X362b science-archive mode: downloads only the 2.365 GB _001_of_001.tar product to Google Drive.')
    drive.mount('/content/drive',force_remount=False)
    size,url,name,desc=get_science_product()
    print(f'Selected archive: {name} | {fmt_bytes(size)}')
    archive=resumable_download(url,name,size)
    print(f'Archive saved: {archive}')
    members=rank_members(archive)
    print(f'FITS members found: {len(members)}')
    diagnostics=[]
    for score,msize,mname in members[:80]:
        print(f'Trying: score={score:4d} | {fmt_bytes(msize)} | {mname}')
        try:
            fpath=extract_member(archive,mname)
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
            diagnostics.append(f'{mname}: {e}')
            try:
                if 'fpath' in locals() and fpath.exists(): fpath.unlink()
            except: pass
    print('ERROR: no FITS cube covering 279.901 GHz was found in the X362b science archive.')
    for d in diagnostics[-20:]: print(' -',d)
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')

main()
