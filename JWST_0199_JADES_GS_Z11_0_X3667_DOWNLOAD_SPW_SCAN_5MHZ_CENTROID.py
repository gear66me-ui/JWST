# JWST_0199
import os, sys, io, tarfile, shutil, subprocess, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

for pkg in ['requests','numpy','pandas','astropy','matplotlib']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

VERSION='JWST_0199'
TARGET_GHZ=279.901000
BIN_MHZ=5.0
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S')
META=ROOT/'METADATA'/'JWST_0192_ALMA_PRODUCT_INVENTORY.csv'
ARCHIVE_DIR=ROOT/'X3667_SCIENCE'/'ARCHIVE'
EXTRACT_DIR=ROOT/'X3667_SCIENCE'/'SELECTED_CUBE'
OUT_CSV=ROOT/'X3667_SCIENCE'/'CSV'
OUT_PNG=ROOT/'X3667_SCIENCE'/'PNG'
ARCHIVE_NAME='2023.1.00336.S_uid___A001_X3667_Xe0_001_of_001.tar'
ARCHIVE=ARCHIVE_DIR/ARCHIVE_NAME

for p in [ARCHIVE_DIR,EXTRACT_DIR,OUT_CSV,OUT_PNG]: p.mkdir(parents=True,exist_ok=True)


def human(n):
    n=float(n)
    for u in ['B','KB','MB','GB','TB']:
        if n<1024 or u=='TB': return f'{n:.2f} {u}'
        n/=1024


def archive_url():
    if not META.exists(): raise FileNotFoundError(f'Missing inventory CSV: {META}')
    df=pd.read_csv(META)
    m=df[df.get('filename','').astype(str).eq(ARCHIVE_NAME)]
    if m.empty: raise RuntimeError('X3667 science archive URL not found in JWST_0192 inventory.')
    for _,r in m.iterrows():
        u=str(r.get('access_url',''))
        if u.startswith('http'): return u, int(float(r.get('size_bytes',0) or 0))
    raise RuntimeError('Inventory row found but access_url is missing.')


def download_resume(url, expected=0):
    have=ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
    if expected and have==expected:
        print(f'Archive already complete: {ARCHIVE} | {human(have)}')
        return
    headers={'Range':f'bytes={have}-'} if have else {}
    mode='ab' if have else 'wb'
    with requests.get(url,headers=headers,stream=True,timeout=(60,600)) as r:
        if have and r.status_code==200:
            have=0; mode='wb'
        r.raise_for_status()
        total=expected or (have+int(r.headers.get('content-length',0)))
        with open(ARCHIVE,mode) as f:
            done=have
            for chunk in r.iter_content(8*1024*1024):
                if not chunk: continue
                f.write(chunk); done+=len(chunk)
                if total:
                    print(f'\rDownloading to Drive {100*done/total:6.2f}% ({human(done)}/{human(total)})',end='',flush=True)
    print()
    final=ARCHIVE.stat().st_size
    if expected and final!=expected: raise RuntimeError(f'Download size mismatch: {final} != {expected}')
    print(f'Archive saved: {ARCHIVE} | {human(final)}')


def spectral_axis(header):
    naxis=int(header.get('NAXIS',0))
    for i in range(1,naxis+1):
        c=str(header.get(f'CTYPE{i}','')).upper()
        if any(k in c for k in ['FREQ','VRAD','VELO','VOPT']): return i
    return None


def frequency_grid(header, axis):
    n=int(header[f'NAXIS{axis}'])
    pix=np.arange(1,n+1,dtype=float)
    crpix=float(header.get(f'CRPIX{axis}',1.0))
    crval=float(header.get(f'CRVAL{axis}',0.0))
    cdelt=float(header.get(f'CDELT{axis}',header.get(f'CD{axis}_{axis}',1.0)))
    vals=crval+(pix-crpix)*cdelt
    unit=str(header.get(f'CUNIT{axis}','Hz')).lower()
    ctype=str(header.get(f'CTYPE{axis}','')).upper()
    if 'FREQ' not in ctype: return None
    if unit=='hz': vals/=1e9
    elif unit=='khz': vals/=1e6
    elif unit=='mhz': vals/=1e3
    elif unit=='ghz': pass
    else: vals/=1e9
    return vals


def read_header_from_member(tf, member):
    fh=tf.extractfile(member)
    if fh is None: return None
    data=fh.read(min(member.size,4*1024*1024))
    return fits.Header.fromstring(data.decode('ascii','ignore'),sep='')


def scan_cubes():
    records=[]; selected=None
    with tarfile.open(ARCHIVE,'r:*') as tf:
        members=[m for m in tf.getmembers() if m.isfile()]
        cubes=[m for m in members if 'JADES-GS-z11-0_sci' in m.name and '.cube.I.pbcor.fits' in m.name and 'mask' not in m.name]
        print(f'JADES pbcor cube candidates: {len(cubes)}')
        for m in cubes:
            try:
                h=read_header_from_member(tf,m)
                ax=spectral_axis(h) if h else None
                freq=frequency_grid(h,ax) if ax else None
                if freq is None: raise RuntimeError('No frequency WCS axis')
                lo,hi=float(np.nanmin(freq)),float(np.nanmax(freq))
                covers=lo<=TARGET_GHZ<=hi
                spacing=float(np.nanmedian(np.abs(np.diff(freq)))*1000)
                rec={'member':m.name,'size_bytes':m.size,'size_MB':m.size/1024**2,'spectral_axis':ax,
                     'channels':len(freq),'frequency_min_GHz':lo,'frequency_max_GHz':hi,
                     'native_spacing_MHz':spacing,'covers_279_901_GHz':covers}
                records.append(rec)
                print(f'  {Path(m.name).name} | {lo:.6f} to {hi:.6f} GHz | {spacing:.6f} MHz | covers={covers}')
                if covers and (selected is None or m.size>selected.size): selected=m
            except Exception as e:
                records.append({'member':m.name,'size_bytes':m.size,'error':str(e),'covers_279_901_GHz':False})
    df=pd.DataFrame(records)
    inv=OUT_CSV/f'{VERSION}_X3667_CUBE_WCS_INVENTORY.csv'; df.to_csv(inv,index=False)
    print(f'Cube WCS inventory: {inv}')
    if selected is None: raise RuntimeError('No X3667 JADES pbcor cube covers 279.901 GHz.')
    return selected


def extract_selected(member):
    dst=EXTRACT_DIR/Path(member.name).name
    if dst.exists() and dst.stat().st_size==member.size:
        print(f'Selected cube already extracted: {dst} | {human(dst.stat().st_size)}')
        return dst
    with tarfile.open(ARCHIVE,'r:*') as tf:
        src=tf.extractfile(member)
        if src is None: raise RuntimeError('Could not open selected cube member.')
        tmp=dst.with_suffix(dst.suffix+'.part')
        with open(tmp,'wb') as out: shutil.copyfileobj(src,out,8*1024*1024)
        if tmp.stat().st_size!=member.size: raise RuntimeError('Extracted cube size verification failed.')
        tmp.replace(dst)
    print(f'Selected cube extracted: {dst} | {human(dst.stat().st_size)}')
    return dst


def cube_and_freq(path):
    with fits.open(path,memmap=True) as hdul:
        h=hdul[0].header.copy(); data=np.asarray(hdul[0].data,dtype=np.float32)
    ax=spectral_axis(h); freq=frequency_grid(h,ax)
    np_axis=data.ndim-ax
    data=np.moveaxis(data,np_axis,0)
    while data.ndim>3:
        data=data[:,0]
    if data.ndim!=3: raise RuntimeError(f'Unexpected cube shape after axis normalization: {data.shape}')
    if freq[0]>freq[-1]: freq=freq[::-1]; data=data[::-1]
    return data,freq,h


def extract_spectrum(cube,freq):
    near=np.abs(freq-TARGET_GHZ)<=0.20
    if near.sum()<5: near=np.abs(freq-TARGET_GHZ)<=0.50
    moment=np.nanmax(cube[near],axis=0)-np.nanmedian(cube[near],axis=0)
    y,x=np.unravel_index(np.nanargmax(moment),moment.shape)
    yy,xx=np.ogrid[:cube.shape[1],:cube.shape[2]]
    aperture=(xx-x)**2+(yy-y)**2<=2.0**2
    spec=np.nanmean(cube[:,aperture],axis=1)
    return spec,y,x


def rebin_5mhz(freq,spec):
    width=BIN_MHZ/1000.0
    start=np.floor(freq.min()/width)*width
    edges=np.arange(start,freq.max()+2*width,width)
    idx=np.digitize(freq,edges)-1
    rows=[]
    for b in range(len(edges)-1):
        m=idx==b
        if not np.any(m): continue
        rows.append((np.nanmean(freq[m]),np.nanmean(spec[m]),int(m.sum())))
    return pd.DataFrame(rows,columns=['frequency_GHz','flux_mean','native_channels'])


def centroid_table(df):
    f=df.frequency_GHz.to_numpy(); s=df.flux_mean.to_numpy()
    line=np.abs(f-TARGET_GHZ)<=0.20
    side=(np.abs(f-TARGET_GHZ)>=0.25)&(np.abs(f-TARGET_GHZ)<=0.70)
    baseline=np.nanmedian(s[side]) if np.any(side) else np.nanmedian(s[~line])
    w=np.clip(s-baseline,0,None)
    use=line & np.isfinite(w)
    centroid=float(np.sum(f[use]*w[use])/np.sum(w[use])) if np.sum(w[use])>0 else float(f[line][np.nanargmax(s[line])])
    peak=float(f[line][np.nanargmax(s[line])])
    return centroid,peak,baseline


def save_plot(df,centroid,peak):
    p=OUT_PNG/f'{VERSION}_JADES_GS_Z11_0_5MHZ_SPECTRUM.png'
    fig,ax=plt.subplots(figsize=(12,6))
    ax.plot(df.frequency_GHz,df.flux_mean,lw=1.1)
    ax.axvline(TARGET_GHZ,ls='--',lw=1.0,label=f'Target {TARGET_GHZ:.6f} GHz')
    ax.axvline(centroid,ls=':',lw=1.2,label=f'Flux centroid {centroid:.6f} GHz')
    ax.axvline(peak,ls='-.',lw=1.0,label=f'Peak bin {peak:.6f} GHz')
    ax.set_xlim(TARGET_GHZ-0.5,TARGET_GHZ+0.5)
    ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Mean aperture flux (native cube units)')
    ax.set_title('JADES-GS-z11-0 ALMA spectrum rebinned to 5 MHz')
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(p,dpi=180); plt.close(fig)
    return p


def main():
    print(f'CODE OUTPUT: {VERSION}')
    print('Drive mount reused; no remount requested.')
    url,expected=archive_url()
    print(f'X3667 science archive expected size: {human(expected)}')
    download_resume(url,expected)
    member=scan_cubes()
    cube_path=extract_selected(member)
    cube,freq,h=cube_and_freq(cube_path)
    print(f'Accepted cube: {Path(member.name).name}')
    print(f'Cube shape [channel,y,x]: {cube.shape}')
    print(f'Frequency coverage: {freq.min():.6f} to {freq.max():.6f} GHz')
    print(f'Native channel spacing: {np.nanmedian(np.diff(freq))*1000:.6f} MHz')
    spec,y,x=extract_spectrum(cube,freq)
    df=rebin_5mhz(freq,spec)
    centroid,peak,baseline=centroid_table(df)
    df['target_GHz']=TARGET_GHZ; df['centroid_GHz']=centroid
    csv=OUT_CSV/f'{VERSION}_JADES_GS_Z11_0_5MHZ_SPECTRUM.csv'; df.to_csv(csv,index=False)
    summary=pd.DataFrame([{'target_frequency_GHz':TARGET_GHZ,'flux_weighted_centroid_GHz':centroid,
                           'peak_5MHz_bin_GHz':peak,'centroid_minus_target_MHz':(centroid-TARGET_GHZ)*1000,
                           'aperture_center_y':y,'aperture_center_x':x,'baseline':baseline,
                           'source_status':'observed ALMA pipeline cube; derived 5 MHz centroid'}])
    scsv=OUT_CSV/f'{VERSION}_CENTROID_SUMMARY.csv'; summary.to_csv(scsv,index=False)
    png=save_plot(df,centroid,peak)
    print(summary.to_string(index=False))
    print(f'Spectrum CSV: {csv}')
    print(f'Summary CSV: {scsv}')
    print(f'Plot PNG: {png}')
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')

main()
