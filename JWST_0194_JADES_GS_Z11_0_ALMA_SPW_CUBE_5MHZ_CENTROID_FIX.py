# JWST_0194
import os, tarfile, shutil, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

VERSION='JWST_0194'
TARGET_GHZ=279.901
BIN_MHZ=5.0
BASE=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X362B_SCIENCE')
ARCHIVE=BASE/'ARCHIVE/2023.1.00336.S_uid___A001_X362b_Xae6_001_of_001.tar'
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
OUT_PNG.mkdir(parents=True,exist_ok=True); OUT_CSV.mkdir(parents=True,exist_ok=True)
WORK=Path('/content/JWST_0194_WORK'); WORK.mkdir(parents=True,exist_ok=True)


def spectral_axis_from_header(hdr, ndim):
    for i in range(1, int(hdr.get('NAXIS', ndim))+1):
        ctype=str(hdr.get(f'CTYPE{i}','')).upper()
        if any(k in ctype for k in ['FREQ','VRAD','VOPT','VELO']):
            n=int(hdr.get(f'NAXIS{i}',0))
            pix=np.arange(n,dtype=float)+1.0
            world=float(hdr.get(f'CRVAL{i}',0.0))+(pix-float(hdr.get(f'CRPIX{i}',1.0)))*float(hdr.get(f'CDELT{i}',1.0))
            unit=str(hdr.get(f'CUNIT{i}','Hz')).lower()
            if 'ghz' in unit: freq_ghz=world
            elif 'mhz' in unit: freq_ghz=world/1e3
            elif 'khz' in unit: freq_ghz=world/1e6
            else: freq_ghz=world/1e9
            np_axis=ndim-i
            return i,np_axis,freq_ghz,ctype,unit
    raise RuntimeError('No spectral FITS axis found in header.')


def extract_member(tf, member_name):
    dest=WORK/Path(member_name).name
    if dest.exists() and dest.stat().st_size>0:
        return dest
    src=tf.extractfile(member_name)
    if src is None: raise RuntimeError(f'Cannot extract {member_name}')
    with src, open(dest,'wb') as out:
        shutil.copyfileobj(src,out,length=16*1024*1024)
    return dest


def candidate_members(tf):
    names=[m.name for m in tf.getmembers() if m.isfile()]
    good=[]
    for n in names:
        low=n.lower()
        if ('jades-gs-z11-0_sci.spw' in low and '.cube.i.pbcor.fits' in low
                and 'mask' not in low and 'mfs' not in low and 'repbw' not in low):
            good.append(n)
    return sorted(good)


def collapse_to_cube(data, np_spec_axis):
    arr=np.asarray(data,dtype=np.float32)
    while arr.ndim>3:
        singleton=[i for i,s in enumerate(arr.shape) if s==1]
        if not singleton: break
        ax=singleton[0]
        arr=np.take(arr,0,axis=ax)
        if ax<np_spec_axis: np_spec_axis-=1
    if arr.ndim!=3:
        raise RuntimeError(f'Expected 3-D cube after singleton removal; got {arr.shape}')
    arr=np.moveaxis(arr,np_spec_axis,0)
    return arr


def rebin_5mhz(freq, flux):
    order=np.argsort(freq); f=freq[order]; y=flux[order]
    step=BIN_MHZ/1000.0
    lo=np.floor(f.min()/step)*step; hi=np.ceil(f.max()/step)*step
    edges=np.arange(lo,hi+step*1.01,step)
    idx=np.digitize(f,edges)-1
    rows=[]
    for k in range(len(edges)-1):
        q=(idx==k)&np.isfinite(y)
        if q.any(): rows.append((0.5*(edges[k]+edges[k+1]),np.nanmean(y[q]),int(q.sum())))
    return pd.DataFrame(rows,columns=['frequency_GHz','flux_Jy_per_beam','native_channels'])


def main():
    print(f'CODE OUTPUT: {VERSION}')
    print('Robust FITS-axis mode: uses the existing X362b archive; no archive download.')
    if not ARCHIVE.exists(): raise FileNotFoundError(f'Archive not found: {ARCHIVE}')

    chosen=None; diagnostics=[]
    with tarfile.open(ARCHIVE,'r:*') as tf:
        members=candidate_members(tf)
        print(f'Candidate science pbcor cubes: {len(members)}')
        for n in members:
            p=extract_member(tf,n)
            with fits.open(p,memmap=True) as hdul:
                hdu=next(h for h in hdul if h.data is not None)
                hdr=hdu.header; ndim=hdu.data.ndim
                fits_ax,np_ax,freq,ctype,unit=spectral_axis_from_header(hdr,ndim)
                row={'member':n,'shape':str(hdu.data.shape),'fits_spectral_axis':fits_ax,
                     'numpy_spectral_axis':np_ax,'ctype':ctype,'cunit':unit,
                     'freq_min_GHz':float(np.nanmin(freq)),'freq_max_GHz':float(np.nanmax(freq)),
                     'channel_MHz':float(np.nanmedian(np.abs(np.diff(freq)))*1000.0)}
                diagnostics.append(row)
                print(f"{Path(n).name} | shape={hdu.data.shape} | {row['freq_min_GHz']:.6f}..{row['freq_max_GHz']:.6f} GHz | {row['channel_MHz']:.6f} MHz")
                if row['freq_min_GHz']<=TARGET_GHZ<=row['freq_max_GHz'] or row['freq_max_GHz']<=TARGET_GHZ<=row['freq_min_GHz']:
                    chosen=(n,p,hdr,np_ax,freq)
                    break

    diag=pd.DataFrame(diagnostics)
    diag_csv=OUT_CSV/f'{VERSION}_CUBE_FREQUENCY_DIAGNOSTIC.csv'; diag.to_csv(diag_csv,index=False)
    if chosen is None:
        raise RuntimeError(f'No candidate cube covers {TARGET_GHZ:.6f} GHz. Diagnostic CSV: {diag_csv}')

    name,path,hdr,np_ax,freq=chosen
    with fits.open(path,memmap=True) as hdul:
        hdu=next(h for h in hdul if h.data is not None)
        cube=collapse_to_cube(hdu.data,np_ax)
    print(f'Accepted cube: {Path(name).name}')
    print(f'Cube reordered to [frequency,y,x]: {cube.shape}')

    target_i=int(np.nanargmin(np.abs(freq-TARGET_GHZ)))
    df_native=float(np.nanmedian(np.abs(np.diff(freq))))
    half=max(2,int(round(0.060/df_native)))
    lo=max(0,target_i-half); hi=min(cube.shape[0],target_i+half+1)
    line_map=np.nanmean(cube[lo:hi],axis=0)
    ny,nx=line_map.shape
    border=np.ones_like(line_map,dtype=bool)
    border[ny//4:3*ny//4,nx//4:3*nx//4]=False
    bg=np.nanmedian(line_map[border])
    search=line_map-bg
    sy=slice(ny//4,3*ny//4); sx=slice(nx//4,3*nx//4)
    yy,xx=np.unravel_index(np.nanargmax(search[sy,sx]),search[sy,sx].shape)
    y0=yy+ny//4; x0=xx+nx//4
    r=2
    aperture=cube[:,max(0,y0-r):min(ny,y0+r+1),max(0,x0-r):min(nx,x0+r+1)]
    spec=np.nanmean(aperture,axis=(1,2))

    far=np.abs(freq-TARGET_GHZ)>0.12
    continuum=float(np.nanmedian(spec[far])) if far.any() else float(np.nanmedian(spec))
    spec_cs=spec-continuum
    binned=rebin_5mhz(freq,spec_cs)
    win=np.abs(binned['frequency_GHz']-TARGET_GHZ)<=0.08
    local=binned.loc[win].copy()
    noise=float(np.nanstd(binned.loc[np.abs(binned['frequency_GHz']-TARGET_GHZ)>0.12,'flux_Jy_per_beam']))
    weights=np.clip(local['flux_Jy_per_beam'].to_numpy()-max(0.0,noise),0,None)
    if not np.isfinite(weights).any() or weights.sum()<=0:
        weights=np.clip(local['flux_Jy_per_beam'].to_numpy(),0,None)
    centroid=float(np.sum(local['frequency_GHz'].to_numpy()*weights)/np.sum(weights))
    peak=float(local.loc[local['flux_Jy_per_beam'].idxmax(),'frequency_GHz'])

    binned['target_window']=np.abs(binned['frequency_GHz']-TARGET_GHZ)<=0.08
    binned_csv=OUT_CSV/f'{VERSION}_5MHZ_SPECTRUM.csv'; binned.to_csv(binned_csv,index=False)

    fig,ax=plt.subplots(figsize=(12,6.5))
    ax.plot(binned['frequency_GHz'],binned['flux_Jy_per_beam'],lw=1.2,label='Observed ALMA spectrum, 5 MHz bins')
    ax.axvline(TARGET_GHZ,ls='--',lw=1.1,label=f'Reference 279.901 GHz')
    ax.axvline(centroid,ls=':',lw=1.5,label=f'Flux-weighted centroid {centroid:.6f} GHz')
    ax.set_xlim(TARGET_GHZ-0.15,TARGET_GHZ+0.15)
    ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Continuum-subtracted mean aperture flux (Jy beam$^{-1}$)')
    ax.set_title('JADES-GS-z11-0 — ALMA [O III] spectral extraction')
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    png=OUT_PNG/f'{VERSION}_ALMA_5MHZ_CENTROID.png'; fig.savefig(png,dpi=220); plt.show(); plt.close(fig)

    print('')
    print(f'Observed/reference frequency       {TARGET_GHZ:.6f} GHz')
    print(f'5 MHz-binned peak                 {peak:.6f} GHz')
    print(f'Flux-weighted centroid            {centroid:.6f} GHz')
    print(f'Centroid minus 279.901 GHz        {(centroid-TARGET_GHZ)*1000.0:+.3f} MHz')
    print(f'Extraction pixel (x,y)            ({x0},{y0})')
    print(f'Estimated off-line RMS            {noise:.6e} Jy/beam')
    print(f'PNG: {png}')
    print(f'CSV: {binned_csv}')
    print(f'Diagnostic CSV: {diag_csv}')
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')

main()
