# JWST_0184
import sys, subprocess, warnings, tarfile, zipfile, shutil, re, os
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

for pkg in ['astroquery','astropy','photutils','scipy','requests']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.alma import Alma
from IPython.display import display, clear_output, Image
import ipywidgets as widgets
import requests

VERSION='JWST_0184'
PROJECT='2023.1.00336.S'
RA_DEG=53.1647632
DEC_DEG=-27.7746223
NU_REST_GHZ=3393.006244
TARGET_GHZ=279.901
OUT=Path('/content/JWST_OUTPUT')
PNG=OUT/'PNG'; CSV=OUT/'CSV'; DATA=OUT/'ALMA_0184'
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})

status=widgets.Output()
query_btn=widgets.Button(description='1. Query ALMA archive',button_style='info',layout=widgets.Layout(width='240px'))
product_dd=widgets.Dropdown(description='Product:',layout=widgets.Layout(width='95%'))
run_btn=widgets.Button(description='2. Download + extract 5 MHz spectrum',button_style='success',layout=widgets.Layout(width='340px'),disabled=True)
products=[]

def fmt_bytes(x):
    try:
        x=float(x)
        for unit in ['B','KB','MB','GB','TB']:
            if x<1024: return f'{x:.1f} {unit}'
            x/=1024
    except Exception: return 'unknown'

def safe_name(url):
    name=url.split('?')[0].rstrip('/').split('/')[-1]
    return name or 'alma_product'

def query_archive(_=None):
    global products
    with status:
        clear_output(wait=True)
        print('REQUEST SUMMARY:')
        print('• Query the public ALMA archive for project 2023.1.00336.S.')
        print('• Locate reduced spectral products near 279.901 GHz with ~5 MHz channels.')
        print('• Extract channel centers, fluxes, empirical uncertainties, and reproduce the flux-weighted centroid.')
        print('\nQuerying ALMA...')
        Alma.cache_location=str(DATA/'astroquery_cache')
        result=Alma.query(payload={'project_code':PROJECT,'public_data':True})
        if result is None or len(result)==0:
            print('No archive rows returned. Try again later; the ALMA TAP service may be unavailable.')
            return
        uids=[]
        for col in ['member_ous_uid','group_ous_uid','obs_id']:
            if col in result.colnames:
                uids.extend([str(x) for x in result[col] if str(x) not in ('','--','None')])
        uids=list(dict.fromkeys(uids))
        print(f'Archive rows: {len(result)} | candidate UIDs: {len(uids)}')
        rows=[]
        for uid in uids:
            try:
                info=Alma.get_data_info(uid)
                if info is None: continue
                for r in info:
                    names=info.colnames
                    url=str(r['access_url']) if 'access_url' in names else ''
                    fname=str(r['file_name']) if 'file_name' in names else safe_name(url)
                    size=r['content_length'] if 'content_length' in names else np.nan
                    mime=str(r['content_type']) if 'content_type' in names else ''
                    text=(fname+' '+url+' '+mime).lower()
                    score=0
                    if 'fits' in text: score+=8
                    if 'cube' in text or 'image' in text: score+=5
                    if 'science' in text: score+=3
                    if 'auxiliary' in text or 'weblog' in text: score-=8
                    rows.append({'uid':uid,'file_name':fname,'url':url,'size':size,'score':score})
            except Exception as e:
                print(f'UID metadata warning: {uid}: {e}')
        rows=[r for r in rows if r['url']]
        rows.sort(key=lambda r:(-r['score'], float(r['size']) if str(r['size']).replace('.','',1).isdigit() else 1e99))
        products=rows[:150]
        if not products:
            print('No downloadable product URLs were exposed by the archive query.')
            return
        opts=[]
        for i,r in enumerate(products):
            label=f"{i:03d} | score {r['score']:2d} | {fmt_bytes(r['size'])} | {r['file_name'][:90]}"
            opts.append((label,i))
        product_dd.options=opts
        run_btn.disabled=False
        print('Select the most likely reduced FITS cube. Highest-scoring products are listed first.')
        display(product_dd,run_btn)

def download(url):
    path=DATA/safe_name(url)
    if path.exists() and path.stat().st_size>0: return path
    with requests.get(url,stream=True,timeout=120) as r:
        r.raise_for_status()
        total=int(r.headers.get('content-length',0)); done=0
        with open(path,'wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk); done+=len(chunk)
                    if total: print(f'\rDownloading {100*done/total:5.1f}% ({fmt_bytes(done)}/{fmt_bytes(total)})',end='')
    print()
    return path

def unpack(path):
    root=DATA/'unpacked'; root.mkdir(exist_ok=True)
    try:
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as t: t.extractall(root)
        elif zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z: z.extractall(root)
        elif path.suffix.lower() in ['.fits','.fit','.fz']:
            return [path]
    except Exception as e: print('Unpack warning:',e)
    return list(root.rglob('*.fits'))+list(root.rglob('*.fit'))+list(root.rglob('*.fits.gz'))

def spectral_axis_ghz(header,nchan):
    w=WCS(header)
    spec=-1
    for i in range(1,header.get('NAXIS',0)+1):
        c=str(header.get(f'CTYPE{i}','')).upper()
        if any(k in c for k in ['FREQ','VRAD','VELO']): spec=i; break
    if spec<0: raise ValueError('No spectral axis found')
    pix=np.arange(nchan,dtype=float)
    crpix=float(header[f'CRPIX{spec}']); crval=float(header[f'CRVAL{spec}']); cdelt=float(header[f'CDELT{spec}'])
    vals=crval+(pix+1-crpix)*cdelt
    ctype=str(header[f'CTYPE{spec}']).upper(); cunit=str(header.get(f'CUNIT{spec}','Hz')).lower()
    if 'FREQ' in ctype:
        scale=1e-9 if 'hz' in cunit else 1.0
        return vals*scale,spec
    rest=float(header.get('RESTFRQ',header.get('RESTFREQ',NU_REST_GHZ*1e9)))
    vel=vals*(1e-3 if 'm/s' in cunit else 1.0)
    return rest*(1.0-vel/299792.458)*1e-9,spec

def squeeze_cube(data,header):
    arr=np.asarray(data,dtype=float)
    arr=np.squeeze(arr)
    if arr.ndim!=3: raise ValueError(f'Expected 3-D cube after squeeze, got shape {arr.shape}')
    # FITS numpy order reverses axis order. Find spectral numpy axis from CTYPE.
    naxis=int(header['NAXIS']); spec_fits=None
    for i in range(1,naxis+1):
        if any(k in str(header.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec_fits=i
    if spec_fits is None: raise ValueError('No spectral axis')
    spec_np=naxis-spec_fits
    while spec_np>=arr.ndim: spec_np-=1
    arr=np.moveaxis(arr,spec_np,0)
    return arr

def extract_from_fits(path):
    with fits.open(path,memmap=True) as hdul:
        hdu=next((h for h in hdul if getattr(h,'data',None) is not None and np.asarray(h.data).ndim>=3),None)
        if hdu is None: raise ValueError('No image cube HDU')
        hdr=hdu.header.copy(); cube=squeeze_cube(hdu.data,hdr)
    nchan,ny,nx=cube.shape
    freq,_=spectral_axis_ghz(hdr,nchan)
    if not (np.nanmin(freq)<TARGET_GHZ<np.nanmax(freq)):
        raise ValueError(f'Cube frequency range {np.nanmin(freq):.3f}-{np.nanmax(freq):.3f} GHz does not include target')
    dw=np.nanmedian(np.abs(np.diff(freq)))*1000
    wc=WCS(hdr).celestial
    x0,y0=wc.world_to_pixel(SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg))
    yy,xx=np.indices((ny,nx)); rr=np.hypot(xx-x0,yy-y0)
    pixscale=np.mean(np.abs(wc.proj_plane_pixel_scales()))*3600
    rap=max(1.0,0.15/pixscale)
    src=rr<=rap
    ann=(rr>=max(rap*3,4))&(rr<=max(rap*6,8))
    flux=np.nansum(cube[:,src],axis=1)
    # Per-channel uncertainty from identical-aperture-equivalent robust annulus RMS.
    annvals=cube[:,ann]
    med=np.nanmedian(annvals,axis=1)
    mad=np.nanmedian(np.abs(annvals-med[:,None]),axis=1)
    sigma_pix=1.4826*mad
    unc=sigma_pix*np.sqrt(np.count_nonzero(src))
    return freq,flux,unc,dw,pixscale,rap

def gaussian(x,a,mu,sig,c): return c+a*np.exp(-0.5*((x-mu)/sig)**2)

def analyse(freq,flux,unc,source_name,dw):
    order=np.argsort(freq); freq=freq[order]; flux=flux[order]; unc=unc[order]
    broad=np.abs(freq-TARGET_GHZ)<0.12
    line=np.abs(freq-TARGET_GHZ)<0.045
    base=broad & ~line
    p=np.polyfit(freq[base]-TARGET_GHZ,flux[base],1,w=1/np.maximum(unc[base],np.nanmedian(unc[base])))
    continuum=np.polyval(p,freq-TARGET_GHZ)
    f=flux-continuum
    # Paper-style first moment: include the fixed line window, preserving signed flux.
    denom=np.nansum(f[line])
    centroid=np.nansum(freq[line]*f[line])/denom
    z=NU_REST_GHZ/centroid-1
    # Gaussian cross-check.
    x=freq[line]; y=f[line]; e=np.where(np.isfinite(unc[line])&(unc[line]>0),unc[line],np.nanmedian(unc[line]))
    p0=[np.nanmax(y),centroid,0.012,0]
    bounds=([0,x.min(),0.001,-np.inf],[np.inf,x.max(),0.08,np.inf])
    popt,pcov=curve_fit(gaussian,x,y,p0=p0,sigma=e,absolute_sigma=True,bounds=bounds,maxfev=30000)
    gcent=popt[1]; gfwhm=2*np.sqrt(2*np.log(2))*abs(popt[2])*1000
    # Monte Carlo first-moment uncertainty using measured per-channel errors.
    rng=np.random.default_rng(184); vals=[]
    for _ in range(20000):
        fm=f+rng.normal(0,unc)
        d=np.nansum(fm[line])
        if np.isfinite(d) and abs(d)>0: vals.append(np.nansum(freq[line]*fm[line])/d)
    vals=np.asarray(vals); cstd=np.nanstd(vals,ddof=1)
    out=pd.DataFrame({'frequency_GHz':freq,'channel_spacing_MHz':np.r_[np.nan,np.diff(freq)*1000],'flux_native_units':flux,'continuum_native_units':continuum,'line_flux_native_units':f,'uncertainty_native_units':unc,'in_centroid_window':line})
    csv=CSV/f'{VERSION}_ALMA_5MHZ_CHANNELS.csv'; out.to_csv(csv,index=False)
    fig,ax=plt.subplots(figsize=(16,7))
    ax.errorbar(freq[broad],f[broad],yerr=unc[broad],fmt='o',ms=4,lw=.8,capsize=2,label='Extracted 5 MHz channels')
    gx=np.linspace(x.min(),x.max(),1200); ax.plot(gx,gaussian(gx,*popt),lw=2,label='Gaussian cross-check')
    ax.axvline(centroid,ls='--',lw=1.8,label=f'Flux-weighted centroid {centroid:.9f} GHz')
    ax.axvline(gcent,ls='-.',lw=1.6,label=f'Gaussian centroid {gcent:.9f} GHz')
    ax.set_xlabel('Observed frequency [GHz]'); ax.set_ylabel('Continuum-subtracted flux [native cube units]')
    ax.set_title(f'JADES-GS-z11-0 ALMA [O III] 88 µm — extracted channel spectrum\nchannel spacing ≈ {dw:.6f} MHz | source: {source_name}')
    ax.grid(color='#303944',lw=.6,alpha=.75); ax.legend(frameon=False,loc='upper left')
    png=PNG/f'{VERSION}_ALMA_5MHZ_CHANNEL_SPECTRUM.png'; fig.savefig(png,dpi=500,bbox_inches='tight'); plt.close(fig)
    tab=pd.DataFrame([
        ['Flux-weighted centroid',f'{centroid:.9f} GHz','Signed first moment in ±45 MHz window'],
        ['Flux-weighted redshift',f'{z:.9f}',f'z = {NU_REST_GHZ:.6f}/ν − 1'],
        ['Monte Carlo centroid σ',f'{cstd*1000:.6f} MHz','20,000 channel-noise realizations'],
        ['Gaussian centroid',f'{gcent:.9f} GHz','Independent profile-fit cross-check'],
        ['Gaussian FWHM',f'{gfwhm:.6f} MHz','2.35482 σ'],
        ['Median channel spacing',f'{dw:.6f} MHz','Read directly from FITS WCS'],
    ],columns=['Quantity','Value','Method'])
    fig,ax=plt.subplots(figsize=(16,5)); ax.axis('off')
    t=ax.table(cellText=tab.values,colLabels=tab.columns,loc='center',cellLoc='left',colLoc='left',bbox=[.01,.05,.98,.85]); t.auto_set_font_size(False); t.set_fontsize(11)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_facecolor('#111820' if r else '#1c2b38'); cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
    ax.set_title('JADES-GS-z11-0 — 5 MHz channel centroid reproduction',fontsize=15,pad=14)
    tablepng=PNG/f'{VERSION}_CENTROID_RESULTS_TABLE.png'; fig.savefig(tablepng,dpi=500,bbox_inches='tight'); plt.close(fig)
    return centroid,z,gcent,gfwhm,cstd,csv,png,tablepng

def run_extract(_=None):
    with status:
        clear_output(wait=True)
        if not products: print('Run the archive query first.'); return
        r=products[int(product_dd.value)]
        print('Selected:',r['file_name']); print('Downloading public ALMA product...')
        try:
            path=download(r['url']); fits_files=unpack(path)
            if not fits_files: raise RuntimeError('No FITS files found in downloaded product')
            print(f'FITS candidates: {len(fits_files)}')
            successes=[]
            for fp in fits_files:
                try:
                    freq,flux,unc,dw,pix,rap=extract_from_fits(fp)
                    successes.append((fp,freq,flux,unc,dw,pix,rap))
                    print(f'Usable cube: {fp.name} | {freq.min():.3f}-{freq.max():.3f} GHz | Δν={dw:.6f} MHz')
                except Exception: pass
            if not successes: raise RuntimeError('No downloaded FITS cube covered 279.901 GHz. Choose another product from the dropdown.')
            best=min(successes,key=lambda s:abs(s[4]-5.0))
            fp,freq,flux,unc,dw,pix,rap=best
            centroid,z,gcent,gfwhm,cstd,csv,png,tablepng=analyse(freq,flux,unc,fp.name,dw)
            display(Image(filename=str(png))); display(Image(filename=str(tablepng)))
            print(f'CODE OUTPUT: {VERSION}')
            print(f'Archive project: {PROJECT}')
            print(f'Cube: {fp}')
            print(f'Median channel spacing: {dw:.6f} MHz')
            print(f'Flux-weighted centroid: {centroid:.9f} GHz')
            print(f'Flux-weighted redshift: {z:.9f}')
            print(f'Monte Carlo centroid sigma: {cstd*1000:.6f} MHz')
            print(f'Gaussian centroid: {gcent:.9f} GHz')
            print(f'Gaussian FWHM: {gfwhm:.6f} MHz')
            print(f'PNG: {png}')
            print(f'TABLE PNG: {tablepng}')
            print(f'CSV: {csv}')
            print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
            print(f'# {VERSION}')
        except Exception as e:
            print('EXTRACTION STOPPED:',e)
            print('This widget does not invent channel values. Select another reduced cube/product and run again.')

query_btn.on_click(query_archive); run_btn.on_click(run_extract)
display(widgets.VBox([query_btn,product_dd,run_btn,status]))
query_archive()
