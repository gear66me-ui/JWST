# JWST_0185
import sys, subprocess, warnings, tarfile, zipfile, shutil, os, re
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

for pkg in ['astroquery','astropy','scipy','requests','ipywidgets']:
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

VERSION='JWST_0185'
PROJECT='2023.1.00336.S'
TARGET_GHZ=279.901
NU_REST_GHZ=3393.006244
RA_DEG=53.1647632
DEC_DEG=-27.7746223
OUT=Path('/content/JWST_OUTPUT')
PNG=OUT/'PNG'; CSV=OUT/'CSV'; DATA=OUT/'ALMA_0185'
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})

status=widgets.Output()
run_btn=widgets.Button(description='Fetch ALMA cube + extract centroid',button_style='success',layout=widgets.Layout(width='360px'))
gallery_dd=widgets.Dropdown(description='Image:',layout=widgets.Layout(width='95%'))
gallery_out=widgets.Output()
image_paths=[]

def fmt_bytes(n):
    try:
        n=float(n)
        for unit in ['B','KB','MB','GB','TB']:
            if n<1024: return f'{n:.1f} {unit}'
            n/=1024
    except Exception: pass
    return 'unknown'

def safe_name(url):
    return url.split('?')[0].rstrip('/').split('/')[-1] or 'alma_product'

def download(url):
    path=DATA/safe_name(url)
    if path.exists() and path.stat().st_size>0:
        print(f'Using cached: {path.name}')
        return path
    print(f'Downloading: {path.name}')
    with requests.get(url,stream=True,timeout=180,allow_redirects=True) as r:
        r.raise_for_status()
        total=int(r.headers.get('content-length',0)); done=0
        with open(path,'wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk); done+=len(chunk)
                    if total:
                        print(f'\r  {100*done/total:5.1f}%  {fmt_bytes(done)} / {fmt_bytes(total)}',end='')
    print()
    return path

def unpack(path):
    root=DATA/(path.stem+'_unpacked')
    root.mkdir(exist_ok=True)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as t: t.extractall(root)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z: z.extractall(root)
    elif path.suffix.lower() in ['.fits','.fit','.fz'] or path.name.endswith('.fits.gz'):
        return [path]
    return list(root.rglob('*.fits'))+list(root.rglob('*.fit'))+list(root.rglob('*.fits.gz'))

def spectral_axis_ghz(h,n):
    spec=None
    for i in range(1,int(h.get('NAXIS',0))+1):
        c=str(h.get(f'CTYPE{i}','')).upper()
        if any(k in c for k in ['FREQ','VRAD','VELO']): spec=i; break
    if spec is None: raise ValueError('No spectral axis')
    pix=np.arange(n,dtype=float)
    vals=float(h[f'CRVAL{spec}'])+(pix+1-float(h[f'CRPIX{spec}']))*float(h[f'CDELT{spec}'])
    ctype=str(h[f'CTYPE{spec}']).upper(); unit=str(h.get(f'CUNIT{spec}','Hz')).lower()
    if 'FREQ' in ctype:
        if 'ghz' in unit: return vals,spec
        if 'mhz' in unit: return vals/1000.0,spec
        return vals*1e-9,spec
    rest=float(h.get('RESTFRQ',h.get('RESTFREQ',NU_REST_GHZ*1e9)))
    vel=vals*(1e-3 if 'm/s' in unit else 1.0)
    return rest*(1.0-vel/299792.458)*1e-9,spec

def cube_from_hdu(hdu):
    arr=np.asarray(hdu.data,dtype=float)
    arr=np.squeeze(arr)
    if arr.ndim!=3: raise ValueError(f'Expected 3-D cube, got {arr.shape}')
    h=hdu.header
    naxis=int(h['NAXIS']); spec_fits=None
    for i in range(1,naxis+1):
        if any(k in str(h.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec_fits=i; break
    if spec_fits is None: raise ValueError('No spectral axis')
    spec_np=naxis-spec_fits
    while spec_np>=arr.ndim: spec_np-=1
    return np.moveaxis(arr,spec_np,0)

def inspect_cube(path):
    with fits.open(path,memmap=True) as hdul:
        for hdu in hdul:
            if getattr(hdu,'data',None) is None: continue
            try:
                cube=cube_from_hdu(hdu)
                freq,_=spectral_axis_ghz(hdu.header,cube.shape[0])
                if np.nanmin(freq)<TARGET_GHZ<np.nanmax(freq):
                    return cube,hdu.header.copy(),freq
            except Exception:
                continue
    raise ValueError('No suitable spectral cube containing target frequency')

def extract_spectrum(cube,hdr,freq):
    nchan,ny,nx=cube.shape
    wc=WCS(hdr).celestial
    try:
        x0,y0=wc.world_to_pixel(SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg))
    except Exception:
        x0=(nx-1)/2; y0=(ny-1)/2
    yy,xx=np.indices((ny,nx)); rr=np.hypot(xx-x0,yy-y0)
    pixscale=float(np.nanmean(np.abs(wc.proj_plane_pixel_scales()))*3600)
    rap=max(1.0,0.15/max(pixscale,1e-6))
    src=rr<=rap
    ann=(rr>=max(rap*3,4))&(rr<=max(rap*6,8))
    flux=np.nansum(cube[:,src],axis=1)
    annvals=cube[:,ann]
    med=np.nanmedian(annvals,axis=1)
    mad=np.nanmedian(np.abs(annvals-med[:,None]),axis=1)
    unc=1.4826*mad*np.sqrt(np.count_nonzero(src))
    return flux,unc,pixscale,rap

def gaussian(x,a,mu,sig,c):
    return c+a*np.exp(-0.5*((x-mu)/sig)**2)

def analyse(freq,flux,unc,source):
    order=np.argsort(freq); freq=freq[order]; flux=flux[order]; unc=unc[order]
    broad=np.abs(freq-TARGET_GHZ)<0.12
    line=np.abs(freq-TARGET_GHZ)<0.045
    base=broad & ~line
    med_unc=np.nanmedian(unc[np.isfinite(unc)&(unc>0)])
    w=1/np.maximum(np.where(np.isfinite(unc[base]),unc[base],med_unc),med_unc)
    p=np.polyfit(freq[base]-TARGET_GHZ,flux[base],1,w=w)
    cont=np.polyval(p,freq-TARGET_GHZ)
    lf=flux-cont
    denom=np.nansum(lf[line])
    centroid=np.nansum(freq[line]*lf[line])/denom
    z=NU_REST_GHZ/centroid-1
    x=freq[line]; y=lf[line]; e=np.where(np.isfinite(unc[line])&(unc[line]>0),unc[line],med_unc)
    p0=[max(np.nanmax(y),1e-12),centroid,0.012,0]
    bounds=([0,x.min(),0.001,-np.inf],[np.inf,x.max(),0.08,np.inf])
    popt,pcov=curve_fit(gaussian,x,y,p0=p0,sigma=e,absolute_sigma=True,bounds=bounds,maxfev=30000)
    gcent=float(popt[1]); gfwhm=2*np.sqrt(2*np.log(2))*abs(float(popt[2]))*1000
    rng=np.random.default_rng(185); vals=[]
    for _ in range(20000):
        fm=lf+rng.normal(0,np.where(np.isfinite(unc),unc,med_unc))
        d=np.nansum(fm[line])
        if np.isfinite(d) and abs(d)>0: vals.append(np.nansum(freq[line]*fm[line])/d)
    vals=np.asarray(vals); cstd=float(np.nanstd(vals,ddof=1))
    dw=float(np.nanmedian(np.abs(np.diff(freq)))*1000)

    table=pd.DataFrame({'frequency_GHz':freq,'channel_spacing_MHz':np.r_[np.nan,np.diff(freq)*1000],'flux_native_units':flux,'continuum_native_units':cont,'line_flux_native_units':lf,'uncertainty_native_units':unc,'in_centroid_window':line})
    csv_path=CSV/f'{VERSION}_ALMA_5MHZ_CHANNELS.csv'; table.to_csv(csv_path,index=False)

    fig,ax=plt.subplots(figsize=(16,7))
    ax.errorbar(freq[broad],lf[broad],yerr=unc[broad],fmt='o',ms=4,lw=.8,capsize=2,label='Extracted channels')
    gx=np.linspace(x.min(),x.max(),1200); ax.plot(gx,gaussian(gx,*popt),lw=2,label='Gaussian cross-check')
    ax.axvline(centroid,ls='--',lw=1.8,label=f'Flux-weighted centroid {centroid:.9f} GHz')
    ax.axvline(gcent,ls='-.',lw=1.6,label=f'Gaussian centroid {gcent:.9f} GHz')
    ax.set_xlabel('Observed frequency [GHz]'); ax.set_ylabel('Continuum-subtracted flux [native cube units]')
    ax.set_title(f'JADES-GS-z11-0 ALMA [O III] 88 µm channel extraction\nspacing ≈ {dw:.6f} MHz | source: {source}')
    ax.grid(color='#303944',lw=.6,alpha=.75); ax.legend(frameon=False,loc='upper left')
    spectrum_png=PNG/f'{VERSION}_ALMA_CHANNEL_SPECTRUM.png'; fig.savefig(spectrum_png,dpi=500,bbox_inches='tight'); plt.close(fig)

    rows=[
        ['Archive project',PROJECT,'Public ALMA Science Archive'],
        ['Channel spacing',f'{dw:.6f} MHz','Measured from FITS spectral axis'],
        ['Flux-weighted centroid',f'{centroid:.9f} GHz','Signed first moment in ±45 MHz window'],
        ['Flux-weighted redshift',f'{z:.9f}',f'z = {NU_REST_GHZ:.6f}/ν − 1'],
        ['Monte Carlo centroid σ',f'{cstd*1000:.6f} MHz','20,000 channel-noise realizations'],
        ['Gaussian centroid',f'{gcent:.9f} GHz','Cross-check only'],
        ['Gaussian FWHM',f'{gfwhm:.6f} MHz','2.35482 σ'],
        ['CSV rows',str(len(table)),'All extracted channels']]
    fig,ax=plt.subplots(figsize=(16,5.3)); ax.axis('off')
    tab=ax.table(cellText=rows,colLabels=['Quantity','Value','Interpretation'],loc='center',cellLoc='left',colLoc='left',bbox=[0.01,0.03,0.98,0.88])
    tab.auto_set_font_size(False); tab.set_fontsize(10.5)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_linewidth(.8); cell.set_facecolor('#111820' if r else '#1c2b38'); cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
    ax.set_title('JADES-GS-z11-0 — ALMA direct-archive centroid summary',fontsize=15,pad=14)
    table_png=PNG/f'{VERSION}_CENTROID_SUMMARY_TABLE.png'; fig.savefig(table_png,dpi=500,bbox_inches='tight'); plt.close(fig)

    return {'centroid':centroid,'z':z,'cstd':cstd,'gcent':gcent,'gfwhm':gfwhm,'dw':dw,'csv':csv_path,'spectrum_png':spectrum_png,'table_png':table_png}

def candidate_products():
    Alma.cache_location=str(DATA/'astroquery_cache')
    result=Alma.query(payload={'project_code':PROJECT,'public_data':True})
    if result is None or len(result)==0: raise RuntimeError('ALMA archive query returned no rows')
    uids=[]
    for col in ['member_ous_uid','group_ous_uid','obs_id']:
        if col in result.colnames:
            uids.extend([str(v) for v in result[col] if str(v) not in ('','--','None')])
    uids=list(dict.fromkeys(uids))
    rows=[]
    for uid in uids:
        try:
            info=Alma.get_data_info(uid)
            if info is None: continue
            for r in info:
                cols=info.colnames
                url=str(r['access_url']) if 'access_url' in cols else ''
                name=str(r['file_name']) if 'file_name' in cols else safe_name(url)
                size=float(r['content_length']) if 'content_length' in cols else np.nan
                text=(name+' '+url).lower(); score=0
                if 'fits' in text: score+=15
                if 'cube' in text: score+=12
                if 'image' in text: score+=5
                if 'science' in text or 'product' in text: score+=4
                if 'pbcor' in text: score+=3
                if 'auxiliary' in text or 'weblog' in text or 'qa' in text: score-=20
                if url: rows.append({'url':url,'name':name,'size':size,'score':score})
        except Exception as e:
            print(f'Metadata warning for {uid}: {e}')
    rows.sort(key=lambda r:(-r['score'], r['size'] if np.isfinite(r['size']) else 1e99))
    return rows

def show_gallery(change=None):
    with gallery_out:
        clear_output(wait=True)
        if image_paths:
            p=image_paths[gallery_dd.value]
            display(Image(filename=str(p)))

def run(_=None):
    global image_paths
    with status:
        clear_output(wait=True)
        print('CODE OUTPUT:',VERSION)
        print('Direct archive mode: no file upload is required.')
        print(f'Querying ALMA project {PROJECT} ...')
        try:
            products=candidate_products()
            print(f'Candidate downloadable products: {len(products)}')
            chosen=None; last_error=None
            for i,r in enumerate(products[:40],1):
                print(f'[{i:02d}] score={r["score"]:2d} size={fmt_bytes(r["size"])} {r["name"][:100]}')
                try:
                    path=download(r['url'])
                    fits_files=unpack(path)
                    for fp in fits_files:
                        try:
                            cube,hdr,freq=inspect_cube(fp)
                            chosen=(fp,cube,hdr,freq)
                            break
                        except Exception as e:
                            last_error=e
                    if chosen: break
                except Exception as e:
                    last_error=e
                    print('  skipped:',e)
            if not chosen:
                raise RuntimeError(f'No suitable public spectral cube was found automatically. Last error: {last_error}')
            fp,cube,hdr,freq=chosen
            print(f'Using cube: {fp}')
            flux,unc,pixscale,rap=extract_spectrum(cube,hdr,freq)
            result=analyse(freq,flux,unc,fp.name)
            image_paths=[result['spectrum_png'],result['table_png']]
            gallery_dd.options=[(p.name,i) for i,p in enumerate(image_paths)]
            gallery_dd.value=0
            print(f'Channel spacing: {result["dw"]:.6f} MHz')
            print(f'Flux-weighted centroid: {result["centroid"]:.9f} GHz')
            print(f'Flux-weighted redshift: {result["z"]:.9f}')
            print(f'Centroid Monte Carlo sigma: {result["cstd"]*1000:.6f} MHz')
            print(f'Gaussian centroid: {result["gcent"]:.9f} GHz')
            print(f'Gaussian FWHM: {result["gfwhm"]:.6f} MHz')
            print(f'CSV: {result["csv"]}')
            print(f'PNG: {result["spectrum_png"]}')
            print(f'TABLE PNG: {result["table_png"]}')
            print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
            print(f'# {VERSION}')
            display(gallery_dd,gallery_out)
            show_gallery()
        except Exception as e:
            print('ERROR:',e)
            print('No upload prompt will appear. The failure, if any, is from ALMA archive availability or product access.')
            print(f'# {VERSION}')

run_btn.on_click(run)
gallery_dd.observe(show_gallery,names='value')
display(run_btn,status)
