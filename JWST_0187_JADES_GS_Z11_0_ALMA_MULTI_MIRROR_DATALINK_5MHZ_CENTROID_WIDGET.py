# JWST_0187
import sys, subprocess, warnings, os, re, tarfile, zipfile, shutil
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

for pkg in ['pyvo','astropy','scipy','requests','pandas','matplotlib','ipywidgets']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import pyvo
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from IPython.display import display, clear_output, Image
import ipywidgets as widgets

VERSION='JWST_0187'
PROJECT='2023.1.00336.S'
TARGET='JADES-GS-z11-0'
RA_DEG=53.1647632
DEC_DEG=-27.7746223
TARGET_GHZ=279.901
NU_REST_GHZ=3393.006244
OUT=Path('/content/JWST_OUTPUT')
PNG=OUT/'PNG'; CSV=OUT/'CSV'; DATA=OUT/'ALMA_0187'
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)

MIRRORS=[
    ('ESO','https://almascience.eso.org'),
    ('NAOJ','https://almascience.nao.ac.jp'),
    ('NRAO','https://almascience.nrao.edu'),
]

plt.rcParams.update({
    'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black',
    'text.color':'white','axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','axes.edgecolor':'#aeb8c3'
})

status=widgets.Output()
run_btn=widgets.Button(description='Fetch ALMA cube + extract centroid',button_style='success',layout=widgets.Layout(width='340px'))
gallery_btn=widgets.Button(description='Open output gallery',button_style='info',layout=widgets.Layout(width='220px'))


def fmt_bytes(n):
    try:
        n=float(n)
        for unit in ['B','KB','MB','GB','TB']:
            if n<1024: return f'{n:.1f} {unit}'
            n/=1024
    except Exception: pass
    return 'unknown'


def safe_name(url):
    name=url.split('?')[0].rstrip('/').split('/')[-1]
    return re.sub(r'[^A-Za-z0-9._-]+','_',name or 'alma_product')


def tap_query(base):
    service=pyvo.dal.TAPService(base+'/tap')
    adql=("SELECT * FROM ivoa.obscore "
          f"WHERE proposal_id='{PROJECT}'")
    return service.search(adql,maxrec=500).to_table().to_pandas()


def choose_mous(df):
    rows=[]
    for _,r in df.iterrows():
        uid=str(r.get('member_ous_uid',''))
        if not uid or uid in ('nan','None','--'): continue
        text=' '.join(str(r.get(c,'')) for c in ['target_name','obs_title','obs_id','member_ous_uid']).lower()
        score=0
        if 'jades-gs-z11-0' in text: score+=100
        if 'jades' in text: score+=30
        lo=float(r.get('em_min',np.nan)) if pd.notna(r.get('em_min',np.nan)) else np.nan
        hi=float(r.get('em_max',np.nan)) if pd.notna(r.get('em_max',np.nan)) else np.nan
        if np.isfinite(lo) and np.isfinite(hi):
            c=299792458.0
            fmax=c/min(lo,hi)/1e9; fmin=c/max(lo,hi)/1e9
            if fmin<=TARGET_GHZ<=fmax: score+=80
        rows.append((score,uid,text))
    rows=sorted(rows,key=lambda x:-x[0])
    out=[]
    for s,u,t in rows:
        if u not in [x[1] for x in out]: out.append((s,u,t))
    return out


def datalink_urls(base,uid):
    ids=[uid,uid.replace('://','___').replace('/','_')]
    errors=[]
    for ident in ids:
        url=f'{base}/datalink/sync?ID={requests.utils.quote(ident,safe="")}'
        try:
            dl=pyvo.dal.adhoc.DatalinkResults.from_result_url(url)
            rows=[]
            for row in dl:
                access=str(row.get('access_url',''))
                if not access: continue
                desc=str(row.get('description',''))
                sem=str(row.get('semantics',''))
                ctype=str(row.get('content_type',''))
                size=row.get('content_length',np.nan)
                text=(access+' '+desc+' '+sem+' '+ctype).lower()
                score=0
                if 'fits' in text: score+=30
                if 'product' in sem.lower(): score+=10
                if 'auxiliary' in text or 'weblog' in text or 'raw' in text: score-=25
                if 'cube' in text or 'image' in text: score+=12
                if 'science' in text: score+=8
                rows.append({'access_url':access,'description':desc,'content_type':ctype,'content_length':size,'score':score})
            if rows: return sorted(rows,key=lambda r:-r['score'])
        except Exception as e:
            errors.append(f'{ident}: {e}')
    raise RuntimeError(' | '.join(errors))


def download(url):
    path=DATA/safe_name(url)
    headers={}
    if path.exists() and path.stat().st_size>0:
        headers['Range']=f'bytes={path.stat().st_size}-'
        mode='ab'
    else:
        mode='wb'
    with requests.get(url,stream=True,timeout=180,headers=headers,allow_redirects=True) as r:
        if r.status_code==416: return path
        r.raise_for_status()
        total=int(r.headers.get('content-length',0))+(path.stat().st_size if path.exists() and mode=='ab' else 0)
        done=path.stat().st_size if path.exists() and mode=='ab' else 0
        with open(path,mode) as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk); done+=len(chunk)
                    if total:
                        print(f'\rDownloading {100*done/total:5.1f}% ({fmt_bytes(done)}/{fmt_bytes(total)})',end='')
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
    except Exception as e:
        print('Unpack warning:',e)
    return list(root.rglob('*.fits'))+list(root.rglob('*.fit'))+list(root.rglob('*.fits.gz'))


def squeeze_cube(data,header):
    arr=np.squeeze(np.asarray(data,dtype=float))
    if arr.ndim!=3: raise ValueError(f'Expected 3-D cube, got {arr.shape}')
    naxis=int(header['NAXIS']); spec_fits=None
    for i in range(1,naxis+1):
        if any(k in str(header.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec_fits=i
    if spec_fits is None: raise ValueError('No spectral axis')
    spec_np=naxis-spec_fits
    while spec_np>=arr.ndim: spec_np-=1
    return np.moveaxis(arr,spec_np,0),spec_fits


def spectral_axis_ghz(h,nchan,spec):
    pix=np.arange(nchan,dtype=float)
    crpix=float(h[f'CRPIX{spec}']); crval=float(h[f'CRVAL{spec}']); cdelt=float(h[f'CDELT{spec}'])
    vals=crval+(pix+1-crpix)*cdelt
    ctype=str(h[f'CTYPE{spec}']).upper(); unit=str(h.get(f'CUNIT{spec}','Hz')).lower()
    if 'FREQ' in ctype:
        return vals*(1e-9 if 'hz' in unit else 1.0)
    rest=float(h.get('RESTFRQ',h.get('RESTFREQ',NU_REST_GHZ*1e9)))
    vel=vals*(1e-3 if 'm/s' in unit else 1.0)
    return rest*(1.0-vel/299792.458)*1e-9


def extract_cube(path):
    with fits.open(path,memmap=True) as hdul:
        hdu=next((x for x in hdul if getattr(x,'data',None) is not None and np.asarray(x.data).ndim>=3),None)
        if hdu is None: raise ValueError('No cube HDU')
        hdr=hdu.header.copy(); cube,spec=squeeze_cube(hdu.data,hdr)
    freq=spectral_axis_ghz(hdr,cube.shape[0],spec)
    if not (np.nanmin(freq)<=TARGET_GHZ<=np.nanmax(freq)):
        raise ValueError(f'Frequency range {np.nanmin(freq):.3f}-{np.nanmax(freq):.3f} GHz misses target')
    w=WCS(hdr).celestial
    x0,y0=w.world_to_pixel(SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg))
    ny,nx=cube.shape[1:]; yy,xx=np.indices((ny,nx)); rr=np.hypot(xx-x0,yy-y0)
    pixscale=np.mean(np.abs(w.proj_plane_pixel_scales()))*3600
    rap=max(1.0,0.15/pixscale)
    src=rr<=rap
    ann=(rr>=max(rap*3,4))&(rr<=max(rap*6,8))
    flux=np.nansum(cube[:,src],axis=1)
    annvals=cube[:,ann]
    med=np.nanmedian(annvals,axis=1)
    mad=np.nanmedian(np.abs(annvals-med[:,None]),axis=1)
    unc=1.4826*mad*np.sqrt(np.count_nonzero(src))
    return freq,flux,unc,pixscale,rap


def gaussian(x,a,mu,sig,c): return c+a*np.exp(-0.5*((x-mu)/sig)**2)


def analyse(freq,flux,unc,source_name):
    order=np.argsort(freq); freq=freq[order]; flux=flux[order]; unc=unc[order]
    broad=np.abs(freq-TARGET_GHZ)<0.12
    line=np.abs(freq-TARGET_GHZ)<0.045
    base=broad & ~line
    w=1/np.maximum(unc[base],np.nanmedian(unc[base]))
    p=np.polyfit(freq[base]-TARGET_GHZ,flux[base],1,w=w)
    continuum=np.polyval(p,freq-TARGET_GHZ)
    f=flux-continuum
    denom=np.nansum(f[line])
    centroid=np.nansum(freq[line]*f[line])/denom
    z=NU_REST_GHZ/centroid-1
    x=freq[line]; y=f[line]; e=np.where(np.isfinite(unc[line])&(unc[line]>0),unc[line],np.nanmedian(unc[line]))
    p0=[np.nanmax(y),centroid,0.012,0]
    bounds=([0,x.min(),0.001,-np.inf],[np.inf,x.max(),0.08,np.inf])
    popt,_=curve_fit(gaussian,x,y,p0=p0,sigma=e,absolute_sigma=True,bounds=bounds,maxfev=30000)
    gcent=popt[1]; gfwhm=2*np.sqrt(2*np.log(2))*abs(popt[2])*1000
    rng=np.random.default_rng(187); vals=[]
    for _ in range(20000):
        fm=f+rng.normal(0,unc)
        d=np.nansum(fm[line])
        if np.isfinite(d) and abs(d)>0: vals.append(np.nansum(freq[line]*fm[line])/d)
    vals=np.asarray(vals); cstd=np.nanstd(vals,ddof=1)
    dw=np.nanmedian(np.abs(np.diff(freq)))*1000
    out=pd.DataFrame({
        'frequency_GHz':freq,
        'channel_spacing_MHz':np.r_[np.nan,np.diff(freq)*1000],
        'flux_native_units':flux,
        'continuum_native_units':continuum,
        'line_flux_native_units':f,
        'uncertainty_native_units':unc,
        'in_centroid_window':line
    })
    csv=CSV/f'{VERSION}_ALMA_5MHZ_CHANNELS.csv'; out.to_csv(csv,index=False)
    fig,ax=plt.subplots(figsize=(16,7))
    ax.errorbar(freq[broad],f[broad],yerr=unc[broad],fmt='o',ms=4,lw=.8,capsize=2,label='Extracted channels')
    gx=np.linspace(x.min(),x.max(),1200); ax.plot(gx,gaussian(gx,*popt),lw=2,label='Gaussian cross-check')
    ax.axvline(centroid,ls='--',lw=1.8,label=f'Flux-weighted centroid {centroid:.9f} GHz')
    ax.axvline(gcent,ls='-.',lw=1.6,label=f'Gaussian centroid {gcent:.9f} GHz')
    ax.set_xlabel('Observed frequency [GHz]'); ax.set_ylabel('Continuum-subtracted flux [native cube units]')
    ax.set_title(f'JADES-GS-z11-0 ALMA [O III] 88 µm channel spectrum\nspacing ≈ {dw:.6f} MHz | source: {source_name}')
    ax.grid(color='#303944',lw=.6,alpha=.75); ax.legend(frameon=False,loc='upper left')
    png=PNG/f'{VERSION}_ALMA_5MHZ_CHANNEL_SPECTRUM.png'; fig.savefig(png,dpi=500,bbox_inches='tight'); plt.close(fig)
    table=pd.DataFrame([
        ['Flux-weighted centroid',f'{centroid:.9f} GHz','Signed first moment in ±45 MHz window'],
        ['Flux-weighted redshift',f'{z:.9f}',f'z = {NU_REST_GHZ:.6f}/ν − 1'],
        ['Gaussian centroid',f'{gcent:.9f} GHz','Cross-check only'],
        ['Gaussian FWHM',f'{gfwhm:.6f} MHz','Single-Gaussian fit'],
        ['Monte Carlo centroid σ',f'{cstd*1000:.6f} MHz','20,000 channel-noise realizations'],
        ['Median channel spacing',f'{dw:.6f} MHz','From FITS spectral WCS'],
    ],columns=['Quantity','Value','Interpretation'])
    fig2,ax2=plt.subplots(figsize=(16,5)); ax2.axis('off')
    tab=ax2.table(cellText=table.values,colLabels=table.columns,loc='center',cellLoc='left',colLoc='left',bbox=[0.01,0.05,0.98,0.82])
    tab.auto_set_font_size(False); tab.set_fontsize(10.5)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_linewidth(.8); cell.set_facecolor('#111820' if r else '#1c2b38'); cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
    ax2.set_title('JADES-GS-z11-0 — ALMA 5 MHz centroid extraction summary',fontsize=15,pad=14)
    tpng=PNG/f'{VERSION}_CENTROID_SUMMARY_TABLE.png'; fig2.savefig(tpng,dpi=500,bbox_inches='tight'); plt.close(fig2)
    return centroid,z,gcent,gfwhm,cstd,dw,csv,png,tpng


def show_gallery(_=None):
    with status:
        clear_output(wait=True)
        files=sorted(PNG.glob('JWST_0187*.png'))
        if not files:
            print('No JWST_0187 PNG files exist yet. Run the extraction first.'); return
        dd=widgets.Dropdown(options=[(p.name,str(p)) for p in files],description='Image:',layout=widgets.Layout(width='95%'))
        o=widgets.Output()
        def render(change=None):
            with o:
                clear_output(wait=True); display(Image(filename=dd.value))
        dd.observe(render,names='value'); display(dd,o); render()


def run(_=None):
    with status:
        clear_output(wait=True)
        print(f'CODE OUTPUT: {VERSION}')
        print('Multi-mirror mode: tries ESO, NAOJ, then NRAO; no upload required.')
        all_errors=[]
        result=None
        for label,base in MIRRORS:
            print(f'\nQuerying {label} TAP: {base}/tap')
            try:
                df=tap_query(base)
                print(f'Rows returned: {len(df)}')
                mous=choose_mous(df)
                print(f'Candidate MOUS: {len(mous)}')
                for i,(score,uid,_) in enumerate(mous[:12],1):
                    print(f'  {i:02d}. score={score} | {uid}')
                for score,uid,_ in mous[:12]:
                    try:
                        print(f'Fetching DataLink from {label}: {uid}')
                        products=datalink_urls(base,uid)
                        for pr in products[:10]:
                            print(f"  product score={pr['score']:2d} | {fmt_bytes(pr['content_length'])} | {pr['description'][:90]}")
                        for pr in products:
                            if pr['score']<0: continue
                            try:
                                path=download(pr['access_url'])
                                fits_files=unpack(path)
                                for fp in fits_files:
                                    try:
                                        freq,flux,unc,pixscale,rap=extract_cube(fp)
                                        result=analyse(freq,flux,unc,fp.name)
                                        raise StopIteration
                                    except Exception as e:
                                        all_errors.append(f'{fp.name}: {e}')
                            except StopIteration:
                                raise
                            except Exception as e:
                                all_errors.append(f"{pr['access_url']}: {e}")
                    except StopIteration:
                        raise
                    except Exception as e:
                        all_errors.append(f'{label} {uid}: {e}')
            except StopIteration:
                break
            except Exception as e:
                all_errors.append(f'{label} TAP: {e}')
        if result is None:
            print('\nERROR: no usable spectral cube was recovered.')
            print('Last diagnostics:')
            for x in all_errors[-12:]: print(' -',x)
        else:
            centroid,z,gcent,gfwhm,cstd,dw,csv,png,tpng=result
            print('\nRESULTS')
            print(f'Flux-weighted centroid: {centroid:.9f} GHz')
            print(f'Flux-weighted redshift: {z:.9f}')
            print(f'Gaussian centroid: {gcent:.9f} GHz')
            print(f'Gaussian FWHM: {gfwhm:.6f} MHz')
            print(f'Monte Carlo centroid sigma: {cstd*1000:.6f} MHz')
            print(f'Channel spacing: {dw:.6f} MHz')
            print(f'PNG: {png}')
            print(f'TABLE PNG: {tpng}')
            print(f'CSV: {csv}')
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')

run_btn.on_click(run); gallery_btn.on_click(show_gallery)
display(widgets.HBox([run_btn,gallery_btn]),status)
