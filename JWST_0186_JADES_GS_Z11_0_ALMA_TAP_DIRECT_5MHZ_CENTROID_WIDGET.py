# JWST_0186
import sys, subprocess, warnings, os, re, tarfile, zipfile
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
for pkg in ['pyvo','astropy','scipy','requests','ipywidgets']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, pyvo
from scipy.optimize import curve_fit
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from IPython.display import display, clear_output, Image
import ipywidgets as widgets

VERSION='JWST_0186'
PROJECT='2023.1.00336.S'
TARGET='JADES-GS-z11-0'
RA_DEG=53.1647632
DEC_DEG=-27.7746223
NU_REST_GHZ=3393.006244
TARGET_GHZ=279.901
OUT=Path('/content/JWST_OUTPUT'); PNG=OUT/'PNG'; CSV=OUT/'CSV'; DATA=OUT/'ALMA_0186'
for d in (PNG,CSV,DATA): d.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})

status=widgets.Output()
run_btn=widgets.Button(description='Fetch ALMA cube + extract centroid',button_style='success',layout=widgets.Layout(width='360px'))
gallery_btn=widgets.Button(description='Open output gallery',button_style='info',layout=widgets.Layout(width='260px'))

TAP_URLS=[
 'https://almascience.nrao.edu/tap',
 'https://almascience.eso.org/tap',
 'https://almascience.nao.ac.jp/tap'
]

def fmt_bytes(x):
    try:
        x=float(x)
        for unit in ['B','KB','MB','GB','TB']:
            if x<1024: return f'{x:.1f} {unit}'
            x/=1024
    except Exception: return 'unknown'

def q(service,adql):
    return service.search(adql,maxrec=2000).to_table().to_pandas()

def query_products():
    adql=f"""SELECT TOP 2000 * FROM ivoa.obscore
    WHERE obs_collection='ALMA'
    AND proposal_id='{PROJECT}'"""
    errors=[]
    for url in TAP_URLS:
        try:
            print(f'TAP query: {url}')
            df=q(pyvo.dal.TAPService(url),adql)
            if len(df):
                print(f'Rows returned: {len(df)}')
                return df,url
        except Exception as e: errors.append(f'{url}: {e}')
    raise RuntimeError('All TAP mirrors failed\n'+'\n'.join(errors))

def score_rows(df):
    rows=[]
    for i,r in df.iterrows():
        text=' '.join(str(r.get(c,'')) for c in df.columns).lower()
        access=str(r.get('access_url',''))
        if not access: continue
        score=0
        if TARGET.lower() in text: score+=50
        if 'spw.22' in text or 'spw22' in text: score+=20
        if 'cube' in text: score+=18
        if 'fits' in text or str(r.get('access_format','')).lower().find('fits')>=0: score+=12
        if 'science' in text: score+=6
        if 'image' in text: score+=4
        if 'auxiliary' in text or 'weblog' in text or 'qa' in text: score-=30
        em_min=r.get('em_min',np.nan); em_max=r.get('em_max',np.nan)
        try:
            f1=299792458.0/float(em_max)/1e9; f2=299792458.0/float(em_min)/1e9
            if min(f1,f2)<=TARGET_GHZ<=max(f1,f2): score+=35
        except Exception: pass
        size=r.get('access_estsize',np.nan)
        rows.append({'index':i,'score':score,'url':access,'title':str(r.get('obs_title',r.get('obs_id',''))),'size':size,'raw':r})
    rows.sort(key=lambda x:(-x['score'],float(x['size']) if pd.notna(x['size']) else 1e99))
    return rows

def download(url):
    name=re.sub(r'[^A-Za-z0-9._-]+','_',url.split('?')[0].rstrip('/').split('/')[-1] or 'alma_product')
    path=DATA/name
    if path.exists() and path.stat().st_size>1024: return path
    with requests.get(url,stream=True,timeout=(30,300),allow_redirects=True) as r:
        r.raise_for_status(); total=int(r.headers.get('content-length',0)); done=0
        with open(path,'wb') as f:
            for chunk in r.iter_content(1024*1024):
                if chunk:
                    f.write(chunk); done+=len(chunk)
                    if total: print(f'\rDownloading {100*done/total:5.1f}%  {fmt_bytes(done)}/{fmt_bytes(total)}',end='')
    print(); return path

def unpack(path):
    root=DATA/'unpacked'; root.mkdir(exist_ok=True)
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
    pix=np.arange(n,dtype=float); vals=float(h[f'CRVAL{spec}'])+(pix+1-float(h[f'CRPIX{spec}']))*float(h[f'CDELT{spec}'])
    ctype=str(h[f'CTYPE{spec}']).upper(); cunit=str(h.get(f'CUNIT{spec}','Hz')).lower()
    if 'FREQ' in ctype: return vals*(1e-9 if 'hz' in cunit else 1.0)
    rest=float(h.get('RESTFRQ',h.get('RESTFREQ',NU_REST_GHZ*1e9))); vel=vals*(1e-3 if 'm/s' in cunit else 1.0)
    return rest*(1-vel/299792.458)*1e-9

def squeeze_cube(data,h):
    arr=np.squeeze(np.asarray(data,dtype=float)); naxis=int(h['NAXIS']); spec_fits=None
    for i in range(1,naxis+1):
        if any(k in str(h.get(f'CTYPE{i}','')).upper() for k in ['FREQ','VRAD','VELO']): spec_fits=i
    if arr.ndim!=3 or spec_fits is None: raise ValueError(f'Not a usable 3-D spectral cube: {arr.shape}')
    spec_np=naxis-spec_fits
    while spec_np>=arr.ndim: spec_np-=1
    return np.moveaxis(arr,spec_np,0)

def extract(path):
    with fits.open(path,memmap=True) as hdul:
        hdu=next((h for h in hdul if getattr(h,'data',None) is not None and np.asarray(h.data).ndim>=3),None)
        if hdu is None: raise ValueError('No cube HDU')
        hdr=hdu.header.copy(); cube=squeeze_cube(hdu.data,hdr)
    freq=spectral_axis_ghz(hdr,cube.shape[0])
    if not (np.nanmin(freq)<=TARGET_GHZ<=np.nanmax(freq)): raise ValueError(f'Frequency range {np.nanmin(freq):.3f}-{np.nanmax(freq):.3f} GHz misses target')
    wc=WCS(hdr).celestial; x0,y0=wc.world_to_pixel(SkyCoord(RA_DEG*u.deg,DEC_DEG*u.deg))
    yy,xx=np.indices(cube.shape[1:]); rr=np.hypot(xx-x0,yy-y0); pixscale=np.mean(np.abs(wc.proj_plane_pixel_scales()))*3600
    rap=max(1.0,0.15/pixscale); src=rr<=rap; ann=(rr>=max(rap*3,4))&(rr<=max(rap*6,8))
    flux=np.nansum(cube[:,src],axis=1); av=cube[:,ann]; med=np.nanmedian(av,axis=1); mad=np.nanmedian(np.abs(av-med[:,None]),axis=1)
    unc=1.4826*mad*np.sqrt(np.count_nonzero(src)); return freq,flux,unc,np.nanmedian(np.abs(np.diff(freq)))*1000,pixscale,rap

def gauss(x,a,mu,sig,c): return c+a*np.exp(-.5*((x-mu)/sig)**2)

def analyse(freq,flux,unc,source,dw):
    o=np.argsort(freq); freq=freq[o]; flux=flux[o]; unc=unc[o]
    broad=np.abs(freq-TARGET_GHZ)<.12; line=np.abs(freq-TARGET_GHZ)<.045; base=broad & ~line
    w=1/np.maximum(unc[base],np.nanmedian(unc[base])); p=np.polyfit(freq[base]-TARGET_GHZ,flux[base],1,w=w)
    cont=np.polyval(p,freq-TARGET_GHZ); f=flux-cont; centroid=np.sum(freq[line]*f[line])/np.sum(f[line]); z=NU_REST_GHZ/centroid-1
    x=freq[line]; y=f[line]; e=np.where((unc[line]>0)&np.isfinite(unc[line]),unc[line],np.nanmedian(unc[line]))
    popt,_=curve_fit(gauss,x,y,p0=[np.nanmax(y),centroid,.012,0],sigma=e,absolute_sigma=True,bounds=([0,x.min(),.001,-np.inf],[np.inf,x.max(),.08,np.inf]),maxfev=30000)
    rng=np.random.default_rng(186); vals=[]
    for _ in range(20000):
        fm=f+rng.normal(0,unc); d=np.sum(fm[line])
        if np.isfinite(d) and abs(d)>0: vals.append(np.sum(freq[line]*fm[line])/d)
    cstd=np.std(vals,ddof=1); gcent=popt[1]; gfwhm=2*np.sqrt(2*np.log(2))*abs(popt[2])*1000
    table=pd.DataFrame({'frequency_GHz':freq,'channel_spacing_MHz':np.r_[np.nan,np.diff(freq)*1000],'flux_native_units':flux,'continuum_native_units':cont,'line_flux_native_units':f,'uncertainty_native_units':unc,'in_centroid_window':line})
    csv=CSV/f'{VERSION}_ALMA_5MHZ_CHANNELS.csv'; table.to_csv(csv,index=False)
    fig,ax=plt.subplots(figsize=(16,7)); ax.errorbar(freq[broad],f[broad],yerr=unc[broad],fmt='o',ms=4,lw=.8,capsize=2,label='Extracted channels')
    gx=np.linspace(x.min(),x.max(),1200); ax.plot(gx,gauss(gx,*popt),lw=2,label='Gaussian cross-check')
    ax.axvline(centroid,ls='--',lw=1.8,label=f'Flux-weighted centroid {centroid:.9f} GHz'); ax.axvline(gcent,ls='-.',lw=1.6,label=f'Gaussian centroid {gcent:.9f} GHz')
    ax.set(xlabel='Observed frequency [GHz]',ylabel='Continuum-subtracted flux [native cube units]',title=f'{TARGET} ALMA [O III] 88 µm — direct TAP extraction\nchannel spacing ≈ {dw:.6f} MHz | source: {source}')
    ax.grid(color='#303944',lw=.6,alpha=.75); ax.legend(frameon=False,loc='upper left')
    png=PNG/f'{VERSION}_ALMA_5MHZ_CHANNEL_SPECTRUM.png'; fig.savefig(png,dpi=500,bbox_inches='tight'); plt.close(fig)
    s=pd.DataFrame([['Flux-weighted centroid',f'{centroid:.9f} GHz'],['Flux-weighted redshift',f'{z:.9f}'],['Monte Carlo centroid sigma',f'{cstd*1000:.6f} MHz'],['Gaussian centroid',f'{gcent:.9f} GHz'],['Gaussian FWHM',f'{gfwhm:.6f} MHz'],['Measured channel spacing',f'{dw:.6f} MHz']])
    fig,ax=plt.subplots(figsize=(13,4.6)); ax.axis('off'); tab=ax.table(cellText=s.values,colLabels=['Quantity','Value'],loc='center',cellLoc='left',colLoc='left',bbox=[.02,.05,.96,.82]); tab.auto_set_font_size(False); tab.set_fontsize(11)
    for (r,c),cell in tab.get_celld().items(): cell.set_edgecolor('#40505f'); cell.set_facecolor('#111820' if r else '#1c2b38'); cell.get_text().set_color('white')
    ax.set_title(f'{TARGET} — direct ALMA channel-centroid results',fontsize=15,pad=14); tpng=PNG/f'{VERSION}_CENTROID_RESULTS_TABLE.png'; fig.savefig(tpng,dpi=500,bbox_inches='tight'); plt.close(fig)
    return centroid,z,cstd,gcent,gfwhm,csv,png,tpng

def run(_=None):
    with status:
        clear_output(wait=True)
        print(f'CODE OUTPUT: {VERSION}')
        print('Direct TAP mode: bypasses the failing DataLink metadata endpoint.')
        print(f'Querying ALMA project {PROJECT} ...')
        try:
            df,mirror=query_products(); rows=score_rows(df)
            if not rows: raise RuntimeError('No direct-access products returned by TAP')
            pd.DataFrame([{k:v for k,v in r.items() if k!='raw'} for r in rows]).to_csv(CSV/f'{VERSION}_TAP_PRODUCT_CANDIDATES.csv',index=False)
            success=None; errors=[]
            for n,r in enumerate(rows[:25],1):
                print(f'Candidate {n:02d}: score={r["score"]} | {r["title"][:80]}')
                try:
                    path=download(r['url']); fits_files=unpack(path)
                    for fp in fits_files:
                        try:
                            freq,flux,unc,dw,pix,rap=extract(fp); success=(fp,freq,flux,unc,dw,pix,rap); break
                        except Exception as e: errors.append(f'{fp.name}: {e}')
                    if success: break
                except Exception as e: errors.append(f'{r["url"]}: {e}')
            if not success: raise RuntimeError('No usable spectral cube found. Last diagnostics:\n'+'\n'.join(errors[-12:]))
            fp,freq,flux,unc,dw,pix,rap=success
            centroid,z,cstd,gcent,gfwhm,csv,png,tpng=analyse(freq,flux,unc,fp.name,dw)
            print(f'TAP mirror: {mirror}')
            print(f'Cube: {fp}')
            print(f'Channel spacing: {dw:.6f} MHz')
            print(f'Flux-weighted centroid: {centroid:.9f} GHz')
            print(f'Flux-weighted redshift: {z:.9f}')
            print(f'Monte Carlo centroid sigma: {cstd*1000:.6f} MHz')
            print(f'Gaussian centroid: {gcent:.9f} GHz')
            print(f'Gaussian FWHM: {gfwhm:.6f} MHz')
            print(f'PNG: {png}\nTABLE PNG: {tpng}\nCSV: {csv}')
            display(Image(filename=str(png))); display(Image(filename=str(tpng)))
        except Exception as e:
            print('ERROR:',e)
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')

def gallery(_=None):
    with status:
        clear_output(wait=True)
        files=sorted(PNG.glob('JWST_0186*.png'))
        if not files: print('No JWST_0186 PNG files yet. Run the extraction first.'); return
        dd=widgets.Dropdown(options=[(p.name,str(p)) for p in files],description='Image:',layout=widgets.Layout(width='95%'))
        pane=widgets.Output()
        def show(change=None):
            with pane:
                clear_output(wait=True); display(Image(filename=dd.value))
        dd.observe(show,names='value'); display(dd,pane); show()

run_btn.on_click(run); gallery_btn.on_click(gallery)
display(widgets.HBox([run_btn,gallery_btn]),status)
