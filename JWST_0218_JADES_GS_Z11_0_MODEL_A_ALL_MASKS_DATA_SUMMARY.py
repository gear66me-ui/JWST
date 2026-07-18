from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, base64, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import label
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0218'
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
OUT_PNG=Path('/content/JWST_OUTPUT/PNG'); OUT_CSV=Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG=ROOT/'PNG'; DRIVE_CSV=ROOT/'CSV'
for p in (OUT_PNG,OUT_CSV,DRIVE_PNG,DRIVE_CSV): p.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d','text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8','xtick.color':'#c7d4e5','ytick.color':'#c7d4e5','grid.color':'#33485f'})
COL={'model':'#4dd0e1','beam':'#90caf9','seed':'#ffb74d','free':'#ef5350','target':'#ffffff','paper':'#f5f7fa','peak':'#fdd835'}

def robust_rms(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    if x.size==0:return np.nan
    m=np.median(x); mad=np.median(np.abs(x-m))
    return 1.4826*mad if mad>0 else np.std(x)

def spec_axis(h):
    for ax in range(1,int(h['NAXIS'])+1):
        if 'FREQ' in str(h.get(f'CTYPE{ax}','')).upper(): return ax
    raise RuntimeError('No frequency axis')

def axis_values(h,ax):
    n=int(h[f'NAXIS{ax}']); p=np.arange(n)+1.0
    return (float(h[f'CRVAL{ax}'])+(p-float(h[f'CRPIX{ax}']))*float(h.get(f'CDELT{ax}',h.get(f'CD{ax}_{ax}'))))/1e9

def collapse(a,np_spec):
    a=np.moveaxis(a,np_spec,0)
    while a.ndim>3:a=np.nanmean(a,axis=1)
    return a

def component(mask,seed):
    labs,n=label(mask); sy,sx=seed
    if n==0:return np.zeros_like(mask)
    q=labs[sy,sx]
    return labs==q if q>0 else np.zeros_like(mask)

def savefig(fig,name,figs):
    p=OUT_PNG/name; fig.savefig(p,dpi=210,bbox_inches='tight'); plt.close(fig)
    (DRIVE_PNG/p.name).write_bytes(p.read_bytes()); figs.append(p)

def upload(files):
    token=os.environ.get('GITHUB_TOKEN')
    if not token:
        try:
            from google.colab import userdata; token=userdata.get('GITHUB_TOKEN')
        except Exception: token=None
    if not token:return 'GitHub upload skipped: add Colab secret GITHUB_TOKEN.'
    import requests
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); ok=0
    for p in files:
        path=f'results/{VERSION}/{stamp}/{p.name}'
        payload={'message':f'Add {VERSION} result {p.name}','content':base64.b64encode(p.read_bytes()).decode(),'branch':'main'}
        r=requests.put(f'https://api.github.com/repos/gear66me-ui/JWST/contents/{path}',headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'},json=payload,timeout=120)
        ok+=int(r.status_code in (200,201))
    return f'GitHub uploaded: {ok}/{len(files)} files'

def outline(ax,mask,color,label_text):
    ys,xs=np.where(mask)
    for y,x in zip(ys,xs): ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,ec=color,lw=.9))
    ax.plot([],[],color=color,lw=2,label=f'{label_text} ({len(xs)} px)')

def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        w=WCS(h).celestial; tx0,ty0=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(np.asarray(tx0).reshape(-1)[0])); ty=int(np.rint(np.asarray(ty0).reshape(-1)[0]))
        pix=abs(float(h['CDELT2']))*3600; bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600; bpa=float(h.get('BPA',0))
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half); ny=nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq)))*1000; k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups]); cube=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pix; dy=(yy-tyc)*pix; rr=np.hypot(dx,dy)
    th=np.deg2rad(bpa); xr=dx*np.cos(th)+dy*np.sin(th); yr=-dx*np.sin(th)+dy*np.cos(th)
    sx=max(bmaj/2.35482,.15); sy=max(bmin/2.35482,.12)
    template=np.exp(-.5*((xr/sx)**2+(yr/sy)**2)); template/=np.sqrt(np.sum(template**2))
    beam_mask=(template/template.max())>=np.exp(-.5)
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ; line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    off=rr>1.2; moment=np.sum(cube[line_sel],axis=0); map_rms=robust_rms(moment[off]); snmap=moment/map_rms
    seeded=component((snmap>=2)&(rr<=.8),(tyc,txc)); seeded=seeded if seeded.any() else beam_mask.copy()
    cand=(snmap>=2)&(rr<=.8); labs,n=label(cand); free=np.zeros_like(cand)
    if n:
        choices=[(np.sum(snmap[labs==q]),labs==q) for q in range(1,n+1) if np.any(labs==q)]
        if choices: free=max(choices,key=lambda z:z[0])[1]
    if not free.any(): free=beam_mask.copy()
    blank=[]
    for rad_pix in (18,24,30,36,42):
        for ang in np.linspace(0,2*np.pi,16,endpoint=False):
            dxs=int(round(rad_pix*np.cos(ang))); dys=int(round(rad_pix*np.sin(ang)))
            t=np.roll(np.roll(template,dys,axis=0),dxs,axis=1)
            blank.append([np.nansum(im*t) for im in cube])
    blank=np.asarray(blank)
    methods={
        'Model A fixed point template':('template',template),
        'Fixed beam pixel mask':('mask',beam_mask),
        'Target-seeded mask':('mask',seeded),
        'Free optimized mask':('mask',free),
    }
    summaries=[]; spectra=[]
    for name,(kind,obj) in methods.items():
        if kind=='template':
            flux=np.array([np.nansum(im*obj) for im in cube]); noise=robust_rms(np.sum(blank[:,line_sel],axis=1)); pixels=int(beam_mask.sum())
        else:
            flux=np.array([np.nansum(im[obj]) for im in cube]); noise=robust_rms(flux[~line_sel])*np.sqrt(line_sel.sum()); pixels=int(obj.sum())
        line_flux=np.sum(flux[line_sel]); snr=line_flux/noise if noise>0 else np.nan
        pos=np.clip(flux[line_sel],0,None); cen=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
        summaries.append({'method':name,'centroid_GHz':cen,'offset_MHz':(cen-TARGET_GHZ)*1000 if np.isfinite(cen) else np.nan,'line_flux_sum':line_flux,'line_noise_sigma':noise,'integrated_SNR':snr,'mask_pixels':pixels,'line_channels':int(line_sel.sum())})
        spectra.append(pd.DataFrame({'method':name,'channel_index':np.arange(len(centers)),'frequency_GHz':centers,'velocity_km_s':vel,'flux':flux,'line_window':line_sel}))
    summary=pd.DataFrame(summaries); specdf=pd.concat(spectra,ignore_index=True)
    scsv=OUT_CSV/f'{VERSION}_ALL_MASKS_SUMMARY.csv'; pcsv=OUT_CSV/f'{VERSION}_ALL_MASKS_SPECTRA.csv'
    summary.to_csv(scsv,index=False); specdf.to_csv(pcsv,index=False)
    for p in (scsv,pcsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(10,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest')
    ax.contour(template/template.max(),levels=[.5],colors=[COL['model']],linewidths=2.2)
    outline(ax,beam_mask,COL['beam'],'Fixed beam'); outline(ax,seeded,COL['seed'],'Target-seeded'); outline(ax,free,COL['free'],'Free optimized')
    ax.scatter([txc],[tyc],marker='+',s=190,c=COL['target'],linewidths=2.2,label='Exact JWST coordinate')
    ax.set_title('MODEL A — all extraction masks at the exact JWST coordinate'); ax.legend(loc='upper right',fontsize=8); fig.colorbar(im,ax=ax,label='Moment-0 surface brightness')
    savefig(fig,f'{VERSION}_MODEL_A_ALL_MASKS_MOMENT0.png',figs)
    fig,ax=plt.subplots(figsize=(12,7))
    for name,color in zip(methods.keys(),[COL['model'],COL['beam'],COL['seed'],COL['free']]):
        d=specdf[specdf.method==name]; ax.plot(d.frequency_GHz,d.flux,lw=1.7,label=name,color=color)
    ax.axvline(TARGET_GHZ,ls='--',c=COL['paper'],label='Reference 279.901 GHz'); ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.10,color='#ffffff')
    ax.set_title('MODEL A — spectra from all masks'); ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Extracted amplitude / summed flux'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    savefig(fig,f'{VERSION}_MODEL_A_ALL_MASKS_SPECTRA.png',figs)
    show=summary.copy(); show['centroid_GHz']=show['centroid_GHz'].map(lambda x:f'{x:.6f}'); show['offset_MHz']=show['offset_MHz'].map(lambda x:f'{x:+.6f}'); show['integrated_SNR']=show['integrated_SNR'].map(lambda x:f'{x:.6f}')
    fig,ax=plt.subplots(figsize=(15,4.7)); ax.axis('off'); tbl=ax.table(cellText=show.values,colLabels=show.columns,loc='center',cellLoc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1,1.6)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor('#33485f'); cell.set_facecolor('#10202f' if r else '#1c3448'); cell.get_text().set_color('#e8f1ff')
    ax.set_title('MODEL A — all mask numerical results',pad=18,fontsize=15)
    savefig(fig,f'{VERSION}_MODEL_A_ALL_MASKS_TABLE.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='92%')); img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1200px')); cap=widgets.HTML(); state={'i':0}; prevb=widgets.Button(description='Previous'); nextb=widgets.Button(description='Next')
    def show_image(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br>{p}'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show_image(state['i']-1)); nextb.on_click(lambda _:show_image(state['i']+1)); show_image(0)
    display(widgets.VBox([widgets.HTML('<h3>Model A — all masks and numerical summary</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[scsv,pcsv]+figs
    print(f'CODE OUTPUT: {VERSION}')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
