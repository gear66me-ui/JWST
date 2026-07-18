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

VERSION='JWST_0216'
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
COL={'beam':'#4dd0e1','seed':'#ffb74d','free':'#ef5350','target':'#ffffff','peak':'#fdd835','centroid':'#ce93d8'}

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

def weighted_xy(image,mask):
    yy,xx=np.indices(image.shape); w=np.where(mask,np.clip(image,0,None),0.0); s=np.nansum(w)
    if not np.isfinite(s) or s<=0:return np.nan,np.nan
    return np.nansum(xx*w)/s,np.nansum(yy*w)/s

def savefig(fig,name,figs):
    p=OUT_PNG/name; fig.savefig(p,dpi=210,bbox_inches='tight'); plt.close(fig)
    (DRIVE_PNG/p.name).write_bytes(p.read_bytes()); figs.append(p)

def outline_pixels(ax,mask,color,label_text):
    ys,xs=np.where(mask)
    for y,x in zip(ys,xs): ax.add_patch(Rectangle((x-.5,y-.5),1,1,fill=False,ec=color,lw=1.25))
    if len(xs): ax.plot([],[],color=color,lw=2,label=f'{label_text} ({len(xs)} px)')

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

def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists():raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data; shape=data.shape
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
            if (j+1)%5==0 or j==len(groups)-1:print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pix; dy=(yy-tyc)*pix; rr=np.hypot(dx,dy)
    th=np.deg2rad(bpa); xr=dx*np.cos(th)+dy*np.sin(th); yr=-dx*np.sin(th)+dy*np.cos(th)
    sx=max(bmaj/2.35482,.15); sy=max(bmin/2.35482,.12); beam_w=np.exp(-.5*((xr/sx)**2+(yr/sy)**2)); beam=beam_w>=np.exp(-.5)
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ; line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    off=rr>1.2; moment=np.sum(cube[line_sel],axis=0); mr=robust_rms(moment[off]); sn=moment/mr
    seeded=component((sn>=2)&(rr<=.8),(tyc,txc)); seeded=seeded if seeded.any() else beam.copy()
    cand=(sn>=2)&(rr<=.8); labs,n=label(cand); free=np.zeros_like(cand)
    if n:
        choices=[(np.sum(sn[labs==q]),labs==q) for q in range(1,n+1) if np.any(labs==q)]
        if choices:free=max(choices,key=lambda z:z[0])[1]
    if not free.any():free=beam.copy()
    peak_y,peak_x=np.unravel_index(np.nanargmax(moment),moment.shape); cx,cy=weighted_xy(moment,free)
    peak_sep=np.hypot(peak_x-txc,peak_y-tyc)*pix; cen_sep=np.hypot(cx-txc,cy-tyc)*pix
    common_vmin=np.nanpercentile(cube[line_sel],2); common_vmax=np.nanpercentile(cube[line_sel],98)
    figs=[]
    for j in np.where(line_sel)[0]:
        fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cube[j],origin='lower',cmap='viridis',vmin=common_vmin,vmax=common_vmax,interpolation='nearest')
        ax.scatter([txc],[tyc],marker='+',s=170,c=COL['target'],linewidths=2,label='JWST coordinate')
        ax.scatter([peak_x],[peak_y],marker='x',s=90,c=COL['peak'],linewidths=2,label='Moment-0 peak')
        ax.add_patch(Ellipse((9,9),bmaj/pix,bmin/pix,angle=bpa,fill=False,ec=COL['beam'],lw=2)); ax.text(9,3,'beam',ha='center',color=COL['beam'])
        ax.set_title(f'Raw 5 MHz channel {j:02d} | {centers[j]:.6f} GHz | {vel[j]:+.2f} km/s'); ax.set_xlabel('Cutout x pixel'); ax.set_ylabel('Cutout y pixel'); ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Surface brightness, common scale')
        savefig(fig,f'{VERSION}_RAW_CHANNEL_{j:02d}_{centers[j]:.6f}GHz.png',figs)
    for name,mask,color in [('TARGET_SEEDED',seeded,COL['seed']),('FREE_OPTIMIZED',free,COL['free'])]:
        fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest')
        outline_pixels(ax,mask,color,name.replace('_',' ').title()); ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2,label='JWST coordinate')
        ax.scatter([peak_x],[peak_y],marker='x',s=100,c=COL['peak'],linewidths=2,label=f'Peak: {peak_sep:.3f} arcsec')
        if np.isfinite(cx):ax.scatter([cx],[cy],marker='o',facecolors='none',edgecolors=COL['centroid'],s=100,linewidths=2,label=f'Free-mask centroid: {cen_sep:.3f} arcsec')
        ax.add_patch(Ellipse((9,9),bmaj/pix,bmin/pix,angle=bpa,fill=False,ec=COL['beam'],lw=2)); ax.legend(loc='upper right'); ax.set_title(f'Moment-0 pixel audit — {name.replace("_"," ")}'); fig.colorbar(im,ax=ax,label='Integrated surface brightness')
        savefig(fig,f'{VERSION}_{name}_PIXEL_AUDIT.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(sn,origin='lower',cmap='magma',vmin=-3,vmax=max(5,np.nanpercentile(sn,99)),interpolation='nearest')
    ax.contour(sn,levels=[2],colors=['#ffffff'],linewidths=1.1); outline_pixels(ax,seeded,COL['seed'],'Target-seeded'); outline_pixels(ax,free,COL['free'],'Free optimized')
    ax.scatter([txc],[tyc],marker='+',s=180,c=COL['target'],linewidths=2,label='JWST coordinate'); ax.scatter([peak_x],[peak_y],marker='x',s=100,c=COL['peak'],linewidths=2,label='Moment-0 peak'); ax.legend(loc='upper right'); ax.set_title('S/N selection audit: exact selected pixels and S/N=2 boundary'); fig.colorbar(im,ax=ax,label='Moment-0 S/N')
    savefig(fig,f'{VERSION}_SNMAP_EXACT_PIXEL_SELECTION.png',figs)
    positions=pd.DataFrame([{'quantity':'JWST target','x_pixel':txc,'y_pixel':tyc,'offset_arcsec':0.0},{'quantity':'Moment-0 peak','x_pixel':peak_x,'y_pixel':peak_y,'offset_arcsec':peak_sep},{'quantity':'Free-mask positive-flux centroid','x_pixel':cx,'y_pixel':cy,'offset_arcsec':cen_sep}])
    masks=pd.DataFrame([{'mask':'Fixed beam','pixels':int(beam.sum())},{'mask':'Target-seeded','pixels':int(seeded.sum())},{'mask':'Free optimized','pixels':int(free.sum())}])
    pcsv=OUT_CSV/f'{VERSION}_POSITION_AUDIT.csv'; mcsv=OUT_CSV/f'{VERSION}_MASK_AUDIT.csv'; positions.to_csv(pcsv,index=False); masks.to_csv(mcsv,index=False)
    for p in (pcsv,mcsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='92%')); img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(); state={'i':0}; prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1)); show(0)
    display(widgets.VBox([widgets.HTML('<h3>JADES-GS-z11-0 exact mask-pixel and position audit</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[pcsv,mcsv]+figs
    print(f'CODE OUTPUT: {VERSION}'); print(positions.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print(masks.to_string(index=False)); print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__':main()
