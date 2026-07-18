from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, base64, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0222'
MODEL='MODEL_G_TARGET_FREQUENCY_IMAGE_AUDIT'
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
        r=requests.put(f'https://api.github.com/repos/gear66me-ui/JWST/contents/{path}',headers={'Authorization':f'Bearer {token}','Accept':'application/vnd.github+json'},json=payload,timeout=180)
        ok+=int(r.status_code in (200,201))
    return f'GitHub uploaded: {ok}/{len(files)} files'


def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        aw=WCS(h).celestial; tx0,ty0=aw.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); pix=abs(float(h['CDELT2']))*3600
        bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600; bpa=float(h.get('BPA',0))
        half=64; ys=slice(ty-half,ty+half); xs=slice(tx-half,tx+half); ny=nx=128; txc=tyc=64
        native=np.nanmedian(np.abs(np.diff(freq)))*1000; k=max(1,int(round(5/native)))
        near=int(np.argmin(np.abs(freq-TARGET_GHZ))); lo=max(0,near-14*k); hi=min(len(freq),near+14*k)
        groups=[np.arange(i,min(i+k,hi)) for i in range(lo,hi,k)]
        centers=np.array([np.mean(freq[g]) for g in groups])
        lows=np.array([np.min(freq[g]) for g in groups]); highs=np.array([np.max(freq[g]) for g in groups])
        cube=np.empty((len(groups),ny,nx),np.float32)
        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim; sl[np_spec]=g; sl[-2]=ys; sl[-1]=xs
            cube[j]=np.nanmean(collapse(np.asarray(data[tuple(sl)],np.float32),np_spec),axis=0)
            if (j+1)%5==0 or j==len(groups)-1: print(f'5 MHz cube: {j+1:3d}/{len(groups):3d}')
        del data; gc.collect()
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pix; dy=(yy-tyc)*pix
    th=np.deg2rad(bpa); xr=dx*np.cos(th)+dy*np.sin(th); yr=-dx*np.sin(th)+dy*np.cos(th)
    sx=max(bmaj/2.35482,.15); sy=max(bmin/2.35482,.12)
    template=np.exp(-.5*((xr/sx)**2+(yr/sy)**2)); template/=np.sqrt(np.sum(template**2))
    amps=np.array([np.nansum(im*template) for im in cube])
    blank=[]
    for r in (18,24,30,36,42):
        for a in np.linspace(0,2*np.pi,16,endpoint=False):
            sxp=int(round(r*np.cos(a))); syp=int(round(r*np.sin(a)))
            t=np.roll(np.roll(template,syp,0),sxp,1); blank.append([np.nansum(im*t) for im in cube])
    blank=np.asarray(blank); noise=np.array([robust_rms(blank[:,j]) for j in range(len(centers))]); snr=amps/noise
    offset_mhz=(centers-TARGET_GHZ)*1000; velocity=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ
    captures=(lows<=TARGET_GHZ)&(highs>=TARGET_GHZ)
    nearest_order=np.argsort(np.abs(offset_mhz))[:5]
    line_sel=(velocity>=-14)&(velocity<=29)
    moment=np.sum(cube[line_sel],axis=0)
    rows=[]; figs=[]
    common_vmin=np.nanpercentile(cube[nearest_order],2); common_vmax=np.nanpercentile(cube[nearest_order],98)
    for j in nearest_order:
        tag='CAPTURES_TARGET' if captures[j] else 'NEAR_TARGET'
        name=f'{VERSION}_MODEL_G_{tag}_CH{j:02d}_{centers[j]:.6f}GHz_{offset_mhz[j]:+.3f}MHz.png'
        fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cube[j],origin='lower',cmap='viridis',vmin=common_vmin,vmax=common_vmax,interpolation='nearest')
        ax.contour(template/template.max(),levels=[0.5],colors=['#4dd0e1'],linewidths=2)
        ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='Exact JWST coordinate')
        ax.add_patch(Ellipse((9,9),bmaj/pix,bmin/pix,angle=bpa,fill=False,ec='#4dd0e1',lw=2))
        verdict='YES — grouped image contains 279.901 GHz' if captures[j] else 'NO — nearest grouped image only'
        ax.set_title(f'MODEL G — {verdict}\ncenter {centers[j]:.6f} GHz | offset {offset_mhz[j]:+.3f} MHz | target-template S/N {snr[j]:+.3f}')
        ax.set_xlabel('Cutout x pixel'); ax.set_ylabel('Cutout y pixel'); ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Surface brightness, common scale')
        savefig(fig,name,figs)
        rows.append({'rank_by_target_distance':len(rows)+1,'image_filename':name,'channel_index':j,'center_frequency_GHz':centers[j],'group_low_GHz':lows[j],'group_high_GHz':highs[j],'offset_MHz':offset_mhz[j],'velocity_km_s':velocity[j],'captures_279.901_GHz':'YES' if captures[j] else 'NO','target_template_SNR':snr[j]})
    mname=f'{VERSION}_MODEL_G_LINE_WINDOW_MOMENT0_279.901GHz.png'
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest'); ax.contour(template/template.max(),levels=[0.5],colors=['#4dd0e1'],linewidths=2); ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='Exact JWST coordinate'); ax.set_title('MODEL G — eight-channel line-window moment-0 around 279.901 GHz'); ax.legend(); fig.colorbar(im,ax=ax,label='Integrated brightness'); savefig(fig,mname,figs)
    table=pd.DataFrame(rows)
    csv=OUT_CSV/f'{VERSION}_MODEL_G_IMAGE_FREQUENCY_RESULTS.csv'; table.to_csv(csv,index=False); (DRIVE_CSV/csv.name).write_bytes(csv.read_bytes())
    fig,ax=plt.subplots(figsize=(16,4.8)); ax.axis('off'); show=table[['image_filename','center_frequency_GHz','offset_MHz','captures_279.901_GHz','target_template_SNR']].copy(); show['center_frequency_GHz']=show['center_frequency_GHz'].map(lambda x:f'{x:.6f}'); show['offset_MHz']=show['offset_MHz'].map(lambda x:f'{x:+.3f}'); show['target_template_SNR']=show['target_template_SNR'].map(lambda x:f'{x:+.3f}'); t=ax.table(cellText=show.values,colLabels=['Image filename','Center GHz','Offset MHz','Contains 279.901?','Target S/N'],loc='center',cellLoc='center'); t.auto_set_font_size(False); t.set_fontsize(8.5); t.scale(1,1.7); ax.set_title('MODEL G — image names tied directly to frequency results',pad=20,fontsize=15); savefig(fig,f'{VERSION}_MODEL_G_IMAGE_RESULTS_TABLE.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='95%')); img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(); state={'i':0}; prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show_image(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show_image(state['i']-1)); nextb.on_click(lambda _:show_image(state['i']+1)); show_image(0)
    display(widgets.VBox([widgets.HTML('<h3>Model G — 279.901 GHz image/frequency audit</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[csv]+figs
    print(f'CODE OUTPUT: {VERSION}')
    print(table.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    hit=table[table['captures_279.901_GHz']=='YES']
    print('\nTARGET IMAGE RESULT')
    if len(hit): print(hit[['image_filename','center_frequency_GHz','group_low_GHz','group_high_GHz','offset_MHz','target_template_SNR']].to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    else: print('No grouped image spans exactly 279.901 GHz; nearest images are listed above.')
    print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
