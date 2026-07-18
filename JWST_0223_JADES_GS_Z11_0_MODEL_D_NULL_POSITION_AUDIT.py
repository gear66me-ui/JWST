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

VERSION='JWST_0223'
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


def shifted(t,dy,dx):
    return np.roll(np.roll(t,dy,axis=0),dx,axis=1)


def main():
    warnings.filterwarnings('ignore')
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        h=hdul[0].header; data=hdul[0].data
        fax=spec_axis(h); freq=axis_values(h,fax); np_spec=data.ndim-fax
        w=WCS(h).celestial; tx0,ty0=w.world_to_pixel_values(RA_DEG,DEC_DEG)
        tx=int(np.rint(tx0)); ty=int(np.rint(ty0)); pix=abs(float(h['CDELT2']))*3600
        bmaj=float(h.get('BMAJ',0))*3600; bmin=float(h.get('BMIN',0))*3600; bpa=float(h.get('BPA',0))
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
    yy,xx=np.indices((ny,nx)); dx=(xx-txc)*pix; dy=(yy-tyc)*pix
    th=np.deg2rad(bpa); xr=dx*np.cos(th)+dy*np.sin(th); yr=-dx*np.sin(th)+dy*np.cos(th)
    sx=max(bmaj/2.35482,.15); sy=max(bmin/2.35482,.12)
    template=np.exp(-.5*((xr/sx)**2+(yr/sy)**2)); template/=np.sqrt(np.sum(template**2))
    vel=C_KMS*(TARGET_GHZ-centers)/TARGET_GHZ; line_sel=(vel>=-14)&(vel<=29)
    if line_sel.sum()<5:
        ids=np.argsort(np.abs(centers-TARGET_GHZ))[:8]; line_sel=np.zeros(len(centers),bool); line_sel[ids]=True
    target_spec=np.array([np.nansum(im*template) for im in cube])
    offsets=[]; specs=[]
    for r in (18,22,26,30,34,38,42,46):
        for a in np.linspace(0,2*np.pi,24,endpoint=False):
            ox=int(round(r*np.cos(a))); oy=int(round(r*np.sin(a)))
            if abs(ox)>50 or abs(oy)>50: continue
            offsets.append((ox,oy)); t=shifted(template,oy,ox)
            specs.append([np.nansum(im*t) for im in cube])
    specs=np.asarray(specs,float); sums=np.sum(specs[:,line_sel],axis=1)
    null_med=np.median(sums); null_sig=robust_rms(sums-null_med)
    target_sum=np.sum(target_spec[line_sel]); target_snr=(target_sum-null_med)/null_sig
    empirical_p=(1+np.sum(sums>=target_sum))/(len(sums)+1)
    percentile=100*np.mean(sums<target_sum)
    pos=np.clip(target_spec[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    jnear=int(np.argmin(np.abs(centers-TARGET_GHZ)))
    nearest_freq=centers[jnear]; nearest_offset=(nearest_freq-TARGET_GHZ)*1000
    moment=np.sum(cube[line_sel],axis=0)
    py,px=np.unravel_index(np.nanargmax(moment),moment.shape); peak_sep=np.hypot(px-txc,py-tyc)*pix
    summary=pd.DataFrame([{
        'model':'Model D — null-position audit','target_frequency_GHz':TARGET_GHZ,
        'nearest_image_index':jnear,'nearest_image_frequency_GHz':nearest_freq,
        'nearest_image_offset_MHz':nearest_offset,'line_channels':int(line_sel.sum()),
        'target_line_sum':target_sum,'null_median':null_med,'null_sigma':null_sig,
        'target_SNR_vs_null':target_snr,'empirical_p_one_sided':empirical_p,
        'target_percentile':percentile,'centroid_GHz':centroid,
        'centroid_offset_MHz':(centroid-TARGET_GHZ)*1000 if np.isfinite(centroid) else np.nan,
        'moment0_peak_offset_arcsec':peak_sep,'null_positions':len(sums),
        'beam_major_arcsec':bmaj,'beam_minor_arcsec':bmin}])
    null_df=pd.DataFrame({'null_id':np.arange(len(sums)),'dx_pixel':[o[0] for o in offsets],'dy_pixel':[o[1] for o in offsets],'line_sum':sums,'null_sigma_units':(sums-null_med)/null_sig})
    chan_df=pd.DataFrame({'channel_index':np.arange(len(centers)),'frequency_GHz':centers,'offset_MHz':(centers-TARGET_GHZ)*1000,'velocity_km_s':vel,'target_template_amplitude':target_spec,'line_window':line_sel,'nearest_target_image':np.arange(len(centers))==jnear})
    scsv=OUT_CSV/f'{VERSION}_MODEL_D_SUMMARY.csv'; ncsv=OUT_CSV/f'{VERSION}_MODEL_D_NULL_POSITIONS.csv'; ccsv=OUT_CSV/f'{VERSION}_MODEL_D_CHANNELS.csv'
    summary.to_csv(scsv,index=False); null_df.to_csv(ncsv,index=False); chan_df.to_csv(ccsv,index=False)
    for p in (scsv,ncsv,ccsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cube[jnear],origin='lower',cmap='viridis',interpolation='nearest')
    ax.contour(template/template.max(),levels=[0.5],colors=['#4dd0e1'],linewidths=2)
    ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='JWST catalog coordinate')
    ax.scatter([px],[py],marker='x',s=110,c='#fdd835',linewidths=2,label=f'Line-window peak: {peak_sep:.3f} arcsec')
    ax.set_title(f'MODEL D — nearest image to 279.901 GHz\nImage {jnear:02d}: {nearest_freq:.6f} GHz | offset {nearest_offset:+.3f} MHz')
    ax.set_xlabel('Cutout x pixel'); ax.set_ylabel('Cutout y pixel'); ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Surface brightness')
    savefig(fig,f'{VERSION}_MODEL_D_IMAGE_{jnear:02d}_{nearest_freq:.6f}GHz_TARGET_AUDIT.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest')
    ax.contour(template/template.max(),levels=[0.5],colors=['#4dd0e1'],linewidths=2)
    ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='JWST coordinate')
    ax.scatter([px],[py],marker='x',s=110,c='#fdd835',linewidths=2,label=f'Moment-0 peak: {peak_sep:.3f} arcsec')
    ax.set_title(f'MODEL D — 8-channel moment-0 | target S/N vs null = {target_snr:.3f}')
    ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Integrated brightness')
    savefig(fig,f'{VERSION}_MODEL_D_MOMENT0_TARGET_VS_NULL.png',figs)
    fig,ax=plt.subplots(figsize=(11,7)); ax.hist(sums,bins=24,alpha=.8,label=f'{len(sums)} null positions')
    ax.axvline(target_sum,c='#fdd835',lw=2.2,label=f'Target line sum | S/N={target_snr:.3f}')
    ax.axvline(null_med,c='white',ls='--',lw=1.5,label='Null median')
    ax.set_title(f'MODEL D — empirical null distribution | p={empirical_p:.4f} | percentile={percentile:.1f}%')
    ax.set_xlabel('8-channel matched-template sum'); ax.set_ylabel('Null-position count'); ax.grid(alpha=.25); ax.legend()
    savefig(fig,f'{VERSION}_MODEL_D_NULL_DISTRIBUTION.png',figs)
    fig,ax=plt.subplots(figsize=(12,7)); ax.plot(centers,target_spec,lw=1.8,label='Exact JWST-position template spectrum')
    ax.axvline(TARGET_GHZ,ls='--',c='white',label='Target 279.901 GHz'); ax.axvline(nearest_freq,ls=':',c='#fdd835',label=f'Nearest image {jnear:02d}: {nearest_freq:.6f} GHz')
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.10,label='8-channel line window')
    ax.set_title(f'MODEL D spectrum | centroid {centroid:.6f} GHz | offset {(centroid-TARGET_GHZ)*1000:+.3f} MHz')
    ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Matched-template amplitude'); ax.grid(alpha=.25); ax.legend()
    savefig(fig,f'{VERSION}_MODEL_D_TARGET_SPECTRUM_WITH_IMAGE_INDEX.png',figs)
    fig,ax=plt.subplots(figsize=(14,4.8)); ax.axis('off')
    table_data=[['Target frequency',f'{TARGET_GHZ:.6f} GHz'],['Nearest image',f'{jnear:02d} — {nearest_freq:.6f} GHz'],['Nearest-image offset',f'{nearest_offset:+.3f} MHz'],['Target S/N vs null',f'{target_snr:.3f}'],['Empirical one-sided p',f'{empirical_p:.4f}'],['Target percentile',f'{percentile:.1f}%'],['Centroid',f'{centroid:.6f} GHz'],['Centroid offset',f'{(centroid-TARGET_GHZ)*1000:+.3f} MHz'],['Moment-0 peak offset',f'{peak_sep:.3f} arcsec'],['Null positions',str(len(sums))]]
    tab=ax.table(cellText=table_data,colLabels=['MODEL D result','Value'],loc='center',cellLoc='left'); tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1,1.55)
    ax.set_title('MODEL D — result and image identification table',pad=16)
    savefig(fig,f'{VERSION}_MODEL_D_RESULTS_TABLE.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='94%'))
    img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(); state={'i':0}; prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1)); show(0)
    display(widgets.VBox([widgets.HTML('<h3>Model D — null-position audit with exact 279.901 GHz image identification</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[scsv,ncsv,ccsv]+figs
    print(f'CODE OUTPUT: {VERSION}'); print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}')); print('\nIMAGE IDENTIFICATION'); print(f'Nearest 5 MHz image to 279.901 GHz: image {jnear:02d} = {nearest_freq:.6f} GHz ({nearest_offset:+.3f} MHz)'); print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files)); print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
