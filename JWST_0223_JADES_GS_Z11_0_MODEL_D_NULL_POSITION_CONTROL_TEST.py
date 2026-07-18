from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, base64, warnings, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets

VERSION='JWST_0223'
MODEL='MODEL_D_NULL_POSITION_CONTROL_TEST'
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

def shift_template(t,dy,dx):
    return np.roll(np.roll(t,dy,axis=0),dx,axis=1)

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
    target_amps=np.array([np.nansum(im*template) for im in cube])
    target_flux=np.sum(target_amps[line_sel])
    controls=[]
    for r in (18,24,30,36,42):
        for a in np.linspace(0,2*np.pi,32,endpoint=False):
            dxp=int(round(r*np.cos(a))); dyp=int(round(r*np.sin(a)))
            t=shift_template(template,dyp,dxp)
            amps=np.array([np.nansum(im*t) for im in cube])
            controls.append({'radius_pix':r,'dx_pix':dxp,'dy_pix':dyp,'offset_arcsec':np.hypot(dxp,dyp)*pix,'line_flux':np.sum(amps[line_sel]),'peak_channel_SNR':np.nan})
    cdf=pd.DataFrame(controls)
    sigma=robust_rms(cdf['line_flux'].values); target_snr=target_flux/sigma
    cdf['line_SNR']=cdf['line_flux']/sigma
    percentile=100.0*np.mean(cdf['line_flux'].values<=target_flux)
    false_alarm=np.mean(cdf['line_flux'].values>=target_flux)
    pos=np.clip(target_amps[line_sel],0,None)
    centroid=np.sum(centers[line_sel]*pos)/np.sum(pos) if np.sum(pos)>0 else np.nan
    nearest=int(np.argmin(np.abs(centers-TARGET_GHZ)))
    moment=np.sum(cube[line_sel],axis=0)
    peak_y,peak_x=np.unravel_index(np.nanargmax(moment),moment.shape)
    peak_sep=np.hypot(peak_x-txc,peak_y-tyc)*pix
    summary=pd.DataFrame([{'model':'Model D — null-position control test','target_frequency_GHz':TARGET_GHZ,'nearest_channel_GHz':centers[nearest],'nearest_channel_offset_MHz':(centers[nearest]-TARGET_GHZ)*1000,'line_channels':int(line_sel.sum()),'target_line_flux':target_flux,'null_sigma':sigma,'target_integrated_SNR':target_snr,'target_percentile_vs_null':percentile,'false_alarm_fraction':false_alarm,'centroid_GHz':centroid,'centroid_offset_MHz':(centroid-TARGET_GHZ)*1000 if np.isfinite(centroid) else np.nan,'moment0_peak_offset_arcsec':peak_sep,'control_positions':len(cdf),'beam_major_arcsec':bmaj,'beam_minor_arcsec':bmin}])
    scsv=OUT_CSV/f'{VERSION}_MODEL_D_SUMMARY.csv'; ccsv=OUT_CSV/f'{VERSION}_MODEL_D_NULL_POSITIONS.csv'
    summary.to_csv(scsv,index=False); cdf.to_csv(ccsv,index=False)
    for p in (scsv,ccsv):(DRIVE_CSV/p.name).write_bytes(p.read_bytes())
    figs=[]
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(cube[nearest],origin='lower',cmap='viridis',interpolation='nearest')
    ax.contour(template/template.max(),levels=[0.5],colors=['#4dd0e1'],linewidths=2)
    ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='JWST coordinate')
    ax.scatter([peak_x],[peak_y],marker='x',s=100,c='#fdd835',linewidths=2,label=f'Line-window peak offset {peak_sep:.3f} arcsec')
    ax.set_title(f'MODEL D target-frequency image\nnearest grouped channel {centers[nearest]:.6f} GHz | offset {(centers[nearest]-TARGET_GHZ)*1000:+.3f} MHz')
    ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Surface brightness')
    savefig(fig,f'{VERSION}_MODEL_D_TARGET_IMAGE_{centers[nearest]:.6f}GHz_OFFSET_{(centers[nearest]-TARGET_GHZ)*1000:+.3f}MHz.png',figs)
    fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(moment,origin='lower',cmap='viridis',interpolation='nearest')
    ax.contour(template/template.max(),levels=[0.5],colors=['#4dd0e1'],linewidths=2)
    ax.scatter([txc],[tyc],marker='+',s=190,c='white',linewidths=2,label='JWST coordinate')
    ax.scatter([peak_x],[peak_y],marker='x',s=100,c='#fdd835',linewidths=2,label=f'Moment-0 peak offset {peak_sep:.3f} arcsec')
    ax.set_title(f'MODEL D line-window moment-0 | target S/N={target_snr:.3f} | null FAP={false_alarm:.3f}')
    ax.legend(loc='upper right'); fig.colorbar(im,ax=ax,label='Integrated brightness')
    savefig(fig,f'{VERSION}_MODEL_D_MOMENT0_SNR_{target_snr:.3f}_FAP_{false_alarm:.3f}.png',figs)
    fig,ax=plt.subplots(figsize=(11,7)); ax.hist(cdf['line_SNR'],bins=24,alpha=.75,label=f'{len(cdf)} null positions')
    ax.axvline(target_snr,ls='--',lw=2,c='white',label=f'JWST target S/N={target_snr:.3f}')
    ax.set_xlabel('Integrated matched-template S/N'); ax.set_ylabel('Control-position count'); ax.grid(alpha=.25)
    ax.set_title(f'MODEL D null distribution | target percentile={percentile:.1f}% | false-alarm fraction={false_alarm:.3f}')
    ax.legend(); savefig(fig,f'{VERSION}_MODEL_D_NULL_DISTRIBUTION_TARGET_SNR_{target_snr:.3f}.png',figs)
    fig,ax=plt.subplots(figsize=(12,7)); ax.plot(centers,target_amps,lw=1.8,label='Exact JWST-position template amplitude')
    ax.axvline(TARGET_GHZ,ls='--',c='white',label='Target 279.901 GHz'); ax.axvline(centers[nearest],ls=':',c='#fdd835',label=f'Nearest image {centers[nearest]:.6f} GHz')
    ax.axvspan(centers[line_sel].min(),centers[line_sel].max(),alpha=.12,label='8-channel line window')
    ax.set_xlabel('Observed frequency (GHz)'); ax.set_ylabel('Matched-template amplitude'); ax.grid(alpha=.25)
    ax.set_title(f'MODEL D spectrum | centroid={centroid:.6f} GHz | offset={(centroid-TARGET_GHZ)*1000:+.3f} MHz')
    ax.legend(); savefig(fig,f'{VERSION}_MODEL_D_SPECTRUM_CENTROID_{centroid:.6f}GHz.png',figs)
    fig,ax=plt.subplots(figsize=(15,4.8)); ax.axis('off')
    table=ax.table(cellText=[[f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v) for v in summary.iloc[0].values]],colLabels=list(summary.columns),loc='center',cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(7); table.scale(1,1.8); ax.set_title('MODEL D — numerical summary',pad=18)
    savefig(fig,f'{VERSION}_MODEL_D_SUMMARY_TABLE.png',figs)
    opts=[(p.name,str(p)) for p in figs]; dd=widgets.Dropdown(options=opts,value=str(figs[0]),description='Image:',layout=widgets.Layout(width='95%'))
    img=widgets.Image(value=figs[0].read_bytes(),format='png',layout=widgets.Layout(width='100%',max_width='1100px')); cap=widgets.HTML(); state={'i':0}
    prevb=widgets.Button(description='Previous',icon='arrow-left'); nextb=widgets.Button(description='Next',icon='arrow-right')
    def show(i):
        state['i']=i%len(figs); p=figs[state['i']]; dd.value=str(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b><br><span style="color:#8aa0b8">{p}</span>'
    def ondd(ch):
        p=Path(ch['new']); state['i']=figs.index(p); img.value=p.read_bytes(); cap.value=f'<b>{state["i"]+1}/{len(figs)} — {p.name}</b>'
    dd.observe(ondd,names='value'); prevb.on_click(lambda _:show(state['i']-1)); nextb.on_click(lambda _:show(state['i']+1)); show(0)
    display(widgets.VBox([widgets.HTML('<h3>Model D — null-position control test</h3>'),dd,widgets.HBox([prevb,nextb]),cap,img]))
    files=[scsv,ccsv]+figs
    print(f'CODE OUTPUT: {VERSION}')
    print(summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nIMAGE INDEX')
    for i,p in enumerate(figs,1): print(f'{i:2d}. {p.name}')
    print(f'Images in gallery: {len(figs)}'); print(f'Drive cube: {CUBE}'); print(upload(files))
    print('Timestamp Colombia:',datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z')); print(f'# {VERSION}')

if __name__=='__main__': main()
