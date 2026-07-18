from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc, numpy as np, pandas as pd, matplotlib.pyplot as plt
from astropy.io import fits

VERSION='JWST_0201'; TARGET=279.901000
ROOT=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE=ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
OUTPNG=Path('/content/JWST_OUTPUT/PNG'); OUTCSV=Path('/content/JWST_OUTPUT/CSV')
PPNG=ROOT/'PNG'; PCSV=ROOT/'CSV'
for p in (OUTPNG,OUTCSV,PPNG,PCSV): p.mkdir(parents=True,exist_ok=True)

def rsig(x):
 x=np.asarray(x,float); m=np.nanmedian(x); d=np.nanmedian(np.abs(x-m)); return 1.4826*d if d>0 else np.nanstd(x)

def main():
 print(f'CODE OUTPUT: {VERSION}'); print('Drive mount reused; no remount requested.')
 if not CUBE.exists(): raise FileNotFoundError(CUBE)
 print(f'Cube: {CUBE}\nCube size: {CUBE.stat().st_size/1024**3:.2f} GB')
 with fits.open(CUBE,memmap=True) as h:
  q=next(z for z in h if getattr(z,'data',None) is not None and z.data.ndim>=3)
  a=q.data; H=q.header; sh=a.shape; na=int(H.get('NAXIS',len(sh)))
  fa=next(i for i in range(1,na+1) if 'FREQ' in str(H.get(f'CTYPE{i}','')).upper())
  npax=len(sh)-fa; n=int(H[f'NAXIS{fa}']); crv=float(H[f'CRVAL{fa}']); crp=float(H[f'CRPIX{fa}']); cdl=float(H.get(f'CDELT{fa}',H.get(f'CD{fa}_{fa}')))
  f=(crv+(np.arange(n)+1-crp)*cdl)/1e9
  a=np.moveaxis(a,npax,0)
  while a.ndim>3:
   k=next((i for i,s in enumerate(a.shape[1:],1) if s==1),None)
   if k is None: raise RuntimeError(f'Unexpected shape {a.shape}')
   a=np.take(a,0,axis=k)
  nc,ny,nx=a.shape; x0=int(np.clip(round(float(H.get('CRPIX1',(nx+1)/2))-1),0,nx-1)); y0=int(np.clip(round(float(H.get('CRPIX2',(ny+1)/2))-1),0,ny-1))
  pix=abs(float(H.get('CDELT1',1/3600)))*3600; beam=np.nanmean([float(H.get('BMAJ',np.nan))*3600,float(H.get('BMIN',np.nan))*3600]); r=max(2.0,0.60*beam/pix) if np.isfinite(beam) else 3.0
  hlf=int(np.ceil(r))+1; ys=slice(max(0,y0-hlf),min(ny,y0+hlf+1)); xs=slice(max(0,x0-hlf),min(nx,x0+hlf+1))
  yy,xx=np.mgrid[ys.start:ys.stop,xs.start:xs.stop]; mask=(xx-x0)**2+(yy-y0)**2<=r*r
  s=np.empty(nc,float)
  for i in range(0,nc,32):
   slab=np.asarray(a[i:min(i+32,nc),ys,xs],dtype=np.float32); s[i:min(i+32,nc)]=np.nanmean(slab[:,mask],axis=1); del slab
   if i%320==0: print(f'Aperture extraction: {min(i+32,nc)}/{nc} channels')
  del a; gc.collect()
 o=np.argsort(f); f=f[o]; s=s[o]; broad=np.abs(f-TARGET)<=0.45; line=np.abs(f-TARGET)<=0.12; base=broad&(~line)
 p=np.polyfit(f[base]-TARGET,s[base],1); res=s-np.polyval(p,f-TARGET); native_rms=rsig(res[base])
 edges=np.arange(TARGET-0.4525,TARGET+0.4525+0.005,0.005); cen=(edges[:-1]+edges[1:])/2; rb=np.full(cen.size,np.nan); nn=np.zeros(cen.size,int)
 for j in range(cen.size):
  m=(f>=edges[j])&(f<edges[j+1]); nn[j]=m.sum(); rb[j]=np.nanmean(res[m]) if m.any() else np.nan
 lm=np.abs(cen-TARGET)<=0.12; rms=rsig(rb[~lm]); w=np.clip(rb[lm],0,None); cent=np.sum(cen[lm]*w)/np.sum(w) if np.sum(w)>0 else np.nan; ps=np.nanmax(rb[lm])/rms; ins=np.nansum(rb[lm])/(rms*np.sqrt(np.sum(np.isfinite(rb[lm]))))
 df=pd.DataFrame({'frequency_GHz':cen,'velocity_offset_km_s':299792.458*(TARGET-cen)/TARGET,'continuum_subtracted_mean_Jy_per_beam':rb,'estimated_rms_Jy_per_beam':rms,'snr':rb/rms,'native_channels':nn,'inside_centroid_window':lm})
 cn=f'{VERSION}_SPW23_5MHZ_DIAGNOSTIC_SPECTRUM.csv'
 for d in (OUTCSV/cn,PCSV/cn): df.to_csv(d,index=False)
 plt.figure(figsize=(12,7)); ax=plt.gca(); ax.set_facecolor('#08111d'); plt.gcf().patch.set_facecolor('#08111d'); ax.plot(cen,rb*1e3,lw=1.2,label='5 MHz aperture spectrum'); ax.axvline(TARGET,ls='--',lw=1.1,label='279.901 GHz reference')
 if np.isfinite(cent): ax.axvline(cent,ls=':',lw=1.2,label=f'Flux-weighted centroid {cent:.6f} GHz')
 ax.axhline(0,lw=.8); ax.fill_between(cen,-rms*1e3,rms*1e3,alpha=.15,label='±1σ RMS'); ax.set(xlim=(TARGET-.45,TARGET+.45),xlabel='Observed frequency (GHz)',ylabel='Continuum-subtracted aperture mean (mJy beam⁻¹)',title='JADES-GS-z11-0 — ALMA SPW23 memory-safe diagnostic'); ax.grid(alpha=.18); ax.legend(); ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white'); ax.title.set_color('white')
 for z in ax.spines.values(): z.set_color('#8fa3b8')
 pn=f'{VERSION}_SPW23_5MHZ_DIAGNOSTIC_SPECTRUM.png'; plt.tight_layout()
 for d in (OUTPNG/pn,PPNG/pn): plt.savefig(d,dpi=180,bbox_inches='tight')
 plt.show(); plt.close()
 status='candidate signal' if np.isfinite(ps) and ps>=5 else 'diagnostic only; no >=5 sigma peak'
 rows=[['Cube shape',f'{nc} x {ny} x {nx}','observed data product'],['Frequency coverage',f'{f.min():.6f} to {f.max():.6f} GHz','FITS WCS'],['Native spacing',f'{np.nanmedian(np.diff(f))*1000:.6f} MHz','FITS WCS'],['Pixel scale',f'{pix:.4f} arcsec','FITS WCS'],['Beam FWHM',f'{beam:.4f} arcsec','FITS header'],['Aperture radius',f'{r:.2f} pixels','derived'],['Native RMS',f'{native_rms*1e3:.6f} mJy beam^-1','derived'],['5 MHz RMS',f'{rms*1e3:.6f} mJy beam^-1','derived'],['Peak SNR',f'{ps:.3f}','derived'],['Integrated-window SNR',f'{ins:.3f}','derived'],['Flux-weighted centroid',f'{cent:.6f} GHz' if np.isfinite(cent) else 'undefined','derived'],['Status',status,'diagnostic classification']]
 print('\n'+pd.DataFrame(rows,columns=['quantity','value','classification']).to_string(index=False)); print(f'\nCSV: {OUTCSV/cn}\nPlot: {OUTPNG/pn}\nPersistent CSV: {PCSV/cn}\nPersistent plot: {PPNG/pn}'); print('Timestamp Colombia: '+datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S -0500')); print(f'# {VERSION}')
if __name__=='__main__': main()
