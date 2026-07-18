# JWST_0181
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

BASE='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0165_JADES_GS_Z11_0_HLSP_VALIDATED_PRODUCT_WIDGET.py'
source=requests.get(BASE,timeout=120).text
ns={}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(source,'JWST_0165_runtime.py','exec'),ns)

x=ns['x']; y=ns['y']; s=ns['s']; PNG=Path(ns['PNG']); CSV=Path(ns['CSV'])
LYA=float(ns['LYA']); gf=ns['gaussian_filter1d']
VERSION='JWST_0181'
Z_OFFICIAL=11.1221000; Z_SIGMA=0.0006000
NU_REST_GHZ=3393.006244; NU_ROUNDED_GHZ=279.901000; NU_SIGMA_GHZ=0.014000
C_KMS=299792.458
Z_DERIVED=NU_REST_GHZ/NU_ROUNDED_GHZ-1.0
NU_OFFICIAL_GHZ=NU_REST_GHZ/(1.0+Z_OFFICIAL)
LYA_DERIVED=LYA*(1.0+Z_DERIVED); LYA_OFFICIAL=LYA*(1.0+Z_OFFICIAL)
DELTA_Z=Z_DERIVED-Z_OFFICIAL
DELTA_NU_MHZ=(NU_OFFICIAL_GHZ-NU_ROUNDED_GHZ)*1000.0
R=NU_ROUNDED_GHZ/NU_OFFICIAL_GHZ
DV_EQ=C_KMS*((R*R)-1.0)/((R*R)+1.0)
DELTA_LYA_NM=(LYA_DERIVED-LYA_OFFICIAL)*1000.0
for d in (PNG,CSV): d.mkdir(parents=True,exist_ok=True)

H=6.62607015e-34; KB=1.380649e-23; C=2.99792458e8
def planck(lam_um,temp):
    lm=np.maximum(lam_um,1e-6)*1e-6
    e=np.clip(H*C/(lm*KB*temp),1e-8,700)
    return (2*H*C*C/lm**5)/(np.exp(e)-1)

def template(temp,nebfrac,lognhi):
    rest=x/(1.0+Z_OFFICIAL)
    stellar=planck(rest,temp)
    norm=(rest>0.135)&(rest<0.18)
    stellar/=np.nanmedian(stellar[norm])
    neb=(np.maximum(rest,0.0912)/0.15)**-1.15
    neb*=np.exp(-np.maximum(rest-0.40,0)/0.55)
    intrinsic=(1-nebfrac)*stellar+nebfrac*neb
    blue=np.where(rest<LYA,np.exp(-80*np.maximum((LYA-rest)/LYA,0)**0.75),1.0)
    strength=10**(lognhi-22.0)
    red=np.where(rest>=LYA,np.exp(-0.020*strength/np.maximum(rest-LYA+0.0015,0.0015)),1.0)
    raw=intrinsic*blue*red
    d=np.diff(x); d=d[np.isfinite(d)&(d>0)]
    sig=max(float(np.nanmedian(x)/(100*2.35482*np.nanmedian(d))),0.35)
    return gf(raw,sig,mode='nearest')

def fit_one(temp,nf,lnhi):
    u=template(temp,nf,lnhi); wt=1/s**2
    den=np.sum(wt*u*u)
    if not np.isfinite(den) or den<=0: return 1e99,0.0,u
    amp=np.sum(wt*y*u)/den
    chi=np.sum(((y-amp*u)/s)**2)
    return float(chi),float(amp),u

best=None; rows=[]
for t in [12000.,16000.,20000.,24000.,30000.]:
    for nf in [0.0,0.15,0.30,0.45]:
        for ln in [21.8,22.1,22.4,22.7]:
            chi,amp,u=fit_one(t,nf,ln); rows.append((t,nf,ln,chi))
            if best is None or chi<best[0]: best=(chi,t,nf,ln,amp,u.copy())
chi,tbest,nfbest,lnbest,ampbest,ubest=best
model=ampbest*ubest; resid=(y-model)/s
nu=np.linspace(279.84,279.96,1200)
line=np.exp(-0.5*((nu-NU_OFFICIAL_GHZ)/NU_SIGMA_GHZ)**2)

summary=pd.DataFrame([
    ['Our rounded-centroid derivation',NU_ROUNDED_GHZ,Z_DERIVED,LYA_DERIVED,DELTA_Z,DELTA_LYA_NM,0.0],
    ['Official ALMA full-precision result',NU_OFFICIAL_GHZ,Z_OFFICIAL,LYA_OFFICIAL,0.0,0.0,DV_EQ],
],columns=['method','OIII_frequency_GHz','redshift_z','lya_reference_um','delta_z','delta_lya_nm','equivalent_offset_km_s'])
summary.to_csv(CSV/f'{VERSION}_RESULTS.csv',index=False)
pd.DataFrame(rows,columns=['temperature_K','nebular_fraction','logNHI_cm2','chi2']).to_csv(CSV/f'{VERSION}_SYSTEMIC_GRID.csv',index=False)
pd.DataFrame({'wavelength_um':x,'flux':y,'sigma':s,'systemic_model':model,'residual_sigma':resid}).to_csv(CSV/f'{VERSION}_FULL_SPECTRUM.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})

def table_png():
    fig,ax=plt.subplots(figsize=(16,4.8)); ax.axis('off')
    show=pd.DataFrame({
        'Quantity':['[O III] centroid','Systemic redshift','Lyα reference','Difference from official','Equivalent centroid offset'],
        'Our derivation':[f'{NU_ROUNDED_GHZ:.6f} GHz',f'{Z_DERIVED:.9f}',f'{LYA_DERIVED:.9f} µm',f'{DELTA_Z:.9f}',f'{DV_EQ:.6f} km/s'],
        'Official ALMA':[f'{NU_OFFICIAL_GHZ:.9f} GHz',f'{Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}',f'{LYA_OFFICIAL:.9f} µm','reference','reference'],
        'Difference / note':[f'{DELTA_NU_MHZ:.6f} MHz',f'{DELTA_Z:.9f}',f'{DELTA_LYA_NM:.6f} nm','rounded-frequency effect','not peculiar velocity']})
    tb=ax.table(cellText=show.values,colLabels=show.columns,loc='center',cellLoc='left',colLoc='left',bbox=[.01,.05,.98,.82])
    tb.auto_set_font_size(False); tb.set_fontsize(10.5)
    for (r,c),cell in tb.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_linewidth(.8); cell.set_facecolor('#111820' if r else '#1c2b38'); cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
    ax.set_title('JADES-GS-z11-0 — ALMA-only systemic-redshift cross-reference',fontsize=15,pad=14)
    p=PNG/f'{VERSION}_RESULTS_TABLE.png'; fig.savefig(p,dpi=500,bbox_inches='tight'); plt.close(fig); return p

run=widgets.Button(description='Plot full ALMA-only verification',button_style='success',layout=widgets.Layout(width='310px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,21)); gs=fig.add_gridspec(4,1,height_ratios=[2.7,1.0,1.0,1.35],hspace=.23)
        a1,a2,a3,a4=[fig.add_subplot(gs[i]) for i in range(4)]
        for a in (a1,a2,a3,a4): a.grid(color='#303944',lw=.6,alpha=.75)
        m=(x>=1.15)&(x<=2.30)
        a1.fill_between(x[m],y[m]-s[m],y[m]+s[m],step='mid',color='gray',alpha=.28,label='1σ uncertainty')
        a1.step(x[m],y[m],where='mid',color='#4f79b9',lw=1,label='Validated JADES PRISM spectrum')
        a1.plot(x[m],model[m],color='#62dfd1',lw=2.4,label=f'Fixed-systemic H/He + nebular + DLA model, z={Z_OFFICIAL:.7f}')
        a1.axvline(LYA_OFFICIAL,color='#ffb347',ls='-.',lw=1.9,label=f'Official Lyα reference = {LYA_OFFICIAL:.9f} µm')
        a1.axvline(LYA_DERIVED,color='#ff5a36',ls='--',lw=1.6,label=f'Our Lyα verification = {LYA_DERIVED:.9f} µm')
        a1.set_xlim(1.15,2.30); q=np.nanpercentile(y[m],[1,99]); pad=.2*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad)
        a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=8.8)
        a1.set_title(f'Full JWST/NIRSpec systemic-redshift analysis — our rounded-centroid derivation z={Z_DERIVED:.9f}')
        a2.axhline(0,color='white',lw=.6); a2.step(x[m],resid[m],where='mid',color='#62dfd1',lw=.95,label='Fixed-systemic model residual / σ')
        a2.axhspan(-1,1,color='gray',alpha=.16); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False)
        vals=[Z_OFFICIAL,Z_DERIVED]; ypos=[0,1]; cols=['#ffb347','#ff5a36']
        a3.scatter(vals,ypos,s=110,color=cols,zorder=4)
        a3.annotate(f'Official ALMA\nz={Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}',xy=(Z_OFFICIAL,0),xytext=(-24,24),textcoords='offset points',ha='right',va='bottom',color=cols[0],arrowprops=dict(arrowstyle='-',color=cols[0]))
        a3.annotate(f'Our derivation\nz={Z_DERIVED:.9f}\nLyα={LYA_DERIVED:.9f} µm',xy=(Z_DERIVED,1),xytext=(24,-28),textcoords='offset points',ha='left',va='top',color=cols[1],arrowprops=dict(arrowstyle='-',color=cols[1]))
        a3.set_yticks([]); a3.set_xlim(min(vals)-.0009,max(vals)+.0009); a3.set_xlabel('Systemic redshift z')
        a3.set_title(f'Fixed-systemic parameters: T={tbest:.0f} K, nebular fraction={nfbest:.2f}, log N(H I)={lnbest:.2f}')
        a4.plot(nu,line,color='#7fd5cb',lw=2.3,label='Reconstructed ALMA [O III] 88 µm line profile')
        a4.axvline(NU_ROUNDED_GHZ,color='#ff5a36',ls='--',lw=2,label=f'Rounded published centroid {NU_ROUNDED_GHZ:.6f} GHz')
        a4.axvline(NU_OFFICIAL_GHZ,color='#ffb347',ls='-.',lw=2,label=f'Official-z implied centroid {NU_OFFICIAL_GHZ:.9f} GHz')
        a4.axvspan(NU_ROUNDED_GHZ-NU_SIGMA_GHZ,NU_ROUNDED_GHZ+NU_SIGMA_GHZ,color='#ff5a36',alpha=.10,label='Published ±0.014 GHz')
        a4.set_xlim(279.84,279.96); a4.set_ylim(-.03,1.08); a4.set_xlabel('Observed [O III] 88 µm frequency [GHz]'); a4.set_ylabel('Normalized line amplitude')
        a4.legend(frameon=False,fontsize=8.5,ncol=1,loc='upper left')
        txt=(f'Our z={Z_DERIVED:.9f}\nOfficial z={Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}\nΔz={DELTA_Z:.9f}\nΔν={DELTA_NU_MHZ:.6f} MHz\nEquivalent offset={DV_EQ:.6f} km/s')
        a4.text(.02,.60,txt,transform=a4.transAxes,va='top',fontsize=10,bbox=dict(boxstyle='round,pad=.45',facecolor='#111820',edgecolor='#72808e',alpha=.95))
        a4.set_title('ALMA [O III] centroid verification — rounded-table derivation versus official full precision')
        fig.suptitle('JADES-GS-z11-0 — ALMA-only full-spectrum systemic-redshift verification',fontsize=16,y=.997)
        main=PNG/f'{VERSION}_FULL_SPECTRUM_ALMA_ONLY.png'; fig.savefig(main,dpi=500,bbox_inches='tight'); plt.show()
        tab=table_png(); display(plt.imread(tab))
        print(f'CODE OUTPUT: {VERSION}')
        print(f'Our derived redshift: z = {Z_DERIVED:.9f}')
        print(f'Official ALMA redshift: z = {Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}')
        print(f'Our derived Lyα reference: {LYA_DERIVED:.9f} µm')
        print(f'Official Lyα reference: {LYA_OFFICIAL:.9f} µm')
        print(f'Lyα difference: {DELTA_LYA_NM:.6f} nm')
        print('No galaxy peculiar-velocity subtraction was applied; the calculated km/s value is only the centroid-rounding equivalent.')
        print(f'PNG: {main}'); print(f'TABLE PNG: {tab}'); print(f'CSV: {CSV/f"{VERSION}_RESULTS.csv"}')
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')
run.on_click(draw); display(run,out); draw()
