# JWST_0176
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
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

x=ns['x']; y=ns['y']; s=ns['s']; model_emp=ns['best']; z_emp=float(ns['zbest'])
LYA=float(ns['LYA']); ZP=float(ns['ZP']); PNG=ns['PNG']; CSV=ns['CSV']
VERSION='JWST_0176'
ZSYS=11.1221; ZSYS_SIG=0.0006; LYA_SYS=LYA*(1+ZSYS)
H=6.62607015e-34; KB=1.380649e-23; C=2.99792458e8; C_KMS=299792.458
NU_REST=3393.006244; NU_TABLE=279.901; NU_TABLE_SIG=0.014
NU_EXACT=NU_REST/(1+ZSYS); Z_TABLE=NU_REST/NU_TABLE-1
R=NU_TABLE/NU_EXACT
DV_REL=C_KMS*((R*R)-1)/((R*R)+1)
DNU_MHZ=(NU_EXACT-NU_TABLE)*1e3
DZ=Z_TABLE-ZSYS

for d in (PNG,CSV): Path(d).mkdir(parents=True,exist_ok=True)

def planck(lam_um,temp):
    lm=np.maximum(lam_um,1e-6)*1e-6
    expo=np.clip(H*C/(lm*KB*temp),1e-8,700)
    return (2*H*C*C/lm**5)/(np.exp(expo)-1)

def template(temp,nebfrac,lognhi):
    rest=x/(1+ZSYS)
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
    return ns['gaussian_filter1d'](raw,sig,mode='nearest')

def fit_one(temp,nf,lnhi):
    u=template(temp,nf,lnhi); wt=1/s**2
    den=np.sum(wt*u*u)
    if not np.isfinite(den) or den<=0: return 1e99,0.0,u
    amp=np.sum(wt*y*u)/den
    chi=np.sum(((y-amp*u)/s)**2)
    return float(chi),float(amp),u

temps=np.array([12000.,16000.,20000.,24000.,30000.])
nebfracs=np.array([0.0,0.15,0.30,0.45])
lognhis=np.array([21.8,22.1,22.4,22.7])
rows=[]; best=None
for t in temps:
    for nf in nebfracs:
        for ln in lognhis:
            chi,amp,u=fit_one(t,nf,ln)
            rows.append((t,nf,ln,chi))
            if best is None or chi<best[0]: best=(chi,t,nf,ln,amp,u.copy())
chi_sys,tbest,nfbest,lnbest,ampbest,ubest=best
model_sys=ampbest*ubest
resid_sys=(y-model_sys)/s; resid_emp=(y-model_emp)/s
lya_emp=LYA*(1+z_emp); lya_paper=LYA*(1+ZP)

nu=np.linspace(279.84,279.96,1000)
line=np.exp(-0.5*((nu-NU_EXACT)/NU_TABLE_SIG)**2)

summary=pd.DataFrame([
    ['Official ALMA systemic',ZSYS,LYA_SYS,NU_EXACT,np.nan,np.nan],
    ['Rounded published centroid',Z_TABLE,LYA*(1+Z_TABLE),NU_TABLE,DNU_MHZ,DV_REL],
    ['Empirical continuum break',z_emp,lya_emp,np.nan,np.nan,np.nan],
    ['Published SMDS fit',ZP,lya_paper,np.nan,np.nan,np.nan],
],columns=['method','redshift_z','lya_reference_um','observed_OIII_frequency_GHz','centroid_correction_MHz','equivalent_velocity_km_s'])
summary.to_csv(Path(CSV)/f'{VERSION}_FULL_METHOD_COMPARISON.csv',index=False)
pd.DataFrame(rows,columns=['temperature_K','nebular_fraction','logNHI_cm2','chi2']).to_csv(Path(CSV)/f'{VERSION}_FIXED_SYSTEMIC_GRID.csv',index=False)
pd.DataFrame({'wavelength_um':x,'flux':y,'sigma':s,'empirical_model':model_emp,'systemic_surrogate':model_sys,'empirical_residual_sigma':resid_emp,'systemic_residual_sigma':resid_sys}).to_csv(Path(CSV)/f'{VERSION}_FULL_SPECTRUM.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Plot full corrected analysis',button_style='success',layout=widgets.Layout(width='290px'))
out=widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,21))
        gs=fig.add_gridspec(4,1,height_ratios=[2.7,1.0,1.0,1.35],hspace=.22)
        a1,a2,a3,a4=[fig.add_subplot(gs[i]) for i in range(4)]
        for a in (a1,a2,a3,a4): a.grid(color='#303944',lw=.6,alpha=.75)

        m=(x>=1.15)&(x<=2.30)
        a1.fill_between(x[m],y[m]-s[m],y[m]+s[m],step='mid',color='gray',alpha=.28,label='1σ uncertainty')
        a1.step(x[m],y[m],where='mid',color='#4f79b9',lw=1,label='Validated JADES PRISM spectrum')
        a1.plot(x[m],model_sys[m],color='#62dfd1',lw=2.4,label=f'Systemic-anchored H/He + nebular + DLA surrogate, z={ZSYS:.4f}')
        a1.plot(x[m],model_emp[m],color='#e68645',lw=1.9,label=f'Empirical continuum-break model, z={z_emp:.5f}')
        a1.axvline(LYA_SYS,color='#8fd3ff',ls='-.',lw=1.9,label=f'Official systemic Lyα reference = {LYA_SYS:.6f} µm')
        a1.axvline(lya_emp,color='#e68645',ls=':',lw=1.2,label=f'Empirical break anchor = {lya_emp:.6f} µm')
        a1.axvline(lya_paper,color='#ff6b6b',ls='--',lw=1.2,label=f'Paper SMDS anchor = {lya_paper:.6f} µm')
        # Full-spectrum line/reference labels used in the calculations.
        for lam,lab,dy in [(LYA_SYS,'Lyα systemic reference',.90),(lya_emp,'Empirical break',.78),(lya_paper,'SMDS-model break',.66)]:
            a1.annotate(lab,xy=(lam,dy),xycoords=('data','axes fraction'),xytext=(8,0),textcoords='offset points',rotation=90,va='top',ha='left',fontsize=8)
        a1.set_xlim(1.15,2.30); q=np.nanpercentile(y[m],[1,99]); pad=.2*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad)
        a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
        a1.legend(frameon=False,ncol=2,fontsize=8.5)
        a1.set_title('Full JWST/NIRSpec spectrum and fixed-systemic H/He + nebular + DLA model')

        a2.axhline(0,color='white',lw=.6)
        a2.step(x[m],resid_emp[m],where='mid',color='#e68645',lw=.9,label='Empirical residual / σ')
        a2.step(x[m],resid_sys[m],where='mid',color='#62dfd1',lw=.9,label='Systemic-anchored surrogate residual / σ')
        a2.axhspan(-1,1,color='gray',alpha=.16); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6)
        a2.set_ylabel('Residual / σ'); a2.legend(frameon=False,ncol=2)

        labels=['Official systemic','Empirical break','Paper SMDS']; vals=[ZSYS,z_emp,ZP]
        a3.scatter(vals,[2,1,0],s=90,color=['#8fd3ff','#e68645','#ff6b6b'])
        for yy,v,lab,col,off in zip([2,1,0],vals,labels,['#8fd3ff','#e68645','#ff6b6b'],[(18,14),(18,0),(18,-14)]):
            a3.annotate(f'{lab}: z={v:.5f}',xy=(v,yy),xytext=off,textcoords='offset points',va='center',fontsize=10,color=col,arrowprops=dict(arrowstyle='-',lw=.8,color=col))
        a3.set_yticks([]); a3.set_xlim(min(vals)-.04,max(vals)+.08); a3.set_xlabel('Redshift z')
        a3.set_title(f'Fixed-systemic fit parameters: T={tbest:.0f} K, nebular fraction={nfbest:.2f}, log N(H I)={lnbest:.2f}')

        a4.plot(nu,line,color='#7fd5cb',lw=2.2,label='Reconstructed [O III] 88 μm line profile')
        a4.axvline(NU_TABLE,color='#ff5a36',ls='--',lw=2.0,label=f'Published rounded centroid {NU_TABLE:.6f} GHz')
        a4.axvline(NU_EXACT,color='#ffb347',ls='-.',lw=2.0,label=f'Official-z implied centroid {NU_EXACT:.6f} GHz')
        a4.axvspan(NU_TABLE-NU_TABLE_SIG,NU_TABLE+NU_TABLE_SIG,color='#ff5a36',alpha=.10,label='Published ±0.014 GHz')
        a4.set_xlim(279.84,279.96); a4.set_ylim(-.03,1.08)
        a4.set_xlabel('Observed [O III] 88 μm frequency [GHz]'); a4.set_ylabel('Normalized line amplitude')
        a4.legend(frameon=False,fontsize=8.5,ncol=2)
        txt=(f'Rounded centroid → z={Z_TABLE:.9f}\nOfficial z={ZSYS:.7f} → ν={NU_EXACT:.9f} GHz\n'
             f'Δν={DNU_MHZ:.6f} MHz   Δz={DZ:.9f}   equivalent offset={DV_REL:.6f} km/s')
        a4.text(.02,.95,txt,transform=a4.transAxes,va='top',fontsize=10,
                bbox=dict(boxstyle='round,pad=.45',facecolor='#111820',edgecolor='#72808e',alpha=.95))
        a4.set_title('ALMA [O III] centroid rounding correction and equivalent velocity offset')

        fig.suptitle('JADES-GS-z11-0 — full-spectrum systemic-redshift analysis with ALMA centroid correction\nAll JWST panels retained; ALMA correction added as a fourth panel',fontsize=15,y=.996)
        outpng=Path(PNG)/f'{VERSION}_FULL_SPECTRUM_ALMA_VELOCITY_CORRECTED.png'
        fig.savefig(outpng,dpi=500,bbox_inches='tight')
        plt.show()
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False,float_format=lambda v:f'{v:.9f}'))
        print(f'PNG: {outpng}')
        print(f'CSV: {Path(CSV)/f"{VERSION}_FULL_METHOD_COMPARISON.csv"}')
        print(f'# {VERSION}')

run.on_click(draw); display(run,out); draw()
