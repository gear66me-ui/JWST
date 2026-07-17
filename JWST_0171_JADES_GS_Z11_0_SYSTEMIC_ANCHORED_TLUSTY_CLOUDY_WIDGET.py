# JWST_0171
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
BASE='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0165_JADES_GS_Z11_0_HLSP_VALIDATED_PRODUCT_WIDGET.py'
source=requests.get(BASE,timeout=120).text
ns={}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(source,'JWST_0165_runtime.py','exec'),ns)

np=ns['np']; pd=ns['pd']; plt=ns['plt']; widgets=ns['widgets']; display=ns['display']; clear_output=ns['clear_output']
PNG=ns['PNG']; CSV=ns['CSV']; x=ns['x']; y=ns['y']; s=ns['s']; model_emp=ns['best']; z_emp=float(ns['zbest']); LYA=float(ns['LYA']); ZP=float(ns['ZP'])
VERSION='JWST_0171'; ZSYS=11.1221; LYA_SYS=LYA*(1+ZSYS)
H=6.62607015e-34; KB=1.380649e-23; C=2.99792458e8

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

summary=pd.DataFrame([
    ['Official ALMA [O III] systemic','reported',ZSYS,LYA_SYS,np.nan,np.nan,np.nan],
    ['Systemic-anchored H/He+nebular+DLA surrogate','physical surrogate fixed-z',ZSYS,LYA_SYS,tbest,nfbest,lnbest],
    ['Empirical continuum-break fit','empirical',z_emp,lya_emp,np.nan,np.nan,np.nan],
    ['Published SMDS paper fit','reported',ZP,lya_paper,np.nan,np.nan,np.nan],
],columns=['method','status','redshift_z','lya_reference_um','temperature_K','nebular_fraction','logNHI_cm2'])
summary.to_csv(CSV/f'{VERSION}_METHOD_COMPARISON.csv',index=False)
pd.DataFrame(rows,columns=['temperature_K','nebular_fraction','logNHI_cm2','chi2']).to_csv(CSV/f'{VERSION}_FIXED_SYSTEMIC_GRID.csv',index=False)
pd.DataFrame({'wavelength_um':x,'flux':y,'sigma':s,'empirical_model':model_emp,'systemic_anchored_surrogate':model_sys,'empirical_residual_sigma':resid_emp,'systemic_surrogate_residual_sigma':resid_sys}).to_csv(CSV/f'{VERSION}_SPECTRAL_COMPARISON.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Plot corrected systemic fit',button_style='success',layout=widgets.Layout(width='270px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(16,15)); gs=fig.add_gridspec(3,1,height_ratios=[2.5,1.0,1.0],hspace=.18)
        a1,a2,a3=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a1,a2,a3): a.grid(color='#303944',lw=.6,alpha=.75)
        m=(x>=1.15)&(x<=2.30)
        a1.fill_between(x[m],y[m]-s[m],y[m]+s[m],step='mid',color='gray',alpha=.28,label='1σ uncertainty')
        a1.step(x[m],y[m],where='mid',color='#4f79b9',lw=1,label='Validated JADES PRISM spectrum')
        a1.plot(x[m],model_sys[m],color='#62dfd1',lw=2.4,label=f'Systemic-anchored H/He + nebular + DLA surrogate, z={ZSYS:.4f}')
        a1.plot(x[m],model_emp[m],color='#e68645',lw=1.9,label=f'Empirical continuum-break model, z={z_emp:.5f}')
        a1.axvline(LYA_SYS,color='#8fd3ff',ls='-.',lw=1.9,label=f'Official systemic Lyα reference = {LYA_SYS:.6f} µm')
        a1.axvline(lya_emp,color='#e68645',ls=':',lw=1.2,label=f'Empirical break anchor = {lya_emp:.6f} µm')
        a1.axvline(lya_paper,color='#ff6b6b',ls='--',lw=1.2,label=f'Paper SMDS anchor = {lya_paper:.6f} µm')
        a1.set_xlim(1.15,2.30); q=np.nanpercentile(y[m],[1,99]); pad=.2*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad)
        a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=8.7)
        a2.axhline(0,color='white',lw=.6); a2.step(x[m],resid_emp[m],where='mid',color='#e68645',lw=.9,label='Empirical residual / σ'); a2.step(x[m],resid_sys[m],where='mid',color='#62dfd1',lw=.9,label='Systemic-anchored surrogate residual / σ'); a2.axhspan(-1,1,color='gray',alpha=.16); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False,ncol=2)
        labels=['Official systemic','Empirical break','Paper SMDS']; vals=[ZSYS,z_emp,ZP]
        a3.scatter(vals,[2,1,0],s=90)
        for yy,v,lab in zip([2,1,0],vals,labels): a3.text(v+.004,yy,f'{lab}: z={v:.5f}',va='center',fontsize=10)
        a3.set_yticks([]); a3.set_xlim(min(vals)-.04,max(vals)+.08); a3.set_xlabel('Redshift z')
        a3.set_title(f'Fixed systemic redshift; best surrogate parameters: T={tbest:.0f} K, nebular fraction={nfbest:.2f}, log N(H I)={lnbest:.2f}')
        fig.suptitle('JADES-GS-z11-0 — corrected systemic-anchored TLUSTY→CLOUDY-method surrogate\nNo free redshift optimization: z fixed to official ALMA [O III] value 11.1221',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_SYSTEMIC_ANCHORED_COMPARISON.png',dpi=500,bbox_inches='tight')
        plt.show()
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
        print(f'PNG: {PNG/f"{VERSION}_SYSTEMIC_ANCHORED_COMPARISON.png"}')
        print(f'CSV: {CSV/f"{VERSION}_METHOD_COMPARISON.csv"}')
        print(f'# {VERSION}')
run.on_click(draw); display(run,out); draw()
