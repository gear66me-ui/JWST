# JWST_0168
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
BASE='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0165_JADES_GS_Z11_0_HLSP_VALIDATED_PRODUCT_WIDGET.py'
source=requests.get(BASE,timeout=120).text
ns={}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(source,'JWST_0165_runtime.py','exec'),ns)

np=ns['np']; pd=ns['pd']; plt=ns['plt']; widgets=ns['widgets']; display=ns['display']; clear_output=ns['clear_output']
PNG=ns['PNG']; CSV=ns['CSV']; x=ns['x']; y=ns['y']; s=ns['s']; z_emp=float(ns['zbest']); ZP=float(ns['ZP']); LYA=float(ns['LYA'])
VERSION='JWST_0168'

# Paper-informed H/He stellar-atmosphere + nebular surrogate.
# This is NOT the authors' unreleased TLUSTY/CLOUDY grid; it reproduces the method structure:
# primordial stellar continuum -> H/He nebular continuum -> IGM/DLA attenuation -> NIRSpec convolution.
C_UM_S=2.99792458e14; H=6.62607015e-34; KB=1.380649e-23

def planck_lambda_um(lam_um,temp):
    lam_m=np.maximum(lam_um,1e-6)*1e-6
    expo=np.clip(H*2.99792458e8/(lam_m*KB*temp),1e-8,700)
    return (2*H*(2.99792458e8**2)/lam_m**5)/(np.exp(expo)-1)

def physical_template(z,temp,nebfrac,lognhi):
    rest=x/(1+z)
    stellar=planck_lambda_um(rest,temp)
    stellar/=np.nanmedian(stellar[(rest>0.135)&(rest<0.18)])
    # Smooth primordial H/He nebular continuum surrogate: flatter free-bound/free-free tail.
    neb=(np.maximum(rest,0.0912)/0.15)**-1.15
    neb*=np.exp(-np.maximum(rest-0.40,0)/0.55)
    intrinsic=(1-nebfrac)*stellar+nebfrac*neb
    # Ly-alpha forest/Gunn-Peterson cutoff plus red damping wing controlled by N_HI.
    edge=LYA
    blue=np.where(rest<edge,np.exp(-80*np.maximum((edge-rest)/edge,0)**0.75),1.0)
    strength=10**(lognhi-22.0)
    redwing=np.where(rest>=edge,np.exp(-0.020*strength/np.maximum(rest-edge+0.0015,0.0015)),1.0)
    trans=blue*redwing
    raw=intrinsic*trans
    d=np.diff(x); d=d[np.isfinite(d)&(d>0)]
    dl=float(np.nanmedian(d)); sigma_pix=max(float(np.nanmedian(x)/(100*2.35482*dl)),0.35)
    return ns['gaussian_filter1d'](raw,sigma_pix,mode='nearest')

def chi_for(z,temp,nebfrac,lognhi):
    u=physical_template(z,temp,nebfrac,lognhi)
    wt=1/s**2; den=np.sum(wt*u*u)
    if not np.isfinite(den) or den<=0: return 1e99,0.0,u
    amp=np.sum(wt*y*u)/den
    val=np.sum(((y-amp*u)/s)**2)
    return float(val),float(amp),u

zgrid=np.linspace(10.90,11.55,261)
temps=np.array([12000.,16000.,20000.,24000.,30000.])
nebfracs=np.array([0.0,0.15,0.30,0.45])
lognhis=np.array([21.8,22.1,22.4,22.7])
records=[]; best_tuple=None
for temp in temps:
    for nf in nebfracs:
        for lnhi in lognhis:
            chis=[]
            for z in zgrid:
                c,a,u=chi_for(z,temp,nf,lnhi); chis.append(c)
                if best_tuple is None or c<best_tuple[0]: best_tuple=(c,z,temp,nf,lnhi,a,u.copy())
            chis=np.asarray(chis)
            records.append(pd.DataFrame({'z':zgrid,'chi2':chis,'temperature_K':temp,'nebular_fraction':nf,'logNHI_cm2':lnhi}))

chi_phys,z_phys,t_phys,nf_phys,lnhi_phys,amp_phys,u_phys=best_tuple
model_phys=amp_phys*u_phys
allgrid=pd.concat(records,ignore_index=True)
profile=allgrid.groupby('z',as_index=False)['chi2'].min()
p=np.exp(np.clip(-0.5*(profile.chi2-profile.chi2.min()),-700,0)); p/=np.trapz(p,profile.z)
c=np.cumsum(p); c/=c[-1]
z16,z50,z84=[np.interp(q,c,profile.z) for q in (0.16,0.50,0.84)]

# Empirical model from prior validated run, carried only for comparison.
model_emp=ns['best']; resid_phys=(y-model_phys)/s; resid_emp=(y-model_emp)/s
lya_phys=LYA*(1+z_phys); lya_emp=LYA*(1+z_emp); lya_paper=LYA*(1+ZP)

summary=pd.DataFrame([
    ['Empirical tanh continuum break','empirical',z_emp,np.nan,np.nan,np.nan,np.sum(resid_emp**2)],
    ['Paper-informed TLUSTY→CLOUDY surrogate','physical surrogate',z_phys,t_phys,nf_phys,lnhi_phys,chi_phys],
    ['Published dark-star paper fit','reported',ZP,np.nan,np.nan,np.nan,np.nan],
    ['Modern line+DLA spectroscopic result','reported',11.122,np.nan,np.nan,22.42,np.nan],
],columns=['method','status','redshift_z','temperature_K','nebular_fraction','logNHI_cm2','chi2'])
summary.to_csv(CSV/f'{VERSION}_METHOD_COMPARISON.csv',index=False)
allgrid.to_csv(CSV/f'{VERSION}_PHYSICAL_GRID.csv',index=False)
pd.DataFrame({'wavelength_um':x,'flux':y,'sigma':s,'empirical_model':model_emp,'physical_surrogate_model':model_phys,'empirical_residual_sigma':resid_emp,'physical_residual_sigma':resid_phys}).to_csv(CSV/f'{VERSION}_SPECTRAL_COMPARISON.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Plot TLUSTY–CLOUDY comparison',button_style='success',layout=widgets.Layout(width='290px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(16,17)); gs=fig.add_gridspec(4,1,height_ratios=[0.9,2.5,1.0,1.0],hspace=.18)
        a0,a1,a2,a3=[fig.add_subplot(gs[i]) for i in range(4)]
        for a in (a0,a1,a2,a3): a.grid(color='#303944',lw=.6,alpha=.75)
        a0.plot(profile.z,p,color='#62dfd1',lw=2.2,label=f'Physical-surrogate posterior z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})')
        a0.axvline(z_phys,color='#ffd400',lw=1.4,label=f'Best surrogate z={z_phys:.5f}')
        a0.axvline(z_emp,color='#e68645',ls='--',lw=1.2,label=f'Empirical break z={z_emp:.5f}')
        a0.axvline(11.122,color='#8fd3ff',ls=':',lw=1.2,label='Modern line+DLA z=11.122')
        a0.axvline(ZP,color='#ff6b6b',ls='--',lw=1.0,label='Paper SMDS fit z=11.38')
        a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False,ncol=2,fontsize=9)

        m=(x>=1.15)&(x<=2.30)
        a1.fill_between(x[m],y[m]-s[m],y[m]+s[m],step='mid',color='gray',alpha=.28,label='1σ uncertainty')
        a1.step(x[m],y[m],where='mid',color='#4f79b9',lw=1.0,label='Validated JADES PRISM spectrum')
        a1.plot(x[m],model_emp[m],color='#e68645',lw=2.0,label=f'Empirical continuum-break model, z={z_emp:.5f}')
        a1.plot(x[m],model_phys[m],color='#62dfd1',lw=2.3,label=f'Primordial H/He + nebular + DLA surrogate, z={z_phys:.5f}')
        a1.axvline(lya_phys,color='#ffd400',ls='--',lw=1.4,label=f'Surrogate Lyα anchor {lya_phys:.6f} µm')
        a1.axvline(lya_emp,color='#e68645',ls=':',lw=1.2,label=f'Empirical anchor {lya_emp:.6f} µm')
        a1.set_xlim(1.15,2.30); q=np.nanpercentile(y[m],[1,99]); pad=.20*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad)
        a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=8.7)

        a2.axhline(0,color='white',lw=.6); a2.step(x[m],resid_emp[m],where='mid',color='#e68645',lw=.9,label='Empirical residual / σ')
        a2.step(x[m],resid_phys[m],where='mid',color='#62dfd1',lw=.9,label='Physical-surrogate residual / σ')
        a2.axhspan(-1,1,color='gray',alpha=.16); a2.set_xlim(1.15,2.30); a2.set_ylim(-6,6); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False,ncol=2)

        labels=['Modern line+DLA','Physical surrogate','Empirical break','Paper SMDS fit']; vals=[11.122,z_phys,z_emp,ZP]
        a3.scatter(vals,[3,2,1,0],s=80)
        for yy,v,lab in zip([3,2,1,0],vals,labels): a3.text(v+.004,yy,f'{lab}: z={v:.5f}',va='center',fontsize=10)
        a3.set_yticks([]); a3.set_xlim(min(vals)-.05,max(vals)+.09); a3.set_xlabel('Redshift z')
        a3.set_title(f'Best physical-surrogate parameters: T={t_phys:.0f} K, nebular fraction={nf_phys:.2f}, log N(H I)={lnhi_phys:.2f}')
        fig.suptitle('JADES-GS-z11-0 — TLUSTY→CLOUDY method reconstruction and redshift comparison\nPhysical surrogate only: exact author TLUSTY/CLOUDY grid and fitting code are not publicly released',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_TLUSTY_CLOUDY_PHYSICAL_COMPARISON.png',dpi=500,bbox_inches='tight')
        plt.show()
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
        print(f'PNG: {PNG/f"{VERSION}_TLUSTY_CLOUDY_PHYSICAL_COMPARISON.png"}')
        print(f'CSV: {CSV/f"{VERSION}_METHOD_COMPARISON.csv"}')
        print(f'# {VERSION}')
run.on_click(draw); display(run,out); draw()
