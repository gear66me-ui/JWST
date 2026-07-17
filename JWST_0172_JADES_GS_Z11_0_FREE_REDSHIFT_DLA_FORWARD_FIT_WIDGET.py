# JWST_0172
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
BASE='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0165_JADES_GS_Z11_0_HLSP_VALIDATED_PRODUCT_WIDGET.py'
source=requests.get(BASE,timeout=120).text
ns={}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(source,'JWST_0165_runtime.py','exec'),ns)

np=ns['np']; pd=ns['pd']; plt=ns['plt']; widgets=ns['widgets']; display=ns['display']; clear_output=ns['clear_output']
PNG=ns['PNG']; CSV=ns['CSV']; x0=ns['x']; y0=ns['y']; s0=ns['s']; model_emp0=ns['best']; z_emp=float(ns['zbest']); LYA=float(ns['LYA']); ZP=float(ns['ZP'])
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
VERSION='JWST_0172'; Z_ALMA=11.1221

# Fit only the continuum-break region. ALMA redshift is comparison-only and never enters the fit.
m=(x0>=1.28)&(x0<=1.82)&np.isfinite(x0)&np.isfinite(y0)&np.isfinite(s0)&(s0>0)
x=x0[m]; y=y0[m]; s=s0[m]
ordr=np.argsort(x); x,y,s=x[ordr],y[ordr],s[ordr]

# Paper-method surrogate: hot primordial continuum + nebular continuum + neutral-H DLA damping wing.
# This is a forward fit to the observed PRISM spectrum; redshift is a free parameter.
def forward(theta):
    z,beta,nebfrac,lognhi,width=theta
    rest=x/(1+z)
    stellar=(np.maximum(rest,0.09)/0.16)**beta
    neb=(np.maximum(rest,0.0912)/0.16)**-1.05*np.exp(-np.maximum(rest-0.40,0)/0.55)
    intrinsic=(1-nebfrac)*stellar+nebfrac*neb
    dlam=rest-LYA
    blue=np.where(dlam<0,np.exp(-np.clip((-dlam)/0.006,0,80)**1.15),1.0)
    nhi=10**(lognhi-22.0)
    # Smooth Lorentz-wing surrogate: strongest at Ly-alpha and decays redward.
    red_tau=np.where(dlam>=0,0.0038*nhi/(dlam*dlam+width*width),0.0)
    trans=blue*np.exp(-np.clip(red_tau,0,80))
    raw=intrinsic*trans
    dx=np.diff(x); dx=dx[np.isfinite(dx)&(dx>0)]
    sig=max(float(np.nanmedian(x)/(100*2.35482*np.nanmedian(dx))),0.35)
    return gaussian_filter1d(raw,sig,mode='nearest')

def residual(theta):
    u=forward(theta)
    A=np.vstack([u,np.ones_like(u),x-np.nanmedian(x)]).T
    Aw=A/s[:,None]; yw=y/s
    coeff=np.linalg.lstsq(Aw,yw,rcond=None)[0]
    model=A@coeff
    return (y-model)/s

bounds_lo=[10.90,-4.0,0.0,20.0,0.001]
bounds_hi=[11.50, 1.0,0.9,23.8,0.030]
starts=[]
for z in [11.00,11.10,11.18,11.30,11.40]:
    for ln in [21.0,22.0,23.0]:
        starts.append([z,-1.8,0.25,ln,0.008])
solutions=[]
for st in starts:
    r=least_squares(residual,st,bounds=(bounds_lo,bounds_hi),max_nfev=5000,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    solutions.append(r)
best=min(solutions,key=lambda r:np.sum(r.fun*r.fun))
theta=best.x; zfit,beta,nebfrac,lognhi,width=theta
u=forward(theta)
A=np.vstack([u,np.ones_like(u),x-np.nanmedian(x)]).T
coeff=np.linalg.lstsq(A/s[:,None],y/s,rcond=None)[0]
model=A@coeff; resid=(y-model)/s; chi2=float(np.sum(resid**2)); dof=max(len(y)-len(theta)-3,1)
lya_fit=LYA*(1+zfit); lya_alma=LYA*(1+Z_ALMA); lya_emp=LYA*(1+z_emp); lya_paper=LYA*(1+ZP)

# Profile likelihood in redshift, re-optimizing all nuisance parameters at each grid point.
zgrid=np.linspace(10.95,11.45,251); prof=[]
seed=theta.copy()
for z in zgrid:
    def rz(q): return residual([z,q[0],q[1],q[2],q[3]])
    rr=least_squares(rz,seed[1:],bounds=([bounds_lo[1],bounds_lo[2],bounds_lo[3],bounds_lo[4]],[bounds_hi[1],bounds_hi[2],bounds_hi[3],bounds_hi[4]]),max_nfev=1200)
    prof.append(np.sum(rr.fun*rr.fun)); seed[1:]=rr.x
prof=np.asarray(prof,float); dp=prof-np.nanmin(prof)
post=np.exp(np.clip(-0.5*dp,-700,0)); area=np.trapz(post,zgrid); post/=area
cdf=np.cumsum(post); cdf/=cdf[-1]
z16,z50,z84=[float(np.interp(q,cdf,zgrid)) for q in (0.16,0.50,0.84)]

summary=pd.DataFrame([
 ['Free-redshift H/He+nebular+DLA forward fit','derived from PRISM',zfit,lya_fit,chi2/dof,beta,nebfrac,lognhi,width],
 ['Official ALMA [O III] systemic','external comparison only',Z_ALMA,lya_alma,np.nan,np.nan,np.nan,np.nan,np.nan],
 ['Empirical continuum-break fit','comparison',z_emp,lya_emp,np.nan,np.nan,np.nan,np.nan,np.nan],
 ['Published SMDS paper fit','reported',ZP,lya_paper,np.nan,np.nan,np.nan,np.nan,np.nan],
],columns=['method','status','redshift_z','lya_reference_um','reduced_chi2','continuum_beta','nebular_fraction','logNHI_cm2','wing_width_um'])
summary.to_csv(CSV/f'{VERSION}_FIT_SUMMARY.csv',index=False)
pd.DataFrame({'redshift_z':zgrid,'delta_chi2':dp,'posterior_density':post}).to_csv(CSV/f'{VERSION}_REDSHIFT_PROFILE.csv',index=False)
pd.DataFrame({'wavelength_um':x,'flux':y,'sigma':s,'forward_model':model,'residual_sigma':resid}).to_csv(CSV/f'{VERSION}_SPECTRUM_FIT.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Plot free-redshift DLA fit',button_style='success',layout=widgets.Layout(width='270px')); out=widgets.Output()
def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(16,15)); gs=fig.add_gridspec(3,1,height_ratios=[0.9,2.5,1.0],hspace=.20)
        a0,a1,a2=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a0,a1,a2): a.grid(color='#303944',lw=.6,alpha=.75)
        a0.plot(zgrid,post,lw=2.2,label=f'Data-derived posterior z={z50:.5f} (+{z84-z50:.5f}/-{z50-z16:.5f})')
        a0.axvline(zfit,lw=1.5,label=f'Best forward fit z={zfit:.5f}')
        a0.axvline(Z_ALMA,ls='--',lw=1.3,label='ALMA comparison z=11.1221')
        a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False,ncol=2)
        a1.fill_between(x,y-s,y+s,step='mid',alpha=.25,label='1σ uncertainty')
        a1.step(x,y,where='mid',lw=1.0,label='Validated JADES PRISM spectrum')
        a1.plot(x,model,lw=2.4,label=f'Free-z H/He + nebular + DLA forward fit, z={zfit:.5f}')
        a1.axvline(lya_fit,ls='--',lw=1.8,label=f'Derived Lyα systemic wavelength = {lya_fit:.6f} µm')
        a1.axvline(lya_alma,ls=':',lw=1.4,label=f'ALMA comparison = {lya_alma:.6f} µm')
        a1.set_xlim(1.28,1.82); q=np.nanpercentile(y,[1,99]); pad=.25*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad)
        a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]'); a1.legend(frameon=False,ncol=2,fontsize=9)
        a2.axhline(0,lw=.7); a2.step(x,resid,where='mid',lw=1.0,label='Forward-fit residual / σ'); a2.axhspan(-1,1,alpha=.16); a2.set_xlim(1.28,1.82); a2.set_ylim(-6,6); a2.set_xlabel('Observed wavelength [µm]'); a2.set_ylabel('Residual / σ'); a2.legend(frameon=False)
        fig.suptitle('JADES-GS-z11-0 — free-redshift H/He + nebular + DLA forward fit\nALMA value shown only after the fit for comparison; it is not imposed',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_FREE_REDSHIFT_DLA_FORWARD_FIT.png',dpi=500,bbox_inches='tight')
        plt.show()
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False,float_format=lambda v:f'{v:.6f}'))
        print(f'PNG: {PNG/f"{VERSION}_FREE_REDSHIFT_DLA_FORWARD_FIT.png"}')
        print(f'CSV: {CSV/f"{VERSION}_FIT_SUMMARY.csv"}')
        print(f'# {VERSION}')
run.on_click(draw); display(run,out); draw()
