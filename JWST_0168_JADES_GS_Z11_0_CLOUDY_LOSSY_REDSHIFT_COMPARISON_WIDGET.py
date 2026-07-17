# JWST_0168
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
BASE='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0165_JADES_GS_Z11_0_HLSP_VALIDATED_PRODUCT_WIDGET.py'
source=requests.get(BASE,timeout=120).text
ns={}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(source,'JWST_0165_runtime.py','exec'),ns)

np=ns['np']; pd=ns['pd']; plt=ns['plt']; widgets=ns['widgets']; display=ns['display']; clear_output=ns['clear_output']
PNG=ns['PNG']; CSV=ns['CSV']; wave=ns['wave']; flux=ns['flux']; err=ns['err']; z_cont=float(ns['zbest']); ZP=float(ns['ZP'])
VERSION='JWST_0168'; LYA=0.121567

# CLOUDY-informed lossy surrogate: continuum break + a compact nebular line basis.
# This is not an exact CLOUDY run and does not reproduce the authors' unpublished grid.
line_names=np.array(['He II 1640','O III] 1661','O III] 1666','C III] 1907','C III] 1909','[O II] 3727','H gamma 4340','H beta 4861','[O III] 4959','[O III] 5007'])
line_rest=np.array([0.1640,0.1661,0.1666,0.1907,0.1909,0.3727,0.4340,0.4861,0.4959,0.5007])
line_prior=np.array([0.45,0.18,0.35,0.20,0.28,0.12,0.16,0.32,0.70,2.10])

m=np.isfinite(wave)&np.isfinite(flux)&np.isfinite(err)&(err>0)&(wave>=1.15)&(wave<=5.20)
x,y,s=wave[m],flux[m],err[m]
o=np.argsort(x); x,y,s=x[o],y[o],s[o]
if x.size<80: raise RuntimeError(f'Need at least 80 finite samples from 1.15–5.20 µm; found {x.size}')

R=100.0

def continuum(z,beta=-1.8,width=0.018):
    edge=LYA*(1+z)
    return (np.maximum(x,edge)/1.7)**beta * 0.5*(1+np.tanh((x-edge)/width))

def design(z):
    cols=[continuum(z)]
    for lam0,prior in zip(line_rest,line_prior):
        obs=lam0*(1+z)
        sig=max(obs/(R*2.354820045),0.004)
        cols.append(prior*np.exp(-0.5*((x-obs)/sig)**2))
    cols.append(np.ones_like(x))
    return np.column_stack(cols)

def fit_at_z(z, ridge=2.0):
    A=design(z); Aw=A/s[:,None]; yw=y/s
    reg=np.zeros((A.shape[1],A.shape[1])); reg[1:-1,1:-1]=ridge*np.eye(A.shape[1]-2)
    lhs=Aw.T@Aw+reg; rhs=Aw.T@yw
    try: coef=np.linalg.solve(lhs,rhs)
    except np.linalg.LinAlgError: coef=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    coef[1:-1]=np.maximum(coef[1:-1],0.0)
    model=A@coef
    chi=float(np.sum(((y-model)/s)**2))
    return chi,coef,model

zg=np.linspace(10.90,11.55,651)
chi=np.empty_like(zg)
for i,z in enumerate(zg): chi[i]=fit_at_z(float(z))[0]
i0=int(np.nanargmin(chi)); z_lossy=float(zg[i0]); chi0,coef,best=fit_at_z(z_lossy)
p=np.exp(np.clip(-0.5*(chi-np.nanmin(chi)),-700,0)); p/=np.trapz(p,zg)
c=np.cumsum(p); c/=c[-1]; z16,z50,z84=[float(np.interp(q,c,zg)) for q in (.16,.5,.84)]
resid=(y-best)/s
line_obs=line_rest*(1+z_lossy)

summary=pd.DataFrame([
    ['Continuum-break fit',z_cont,np.nan,np.nan,'Empirical Lyα-anchored continuum step'],
    ['CLOUDY-informed lossy fit',z50,z50-z16,z84-z50,'Continuum + nebular line basis; approximate'],
    ['Paper SMDS fit',ZP,np.nan,np.nan,'Reported model-fit value']
],columns=['method','redshift_z','minus_1sigma','plus_1sigma','status'])
summary.to_csv(CSV/f'{VERSION}_REDSHIFT_METHOD_COMPARISON.csv',index=False)
pd.DataFrame({'redshift_z':zg,'chi2':chi,'posterior':p}).to_csv(CSV/f'{VERSION}_LOSSY_CLOUDY_POSTERIOR.csv',index=False)
pd.DataFrame({'line':line_names,'rest_um':line_rest,'observed_um_at_best_z':line_obs,'relative_prior_weight':line_prior,'fitted_nonnegative_coefficient':coef[1:-1]}).to_csv(CSV/f'{VERSION}_LOSSY_CLOUDY_LINE_AUDIT.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Run lossy CLOUDY comparison',button_style='success',layout=widgets.Layout(width='285px')); out=widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(16,16)); gs=fig.add_gridspec(4,1,height_ratios=[.9,2.4,.9,.85],hspace=.18)
        a0,a1,a2,a3=[fig.add_subplot(gs[i]) for i in range(4)]
        for a in (a0,a1,a2,a3): a.grid(color='#303944',lw=.6,alpha=.72)
        a0.plot(zg,p,lw=2.2,label=f'Lossy CLOUDY-informed posterior z={z50:.5f} (+{z84-z50:.5f}/-{z50-z16:.5f})')
        a0.axvline(z_cont,ls='--',lw=1.3,label=f'Continuum-break z={z_cont:.5f}')
        a0.axvline(ZP,ls=':',lw=1.2,label='Paper SMDS z=11.38')
        a0.axvspan(z16,z84,alpha=.14); a0.set_ylabel('Posterior density'); a0.set_xlabel('Redshift z'); a0.legend(frameon=False,ncol=2,fontsize=9)

        a1.fill_between(x,y-s,y+s,step='mid',alpha=.25,label='1σ uncertainty')
        a1.step(x,y,where='mid',lw=.9,label='Validated JADES 1D spectrum')
        a1.plot(x,best,lw=2.2,label=f'CLOUDY-informed lossy model z={z_lossy:.5f}')
        a1.axvline(LYA*(1+z_lossy),ls='--',lw=1.3,label='Lyα continuum-break anchor')
        for name,obs in zip(line_names,line_obs):
            if 1.15<=obs<=5.20: a1.axvline(obs,lw=.45,alpha=.45)
        a1.set_xlim(1.15,5.20); q=np.nanpercentile(y,[1,99]); pad=.18*(q[1]-q[0]); a1.set_ylim(q[0]-pad,q[1]+pad)
        a1.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
        a1.legend(frameon=False,ncol=2,fontsize=9)

        a2.axhline(0,lw=.6); a2.step(x,resid,where='mid',lw=.9); a2.axhspan(-1,1,alpha=.18,label='±1σ')
        a2.set_xlim(1.15,5.20); a2.set_ylim(-6,6); a2.set_ylabel('Residual / σ'); a2.set_xlabel('Observed wavelength [µm]'); a2.legend(frameon=False)

        labels=['Continuum break','Lossy CLOUDY-informed','Paper SMDS']; vals=[z_cont,z50,ZP]
        a3.scatter(vals,[0,1,2],s=90)
        for yy,v,lab in zip([0,1,2],vals,labels): a3.text(v+.004,yy,f'{lab}: z={v:.5f}',va='center',fontsize=11)
        a3.set_xlim(min(vals)-.08,max(vals)+.12); a3.set_yticks([]); a3.set_xlabel('Redshift z'); a3.set_title('Method comparison — observed/reported status kept separate')

        fig.suptitle('JADES-GS-z11-0 — CLOUDY-informed lossy redshift derivation\nApproximate nebular template, not an exact CLOUDY/TLUSTY reproduction',fontsize=15,y=.995)
        fig.savefig(PNG/f'{VERSION}_CLOUDY_LOSSY_REDSHIFT_COMPARISON.png',dpi=500,bbox_inches='tight')
        plt.show()

run.on_click(draw); display(run,out); draw()
print(f'CODE OUTPUT: {VERSION}')
print(summary.to_string(index=False))
print(f'PNG: {PNG}/{VERSION}_CLOUDY_LOSSY_REDSHIFT_COMPARISON.png')
print(f'CSV: {CSV}/{VERSION}_REDSHIFT_METHOD_COMPARISON.csv')
print(f'# {VERSION}')
