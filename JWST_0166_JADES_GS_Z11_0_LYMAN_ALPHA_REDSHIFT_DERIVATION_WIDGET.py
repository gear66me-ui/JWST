# JWST_0166
import io, contextlib, requests, warnings
warnings.filterwarnings('ignore')
BASE='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0165_JADES_GS_Z11_0_HLSP_VALIDATED_PRODUCT_WIDGET.py'
source=requests.get(BASE,timeout=120).text
ns={}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(source,'JWST_0165_runtime.py','exec'),ns)

np=ns['np']; pd=ns['pd']; plt=ns['plt']; widgets=ns['widgets']; display=ns['display']; clear_output=ns['clear_output']
PNG=ns['PNG']; CSV=ns['CSV']; x=ns['x']; y=ns['y']; s=ns['s']; best=ns['best']; zbest=float(ns['zbest']); ZP=float(ns['ZP']); LYA=float(ns['LYA'])
VERSION='JWST_0166'; lya_obs=LYA*(1+zbest); paper_obs=LYA*(1+ZP)

run=widgets.Button(description='Plot Ly-alpha redshift derivation',button_style='success',layout=widgets.Layout(width='290px'))
out=widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,11),facecolor='black')
        gs=fig.add_gridspec(2,1,height_ratios=[2.2,1.25],hspace=.24)
        ax=fig.add_subplot(gs[0]); ax2=fig.add_subplot(gs[1])
        for a in (ax,ax2):
            a.set_facecolor('black'); a.grid(color='#303944',lw=.6,alpha=.75)
        m=(x>=1.34)&(x<=1.68)
        ax.fill_between(x[m],y[m]-s[m],y[m]+s[m],step='mid',color='gray',alpha=.32,label='1σ uncertainty')
        ax.step(x[m],y[m],where='mid',color='#4f79b9',lw=1.1,label='Validated JADES PRISM spectrum')
        ax.plot(x[m],best[m],color='#e68645',lw=2.6,label=f'Best Lyman-break fit, z={zbest:.5f}')
        ax.axvline(lya_obs,color='#ffd400',ls='--',lw=1.7,label=f'Fitted Lyα break = {lya_obs:.6f} μm')
        ax.axvline(paper_obs,color='#ff5f5f',ls=':',lw=1.5,label=f'Paper z=11.38 → {paper_obs:.6f} μm')
        ax.set_xlim(1.34,1.68); q=np.nanpercentile(y[m],[1,99]); pad=.22*(q[1]-q[0]); ax.set_ylim(q[0]-pad,q[1]+pad)
        ax.set_ylabel(r'Flux density [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]')
        ax.set_title('Observed Lyman-α break used as the redshift anchor',fontsize=14)
        ax.legend(frameon=False,ncol=2,fontsize=9)

        lam=np.linspace(1.35,1.68,500); zcurve=lam/LYA-1
        ax2.plot(lam,zcurve,color='#62dfd1',lw=2.2,label=r'$z=\lambda_{obs}/\lambda_{rest}-1$')
        ax2.axvline(lya_obs,color='#ffd400',ls='--',lw=1.5)
        ax2.axhline(zbest,color='#ffd400',ls='--',lw=1.2)
        ax2.scatter([lya_obs],[zbest],s=55,color='#ffd400',zorder=5)
        text=(f'λ_rest(Lyα) = {LYA:.6f} μm\n'
              f'λ_obs(fit) = {lya_obs:.6f} μm\n'
              f'z = ({lya_obs:.6f} / {LYA:.6f}) − 1\n'
              f'z = {zbest:.6f}')
        ax2.text(.02,.96,text,transform=ax2.transAxes,va='top',ha='left',fontsize=12,
                 bbox=dict(boxstyle='round,pad=.5',facecolor='#111820',edgecolor='#72808e',alpha=.95))
        ax2.set_xlim(1.35,1.68); ax2.set_ylim(zcurve.min(),zcurve.max())
        ax2.set_xlabel('Observed Lyα-break wavelength [μm]'); ax2.set_ylabel('Derived redshift, z')
        ax2.set_title('Exact wavelength-to-redshift calculation',fontsize=13)
        ax2.legend(frameon=False,loc='lower right')
        fig.suptitle('JADES-GS-z11-0 — direct Lyα-break redshift derivation',fontsize=16,y=.995)
        fig.savefig(PNG/f'{VERSION}_LYMAN_ALPHA_REDSHIFT_DERIVATION.png',dpi=500,bbox_inches='tight')
        pd.DataFrame({'lambda_obs_um':lam,'z_from_lya':zcurve}).to_csv(CSV/f'{VERSION}_LYMAN_ALPHA_REDSHIFT_CURVE.csv',index=False)
        plt.show()
run.on_click(draw); display(run,out); draw()
