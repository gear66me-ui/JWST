# JWST_0182
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION='JWST_0182'
OUT=Path('/content/JWST_OUTPUT')
PNG=OUT/'PNG'; CSV=OUT/'CSV'
for d in (PNG,CSV): d.mkdir(parents=True,exist_ok=True)

NU_REST_GHZ=3393.006244
NU_ROUNDED_GHZ=279.901000
NU_SIGMA_GHZ=0.014000
Z_OFFICIAL=11.1221000
Z_OFFICIAL_SIGMA=0.0006000
LYA_REST_UM=0.121567
C_KMS=299792.458

Z_DERIVED=NU_REST_GHZ/NU_ROUNDED_GHZ-1.0
NU_REQUIRED_GHZ=NU_REST_GHZ/(1.0+Z_OFFICIAL)
DELTA_NU_GHZ=NU_REQUIRED_GHZ-NU_ROUNDED_GHZ
DELTA_NU_MHZ=DELTA_NU_GHZ*1000.0
DELTA_Z=Z_DERIVED-Z_OFFICIAL
DZ_DNU=-NU_REST_GHZ/(NU_ROUNDED_GHZ**2)
SIGMA_Z_LINEAR=abs(DZ_DNU)*NU_SIGMA_GHZ
ROUND_HALF_MHZ=0.0005*1000.0
SIGMA_Z_1MHZ=abs(DZ_DNU)*0.001
LYA_DERIVED_UM=LYA_REST_UM*(1.0+Z_DERIVED)
LYA_OFFICIAL_UM=LYA_REST_UM*(1.0+Z_OFFICIAL)
DELTA_LYA_NM=(LYA_DERIVED_UM-LYA_OFFICIAL_UM)*1000.0
R=(1.0+Z_DERIVED)/(1.0+Z_OFFICIAL)
DV_EQUIV_KMS=C_KMS*((R*R)-1.0)/((R*R)+1.0)
SIGMA_RATIO=DELTA_Z/Z_OFFICIAL_SIGMA

rng=np.random.default_rng(182)
nu_mc=rng.normal(NU_ROUNDED_GHZ,NU_SIGMA_GHZ,250000)
z_mc=NU_REST_GHZ/nu_mc-1.0
mc_mean=float(np.mean(z_mc)); mc_std=float(np.std(z_mc,ddof=1))
mc_p16,mc_p50,mc_p84=np.percentile(z_mc,[16,50,84])

summary=pd.DataFrame([
    ['Rounded published centroid',NU_ROUNDED_GHZ,Z_DERIVED,LYA_DERIVED_UM,DELTA_Z,DELTA_LYA_NM,DV_EQUIV_KMS],
    ['Full-precision centroid required by official z',NU_REQUIRED_GHZ,Z_OFFICIAL,LYA_OFFICIAL_UM,0.0,0.0,0.0],
],columns=['case','observed_frequency_GHz','redshift_z','lya_reference_um','delta_z','delta_lya_nm','equivalent_velocity_km_s'])
summary.to_csv(CSV/f'{VERSION}_SUMMARY.csv',index=False)

error_budget=pd.DataFrame([
    ['Displayed centroid rounding needed to match official z',abs(DELTA_Z),'deterministic offset'],
    ['1 MHz centroid perturbation',SIGMA_Z_1MHZ,'local sensitivity'],
    ['Published ±0.014 GHz centroid uncertainty',SIGMA_Z_LINEAR,'1σ propagated'],
    ['Official quoted redshift uncertainty',Z_OFFICIAL_SIGMA,'1σ reported'],
],columns=['source','redshift_scale','status'])
error_budget.to_csv(CSV/f'{VERSION}_ERROR_BUDGET.csv',index=False)

plt.rcParams.update({
    'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black',
    'text.color':'white','axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','axes.edgecolor':'#aeb8c3'
})

def table_png():
    fig,ax=plt.subplots(figsize=(17,5.4))
    ax.axis('off')
    rows=[
        ['Our rounded-centroid redshift',f'{Z_DERIVED:.9f}','Direct calculation'],
        ['Official ALMA redshift',f'{Z_OFFICIAL:.7f} ± {Z_OFFICIAL_SIGMA:.7f}','Published result'],
        ['Difference',f'{DELTA_Z:.9f}',f'{SIGMA_RATIO:.3f} σ of official uncertainty'],
        ['Required centroid',f'{NU_REQUIRED_GHZ:.9f} GHz',f'{DELTA_NU_MHZ:.6f} MHz above rounded value'],
        ['Equivalent velocity scale',f'{DV_EQUIV_KMS:.6f} km/s','Not a peculiar-velocity measurement'],
        ['Our Lyα reference',f'{LYA_DERIVED_UM:.9f} µm',f'+{DELTA_LYA_NM:.6f} nm from official'],
        ['Official Lyα reference',f'{LYA_OFFICIAL_UM:.9f} µm','Reference'],
        ['Monte Carlo σ(z)',f'{mc_std:.9f}','Using ±0.014 GHz centroid uncertainty'],
    ]
    tab=ax.table(cellText=rows,colLabels=['Quantity','Value','Interpretation'],loc='center',cellLoc='left',colLoc='left',bbox=[0.01,0.03,0.98,0.88])
    tab.auto_set_font_size(False); tab.set_fontsize(10.5)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_linewidth(.8)
        cell.set_facecolor('#111820' if r else '#1c2b38')
        cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
    ax.set_title('JADES-GS-z11-0 — mathematical redshift error budget',fontsize=15,pad=14)
    path=PNG/f'{VERSION}_ERROR_BUDGET_TABLE.png'
    fig.savefig(path,dpi=500,bbox_inches='tight'); plt.close(fig)
    return path

run=widgets.Button(description='Plot mathematical audit',button_style='success',layout=widgets.Layout(width='280px'))
out=widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,16))
        gs=fig.add_gridspec(3,1,height_ratios=[1.35,1.05,1.15],hspace=.28)
        a1,a2,a3=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a1,a2,a3): a.grid(color='#303944',lw=.6,alpha=.75)

        nu_local=np.linspace(NU_ROUNDED_GHZ-0.02,NU_ROUNDED_GHZ+0.02,1400)
        z_local=NU_REST_GHZ/nu_local-1.0
        a1.plot(nu_local,z_local,lw=2.2,label='Exact relation z = νrest/νobs − 1')
        a1.axvline(NU_ROUNDED_GHZ,ls='--',lw=1.8,label=f'Rounded centroid {NU_ROUNDED_GHZ:.6f} GHz')
        a1.axvline(NU_REQUIRED_GHZ,ls='-.',lw=1.8,label=f'Centroid required by official z {NU_REQUIRED_GHZ:.9f} GHz')
        a1.scatter([NU_ROUNDED_GHZ,NU_REQUIRED_GHZ],[Z_DERIVED,Z_OFFICIAL],s=90,zorder=5)
        a1.set_xlabel('Observed [O III] frequency [GHz]'); a1.set_ylabel('Derived redshift z')
        a1.set_title(f'Centroid sensitivity: Δν = {DELTA_NU_MHZ:.6f} MHz produces Δz = {DELTA_Z:.9f}')
        a1.legend(frameon=False,loc='upper right')

        names=['Actual offset','1 MHz sensitivity','Published centroid σ','Official z σ']
        vals=[abs(DELTA_Z),SIGMA_Z_1MHZ,SIGMA_Z_LINEAR,Z_OFFICIAL_SIGMA]
        a2.barh(names,vals)
        a2.axvline(abs(DELTA_Z),ls='--',lw=1.2,label='Observed discrepancy')
        a2.set_xlabel('Redshift scale |Δz|')
        a2.set_title('Error-budget comparison')
        a2.legend(frameon=False)

        bins=np.linspace(Z_DERIVED-4*mc_std,Z_DERIVED+4*mc_std,110)
        a3.hist(z_mc,bins=bins,density=True,alpha=.65,label='Monte Carlo from ±0.014 GHz centroid uncertainty')
        a3.axvline(Z_DERIVED,ls='--',lw=1.8,label=f'Our derivation {Z_DERIVED:.9f}')
        a3.axvline(Z_OFFICIAL,ls='-.',lw=1.8,label=f'Official {Z_OFFICIAL:.7f}')
        a3.set_xlabel('Redshift z'); a3.set_ylabel('Probability density')
        a3.set_title(f'Official value lies {SIGMA_RATIO:.3f}σ from our rounded-centroid result')
        a3.legend(frameon=False)

        fig.suptitle('JADES-GS-z11-0 — mathematical audit of the small redshift difference\nDominant cause: published centroid rounding / finite centroid precision',fontsize=16,y=.995)
        main_png=PNG/f'{VERSION}_MATHEMATICAL_AUDIT.png'
        fig.savefig(main_png,dpi=500,bbox_inches='tight')
        plt.show()
        tab=table_png(); display(plt.imread(tab))

        print(f'CODE OUTPUT: {VERSION}')
        print(f'Our redshift: {Z_DERIVED:.9f}')
        print(f'Official redshift: {Z_OFFICIAL:.7f} ± {Z_OFFICIAL_SIGMA:.7f}')
        print(f'Difference: {DELTA_Z:.9f} = {SIGMA_RATIO:.3f} sigma')
        print(f'Frequency needed to match official z: {NU_REQUIRED_GHZ:.9f} GHz')
        print(f'Centroid adjustment: {DELTA_NU_MHZ:.6f} MHz')
        print(f'Equivalent velocity scale: {DV_EQUIV_KMS:.6f} km/s')
        print(f'Monte Carlo sigma(z): {mc_std:.9f}')
        print(f'PNG: {main_png}')
        print(f'TABLE PNG: {tab}')
        print(f'CSV: {CSV/f"{VERSION}_ERROR_BUDGET.csv"}')
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')

run.on_click(draw)
display(run,out)
draw()
