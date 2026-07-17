# JWST_0175
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION='JWST_0175'
ROOT=Path('/content/JWST_OUTPUT'); PNG=ROOT/'PNG'; CSV=ROOT/'CSV'
for d in (PNG,CSV): d.mkdir(parents=True,exist_ok=True)

C_KMS=299792.458
NU_REST_GHZ=3393.006244
NU_TABLE_GHZ=279.901
NU_TABLE_SIGMA_GHZ=0.014
Z_OFFICIAL=11.1221
Z_OFFICIAL_SIGMA=0.0006
LYA_REST_UM=0.121567

NU_EXACT_GHZ=NU_REST_GHZ/(1.0+Z_OFFICIAL)
Z_TABLE=NU_REST_GHZ/NU_TABLE_GHZ-1.0
DELTA_NU_GHZ=NU_EXACT_GHZ-NU_TABLE_GHZ
DELTA_Z=Z_TABLE-Z_OFFICIAL
RATIO=NU_TABLE_GHZ/NU_EXACT_GHZ
DV_REL_KMS=C_KMS*((RATIO*RATIO)-1.0)/((RATIO*RATIO)+1.0)
DV_APPROX_KMS=C_KMS*(NU_TABLE_GHZ-NU_EXACT_GHZ)/NU_EXACT_GHZ
LYA_SYS_UM=LYA_REST_UM*(1.0+Z_OFFICIAL)

# Reconstructed line profile for visual comparison only.
nu=np.linspace(279.80,280.00,1200)
sigma_nu=NU_TABLE_SIGMA_GHZ
profile=np.exp(-0.5*((nu-NU_EXACT_GHZ)/sigma_nu)**2)

summary=pd.DataFrame([
    ['Published rounded centroid',NU_TABLE_GHZ,Z_TABLE,np.nan,np.nan],
    ['Official-z implied centroid',NU_EXACT_GHZ,Z_OFFICIAL,DELTA_NU_GHZ,DV_REL_KMS],
],columns=['quantity','frequency_GHz','redshift_z','correction_GHz','equivalent_velocity_km_s'])
summary.to_csv(CSV/f'{VERSION}_CENTROID_VELOCITY_CORRECTION.csv',index=False)
pd.DataFrame({'frequency_GHz':nu,'normalized_profile':profile}).to_csv(CSV/f'{VERSION}_RECONSTRUCTED_LINE_PROFILE.csv',index=False)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Plot corrected ALMA derivation',button_style='success',layout=widgets.Layout(width='290px'))
out=widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(15,13))
        gs=fig.add_gridspec(3,1,height_ratios=[1.6,1.1,1.3],hspace=.34)
        a0,a1,a2=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a0,a1,a2): a.grid(color='#303944',lw=.6,alpha=.75)

        a0.plot(nu,profile,lw=2.3,color='#7fd5cb',label='Reconstructed [O III] 88 μm Gaussian profile')
        a0.axvline(NU_TABLE_GHZ,color='#ff6b3d',ls='--',lw=2.0,label=f'Published rounded centroid: {NU_TABLE_GHZ:.6f} GHz')
        a0.axvline(NU_EXACT_GHZ,color='#ffd166',ls='-.',lw=2.0,label=f'Official-z implied centroid: {NU_EXACT_GHZ:.6f} GHz')
        a0.axvspan(NU_TABLE_GHZ-NU_TABLE_SIGMA_GHZ,NU_TABLE_GHZ+NU_TABLE_SIGMA_GHZ,color='#ff6b3d',alpha=.10,label='Published ±0.014 GHz')
        a0.set_xlim(279.84,279.96); a0.set_ylim(-.03,1.08)
        a0.set_xlabel('Observed [O III] 88 μm frequency [GHz]'); a0.set_ylabel('Normalized amplitude')
        a0.set_title('Published rounded centroid versus centroid implied by the official full-precision redshift')
        a0.legend(frameon=False,loc='upper right',fontsize=9)

        y_table,y_exact=1.0,0.0
        a1.errorbar(Z_TABLE,y_table,xerr=NU_REST_GHZ*NU_TABLE_SIGMA_GHZ/(NU_TABLE_GHZ**2),fmt='o',ms=9,capsize=4,color='#ff6b3d')
        a1.errorbar(Z_OFFICIAL,y_exact,xerr=Z_OFFICIAL_SIGMA,fmt='o',ms=9,capsize=4,color='#ffd166')
        a1.annotate(f'Rounded table result\nz = {Z_TABLE:.9f}',xy=(Z_TABLE,y_table),xytext=(22,18),textcoords='offset points',ha='left',va='bottom',color='#ff8f66',arrowprops=dict(arrowstyle='-',color='#ff8f66',lw=.8))
        a1.annotate(f'Official full-precision result\nz = {Z_OFFICIAL:.7f}',xy=(Z_OFFICIAL,y_exact),xytext=(22,-20),textcoords='offset points',ha='left',va='top',color='#ffe08a',arrowprops=dict(arrowstyle='-',color='#ffe08a',lw=.8))
        a1.set_yticks([]); a1.set_xlabel('Systemic redshift z')
        pad=max(abs(DELTA_Z)*7,0.0015); a1.set_xlim(min(Z_TABLE,Z_OFFICIAL)-pad,max(Z_TABLE,Z_OFFICIAL)+pad)
        a1.set_ylim(-.65,1.65)
        a1.set_title('Labels offset from error bars for readability')

        a2.axis('off')
        text=(
            'CENTROID CORRECTION DERIVATION\n\n'
            f'ν_rest = {NU_REST_GHZ:.6f} GHz\n'
            f'ν_obs,published = {NU_TABLE_GHZ:.6f} ± {NU_TABLE_SIGMA_GHZ:.6f} GHz\n'
            f'z(published rounded centroid) = ν_rest/ν_obs − 1 = {Z_TABLE:.9f}\n\n'
            f'z_official = {Z_OFFICIAL:.7f} ± {Z_OFFICIAL_SIGMA:.7f}\n'
            f'ν_obs implied by z_official = ν_rest/(1+z) = {NU_EXACT_GHZ:.9f} GHz\n'
            f'Δν = ν_exact − ν_table = {DELTA_NU_GHZ*1e3:.6f} MHz\n'
            f'Δz = {DELTA_Z:.9f}\n'
            f'Equivalent relativistic velocity offset = {DV_REL_KMS:.6f} km/s\n'
            f'Nominal systemic Lyα wavelength = {LYA_SYS_UM:.6f} μm\n\n'
            'Interpretation: this ~1.62 km/s quantity is the equivalent offset caused by\n'
            'the rounded tabulated centroid versus the full-precision centroid. It is not\n'
            'an independently measured peculiar velocity of the galaxy.'
        )
        a2.text(.5,.5,text,ha='center',va='center',fontsize=12.5,
                bbox=dict(boxstyle='round,pad=.8',facecolor='#111820',edgecolor='#72808e',alpha=.97))

        fig.suptitle('JADES-GS-z11-0 — corrected ALMA [O III] centroid and equivalent velocity offset\nOfficial systemic redshift retained separately: z = 11.1221 ± 0.0006',fontsize=15,y=.995)
        outpng=PNG/f'{VERSION}_ALMA_CENTROID_VELOCITY_CORRECTION.png'
        fig.savefig(outpng,dpi=500,bbox_inches='tight')
        plt.show()
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False,float_format=lambda v:f'{v:.9f}'))
        print(f'PNG: {outpng}')
        print(f'CSV: {CSV/f"{VERSION}_CENTROID_VELOCITY_CORRECTION.csv"}')
        print(f'# {VERSION}')

run.on_click(draw)
display(run,out)
draw()
