# JWST_0178
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

x=ns['x']; y=ns['y']; s=ns['s']; PNG=ns['PNG']; CSV=ns['CSV']
LYA=float(ns['LYA'])
VERSION='JWST_0178'
Z_OFFICIAL=11.1221
Z_SIGMA=0.0006
NU_REST_GHZ=3393.006244
NU_PUBLISHED_GHZ=279.901
NU_PUBLISHED_SIGMA_GHZ=0.014
C_KMS=299792.458

Z_DERIVED=NU_REST_GHZ/NU_PUBLISHED_GHZ-1
NU_OFFICIAL_GHZ=NU_REST_GHZ/(1+Z_OFFICIAL)
LYA_DERIVED_UM=LYA*(1+Z_DERIVED)
LYA_OFFICIAL_UM=LYA*(1+Z_OFFICIAL)
DELTA_Z=Z_DERIVED-Z_OFFICIAL
DELTA_NU_MHZ=(NU_OFFICIAL_GHZ-NU_PUBLISHED_GHZ)*1000.0
R=NU_PUBLISHED_GHZ/NU_OFFICIAL_GHZ
DV_EQUIV_KMS=C_KMS*((R*R)-1)/((R*R)+1)

# Important: no peculiar-velocity correction is applied. The [O III] centroid already defines
# the observed systemic redshift. A true peculiar-velocity subtraction would require an
# independent peculiar-velocity measurement, which is not supplied by these data.

for d in (PNG,CSV): Path(d).mkdir(parents=True,exist_ok=True)

summary=pd.DataFrame([
    ['Our direct calculation from rounded published [O III] centroid',Z_DERIVED,LYA_DERIVED_UM,NU_PUBLISHED_GHZ,0.0,0.0],
    ['Official ALMA full-precision systemic result',Z_OFFICIAL,LYA_OFFICIAL_UM,NU_OFFICIAL_GHZ,DELTA_NU_MHZ,DV_EQUIV_KMS],
],columns=['method','redshift_z','lya_reference_um','observed_OIII_frequency_GHz','frequency_adjustment_MHz','equivalent_centroid_offset_km_s'])
summary.to_csv(Path(CSV)/f'{VERSION}_ALMA_ONLY_REDSHIFT_VERIFICATION.csv',index=False)

nu=np.linspace(279.84,279.96,1200)
sigma_line=NU_PUBLISHED_SIGMA_GHZ
line=np.exp(-0.5*((nu-NU_OFFICIAL_GHZ)/sigma_line)**2)

plt.rcParams.update({'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black','text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white','axes.edgecolor':'#aeb8c3'})
run=widgets.Button(description='Plot ALMA-only verification',button_style='success',layout=widgets.Layout(width='280px'))
out=widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        print('REQUEST SUMMARY:')
        print('• Derive only the z=11.1221 ALMA systemic result; include no z=11.38 references.')
        print('• Compare our rounded-centroid calculation with the official full-precision value.')
        print('• Show how the Lyα reference 1.473647 µm is computed from each redshift.')
        print('• Do not claim a peculiar-velocity correction without independent velocity data.')
        fig=plt.figure(figsize=(17,18))
        gs=fig.add_gridspec(4,1,height_ratios=[1.55,1.05,1.15,1.25],hspace=.28)
        a1,a2,a3,a4=[fig.add_subplot(gs[i]) for i in range(4)]
        for a in (a1,a2,a3,a4): a.grid(color='#303944',lw=.6,alpha=.75)

        a1.plot(nu,line,color='#7fd5cb',lw=2.4,label='Reconstructed ALMA [O III] 88 µm line profile')
        a1.axvline(NU_PUBLISHED_GHZ,color='#ff5a36',ls='--',lw=2.1,label=f'Published rounded centroid = {NU_PUBLISHED_GHZ:.6f} GHz')
        a1.axvline(NU_OFFICIAL_GHZ,color='#ffb347',ls='-.',lw=2.1,label=f'Official-z implied centroid = {NU_OFFICIAL_GHZ:.9f} GHz')
        a1.axvspan(NU_PUBLISHED_GHZ-NU_PUBLISHED_SIGMA_GHZ,NU_PUBLISHED_GHZ+NU_PUBLISHED_SIGMA_GHZ,color='#ff5a36',alpha=.10,label='Published ±0.014 GHz')
        a1.set_xlim(279.84,279.96); a1.set_ylim(-.03,1.08)
        a1.set_xlabel('Observed [O III] 88 µm frequency [GHz]'); a1.set_ylabel('Normalized line amplitude')
        a1.legend(frameon=False,fontsize=9,ncol=2,loc='upper right')
        box=(f'OUR DERIVATION\n'
             f'z = νrest/νobs − 1\n'
             f'z = {NU_REST_GHZ:.6f}/{NU_PUBLISHED_GHZ:.6f} − 1\n'
             f'z = {Z_DERIVED:.9f}\n\n'
             f'OFFICIAL\n'
             f'z = {Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}')
        a1.text(.025,.37,box,transform=a1.transAxes,va='top',ha='left',fontsize=10,
                bbox=dict(boxstyle='round,pad=.5',facecolor='#111820',edgecolor='#72808e',alpha=.95))
        a1.set_title('ALMA [O III] centroid: our rounded-table derivation versus official full-precision result')

        methods=['Our rounded-centroid derivation','Official ALMA result']
        zs=[Z_DERIVED,Z_OFFICIAL]
        cols=['#ff5a36','#ffb347']
        ypos=[1,0]
        a2.scatter(zs,ypos,s=110,color=cols,zorder=4)
        a2.hlines(ypos,min(zs)-0.0008,zs,color=cols,lw=1.2)
        a2.annotate(f'Our derivation: z={Z_DERIVED:.9f}',xy=(Z_DERIVED,1),xytext=(-12,-26),textcoords='offset points',ha='right',va='top',color=cols[0],fontsize=11,arrowprops=dict(arrowstyle='-',color=cols[0],lw=.8))
        a2.annotate(f'Official: z={Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}',xy=(Z_OFFICIAL,0),xytext=(12,24),textcoords='offset points',ha='left',va='bottom',color=cols[1],fontsize=11,arrowprops=dict(arrowstyle='-',color=cols[1],lw=.8))
        a2.set_yticks([]); a2.set_xlim(min(zs)-0.0010,max(zs)+0.0010); a2.set_xlabel('Systemic redshift z')
        a2.set_title(f'Difference: Δz={DELTA_Z:.9f}; equivalent centroid offset={DV_EQUIV_KMS:.6f} km/s')

        a3.scatter([LYA_DERIVED_UM,LYA_OFFICIAL_UM],[1,0],s=110,color=cols,zorder=4)
        a3.axvline(LYA_DERIVED_UM,color=cols[0],ls='--',lw=1.7)
        a3.axvline(LYA_OFFICIAL_UM,color=cols[1],ls='-.',lw=1.7)
        a3.annotate(f'Our Lyα verification\nλ = 0.121567(1+{Z_DERIVED:.9f})\n= {LYA_DERIVED_UM:.9f} µm',xy=(LYA_DERIVED_UM,1),xytext=(-18,-34),textcoords='offset points',ha='right',va='top',color=cols[0],fontsize=10,arrowprops=dict(arrowstyle='-',color=cols[0],lw=.8))
        a3.annotate(f'Official Lyα reference\nλ = 0.121567(1+{Z_OFFICIAL:.7f})\n= {LYA_OFFICIAL_UM:.9f} µm',xy=(LYA_OFFICIAL_UM,0),xytext=(18,34),textcoords='offset points',ha='left',va='bottom',color=cols[1],fontsize=10,arrowprops=dict(arrowstyle='-',color=cols[1],lw=.8))
        a3.set_yticks([]); a3.set_xlim(min(LYA_DERIVED_UM,LYA_OFFICIAL_UM)-0.00003,max(LYA_DERIVED_UM,LYA_OFFICIAL_UM)+0.00003)
        a3.set_xlabel('Observed Lyα reference wavelength [µm]')
        a3.set_title('Independent wavelength verification from the two redshift values')

        a4.axis('off')
        text=(f'VELOCITY QUESTION\n\n'
              f'Rounded centroid → official centroid adjustment: {DELTA_NU_MHZ:.6f} MHz\n'
              f'Equivalent relativistic centroid offset: {DV_EQUIV_KMS:.6f} km/s\n\n'
              f'This is a frequency-rounding equivalence, not a measured peculiar velocity of the galaxy.\n'
              f'No galaxy-through-space velocity was subtracted because no independent peculiar-velocity\n'
              f'measurement is contained in the ALMA centroid or the JWST spectrum.\n\n'
              f'Final comparison:\n'
              f'Our rounded-table derivation  z = {Z_DERIVED:.9f}\n'
              f'Official full-precision value z = {Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}\n'
              f'Official Lyα reference        λ = {LYA_OFFICIAL_UM:.9f} µm')
        a4.text(.03,.95,text,transform=a4.transAxes,va='top',ha='left',fontsize=12,
                bbox=dict(boxstyle='round,pad=.65',facecolor='#111820',edgecolor='#72808e',alpha=.96))

        fig.suptitle('JADES-GS-z11-0 — ALMA-only systemic-redshift verification\nNo z=11.38 model or curve is included',fontsize=16,y=.997)
        outpng=Path(PNG)/f'{VERSION}_ALMA_ONLY_REDSHIFT_VERIFICATION.png'
        fig.savefig(outpng,dpi=500,bbox_inches='tight')
        plt.show()
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False,float_format=lambda v:f'{v:.9f}'))
        print(f'PNG: {outpng}')
        print(f'CSV: {Path(CSV)/f"{VERSION}_ALMA_ONLY_REDSHIFT_VERIFICATION.csv"}')
        print(f'# {VERSION}')

run.on_click(draw); display(run,out); draw()
