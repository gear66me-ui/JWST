# JWST_0180
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

VERSION='JWST_0180'
OUT=Path('/content/JWST_OUTPUT')
PNG=OUT/'PNG'; CSV=OUT/'CSV'
for d in (PNG,CSV): d.mkdir(parents=True,exist_ok=True)

NU_REST_GHZ=3393.006244
NU_ROUNDED_GHZ=279.901000
NU_SIGMA_GHZ=0.014000
Z_OFFICIAL=11.1221000
Z_SIGMA=0.0006000
LYA_REST_UM=0.121567
C_KMS=299792.458

Z_DERIVED=NU_REST_GHZ/NU_ROUNDED_GHZ-1.0
NU_OFFICIAL_GHZ=NU_REST_GHZ/(1.0+Z_OFFICIAL)
LYA_DERIVED_UM=LYA_REST_UM*(1.0+Z_DERIVED)
LYA_OFFICIAL_UM=LYA_REST_UM*(1.0+Z_OFFICIAL)
DELTA_Z=Z_DERIVED-Z_OFFICIAL
DELTA_NU_MHZ=(NU_OFFICIAL_GHZ-NU_ROUNDED_GHZ)*1000.0
R=NU_ROUNDED_GHZ/NU_OFFICIAL_GHZ
DV_EQUIV_KMS=C_KMS*((R*R)-1.0)/((R*R)+1.0)
DELTA_LYA_NM=(LYA_DERIVED_UM-LYA_OFFICIAL_UM)*1000.0

summary=pd.DataFrame([
    ['Our rounded-centroid derivation',NU_ROUNDED_GHZ,Z_DERIVED,LYA_DERIVED_UM,DELTA_Z,DELTA_LYA_NM,0.0],
    ['Official ALMA full-precision result',NU_OFFICIAL_GHZ,Z_OFFICIAL,LYA_OFFICIAL_UM,0.0,0.0,DV_EQUIV_KMS],
],columns=['method','observed_OIII_frequency_GHz','redshift_z','lya_reference_um','delta_z_from_official','delta_lya_nm_from_official','equivalent_centroid_offset_km_s'])
summary.to_csv(CSV/f'{VERSION}_ALMA_ONLY_RESULTS.csv',index=False)

nu=np.linspace(279.84,279.96,1200)
line=np.exp(-0.5*((nu-NU_OFFICIAL_GHZ)/NU_SIGMA_GHZ)**2)

plt.rcParams.update({
    'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black',
    'text.color':'white','axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','axes.edgecolor':'#aeb8c3'
})

run=widgets.Button(description='Plot ALMA-only verification',button_style='success',layout=widgets.Layout(width='285px'))
out=widgets.Output()

def styled_table_png():
    fig,ax=plt.subplots(figsize=(16,5.2))
    ax.axis('off')
    rows=[
        ['Observed [O III] centroid',f'{NU_ROUNDED_GHZ:.6f} GHz',f'{NU_OFFICIAL_GHZ:.9f} GHz',f'{DELTA_NU_MHZ:.6f} MHz'],
        ['Systemic redshift',f'{Z_DERIVED:.9f}',f'{Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}',f'Δz = {DELTA_Z:.9f}'],
        ['Lyα reference wavelength',f'{LYA_DERIVED_UM:.9f} µm',f'{LYA_OFFICIAL_UM:.9f} µm',f'Δλ = {DELTA_LYA_NM:.6f} nm'],
        ['Equivalent centroid offset','—','—',f'{DV_EQUIV_KMS:.6f} km/s'],
        ['Interpretation','Rounded table input','Full-precision published fit','Frequency-rounding equivalence; not peculiar velocity'],
    ]
    cols=['Quantity','Our derivation','Official ALMA','Difference / note']
    tbl=ax.table(cellText=rows,colLabels=cols,cellLoc='left',colLoc='left',loc='center',bbox=[0.01,0.05,0.98,0.82])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5)
    for (r,c),cell in tbl.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_linewidth(.8)
        cell.set_facecolor('#1c2b38' if r==0 else '#111820')
        cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
        if r>0 and c==1: cell.get_text().set_color('#ff6a45')
        if r>0 and c==2: cell.get_text().set_color('#ffb347')
    ax.set_title('JADES-GS-z11-0 — ALMA systemic-redshift verification summary',fontsize=15,pad=14)
    path=PNG/f'{VERSION}_SUMMARY_TABLE.png'
    fig.savefig(path,dpi=500,bbox_inches='tight'); plt.close(fig)
    return path

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig=plt.figure(figsize=(17,19))
        gs=fig.add_gridspec(4,1,height_ratios=[1.5,1.0,1.1,1.2],hspace=.30)
        a1,a2,a3,a4=[fig.add_subplot(gs[i]) for i in range(4)]
        for a in (a1,a2,a3): a.grid(color='#303944',lw=.6,alpha=.75)

        a1.plot(nu,line,color='#7fd5cb',lw=2.4,label='Reconstructed ALMA [O III] 88 µm line profile')
        a1.axvline(NU_ROUNDED_GHZ,color='#ff5a36',ls='--',lw=2.1,label=f'Rounded published centroid = {NU_ROUNDED_GHZ:.6f} GHz')
        a1.axvline(NU_OFFICIAL_GHZ,color='#ffb347',ls='-.',lw=2.1,label=f'Official-z implied centroid = {NU_OFFICIAL_GHZ:.9f} GHz')
        a1.axvspan(NU_ROUNDED_GHZ-NU_SIGMA_GHZ,NU_ROUNDED_GHZ+NU_SIGMA_GHZ,color='#ff5a36',alpha=.10,label='Published ±0.014 GHz')
        a1.set_xlim(279.84,279.96); a1.set_ylim(-.03,1.08)
        a1.set_xlabel('Observed [O III] 88 µm frequency [GHz]'); a1.set_ylabel('Normalized line amplitude')
        a1.legend(frameon=False,fontsize=8.8,ncol=1,loc='upper left',bbox_to_anchor=(0.01,0.98))
        a1.set_title('ALMA [O III] centroid verification')

        a2.scatter([Z_OFFICIAL,Z_DERIVED],[0,1],s=125,color=['#ffb347','#ff5a36'],zorder=4)
        a2.annotate(f'Official ALMA\nz = {Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}',xy=(Z_OFFICIAL,0),xytext=(-36,24),textcoords='offset points',ha='right',va='bottom',color='#ffb347',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ffb347',lw=.9))
        a2.annotate(f'Our derivation\nz = {Z_DERIVED:.9f}',xy=(Z_DERIVED,1),xytext=(36,-24),textcoords='offset points',ha='left',va='top',color='#ff5a36',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ff5a36',lw=.9))
        a2.set_yticks([]); a2.set_xlim(min(Z_OFFICIAL,Z_DERIVED)-0.0008,max(Z_OFFICIAL,Z_DERIVED)+0.0008)
        a2.set_xlabel('Systemic redshift z'); a2.set_title(f'Our derivation versus official result — Δz = {DELTA_Z:.9f}')

        a3.axvline(0.0,color='#ffb347',ls='-.',lw=1.8)
        a3.axvline(DELTA_LYA_NM,color='#ff5a36',ls='--',lw=1.8)
        a3.scatter([0.0,DELTA_LYA_NM],[0,1],s=125,color=['#ffb347','#ff5a36'],zorder=4)
        a3.annotate(f'Official Lyα reference\n{LYA_OFFICIAL_UM:.9f} µm\n0.000000 nm offset',xy=(0,0),xytext=(-42,30),textcoords='offset points',ha='right',va='bottom',color='#ffb347',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ffb347',lw=.9))
        a3.annotate(f'Our Lyα verification\n{LYA_DERIVED_UM:.9f} µm\n+{DELTA_LYA_NM:.6f} nm',xy=(DELTA_LYA_NM,1),xytext=(42,-30),textcoords='offset points',ha='left',va='top',color='#ff5a36',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ff5a36',lw=.9))
        pad=max(abs(DELTA_LYA_NM)*0.9,0.004)
        a3.set_xlim(-pad,DELTA_LYA_NM+pad); a3.set_yticks([])
        a3.set_xlabel('Lyα wavelength offset from official reference [nm]')
        a3.set_title(f'Independent Lyα verification — Δλ = {DELTA_LYA_NM:.6f} nm')

        a4.axis('off')
        table_rows=[
            ['Observed [O III] centroid',f'{NU_ROUNDED_GHZ:.6f} GHz',f'{NU_OFFICIAL_GHZ:.9f} GHz',f'{DELTA_NU_MHZ:.6f} MHz'],
            ['Systemic redshift',f'{Z_DERIVED:.9f}',f'{Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}',f'{DELTA_Z:.9f}'],
            ['Lyα reference',f'{LYA_DERIVED_UM:.9f} µm',f'{LYA_OFFICIAL_UM:.9f} µm',f'{DELTA_LYA_NM:.6f} nm'],
            ['Velocity-equivalent offset','—','—',f'{DV_EQUIV_KMS:.6f} km/s'],
        ]
        table=a4.table(cellText=table_rows,colLabels=['Quantity','Our derivation','Official ALMA','Difference'],loc='center',cellLoc='left',colLoc='left',bbox=[0.02,0.08,0.96,0.80])
        table.auto_set_font_size(False); table.set_fontsize(10.5)
        for (r,c),cell in table.get_celld().items():
            cell.set_edgecolor('#40505f'); cell.set_linewidth(.8)
            cell.set_facecolor('#1c2b38' if r==0 else '#111820')
            cell.get_text().set_color('white')
            if r==0: cell.get_text().set_weight('bold')
            if r>0 and c==1: cell.get_text().set_color('#ff6a45')
            if r>0 and c==2: cell.get_text().set_color('#ffb347')
        a4.set_title('Cross-reference table',fontsize=14,pad=10)

        fig.suptitle('JADES-GS-z11-0 — ALMA-only systemic-redshift and Lyα verification',fontsize=16,y=.996)
        main_png=PNG/f'{VERSION}_ALMA_ONLY_VERIFICATION.png'
        fig.savefig(main_png,dpi=500,bbox_inches='tight')
        plt.show()
        table_png=styled_table_png()
        display(plt.imread(table_png))

        print(f'CODE OUTPUT: {VERSION}')
        print(f'Our derived redshift: z = {Z_DERIVED:.9f}')
        print(f'Official ALMA redshift: z = {Z_OFFICIAL:.7f} ± {Z_SIGMA:.7f}')
        print(f'Our derived Lyα reference: {LYA_DERIVED_UM:.9f} µm')
        print(f'Official Lyα reference: {LYA_OFFICIAL_UM:.9f} µm')
        print(f'Lyα wavelength difference: {DELTA_LYA_NM:.6f} nm')
        print(f'Equivalent centroid offset: {DV_EQUIV_KMS:.6f} km/s')
        print('No peculiar galaxy velocity was subtracted; the available data do not independently measure one.')
        print(f'PNG: {main_png}')
        print(f'TABLE PNG: {table_png}')
        print(f'CSV: {CSV/f"{VERSION}_ALMA_ONLY_RESULTS.csv"}')
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')

run.on_click(draw)
display(run,out)
draw()
