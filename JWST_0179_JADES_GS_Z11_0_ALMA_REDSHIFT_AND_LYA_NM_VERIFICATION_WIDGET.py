# JWST_0179
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

VERSION='JWST_0179'
OUT=Path('/content/JWST_OUTPUT')
PNG=OUT/'PNG'; CSV=OUT/'CSV'
for d in (PNG,CSV): d.mkdir(parents=True,exist_ok=True)

# Published ALMA [O III] 88 micron quantities used for this verification only.
NU_REST_GHZ=3393.006244
NU_ROUNDED_GHZ=279.901000
NU_ROUNDED_SIGMA_GHZ=0.014000
Z_OFFICIAL=11.1221000
Z_OFFICIAL_SIGMA=0.0006000
LYA_REST_UM=0.121567
C_KMS=299792.458

# Direct derivation from the rounded published centroid.
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
    ['Official full-precision ALMA result',NU_OFFICIAL_GHZ,Z_OFFICIAL,LYA_OFFICIAL_UM,0.0,0.0,DV_EQUIV_KMS],
],columns=['method','observed_OIII_frequency_GHz','redshift_z','lya_reference_um','delta_z_from_official','delta_lya_nm_from_official','equivalent_frequency_offset_km_s'])
summary.to_csv(CSV/f'{VERSION}_ALMA_REDSHIFT_LYA_VERIFICATION.csv',index=False)

plt.rcParams.update({
    'figure.facecolor':'black','axes.facecolor':'black','savefig.facecolor':'black',
    'text.color':'white','axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','axes.edgecolor':'#aeb8c3'
})

run=widgets.Button(description='Plot ALMA-only derivation',button_style='success',layout=widgets.Layout(width='280px'))
out=widgets.Output()

def make_table_png():
    fig,ax=plt.subplots(figsize=(16,4.8))
    ax.axis('off')
    show=pd.DataFrame({
        'Quantity':['Observed [O III] centroid','Derived systemic redshift','Lyα reference wavelength','Difference from official','Equivalent frequency offset'],
        'Our derivation':[f'{NU_ROUNDED_GHZ:.6f} GHz',f'{Z_DERIVED:.9f}',f'{LYA_DERIVED_UM:.9f} µm',f'{DELTA_Z:.9f}',f'{DV_EQUIV_KMS:.6f} km/s'],
        'Official ALMA':[f'{NU_OFFICIAL_GHZ:.9f} GHz',f'{Z_OFFICIAL:.7f} ± {Z_OFFICIAL_SIGMA:.7f}',f'{LYA_OFFICIAL_UM:.9f} µm','reference','reference'],
        'Difference / note':[f'{DELTA_NU_MHZ:.6f} MHz',f'{DELTA_Z:.9f}',f'{DELTA_LYA_NM:.6f} nm','rounded-centroid effect','not a peculiar-velocity measurement']
    })
    table=ax.table(cellText=show.values,colLabels=show.columns,loc='center',cellLoc='left',colLoc='left',bbox=[0.01,0.05,0.98,0.82])
    table.auto_set_font_size(False); table.set_fontsize(10.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#40505f'); cell.set_linewidth(.8)
        cell.set_facecolor('#111820' if r else '#1c2b38')
        cell.get_text().set_color('white')
        if r==0: cell.get_text().set_weight('bold')
    ax.set_title('JADES-GS-z11-0 — ALMA systemic-redshift and Lyα cross-reference table',fontsize=15,pad=14)
    path=PNG/f'{VERSION}_CROSS_REFERENCE_TABLE.png'
    fig.savefig(path,dpi=500,bbox_inches='tight'); plt.close(fig)
    return path

def draw(_=None):
    with out:
        clear_output(wait=True)
        print('REQUEST SUMMARY:')
        print('• Show only the revised ALMA systemic-redshift verification.')
        print('• Compare our rounded-centroid derivation with the official full-precision value.')
        print('• Express the Lyα wavelength difference in nanometers, not as a misleading micrometer-scale gap.')
        print('• Keep the official Lyα marker on the left and our verification on the right.')
        print('• Save a separate styled PNG table for cross-reference.')

        fig=plt.figure(figsize=(17,14))
        gs=fig.add_gridspec(3,1,height_ratios=[1.15,1.05,1.25],hspace=.31)
        a1,a2,a3=[fig.add_subplot(gs[i]) for i in range(3)]
        for a in (a1,a2,a3): a.grid(color='#303944',lw=.6,alpha=.75)

        # Panel 1: frequency centroids only; no unrelated spectrum or model curve.
        freq_vals=[NU_ROUNDED_GHZ,NU_OFFICIAL_GHZ]
        ypos=[1,0]
        cols=['#ff5a36','#ffb347']
        labels=['Rounded published centroid','Official-z implied centroid']
        a1.scatter(freq_vals,ypos,s=125,color=cols,zorder=4)
        for yy,v,lab,col,off in zip(ypos,freq_vals,labels,cols,[(-24,-28),(24,28)]):
            a1.annotate(f'{lab}\n{v:.9f} GHz',xy=(v,yy),xytext=off,textcoords='offset points',
                        ha='right' if off[0]<0 else 'left',va='top' if off[1]<0 else 'bottom',
                        color=col,fontsize=11,arrowprops=dict(arrowstyle='-',color=col,lw=.9))
        a1.set_yticks([])
        a1.set_xlim(min(freq_vals)-0.003,max(freq_vals)+0.003)
        a1.set_xlabel('Observed [O III] 88 µm frequency [GHz]')
        a1.set_title(f'ALMA centroid comparison — Δν = {DELTA_NU_MHZ:.6f} MHz')

        # Panel 2: our redshift derivation versus official value.
        zs=[Z_OFFICIAL,Z_DERIVED]
        zcols=['#ffb347','#ff5a36']
        a2.scatter(zs,[0,1],s=125,color=zcols,zorder=4)
        a2.annotate(f'Official ALMA\nz = {Z_OFFICIAL:.7f} ± {Z_OFFICIAL_SIGMA:.7f}',
                    xy=(Z_OFFICIAL,0),xytext=(-26,24),textcoords='offset points',ha='right',va='bottom',
                    color=zcols[0],fontsize=11,arrowprops=dict(arrowstyle='-',color=zcols[0],lw=.9))
        a2.annotate(f'Our rounded-centroid derivation\nz = {Z_DERIVED:.9f}',
                    xy=(Z_DERIVED,1),xytext=(26,-24),textcoords='offset points',ha='left',va='top',
                    color=zcols[1],fontsize=11,arrowprops=dict(arrowstyle='-',color=zcols[1],lw=.9))
        a2.set_yticks([]); a2.set_xlim(min(zs)-0.0008,max(zs)+0.0008)
        a2.set_xlabel('Systemic redshift z')
        a2.set_title(f'Our derivation versus official result — Δz = {DELTA_Z:.9f}')

        # Panel 3: wavelength difference plotted in nanometers relative to the official anchor.
        offsets_nm=[0.0,DELTA_LYA_NM]
        a3.axvline(0,color='#ffb347',ls='-.',lw=1.8)
        a3.axvline(DELTA_LYA_NM,color='#ff5a36',ls='--',lw=1.8)
        a3.scatter(offsets_nm,[0,1],s=125,color=['#ffb347','#ff5a36'],zorder=4)
        a3.annotate(f'Official Lyα reference\n{LYA_OFFICIAL_UM:.9f} µm\n0.000000 nm offset',
                    xy=(0,0),xytext=(-34,28),textcoords='offset points',ha='right',va='bottom',
                    color='#ffb347',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ffb347',lw=.9))
        a3.annotate(f'Our Lyα verification\n{LYA_DERIVED_UM:.9f} µm\n+{DELTA_LYA_NM:.6f} nm',
                    xy=(DELTA_LYA_NM,1),xytext=(34,-28),textcoords='offset points',ha='left',va='top',
                    color='#ff5a36',fontsize=11,arrowprops=dict(arrowstyle='-',color='#ff5a36',lw=.9))
        pad=max(abs(DELTA_LYA_NM)*0.85,0.004)
        a3.set_xlim(-pad,DELTA_LYA_NM+pad); a3.set_yticks([])
        a3.set_xlabel('Lyα wavelength offset from official reference [nm]')
        a3.set_title(f'Independent Lyα verification — difference = {DELTA_LYA_NM:.6f} nm')

        fig.suptitle('JADES-GS-z11-0 — ALMA systemic-redshift and Lyα verification',fontsize=16,y=.995)
        main_png=PNG/f'{VERSION}_ALMA_REDSHIFT_LYA_NM_VERIFICATION.png'
        fig.savefig(main_png,dpi=500,bbox_inches='tight')
        plt.show()
        table_png=make_table_png()
        display(plt.imread(table_png))

        print(f'CODE OUTPUT: {VERSION}')
        print(f'Our derived redshift: z = {Z_DERIVED:.9f}')
        print(f'Official ALMA redshift: z = {Z_OFFICIAL:.7f} ± {Z_OFFICIAL_SIGMA:.7f}')
        print(f'Our derived Lyα reference: {LYA_DERIVED_UM:.9f} µm')
        print(f'Official Lyα reference: {LYA_OFFICIAL_UM:.9f} µm')
        print(f'Lyα difference: {DELTA_LYA_NM:.6f} nm')
        print(f'Equivalent rounded-centroid frequency offset: {DV_EQUIV_KMS:.6f} km/s')
        print('No peculiar velocity of the galaxy was subtracted; these data do not independently measure one.')
        print(f'PNG: {main_png}')
        print(f'TABLE PNG: {table_png}')
        print(f'CSV: {CSV/f"{VERSION}_ALMA_REDSHIFT_LYA_VERIFICATION.csv"}')
        print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
        print(f'# {VERSION}')

run.on_click(draw)
display(run,out)
draw()
