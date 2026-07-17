# JWST_0177
import requests

SOURCE_URL='https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0176_JADES_GS_Z11_0_FULL_SPECTRUM_ALMA_VELOCITY_CORRECTED_WIDGET.py'
source=requests.get(SOURCE_URL,timeout=120).text
source=source.replace("VERSION='JWST_0176'","VERSION='JWST_0177'")
source=source.replace('JWST_0176_','JWST_0177_')
source=source.replace("a3.annotate(f'{lab}: z={v:.5f}',xy=(v,yy),xytext=off,textcoords='offset points',va='center',fontsize=10,color=col,arrowprops=dict(arrowstyle='-',lw=.8,color=col))",
"a3.annotate(f'{lab}: z={v:.6f}',xy=(v,yy),xytext=off,textcoords='offset points',va='center',fontsize=10,color=col,arrowprops=dict(arrowstyle='-',lw=.8,color=col))")
source=source.replace("for yy,v,lab,col,off in zip([2,1,0],vals,labels,['#8fd3ff','#e68645','#ff6b6b'],[(18,14),(18,0),(18,-14)]):",
"for yy,v,lab,col,off in zip([2,1,0],vals,labels,['#8fd3ff','#e68645','#ff6b6b'],[(18,-18),(18,8),(18,22)]):")
source=source.replace("a3.set_yticks([]); a3.set_xlim(min(vals)-.04,max(vals)+.08); a3.set_xlabel('Redshift z')",
"a3.set_yticks([]); a3.set_ylim(-0.45,2.35); a3.set_xlim(min(vals)-.04,max(vals)+.08); a3.set_xlabel('Redshift z')")
source=source.replace("a4.legend(frameon=False,fontsize=8.5,ncol=2)","a4.legend(frameon=False,fontsize=8.5,ncol=2,loc='upper right')")
source=source.replace("a4.text(.02,.95,txt,transform=a4.transAxes,va='top',fontsize=10,",
"a4.text(.02,.06,txt,transform=a4.transAxes,va='bottom',ha='left',fontsize=10,")
source=source.replace("txt=(f'Rounded centroid → z={Z_TABLE:.9f}\\nOfficial z={ZSYS:.7f} → ν={NU_EXACT:.9f} GHz\\n'",
"txt=(f'DERIVED FROM PUBLISHED ROUNDED CENTROID: z={Z_TABLE:.9f}\\nOFFICIAL FULL-PRECISION RESULT: z={ZSYS:.7f}\\nOfficial-z implied centroid: ν={NU_EXACT:.9f} GHz\\n'")
source=source.replace("f'Δν={DNU_MHZ:.6f} MHz   Δz={DZ:.9f}   equivalent offset={DV_REL:.6f} km/s')",
"f'Δν={DNU_MHZ:.6f} MHz   Δz={DZ:.9f}\\nEquivalent centroid-rounding velocity offset={DV_REL:.6f} km/s\\nThis is not an independently measured galaxy peculiar-velocity correction.')")
source=source.replace("fig.suptitle('JADES-GS-z11-0 — full-spectrum systemic-redshift analysis with ALMA centroid correction\\nAll JWST panels retained; ALMA correction added as a fourth panel',fontsize=15,y=.996)",
"fig.suptitle(f'JADES-GS-z11-0 — full-spectrum systemic-redshift analysis with ALMA centroid correction\\nDerived rounded-centroid redshift z={Z_TABLE:.9f}; official full-precision redshift z={ZSYS:.7f}',fontsize=15,y=.996)")
source=source.replace("print(f'CODE OUTPUT: {VERSION}')",
"print(f'CODE OUTPUT: {VERSION}')\n        print(f'DERIVED REDSHIFT FROM PUBLISHED ROUNDED CENTROID: {Z_TABLE:.9f}')\n        print(f'OFFICIAL FULL-PRECISION SYSTEMIC REDSHIFT: {ZSYS:.7f} ± {ZSYS_SIG:.7f}')\n        print(f'REDSHIFT DIFFERENCE: {DZ:.9f}')\n        print(f'EQUIVALENT CENTROID-ROUNDING VELOCITY OFFSET: {DV_REL:.6f} km/s')")
exec(compile(source,'JWST_0177_runtime.py','exec'),globals())
