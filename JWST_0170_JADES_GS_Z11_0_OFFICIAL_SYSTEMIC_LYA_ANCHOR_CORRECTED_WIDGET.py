# JWST_0170
import requests

SOURCE_URL = 'https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0169_JADES_GS_Z11_0_TLUSTY_CLOUDY_PHYSICAL_COMPARISON_FIXED_WIDGET.py'
source = requests.get(SOURCE_URL, timeout=120).text
source = source.replace("VERSION='JWST_0169'", "VERSION='JWST_0170'")
source = source.replace('JWST_0169_', 'JWST_0170_')
source = source.replace("['Modern line+DLA spectroscopic result','reported',11.122,np.nan,np.nan,22.42,np.nan]", "['Official ALMA [O III] systemic redshift','reported',11.1221,np.nan,np.nan,22.42,np.nan]")
source = source.replace("a0.axvline(11.122,color='#8fd3ff',ls=':',lw=1.2,label='Modern line+DLA z=11.122')", "a0.axvline(11.1221,color='#8fd3ff',ls=':',lw=1.4,label='Official ALMA systemic z=11.1221')")
source = source.replace("lya_phys=LYA*(1+z_phys); lya_emp=LYA*(1+z_emp); lya_paper=LYA*(1+ZP)", "lya_phys=LYA*(1+z_phys); lya_emp=LYA*(1+z_emp); lya_paper=LYA*(1+ZP); z_systemic=11.1221; lya_systemic=LYA*(1+z_systemic)")
source = source.replace("label=f'Surrogate Lyα anchor {lya_phys:.6f} µm'", "label=f'Surrogate grid-boundary anchor {lya_phys:.6f} µm (not systemic)'")
source = source.replace("a1.axvline(lya_emp,color='#e68645',ls=':',lw=1.2,label=f'Empirical anchor {lya_emp:.6f} µm')", "a1.axvline(lya_emp,color='#e68645',ls=':',lw=1.2,label=f'Empirical continuum-break anchor {lya_emp:.6f} µm')\n        a1.axvline(lya_systemic,color='#8fd3ff',ls='-.',lw=1.8,label=f'Official ALMA systemic Lyα reference {lya_systemic:.6f} µm (z=11.1221)')")
source = source.replace("labels=['Modern line+DLA','Physical surrogate','Empirical break','Paper SMDS fit']; vals=[11.122,z_phys,z_emp,ZP]", "labels=['Official ALMA systemic','Surrogate grid result','Empirical break','Paper SMDS fit']; vals=[11.1221,z_phys,z_emp,ZP]")
source = source.replace("a3.set_title(f'Best physical-surrogate parameters: T={t_phys:.0f} K, nebular fraction={nf_phys:.2f}, log N(H I)={lnhi_phys:.2f}')", "a3.set_title(f'Surrogate best grid result: T={t_phys:.0f} K, nebular fraction={nf_phys:.2f}, log N(H I)={lnhi_phys:.2f} — official systemic z=11.1221')")
source = source.replace("Physical surrogate only: exact author TLUSTY/CLOUDY grid and fitting code are not publicly released", "Official ALMA systemic anchor added at 1.473647 µm; surrogate grid result remains model-dependent")
exec(compile(source, 'JWST_0170_runtime.py', 'exec'), globals())
