# JWST_0169
import requests

SOURCE_URL = 'https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0168_JADES_GS_Z11_0_TLUSTY_CLOUDY_PHYSICAL_COMPARISON_WIDGET.py'
source = requests.get(SOURCE_URL, timeout=120).text
source = source.replace("VERSION='JWST_0168'", "VERSION='JWST_0169'")
source = source.replace(
"profile=allgrid.groupby('z',as_index=False)['chi2'].min()\np=np.exp(np.clip(-0.5*(profile.chi2-profile.chi2.min()),-700,0)); p/=np.trapz(p,profile.z)\nc=np.cumsum(p); c/=c[-1]\nz16,z50,z84=[np.interp(q,c,profile.z) for q in (0.16,0.50,0.84)]",
"profile=allgrid.groupby('z',as_index=False)['chi2'].min().sort_values('z').reset_index(drop=True)\nz_profile=profile['z'].to_numpy(dtype=float)\nchi_profile=profile['chi2'].to_numpy(dtype=float)\ngood=np.isfinite(z_profile)&np.isfinite(chi_profile)\nz_profile=z_profile[good]; chi_profile=chi_profile[good]\nif z_profile.size<3: raise RuntimeError('Physical-grid redshift profile has fewer than 3 finite points')\np=np.exp(np.clip(-0.5*(chi_profile-np.nanmin(chi_profile)),-700,0))\narea=np.trapz(p,z_profile)\nif not np.isfinite(area) or area<=0: raise RuntimeError('Physical-grid posterior normalization failed')\np=p/area\nc=np.cumsum(p)\nif not np.isfinite(c[-1]) or c[-1]<=0: raise RuntimeError('Physical-grid cumulative posterior failed')\nc=c/c[-1]\nz16,z50,z84=[np.interp(q,c,z_profile) for q in (0.16,0.50,0.84)]"
)
source = source.replace("a0.plot(profile.z,p,", "a0.plot(z_profile,p,")
source = source.replace('JWST_0168_', 'JWST_0169_')
exec(compile(source, 'JWST_0169_runtime.py', 'exec'), globals())
