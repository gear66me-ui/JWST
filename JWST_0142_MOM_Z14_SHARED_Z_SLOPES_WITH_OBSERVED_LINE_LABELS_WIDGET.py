# JWST_0142
import requests

VERSION = "JWST_0142"
SOURCE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/JWST_0141_MOM_Z14_SHARED_Z_COMPONENT_SLOPES_WIDGET.py"

response = requests.get(SOURCE_URL, timeout=120)
response.raise_for_status()
source = response.text
source = source.replace("# JWST_0141", "# JWST_0142", 1)
source = source.replace('VERSION="JWST_0141"', 'VERSION="JWST_0142"')
source = source.replace(
    'axs.fill_between(lam,y-e,y+e,step="mid",color="#9a9a9a",alpha=.35,label="1σ uncertainty"); axs.step(lam,y,where="mid",color="#67e0d1",lw=1.6,label="DJA spectrum"); axs.plot(lam,best_model,color="#ffd400",lw=2,label=f"Best joint model, z={z_best:.4f}"); axs.plot(lam,continuum,color="white",ls=":",lw=1.5,label=f"Continuum slope={slope_coeff:.4f} per μm")',
    'axs.fill_between(lam,y-e,y+e,step="mid",color="#9a9a9a",alpha=.35,label="1σ uncertainty"); axs.step(lam,y,where="mid",color="#67e0d1",lw=1.6,label="DJA spectrum"); axs.plot(lam,best_model,color="#ffd400",lw=2,label=f"Best joint model, z={z_best:.4f}"); axs.plot(lam,continuum,color="white",ls=":",lw=1.5,label=f"Continuum slope={slope_coeff:.4f} per μm")\n        lya_x=REST["Lyα break"]*(1+z_best)\n        if lo_lam<=lya_x<=hi_lam:\n            axs.axvline(lya_x,color="white",ls="--",lw=1.15,alpha=.95,zorder=6)\n            axs.text(lya_x,.975,"Lyα",rotation=90,transform=axs.get_xaxis_transform(),ha="right",va="top",color="white",fontsize=10,fontweight="bold")\n        for obs_name in selected:\n            obs_x=REST[obs_name]*(1+z_best)\n            if lo_lam<=obs_x<=hi_lam:\n                axs.axvline(obs_x,color="white",ls="--",lw=.9,alpha=.92,zorder=6)\n                axs.text(obs_x,.975,obs_name,rotation=90,transform=axs.get_xaxis_transform(),ha="right",va="top",color="white",fontsize=9)'
)
source = source.replace("SHARED-z LIKELIHOOD + INDIVIDUAL LINE COMPONENT SLOPES", "SHARED-z LIKELIHOOD + SLOPES + OBSERVED-LINE LABELS")
source = source.replace("SHARED_Z_COMPONENT_SLOPES", "SHARED_Z_SLOPES_OBSERVED_LINE_LABELS")

exec(compile(source, SOURCE_URL, "exec"), globals(), globals())
