# JWST_0153
import io, os, sys, warnings, contextlib, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")

for pkg in ["numpy", "pandas", "matplotlib", "requests", "scipy"]:
    try:
        __import__(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d

VERSION = "JWST_0153"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
for d in (PNG, CSV, DATA):
    d.mkdir(parents=True, exist_ok=True)

# Reported ALMA [O III] 88 micron line measurement (Schouws et al.)
NU_REST_GHZ = 3393.006244
NU_OBS_GHZ = 223.528
SIGMA_NU_GHZ = 0.009
Z_REPORTED = 14.1793
SIGMA_Z_REPORTED = 0.0007
LYA_REST_UM = 0.121567
CIII_REST_UM = 0.1908

# Exact public JWST/NIRSpec 1D spectrum cited by the paper
SPEC_URL = "https://zenodo.org/records/12578543/files/JADES-GS-z14-0_spec1D.txt?download=1"
SPEC_PATH = DATA / "JADES-GS-z14-0_spec1D.txt"
if (not SPEC_PATH.exists()) or SPEC_PATH.stat().st_size < 10000:
    with requests.get(SPEC_URL, timeout=120) as r:
        r.raise_for_status()
        SPEC_PATH.write_bytes(r.content)

# Flexible parser for the published ASCII table
raw = np.genfromtxt(SPEC_PATH, comments="#", invalid_raise=False)
if raw.ndim == 1:
    raw = raw[None, :]
raw = raw[np.all(np.isfinite(raw[:, :min(raw.shape[1], 3)]), axis=1)]
if raw.shape[1] < 3:
    raise RuntimeError("Published Zenodo spectrum does not contain at least wavelength, flux, uncertainty columns")
wave = raw[:, 0].astype(float)
flux = raw[:, 1].astype(float)
err = np.abs(raw[:, 2].astype(float))
if np.nanmedian(wave) > 100:
    wave = wave / 1e4
mask = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
wave, flux, err = wave[mask], flux[mask], err[mask]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]

# ALMA redshift derivation: 1 + z = nu_rest / nu_obs
z_derived = NU_REST_GHZ / NU_OBS_GHZ - 1.0
sigma_z = NU_REST_GHZ * SIGMA_NU_GHZ / (NU_OBS_GHZ ** 2)

# Build an ALMA frequency likelihood directly from the published line centroid
nu_grid = np.linspace(NU_OBS_GHZ - 0.08, NU_OBS_GHZ + 0.08, 1201)
like_nu = np.exp(-0.5 * ((nu_grid - NU_OBS_GHZ) / SIGMA_NU_GHZ) ** 2)
like_nu /= np.trapz(like_nu, nu_grid)
z_grid = NU_REST_GHZ / nu_grid - 1.0

# DLA-informed display model for the exact JWST spectrum.
# Redshift is fixed by ALMA; the fit only adjusts continuum and damping-wing shape.
sel = (wave >= 1.60) & (wave <= 2.35)
lam, y, e = wave[sel], flux[sel], err[sel]
lya_obs = LYA_REST_UM * (1.0 + z_derived)

# Normalize only for stable optimization; plotted values retain original units.
scale = np.nanmedian(np.abs(y[(lam > 2.0) & (lam < 2.3)]))
if not np.isfinite(scale) or scale == 0:
    scale = np.nanpercentile(np.abs(y), 75)
if not np.isfinite(scale) or scale == 0:
    scale = 1.0
yn, en = y / scale, e / scale

# Phenomenological damping-wing model anchored to the ALMA systemic redshift.
def model(theta):
    amp, beta, tau0, wing_um, floor = theta
    continuum = amp * np.clip(lam / 2.10, 0.1, None) ** beta
    blue_cut = 0.5 * (1.0 + np.tanh((lam - lya_obs) / 0.010))
    delta = np.maximum(lam - lya_obs, 1e-4)
    damping = np.exp(-tau0 * (wing_um / delta) ** 2)
    transmission = np.where(lam < lya_obs, floor, damping)
    return continuum * blue_cut * transmission

def residual(theta):
    return (yn - model(theta)) / en

x0 = np.array([1.0, -2.0, 0.25, 0.025, 0.01])
lo = np.array([0.0, -6.0, 0.0, 0.001, 0.0])
hi = np.array([10.0, 4.0, 50.0, 0.20, 0.30])
fit = least_squares(residual, x0, bounds=(lo, hi), max_nfev=5000)
best_model = model(fit.x) * scale
best_model = gaussian_filter1d(best_model, sigma=0.8)

# Save audit tables
pd.DataFrame({
    "quantity": ["nu_rest_GHz", "nu_obs_GHz", "sigma_nu_GHz", "z_derived", "sigma_z_propagated", "z_reported", "sigma_z_reported"],
    "value": [NU_REST_GHZ, NU_OBS_GHZ, SIGMA_NU_GHZ, z_derived, sigma_z, Z_REPORTED, SIGMA_Z_REPORTED],
    "status": ["laboratory", "reported ALMA", "reported ALMA", "derived", "propagated", "reported", "reported"]
}).to_csv(CSV / f"{VERSION}_ALMA_REDSHIFT_AUDIT.csv", index=False)

pd.DataFrame({
    "wavelength_um": wave,
    "flux": flux,
    "uncertainty": err
}).to_csv(CSV / f"{VERSION}_ZENODO_JWST_SPECTRUM.csv", index=False)

# Plot styling
plt.rcParams.update({
    "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
    "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
    "ytick.color": "white", "axes.edgecolor": "white", "font.size": 10
})

fig = plt.figure(figsize=(16, 15), facecolor="black")
gs = fig.add_gridspec(3, 1, height_ratios=[1.15, 1.35, 2.2], hspace=0.25)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
for ax in (ax0, ax1, ax2):
    ax.set_facecolor("black")
    for sp in ax.spines.values():
        sp.set_color("white")
    ax.grid(color="#303030", lw=0.6, alpha=0.8)

# Panel 1: ALMA line-centroid likelihood and redshift derivation
ax0.plot(nu_grid, like_nu, color="#67e0d1", lw=2.0,
         label=f"Published ALMA centroid: {NU_OBS_GHZ:.3f} ± {SIGMA_NU_GHZ:.3f} GHz")
ax0.axvline(NU_OBS_GHZ, color="#ffd400", lw=1.5)
ax0.fill_between(nu_grid, 0, like_nu,
                 where=(nu_grid >= NU_OBS_GHZ-SIGMA_NU_GHZ) & (nu_grid <= NU_OBS_GHZ+SIGMA_NU_GHZ),
                 color="#ffd400", alpha=0.18)
ax0.set_xlabel("Observed [O III] 88 μm frequency [GHz]")
ax0.set_ylabel("Normalized likelihood density")
ax0.set_title("ALMA [O III] 88 μm line centroid → systemic redshift", fontsize=13)
ax0.text(0.985, 0.92,
         f"1 + z = νrest / νobs\nνrest = {NU_REST_GHZ:.6f} GHz\n"
         f"z = {z_derived:.7f}\nσz = {sigma_z:.7f}",
         transform=ax0.transAxes, ha="right", va="top", fontsize=11,
         bbox=dict(facecolor="black", edgecolor="#777", alpha=0.9, pad=8))
lg = ax0.legend(facecolor="black", edgecolor="#777", loc="upper left")
for t in lg.get_texts(): t.set_color("white")

# Panel 2: exact frequency-to-redshift mapping and JWST cross-checks
order_z = np.argsort(z_grid)
ax1.plot(nu_grid[order_z], z_grid[order_z], color="#ffd400", lw=2.0,
         label="Exact relation: z = νrest/νobs − 1")
ax1.axhline(Z_REPORTED, color="#ff6666", ls="--", lw=1.0,
            label=f"Published systemic z = {Z_REPORTED:.4f} ± {SIGMA_Z_REPORTED:.4f}")
ax1.scatter([NU_OBS_GHZ], [z_derived], s=85, marker="D",
            facecolors="none", edgecolors="white", linewidths=1.3, zorder=5,
            label=f"Derived point: z = {z_derived:.7f}")
ax1.fill_between(nu_grid[order_z], Z_REPORTED-SIGMA_Z_REPORTED,
                 Z_REPORTED+SIGMA_Z_REPORTED, color="#ff3333", alpha=0.14)
ax1.set_xlabel("Observed [O III] 88 μm frequency [GHz]")
ax1.set_ylabel("Spectroscopic redshift, z")
ax1.set_title("Laboratory rest frequency mapped to the reported ALMA line centroid", fontsize=12)
lg = ax1.legend(facecolor="black", edgecolor="#777", loc="upper left")
for t in lg.get_texts(): t.set_color("white")

# Panel 3: exact released JWST spectrum, with ALMA systemic redshift anchoring the DLA model
ax2.fill_between(lam, y-e, y+e, step="mid", color="#888", alpha=0.28, label="Published 1σ uncertainty")
ax2.step(lam, y, where="mid", color="#67e0d1", lw=1.15, label="Zenodo JWST/NIRSpec PRISM spectrum")
ax2.plot(lam, best_model, color="#ffd400", lw=1.55,
         label=f"DLA-informed continuum model fixed at ALMA z={z_derived:.4f}")
ax2.axvline(lya_obs, color="white", ls="--", lw=0.9)
ax2.text(lya_obs, 0.97, "Systemic Lyα wavelength", rotation=90,
         transform=ax2.get_xaxis_transform(), ha="right", va="top", fontsize=9)
ciii_obs = CIII_REST_UM * (1.0 + z_derived)
ax2.axvline(ciii_obs, color="#72e0ff", ls=":", lw=0.9)
ax2.text(ciii_obs, 0.97, "C III] expected", rotation=90,
         transform=ax2.get_xaxis_transform(), ha="right", va="top", fontsize=9, color="#72e0ff")
ax2.set_xlim(1.60, 2.35)
ax2.set_xlabel("Observed wavelength [μm]")
ax2.set_ylabel("Published flux density [native table units]")
ax2.set_title("Exact released JWST spectrum — ALMA fixes z; the DLA wing shapes the UV break", fontsize=12)
lg = ax2.legend(facecolor="black", edgecolor="#777", loc="best")
for t in lg.get_texts(): t.set_color("white")

fig.suptitle("JADES-GS-z14-0 — scientific derivation of z = 14.1793 from ALMA [O III] 88 μm", fontsize=15, y=0.995)
fig.tight_layout(rect=[0.03, 0.02, 0.995, 0.98])
out_png = PNG / f"{VERSION}_JADES_GS_Z14_0_ALMA_OIII_REDSHIFT.png"
fig.savefig(out_png, dpi=450, bbox_inches="tight", facecolor="black")
plt.show()
