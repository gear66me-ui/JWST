# JWST_0174
import warnings
from pathlib import Path
from datetime import datetime, timezone, timedelta
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets

VERSION = 'JWST_0174'
ROOT = Path('/content/JWST_OUTPUT')
PNG = ROOT / 'PNG'
CSV = ROOT / 'CSV'
for d in (PNG, CSV):
    d.mkdir(parents=True, exist_ok=True)

# Official ALMA [O III] 88 micron systemic measurement for JADES-GS-z11-0.
NU_REST_GHZ = 3393.006244
NU_OBS_GHZ = 279.901
NU_OBS_SIGMA_GHZ = 0.014
Z_OFFICIAL = 11.1221
Z_OFFICIAL_SIGMA = 0.0006
LYA_REST_UM = 0.121567
C_KMS = 299792.458
FWHM_KMS = 29.0

# Direct redshift derivation from the observed ALMA line centroid.
z_direct = NU_REST_GHZ / NU_OBS_GHZ - 1.0
z_direct_sigma = NU_REST_GHZ * NU_OBS_SIGMA_GHZ / (NU_OBS_GHZ ** 2)

# Full-precision centroid corresponding to the published official redshift.
nu_obs_exact = NU_REST_GHZ / (1.0 + Z_OFFICIAL)
lya_systemic_um = LYA_REST_UM * (1.0 + Z_OFFICIAL)

# Frequency rounding audit.
delta_nu_mhz = (NU_OBS_GHZ - nu_obs_exact) * 1e3
delta_z = z_direct - Z_OFFICIAL

# Reconstruct only the ALMA [O III] line profile from the published centroid and FWHM.
fwhm_ghz = NU_OBS_GHZ * FWHM_KMS / C_KMS
sigma_ghz = fwhm_ghz / 2.354820045
freq = np.linspace(NU_OBS_GHZ - 0.10, NU_OBS_GHZ + 0.10, 1600)
line = np.exp(-0.5 * ((freq - NU_OBS_GHZ) / sigma_ghz) ** 2)

summary = pd.DataFrame([
    ['[O III] 88 µm rest frequency', NU_REST_GHZ, 'GHz', 'rest-frame atomic reference'],
    ['Published ALMA observed centroid', NU_OBS_GHZ, 'GHz', 'rounded table value'],
    ['Published centroid uncertainty', NU_OBS_SIGMA_GHZ, 'GHz', '1σ'],
    ['Direct redshift from rounded centroid', z_direct, '', 'νrest/νobs − 1'],
    ['Direct propagated uncertainty', z_direct_sigma, '', 'from centroid uncertainty'],
    ['Official ALMA systemic redshift', Z_OFFICIAL, '', 'reported full-precision fit'],
    ['Official ALMA redshift uncertainty', Z_OFFICIAL_SIGMA, '', 'reported 1σ'],
    ['Centroid implied by official z', nu_obs_exact, 'GHz', 'νrest/(1+z)'],
    ['Systemic Lyα reference from official z', lya_systemic_um, 'µm', '0.121567 × (1+z)'],
    ['Frequency-table rounding difference', delta_nu_mhz, 'MHz', 'rounded centroid minus exact implied centroid'],
    ['Redshift rounding difference', delta_z, '', 'direct rounded-centroid result minus official z'],
], columns=['quantity', 'value', 'unit', 'status'])
summary.to_csv(CSV / f'{VERSION}_ALMA_OIII_SYSTEMIC_REDSHIFT_DERIVATION.csv', index=False)

plt.rcParams.update({
    'figure.facecolor': 'black',
    'axes.facecolor': 'black',
    'savefig.facecolor': 'black',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#aeb8c3',
})

button = widgets.Button(
    description='Plot ALMA z=11.1221 derivation',
    button_style='success',
    layout=widgets.Layout(width='300px')
)
out = widgets.Output()

def draw(_=None):
    with out:
        clear_output(wait=True)
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 1.1, 1.2], hspace=0.30)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        for ax in (ax1, ax2):
            ax.grid(color='#303944', lw=0.6, alpha=0.75)

        ax1.plot(freq, line, lw=2.3, label='Reconstructed ALMA [O III] Gaussian line fit')
        ax1.axvline(NU_OBS_GHZ, ls='--', lw=1.8,
                    label=f'Published rounded centroid: {NU_OBS_GHZ:.3f} ± {NU_OBS_SIGMA_GHZ:.3f} GHz')
        ax1.axvline(nu_obs_exact, ls=':', lw=1.8,
                    label=f'Centroid implied by official z=11.1221: {nu_obs_exact:.6f} GHz')
        ax1.axvspan(NU_OBS_GHZ - NU_OBS_SIGMA_GHZ,
                    NU_OBS_GHZ + NU_OBS_SIGMA_GHZ, alpha=0.15, label='Centroid ±1σ')
        ax1.set_xlabel('Observed [O III] 88 µm frequency [GHz]')
        ax1.set_ylabel('Normalized line amplitude')
        ax1.set_title('Observed ALMA [O III] 88 µm line used to derive the systemic redshift')
        ax1.legend(frameon=False, fontsize=9)

        labels = ['Rounded table centroid', 'Official full-precision fit']
        values = [z_direct, Z_OFFICIAL]
        ax2.scatter(values, [1, 0], s=110)
        ax2.errorbar(Z_OFFICIAL, 0, xerr=Z_OFFICIAL_SIGMA, fmt='none', capsize=5, lw=1.5)
        ax2.errorbar(z_direct, 1, xerr=z_direct_sigma, fmt='none', capsize=5, lw=1.5)
        for yv, value, label in zip([1, 0], values, labels):
            ax2.text(value + 0.00012, yv, f'{label}: z={value:.6f}', va='center', fontsize=11)
        lo = min(values) - 0.0010
        hi = max(values) + 0.0012
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(-0.7, 1.7)
        ax2.set_yticks([])
        ax2.set_xlabel('Systemic redshift z')
        ax2.set_title('The official value is 11.1221; the small difference is table-frequency rounding')

        ax3.axis('off')
        derivation = (
            f'Observed quantity: ALMA [O III] 88 µm line centroid\n\n'
            f'ν_rest = {NU_REST_GHZ:.6f} GHz\n'
            f'ν_obs(table) = {NU_OBS_GHZ:.3f} ± {NU_OBS_SIGMA_GHZ:.3f} GHz\n\n'
            f'z = ν_rest / ν_obs − 1\n'
            f'z = {NU_REST_GHZ:.6f} / {NU_OBS_GHZ:.3f} − 1\n'
            f'z = {z_direct:.6f} ± {z_direct_sigma:.6f}\n\n'
            f'Official full-precision ALMA fit: z = {Z_OFFICIAL:.4f} ± {Z_OFFICIAL_SIGMA:.4f}\n'
            f'Exact centroid implied by z=11.1221: {nu_obs_exact:.6f} GHz\n'
            f'Nominal systemic Lyα wavelength: {lya_systemic_um:.6f} µm\n\n'
            f'No z=11.38 curve is included in this widget.'
        )
        ax3.text(0.5, 0.52, derivation, ha='center', va='center', fontsize=14,
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#111820',
                           edgecolor='#72808e', alpha=0.96))

        fig.suptitle(
            'JADES-GS-z11-0 — independent ALMA [O III] systemic-redshift derivation\n'
            'Official revised result: z = 11.1221 ± 0.0006',
            fontsize=16, y=0.995
        )
        out_png = PNG / f'{VERSION}_ALMA_OIII_SYSTEMIC_REDSHIFT_DERIVATION.png'
        fig.savefig(out_png, dpi=500, bbox_inches='tight')
        plt.show()

        now = datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S UTC-05')
        print(f'CODE OUTPUT: {VERSION}')
        print(summary.to_string(index=False, float_format=lambda v: f'{v:.6f}'))
        print(f'PNG: {out_png}')
        print(f'CSV: {CSV / f"{VERSION}_ALMA_OIII_SYSTEMIC_REDSHIFT_DERIVATION.csv"}')
        print(f'Timestamp Colombia: {now}')
        print(f'# {VERSION}')

button.on_click(draw)
display(button, out)
draw()
