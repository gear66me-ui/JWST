# JWST_0112
# Audit reference: interactive N III] G235H spectrum for verified MoM-z14 source 10003.

from pathlib import Path
import numpy as np
from astropy.io import fits
import plotly.graph_objects as go

VERSION = "JWST_0112"
print(f"CODE OUTPUT: {VERSION}")

Z_ADOPTED = 9.31102
REST_MIN_A = 1700.0
REST_MAX_A = 1800.0
NIII_LINES_A = [1746.82, 1748.65, 1749.67, 1752.16, 1753.99]

ROOTS = [
    Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS"),
    Path("/content/JWST_BACKUP_REPO/FITS_DATA/MOM_Z14_SOURCE_10003"),
]
FILENAME = "hlsp_glass-jwst_jwst_nirspec_abell2744-1324-10003_f170lp-g235h_v1_spec.fits"

matches = []
for root in ROOTS:
    if root.exists():
        matches.extend(root.rglob(FILENAME))

# Deduplicate exact paths.
unique = []
seen = set()
for p in matches:
    s = str(p.resolve())
    if s not in seen:
        unique.append(p)
        seen.add(s)

if not unique:
    raise FileNotFoundError(FILENAME)

path = unique[0]

with fits.open(path, memmap=False) as hdul:
    tab = hdul["SPEC1D"].data
    wave_um = np.asarray(tab["wave"], dtype=float)
    flux = np.asarray(tab["flux"], dtype=float)
    err = np.asarray(tab["err"], dtype=float)

rest_a = wave_um * 10000.0 / (1.0 + Z_ADOPTED)
valid = (
    np.isfinite(wave_um)
    & np.isfinite(flux)
    & np.isfinite(err)
    & (err > 0.0)
    & (flux != 0.0)
    & (rest_a >= REST_MIN_A)
    & (rest_a <= REST_MAX_A)
)

if not np.any(valid):
    raise RuntimeError("No valid N III] samples in verified source-10003 G235H spectrum.")

wave_um = wave_um[valid]
flux = flux[valid]
err = err[valid]
rest_a = rest_a[valid]
order = np.argsort(wave_um)
wave_um = wave_um[order]
flux = flux[order]
err = err[order]
rest_a = rest_a[order]

# Redshift inferred if the hovered wavelength is identified with the central N III] 1749.67 A component.
z_from_central = wave_um * 10000.0 / 1749.67 - 1.0
custom = np.column_stack((wave_um * 10000.0, rest_a, err, z_from_central))

fig = go.FigureWidget()
fig.add_trace(
    go.Scatter(
        x=wave_um,
        y=flux,
        mode="lines+markers",
        name="Native G235H flux",
        line=dict(width=1),
        marker=dict(size=4),
        error_y=dict(type="data", array=err, visible=False),
        customdata=custom,
        hovertemplate=(
            "Observed: %{x:.7f} µm<br>"
            "Observed: %{customdata[0]:.3f} Å<br>"
            "Rest at z=9.31102: %{customdata[1]:.3f} Å<br>"
            "Flux: %{y:.6g}<br>"
            "1σ: %{customdata[2]:.6g}<br>"
            "z if N III] 1749.67 Å: %{customdata[3]:.6f}"
            "<extra></extra>"
        ),
    )
)

for line_a in NIII_LINES_A:
    obs_um = line_a * (1.0 + Z_ADOPTED) / 10000.0
    fig.add_vline(x=obs_um, line_dash="dash", line_width=1)
    fig.add_annotation(
        x=obs_um,
        y=1.0,
        yref="paper",
        text=f"{line_a:.2f} Å",
        showarrow=False,
        textangle=-90,
        yanchor="top",
        font=dict(size=10),
    )

fig.update_layout(
    title=(
        "MoM-z14 / source 10003 — N III] interactive G235H spectrum"
        "<br><sup>Hover over any peak; drag to zoom; double-click to reset</sup>"
    ),
    xaxis_title="Observed wavelength [µm]",
    yaxis_title="Flux [native FITS units]",
    hovermode="closest",
    height=620,
    margin=dict(l=70, r=30, t=100, b=80),
)
fig.update_xaxes(rangeslider_visible=True, showspikes=True, spikemode="across", spikesnap="cursor")
fig.update_yaxes(showspikes=True, spikemode="across", spikesnap="cursor")

print("SOURCE FILE:", path)
print("VALID N III] SAMPLES:", wave_um.size)
print("OBSERVED RANGE [um]:", f"{wave_um.min():.7f} to {wave_um.max():.7f}")
print("ADOPTED REDSHIFT:", Z_ADOPTED)
print("HOVER REDSHIFT REFERENCE: N III] 1749.67 A")
print(f"END {VERSION}")

display(fig)
