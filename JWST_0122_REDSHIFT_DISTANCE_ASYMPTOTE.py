# JWST_0122
# Plot Planck18 comoving distance versus redshift with the high-z asymptote clearly visible.

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import Planck18

VERSION = "JWST_0122"
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
CSV_DIR = ROOT / "CSV"
PNG_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

print(f"CODE OUTPUT: {VERSION}")

# Dense logarithmic sampling so the flattening is visible all the way to extreme z.
z = np.logspace(-3, 9, 2400)
dc_gly = Planck18.comoving_distance(z).to_value(u.Glyr)

# Numerical particle-horizon proxy from an extreme redshift.
z_horizon = 1.0e12
horizon_gly = Planck18.comoving_distance(z_horizon).to_value(u.Glyr)
remaining_gly = horizon_gly - dc_gly
fraction = dc_gly / horizon_gly

# Useful high-z milestones.
milestone_z = np.array([1, 3, 6, 10, 14.44, 20, 50, 100, 1e3, 1e5, 1e7, 1e9])
milestone_d = Planck18.comoving_distance(milestone_z).to_value(u.Glyr)
milestone_pct = 100.0 * milestone_d / horizon_gly

# Locate where the curve reaches selected fractions of the asymptote.
levels = [0.90, 0.95, 0.99, 0.999]
level_rows = []
for level in levels:
    idx = int(np.argmax(fraction >= level))
    level_rows.append({
        "fraction_of_asymptote": level,
        "percent": 100.0 * level,
        "redshift_z": z[idx],
        "comoving_distance_Gly": dc_gly[idx],
        "remaining_to_horizon_Gly": remaining_gly[idx],
    })

curve_csv = CSV_DIR / f"{VERSION}_REDSHIFT_DISTANCE_CURVE.csv"
pd.DataFrame({
    "redshift_z": z,
    "comoving_distance_Gly": dc_gly,
    "fraction_of_asymptote": fraction,
    "remaining_to_horizon_Gly": remaining_gly,
}).to_csv(curve_csv, index=False)

milestone_csv = CSV_DIR / f"{VERSION}_MILESTONES.csv"
pd.DataFrame({
    "redshift_z": milestone_z,
    "comoving_distance_Gly": milestone_d,
    "percent_of_asymptote": milestone_pct,
}).to_csv(milestone_csv, index=False)

levels_csv = CSV_DIR / f"{VERSION}_ASYMPTOTE_LEVELS.csv"
pd.DataFrame(level_rows).to_csv(levels_csv, index=False)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(15, 8.5))

ax.plot(z, dc_gly, lw=2.1, label="Planck18 comoving distance")
ax.axhline(horizon_gly, lw=1.2, ls="--", alpha=0.9,
           label=f"High-z asymptote ≈ {horizon_gly:.3f} Gly")

for level in levels:
    y = level * horizon_gly
    ax.axhline(y, lw=0.45, ls=":", alpha=0.35)

for zz, dd, pp in zip(milestone_z, milestone_d, milestone_pct):
    if zz in (1, 10, 14.44, 100, 1e3, 1e5, 1e7):
        ax.scatter([zz], [dd], s=22, zorder=4)
        label = f"z={zz:g}\n{dd:.2f} Gly\n{pp:.2f}%"
        ax.annotate(label, xy=(zz, dd), xytext=(8, 10),
                    textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.55))

ax.set_xscale("log")
ax.set_xlim(z.min(), z.max())
ax.set_ylim(0, horizon_gly * 1.035)
ax.set_xlabel("Redshift z (log scale)", fontsize=13)
ax.set_ylabel("Present-day comoving distance [Gly]", fontsize=13)
ax.set_title("Redshift–Distance Relation and the High-z Asymptote", fontsize=18, weight="bold", pad=16)
ax.text(0.5, 1.005,
        "Planck18 ΛCDM • the curve approaches the finite particle-horizon scale",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=11, alpha=0.75)
ax.grid(alpha=0.18)
ax.legend(loc="lower right", framealpha=0.9)

inset = ax.inset_axes([0.57, 0.13, 0.39, 0.33])
zin = z[z >= 10]
din = dc_gly[z >= 10]
inset.plot(zin, horizon_gly - din, lw=1.7)
inset.set_xscale("log")
inset.set_yscale("log")
inset.set_xlabel("z", fontsize=8)
inset.set_ylabel("Horizon − distance [Gly]", fontsize=8)
inset.set_title("Residual distance to asymptote", fontsize=9)
inset.grid(alpha=0.16)
inset.tick_params(labelsize=8)

fig.tight_layout()
png_path = PNG_DIR / f"{VERSION}_REDSHIFT_DISTANCE_ASYMPTOTE.png"
fig.savefig(png_path, dpi=220, bbox_inches="tight")
plt.show()

print(f"ASYMPTOTIC COMOVING RADIUS: {horizon_gly:.6f} Gly")
print("\nASYMPTOTE LEVELS")
print(pd.DataFrame(level_rows).to_string(index=False, formatters={
    "fraction_of_asymptote": "{:.4f}".format,
    "percent": "{:.3f}".format,
    "redshift_z": "{:.6g}".format,
    "comoving_distance_Gly": "{:.6f}".format,
    "remaining_to_horizon_Gly": "{:.6f}".format,
}))
print("\nOUTPUT SUMMARY")
print(f"Plot PNG: {png_path}")
print(f"Curve CSV: {curve_csv}")
print(f"Milestones CSV: {milestone_csv}")
print(f"Asymptote levels CSV: {levels_csv}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
