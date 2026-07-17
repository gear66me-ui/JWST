from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.cosmology import Planck18
import astropy.units as u

VERSION = "JWST_0120"
OUT_PNG = Path("/content/JWST_OUTPUT/PNG")
OUT_CSV = Path("/content/JWST_OUTPUT/CSV")
OUT_PNG.mkdir(parents=True, exist_ok=True)
OUT_CSV.mkdir(parents=True, exist_ok=True)

# Dense logarithmic sampling shows how comoving distance approaches a finite limit
# as redshift tends toward infinity in a standard FLRW cosmology.
z = np.concatenate(([0.0], np.geomspace(1e-4, 1e7, 2400)))
dc_gly = Planck18.comoving_distance(z).to_value(u.Glyr)
lookback_gyr = Planck18.lookback_time(z).to_value(u.Gyr)
age_myr = Planck18.age(z).to_value(u.Myr)

# Numerical high-z proxy for the particle-horizon/comoving-radius limit.
z_limit = 1e9
horizon_gly = Planck18.comoving_distance(z_limit).to_value(u.Glyr)

# Reference redshifts, including MoM-z14.
ref_z = np.array([1, 3, 6, 10, 14.44, 20, 50, 100, 1000], dtype=float)
ref_dc = Planck18.comoving_distance(ref_z).to_value(u.Glyr)
ref_lb = Planck18.lookback_time(ref_z).to_value(u.Gyr)
ref_age = Planck18.age(ref_z).to_value(u.Myr)

summary = pd.DataFrame({
    "redshift_z": ref_z,
    "comoving_distance_Gly": ref_dc,
    "lookback_time_Gyr": ref_lb,
    "universe_age_Myr": ref_age,
    "fraction_of_horizon": ref_dc / horizon_gly,
})
summary.to_csv(OUT_CSV / f"{VERSION}_REFERENCE_POINTS.csv", index=False)

curve = pd.DataFrame({
    "redshift_z": z,
    "comoving_distance_Gly": dc_gly,
    "lookback_time_Gyr": lookback_gyr,
    "universe_age_Myr": age_myr,
})
curve.to_csv(OUT_CSV / f"{VERSION}_CURVE.csv", index=False)

fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=160)
fig.patch.set_facecolor("#07111f")
ax.set_facecolor("#07111f")

ax.plot(z[1:], dc_gly[1:], color="#63d5ff", lw=2.3,
        label="Planck18 comoving distance")
ax.axhline(horizon_gly, color="#ffbf69", lw=1.2, ls="--",
           label=f"Asymptotic comoving radius ≈ {horizon_gly:.2f} Gly")
ax.axhline(40.0, color="#d9e2ec", lw=0.8, ls=":", alpha=0.75,
           label="40 Gly reference")

for rz, rd in zip(ref_z, ref_dc):
    ax.scatter(rz, rd, s=34, edgecolor="#07111f", linewidth=0.7, zorder=4)

mom_z = 14.44
mom_dc = Planck18.comoving_distance(mom_z).to_value(u.Glyr)
ax.scatter([mom_z], [mom_dc], s=105, marker="*", color="#ff6b6b",
           edgecolor="white", linewidth=0.7, zorder=6)
ax.annotate(
    f"MoM-z14\nz = {mom_z:.2f}\nD₍c₎ = {mom_dc:.2f} Gly",
    xy=(mom_z, mom_dc), xytext=(32, 37.0),
    color="white", fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#ff6b6b", lw=1.0),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#101d2d",
              edgecolor="#ff6b6b", alpha=0.95),
)

ax.text(
    2.1e4, horizon_gly - 2.0,
    "The curve flattens toward the particle-horizon scale.\n"
    "It does not asymptote at 40 Gly; 40 Gly is crossed at finite redshift.",
    color="#f4f7fb", fontsize=10.5, ha="center", va="top",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#101d2d",
              edgecolor="#6f8398", alpha=0.92),
)

ax.set_xscale("log")
ax.set_xlim(1e-3, 1e7)
ax.set_ylim(0, horizon_gly + 2.0)
ax.set_xlabel("Redshift z (log scale)", color="white", fontsize=12)
ax.set_ylabel("Present-day comoving distance (billion light-years)",
              color="white", fontsize=12)
ax.set_title("Redshift–Distance Relation and Its High-z Asymptote",
             color="white", fontsize=17, weight="bold", pad=16)
ax.text(
    0.5, 1.015,
    "Planck18 ΛCDM cosmology • distance is comoving, not light-travel distance",
    transform=ax.transAxes, ha="center", va="bottom",
    color="#aebdcc", fontsize=10.5,
)

ax.grid(True, which="major", alpha=0.22, linewidth=0.7)
ax.grid(True, which="minor", alpha=0.08, linewidth=0.45)
ax.tick_params(colors="white", which="both")
for spine in ax.spines.values():
    spine.set_color("#668099")

legend = ax.legend(loc="lower right", frameon=True, fontsize=9.5)
legend.get_frame().set_facecolor("#101d2d")
legend.get_frame().set_edgecolor("#668099")
for text in legend.get_texts():
    text.set_color("white")

png_path = OUT_PNG / f"{VERSION}_REDSHIFT_COMOVING_DISTANCE_ASYMPTOTE.png"
fig.tight_layout()
fig.savefig(png_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()

z_40 = np.interp(40.0, dc_gly, z)

print(f"CODE OUTPUT: {VERSION}")
print("REDSHIFT–COMOVING DISTANCE ASYMPTOTE")
print("-" * 78)
print(f"Planck18 asymptotic comoving radius  {horizon_gly:10.6f} Gly")
print(f"Redshift where D_c ≈ 40 Gly         {z_40:10.6f}")
print(f"MoM-z14 redshift                    {mom_z:10.6f}")
print(f"MoM-z14 comoving distance           {mom_dc:10.6f} Gly")
print("-" * 78)
print(f"Plot PNG: {png_path}")
print(f"Curve CSV: {OUT_CSV / f'{VERSION}_CURVE.csv'}")
print(f"Reference CSV: {OUT_CSV / f'{VERSION}_REFERENCE_POINTS.csv'}")
print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")
print(f"# {VERSION}")
