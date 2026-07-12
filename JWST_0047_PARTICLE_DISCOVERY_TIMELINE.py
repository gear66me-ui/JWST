from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0047"
OUT = Path("/content/JWST_OUTPUT")
PNG = OUT / "PNG"
CSV = OUT / "CSV"
PNG.mkdir(parents=True, exist_ok=True)
CSV.mkdir(parents=True, exist_ok=True)

# Broad, inclusive historical reconstruction.
# Counts intentionally include named charge states, antiparticles,
# composite hadrons, resonances and exotic hadrons when treated as
# distinct experimentally reported species/states.
# Because particle-counting conventions differ, these are rounded
# lower-bound estimates rather than a literal PDG census.
rows = [
    ("1900s",   1, "photon"),
    ("1910s",   1, "proton"),
    ("1920s",   0, "no major new established species"),
    ("1930s",   3, "neutron, positron, muon"),
    ("1940s",  10, "charged/neutral pions, kaons, Lambda and partners"),
    ("1950s",  90, "neutrino, antiproton, antineutron, strange-particle zoo"),
    ("1960s", 220, "many mesons, baryons, resonances, quark classification"),
    ("1970s",  40, "charm, tau, bottom, gluon, charmed/bottom hadrons"),
    ("1980s",  15, "W+, W-, Z0 and additional heavy-hadron states"),
    ("1990s",  15, "top quark, antihydrogen, heavy baryons and mesons"),
    ("2000s",  25, "tau neutrino, X/Y states, bottom-baryon families"),
    ("2010s",  60, "Higgs boson plus first large wave of LHC hadrons"),
    ("2020s",  26, "additional tetraquarks, pentaquarks and heavy hadrons"),
]

df = pd.DataFrame(rows, columns=["decade", "new_species_est", "examples"])
df["cumulative_est"] = df["new_species_est"].cumsum()

csv_path = CSV / f"{VERSION}_PARTICLE_DISCOVERY_TIMELINE.csv"
png_path = PNG / f"{VERSION}_PARTICLE_DISCOVERY_TIMELINE.png"
df.to_csv(csv_path, index=False)

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor("#09111d")
ax.set_facecolor("#09111d")

x = range(len(df))
ax.bar(x, df["new_species_est"], alpha=0.75, label="Estimated new species/states")
ax.plot(x, df["cumulative_est"], marker="o", linewidth=2.4,
        label="Estimated cumulative count")

for i, value in enumerate(df["cumulative_est"]):
    ax.annotate(f"{int(value)}", (i, value), xytext=(0, 8),
                textcoords="offset points", ha="center", fontsize=9)

ax.set_xticks(list(x))
ax.set_xticklabels(df["decade"], rotation=35, ha="right")
ax.set_ylabel("Count of named particle species / states")
ax.set_xlabel("Discovery decade")
ax.set_title("Growth of the Particle Zoo, 1900s–2020s",
             fontsize=18, pad=18, weight="bold")
ax.text(0.01, 1.01,
        "Inclusive lower-bound reconstruction: antiparticles, charge states, resonances and exotic hadrons counted separately",
        transform=ax.transAxes, fontsize=10, alpha=0.85)
ax.grid(True, axis="y", alpha=0.18)
ax.legend(loc="upper left")

note = (
    "Counting rule is intentionally broad. Exact totals depend on whether one counts "
    "charge partners, antiparticles, resonances, nuclei and tentative states separately."
)
fig.text(0.01, 0.01, note, fontsize=9, alpha=0.75)
fig.tight_layout(rect=[0, 0.045, 1, 1])
fig.savefig(png_path, dpi=220, bbox_inches="tight")
plt.show()

print(f"CODE OUTPUT: {VERSION}")
print(df[["decade", "new_species_est", "cumulative_est"]].to_string(index=False))
print(f"PNG  {png_path}")
print(f"CSV  {csv_path}")
print(datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"))
print(f"# {VERSION}")
