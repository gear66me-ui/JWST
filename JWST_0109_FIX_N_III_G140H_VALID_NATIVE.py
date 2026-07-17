# JWST_0109
# Audit: reject zero-padded N III spectra and select valid G140H coverage

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

VERSION = "JWST_0109"
Z = 9.31102
REST_MIN_A = 1700.0
REST_MAX_A = 1800.0
ROOT = Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS")
OUTPNG = Path("/content/JWST_OUTPUT/PNG/JWST_0109_N_III_G140H_VALID_NATIVE.png")

print(f"CODE OUTPUT: {VERSION}")


def read_1d_spectra(path):
    spectra = []
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu_index, hdu in enumerate(hdul):
                data = hdu.data
                if data is None or not getattr(data, "names", None):
                    continue

                names = {name.lower(): name for name in data.names}
                wave_name = next((names[x] for x in ["wavelength", "wave", "lambda", "lam"] if x in names), None)
                flux_name = next((names[x] for x in ["flux", "flam", "fnu", "flux_cal", "sci"] if x in names), None)
                err_name = next((names[x] for x in ["error", "err", "flux_error", "flux_err", "sigma"] if x in names), None)

                if wave_name is None or flux_name is None:
                    continue

                wave = np.asarray(data[wave_name], dtype=float).ravel()
                flux = np.asarray(data[flux_name], dtype=float).ravel()
                err = np.asarray(data[err_name], dtype=float).ravel() if err_name else np.full(flux.shape, np.nan)

                if wave.size != flux.size:
                    continue

                finite_wave = wave[np.isfinite(wave)]
                if finite_wave.size == 0:
                    continue

                median_wave = np.nanmedian(finite_wave)
                if median_wave > 1000:
                    wave_um = wave / 10000.0
                elif median_wave > 100:
                    wave_um = wave / 1000.0
                else:
                    wave_um = wave

                spectra.append({
                    "hdu": hdu_index,
                    "wave_um": wave_um,
                    "flux": flux,
                    "err": err,
                })
    except Exception:
        return []

    return spectra


candidates = []

for path in ROOT.rglob("*.fits"):
    text = str(path).upper()
    if "G140H" not in text:
        continue

    for spectrum in read_1d_spectra(path):
        wave_um = spectrum["wave_um"]
        flux = spectrum["flux"]
        err = spectrum["err"]
        rest_a = wave_um * 10000.0 / (1.0 + Z)

        region = (
            np.isfinite(rest_a)
            & np.isfinite(flux)
            & (rest_a >= REST_MIN_A)
            & (rest_a <= REST_MAX_A)
        )

        valid = region & (flux != 0.0)
        if np.any(np.isfinite(err)):
            valid &= np.isfinite(err) & (err > 0.0)

        n_region = int(np.count_nonzero(region))
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            continue

        region_flux = flux[valid]
        scatter = float(np.nanpercentile(region_flux, 84) - np.nanpercentile(region_flux, 16))

        candidates.append({
            "path": path,
            "hdu": spectrum["hdu"],
            "rest_a": rest_a,
            "flux": flux,
            "err": err,
            "valid": valid,
            "n_region": n_region,
            "n_valid": n_valid,
            "scatter": scatter,
        })

candidates.sort(key=lambda x: (x["n_valid"], x["scatter"]), reverse=True)

print(f"VALID N III] CANDIDATES: {len(candidates)}")
for rank, candidate in enumerate(candidates[:20], start=1):
    print(
        f"{rank:2d}  valid={candidate['n_valid']:5d}  "
        f"region={candidate['n_region']:5d}  HDU={candidate['hdu']:2d}  "
        f"{candidate['path'].name}"
    )

if not candidates:
    raise RuntimeError("No downloaded G140H FITS file has valid nonzero N III] coverage.")

best = candidates[0]
x = best["rest_a"][best["valid"]]
y = best["flux"][best["valid"]]
e = best["err"][best["valid"]]
order = np.argsort(x)
x, y, e = x[order], y[order], e[order]

print("\nSELECTED FILE")
print(best["path"])
print("HDU:", best["hdu"])
print("VALID SAMPLES:", best["n_valid"])
print("REST RANGE:", f"{x.min():.6f} to {x.max():.6f} A")

plt.figure(figsize=(13, 6))
plt.plot(x, y, linewidth=0.7, label="Native flux")
plt.scatter(x, y, s=9, label="Valid native samples")

good_error = np.isfinite(e) & (e > 0)
if np.any(good_error):
    plt.fill_between(
        x[good_error],
        y[good_error] - e[good_error],
        y[good_error] + e[good_error],
        alpha=0.18,
        label="+/-1 sigma",
    )

for line_a in [1746.82, 1748.65, 1749.67, 1752.16, 1753.99]:
    plt.axvline(line_a, linestyle="--", linewidth=0.7, alpha=0.7)

plt.xlim(REST_MIN_A, REST_MAX_A)
plt.xlabel("Rest-frame wavelength [A]")
plt.ylabel("Flux [native FITS units]")
plt.title(
    f"N III] | G140H | valid native samples n={best['n_valid']}\n"
    f"{best['path'].name}"
)
plt.grid(alpha=0.22)
plt.legend()
plt.tight_layout()
OUTPNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPNG, dpi=300, bbox_inches="tight")
plt.show()

print("\nOUTPUT PNG:", OUTPNG)
print(f"End {VERSION}")
