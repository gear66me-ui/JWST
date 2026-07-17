# JWST_0110
# Audit fix: search both G140H/F100LP and G235H/F170LP for valid N III] coverage.

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

VERSION = "JWST_0110"
print(f"CODE OUTPUT: {VERSION}")

Z = 9.31102
REST_MIN_A = 1700.0
REST_MAX_A = 1800.0
ROOT = Path("/content/JWST_OUTPUT/DATA/JWST_0108/FITS")
OUTPNG = Path("/content/JWST_OUTPUT/PNG/JWST_0110_N_III_VALID_NATIVE.png")


def wavelength_to_um(wave):
    finite = wave[np.isfinite(wave)]
    if finite.size == 0:
        return wave
    median = float(np.nanmedian(finite))
    if median > 1000.0:
        return wave / 10000.0
    if median > 100.0:
        return wave / 1000.0
    return wave


def read_table_spectra(path):
    output = []
    try:
        with fits.open(path, memmap=False) as hdul:
            for hdu_index, hdu in enumerate(hdul):
                data = hdu.data
                names_raw = getattr(data, "names", None)
                if data is None or not names_raw:
                    continue

                names = {str(name).lower(): name for name in names_raw}
                wave_name = next((names[k] for k in ["wavelength", "wave", "lambda", "lam"] if k in names), None)
                flux_name = next((names[k] for k in ["flux", "flam", "fnu", "flux_cal", "sci"] if k in names), None)
                err_name = next((names[k] for k in ["error", "err", "flux_error", "flux_err", "sigma"] if k in names), None)
                dq_name = next((names[k] for k in ["dq", "data_quality", "quality"] if k in names), None)

                if wave_name is None or flux_name is None:
                    continue

                wave = np.asarray(data[wave_name], dtype=float).ravel()
                flux = np.asarray(data[flux_name], dtype=float).ravel()
                if wave.size != flux.size or wave.size == 0:
                    continue

                err = np.asarray(data[err_name], dtype=float).ravel() if err_name is not None else np.full(flux.shape, np.nan)
                dq = np.asarray(data[dq_name]).ravel() if dq_name is not None else np.zeros(flux.shape, dtype=int)

                if err.size != flux.size:
                    err = np.full(flux.shape, np.nan)
                if dq.size != flux.size:
                    dq = np.zeros(flux.shape, dtype=int)

                output.append({
                    "hdu": hdu_index,
                    "wave_um": wavelength_to_um(wave),
                    "flux": flux,
                    "err": err,
                    "dq": dq,
                })
    except Exception:
        return []
    return output


candidates = []
files_checked = 0
spectra_checked = 0

for path in ROOT.rglob("*.fits"):
    upper = str(path).upper()
    if "G140H" not in upper and "G235H" not in upper:
        continue

    grating = "G140H" if "G140H" in upper else "G235H"
    files_checked += 1

    for spec in read_table_spectra(path):
        spectra_checked += 1
        rest_a = spec["wave_um"] * 10000.0 / (1.0 + Z)
        flux = spec["flux"]
        err = spec["err"]
        dq = spec["dq"]

        region = (
            np.isfinite(rest_a)
            & np.isfinite(flux)
            & (rest_a >= REST_MIN_A)
            & (rest_a <= REST_MAX_A)
        )

        valid = region & (flux != 0.0)
        if np.any(np.isfinite(err)):
            valid &= np.isfinite(err) & (err > 0.0)
        if dq.size == flux.size:
            valid &= (dq == 0)

        n_region = int(np.count_nonzero(region))
        n_valid = int(np.count_nonzero(valid))
        if n_valid == 0:
            continue

        x = rest_a[valid]
        y = flux[valid]
        e = err[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        e = e[order]

        finite_y = y[np.isfinite(y)]
        scatter = float(np.nanpercentile(finite_y, 84) - np.nanpercentile(finite_y, 16)) if finite_y.size else 0.0
        coverage = float(x.max() - x.min()) if x.size else 0.0

        candidates.append({
            "path": path,
            "grating": grating,
            "hdu": spec["hdu"],
            "x": x,
            "y": y,
            "e": e,
            "n_region": n_region,
            "n_valid": n_valid,
            "coverage": coverage,
            "scatter": scatter,
        })

candidates.sort(key=lambda c: (c["n_valid"], c["coverage"], c["scatter"]), reverse=True)

print(f"FILES CHECKED          {files_checked}")
print(f"TABLE SPECTRA CHECKED  {spectra_checked}")
print(f"VALID N III] CANDIDATES {len(candidates)}")

for rank, c in enumerate(candidates[:20], start=1):
    print(
        f"{rank:2d}  {c['grating']:5s}  valid={c['n_valid']:5d}  "
        f"region={c['n_region']:5d}  coverage={c['coverage']:8.3f} A  "
        f"HDU={c['hdu']:2d}  {c['path'].name}"
    )

if not candidates:
    print("STATUS                 NO VALID DOWNLOADED N III] COVERAGE")
    print("NEXT ACTION            Download additional G235H/F170LP target products or verify target aperture identity.")
else:
    best = candidates[0]
    x = best["x"]
    y = best["y"]
    e = best["e"]

    print()
    print(f"SELECTED GRATING       {best['grating']}")
    print(f"SELECTED FILE          {best['path']}")
    print(f"SELECTED HDU           {best['hdu']}")
    print(f"VALID SAMPLES          {best['n_valid']}")
    print(f"REST RANGE             {x.min():.6f} to {x.max():.6f} A")

    plt.figure(figsize=(13, 6))
    plt.plot(x, y, linewidth=0.7, label="Native flux")
    plt.scatter(x, y, s=8, label="Valid native samples")

    good_e = np.isfinite(e) & (e > 0.0)
    if np.any(good_e):
        plt.fill_between(x[good_e], y[good_e] - e[good_e], y[good_e] + e[good_e], alpha=0.18, label="+/- 1 sigma")

    for line_a in [1746.82, 1748.65, 1749.67, 1752.16, 1753.99]:
        plt.axvline(line_a, linestyle="--", linewidth=0.7, alpha=0.7)

    plt.xlim(REST_MIN_A, REST_MAX_A)
    plt.xlabel("Rest-frame wavelength [A]")
    plt.ylabel("Flux [native FITS units]")
    plt.title(f"N III] | {best['grating']} | valid native samples n={best['n_valid']}\n{best['path'].name}")
    plt.grid(alpha=0.22)
    plt.legend()
    plt.tight_layout()

    OUTPNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPNG, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"OUTPUT PNG             {OUTPNG}")

print(f"END {VERSION}")
