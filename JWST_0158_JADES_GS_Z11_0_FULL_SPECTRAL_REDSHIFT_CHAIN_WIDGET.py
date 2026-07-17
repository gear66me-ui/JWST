# JWST_0158
import os, io, sys, warnings, contextlib, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")

for pkg in ["numpy", "pandas", "matplotlib", "astropy", "astroquery", "scipy", "ipywidgets"]:
    try:
        __import__(pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astroquery.mast import Observations
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from IPython.display import display, clear_output, HTML
import ipywidgets as widgets

VERSION = "JWST_0158"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
DATA = ROOT / "DATA" / VERSION
for d in (PNG, CSV, DATA):
    d.mkdir(parents=True, exist_ok=True)

TARGET_ID = "10014220"
PROGRAM_ID = "1210"
Z_PAPER = 11.38
LYA_REST = 0.121567
HEII_REST = 0.1640


def download_exact_jades_spectrum():
    cached = DATA / "JADES_GS_Z11_0_10014220_PRISM_SPEC1D.fits"
    if cached.exists() and cached.stat().st_size > 50000:
        return cached

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        obs = Observations.query_criteria(obs_collection="JWST", proposal_id=PROGRAM_ID)
        products = Observations.get_product_list(obs)

    names = np.array([str(x).lower() for x in products["productFilename"]])
    mask = np.array([
        TARGET_ID in n and n.endswith(".fits") and
        ("prism" in n or "clear" in n) and
        ("x1d" in n or "spec1d" in n)
        for n in names
    ])
    candidates = products[mask]
    if len(candidates) == 0:
        mask = np.array([
            TARGET_ID in n and n.endswith(".fits") and ("prism" in n or "clear" in n)
            for n in names
        ])
        candidates = products[mask]
    if len(candidates) == 0:
        raise RuntimeError("Exact JADES NIRSpec ID 10014220 PRISM product was not found in program 1210")

    score = []
    for row in candidates:
        n = str(row["productFilename"]).lower()
        s = 0
        s += 100 if "spec1d" in n else 0
        s += 80 if "x1d" in n else 0
        s += 40 if "clear-prism" in n else 0
        s += 20 if "prism" in n else 0
        score.append(s)
    selected = candidates[int(np.argmax(score))]

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        manifest = Observations.download_products(Table(rows=[selected]), download_dir=str(DATA), cache=True)
    src = Path(str(manifest["Local Path"][0]))
    if not src.exists():
        raise RuntimeError("MAST download did not produce the selected JADES spectrum")
    cached.write_bytes(src.read_bytes())
    return cached


def read_spectrum(path):
    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is None or not hasattr(data, "names") or data.names is None:
                continue
            names = {n.upper(): n for n in data.names}

            def pick(*keys):
                for key in keys:
                    if key.upper() in names:
                        return np.ravel(np.asarray(data[names[key.upper()]], float))
                return None

            wave = pick("WAVELENGTH", "WAVE", "LAMBDA")
            flux = pick("FLUX", "FLUX_CORR", "FLAM", "FNU")
            err = pick("ERROR", "ERR", "FLUX_ERROR", "FLUX_ERR", "SIGMA")
            if wave is None or flux is None:
                continue
            if err is None:
                ivar = pick("IVAR", "INVERSE_VARIANCE")
                if ivar is not None:
                    err = np.where(ivar > 0, 1 / np.sqrt(ivar), np.nan)
            if err is None:
                err = np.full_like(flux, np.nan)
            unit = ""
            for k in ("FLUX", "FLUX_CORR", "FLAM", "FNU"):
                if k in names:
                    try:
                        unit = str(hdu.columns[names[k]].unit or "").lower()
                    except Exception:
                        pass
                    break
            return wave, flux, err, unit
    raise RuntimeError("No wavelength/flux table found in the selected FITS product")


PATH = download_exact_jades_spectrum()
wave, flux, err, unit = read_spectrum(PATH)
if np.nanmedian(wave) > 100:
    wave = wave / 1e4
elif np.nanmedian(wave) > 10:
    wave = wave / 1e3

if "jy" in unit or np.nanmedian(np.abs(flux[np.isfinite(flux)])) < 1e-10:
    conv = 1e-6 * 1e-23 * 2.99792458e18 / (wave * 1e4) ** 2 / 1e-21
    flux = flux * conv
    err = err * conv
else:
    scale_guess = np.nanmedian(np.abs(flux[(wave > 1.7) & (wave < 2.5)]))
    if np.isfinite(scale_guess) and scale_guess < 1e-8:
        flux = flux / 1e-21
        err = err / 1e-21

mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0.75) & (wave < 5.35)
wave, flux, err = wave[mask], flux[mask], err[mask]
order = np.argsort(wave)
wave, flux, err = wave[order], flux[order], err[order]
good = np.isfinite(err) & (err > 0)
if not np.any(good):
    mad = np.nanmedian(np.abs(flux - np.nanmedian(flux)))
    err = np.full_like(flux, max(1.4826 * mad, 1e-6))
else:
    err[~good] = np.nanmedian(err[good])

fitmask = (wave >= 1.05) & (wave <= 5.20)
lam, y, s = wave[fitmask], flux[fitmask], err[fitmask]


def empirical_sed(lam_um, z, beta, width, amplitude):
    edge = LYA_REST * (1 + z)
    continuum = amplitude * (np.maximum(lam_um, edge) / 1.7) ** beta
    transmission = 0.5 * (1 + np.tanh((lam_um - edge) / width))
    model = continuum * transmission
    dl = np.nanmedian(np.diff(lam_um))
    sigma_lam = max(np.nanmedian(lam_um) / (100.0 * 2.354820045), dl * 0.5)
    return gaussian_filter1d(model, sigma_lam / dl, mode="nearest")


def objective(theta, yy=y):
    z, beta, width = theta
    if not (10.7 <= z <= 11.9 and -4.5 <= beta <= 0.5 and 0.002 <= width <= 0.080):
        return 1e99
    unit_model = empirical_sed(lam, z, beta, width, 1.0)
    wgt = 1.0 / s**2
    denom = np.sum(wgt * unit_model**2)
    if denom <= 0:
        return 1e99
    amp = np.sum(wgt * yy * unit_model) / denom
    model = amp * unit_model
    return np.sum(((yy - model) / s) ** 2)


result = minimize(objective, x0=[Z_PAPER, -1.8, 0.018], method="Nelder-Mead",
                  options={"maxiter": 2200, "xatol": 1e-7, "fatol": 1e-5})
z_best, beta_best, width_best = result.x
unit_best = empirical_sed(lam, z_best, beta_best, width_best, 1.0)
wgt = 1.0 / s**2
amp_best = np.sum(wgt * y * unit_best) / np.sum(wgt * unit_best**2)
best = amp_best * unit_best

# Redshift likelihood profile: optimize beta and width at each z, then convert Δχ² to posterior.
z_grid = np.linspace(10.95, 11.75, 801)
chi2 = np.empty_like(z_grid)
for i, z in enumerate(z_grid):
    r = minimize(lambda q: objective([z, q[0], q[1]]), x0=[beta_best, width_best],
                 method="Nelder-Mead", options={"maxiter": 350, "xatol": 2e-5, "fatol": 1e-3})
    chi2[i] = r.fun
posterior = np.exp(np.clip(-0.5 * (chi2 - np.nanmin(chi2)), -700, 0))
posterior /= np.trapz(posterior, z_grid)

cdf = np.cumsum(posterior)
cdf /= cdf[-1]
z16 = np.interp(0.16, cdf, z_grid)
z50 = np.interp(0.50, cdf, z_grid)
z84 = np.interp(0.84, cdf, z_grid)

lya_obs = LYA_REST * (1 + z_best)
heii_obs = HEII_REST * (1 + z_best)
residual = (y - best) / s

plt.rcParams.update({
    "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
    "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
    "ytick.color": "white", "axes.edgecolor": "#b8c4d0", "font.size": 10
})

display(HTML("""<style>.jwst{background:#000;padding:10px;border:1px solid #555;border-radius:8px}.jwst label,.jwst .widget-label{color:#fff!important;font-weight:600}.jwst button{background:#202020!important;color:#fff!important;border:1px solid #888!important}</style>"""))
run = widgets.Button(description="Re-fit full spectrum", button_style="success", layout=widgets.Layout(width="220px"))
out = widgets.Output(layout=widgets.Layout(border="1px solid #333"))


def draw(_=None):
    with out:
        clear_output(wait=True)
        fig = plt.figure(figsize=(16, 15), facecolor="black")
        gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 2.5, 0.9], hspace=0.12)
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharex=ax1)
        for ax in (ax0, ax1, ax2):
            ax.grid(color="#303944", lw=0.6, alpha=0.72)
            for sp in ax.spines.values():
                sp.set_color("#b8c4d0")

        ax0.plot(z_grid, posterior, color="#62dfd1", lw=2.0,
                 label=f"Full-spectrum posterior: z={z50:.4f} (+{z84-z50:.4f}/-{z50-z16:.4f})")
        ax0.axvline(z_best, color="#ffd400", lw=1.5, label=f"Best fit z={z_best:.4f}")
        ax0.axvline(Z_PAPER, color="#ff6b6b", lw=1.0, ls="--", label="Paper value z=11.38")
        ax0.axvspan(z16, z84, color="#62dfd1", alpha=0.12, label="68% interval")
        ax0.set_xlim(z_grid.min(), z_grid.max())
        ax0.set_xlabel("Redshift, z")
        ax0.set_ylabel("Normalized posterior density")
        ax0.set_title("Scientific chain, step 1 — shift the complete spectral model until the Lyman break and continuum align")
        lg = ax0.legend(facecolor="black", edgecolor="#666", ncol=2, fontsize=9)
        [t.set_color("white") for t in lg.get_texts()]

        ax1.fill_between(lam, y-s, y+s, step="mid", color="#8f8f8f", alpha=0.30, label="1σ uncertainty")
        ax1.step(lam, y, where="mid", color="#4f79b9", lw=1.05, label="JADES-GS-z11-0 NIRSpec/PRISM data")
        ax1.plot(lam, best, color="#e68645", lw=2.5,
                 label=f"Best empirical full-spectrum fit: z={z_best:.4f}, β={beta_best:.2f}")
        ax1.axvline(lya_obs, color="#ffd400", lw=1.35, ls="--",
                    label=f"Redshift anchor: Lyα rest 0.121567 μm → observed {lya_obs:.4f} μm")
        ax1.axvline(heii_obs, color="#ff4d4d", lw=1.15, ls="--",
                    label=f"He II 1640 reference after fitting z → {heii_obs:.4f} μm")
        ax1.axhline(0, color="#9aa5b1", lw=0.55, ls=":")
        ax1.set_xlim(1.0, 5.25)
        qlo, qhi = np.nanpercentile(y, [1, 99])
        pad = 0.18 * (qhi - qlo)
        ax1.set_ylim(qlo-pad, qhi+pad)
        ax1.set_ylabel(r"$F_\lambda$ [$10^{-21}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]")
        ax1.set_title("Scientific chain, step 2 — complete observed spectrum, 1σ band, best fit, Lyα redshift anchor, and He II reference")
        lg = ax1.legend(facecolor="black", edgecolor="#666", fontsize=9, ncol=2, loc="upper right")
        [t.set_color("white") for t in lg.get_texts()]

        ax2.axhspan(-1, 1, color="#9a9a9a", alpha=0.20, label="±1σ residual band")
        ax2.axhline(0, color="white", lw=0.7)
        ax2.plot(lam, residual, color="#e7e7e7", lw=0.9)
        ax2.axhline(1, color="#ff5a5a", lw=0.85)
        ax2.axhline(-1, color="#ff5a5a", lw=0.85)
        ax2.axvline(lya_obs, color="#ffd400", lw=1.0, ls="--")
        ax2.axvline(heii_obs, color="#ff4d4d", lw=1.0, ls="--")
        ax2.set_ylim(-3.2, 3.2)
        ax2.set_xlabel("Observed wavelength [μm]")
        ax2.set_ylabel("Normalized residual")
        ax2.set_title("Scientific chain, step 3 — residual audit")

        fig.suptitle("JADES-GS-z11-0 — full-spectrum redshift-chain reproduction of the paper-style analysis", fontsize=16, y=0.995)
        fig.savefig(PNG / f"{VERSION}_JADES_GS_Z11_0_FULL_SPECTRAL_REDSHIFT_CHAIN.png",
                    dpi=430, bbox_inches="tight", facecolor="black")

        pd.DataFrame({
            "quantity": ["best_fit_z", "posterior_z16", "posterior_z50", "posterior_z84",
                         "beta", "break_width_um", "Lyalpha_rest_um", "Lyalpha_observed_um",
                         "HeII_rest_um", "HeII_expected_observed_um"],
            "value": [z_best, z16, z50, z84, beta_best, width_best,
                      LYA_REST, lya_obs, HEII_REST, heii_obs],
            "status": ["derived full-spectrum fit", "derived posterior", "derived posterior",
                       "derived posterior", "fitted continuum", "fitted break smoothing",
                       "laboratory reference", "redshift anchor", "laboratory reference",
                       "reference after redshift fit"]
        }).to_csv(CSV / f"{VERSION}_JADES_GS_Z11_0_REDSHIFT_CHAIN.csv", index=False)

        pd.DataFrame({
            "wavelength_um": lam,
            "flux_1e-21": y,
            "sigma_1e-21": s,
            "best_fit_1e-21": best,
            "normalized_residual": residual
        }).to_csv(CSV / f"{VERSION}_JADES_GS_Z11_0_SPECTRUM_AND_FIT.csv", index=False)
        plt.show()


run.on_click(draw)
display(run)
display(out)
draw()
