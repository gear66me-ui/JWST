# JWST_0149
import os, io, sys, warnings, contextlib, subprocess
warnings.filterwarnings("ignore")

# Quiet dependency install for Colab
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        from astroquery.mast import Observations
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "astroquery", "astropy"], check=True)
        from astroquery.mast import Observations
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.table import Table

OUT_PNG = "/content/JWST_OUTPUT/PNG"
OUT_CSV = "/content/JWST_OUTPUT/CSV"
os.makedirs(OUT_PNG, exist_ok=True)
os.makedirs(OUT_CSV, exist_ok=True)

RA, DEC = 53.1537083, -27.7803694
Z_NIRSPEC = 14.32
Z_ALMA = 14.1793
LYA_REST = 0.121567
CIII_REST = 0.1908

# Public JWST archive query at the published sky position
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    obs = Observations.query_region(f"{RA} {DEC}", radius="1.5 arcsec")
    keep = np.array([str(x).upper() == "JWST" for x in obs["obs_collection"]])
    obs = obs[keep]
    products = Observations.get_product_list(obs)
    names = np.array([str(x).lower() for x in products["productFilename"]])
    mask = np.array([("x1d" in n or "spec1d" in n) and n.endswith(".fits") for n in names])
    cand = products[mask]
    if len(cand) == 0:
        mask = np.array([n.endswith(".fits") and ("s2d" in n or "cal" in n) for n in names])
        cand = products[mask]
    if len(cand) == 0:
        raise RuntimeError("No public JWST spectral FITS product found at JADES-GS-z14-0 position")
    # Prefer program 1287 and prism products
    score = []
    for r in cand:
        n = str(r["productFilename"]).lower()
        s = (10 if "1287" in n else 0) + (6 if "prism" in n else 0) + (4 if "x1d" in n else 0)
        score.append(s)
    row = cand[int(np.argmax(score))]
    manifest = Observations.download_products(Table(rows=[row]), download_dir="/content/jwst_0149_mast", cache=True)
    fits_path = str(manifest["Local Path"][0])

# Extract wavelength, flux and uncertainty from any suitable table extension
def read_spectrum(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            d = getattr(hdu, "data", None)
            if d is None or not hasattr(d, "names") or d.names is None:
                continue
            names = {n.lower(): n for n in d.names}
            wkey = next((names[k] for k in names if k in ("wavelength", "wave", "lambda")), None)
            fkey = next((names[k] for k in names if k in ("flux", "flam", "fnu", "surf_bright")), None)
            ekey = next((names[k] for k in names if k in ("error", "err", "flux_error", "sigma")), None)
            if wkey and fkey:
                w = np.asarray(d[wkey], float).ravel()
                f = np.asarray(d[fkey], float).ravel()
                if ekey:
                    e = np.asarray(d[ekey], float).ravel()
                elif "var_poisson" in names:
                    e = np.sqrt(np.abs(np.asarray(d[names["var_poisson"]], float).ravel()))
                else:
                    e = np.full_like(f, np.nan)
                return w, f, e
    raise RuntimeError("No wavelength/flux table found in downloaded FITS")

w, f, e = read_spectrum(fits_path)
if np.nanmedian(w) > 100:
    w = w / 1e4
m = np.isfinite(w) & np.isfinite(f) & (w > 0.75) & (w < 5.35)
w, f, e = w[m], f[m], e[m]
order = np.argsort(w)
w, f, e = w[order], f[order], e[order]
if not np.any(np.isfinite(e) & (e > 0)):
    med = np.nanmedian(np.abs(f - np.nanmedian(f)))
    e = np.full_like(f, 1.4826 * med)
else:
    good_e = np.isfinite(e) & (e > 0)
    e[~good_e] = np.nanmedian(e[good_e])

# Robust scaling for display
scale = np.nanpercentile(np.abs(f), 90)
if not np.isfinite(scale) or scale == 0:
    scale = 1.0
fn, en = f / scale, e / scale

# Shared-redshift model: continuum, softened Lyman break, and CIII] feature
zgrid = np.linspace(13.75, 14.55, 801)
chi2 = np.full_like(zgrid, np.nan)
models = []

for iz, z in enumerate(zgrid):
    lbreak = LYA_REST * (1 + z)
    ciii = CIII_REST * (1 + z)
    width = 0.018
    transmission = 0.5 * (1 + np.tanh((w - lbreak) / width))
    x = np.clip(w / 2.5, 0.2, 3.0)
    cont0 = transmission
    cont1 = transmission * np.log(x)
    sigma = 0.018 + 0.004 * np.clip(ciii - 2.5, 0, 3)
    line = np.exp(-0.5 * ((w - ciii) / sigma) ** 2)
    A = np.column_stack([cont0, cont1, line])
    fitmask = np.isfinite(fn) & np.isfinite(en) & (en > 0)
    Aw = A[fitmask] / en[fitmask, None]
    yw = fn[fitmask] / en[fitmask]
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    coef[2] = max(0.0, coef[2])
    model = A @ coef
    chi2[iz] = np.nansum(((fn[fitmask] - model[fitmask]) / en[fitmask]) ** 2)
    models.append(model)

imin = int(np.nanargmin(chi2))
zbest = float(zgrid[imin])
best = models[imin]
like = np.exp(-0.5 * (chi2 - np.nanmin(chi2)))
like /= np.trapz(like, zgrid)

# Wavelength relation using the joint best redshift
rest_points = np.array([LYA_REST, CIII_REST])
obs_points = rest_points * (1 + zbest)
xx = np.linspace(0.115, 0.205, 300)

plt.rcParams.update({
    "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
    "text.color": "white", "axes.labelcolor": "white", "xtick.color": "white",
    "ytick.color": "white", "axes.edgecolor": "#8aa0b8", "font.size": 10
})
fig = plt.figure(figsize=(15, 12), constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.8])

ax0 = fig.add_subplot(gs[0])
ax0.plot(zgrid, like, color="#ffd23f", lw=1.4)
ax0.axvline(zbest, color="white", lw=0.8, ls="--")
ax0.axvline(Z_NIRSPEC, color="#ff5a5f", lw=0.9, ls=":", label="Published NIRSpec z = 14.32")
ax0.axvline(Z_ALMA, color="#54d6ff", lw=0.9, ls="-.", label="ALMA [O III] z = 14.1793")
ax0.set_xlim(zgrid.min(), zgrid.max())
ax0.set_ylabel("Normalized joint likelihood")
ax0.set_title("JADES-GS-z14-0 — Lyman-break + C III] shared-redshift likelihood", fontsize=13)
ax0.grid(alpha=0.16)
ax0.legend(frameon=False, ncol=2, loc="upper left")

ax1 = fig.add_subplot(gs[1])
ax1.plot(xx, (1 + zbest) * xx, color="#ffd23f", lw=1.8, label=f"Joint model: z = {zbest:.5f}")
ax1.plot(xx, (1 + Z_NIRSPEC) * xx, color="#ff5a5f", lw=0.9, ls="--", label="Published NIRSpec relation")
ax1.scatter(rest_points, obs_points, s=[58, 42], marker="s", facecolor="#2f8cff", edgecolor="white", linewidth=0.7, zorder=5)
ax1.annotate("Lyman-α break", (rest_points[0], obs_points[0]), xytext=(10, 15), textcoords="offset points", color="#8fc5ff")
ax1.annotate("C III]", (rest_points[1], obs_points[1]), xytext=(-45, 15), textcoords="offset points", color="#8fc5ff")
ax1.set_xlabel("Rest wavelength [μm]")
ax1.set_ylabel("Observed wavelength [μm]")
ax1.set_title("Wavelength mapping: λobs = (1 + zjoint) λrest", fontsize=12)
ax1.grid(alpha=0.16)
ax1.legend(frameon=False, loc="upper left")

ax2 = fig.add_subplot(gs[2])
ax2.fill_between(w, fn - en, fn + en, color="#8c9aa8", alpha=0.23, linewidth=0)
ax2.step(w, fn, where="mid", color="#3f95ff", lw=0.75, label="Public JWST/NIRSpec spectrum")
ax2.plot(w, best, color="#ffd23f", lw=1.35, label="Best joint model")
lb = LYA_REST * (1 + zbest)
ci = CIII_REST * (1 + zbest)
ax2.axvline(lb, color="#ff5a5f", lw=1.0, ls="--")
ax2.axvline(ci, color="#72e0ff", lw=0.85, ls=":")
ax2.text(lb + 0.025, np.nanpercentile(fn, 88), "Lyman-α break", color="#ff8589", rotation=90, va="top")
ax2.text(ci + 0.025, np.nanpercentile(fn, 82), "C III]", color="#72e0ff", rotation=90, va="top")
ax2.axhline(0, color="#8694a6", lw=0.5, alpha=0.7)
ax2.set_xlim(0.8, 5.25)
lo, hi = np.nanpercentile(fn, [2, 98])
pad = 0.18 * (hi - lo if hi > lo else 1)
ax2.set_ylim(lo - pad, hi + pad)
ax2.set_xlabel("Observed wavelength [μm]")
ax2.set_ylabel("Scaled flux density")
ax2.set_title("Native spectral granularity — blue data, gray 1σ, yellow joint model", fontsize=12)
ax2.grid(alpha=0.12)
ax2.legend(frameon=False, loc="upper right")

fig.suptitle("JADES-GS-z14-0 — the next spectroscopically confirmed galaxy beyond MoM-z14", fontsize=15, y=1.01)
fig.savefig(os.path.join(OUT_PNG, "JWST_0149_JADES_GS_Z14_0_LYMAN_BREAK_JOINT_MODEL.png"), dpi=350, bbox_inches="tight")
plt.show()
