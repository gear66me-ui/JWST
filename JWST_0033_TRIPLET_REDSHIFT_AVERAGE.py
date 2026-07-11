# JWST_0033
# Audit: Redshift derived from three observed JWST/NIRSpec H-alpha + [N II] peaks. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import getpass
import subprocess
import importlib
import warnings

VERSION = "JWST_0033"
PROJECT = "JWST THREE-LINE REDSHIFT AVERAGE"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0033_TRIPLET_REDSHIFT_AVERAGE"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_DIRECT_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen", 0.45),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha", 1.00),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen", 0.70),
]

SCAN_LINES = [
    ("H-beta", 0.486133, 0.45),
    ("[O III] 4959", 0.495891, 0.50),
    ("[O III] 5007", 0.500684, 1.00),
    ("[N II] 6548", 0.654805, 0.45),
    ("H-alpha", 0.656281, 1.00),
    ("[N II] 6583", 0.658345, 0.70),
    ("[S II] 6716", 0.671647, 0.35),
    ("[S II] 6731", 0.673085, 0.35),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "scipy", "requests"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)


def get_token(optional=True):
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        if token:
            return token.strip()
    except Exception:
        pass
    if optional:
        return ""
    return getpass.getpass("GitHub token, hidden input: ").strip()


def gh_headers(token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": f"{VERSION}-upload"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def github_get_sha(token, path):
    import requests
    if not token:
        return None
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    r = requests.get(url, headers=gh_headers(token), params={"ref": BRANCH}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def github_put_file(token, local_path, repo_path):
    import requests
    if not token:
        return "", "skipped-no-token"
    sha = github_get_sha(token, repo_path)
    payload = {
        "message": f"{VERSION}: upload {local_path.name}",
        "branch": BRANCH,
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    url = f"https://api.github.com/repos/{REPO}/contents/{repo_path}"
    r = requests.put(url, headers=gh_headers(token), json=payload, timeout=90)
    r.raise_for_status()
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{repo_path}", "updated" if sha else "created"


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def download_spectrum():
    import requests
    local = DATA / PRODUCT
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    try:
        need("astroquery")
        from astroquery.mast import Observations
        Observations.download_file(MAST_URI, local_path=str(local))
        if local.exists() and local.stat().st_size > 100000:
            return local, "astroquery_mast"
    except Exception as exc:
        print(f"astroquery MAST download failed; trying direct MAST URL: {exc}")
    r = requests.get(MAST_DIRECT_URL, timeout=120)
    r.raise_for_status()
    local.write_bytes(r.content)
    if local.stat().st_size < 100000:
        raise RuntimeError(f"Downloaded file is unexpectedly small: {local.stat().st_size} bytes")
    return local, "direct_mast_url"


def read_spectrum(path):
    import numpy as np
    from astropy.io import fits
    with fits.open(path) as hdul:
        ext = None
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                ext = hdu
                break
        if ext is None:
            raise RuntimeError("No WAVELENGTH/FLUX FITS table found.")
        wave = np.asarray(ext.data["WAVELENGTH"], dtype=float)
        flux = np.asarray(ext.data["FLUX"], dtype=float)
        unit = str(ext.header.get("TUNIT2", "unknown"))
    mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[mask]
    flux = flux[mask]
    if unit.lower() == "jy":
        flux = flux * 1e6
        unit = "microJy"
    order = np.argsort(wave)
    return wave[order], flux[order], unit


def continuum_residual(wave, flux):
    import numpy as np
    from scipy.ndimage import median_filter
    from scipy.signal import savgol_filter
    n = len(flux)
    k = max(101, int(n * 0.055) // 2 * 2 + 1)
    k = min(k, n - 1 if (n - 1) % 2 else n - 2)
    cont = median_filter(flux, size=k, mode="nearest")
    w = max(51, int(n * 0.035) // 2 * 2 + 1)
    w = min(w, n - 1 if (n - 1) % 2 else n - 2)
    if w >= 7:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cont = savgol_filter(cont, w, 2, mode="interp")
    residual = flux - cont
    sw = max(7, int(n * 0.0055) // 2 * 2 + 1)
    sw = min(sw, n - 1 if (n - 1) % 2 else n - 2)
    smooth = savgol_filter(residual, sw, 2, mode="interp") if sw >= 7 else residual
    return cont, residual, smooth


def robust_norm(y):
    import numpy as np
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(y)
    return (y - med) / (scale if scale and scale == scale else 1.0)


def scan_z(wave, smooth):
    import numpy as np
    y = robust_norm(smooth)
    z_grid = np.linspace(2.20, 2.75, 2201)
    scores = []
    for z in z_grid:
        s = 0.0
        wsum = 0.0
        for _, rest, weight in SCAN_LINES:
            x = rest * (1 + z)
            if wave.min() <= x <= wave.max():
                s += weight * np.interp(x, wave, y)
                wsum += weight
        scores.append(s / wsum if wsum else float("nan"))
    scores = np.asarray(scores)
    i = int(np.nanargmax(scores))
    z = float(z_grid[i])
    if 2 <= i < len(z_grid) - 2:
        xs = z_grid[i-2:i+3]
        ys = scores[i-2:i+3]
        c = np.polyfit(xs, ys, 2)
        if c[0] < 0:
            zv = -c[1] / (2 * c[0])
            if xs.min() <= zv <= xs.max():
                z = float(zv)
    return z_grid, scores, z


def local_peak(wave, smooth, center, half_width=0.018):
    import numpy as np
    m = (wave >= center - half_width) & (wave <= center + half_width)
    if m.sum() < 5:
        return float("nan"), float("nan")
    x = wave[m]
    y = smooth[m]
    i = int(np.nanargmax(y))
    px = float(x[i])
    py = float(y[i])
    j0 = max(0, i - 4)
    j1 = min(len(x), i + 5)
    if j1 - j0 >= 5:
        try:
            coeff = np.polyfit(x[j0:j1], y[j0:j1], 2)
            if coeff[0] < 0:
                xv = -coeff[1] / (2 * coeff[0])
                if x[j0] <= xv <= x[j1 - 1]:
                    px = float(xv)
                    py = float(np.polyval(coeff, xv))
        except Exception:
            pass
    return px, py


def direct_triplet_z(wave, smooth, z_seed):
    import numpy as np
    import pandas as pd
    rows = []
    for n, label, rest, species, weight in TRIPLET:
        predicted = rest * (1 + z_seed)
        peak, amp = local_peak(wave, smooth, predicted)
        z_line = peak / rest - 1 if peak == peak else float("nan")
        rows.append({
            "n": n,
            "line": label,
            "species": species,
            "rest_um": rest,
            "predicted_observed_um": predicted,
            "measured_peak_um": peak,
            "z_line": z_line,
            "offset_nm_vs_scan": (peak - predicted) * 1000 if peak == peak else float("nan"),
            "relative_line_weight": weight,
            "measured_peak_residual_flux": amp,
        })
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["z_line"])
    z_mean = float(valid["z_line"].mean())
    z_std = float(valid["z_line"].std(ddof=1)) if len(valid) > 1 else float("nan")
    amp = valid["measured_peak_residual_flux"].clip(lower=0)
    if float(amp.sum()) > 0:
        z_amp_weighted = float((valid["z_line"] * amp).sum() / amp.sum())
    else:
        z_amp_weighted = z_mean
    catalog_weight = valid["relative_line_weight"]
    z_catalog_weighted = float((valid["z_line"] * catalog_weight).sum() / catalog_weight.sum())
    return df, z_mean, z_std, z_amp_weighted, z_catalog_weighted


def shared_gaussian_fit(wave, residual, z_seed):
    import numpy as np
    from scipy.optimize import curve_fit
    center = 0.656281 * (1 + z_seed)
    m = (wave >= center - 0.085) & (wave <= center + 0.085)
    x = wave[m]
    y = residual[m]
    if len(x) < 20:
        return {}
    base = float(np.nanmedian(y))
    amp = float(np.nanmax(y - base))
    if not amp > 0:
        amp = float(np.nanstd(y))

    def model(x, z, sigma, a1, a2, a3, c0, c1):
        c1x = TRIPLET[0][2] * (1 + z)
        c2x = TRIPLET[1][2] * (1 + z)
        c3x = TRIPLET[2][2] * (1 + z)
        return (c0 + c1 * (x - center)
                + a1 * np.exp(-0.5 * ((x - c1x) / sigma) ** 2)
                + a2 * np.exp(-0.5 * ((x - c2x) / sigma) ** 2)
                + a3 * np.exp(-0.5 * ((x - c3x) / sigma) ** 2))

    p0 = [z_seed, 0.0045, 0.5 * amp, 1.0 * amp, 0.7 * amp, base, 0.0]
    lo = [z_seed - 0.06, 0.0007, 0.0, 0.0, 0.0, base - 8 * abs(amp), -1e6]
    hi = [z_seed + 0.06, 0.0200, 10 * abs(amp), 10 * abs(amp), 10 * abs(amp), base + 8 * abs(amp), 1e6]
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=80000)
        fit_y = model(x, *popt)
        return {
            "z_fit": float(popt[0]),
            "sigma_um": float(popt[1]),
            "x": x,
            "y": y,
            "fit_y": fit_y,
            "params": popt.tolist(),
            "status": "ok",
        }
    except Exception as exc:
        return {"status": f"failed: {exc}"}


def dark_axis(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.68)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.2, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_triplet(wave, flux, cont, residual, smooth, df, z_mean, z_fit, unit):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    ha = 0.656281 * (1 + z_mean)
    m = (wave >= ha - 0.09) & (wave <= ha + 0.09)
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    fig = plt.figure(figsize=(18.0, 10.6))
    fig.patch.set_facecolor("#050712")
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.35, 1.25], height_ratios=[1, 1], hspace=0.09, wspace=0.035)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    lane = fig.add_subplot(gs[:, 1])
    dark_axis(fig, ax1)
    dark_axis(fig, ax2)
    lane.set_facecolor("#050712")
    lane.axis("off")

    ax1.plot(wave[m], flux[m], linewidth=1.1, color="#93c5fd", label="Observed flux")
    ax1.plot(wave[m], cont[m], linewidth=1.2, color="#facc15", label="Continuum estimate")
    ax2.plot(wave[m], residual[m], linewidth=0.80, color="#fb7185", alpha=0.45, label="Residual")
    ax2.plot(wave[m], smooth[m], linewidth=1.35, color="#f97316", label="Smoothed residual")
    if z_fit and "x" in z_fit:
        ax2.plot(z_fit["x"], z_fit["fit_y"], linewidth=1.6, color="#22c55e", label=f"3-Gaussian shared z={z_fit['z_fit']:.6f}")

    for i, (_, r) in enumerate(df.iterrows()):
        c = colors[i]
        x = r["measured_peak_um"]
        pred = r["predicted_observed_um"]
        ax1.axvline(x, color=c, linewidth=1.0, alpha=0.72)
        ax2.axvline(x, color=c, linewidth=1.0, alpha=0.72)
        ax2.scatter([x], [r["measured_peak_residual_flux"]], s=62, color=c, edgecolor="#f8fafc", linewidth=0.8, zorder=9)
        ax1.text(x, ax1.get_ylim()[1], f" {int(r['n'])}", color="#050712", fontsize=10.2, weight="bold", ha="left", va="top", bbox=dict(boxstyle="circle,pad=0.20", facecolor=c, edgecolor="#f8fafc"))
        ax2.text(x, ax2.get_ylim()[1], f" {int(r['n'])}", color="#050712", fontsize=10.2, weight="bold", ha="left", va="top", bbox=dict(boxstyle="circle,pad=0.20", facecolor=c, edgecolor="#f8fafc"))

    ax1.set_title(f"{VERSION} — Derive redshift from three observed lines: [N II] / H-alpha / [N II]")
    ax1.set_ylabel(f"Flux, {unit}")
    ax2.set_ylabel(f"Residual, {unit}")
    ax2.set_xlabel("Observed wavelength, micron")
    legend(ax1, "upper left")
    legend(ax2, "upper left")

    lane.text(0.02, 0.985, "Redshift from each line", color="#f8fafc", fontsize=15, weight="bold", va="top")
    lane.text(0.02, 0.945, "Formula: z = observed wavelength / rest wavelength − 1", color="#cbd5e1", fontsize=9.2, va="top", wrap=True)
    lane.text(0.02, 0.898, f"Unweighted mean z = {z_mean:.6f}", color="#e0f2fe", fontsize=10.4)
    if z_fit and "z_fit" in z_fit:
        lane.text(0.02, 0.868, f"Shared Gaussian z = {z_fit['z_fit']:.6f}", color="#bbf7d0", fontsize=10.4)
    y = 0.765
    for i, (_, r) in enumerate(df.iterrows()):
        c = colors[i]
        lane.text(0.02, y, f"{int(r['n'])}", color="#050712", fontsize=10.5, weight="bold", bbox=dict(boxstyle="circle,pad=0.25", facecolor=c, edgecolor="#f8fafc"))
        lane.text(0.13, y + 0.024, str(r["line"]), color=c, fontsize=11.4, weight="bold", va="center")
        lane.text(0.13, y - 0.012, f"rest λ = {r['rest_um']:.6f} µm", color="#94a3b8", fontsize=8.6)
        lane.text(0.13, y - 0.046, f"observed peak λ = {r['measured_peak_um']:.6f} µm", color="#dbeafe", fontsize=8.6)
        lane.text(0.13, y - 0.080, f"z = {r['z_line']:.6f}", color="#f8fafc", fontsize=8.9)
        y -= 0.190
    lane.text(0.02, 0.070, "Interpretation:\nThe line species are identified because the known rest-frame spacing is preserved after multiplying all wavelengths by the same (1+z).", color="#f8fafc", fontsize=9.6, bbox=dict(boxstyle="round,pad=0.45", facecolor="#020617", edgecolor="#475569"))

    fig.tight_layout()
    path = PNG / f"{VERSION}_THREE_LINE_REDSHIFT_DERIVATION.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_table(df, z_mean, z_std, z_amp, z_catalog, z_fit):
    import matplotlib.pyplot as plt
    view = df[["n", "line", "rest_um", "measured_peak_um", "z_line", "offset_nm_vs_scan"]].copy()
    for c in ["rest_um", "measured_peak_um", "z_line", "offset_nm_vs_scan"]:
        view[c] = view[c].map(lambda x: f"{x:.6f}")
    view.columns = ["#", "line", "rest λ µm", "observed peak λ µm", "z line", "offset nm"]
    fig, ax = plt.subplots(figsize=(16.4, 5.1))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    fit_txt = f" | shared-fit z={z_fit['z_fit']:.6f}" if z_fit and "z_fit" in z_fit else ""
    ax.set_title(f"{VERSION} — Three-line redshift average | mean z={z_mean:.6f}, std={z_std:.6f}, amp-weighted z={z_amp:.6f}{fit_txt}", color="#f8fafc", fontsize=13.2, pad=12)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.1)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#020617" if r % 2 else "#0f172a")
            cell.get_text().set_color("#dbeafe")
            if "H-alpha" in str(view.iloc[r-1, 1]):
                cell.set_facecolor("#450a0a")
                cell.get_text().set_color("#fecaca")
    fig.tight_layout()
    path = PNG / f"{VERSION}_THREE_LINE_REDSHIFT_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def upload(paths, token):
    import pandas as pd
    rows = []
    for p in paths:
        if p is None:
            continue
        sub = "PNG" if p.suffix.lower() == ".png" else "CSV"
        repo_path = f"{DEST_ROOT}/{sub}/{p.name}"
        raw, status = github_put_file(token, p, repo_path)
        rows.append({"type": sub.lower(), "status": status, "file": p.name, "raw_url": raw})
    links_csv = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    links_json = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"
    pd.DataFrame(rows).to_csv(links_csv, index=False)
    links_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for p in [links_csv, links_json]:
        raw, status = github_put_file(token, p, f"{DEST_ROOT}/CSV/{p.name}")
        rows.append({"type": p.suffix.lower().replace(".", ""), "status": status, "file": p.name, "raw_url": raw})
    return rows


def display_links(rows):
    print("\nIMAGES AVAILABLE HERE:")
    for r in rows:
        if r["type"] == "png":
            print(r["file"])
            print(r["raw_url"] if r["raw_url"] else "local only; no upload token available")
    try:
        from IPython.display import HTML, display
        previews = "".join(f'<h4 style="color:#e5e7eb">{r["file"]}</h4><a href="{r["raw_url"]}" target="_blank"><img src="{r["raw_url"]}" style="max-width:100%;border:1px solid #475569;border-radius:8px"></a>' for r in rows if r["type"] == "png" and r["raw_url"])
        if previews:
            display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Three-line redshift derivation</h2>{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    token = get_token(optional=True)
    path, dl = download_spectrum()
    wave, flux, unit = read_spectrum(path)
    cont, residual, smooth = continuum_residual(wave, flux)
    z_grid, scores, z_seed = scan_z(wave, smooth)
    triplet_df, z_mean, z_std, z_amp, z_catalog = direct_triplet_z(wave, smooth, z_seed)
    fit = shared_gaussian_fit(wave, residual, z_mean)

    import pandas as pd
    triplet_csv = CSV / f"{VERSION}_LINE_BY_LINE_REDSHIFT.csv"
    summary_csv = CSV / f"{VERSION}_REDSHIFT_AVERAGE_SUMMARY.csv"
    score_csv = CSV / f"{VERSION}_REDSHIFT_SCAN_SCORE.csv"
    triplet_df.to_csv(triplet_csv, index=False)
    pd.DataFrame({"z": z_grid, "score": scores}).to_csv(score_csv, index=False)
    pd.DataFrame([{
        "z_scan_seed": z_seed,
        "z_unweighted_mean": z_mean,
        "z_sample_std": z_std,
        "z_amplitude_weighted": z_amp,
        "z_catalog_weighted": z_catalog,
        "z_shared_gaussian_fit": fit.get("z_fit") if fit and "z_fit" in fit else None,
        "gaussian_sigma_um": fit.get("sigma_um") if fit and "sigma_um" in fit else None,
        "download_status": dl,
        "mast_product": PRODUCT,
    }]).to_csv(summary_csv, index=False)

    png1 = plot_triplet(wave, flux, cont, residual, smooth, triplet_df, z_mean, fit, unit)
    png2 = plot_table(triplet_df, z_mean, z_std, z_amp, z_catalog, fit)
    uploaded = upload([png1, png2, triplet_csv, summary_csv, score_csv], token)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("MAST product", PRODUCT),
        ("Download status", dl),
        ("Formula", "z = lambda_observed / lambda_rest - 1"),
        ("Line-comb seed z", f"{z_seed:.6f}"),
        ("Unweighted mean z", f"{z_mean:.6f}"),
        ("Sample std z", f"{z_std:.6f}"),
        ("Amplitude-weighted z", f"{z_amp:.6f}"),
        ("Catalog-weighted z", f"{z_catalog:.6f}"),
        ("Shared Gaussian z", f"{fit.get('z_fit'):.6f}" if fit and "z_fit" in fit else "not used"),
    ], ["Field", "Value"])

    print("\nLINE-BY-LINE REDSHIFT")
    rows = []
    for _, r in triplet_df.iterrows():
        rows.append((int(r["n"]), r["line"], f"{r['rest_um']:.6f}", f"{r['measured_peak_um']:.6f}", f"{r['z_line']:.6f}", f"{r['offset_nm_vs_scan']:.3f}"))
    print_table(rows, ["#", "Line", "Rest um", "Observed um", "z", "Offset nm"])

    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"] or "local only") for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)

    print("\nCOMMENTS")
    print("For learning, averaging the three z values is a valid direct curve-reading method.")
    print("For publication-grade spectroscopy, the cleaner method is a simultaneous fit or cross-correlation using many lines and uncertainties.")
    print("The three-line pattern identifies the species because the known rest-wavelength spacing is preserved by one shared redshift.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
