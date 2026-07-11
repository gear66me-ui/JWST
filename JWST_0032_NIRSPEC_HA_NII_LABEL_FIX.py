# JWST_0032
# Audit: Actual JWST/NIRSpec H-alpha + [N II] redshift plot with label lane. Python/matplotlib only. No AI images.

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

VERSION = "JWST_0032"
PROJECT = "JWST NIRSPEC H-ALPHA NII REDSHIFT LABEL FIX"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0032_NIRSPEC_HA_NII_LABEL_FIX"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_DIRECT_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"

# Rest-frame vacuum/air-near optical reference wavelengths in microns for the educational line ID.
# These are the classic H-alpha + [N II] triplet used by the STScI notebook redshift example.
TRIPLET = [
    (1, "[N II] 6548", 0.654805, "forbidden nitrogen line", 0.45),
    (2, "Hα 6563",     0.656281, "hydrogen Balmer alpha", 1.00),
    (3, "[N II] 6583", 0.658345, "forbidden nitrogen line", 0.70),
]

FULL_LINES = [
    ("Hβ", 0.486133, 0.45),
    ("[O III] 4959", 0.495891, 0.45),
    ("[O III] 5007", 0.500684, 0.90),
    ("[N II] 6548", 0.654805, 0.45),
    ("Hα", 0.656281, 1.00),
    ("[N II] 6583", 0.658345, 0.70),
    ("[S II] 6716", 0.671647, 0.30),
    ("[S II] 6731", 0.673085, 0.30),
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
    if not rows:
        widths = [len(str(h)) for h in headers]
    else:
        widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
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
            if data is None:
                continue
            names = list(getattr(data, "names", []) or [])
            if "WAVELENGTH" in names and "FLUX" in names:
                ext = hdu
                break
        if ext is None:
            raise RuntimeError("No FITS table extension with WAVELENGTH and FLUX columns found.")
        wave = np.asarray(ext.data["WAVELENGTH"], dtype=float)
        flux = np.asarray(ext.data["FLUX"], dtype=float)
        flux_unit = str(ext.header.get("TUNIT2", "unknown"))
    mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave = wave[mask]
    flux = flux[mask]
    if flux_unit.lower() == "jy":
        flux_plot = flux * 1.0e6
        plot_unit = "microJy"
    else:
        flux_plot = flux
        plot_unit = flux_unit
    order = np.argsort(wave)
    return wave[order], flux_plot[order], plot_unit


def continuum_and_residual(wave, flux):
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import median_filter
    n = len(flux)
    kernel = max(101, int(n * 0.055) // 2 * 2 + 1)
    kernel = min(kernel, n - 1 if (n - 1) % 2 == 1 else n - 2)
    cont = median_filter(flux, size=kernel, mode="nearest")
    sg_window = max(51, int(n * 0.035) // 2 * 2 + 1)
    sg_window = min(sg_window, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if sg_window >= 7:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cont = savgol_filter(cont, sg_window, 2, mode="interp")
    residual = flux - cont
    res_window = max(7, int(n * 0.0055) // 2 * 2 + 1)
    res_window = min(res_window, n - 1 if (n - 1) % 2 == 1 else n - 2)
    residual_smooth = savgol_filter(residual, res_window, 2, mode="interp") if res_window >= 7 else residual
    return cont, residual, residual_smooth


def robust_norm(values):
    import numpy as np
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(values)
    return (values - med) / (scale if scale and scale == scale else 1.0)


def coarse_redshift_scan(wave, residual_smooth):
    import numpy as np
    yy = robust_norm(residual_smooth)
    z_grid = np.linspace(2.20, 2.75, 2201)
    scores = []
    for z in z_grid:
        total = 0.0
        wsum = 0.0
        for label, rest, weight in FULL_LINES:
            obs = rest * (1 + z)
            if wave.min() <= obs <= wave.max():
                total += weight * np.interp(obs, wave, yy)
                wsum += weight
        scores.append(total / wsum if wsum > 0 else np.nan)
    scores = np.asarray(scores)
    idx = int(np.nanargmax(scores))
    z = float(z_grid[idx])
    if 2 <= idx < len(z_grid) - 2:
        xs = z_grid[idx-2:idx+3]
        ys = scores[idx-2:idx+3]
        coeff = np.polyfit(xs, ys, 2)
        if coeff[0] < 0:
            zv = -coeff[1] / (2 * coeff[0])
            if xs.min() <= zv <= xs.max():
                z = float(zv)
    return z_grid, scores, z


def local_peak(wave, residual_smooth, center_um, half_width_um):
    import numpy as np
    mask = (wave >= center_um - half_width_um) & (wave <= center_um + half_width_um)
    if mask.sum() < 5:
        return float("nan"), float("nan")
    ww = wave[mask]
    yy = residual_smooth[mask]
    i = int(np.nanargmax(yy))
    peak_x = float(ww[i])
    peak_y = float(yy[i])
    j0 = max(0, i - 4)
    j1 = min(len(ww), i + 5)
    if j1 - j0 >= 5:
        try:
            coeff = np.polyfit(ww[j0:j1], yy[j0:j1], 2)
            if coeff[0] < 0:
                xv = -coeff[1] / (2 * coeff[0])
                if ww[j0] <= xv <= ww[j1 - 1]:
                    peak_x = float(xv)
                    peak_y = float(np.polyval(coeff, xv))
        except Exception:
            pass
    return peak_x, peak_y


def triplet_measurements(wave, residual_smooth, z_guess):
    import pandas as pd
    rows = []
    for n, label, rest, species, weight in TRIPLET:
        predicted = rest * (1.0 + z_guess)
        peak_um, peak_flux = local_peak(wave, residual_smooth, predicted, half_width_um=0.018)
        z_peak = peak_um / rest - 1.0 if peak_um == peak_um else float("nan")
        rows.append({
            "n": n,
            "line": label,
            "species": species,
            "rest_wavelength_um": rest,
            "predicted_observed_um_from_scan_z": predicted,
            "local_peak_observed_um": peak_um,
            "z_from_line_peak": z_peak,
            "offset_from_scan_nm": (peak_um - predicted) * 1000.0 if peak_um == peak_um else float("nan"),
            "relative_weight": weight,
            "peak_residual_flux": peak_flux,
        })
    df = pd.DataFrame(rows)
    valid = df["z_from_line_peak"].dropna()
    z_mean = float(valid.mean()) if len(valid) else float("nan")
    z_std = float(valid.std(ddof=1)) if len(valid) > 1 else float("nan")
    return df, z_mean, z_std


def shared_z_gaussian_fit(wave, residual, z_init):
    import numpy as np
    from scipy.optimize import curve_fit

    center = 0.656281 * (1 + z_init)
    mask = (wave >= center - 0.080) & (wave <= center + 0.080)
    x = wave[mask]
    y = residual[mask]
    if len(x) < 20:
        return None
    ymed = np.nanmedian(y)
    ymax = np.nanmax(y - ymed)
    ymax = ymax if ymax > 0 else np.nanstd(y)

    def model(x, z, sigma, a1, a2, a3, c0, c1):
        centers = [rest * (1 + z) for _, _, rest, _, _ in TRIPLET]
        base = c0 + c1 * (x - center)
        return base + a1 * np.exp(-0.5 * ((x - centers[0]) / sigma) ** 2) + a2 * np.exp(-0.5 * ((x - centers[1]) / sigma) ** 2) + a3 * np.exp(-0.5 * ((x - centers[2]) / sigma) ** 2)

    p0 = [z_init, 0.0045, ymax * 0.5, ymax, ymax * 0.7, ymed, 0.0]
    lo = [z_init - 0.08, 0.0008, 0.0, 0.0, 0.0, ymed - 5 * abs(ymax), -1e6]
    hi = [z_init + 0.08, 0.0200, 10 * abs(ymax), 10 * abs(ymax), 10 * abs(ymax), ymed + 5 * abs(ymax), 1e6]
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=50000)
        z_fit = float(popt[0])
        sigma = float(popt[1])
        yfit = model(x, *popt)
        rss = float(np.nansum((y - yfit) ** 2))
        return {"z_fit": z_fit, "sigma_um": sigma, "x": x, "y": y, "yfit": yfit, "rss": rss, "params": popt.tolist()}
    except Exception as exc:
        return {"error": str(exc)}


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


def style_legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.1, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_clean_triplet(wave, flux, cont, residual, residual_smooth, triplet_df, z_scan, z_avg, fit, plot_unit):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    ha_center = 0.656281 * (1 + z_scan)
    x0 = ha_center - 0.085
    x1 = ha_center + 0.085
    mask = (wave >= x0) & (wave <= x1)

    fig = plt.figure(figsize=(17.8, 10.2))
    fig.patch.set_facecolor("#050712")
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.25, 1.25], height_ratios=[1.0, 1.0], hspace=0.10, wspace=0.035)
    ax_flux = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_flux)
    lane = fig.add_subplot(gs[:, 1])
    dark_axis(fig, ax_flux)
    dark_axis(fig, ax_res)
    lane.set_facecolor("#050712")
    lane.axis("off")

    ax_flux.plot(wave[mask], flux[mask], color="#93c5fd", linewidth=1.15, label="Observed flux")
    ax_flux.plot(wave[mask], cont[mask], color="#facc15", linewidth=1.20, label="Continuum estimate")
    ax_res.plot(wave[mask], residual[mask], color="#fb7185", linewidth=0.85, alpha=0.50, label="Residual")
    ax_res.plot(wave[mask], residual_smooth[mask], color="#f97316", linewidth=1.35, label="Smoothed residual")

    if fit and "x" in fit:
        ax_res.plot(fit["x"], fit["yfit"], color="#22c55e", linewidth=1.55, label=f"3-line shared-z fit, z={fit['z_fit']:.6f}")

    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    y_flux_top = ax_flux.get_ylim()[1]
    y_res_top = ax_res.get_ylim()[1]
    y_res_bottom = ax_res.get_ylim()[0]
    for (_, row), color in zip(triplet_df.iterrows(), colors):
        n = int(row["n"])
        x_pred = row["predicted_observed_um_from_scan_z"]
        x_peak = row["local_peak_observed_um"]
        for ax in [ax_flux, ax_res]:
            ax.axvline(x_pred, color=color, linewidth=0.95, alpha=0.72)
        ax_res.scatter([x_peak], [row["peak_residual_flux"]], s=62, color=color, edgecolor="#f8fafc", linewidth=0.8, zorder=8)
        ax_res.text(x_pred, y_res_top, f" {n}", color="#f8fafc", fontsize=11, weight="bold", va="top", ha="left",
                    bbox=dict(boxstyle="circle,pad=0.22", facecolor=color, edgecolor="#f8fafc", alpha=0.90))
        ax_flux.text(x_pred, y_flux_top, f" {n}", color="#f8fafc", fontsize=10.5, weight="bold", va="top", ha="left",
                     bbox=dict(boxstyle="circle,pad=0.18", facecolor=color, edgecolor="#f8fafc", alpha=0.90))

    ax_flux.set_ylabel(f"Flux, {plot_unit}")
    ax_res.set_ylabel(f"Residual, {plot_unit}")
    ax_res.set_xlabel("Observed wavelength, micron")
    ax_flux.set_title(f"{VERSION} — H-alpha + [N II] triplet, labels moved into right lane")
    style_legend(ax_flux, "upper left")
    style_legend(ax_res, "upper left")

    lane.text(0.02, 0.985, "Right-side label lane", color="#f8fafc", fontsize=15, weight="bold", va="top")
    lane.text(0.02, 0.945, "Species are identified by matching a known rest-wavelength pattern to the observed peaks.", color="#cbd5e1", fontsize=9.2, va="top", wrap=True)
    lane.text(0.02, 0.880, f"Scan z       = {z_scan:.6f}", color="#e0f2fe", fontsize=10.2)
    lane.text(0.02, 0.850, f"Triplet avg z = {z_avg:.6f}", color="#e0f2fe", fontsize=10.2)
    if fit and "z_fit" in fit:
        lane.text(0.02, 0.820, f"3-Gaussian z = {fit['z_fit']:.6f}", color="#bbf7d0", fontsize=10.2)

    y = 0.740
    for i, (_, row) in enumerate(triplet_df.iterrows()):
        color = colors[i]
        lane.text(0.02, y, f"{int(row['n'])}", color="#050712", fontsize=10.5, weight="bold",
                  bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="#f8fafc", linewidth=0.8))
        lane.text(0.13, y + 0.022, row["line"], color=color, fontsize=11.5, weight="bold", va="center")
        lane.text(0.13, y - 0.013, row["species"], color="#dbeafe", fontsize=8.8, va="center")
        lane.text(0.13, y - 0.047, f"rest λ = {row['rest_wavelength_um']:.6f} µm", color="#94a3b8", fontsize=8.4, va="center")
        lane.text(0.13, y - 0.079, f"peak λobs = {row['local_peak_observed_um']:.6f} µm", color="#e0f2fe", fontsize=8.4, va="center")
        lane.text(0.13, y - 0.111, f"z = λobs/λrest - 1 = {row['z_from_line_peak']:.6f}", color="#f8fafc", fontsize=8.4, va="center")
        y -= 0.205

    lane.text(0.02, 0.070, "Core rule:\nz = λ_observed / λ_rest − 1\nSame z for all three lines = species ID confidence.",
              color="#f8fafc", fontsize=10.0,
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#020617", edgecolor="#475569", alpha=0.92))

    fig.tight_layout()
    path = PNG / f"{VERSION}_HALPHA_NII_TRIPLET_LABEL_LANE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_rest_vs_observed(triplet_df, z_avg, fit):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.8, 7.2))
    dark_axis(fig, ax)
    ax.scatter(triplet_df["rest_wavelength_um"], triplet_df["local_peak_observed_um"], s=90, color="#38bdf8", edgecolor="#f8fafc", label="Measured peak")
    xmin = triplet_df["rest_wavelength_um"].min() - 0.0012
    xmax = triplet_df["rest_wavelength_um"].max() + 0.0012
    xs = [xmin, xmax]
    ax.plot(xs, [(1 + z_avg) * x for x in xs], color="#f97316", linewidth=1.7, label=f"λobs=(1+z)λrest, z={z_avg:.6f}")
    if fit and "z_fit" in fit:
        ax.plot(xs, [(1 + fit["z_fit"]) * x for x in xs], color="#22c55e", linewidth=1.3, linestyle="--", label=f"shared-z Gaussian fit, z={fit['z_fit']:.6f}")
    for _, row in triplet_df.iterrows():
        ax.text(row["rest_wavelength_um"], row["local_peak_observed_um"], f" {int(row['n'])} {row['line']}", color="#dbeafe", fontsize=9, va="bottom")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed peak wavelength, micron")
    ax.set_title("Redshift derivation: the H-alpha/[N II] triplet falls on one straight scaling line")
    style_legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_REST_VS_OBSERVED_REDSHIFT_LINE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_table_png(triplet_df, z_scan, z_avg, z_std, fit):
    import matplotlib.pyplot as plt
    view = triplet_df[["n", "line", "rest_wavelength_um", "local_peak_observed_um", "z_from_line_peak", "offset_from_scan_nm"]].copy()
    view["n"] = view["n"].map(lambda x: f"{int(x)}")
    for c in ["rest_wavelength_um", "local_peak_observed_um", "z_from_line_peak", "offset_from_scan_nm"]:
        view[c] = view[c].map(lambda x: "" if x != x else f"{x:.6f}")
    view.columns = ["#", "species line", "rest λ µm", "observed peak λ µm", "z", "offset nm"]
    fit_text = f" | fit z={fit['z_fit']:.6f}" if fit and "z_fit" in fit else ""
    fig, ax = plt.subplots(figsize=(15.6, 4.5))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — Direct redshift from H-alpha/[N II] | scan z={z_scan:.6f}, avg z={z_avg:.6f}, std={z_std:.6f}{fit_text}", color="#f8fafc", fontsize=13.5, pad=12)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
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
            if "H" in str(view.iloc[r-1, 1]):
                cell.set_facecolor("#450a0a")
                cell.get_text().set_color("#fecaca")
    fig.tight_layout()
    path = PNG / f"{VERSION}_TRIPLET_REDSHIFT_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def upload_outputs(paths, token):
    import pandas as pd
    rows = []
    for local_path in paths:
        if local_path is None:
            continue
        sub = "PNG" if local_path.suffix.lower() == ".png" else "CSV"
        repo_path = f"{DEST_ROOT}/{sub}/{local_path.name}"
        raw, status = github_put_file(token, local_path, repo_path)
        rows.append({"type": sub.lower(), "status": status, "file": local_path.name, "raw_url": raw})
    links_csv = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    links_json = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"
    pd.DataFrame(rows).to_csv(links_csv, index=False)
    links_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for p in [links_csv, links_json]:
        repo_path = f"{DEST_ROOT}/CSV/{p.name}"
        raw, status = github_put_file(token, p, repo_path)
        rows.append({"type": p.suffix.lower().replace(".", ""), "status": status, "file": p.name, "raw_url": raw})
    return rows


def display_links(rows):
    image_rows = [r for r in rows if r["type"] == "png"]
    print("\nIMAGES AVAILABLE HERE:")
    for r in image_rows:
        print(r["file"])
        print(r["raw_url"] if r["raw_url"] else "local only; no upload token available")
    try:
        from IPython.display import HTML, display
        links = "".join(f'<li><a href="{r["raw_url"]}" target="_blank">{r["file"]}</a><br><code>{r["raw_url"]}</code></li>' for r in image_rows if r["raw_url"])
        previews = "".join(f'<h4 style="color:#e5e7eb">{r["file"]}</h4><a href="{r["raw_url"]}" target="_blank"><img src="{r["raw_url"]}" style="max-width:100%;border:1px solid #475569;border-radius:8px"></a>' for r in image_rows if r["raw_url"])
        if links:
            display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Images available here</h2><p>H-alpha/[N II] triplet labels moved right; redshift calculation included.</p><ul>{links}</ul><hr style="border-color:#334155">{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    token = get_token(optional=True)
    path, download_status = download_spectrum()
    wave, flux, plot_unit = read_spectrum(path)
    cont, residual, residual_smooth = continuum_and_residual(wave, flux)
    z_grid, score, z_scan = coarse_redshift_scan(wave, residual_smooth)
    triplet_df, z_avg, z_std = triplet_measurements(wave, residual_smooth, z_scan)
    fit = shared_z_gaussian_fit(wave, residual, z_avg)

    import pandas as pd
    triplet_csv = CSV / f"{VERSION}_HALPHA_NII_TRIPLET_REDSHIFT.csv"
    spectrum_csv = CSV / f"{VERSION}_ZOOM_SPECTRUM_DATA.csv"
    fit_csv = CSV / f"{VERSION}_GAUSSIAN_FIT_SUMMARY.csv"
    triplet_df.to_csv(triplet_csv, index=False)
    center = 0.656281 * (1 + z_scan)
    mask = (wave >= center - 0.10) & (wave <= center + 0.10)
    pd.DataFrame({
        "wavelength_um": wave[mask],
        f"flux_{plot_unit}": flux[mask],
        f"continuum_{plot_unit}": cont[mask],
        f"residual_{plot_unit}": residual[mask],
        f"residual_smooth_{plot_unit}": residual_smooth[mask],
    }).to_csv(spectrum_csv, index=False)
    pd.DataFrame([{
        "z_scan": z_scan,
        "z_triplet_average": z_avg,
        "z_triplet_std": z_std,
        "z_gaussian_shared_fit": fit.get("z_fit") if fit and "z_fit" in fit else None,
        "gaussian_sigma_um": fit.get("sigma_um") if fit and "sigma_um" in fit else None,
        "fit_status": "ok" if fit and "z_fit" in fit else str(fit),
    }]).to_csv(fit_csv, index=False)

    png1 = plot_clean_triplet(wave, flux, cont, residual, residual_smooth, triplet_df, z_scan, z_avg, fit, plot_unit)
    png2 = plot_rest_vs_observed(triplet_df, z_avg, fit)
    png3 = plot_table_png(triplet_df, z_scan, z_avg, z_std, fit)
    uploaded = upload_outputs([png1, png2, png3, triplet_csv, spectrum_csv, fit_csv], token)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("MAST product", PRODUCT),
        ("Download status", download_status),
        ("Core species ID", "left=[N II] 6548, center=H-alpha 6563, right=[N II] 6583"),
        ("Formula", "z = lambda_observed / lambda_rest - 1"),
        ("Line-comb scan z", f"{z_scan:.6f}"),
        ("Triplet average z", f"{z_avg:.6f}"),
        ("Triplet std z", f"{z_std:.6f}"),
        ("Shared-z Gaussian fit", f"{fit.get('z_fit'):.6f}" if fit and "z_fit" in fit else "fit failed / not used"),
    ], ["Field", "Value"])

    print("\nDIRECT LINE REDSHIFT")
    rows = []
    for _, r in triplet_df.iterrows():
        rows.append((int(r["n"]), r["line"], f"{r['rest_wavelength_um']:.6f}", f"{r['local_peak_observed_um']:.6f}", f"{r['z_from_line_peak']:.6f}", f"{r['offset_from_scan_nm']:.3f}"))
    print_table(rows, ["#", "Line", "Rest um", "Peak obs um", "z", "Offset nm"])

    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"] or "local only") for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)

    print("\nCOMMENTS")
    print("The three close peaks are identified by their spacing pattern: [N II] 6548, H-alpha 6563, [N II] 6583.")
    print("The same redshift applied to their known rest wavelengths predicts the observed triplet positions.")
    print("Species ID is pattern matching plus physics of known emission transitions, not just peak height.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
