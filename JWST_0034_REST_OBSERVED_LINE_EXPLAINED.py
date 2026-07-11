# JWST_0034
# Audit: Explain observed-peak wavelength versus rest-frame wavelength using actual JWST/NIRSpec triplet data. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import os, sys, json, base64, getpass, subprocess, importlib, warnings

VERSION = "JWST_0034"
PROJECT = "REST VS OBSERVED REDSHIFT LINE EXPLAINED"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0034_REST_OBSERVED_LINE_EXPLAINED"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen, forbidden line", 0.45),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha", 1.00),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen, forbidden line", 0.70),
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


def get_token():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        return token.strip() if token else ""
    except Exception:
        return ""


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
    r = requests.get(MAST_URL, timeout=120)
    r.raise_for_status()
    local.write_bytes(r.content)
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
            raise RuntimeError("No WAVELENGTH/FLUX table found in FITS file.")
        wave = np.asarray(ext.data["WAVELENGTH"], dtype=float)
        flux = np.asarray(ext.data["FLUX"], dtype=float)
        unit = str(ext.header.get("TUNIT2", "unknown"))
    mask = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[mask], flux[mask]
    if unit.lower() == "jy":
        flux = flux * 1.0e6
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
    resid = flux - cont
    sw = max(7, int(n * 0.0055) // 2 * 2 + 1)
    sw = min(sw, n - 1 if (n - 1) % 2 else n - 2)
    smooth = savgol_filter(resid, sw, 2, mode="interp") if sw >= 7 else resid
    return cont, resid, smooth


def robust_norm(y):
    import numpy as np
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(y)
    return (y - med) / (scale if scale and scale == scale else 1.0)


def scan_z(wave, smooth):
    import numpy as np
    yy = robust_norm(smooth)
    z_grid = np.linspace(2.20, 2.75, 2201)
    score = []
    for z in z_grid:
        total = 0.0
        wsum = 0.0
        for _, rest, weight in SCAN_LINES:
            x = rest * (1 + z)
            if wave.min() <= x <= wave.max():
                total += weight * np.interp(x, wave, yy)
                wsum += weight
        score.append(total / wsum if wsum else float("nan"))
    score = np.asarray(score)
    i = int(np.nanargmax(score))
    z = float(z_grid[i])
    if 2 <= i < len(z_grid) - 2:
        xs, ys = z_grid[i - 2:i + 3], score[i - 2:i + 3]
        c = np.polyfit(xs, ys, 2)
        if c[0] < 0:
            zv = -c[1] / (2 * c[0])
            if xs.min() <= zv <= xs.max():
                z = float(zv)
    return z_grid, score, z


def local_peak(wave, smooth, center, half_width=0.018):
    import numpy as np
    m = (wave >= center - half_width) & (wave <= center + half_width)
    if m.sum() < 5:
        return float("nan"), float("nan")
    x, y = wave[m], smooth[m]
    i = int(np.nanargmax(y))
    px, py = float(x[i]), float(y[i])
    j0, j1 = max(0, i - 4), min(len(x), i + 5)
    if j1 - j0 >= 5:
        try:
            c = np.polyfit(x[j0:j1], y[j0:j1], 2)
            if c[0] < 0:
                xv = -c[1] / (2 * c[0])
                if x[j0] <= xv <= x[j1 - 1]:
                    px = float(xv)
                    py = float(np.polyval(c, xv))
        except Exception:
            pass
    return px, py


def measure_triplet(wave, smooth, z_seed):
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
            "predicted_from_seed_um": predicted,
            "observed_peak_um": peak,
            "z_line": z_line,
            "peak_residual_flux": amp,
            "offset_nm_vs_seed": (peak - predicted) * 1000 if peak == peak else float("nan"),
        })
    df = pd.DataFrame(rows)
    z_mean = float(df["z_line"].mean())
    z_std = float(df["z_line"].std(ddof=1))
    return df, z_mean, z_std


def shared_gaussian_fit(wave, resid, z_seed):
    import numpy as np
    from scipy.optimize import curve_fit
    center = 0.656281 * (1 + z_seed)
    m = (wave >= center - 0.085) & (wave <= center + 0.085)
    x, y = wave[m], resid[m]
    if len(x) < 20:
        return {}
    base = float(np.nanmedian(y))
    amp = float(np.nanmax(y - base))
    if not amp > 0:
        amp = float(np.nanstd(y))

    def model(x, z, sigma, a1, a2, a3, c0, c1):
        cxs = [rest * (1 + z) for _, _, rest, _, _ in TRIPLET]
        return (c0 + c1 * (x - center)
                + a1 * np.exp(-0.5 * ((x - cxs[0]) / sigma) ** 2)
                + a2 * np.exp(-0.5 * ((x - cxs[1]) / sigma) ** 2)
                + a3 * np.exp(-0.5 * ((x - cxs[2]) / sigma) ** 2))

    p0 = [z_seed, 0.0045, 0.5 * amp, amp, 0.7 * amp, base, 0.0]
    lo = [z_seed - 0.06, 0.0007, 0.0, 0.0, 0.0, base - 8 * abs(amp), -1e6]
    hi = [z_seed + 0.06, 0.0200, 10 * abs(amp), 10 * abs(amp), 10 * abs(amp), base + 8 * abs(amp), 1e6]
    try:
        popt, pcov = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=80000)
        return {"z_fit": float(popt[0]), "sigma_um": float(popt[1]), "x": x, "yfit": model(x, *popt), "status": "ok"}
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
    for s in ax.spines.values():
        s.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.5, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_triplet_zoom(wave, flux, cont, resid, smooth, df, fit, z_mean, unit):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    center = 0.656281 * (1 + z_mean)
    m = (wave >= center - 0.09) & (wave <= center + 0.09)
    fig = plt.figure(figsize=(18.0, 10.2))
    fig.patch.set_facecolor("#050712")
    gs = gridspec.GridSpec(2, 2, width_ratios=[3.35, 1.35], height_ratios=[1, 1], hspace=0.08, wspace=0.035)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    lane = fig.add_subplot(gs[:, 1])
    dark_axis(fig, ax1); dark_axis(fig, ax2)
    lane.set_facecolor("#050712"); lane.axis("off")
    ax1.plot(wave[m], flux[m], color="#93c5fd", lw=1.10, label="Observed flux")
    ax1.plot(wave[m], cont[m], color="#facc15", lw=1.15, label="Continuum")
    ax2.plot(wave[m], resid[m], color="#fb7185", lw=0.75, alpha=0.45, label="Residual")
    ax2.plot(wave[m], smooth[m], color="#f97316", lw=1.35, label="Smoothed residual")
    if fit and "x" in fit:
        ax2.plot(fit["x"], fit["yfit"], color="#22c55e", lw=1.60, label=f"Shared-z fit: z={fit['z_fit']:.6f}")
    for i, (_, r) in enumerate(df.iterrows()):
        color = colors[i]
        x = r["observed_peak_um"]
        for ax in [ax1, ax2]:
            ax.axvline(x, color=color, lw=1.0, alpha=0.80)
            ax.text(x, ax.get_ylim()[1], f" {int(r['n'])}", color="#050712", fontsize=11, weight="bold", ha="left", va="top", bbox=dict(boxstyle="circle,pad=0.24", facecolor=color, edgecolor="#f8fafc"))
        ax2.scatter([x], [r["peak_residual_flux"]], s=70, color=color, edgecolor="#f8fafc", zorder=10)
    ax1.set_title(f"{VERSION} — all three peaks numbered: 1=[N II] 6548, 2=H-alpha, 3=[N II] 6583")
    ax1.set_ylabel(f"Flux, {unit}")
    ax2.set_ylabel(f"Residual, {unit}")
    ax2.set_xlabel("Observed wavelength, micron")
    legend(ax1, "upper left"); legend(ax2, "upper left")
    lane.text(0.02, 0.985, "Peak label lane", color="#f8fafc", fontsize=16, weight="bold", va="top")
    lane.text(0.02, 0.945, "Each peak is converted to redshift using z = λobs / λrest − 1.", color="#cbd5e1", fontsize=9.5, wrap=True, va="top")
    y = 0.835
    for i, (_, r) in enumerate(df.iterrows()):
        color = colors[i]
        lane.text(0.02, y, f"{int(r['n'])}", color="#050712", fontsize=11, weight="bold", bbox=dict(boxstyle="circle,pad=0.30", facecolor=color, edgecolor="#f8fafc"))
        lane.text(0.13, y + 0.030, str(r["line"]), color=color, fontsize=12, weight="bold")
        lane.text(0.13, y - 0.006, f"rest λ = {r['rest_um']:.6f} µm", color="#94a3b8", fontsize=8.6)
        lane.text(0.13, y - 0.040, f"observed peak = {r['observed_peak_um']:.6f} µm", color="#dbeafe", fontsize=8.6)
        lane.text(0.13, y - 0.074, f"z_line = {r['z_line']:.6f}", color="#f8fafc", fontsize=8.9)
        y -= 0.205
    lane.text(0.02, 0.090, f"Average of the three z values:\nz_mean = {z_mean:.6f}\n\nPattern test:\nknown rest spacing × (1+z) lands on all three observed peaks.", color="#f8fafc", fontsize=10, bbox=dict(boxstyle="round,pad=0.48", facecolor="#020617", edgecolor="#475569"))
    fig.tight_layout()
    p = PNG / f"{VERSION}_TRIPLET_ALL_1_2_3_LABELS.png"
    fig.savefig(p, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return p


def plot_rest_observed(df, z_mean, fit):
    import matplotlib.pyplot as plt
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    fig, ax = plt.subplots(figsize=(15.2, 8.2))
    dark_axis(fig, ax)
    xmin = df["rest_um"].min() - 0.0014
    xmax = df["rest_um"].max() + 0.0014
    xs = [xmin, xmax]
    orange_y = [(1 + z_mean) * x for x in xs]
    ax.plot(xs, orange_y, color="#f97316", lw=2.0, label=f"Orange: unweighted average redshift, λobs=(1+{z_mean:.6f})λrest")
    if fit and "z_fit" in fit:
        green_y = [(1 + fit["z_fit"]) * x for x in xs]
        ax.plot(xs, green_y, color="#22c55e", lw=1.55, ls="--", label=f"Green dashed: simultaneous 3-Gaussian shared-z fit, z={fit['z_fit']:.6f}")
    for i, (_, r) in enumerate(df.iterrows()):
        color = colors[i]
        ax.scatter([r["rest_um"]], [r["observed_peak_um"]], s=125, color=color, edgecolor="#f8fafc", zorder=10)
        ax.text(r["rest_um"], r["observed_peak_um"], f"  {int(r['n'])} {r['line']}\nz={r['z_line']:.6f}", color="#f8fafc", fontsize=9, va="center", ha="left")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Measured observed peak wavelength, micron")
    ax.set_title(f"{VERSION} — data behind the rest-vs-observed redshift line")
    ax.set_xlim(xmin, xmax)
    yvals = list(df["observed_peak_um"]) + orange_y
    if fit and "z_fit" in fit:
        yvals += [(1 + fit["z_fit"]) * x for x in xs]
    pad = (max(yvals) - min(yvals)) * 0.22
    ax.set_ylim(min(yvals) - pad, max(yvals) + pad)
    legend(ax, "upper left")
    fig.tight_layout()
    p = PNG / f"{VERSION}_REST_VS_OBSERVED_ORANGE_GREEN_EXPLAINED.png"
    fig.savefig(p, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return p


def plot_table(df, z_mean, z_std, fit):
    import matplotlib.pyplot as plt
    view = df[["n", "line", "rest_um", "observed_peak_um", "z_line"]].copy()
    for c in ["rest_um", "observed_peak_um", "z_line"]:
        view[c] = view[c].map(lambda x: f"{x:.6f}")
    view.columns = ["#", "line", "rest λ µm", "observed peak λ µm", "z = λobs/λrest - 1"]
    fit_txt = f" | green fit z={fit['z_fit']:.6f}" if fit and "z_fit" in fit else ""
    fig, ax = plt.subplots(figsize=(15.8, 4.7))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — combining the three lines | orange mean z={z_mean:.6f}, std={z_std:.6f}{fit_txt}", color="#f8fafc", fontsize=13.6, pad=12)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
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
    p = PNG / f"{VERSION}_REST_OBSERVED_DATA_TABLE.png"
    fig.savefig(p, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return p


def upload(paths, token):
    import pandas as pd
    rows = []
    for p in paths:
        if not p:
            continue
        sub = "PNG" if p.suffix.lower() == ".png" else "CSV"
        raw, status = github_put_file(token, p, f"{DEST_ROOT}/{sub}/{p.name}")
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
            display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Rest-vs-observed redshift line explained</h2>{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    token = get_token()
    path, status = download_spectrum()
    wave, flux, unit = read_spectrum(path)
    cont, resid, smooth = continuum_residual(wave, flux)
    z_grid, score, z_seed = scan_z(wave, smooth)
    df, z_mean, z_std = measure_triplet(wave, smooth, z_seed)
    fit = shared_gaussian_fit(wave, resid, z_mean)

    import pandas as pd
    triplet_csv = CSV / f"{VERSION}_TRIPLET_REST_OBSERVED_DATA.csv"
    summary_csv = CSV / f"{VERSION}_ORANGE_GREEN_LINE_SUMMARY.csv"
    df.to_csv(triplet_csv, index=False)
    pd.DataFrame([{
        "orange_line_definition": "lambda_observed = (1 + z_mean_from_three_peak_readings) * lambda_rest",
        "green_line_definition": "lambda_observed = (1 + z_shared_gaussian_fit) * lambda_rest",
        "z_seed_line_comb": z_seed,
        "z_mean_orange": z_mean,
        "z_std_three_lines": z_std,
        "z_fit_green": fit.get("z_fit") if fit and "z_fit" in fit else None,
        "mast_product": PRODUCT,
        "download_status": status,
    }]).to_csv(summary_csv, index=False)

    png1 = plot_triplet_zoom(wave, flux, cont, resid, smooth, df, fit, z_mean, unit)
    png2 = plot_rest_observed(df, z_mean, fit)
    png3 = plot_table(df, z_mean, z_std, fit)
    rows = upload([png1, png2, png3, triplet_csv, summary_csv], token)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("MAST product", PRODUCT),
        ("Download status", status),
        ("Data used for rest-vs-observed plot", "three measured peak wavelengths + three known rest wavelengths"),
        ("Orange line", "lambda_obs = (1 + mean z from 1,2,3) * lambda_rest"),
        ("Green line", "lambda_obs = (1 + shared Gaussian fit z) * lambda_rest"),
        ("z seed from line-comb scan", f"{z_seed:.6f}"),
        ("orange mean z", f"{z_mean:.6f}"),
        ("three-line z std", f"{z_std:.6f}"),
        ("green fit z", f"{fit.get('z_fit'):.6f}" if fit and "z_fit" in fit else "not available"),
    ], ["Field", "Value"])

    print("\nPOINTS USED IN REST-VS-OBSERVED PLOT")
    print_table([(int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.observed_peak_um:.6f}", f"{r.z_line:.6f}") for r in df.itertuples()], ["#", "Line", "Rest um", "Observed peak um", "z"])
    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"] or "local only") for r in rows], ["Type", "Status", "File", "HTTP link"])
    display_links(rows)
    print("\nCOMMENTS")
    print("Plot 1 now shows all three labels: 1, 2, and 3.")
    print("The rest-vs-observed plot uses x=known rest wavelength and y=measured observed peak wavelength.")
    print("Orange is the simple average-redshift line; green is the simultaneous shared-z Gaussian fit line.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
