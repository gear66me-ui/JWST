# JWST_0031
# Audit: Actual JWST/NIRSpec redshift signature from a public MAST spectrum. Python/matplotlib only. No AI images.

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

VERSION = "JWST_0031"
PROJECT = "JWST NIRSPEC REDSHIFT SIGNATURE"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0031_NIRSPEC_REDSHIFT_SIGNATURE"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PRODUCT = "jw02736-o007_s000009239_nirspec_f170lp-g235m_x1d.fits"
MAST_URI = f"mast:JWST/product/{PRODUCT}"
MAST_DIRECT_URL = f"https://mast.stsci.edu/api/v0.1/Download/file?uri={MAST_URI}"

LINE_DEFS = [
    ("Hβ", 0.486133, 0.55),
    ("[O III] 4959", 0.495891, 0.55),
    ("[O III] 5007", 0.500684, 1.00),
    ("[N II] 6548", 0.654805, 0.45),
    ("Hα", 0.656281, 1.00),
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
        wave_unit = str(ext.header.get("TUNIT1", "um"))
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
    return wave[order], flux_plot[order], wave_unit, plot_unit


def continuum_and_residual(wave, flux):
    import numpy as np
    from scipy.signal import savgol_filter
    from scipy.ndimage import median_filter
    n = len(flux)
    kernel = max(101, int(n * 0.055) // 2 * 2 + 1)
    kernel = min(kernel, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if kernel < 9:
        kernel = 9
    cont = median_filter(flux, size=kernel, mode="nearest")
    sg_window = max(51, int(n * 0.035) // 2 * 2 + 1)
    sg_window = min(sg_window, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if sg_window >= 7:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cont = savgol_filter(cont, sg_window, 2, mode="interp")
    residual = flux - cont
    res_window = max(7, int(n * 0.006) // 2 * 2 + 1)
    res_window = min(res_window, n - 1 if (n - 1) % 2 == 1 else n - 2)
    if res_window >= 7:
        residual_smooth = savgol_filter(residual, res_window, 2, mode="interp")
    else:
        residual_smooth = residual
    return cont, residual, residual_smooth


def robust_norm(values):
    import numpy as np
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad if mad > 0 else np.nanstd(values)
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return (values - med) / scale


def cross_correlation_redshift(wave, residual_smooth):
    import numpy as np
    y = robust_norm(residual_smooth)
    z_grid = np.linspace(1.9, 3.1, 2401)
    scores = []
    contributing = []
    for z in z_grid:
        score = 0.0
        wsum = 0.0
        used = 0
        for label, rest, weight in LINE_DEFS:
            obs = rest * (1.0 + z)
            if wave.min() <= obs <= wave.max():
                val = np.interp(obs, wave, y)
                score += weight * val
                wsum += weight
                used += 1
        scores.append(score / wsum if wsum > 0 else np.nan)
        contributing.append(used)
    scores = np.asarray(scores)
    idx = int(np.nanargmax(scores))
    z_best = float(z_grid[idx])
    if 2 <= idx < len(z_grid) - 2:
        xs = z_grid[idx-2:idx+3]
        ys = scores[idx-2:idx+3]
        coeff = np.polyfit(xs, ys, 2)
        if coeff[0] < 0:
            z_vertex = -coeff[1] / (2 * coeff[0])
            if xs.min() <= z_vertex <= xs.max():
                z_best = float(z_vertex)
    return z_grid, scores, contributing, z_best


def local_peak(wave, residual_smooth, center, half_width=0.020):
    import numpy as np
    mask = (wave >= center - half_width) & (wave <= center + half_width)
    if mask.sum() < 5:
        return np.nan, np.nan
    ww = wave[mask]
    yy = residual_smooth[mask]
    imax = int(np.nanargmax(yy))
    peak_wave = float(ww[imax])
    peak_flux = float(yy[imax])
    j0 = max(0, imax - 3)
    j1 = min(len(ww), imax + 4)
    if j1 - j0 >= 5:
        try:
            coeff = np.polyfit(ww[j0:j1], yy[j0:j1], 2)
            if coeff[0] < 0:
                vertex = -coeff[1] / (2 * coeff[0])
                if ww[j0] <= vertex <= ww[j1-1]:
                    peak_wave = float(vertex)
                    peak_flux = float(np.polyval(coeff, vertex))
        except Exception:
            pass
    return peak_wave, peak_flux


def analyze_lines(wave, residual_smooth, z_best):
    import pandas as pd
    rows = []
    for label, rest, weight in LINE_DEFS:
        predicted = rest * (1.0 + z_best)
        if wave.min() <= predicted <= wave.max():
            peak_wave, peak_flux = local_peak(wave, residual_smooth, predicted, half_width=0.018)
            z_direct = peak_wave / rest - 1.0 if peak_wave == peak_wave else float("nan")
            offset_nm = (peak_wave - predicted) * 1000.0 if peak_wave == peak_wave else float("nan")
            rows.append({
                "line": label,
                "rest_wavelength_um": rest,
                "predicted_observed_um": predicted,
                "local_peak_um": peak_wave,
                "z_from_local_peak": z_direct,
                "offset_from_template_nm": offset_nm,
                "relative_weight": weight,
                "local_peak_residual_flux": peak_flux,
            })
    return pd.DataFrame(rows)


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
    leg = ax.legend(loc=loc, fontsize=8.4, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_full_spectrum(wave, flux, cont, residual_smooth, line_df, z_best, plot_unit):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16.4, 8.8))
    dark_axis(fig, ax)
    ax.plot(wave, flux, linewidth=1.05, color="#93c5fd", alpha=0.95, label="Observed NIRSpec spectrum")
    ax.plot(wave, cont, linewidth=1.45, color="#facc15", alpha=0.85, label="Median continuum estimate")
    y0 = min(flux) - 0.18 * (max(flux) - min(flux))
    scale = 0.22 * (max(flux) - min(flux)) / max(1e-12, max(abs(residual_smooth)))
    ax.plot(wave, y0 + residual_smooth * scale, linewidth=1.0, color="#fb7185", alpha=0.90, label="Continuum-subtracted residual, scaled")
    ymax = max(flux)
    for _, row in line_df.iterrows():
        x = row["predicted_observed_um"]
        ax.axvline(x, color="#22c55e", linewidth=0.85, alpha=0.72)
        ax.text(x, ymax, row["line"], color="#bbf7d0", fontsize=8.1, rotation=90, va="top", ha="right")
    ax.set_xlim(wave.min(), wave.max())
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, {plot_unit}")
    ax.set_title(f"{VERSION} — Actual JWST/NIRSpec spectrum with redshifted emission-line markers, z about {z_best:.5f}")
    style_legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_FULL_NIRSPEC_SPECTRUM_WITH_LINES.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_halpha_zoom(wave, flux, cont, residual_smooth, line_df, z_best, plot_unit):
    import matplotlib.pyplot as plt
    ha = line_df[line_df["line"] == "Hα"].iloc[0]
    center = ha["predicted_observed_um"]
    x0, x1 = center - 0.075, center + 0.075
    mask = (wave >= x0) & (wave <= x1)
    fig, ax = plt.subplots(figsize=(15.8, 8.2))
    dark_axis(fig, ax)
    ax.plot(wave[mask], flux[mask], linewidth=1.15, color="#93c5fd", label="Observed flux")
    ax.plot(wave[mask], cont[mask], linewidth=1.4, color="#facc15", label="Continuum estimate")
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax2.plot(wave[mask], residual_smooth[mask], linewidth=1.3, color="#fb7185", label="Continuum-subtracted residual")
    ax2.tick_params(colors="#fecdd3", labelsize=9)
    ax2.yaxis.label.set_color("#fecdd3")
    for spine in ax2.spines.values():
        spine.set_color("#94a3b8")
    zoom_lines = line_df[line_df["line"].isin(["[N II] 6548", "Hα", "[N II] 6583", "[S II] 6716", "[S II] 6731"])]
    for _, row in zoom_lines.iterrows():
        x = row["predicted_observed_um"]
        ax.axvline(x, color="#22c55e", linewidth=1.0, alpha=0.82)
        ax.text(x, ax.get_ylim()[1], row["line"], color="#bbf7d0", fontsize=9, rotation=90, va="top", ha="right")
        if row["local_peak_um"] == row["local_peak_um"]:
            ax2.scatter([row["local_peak_um"]], [row["local_peak_residual_flux"]], s=42, color="#f97316", edgecolor="#f8fafc", zorder=8)
    ax.set_xlim(x0, x1)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Flux, {plot_unit}")
    ax2.set_ylabel(f"Residual flux, {plot_unit}")
    ax.set_title(f"H-alpha / [N II] complex — direct curve-reading redshift near z about {z_best:.5f}")
    style_legend(ax, "upper left")
    leg2 = ax2.legend(loc="upper right", fontsize=8.4, facecolor="#020617", edgecolor="#475569")
    for t in leg2.get_texts():
        t.set_color("#f8fafc")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HALPHA_NII_ZOOM_DIRECT_REDSHIFT.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_redshift_score(z_grid, scores, z_best):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14.8, 7.6))
    dark_axis(fig, ax)
    ax.plot(z_grid, scores, linewidth=1.45, color="#67e8f9", label="Line-comb redshift score")
    ax.axvline(z_best, color="#fb7185", linewidth=1.55, linestyle="--", label=f"Best z = {z_best:.5f}")
    ax.set_xlabel("Trial redshift z")
    ax.set_ylabel("Weighted normalized line score")
    ax.set_title("Automated redshift scan from actual JWST spectrum")
    style_legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_REDSHIFT_SCORE_CURVE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def make_table_png(line_df, z_best, z_ha):
    import matplotlib.pyplot as plt
    view = line_df.copy()
    view = view[["line", "rest_wavelength_um", "predicted_observed_um", "local_peak_um", "z_from_local_peak", "offset_from_template_nm"]]
    for col in ["rest_wavelength_um", "predicted_observed_um", "local_peak_um", "z_from_local_peak", "offset_from_template_nm"]:
        view[col] = view[col].map(lambda x: "" if x != x else f"{x:.6f}")
    view.columns = ["line", "rest micron", "predicted obs micron", "local peak micron", "z from peak", "offset nm"]
    fig, ax = plt.subplots(figsize=(16.2, 5.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — Redshift line table | scan z={z_best:.5f}, H-alpha direct z={z_ha:.5f}", color="#f8fafc", fontsize=15, pad=14)
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
            if str(view.iloc[r-1, 0]) == "Hα":
                cell.set_facecolor("#450a0a")
                cell.get_text().set_color("#fecaca")
    fig.tight_layout()
    path = PNG / f"{VERSION}_REDSHIFT_LINE_TABLE.png"
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
            display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Images available here</h2><p>Actual JWST/NIRSpec redshift plots. Click a preview to open the PNG.</p><ul>{links}</ul><hr style="border-color:#334155">{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    token = get_token(optional=True)
    path, download_status = download_spectrum()
    wave, flux, wave_unit, plot_unit = read_spectrum(path)
    cont, residual, residual_smooth = continuum_and_residual(wave, flux)
    z_grid, scores, contributing, z_best = cross_correlation_redshift(wave, residual_smooth)
    line_df = analyze_lines(wave, residual_smooth, z_best)
    ha = line_df[line_df["line"] == "Hα"].iloc[0]
    z_ha = float(ha["z_from_local_peak"])

    import pandas as pd
    spectrum_csv = CSV / f"{VERSION}_OBSERVED_NIRSPEC_SPECTRUM.csv"
    score_csv = CSV / f"{VERSION}_REDSHIFT_SCORE_CURVE.csv"
    lines_csv = CSV / f"{VERSION}_LINE_REDSHIFT_MEASUREMENTS.csv"
    pd.DataFrame({
        "wavelength_um": wave,
        f"flux_{plot_unit}": flux,
        f"continuum_{plot_unit}": cont,
        f"residual_{plot_unit}": residual,
        f"residual_smooth_{plot_unit}": residual_smooth,
    }).to_csv(spectrum_csv, index=False)
    pd.DataFrame({"z": z_grid, "score": scores, "contributing_lines": contributing}).to_csv(score_csv, index=False)
    line_df.to_csv(lines_csv, index=False)

    png1 = plot_full_spectrum(wave, flux, cont, residual_smooth, line_df, z_best, plot_unit)
    png2 = plot_halpha_zoom(wave, flux, cont, residual_smooth, line_df, z_best, plot_unit)
    png3 = plot_redshift_score(z_grid, scores, z_best)
    png4 = make_table_png(line_df, z_best, z_ha)
    uploaded = upload_outputs([png1, png2, png3, png4, spectrum_csv, score_csv, lines_csv], token)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Observed source", "JWST/NIRSpec 1D spectrum, MAST public product"),
        ("MAST product", PRODUCT),
        ("Download status", download_status),
        ("Wavelength range micron", f"{wave.min():.6f} to {wave.max():.6f}"),
        ("Flux unit plotted", plot_unit),
        ("Line-comb best z", f"{z_best:.6f}"),
        ("H-alpha direct curve-read z", f"{z_ha:.6f}"),
        ("H-alpha local peak micron", f"{ha['local_peak_um']:.6f}"),
        ("H-alpha rest micron", f"{ha['rest_wavelength_um']:.6f}"),
    ], ["Field", "Value"])

    print("\nLINE MEASUREMENTS")
    rows = []
    for _, r in line_df.iterrows():
        rows.append((r["line"], f"{r['rest_wavelength_um']:.6f}", f"{r['predicted_observed_um']:.6f}", f"{r['local_peak_um']:.6f}", f"{r['z_from_local_peak']:.6f}", f"{r['offset_from_template_nm']:.3f}"))
    print_table(rows, ["Line", "Rest micron", "Pred obs micron", "Peak micron", "z_peak", "Offset nm"])

    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"] or "local only") for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)

    print("\nCOMMENTS")
    print("This downloads an actual public 1D JWST/NIRSpec spectrum from MAST, not an image product.")
    print("Redshift is estimated two ways: weighted emission-line scan and direct H-alpha peak reading from the curve.")
    print("The H-alpha/[N II] complex is blended, so the direct H-alpha value is an educational curve-read, not a publication-grade fit.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
