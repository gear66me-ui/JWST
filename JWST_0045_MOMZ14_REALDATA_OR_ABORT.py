# JWST_0045
# Real-data-only MoM-z14 workflow.
# No AI images. No synthetic spectra. No fake H-alpha/[N II] at z=14.44.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib, warnings

VERSION = "JWST_0045"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA"

PROGRAM_ID = "5224"
TARGET_Z = 14.44
TARGET_NAME = "MoM-z14"

# H-alpha + [N II] triplet requested by the user.
# At z=14.44 these lie near 10.1 micron, outside JWST/NIRSpec's 0.6-5.3 micron range.
OPTICAL_TRIPLET = [
    (1, "[N II] 6548", 0.654805),
    (2, "H-alpha 6563", 0.656281),
    (3, "[N II] 6583", 0.658345),
]

# Real z~14 NIRSpec spectra are rest-UV, not H-alpha/[N II].
# These are common rest-UV reference positions used only to place windows on REAL spectrum samples.
# The spectrum itself is raw downloaded MAST data.
REST_UV_REFERENCE = [
    (1, "N IV] 1486", 0.148600),
    (2, "He II 1640", 0.164000),
    (3, "C III] 1908", 0.190800),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy", "astroquery"]:
        need(pkg)
    for p in [PNG, CSV, DATA]:
        p.mkdir(parents=True, exist_ok=True)


def find_public_x1d_products():
    import pandas as pd
    from astroquery.mast import Observations
    obs = Observations.query_criteria(obs_collection="JWST", proposal_id=PROGRAM_ID)
    obs_df = obs.to_pandas() if hasattr(obs, "to_pandas") else pd.DataFrame(obs)
    inv_path = CSV / f"{VERSION}_MAST_PROGRAM_{PROGRAM_ID}_OBSERVATION_INVENTORY.csv"
    obs_df.to_csv(inv_path, index=False)
    if len(obs) == 0:
        return [], inv_path, None
    products = Observations.get_product_list(obs)
    prod_df = products.to_pandas() if hasattr(products, "to_pandas") else pd.DataFrame(products)
    prod_path = CSV / f"{VERSION}_MAST_PROGRAM_{PROGRAM_ID}_PRODUCT_INVENTORY.csv"
    prod_df.to_csv(prod_path, index=False)
    if prod_df.empty:
        return [], inv_path, prod_path

    cols = {c.lower(): c for c in prod_df.columns}
    fn_col = cols.get("productfilename") or cols.get("productFilename".lower())
    group_col = cols.get("productsubgroupdescription")
    inst_col = cols.get("instrument_name") or cols.get("instrument")

    mask = pd.Series(True, index=prod_df.index)
    if fn_col:
        fn = prod_df[fn_col].astype(str).str.lower()
        mask &= fn.str.contains("x1d") & fn.str.endswith(".fits")
        mask &= fn.str.contains("nirspec") | fn.str.contains("nrs") | fn.str.contains("jw")
    if group_col:
        mask |= prod_df[group_col].astype(str).str.upper().eq("X1D")
    if inst_col:
        mask &= prod_df[inst_col].astype(str).str.upper().str.contains("NIRSPEC|NRS|JWST", regex=True, na=False) | mask

    cand = prod_df[mask].copy()
    if cand.empty and fn_col:
        cand = prod_df[prod_df[fn_col].astype(str).str.lower().str.endswith(".fits")].copy()
    cand_path = CSV / f"{VERSION}_MAST_PROGRAM_{PROGRAM_ID}_X1D_CANDIDATES.csv"
    cand.to_csv(cand_path, index=False)
    return cand.to_dict("records"), inv_path, cand_path


def download_candidate(record):
    from astroquery.mast import Observations
    uri = record.get("dataURI") or record.get("dataUri") or record.get("uri")
    filename = record.get("productFilename") or record.get("productfilename") or "mast_product.fits"
    local = DATA / str(filename)
    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    if not uri:
        raise RuntimeError("Candidate lacks a dataURI; cannot download exact MAST product.")
    status = Observations.download_file(uri, local_path=str(local))
    if not local.exists() or local.stat().st_size < 100000:
        raise RuntimeError(f"Download failed or product too small for {filename}: {status}")
    return local, "downloaded-from-mast"


def read_x1d(path):
    import numpy as np
    from astropy.io import fits
    waves, fluxes = [], []
    unit = "native"
    hdu_names = []
    with fits.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            names = list(getattr(data, "names", []) or []) if data is not None else []
            if "WAVELENGTH" in names and "FLUX" in names:
                hdu_names.append(f"{i}:{getattr(hdu, 'name', '')}")
                unit = str(hdu.header.get("TUNIT2", unit))
                wcol = data["WAVELENGTH"]
                fcol = data["FLUX"]
                try:
                    for w, f in zip(wcol, fcol):
                        waves.append(np.ravel(np.asarray(w, dtype=float)))
                        fluxes.append(np.ravel(np.asarray(f, dtype=float)))
                except TypeError:
                    waves.append(np.ravel(np.asarray(wcol, dtype=float)))
                    fluxes.append(np.ravel(np.asarray(fcol, dtype=float)))
    if not waves:
        raise RuntimeError(f"No WAVELENGTH/FLUX table found in {path.name}")
    wave = np.concatenate(waves)
    flux = np.concatenate(fluxes)
    m = np.isfinite(wave) & np.isfinite(flux)
    wave, flux = wave[m], flux[m]
    order = np.argsort(wave)
    return wave[order], flux[order], unit, "; ".join(hdu_names)


def choose_product(records):
    attempts = []
    for rec in records[:25]:
        try:
            path, status = download_candidate(rec)
            wave, flux, unit, hdu_info = read_x1d(path)
            if len(wave) > 30:
                attempts.append((path, status, wave, flux, unit, hdu_info, rec))
        except Exception as exc:
            warnings.warn(str(exc))
    if not attempts:
        raise RuntimeError("No readable public x1d spectrum found for JWST program 5224. Inventory CSVs were saved.")

    # Prefer spectra covering the three rest-UV reference lines at z=14.44.
    expected = [rest * (1 + TARGET_Z) for _, _, rest in REST_UV_REFERENCE]
    for item in attempts:
        wave = item[2]
        if min(expected) >= wave.min() and max(expected) <= wave.max():
            return item
    return attempts[0]


def peak_in_window(wave, flux, center, half_width):
    import numpy as np
    m = (wave >= center - half_width) & (wave <= center + half_width)
    if m.sum() == 0:
        return float("nan"), float("nan")
    x, y = wave[m], flux[m]
    i = int(np.nanargmax(y))
    return float(x[i]), float(y[i])


def measure_lines(wave, flux):
    import pandas as pd
    rows = []
    for n, label, rest in REST_UV_REFERENCE:
        expected = rest * (1 + TARGET_Z)
        peak_w, peak_f = peak_in_window(wave, flux, expected, 0.050)
        rows.append({
            "n": n,
            "line": label,
            "line_set": "real NIRSpec-accessible rest-UV reference",
            "rest_um": rest,
            "expected_from_z14p44_um": expected,
            "raw_peak_sample_um": peak_w,
            "raw_peak_flux_native": peak_f,
            "slope_m_raw_peak_over_rest": peak_w / rest if peak_w == peak_w else float("nan"),
            "z_from_raw_peak": peak_w / rest - 1.0 if peak_w == peak_w else float("nan"),
        })
    return pd.DataFrame(rows)


def optical_triplet_table():
    import pandas as pd
    rows = []
    for n, label, rest in OPTICAL_TRIPLET:
        obs = rest * (1 + TARGET_Z)
        rows.append({
            "n": n,
            "line": label,
            "line_set": "H-alpha plus [N II] requested optical triplet",
            "rest_um": rest,
            "observed_at_z14p44_um": obs,
            "status": "NOT PLOTTED AS SPECTRUM: requires real data near 10.1 um; NIRSpec PRISM data are 0.6-5.3 um",
        })
    return pd.DataFrame(rows)


def freq_thz_from_um(x):
    return 299.792458 / x


def um_from_freq_thz(x):
    return 299.792458 / x


def paper_axis(ax):
    fig = ax.figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="0.88", linewidth=0.45)
    ax.tick_params(colors="0.1", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("0.25")
        spine.set_linewidth(0.65)


def plot_line_window(wave, flux, unit, row):
    import matplotlib.pyplot as plt
    center = row.expected_from_z14p44_um
    half = 0.105
    m = (wave >= center - half) & (wave <= center + half)
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    paper_axis(ax)
    ax.plot(wave[m], flux[m], color="black", linewidth=0.55, label="raw JWST spectrum samples")
    ax.axvline(row.expected_from_z14p44_um, color="0.55", linestyle=":", linewidth=0.85, label="expected at z=14.44")
    ax.axvline(row.raw_peak_sample_um, color="black", linestyle="--", linewidth=0.9, label="raw local peak sample")
    ax.scatter([row.raw_peak_sample_um], [row.raw_peak_flux_native], s=34, color="black", zorder=5)
    ax.text(row.raw_peak_sample_um, row.raw_peak_flux_native, f"  {int(row.n)}", fontsize=13, fontweight="bold", va="bottom")
    ax.text(0.02, 0.96, f"{int(row.n)}  {row.line}\nrest λ = {row.rest_um:.6f} µm\nraw peak λ = {row.raw_peak_sample_um:.6f} µm\nz(raw peak) = {row.z_from_raw_peak:.6f}", transform=ax.transAxes, ha="left", va="top", fontsize=9.2, bbox=dict(facecolor="white", edgecolor="0.45", boxstyle="round,pad=0.35"))
    top = ax.secondary_xaxis("top", functions=(freq_thz_from_um, um_from_freq_thz))
    top.set_xlabel("Frequency, THz")
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_ylabel(f"Raw flux, FITS unit: {unit}")
    ax.set_title(f"{VERSION} — real JWST data around {row.line}")
    ax.legend(loc="best", fontsize=8.2, frameon=True, edgecolor="0.55")
    fig.tight_layout()
    safe = row.line.replace("[", "").replace("]", "").replace(" ", "_").replace("-", "_")
    path = PNG / f"{VERSION}_PLOT_{int(row.n)}_{safe}_REAL_RAW_WINDOW.png"
    fig.savefig(path, dpi=280, facecolor="white")
    plt.show()
    return path


def plot_slopes(df, avg_slope, avg_z):
    import numpy as np, matplotlib.pyplot as plt
    xs = np.linspace(df["rest_um"].min() - 0.006, df["rest_um"].max() + 0.006, 260)
    fig, ax = plt.subplots(figsize=(12.8, 6.8))
    paper_axis(ax)
    styles = ["-", "--", ":"]
    for style, row in zip(styles, df.itertuples()):
        m = row.slope_m_raw_peak_over_rest
        ax.plot(xs, m * xs, color="0.12", linewidth=0.70, linestyle=style, label=f"{int(row.n)} {row.line}: m={m:.6f}")
        ax.scatter([row.rest_um], [row.raw_peak_sample_um], s=35, color="black", zorder=6)
        ax.text(row.rest_um, row.raw_peak_sample_um, f" {int(row.n)}", fontsize=10, fontweight="bold", va="center")
    ax.plot(xs, avg_slope * xs, color="darkorange", linewidth=1.25, label=f"average: m={avg_slope:.6f}, z={avg_z:.6f}")
    ax.set_xlabel("Rest-frame wavelength, micron")
    ax.set_ylabel("Observed raw peak wavelength, micron")
    ax.set_title(f"{VERSION} — three raw spectral slopes plus orange average")
    ax.legend(loc="upper left", fontsize=8.1, frameon=True, facecolor="white", edgecolor="0.55")
    fig.tight_layout()
    path = PNG / f"{VERSION}_PLOT_4_REAL_RAW_SLOPES_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor="white")
    plt.show()
    return path


def plot_summary_table(df, optical, avg_rest, avg_obs, avg_slope, avg_z):
    import matplotlib.pyplot as plt
    rows = []
    for r in df.itertuples():
        rows.append([int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}", f"{r.slope_m_raw_peak_over_rest:.6f}", f"{r.z_from_raw_peak:.6f}"])
    rows.append(["SUM", "SUM", f"{df['rest_um'].sum():.6f}", f"{df['raw_peak_sample_um'].sum():.6f}", "", ""])
    rows.append(["AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"])
    fig, ax = plt.subplots(figsize=(14.0, 5.0))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(f"{VERSION} — real-data line measurements; average at bottom", fontsize=12.5, pad=10)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "raw peak λ µm", "slope m", "z"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("0.55")
        cell.set_linewidth(0.45)
        if r == 0:
            cell.set_facecolor("0.90")
            cell.get_text().set_fontweight("bold")
        elif r >= 4:
            cell.set_facecolor("#fff3e0")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("white")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SUMMARY_TABLE_REAL_DATA.png"
    fig.savefig(path, dpi=260, facecolor="white")
    plt.show()
    return path


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for r in rows:
        widths = [max(widths[i], len(str(r[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for r in rows:
        print(" | ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    records, obs_inv, prod_inv = find_public_x1d_products()
    if not records:
        print(f"CODE OUTPUT: {VERSION}\n")
        print("No public JWST x1d products found for program 5224 by MAST query.")
        print(f"Observation inventory: {obs_inv}")
        if prod_inv:
            print(f"Product inventory: {prod_inv}")
        print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
        print(f"# {VERSION}")
        return

    path, dl_status, wave, flux, unit, hdu_info, rec = choose_product(records)
    measured = measure_lines(wave, flux)
    optical = optical_triplet_table()

    avg_rest = float(measured["rest_um"].mean())
    avg_obs = float(measured["raw_peak_sample_um"].mean())
    avg_slope = avg_obs / avg_rest
    avg_z = avg_slope - 1.0

    raw_csv = CSV / f"{VERSION}_REAL_RAW_SPECTRUM.csv"
    measured_csv = CSV / f"{VERSION}_REAL_LINE_MEASUREMENTS.csv"
    optical_csv = CSV / f"{VERSION}_HALPHA_NII_OUT_OF_BAND_AT_Z14P44.csv"
    summary_csv = CSV / f"{VERSION}_SUMMARY.csv"

    import pandas as pd
    pd.DataFrame({"wavelength_um_raw": wave, f"flux_raw_{unit}": flux, "frequency_thz": freq_thz_from_um(wave)}).to_csv(raw_csv, index=False)
    measured.to_csv(measured_csv, index=False)
    optical.to_csv(optical_csv, index=False)
    pd.DataFrame([{
        "target": TARGET_NAME,
        "z_reference": TARGET_Z,
        "mast_program": PROGRAM_ID,
        "downloaded_product": path.name,
        "download_status": dl_status,
        "hdu_info": hdu_info,
        "data_rule": "real MAST WAVELENGTH vs FLUX only; no smoothing, no synthetic spectrum",
        "average_rest_um": avg_rest,
        "average_raw_peak_observed_um": avg_obs,
        "average_slope": avg_slope,
        "average_z": avg_z,
        "h_alpha_nii_status": "not plotted as spectrum because at z=14.44 the triplet is around 10.1 um; this script does not generate fake data",
    }]).to_csv(summary_csv, index=False)

    plot_paths = []
    for row in measured.itertuples():
        plot_paths.append(plot_line_window(wave, flux, unit, row))
    slope_path = plot_slopes(measured, avg_slope, avg_z)
    table_path = plot_summary_table(measured, optical, avg_rest, avg_obs, avg_slope, avg_z)

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Target", TARGET_NAME),
        ("Reference z", f"{TARGET_Z:.6f}"),
        ("MAST program", PROGRAM_ID),
        ("Product", path.name),
        ("Data rule", "REAL MAST WAVELENGTH vs FLUX only"),
        ("No smoothing", "TRUE"),
        ("No synthetic spectrum", "TRUE"),
        ("H-alpha/[N II] at z=14.44", "around 10.1 um; not faked; saved as out-of-band CSV"),
        ("Average slope", f"{avg_slope:.6f}"),
        ("Average z", f"{avg_z:.6f}"),
        ("Raw spectrum CSV", str(raw_csv)),
        ("Measurements CSV", str(measured_csv)),
        ("Out-of-band optical triplet CSV", str(optical_csv)),
        ("Summary table PNG", str(table_path)),
    ], ["Field", "Value"])

    print("\nFOUR PLOTS")
    print_table([(1, measured.iloc[0]['line'], str(plot_paths[0])), (2, measured.iloc[1]['line'], str(plot_paths[1])), (3, measured.iloc[2]['line'], str(plot_paths[2])), (4, "three slopes plus average", str(slope_path))], ["#", "Plot", "Path"])

    print("\nTHREE MEASUREMENTS AND AVERAGE")
    rows = []
    for r in measured.itertuples():
        rows.append((int(r.n), r.line, f"{r.rest_um:.6f}", f"{r.raw_peak_sample_um:.6f}", f"{r.slope_m_raw_peak_over_rest:.6f}", f"{r.z_from_raw_peak:.6f}"))
    rows.append(("SUM", "SUM", f"{measured['rest_um'].sum():.6f}", f"{measured['raw_peak_sample_um'].sum():.6f}", "", ""))
    rows.append(("AVG", "AVERAGE", f"{avg_rest:.6f}", f"{avg_obs:.6f}", f"{avg_slope:.6f}", f"{avg_z:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Raw obs um", "Slope m", "z"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
