# JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK.py
# Fast JWST redshift-curve dashboard with strict online timeout and embedded fallback.
# Matplotlib only. No AI images. No FITS/image downloads.

from pathlib import Path
from datetime import datetime
import subprocess
import sys
import warnings
import signal
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK"
PROJECT_NAME = "JWST"
ROOT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = ROOT / "PNG"
OUTPUT_CSV = ROOT / "CSV"

Z_MIN = 0.0
Z_MAX = 15.0
BIN_WIDTH_Z = 0.25
SMOOTH_SIGMA_BINS = 1.15
RANDOM_SEED = 66
ONLINE_TIMEOUT_SECONDS = 45
VIZIER_ROW_LIMIT = 2500
MIN_ACCEPTED_ROWS = 150

PNG_CURVE = OUTPUT_PNG / f"{VERSION}.png"
PNG_DISTANCE = OUTPUT_PNG / f"{VERSION}_DISTANCE_AGE_CURVES.png"
PNG_TABLE = OUTPUT_PNG / f"{VERSION}_SUMMARY_TABLE.png"
CSV_REDSHIFTS = OUTPUT_CSV / f"{VERSION}_REDSHIFT_SAMPLE.csv"
CSV_HISTOGRAM = OUTPUT_CSV / f"{VERSION}_REDSHIFT_CURVE.csv"
CSV_DISTANCE = OUTPUT_CSV / f"{VERSION}_COSMOLOGY_CURVE.csv"
CSV_AUDIT = OUTPUT_CSV / f"{VERSION}_AUDIT.csv"

# This fallback is intentionally a compact teaching distribution, not a live catalog.
# It creates a redshift count curve that runs instantly when online catalog retrieval stalls.
FALLBACK_COMPONENTS = [
    {"label": "nearby/low-z calibration galaxies", "mean": 0.8, "sigma": 0.35, "n": 850},
    {"label": "intermediate JWST field galaxies", "mean": 2.0, "sigma": 0.65, "n": 1250},
    {"label": "deep near-IR galaxy population", "mean": 4.0, "sigma": 0.85, "n": 980},
    {"label": "reionization-era candidates", "mean": 7.0, "sigma": 0.75, "n": 360},
    {"label": "very high-z tail", "mean": 10.5, "sigma": 0.80, "n": 95},
    {"label": "frontier candidate tail", "mean": 13.0, "sigma": 0.45, "n": 20},
]

REDSHIFT_PRIORITY_NAMES = [
    "zphot", "z_phot", "zphot_best", "z_best", "zbest", "z_med", "zmedian", "z_median",
    "photoz", "photo_z", "photz", "phot_z", "redshift", "zspec", "z_spec", "specz", "spec_z",
]

BAD_REDSHIFT_NAME_FRAGMENTS = [
    "ra", "dec", "err", "e_", "chi", "flag", "prob", "pdf", "mag", "flux", "sn", "id",
    "_min", "_max", "lower", "upper",
]

VIZIER_TARGETS = [
    "JADES photometric redshift JWST",
    "CEERS photometric redshift JWST",
]


class TimeoutException(Exception):
    pass


def timeout_handler(_signum, _frame):
    raise TimeoutException("online query timeout")


def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def ensure_environment():
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    install_if_missing("astropy")


def print_table(title, rows, headers):
    rows = [[str(x) for x in row] for row in rows]
    headers = [str(x) for x in headers]
    table = [headers] + rows
    widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
    rule = "-" * (sum(widths) + 3 * (len(widths) - 1))
    print(title)
    print(rule)
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print(rule)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
    print()


def table_to_dataframe(table):
    try:
        return table.to_pandas()
    except Exception:
        return pd.DataFrame({name: np.array(table[name]) for name in table.colnames})


def redshift_column_score(name):
    low = str(name).lower().replace("-", "_")
    for bad in BAD_REDSHIFT_NAME_FRAGMENTS:
        if bad in low and low != "redshift":
            return -999
    for i, key in enumerate(REDSHIFT_PRIORITY_NAMES):
        if low == key:
            return 1000 - i
    for i, key in enumerate(REDSHIFT_PRIORITY_NAMES):
        if key in low:
            return 500 - i
    if low in ["z", "z1", "z2"]:
        return 50
    return -999


def find_redshift_columns(df):
    candidates = []
    for col in df.columns:
        score = redshift_column_score(col)
        if score <= 0:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        finite = values[np.isfinite(values)]
        in_range = finite[(finite >= Z_MIN) & (finite <= Z_MAX)]
        if len(in_range) < 20:
            continue
        frac = len(in_range) / max(1, len(finite))
        if frac < 0.35:
            continue
        candidates.append((score, col, len(in_range), frac))
    candidates.sort(reverse=True, key=lambda x: (x[0], x[2]))
    return candidates


def try_fast_online_sample():
    audit_rows = []
    records = []
    try:
        install_if_missing("astroquery")
        from astroquery.vizier import Vizier
    except Exception as exc:
        audit_rows.append({"mode": "online", "target": "astroquery", "status": "install_failed", "rows": 0, "detail": str(exc)[:120]})
        return pd.DataFrame(columns=["redshift", "source", "status"]), pd.DataFrame(audit_rows)

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(ONLINE_TIMEOUT_SECONDS)
    try:
        Vizier.ROW_LIMIT = VIZIER_ROW_LIMIT
        seen = set()
        for target in VIZIER_TARGETS:
            print(f"Fast online search: {target}")
            catalogs = Vizier.find_catalogs(target)
            for catalog_key, _meta in list(catalogs.items())[:2]:
                if catalog_key in seen:
                    continue
                seen.add(catalog_key)
                print(f"  Catalog: {catalog_key}")
                tables = Vizier(columns=["**"], row_limit=VIZIER_ROW_LIMIT).get_catalogs(catalog_key)
                for table_index, table in enumerate(tables[:2]):
                    table_name = getattr(table, "meta", {}).get("name", f"table_{table_index}")
                    df = table_to_dataframe(table)
                    candidates = find_redshift_columns(df)
                    if not candidates:
                        audit_rows.append({"mode": "online", "target": target, "catalog": catalog_key, "table": table_name, "status": "no_z", "rows": len(df)})
                        continue
                    _score, z_col, _n, _frac = candidates[0]
                    z = pd.to_numeric(df[z_col], errors="coerce")
                    z = z[np.isfinite(z)]
                    z = z[(z >= Z_MIN) & (z <= Z_MAX)]
                    if len(z) < 20:
                        continue
                    label = f"VizieR:{catalog_key}:{table_name}:{z_col}"
                    records.append(pd.DataFrame({"redshift": z.astype(float), "source": label, "status": "online"}))
                    audit_rows.append({"mode": "online", "target": target, "catalog": catalog_key, "table": table_name, "status": "accepted", "rows": len(z)})
                    if sum(len(x) for x in records) >= VIZIER_ROW_LIMIT:
                        break
                if sum(len(x) for x in records) >= VIZIER_ROW_LIMIT:
                    break
            if sum(len(x) for x in records) >= VIZIER_ROW_LIMIT:
                break
    except TimeoutException as exc:
        audit_rows.append({"mode": "online", "target": "timeout", "catalog": "", "table": "", "status": "timeout", "rows": 0, "detail": str(exc)})
    except Exception as exc:
        audit_rows.append({"mode": "online", "target": "exception", "catalog": "", "table": "", "status": "failed", "rows": 0, "detail": str(exc)[:160]})
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if records:
        sample = pd.concat(records, ignore_index=True)
        sample = sample.drop_duplicates(subset=["redshift", "source"])
        sample = sample[(sample["redshift"] >= Z_MIN) & (sample["redshift"] <= Z_MAX)]
        return sample.reset_index(drop=True), pd.DataFrame(audit_rows)
    return pd.DataFrame(columns=["redshift", "source", "status"]), pd.DataFrame(audit_rows)


def fallback_sample():
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for comp in FALLBACK_COMPONENTS:
        values = rng.normal(loc=comp["mean"], scale=comp["sigma"], size=comp["n"])
        values = values[(values >= Z_MIN) & (values <= Z_MAX)]
        for z in values:
            rows.append({"redshift": float(z), "source": comp["label"], "status": "embedded_fallback"})
    df = pd.DataFrame(rows).sort_values("redshift").reset_index(drop=True)
    return df


def gaussian_kernel(sigma_bins):
    radius = max(2, int(np.ceil(4 * sigma_bins)))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def make_curve(redshifts):
    edges = np.arange(Z_MIN, Z_MAX + BIN_WIDTH_Z, BIN_WIDTH_Z)
    counts, edges = np.histogram(redshifts, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = np.convolve(counts.astype(float), gaussian_kernel(SMOOTH_SIGMA_BINS), mode="same")
    cumulative = np.cumsum(counts) / max(1, counts.sum())
    return pd.DataFrame({
        "z_center": centers,
        "z_min": edges[:-1],
        "z_max": edges[1:],
        "count": counts,
        "count_smooth": smooth,
        "cumulative_fraction": cumulative,
    })


def cosmology_curve():
    from astropy.cosmology import Planck18 as cosmo
    z = np.linspace(0.0, 15.0, 301)
    return pd.DataFrame({
        "redshift_z": z,
        "lookback_time_Gyr": cosmo.lookback_time(z).value,
        "universe_age_Gyr": cosmo.age(z).value,
        "comoving_distance_Gly": cosmo.comoving_distance(z).to("Glyr").value,
    })


def style_axes(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, alpha=0.26, linewidth=0.55, color="#334155")
    ax.tick_params(colors="#dbeafe")
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def plot_redshift_curve(curve, sample, mode_label):
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 11.0), dpi=155, gridspec_kw={"height_ratios": [3.0, 1.15]})
    ax, ax2 = axes
    style_axes(fig, ax)
    style_axes(fig, ax2)

    ax.plot(curve["z_center"], curve["count_smooth"], linewidth=2.7, color="#38bdf8", label="Smoothed count curve")
    ax.fill_between(curve["z_center"], curve["count_smooth"], alpha=0.22, color="#38bdf8")
    ax.plot(curve["z_center"], curve["count"], linewidth=0.8, alpha=0.42, color="#f8fafc", label="Binned counts")
    for x in [1, 2, 4, 6, 8, 10, 12, 15]:
        ax.axvline(x, alpha=0.18, linewidth=0.75, color="#f8fafc")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel(f"Objects per Δz={BIN_WIDTH_Z:g} bin")
    ax.set_title(f"JWST redshift distribution curve, z = 0 to 15\nmode: {mode_label}")
    ax.legend(framealpha=0.32, fontsize=9, facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc")

    ax2.plot(curve["z_center"], curve["cumulative_fraction"] * 100.0, linewidth=2.2, color="#fb923c")
    ax2.set_xlim(Z_MIN, Z_MAX)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Redshift z")
    ax2.set_ylabel("Cumulative [%]")
    ax2.set_title("Cumulative fraction")

    z = sample["redshift"].to_numpy()
    stats_text = (
        f"N={len(sample):,} | median z={np.median(z):.3f} | "
        f"z>6={(z > 6).sum():,} | z>8={(z > 8).sum():,} | "
        f"z>10={(z > 10).sum():,} | z>12={(z > 12).sum():,}"
    )
    fig.text(0.5, 0.017, stats_text, ha="center", fontsize=10, alpha=0.90, color="#f8fafc")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(PNG_CURVE, dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)


def plot_distance_age(cosmo_df):
    fig, ax = plt.subplots(figsize=(15.0, 8.3), dpi=155)
    style_axes(fig, ax)
    ax.plot(cosmo_df["redshift_z"], cosmo_df["comoving_distance_Gly"], linewidth=2.4, color="#38bdf8", label="Comoving distance [Gly]")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Comoving distance [Gly]")
    ax.set_xlim(0, 15)
    ax.set_ylim(bottom=0)

    axr = ax.twinx()
    axr.set_facecolor("#050712")
    axr.plot(cosmo_df["redshift_z"], cosmo_df["universe_age_Gyr"] * 1000.0, linewidth=2.0, color="#fb923c", label="Universe age [Myr]")
    axr.set_ylabel("Universe age [Myr]", color="#f8fafc")
    axr.tick_params(colors="#dbeafe")
    for spine in axr.spines.values():
        spine.set_color("#94a3b8")
    ax.set_title("JWST redshift context: distance and universe age\nPlanck18 cosmology curve; theoretical context, not observed galaxy locations")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", framealpha=0.32,
              facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc")
    fig.tight_layout()
    fig.savefig(PNG_DISTANCE, dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)


def summary_rows(sample, curve, audit, mode_label):
    z = sample["redshift"].to_numpy()
    peak_row = curve.iloc[int(np.argmax(curve["count_smooth"].to_numpy()))]
    online_rows = int((sample["status"] == "online").sum()) if "status" in sample.columns else 0
    fallback_rows = int((sample["status"] == "embedded_fallback").sum()) if "status" in sample.columns else 0
    timed_out = bool(len(audit) and "timeout" in " ".join(audit.astype(str).to_numpy().ravel()).lower())
    return [
        ("mode", mode_label),
        ("online_timeout_s", ONLINE_TIMEOUT_SECONDS),
        ("online_rows", online_rows),
        ("fallback_rows", fallback_rows),
        ("query_timed_out", timed_out),
        ("objects", f"{len(sample):,}"),
        ("source_groups", sample["source"].nunique()),
        ("z_min", f"{np.min(z):.4f}"),
        ("z_median", f"{np.median(z):.4f}"),
        ("z_mean", f"{np.mean(z):.4f}"),
        ("z_max", f"{np.max(z):.4f}"),
        ("curve_peak_z", f"{peak_row['z_center']:.4f}"),
        ("count_z_gt_6", int((z > 6).sum())),
        ("count_z_gt_8", int((z > 8).sum())),
        ("count_z_gt_10", int((z > 10).sum())),
        ("count_z_gt_12", int((z > 12).sum())),
    ]


def plot_summary_table(rows):
    fig, ax = plt.subplots(figsize=(10.8, 8.2), dpi=155)
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    table = ax.table(cellText=[[str(a), str(b)] for a, b in rows], colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.0)
    table.scale(1.0, 1.32)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#082f49" if r % 2 else "#0f172a")
            cell.get_text().set_color("#e5e7eb")
    ax.set_title("JWST fast redshift dashboard summary\nTimeout-protected online search with embedded fallback",
                 color="#f8fafc", fontsize=13.0, pad=18)
    fig.tight_layout()
    fig.savefig(PNG_TABLE, dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)


def main():
    warnings.filterwarnings("ignore")
    print(f"CODE OUTPUT: {VERSION}")
    print()
    ensure_environment()

    online_sample, audit = try_fast_online_sample()
    if len(online_sample) >= MIN_ACCEPTED_ROWS:
        sample = online_sample.copy()
        mode_label = "timeout-protected online VizieR sample"
    else:
        sample = fallback_sample()
        mode_label = "embedded fallback teaching sample"
        extra = pd.DataFrame([{
            "mode": "fallback",
            "target": "embedded sample",
            "status": "used",
            "rows": len(sample),
            "detail": f"online rows below threshold {MIN_ACCEPTED_ROWS}",
        }])
        audit = pd.concat([audit, extra], ignore_index=True)

    curve = make_curve(sample["redshift"].to_numpy())
    cosmo_df = cosmology_curve()

    sample.to_csv(CSV_REDSHIFTS, index=False)
    curve.to_csv(CSV_HISTOGRAM, index=False)
    cosmo_df.to_csv(CSV_DISTANCE, index=False)
    audit.to_csv(CSV_AUDIT, index=False)

    plot_redshift_curve(curve, sample, mode_label)
    plot_distance_age(cosmo_df)
    rows = summary_rows(sample, curve, audit, mode_label)
    plot_summary_table(rows)

    print_table("JWST FAST REDSHIFT DASHBOARD SUMMARY", rows, ["metric", "value"])
    print_table(
        "OUTPUT SUMMARY",
        [
            ("plot_png", str(PNG_CURVE)),
            ("distance_png", str(PNG_DISTANCE)),
            ("table_png", str(PNG_TABLE)),
            ("redshift_sample_csv", str(CSV_REDSHIFTS)),
            ("curve_csv", str(CSV_HISTOGRAM)),
            ("cosmology_csv", str(CSV_DISTANCE)),
            ("audit_csv", str(CSV_AUDIT)),
        ],
        ["item", "value"],
    )
    print("Online catalog retrieval is timeout-protected. If VizieR stalls, the embedded fallback is used automatically.")
    print("Fallback distribution is for learning/visualization only; it is not a live catalog or volume-corrected density function.")
    print("Cosmology curve is theoretical Planck18 context and should not be interpreted as observed galaxy detections.")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
