# JWST_0015_ONLINE_REDSHIFT_CURVE.py
# JWST public-catalog online redshift curve builder.
# Matplotlib only. No AI images. No FITS/image downloads.

from pathlib import Path
from datetime import datetime
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VERSION = "JWST_0015_ONLINE_REDSHIFT_CURVE"
PROJECT_NAME = "JWST"
ROOT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = ROOT / "PNG"
OUTPUT_CSV = ROOT / "CSV"

Z_MIN = 0.0
Z_MAX = 15.0
TARGET_MAX_GALAXIES = 30000
VIZIER_ROW_LIMIT_PER_TABLE = 25000
MAX_CATALOGS_PER_TERM = 6
MAX_TABLES_PER_CATALOG = 6
BIN_WIDTH_Z = 0.25
SMOOTH_SIGMA_BINS = 1.4
RANDOM_SEED = 66

PNG_CURVE = OUTPUT_PNG / f"{VERSION}.png"
PNG_TABLE = OUTPUT_PNG / f"{VERSION}_SUMMARY_TABLE.png"
CSV_REDSHIFTS = OUTPUT_CSV / f"{VERSION}_REDSHIFT_SAMPLE.csv"
CSV_HISTOGRAM = OUTPUT_CSV / f"{VERSION}_REDSHIFT_CURVE.csv"
CSV_AUDIT = OUTPUT_CSV / f"{VERSION}_ONLINE_CATALOG_AUDIT.csv"

SEARCH_TERMS = [
    "ASTRODEEP-JWST photometric redshift",
    "ASTRODEEP JWST",
    "CEERS photometric redshift JWST",
    "JADES photometric redshift JWST",
    "JWST photometric redshift galaxies",
]

REDSHIFT_PRIORITY_NAMES = [
    "zphot", "z_phot", "zphot_best", "z_best", "zbest", "z_med", "zmedian", "z_median",
    "photoz", "photo_z", "photz", "phot_z", "redshift", "zspec", "z_spec", "specz", "spec_z",
]

BAD_REDSHIFT_NAME_FRAGMENTS = [
    "ra", "dec", "err", "e_", "chi", "flag", "prob", "pdf", "mag", "flux", "sn", "id",
    "_min", "_max", "lower", "upper",
]


def install_if_missing(package, import_name=None):
    import_name = import_name or package
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


def ensure_environment():
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    install_if_missing("astroquery")
    install_if_missing("astropy")
    install_if_missing("scipy")


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


def extract_redshifts_from_vizier():
    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = VIZIER_ROW_LIMIT_PER_TABLE
    all_records = []
    audit_rows = []
    catalog_keys_seen = set()

    for term in SEARCH_TERMS:
        print(f"Searching VizieR: {term}")
        try:
            catalogs = Vizier.find_catalogs(term)
        except Exception as exc:
            audit_rows.append({
                "search_term": term,
                "catalog": "SEARCH_FAILED",
                "table": "",
                "column": "",
                "rows": 0,
                "status": str(exc)[:160],
            })
            continue

        catalog_items = list(catalogs.items())[:MAX_CATALOGS_PER_TERM]
        if not catalog_items:
            audit_rows.append({
                "search_term": term,
                "catalog": "NO_MATCH",
                "table": "",
                "column": "",
                "rows": 0,
                "status": "no VizieR catalogs matched",
            })
            continue

        for catalog_key, _catalog_meta in catalog_items:
            if catalog_key in catalog_keys_seen:
                continue
            catalog_keys_seen.add(catalog_key)
            print(f"  Catalog: {catalog_key}")

            try:
                tables = Vizier(columns=["**"], row_limit=VIZIER_ROW_LIMIT_PER_TABLE).get_catalogs(catalog_key)
            except Exception as exc:
                audit_rows.append({
                    "search_term": term,
                    "catalog": catalog_key,
                    "table": "FETCH_FAILED",
                    "column": "",
                    "rows": 0,
                    "status": str(exc)[:160],
                })
                continue

            for table_index, table in enumerate(tables[:MAX_TABLES_PER_CATALOG]):
                table_name = getattr(table, "meta", {}).get("name", f"table_{table_index}")
                try:
                    df = table_to_dataframe(table)
                except Exception as exc:
                    audit_rows.append({
                        "search_term": term,
                        "catalog": catalog_key,
                        "table": table_name,
                        "column": "",
                        "rows": 0,
                        "status": f"dataframe_failed: {str(exc)[:120]}",
                    })
                    continue

                candidates = find_redshift_columns(df)
                if not candidates:
                    audit_rows.append({
                        "search_term": term,
                        "catalog": catalog_key,
                        "table": table_name,
                        "column": "NO_Z_COLUMN",
                        "rows": len(df),
                        "status": "no usable redshift column",
                    })
                    continue

                _score, z_col, _n_in_range, _frac = candidates[0]
                z = pd.to_numeric(df[z_col], errors="coerce")
                z = z[np.isfinite(z)]
                z = z[(z >= Z_MIN) & (z <= Z_MAX)]
                if len(z) == 0:
                    continue

                source_label = f"VizieR:{catalog_key}:{table_name}:{z_col}"
                chunk = pd.DataFrame({"redshift": z.astype(float).to_numpy(), "source": source_label})
                all_records.append(chunk)
                audit_rows.append({
                    "search_term": term,
                    "catalog": catalog_key,
                    "table": table_name,
                    "column": z_col,
                    "rows": len(chunk),
                    "status": "accepted",
                })
                print(f"    accepted {len(chunk):,} redshifts from {z_col}")

                total = sum(len(x) for x in all_records)
                if total >= TARGET_MAX_GALAXIES:
                    audit = pd.DataFrame(audit_rows)
                    return pd.concat(all_records, ignore_index=True), audit

    audit = pd.DataFrame(audit_rows)
    if all_records:
        return pd.concat(all_records, ignore_index=True), audit
    return pd.DataFrame(columns=["redshift", "source"]), audit


def dedupe_and_sample(df):
    clean = df.copy()
    clean["redshift"] = pd.to_numeric(clean["redshift"], errors="coerce")
    clean = clean[np.isfinite(clean["redshift"])]
    clean = clean[(clean["redshift"] >= Z_MIN) & (clean["redshift"] <= Z_MAX)]
    clean = clean.drop_duplicates(subset=["redshift", "source"])
    if len(clean) > TARGET_MAX_GALAXIES:
        clean = clean.sample(TARGET_MAX_GALAXIES, random_state=RANDOM_SEED).sort_values("redshift")
    return clean.reset_index(drop=True)


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
    density = counts / max(1, len(redshifts)) / BIN_WIDTH_Z
    density_smooth = smooth / max(1, len(redshifts)) / BIN_WIDTH_Z
    cumulative = np.cumsum(counts) / max(1, counts.sum())
    return pd.DataFrame({
        "z_center": centers,
        "z_min": edges[:-1],
        "z_max": edges[1:],
        "count": counts,
        "count_smooth": smooth,
        "density_per_z": density,
        "density_smooth_per_z": density_smooth,
        "cumulative_fraction": cumulative,
    })


def plot_curve(curve, sample):
    fig, axes = plt.subplots(2, 1, figsize=(15.5, 11.0), dpi=155, gridspec_kw={"height_ratios": [3.0, 1.15]})
    fig.patch.set_facecolor("#050712")

    ax = axes[0]
    ax2 = axes[1]
    for panel in [ax, ax2]:
        panel.set_facecolor("#050712")
        panel.grid(True, alpha=0.24, linewidth=0.55, color="#334155")
        panel.tick_params(colors="#dbeafe")
        for spine in panel.spines.values():
            spine.set_color("#94a3b8")

    ax.plot(curve["z_center"], curve["count_smooth"], linewidth=2.7, color="#38bdf8", label="Smoothed galaxy count curve")
    ax.fill_between(curve["z_center"], curve["count_smooth"], alpha=0.22, color="#38bdf8")
    ax.plot(curve["z_center"], curve["count"], linewidth=0.8, alpha=0.35, color="#f8fafc", label="Binned counts")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Redshift z", color="#f8fafc")
    ax.set_ylabel(f"Galaxies per Δz={BIN_WIDTH_Z:g} bin", color="#f8fafc")
    ax.set_title("JWST public-catalog galaxy redshift distribution, z = 0 to 15", color="#f8fafc")
    ax.legend(framealpha=0.32, fontsize=9, facecolor="#020617", edgecolor="#475569", labelcolor="#f8fafc")

    for x in [1, 2, 4, 6, 8, 10, 12, 15]:
        ax.axvline(x, alpha=0.18, linewidth=0.75, color="#f8fafc")

    ax2.plot(curve["z_center"], curve["cumulative_fraction"] * 100.0, linewidth=2.2, color="#fb923c")
    ax2.set_xlim(Z_MIN, Z_MAX)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Redshift z", color="#f8fafc")
    ax2.set_ylabel("Cumulative [%]", color="#f8fafc")
    ax2.set_title("Cumulative fraction", color="#f8fafc")

    n = len(sample)
    q = np.quantile(sample["redshift"], [0.05, 0.25, 0.5, 0.75, 0.95]) if n else [np.nan] * 5
    counts = {
        "z>6": int((sample["redshift"] > 6).sum()),
        "z>8": int((sample["redshift"] > 8).sum()),
        "z>10": int((sample["redshift"] > 10).sum()),
        "z>12": int((sample["redshift"] > 12).sum()),
        "z>14": int((sample["redshift"] > 14).sum()),
    }
    source_count = sample["source"].nunique() if n else 0
    stats_text = (
        f"N={n:,} galaxies | sources={source_count} | median z={q[2]:.3f} | "
        f"P05/P95={q[0]:.2f}/{q[4]:.2f} | "
        + " | ".join([f"{k}={v:,}" for k, v in counts.items()])
    )
    fig.text(0.5, 0.017, stats_text, ha="center", fontsize=10, alpha=0.90, color="#f8fafc")
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(PNG_CURVE, dpi=230, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()
    plt.close(fig)


def summary_rows(sample, curve, audit):
    z = sample["redshift"].to_numpy()
    if len(z) == 0:
        return []
    peak_row = curve.iloc[int(np.argmax(curve["count_smooth"].to_numpy()))]
    return [
        ("galaxies_mapped", f"{len(sample):,}"),
        ("online_sources_used", sample["source"].nunique()),
        ("accepted_catalog_tables", int((audit["status"] == "accepted").sum()) if len(audit) else 0),
        ("z_min", f"{np.min(z):.4f}"),
        ("z_p05", f"{np.quantile(z, 0.05):.4f}"),
        ("z_median", f"{np.median(z):.4f}"),
        ("z_mean", f"{np.mean(z):.4f}"),
        ("z_p95", f"{np.quantile(z, 0.95):.4f}"),
        ("z_max", f"{np.max(z):.4f}"),
        ("curve_peak_z", f"{peak_row['z_center']:.4f}"),
        ("count_z_gt_6", int((z > 6).sum())),
        ("count_z_gt_8", int((z > 8).sum())),
        ("count_z_gt_10", int((z > 10).sum())),
        ("count_z_gt_12", int((z > 12).sum())),
        ("count_z_gt_14", int((z > 14).sum())),
    ]


def plot_summary_table(rows):
    fig, ax = plt.subplots(figsize=(10.6, 7.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    cell_text = [[str(a), str(b)] for a, b in rows]
    table = ax.table(cellText=cell_text, colLabels=["Metric", "Value"], loc="center", cellLoc="left", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    table.scale(1.0, 1.36)
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
    ax.set_title("JWST online redshift-curve summary\nPublic VizieR catalog search; not volume-corrected cosmic density",
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

    raw_sample, audit = extract_redshifts_from_vizier()
    sample = dedupe_and_sample(raw_sample)

    if len(sample) < 100:
        audit.to_csv(CSV_AUDIT, index=False)
        print("Insufficient online redshift rows retrieved. Check audit CSV for VizieR/connection details.")
        print_table("OUTPUT", [("audit_csv", str(CSV_AUDIT))], ["item", "value"])
        print(datetime.now().astimezone().isoformat(timespec="seconds"))
        print(f"# {VERSION}")
        return

    curve = make_curve(sample["redshift"].to_numpy())
    sample.to_csv(CSV_REDSHIFTS, index=False)
    curve.to_csv(CSV_HISTOGRAM, index=False)
    audit.to_csv(CSV_AUDIT, index=False)

    plot_curve(curve, sample)
    rows = summary_rows(sample, curve, audit)
    plot_summary_table(rows)

    print_table("JWST / VIZIER REDSHIFT CURVE SUMMARY", rows, ["metric", "value"])
    print_table(
        "OUTPUT SUMMARY",
        [
            ("plot_png", str(PNG_CURVE)),
            ("table_png", str(PNG_TABLE)),
            ("redshift_sample_csv", str(CSV_REDSHIFTS)),
            ("curve_csv", str(CSV_HISTOGRAM)),
            ("audit_csv", str(CSV_AUDIT)),
        ],
        ["item", "value"],
    )
    print("Online retrieval uses VizieR catalog discovery for JWST/JADES/CEERS/ASTRODEEP photometric-redshift catalog matches.")
    print("This is a public-catalog redshift distribution curve, not a volume-corrected cosmic galaxy density function.")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
