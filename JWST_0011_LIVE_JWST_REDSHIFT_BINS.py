# JWST_0011
# Audit: Live public JWST galaxy redshift catalog retrieval and binning.
# Matplotlib only. No AI images. No FITS/image products unless explicitly provided by catalog service.

import os
import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime

VERSION = "JWST_0011"
PROJECT = "LIVE JWST GALAXY REDSHIFT BINNING"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_PNG = OUTPUT_DIR / "PNG"
OUTPUT_CSV = OUTPUT_DIR / "CSV"

Z_MIN = 0.0
Z_MAX = 15.0
COARSE_BINS = [(0.0, 5.0, "0 <= z < 5"), (5.0, 10.0, "5 <= z < 10"), (10.0, 15.000001, "10 <= z <= 15")]
SEARCH_TERMS = ["COSMOS2025", "COSMOS-Web", "JADES DR5", "JWST Advanced Deep Extragalactic Survey"]
MAX_CATALOGS_PER_TERM = 8
MAX_OUTPUT_ROWS = 250000


def ensure_package(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name])


def setup():
    ensure_package("numpy")
    ensure_package("pandas")
    ensure_package("matplotlib")
    ensure_package("astropy")
    ensure_package("astroquery")
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)


def print_table(rows, headers):
    widths = []
    for i, header in enumerate(headers):
        width = max(len(str(header)), *(len(str(row[i])) for row in rows)) if rows else len(str(header))
        widths.append(min(width, 54))
    line = " | ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        cells = []
        for i, value in enumerate(row):
            text = str(value)
            if len(text) > widths[i]:
                text = text[:widths[i] - 1] + "…"
            cells.append(text.ljust(widths[i]))
        print(" | ".join(cells))


def redshift_column_score(colname):
    s = str(colname).lower().replace("-", "_")
    bad = ["err", "error", "sigma", "chi", "flag", "qual", "prob", "pdf", "min", "max", "lo", "hi", "low", "high", "ra", "dec"]
    if any(b in s for b in bad):
        return -100
    exact = {"z": 80, "redshift": 90, "zbest": 95, "z_best": 95, "zphot": 95, "z_phot": 95, "photoz": 90, "photo_z": 90}
    if s in exact:
        return exact[s]
    score = 0
    for token, pts in [("redshift", 70), ("zbest", 70), ("z_best", 70), ("zphot", 68), ("z_phot", 68), ("photoz", 65), ("photo_z", 65), ("zmed", 58), ("z_med", 58), ("ez_z", 52)]:
        if token in s:
            score += pts
    if s.startswith("z_") or s.endswith("_z"):
        score += 25
    return score


def choose_redshift_column(df):
    import pandas as pd
    candidates = []
    n = max(len(df), 1)
    for col in df.columns:
        score = redshift_column_score(col)
        if score <= 0:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        valid = vals[(vals >= Z_MIN) & (vals <= Z_MAX)]
        valid_count = int(valid.notna().sum())
        valid_frac = valid_count / n
        if valid_count >= 25 or valid_frac >= 0.01:
            candidates.append((score + 25 * valid_frac, col, valid_count, valid_frac))
    if not candidates:
        return None, None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates


def find_column(df, tokens):
    low = {str(c).lower(): c for c in df.columns}
    for name, original in low.items():
        for token in tokens:
            if token in name:
                return original
    return None


def try_load_user_catalog_url():
    import pandas as pd
    url = os.environ.get("JWST_CATALOG_URL", "").strip()
    if not url:
        return None, []
    audit = []
    try:
        if url.lower().endswith((".csv", ".txt", ".tsv")):
            sep = "\t" if url.lower().endswith((".tsv", ".txt")) else ","
            df = pd.read_csv(url, sep=sep, low_memory=False)
        else:
            from astropy.table import Table
            table = Table.read(url)
            df = table.to_pandas()
        audit.append({"source": "JWST_CATALOG_URL", "catalog": url, "status": "loaded", "rows": len(df), "columns": len(df.columns)})
        return df, audit
    except Exception as exc:
        audit.append({"source": "JWST_CATALOG_URL", "catalog": url, "status": f"failed: {exc}", "rows": 0, "columns": 0})
        return None, audit


def query_vizier_catalogs():
    import pandas as pd
    from astroquery.vizier import Vizier

    Vizier.ROW_LIMIT = -1
    all_frames = []
    audit_rows = []
    seen_catalogs = set()

    for term in SEARCH_TERMS:
        try:
            found = Vizier.find_catalogs(term)
            keys = list(found.keys())[:MAX_CATALOGS_PER_TERM]
            audit_rows.append({"source": "VizieR search", "catalog": term, "status": f"found {len(found)} catalogs; trying {len(keys)}", "rows": 0, "columns": 0})
        except Exception as exc:
            audit_rows.append({"source": "VizieR search", "catalog": term, "status": f"search failed: {exc}", "rows": 0, "columns": 0})
            continue

        for key in keys:
            if key in seen_catalogs:
                continue
            seen_catalogs.add(key)
            try:
                viz = Vizier(columns=["**"], row_limit=-1, timeout=180)
                tables = viz.get_catalogs(key)
                if len(tables) == 0:
                    audit_rows.append({"source": "VizieR", "catalog": key, "status": "no tables", "rows": 0, "columns": 0})
                    continue
                for t_index, table in enumerate(tables):
                    df = table.to_pandas()
                    zcol, candidates = choose_redshift_column(df)
                    if zcol is None:
                        audit_rows.append({"source": "VizieR", "catalog": f"{key}[{t_index}]", "status": "no usable redshift column", "rows": len(df), "columns": len(df.columns)})
                        continue
                    z = pd.to_numeric(df[zcol], errors="coerce")
                    keep = (z >= Z_MIN) & (z <= Z_MAX)
                    if int(keep.sum()) == 0:
                        audit_rows.append({"source": "VizieR", "catalog": f"{key}[{t_index}]", "status": f"redshift column {zcol}, no rows in z range", "rows": len(df), "columns": len(df.columns)})
                        continue

                    ra_col = find_column(df, ["raj2000", "ra_j2000", "alpha", "ra"])
                    dec_col = find_column(df, ["dej2000", "dec_j2000", "delta", "dec"])
                    id_col = find_column(df, ["id", "name", "source", "obj"])
                    mass_col = find_column(df, ["mstar", "stellar_mass", "lmass", "logm", "mass"])
                    quality_col = find_column(df, ["flag", "qual", "quality", "class", "type"])

                    out = pd.DataFrame({
                        "catalog_key": f"{key}[{t_index}]",
                        "source_id": df[id_col].astype(str) if id_col else [f"{key}_{t_index}_{i}" for i in range(len(df))],
                        "redshift_z": z,
                    })
                    if ra_col:
                        out["ra_deg"] = pd.to_numeric(df[ra_col], errors="coerce")
                    if dec_col:
                        out["dec_deg"] = pd.to_numeric(df[dec_col], errors="coerce")
                    if mass_col:
                        out["stellar_mass_proxy"] = pd.to_numeric(df[mass_col], errors="coerce")
                    if quality_col:
                        out["quality_or_type"] = df[quality_col].astype(str)
                    out = out[keep].copy()
                    all_frames.append(out)
                    audit_rows.append({"source": "VizieR", "catalog": f"{key}[{t_index}]", "status": f"loaded redshift column {zcol}", "rows": len(out), "columns": len(df.columns)})
            except Exception as exc:
                audit_rows.append({"source": "VizieR", "catalog": key, "status": f"load failed: {exc}", "rows": 0, "columns": 0})

    if not all_frames:
        return pd.DataFrame(), audit_rows
    combined = pd.concat(all_frames, ignore_index=True)
    return combined, audit_rows


def normalize_and_dedupe(df):
    import pandas as pd
    if df is None or len(df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    if "redshift_z" not in df.columns:
        zcol, _ = choose_redshift_column(df)
        if zcol is None:
            return pd.DataFrame(), pd.DataFrame()
        df["redshift_z"] = pd.to_numeric(df[zcol], errors="coerce")
    df = df[(df["redshift_z"] >= Z_MIN) & (df["redshift_z"] <= Z_MAX)].copy()
    raw = df.copy()

    if {"ra_deg", "dec_deg"}.issubset(df.columns):
        df["_dedupe_key"] = df["ra_deg"].round(5).astype(str) + "_" + df["dec_deg"].round(5).astype(str) + "_" + df["redshift_z"].round(3).astype(str)
        df = df.drop_duplicates("_dedupe_key").drop(columns=["_dedupe_key"])
    else:
        key_cols = [c for c in ["catalog_key", "source_id", "redshift_z"] if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(key_cols)
    return raw.reset_index(drop=True), df.reset_index(drop=True)


def add_cosmology(df):
    from astropy.cosmology import Planck18 as cosmo
    if len(df) == 0:
        return df
    df = df.copy()
    df["comoving_distance_gly"] = [cosmo.comoving_distance(float(z)).to("Glyr").value for z in df["redshift_z"]]
    df["lookback_time_gyr"] = [cosmo.lookback_time(float(z)).value for z in df["redshift_z"]]
    df["universe_age_myr"] = [cosmo.age(float(z)).value * 1000.0 for z in df["redshift_z"]]
    return df


def make_bins(df):
    import pandas as pd
    from astropy.cosmology import Planck18 as cosmo
    rows = []
    total = max(len(df), 1)
    for z0, z1, label in COARSE_BINS:
        if z1 > Z_MAX:
            mask = (df["redshift_z"] >= z0) & (df["redshift_z"] <= Z_MAX)
            z1_display = Z_MAX
        else:
            mask = (df["redshift_z"] >= z0) & (df["redshift_z"] < z1)
            z1_display = z1
        sub = df[mask]
        age_young_myr = cosmo.age(z0).value * 1000.0
        age_old_myr = cosmo.age(z1_display).value * 1000.0
        rows.append({
            "redshift_bin": label,
            "z_min": z0,
            "z_max": z1_display,
            "count": int(len(sub)),
            "share_percent": 100.0 * len(sub) / total if total else 0.0,
            "median_z": float(sub["redshift_z"].median()) if len(sub) else None,
            "mean_z": float(sub["redshift_z"].mean()) if len(sub) else None,
            "universe_age_range_myr": f"{age_old_myr:.0f} to {age_young_myr:.0f}",
            "age_at_high_z_myr": age_old_myr,
            "age_at_low_z_myr": age_young_myr,
        })
    return pd.DataFrame(rows)


def style_dark(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.tick_params(colors="#dbeafe")
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def plot_coarse_bins(bin_df):
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    style_dark(fig, ax)
    x = np.arange(len(bin_df))
    bars = ax.bar(x, bin_df["count"], edgecolor="#cbd5e1", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_df["redshift_bin"], color="#f8fafc")
    ax.set_yscale("log" if bin_df["count"].max() > 1000 else "linear")
    ax.set_ylabel("Galaxy count" + (" (log scale)" if bin_df["count"].max() > 1000 else ""))
    ax.set_xlabel("Redshift bin")
    ax.set_title("JWST public catalog galaxies binned by redshift\n0-5, 5-10, and 10-15")
    for bar, (_, row) in zip(bars, bin_df.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{int(row['count']):,}\n{row['share_percent']:.2f}%",
                ha="center", va="bottom", color="#f8fafc", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#020617", edgecolor="#475569", alpha=0.72))
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_COARSE_REDSHIFT_BIN_COUNTS.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_histogram(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(13.6, 7.8))
    style_dark(fig, ax)
    ax.hist(df["redshift_z"], bins=60, range=(Z_MIN, Z_MAX), edgecolor="#cbd5e1", linewidth=0.25)
    ax.axvspan(0, 5, color="#0e7490", alpha=0.11)
    ax.axvspan(5, 10, color="#1d4ed8", alpha=0.10)
    ax.axvspan(10, 15, color="#7f1d1d", alpha=0.13)
    ax.set_yscale("log" if len(df) > 1000 else "linear")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Galaxy count per fine bin" + (" (log scale)" if len(df) > 1000 else ""))
    ax.set_title("Fine redshift distribution for retrieved JWST public catalog galaxies\nall retrieved rows with usable 0 <= z <= 15 redshifts")
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_FINE_REDSHIFT_HISTOGRAM.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def styled_bin_table(bin_df, raw_count, dedup_count):
    import matplotlib.pyplot as plt
    rows = []
    for _, r in bin_df.iterrows():
        rows.append([
            r["redshift_bin"],
            f"{int(r['count']):,}",
            f"{r['share_percent']:.3f}%",
            "—" if r["median_z"] is None else f"{r['median_z']:.4f}",
            r["universe_age_range_myr"],
        ])
    fig, ax = plt.subplots(figsize=(12.8, 4.7))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    col_labels = ["Redshift bin", "Count", "Share", "Median z", "Universe age range (Myr)"]
    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.1)
    table.scale(1.0, 1.65)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#334155")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(["#082f49", "#172554", "#3b1114"][r - 1])
            cell.get_text().set_color("#e5e7eb")
    title = f"{VERSION} redshift bin count table — raw rows {raw_count:,}, deduped rows {dedup_count:,}"
    ax.set_title(title, color="#f8fafc", fontsize=13.2, pad=16)
    fig.tight_layout()
    path = OUTPUT_PNG / f"{VERSION}_STYLED_BIN_TABLE.png"
    fig.savefig(path, dpi=275, facecolor=fig.get_facecolor())
    plt.show()
    return path


def main():
    setup()
    import pandas as pd

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("LIVE RETRIEVAL")
    print("Searching public catalog services for JWST galaxy redshift tables...")

    user_df, user_audit = try_load_user_catalog_url()
    if user_df is not None:
        raw, deduped = normalize_and_dedupe(user_df)
        audit_rows = user_audit
    else:
        combined, audit_rows = query_vizier_catalogs()
        raw, deduped = normalize_and_dedupe(combined)

    audit_df = pd.DataFrame(audit_rows)
    audit_path = OUTPUT_CSV / f"{VERSION}_CATALOG_RETRIEVAL_AUDIT.csv"
    audit_df.to_csv(audit_path, index=False)

    if len(deduped) == 0:
        notes_path = OUTPUT_CSV / f"{VERSION}_NO_LIVE_REDSHIFT_CATALOG_FOUND.txt"
        notes_path.write_text(
            "No live public catalog with a usable 0 <= z <= 15 redshift column was retrieved.\n"
            "This does not mean JWST has no galaxies; it means the script could not access a unified public redshift catalog from the attempted services.\n"
            "Try setting JWST_CATALOG_URL to a public CSV/FITS/VOTable catalog URL with a redshift column, then rerun.\n"
            "The retrieval audit CSV lists attempted catalogs and failures.\n",
            encoding="utf-8",
        )
        print("")
        print("RESULTS")
        print_table([
            ("Raw rows", 0),
            ("Deduped rows", 0),
            ("Status", "no live redshift catalog retrieved"),
            ("Audit CSV", str(audit_path)),
            ("Notes", str(notes_path)),
        ], ["Metric", "Value"])
        print("")
        print(datetime.now().astimezone().isoformat(timespec="seconds"))
        print(f"# {VERSION}")
        return

    deduped = add_cosmology(deduped)
    bin_df = make_bins(deduped)

    galaxies_path = OUTPUT_CSV / f"{VERSION}_RETRIEVED_DEDUPED_GALAXIES_Z0_Z15.csv"
    if len(deduped) > MAX_OUTPUT_ROWS:
        deduped.sample(MAX_OUTPUT_ROWS, random_state=11).to_csv(galaxies_path, index=False)
        saved_mode = f"sampled {MAX_OUTPUT_ROWS:,} rows from {len(deduped):,} deduped rows"
    else:
        deduped.to_csv(galaxies_path, index=False)
        saved_mode = f"saved all {len(deduped):,} deduped rows"

    bins_path = OUTPUT_CSV / f"{VERSION}_REDSHIFT_BIN_COUNTS.csv"
    bin_df.to_csv(bins_path, index=False)

    coarse_png = plot_coarse_bins(bin_df)
    hist_png = plot_histogram(deduped)
    table_png = styled_bin_table(bin_df, len(raw), len(deduped))

    print("")
    print("RESULTS")
    print_table([
        ("Raw retrieved rows in z range", f"{len(raw):,}"),
        ("Deduped plotted/catalog rows", f"{len(deduped):,}"),
        ("z range used", f"{deduped['redshift_z'].min():.6f} to {deduped['redshift_z'].max():.6f}"),
        ("CSV galaxy save mode", saved_mode),
        ("Important caveat", "not a complete all-sky JWST galaxy census"),
    ], ["Metric", "Value"])
    print("")
    print("BIN COUNTS")
    print_table([
        (r["redshift_bin"], f"{int(r['count']):,}", f"{r['share_percent']:.3f}%", "—" if r["median_z"] is None else f"{r['median_z']:.4f}", r["universe_age_range_myr"])
        for _, r in bin_df.iterrows()
    ], ["z bin", "count", "share", "median z", "universe age Myr"])
    print("")
    print("OUTPUT SUMMARY")
    print_table([
        ("png", str(coarse_png)),
        ("png", str(hist_png)),
        ("png", str(table_png)),
        ("csv", str(galaxies_path)),
        ("csv", str(bins_path)),
        ("csv", str(audit_path)),
    ], ["Type", "Path"])
    print("")
    print("COMMENTS")
    print("This retrieves the maximum public redshift-catalog rows the script can access live, then bins them.")
    print("MAST/JWST observation metadata is not the same thing as a unified galaxy redshift catalog.")
    print("Matplotlib only. No AI images.")
    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
