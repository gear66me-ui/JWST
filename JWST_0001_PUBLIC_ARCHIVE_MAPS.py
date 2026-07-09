# JWST_0001
# Audit: JWST public archive metadata maps; metadata only, no FITS bulk download.

import sys
import subprocess
import importlib
from pathlib import Path
from datetime import datetime, timezone

VERSION = "JWST_0001"
PROJECT = "JWST PUBLIC ARCHIVE MAPPING"
OUTPUT_DIR = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
OUTPUT_CSV = OUTPUT_DIR / "CSV"
OUTPUT_PNG = OUTPUT_DIR / "PNG"

TARGETS = [
    {"target_group": "SMACS 0723 Deep Field", "ra_deg": 110.8375, "dec_deg": -73.4550, "radius_deg": 0.25},
    {"target_group": "Stephan's Quintet", "ra_deg": 339.0144, "dec_deg": 33.9750, "radius_deg": 0.25},
    {"target_group": "NGC 3324 Carina", "ra_deg": 159.3330, "dec_deg": -58.6250, "radius_deg": 0.30},
    {"target_group": "Southern Ring NGC 3132", "ra_deg": 151.7583, "dec_deg": -40.4364, "radius_deg": 0.25},
    {"target_group": "Pillars of Creation M16", "ra_deg": 274.7000, "dec_deg": -13.8067, "radius_deg": 0.30},
    {"target_group": "NGC 628 PHANGS", "ra_deg": 24.1740, "dec_deg": 15.7837, "radius_deg": 0.35},
    {"target_group": "Orion Bar", "ra_deg": 83.8390, "dec_deg": -5.4170, "radius_deg": 0.25},
]

KEEP_COLUMNS = [
    "obs_collection",
    "instrument_name",
    "dataproduct_type",
    "calib_level",
    "intentType",
    "target_name",
    "proposal_id",
    "proposal_pi",
    "obs_id",
    "obsid",
    "s_ra",
    "s_dec",
    "t_min",
    "t_max",
    "t_exptime",
    "filters",
    "em_min",
    "em_max",
    "dataRights",
]

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
    OUTPUT_CSV.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG.mkdir(parents=True, exist_ok=True)

def clean_text(value):
    if value is None:
        return ""
    text = str(value)
    if text in ("--", "nan", "None"):
        return ""
    return text

def table_to_dataframe(table, target_group):
    import numpy as np
    import pandas as pd

    cols = [c for c in KEEP_COLUMNS if c in table.colnames]
    if not cols:
        return pd.DataFrame()

    df = table[cols].to_pandas()
    df["target_group"] = target_group

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(clean_text)

    for col in ["s_ra", "s_dec", "t_min", "t_max", "t_exptime", "em_min", "em_max", "calib_level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def query_jwst_metadata():
    import pandas as pd
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations

    frames = []
    failures = []

    for item in TARGETS:
        try:
            coord = SkyCoord(item["ra_deg"], item["dec_deg"], unit="deg", frame="icrs")
            obs = Observations.query_region(coord, radius=item["radius_deg"] * u.deg)

            if len(obs) == 0:
                failures.append((item["target_group"], "NO OBSERVATIONS"))
                continue

            mask = obs["obs_collection"] == "JWST" if "obs_collection" in obs.colnames else None
            if mask is not None:
                obs = obs[mask]

            if "intentType" in obs.colnames and len(obs) > 0:
                obs = obs[obs["intentType"] == "science"]

            if len(obs) == 0:
                failures.append((item["target_group"], "NO JWST SCIENCE MATCHES"))
                continue

            df = table_to_dataframe(obs, item["target_group"])
            if len(df) > 0:
                df["query_ra_deg"] = item["ra_deg"]
                df["query_dec_deg"] = item["dec_deg"]
                df["query_radius_deg"] = item["radius_deg"]
                frames.append(df)
            else:
                failures.append((item["target_group"], "NO USABLE COLUMNS"))

        except Exception as exc:
            failures.append((item["target_group"], str(exc)[:160]))

    if frames:
        out = pd.concat(frames, ignore_index=True)
    else:
        out = pd.DataFrame()

    return out, failures

def enrich(df):
    import numpy as np
    import pandas as pd
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time

    if df.empty:
        return df

    if "s_ra" in df.columns and "s_dec" in df.columns:
        ok = df["s_ra"].notna() & df["s_dec"].notna()
        df["gal_l_deg"] = np.nan
        df["gal_b_deg"] = np.nan
        if ok.any():
            coords = SkyCoord(df.loc[ok, "s_ra"].to_numpy() * u.deg,
                              df.loc[ok, "s_dec"].to_numpy() * u.deg,
                              frame="icrs")
            df.loc[ok, "gal_l_deg"] = coords.galactic.l.wrap_at(180 * u.deg).deg
            df.loc[ok, "gal_b_deg"] = coords.galactic.b.deg

    if "t_min" in df.columns:
        ok = df["t_min"].notna()
        df["obs_datetime_utc"] = ""
        if ok.any():
            times = Time(df.loc[ok, "t_min"].to_numpy(), format="mjd")
            df.loc[ok, "obs_datetime_utc"] = [t.iso for t in times]

    if "instrument_name" in df.columns:
        df["instrument_short"] = df["instrument_name"].astype(str).str.split("/").str[0].str.upper()
    else:
        df["instrument_short"] = "UNKNOWN"

    if "filters" in df.columns:
        df["filters_clean"] = df["filters"].astype(str).replace({"": "UNSPECIFIED"})
    else:
        df["filters_clean"] = "UNSPECIFIED"

    if "t_exptime" in df.columns:
        df["t_exptime"] = pd.to_numeric(df["t_exptime"], errors="coerce")
    else:
        df["t_exptime"] = np.nan

    return df

def safe_title(text):
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")

def save_sky_maps(df):
    import matplotlib.pyplot as plt

    if df.empty:
        return []

    saved = []

    plot_df = df.dropna(subset=["s_ra", "s_dec"]).copy()
    if len(plot_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, g in plot_df.groupby("target_group"):
            ax.scatter(g["s_ra"], g["s_dec"], s=9, alpha=0.70, label=name)
        ax.invert_xaxis()
        ax.set_xlabel("Right Ascension, deg")
        ax.set_ylabel("Declination, deg")
        ax.set_title("JWST observation positions around selected iconic targets")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(fontsize=7, markerscale=1.5, loc="best")
        fig.tight_layout()
        path = OUTPUT_PNG / f"{VERSION}_RA_DEC_TARGET_MAP.png"
        fig.savefig(path, dpi=180)
        plt.show()
        saved.append(path)

    if "gal_l_deg" in df.columns and "gal_b_deg" in df.columns:
        gdf = df.dropna(subset=["gal_l_deg", "gal_b_deg"]).copy()
        if len(gdf) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            for name, g in gdf.groupby("instrument_short"):
                ax.scatter(g["gal_l_deg"], g["gal_b_deg"], s=9, alpha=0.70, label=name)
            ax.set_xlabel("Galactic longitude, deg, wrapped at 180")
            ax.set_ylabel("Galactic latitude, deg")
            ax.set_title("JWST sample positions in Galactic coordinates")
            ax.grid(True, linewidth=0.4, alpha=0.35)
            ax.legend(fontsize=8, markerscale=1.5, loc="best")
            fig.tight_layout()
            path = OUTPUT_PNG / f"{VERSION}_GALACTIC_COORDINATE_MAP.png"
            fig.savefig(path, dpi=180)
            plt.show()
            saved.append(path)

    return saved

def save_summary_plots(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    saved = []
    if df.empty:
        return saved

    counts = df.groupby(["target_group", "instrument_short"]).size().reset_index(name="count")
    pivot = counts.pivot(index="target_group", columns="instrument_short", values="count").fillna(0)
    if len(pivot) > 0:
        ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6), width=0.8)
        ax.set_xlabel("Target group")
        ax.set_ylabel("Observation count")
        ax.set_title("JWST observation count by target and instrument")
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.35)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        path = OUTPUT_PNG / f"{VERSION}_OBSERVATION_COUNT_BY_TARGET_INSTRUMENT.png"
        plt.savefig(path, dpi=180)
        plt.show()
        saved.append(path)

    exp = df.dropna(subset=["t_exptime"]).copy()
    exp = exp[exp["t_exptime"] > 0]
    if len(exp) > 0:
        exp_sum = exp.groupby("target_group")["t_exptime"].sum().sort_values(ascending=True) / 3600.0
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(exp_sum.index, exp_sum.values)
        ax.set_xlabel("Total exposure time, hours")
        ax.set_ylabel("Target group")
        ax.set_title("Total JWST exposure time in sampled target regions")
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
        fig.tight_layout()
        path = OUTPUT_PNG / f"{VERSION}_EXPOSURE_HOURS_BY_TARGET.png"
        fig.savefig(path, dpi=180)
        plt.show()
        saved.append(path)

    top_filters = (
        df["filters_clean"]
        .replace("", "UNSPECIFIED")
        .value_counts()
        .head(20)
        .sort_values(ascending=True)
    )
    if len(top_filters) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(top_filters.index, top_filters.values)
        ax.set_xlabel("Observation count")
        ax.set_ylabel("Filter / disperser field")
        ax.set_title("Top JWST filter metadata values in sampled regions")
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.35)
        fig.tight_layout()
        path = OUTPUT_PNG / f"{VERSION}_TOP_FILTERS.png"
        fig.savefig(path, dpi=180)
        plt.show()
        saved.append(path)

    if "t_min" in df.columns:
        timeline = df.dropna(subset=["t_min"]).copy()
        if len(timeline) > 0:
            timeline["month"] = pd.to_datetime(timeline["obs_datetime_utc"], errors="coerce").dt.to_period("M").astype(str)
            month_counts = timeline.groupby("month").size()
            if len(month_counts) > 0:
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.plot(month_counts.index, month_counts.values, marker="o", linewidth=1)
                ax.set_xlabel("Observation month")
                ax.set_ylabel("Observation count")
                ax.set_title("JWST observation timeline in sampled target regions")
                ax.grid(True, linewidth=0.4, alpha=0.35)
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                path = OUTPUT_PNG / f"{VERSION}_OBSERVATION_TIMELINE.png"
                fig.savefig(path, dpi=180)
                plt.show()
                saved.append(path)

    return saved

def save_tables(df):
    if df.empty:
        return None, None

    metadata_path = OUTPUT_CSV / f"{VERSION}_JWST_METADATA_SAMPLE.csv"
    df.to_csv(metadata_path, index=False)

    summary = (
        df.groupby(["target_group", "instrument_short"])
        .agg(
            observation_count=("obs_id", "count") if "obs_id" in df.columns else ("instrument_short", "count"),
            exposure_seconds=("t_exptime", "sum"),
        )
        .reset_index()
        .sort_values(["target_group", "instrument_short"])
    )
    summary["exposure_hours"] = summary["exposure_seconds"] / 3600.0
    summary_path = OUTPUT_CSV / f"{VERSION}_JWST_SUMMARY_BY_TARGET_INSTRUMENT.csv"
    summary.to_csv(summary_path, index=False)

    return metadata_path, summary_path

def print_table(rows, headers):
    widths = []
    for i, h in enumerate(headers):
        width = max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        widths.append(min(width, 42))
    line = " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = []
        for i, value in enumerate(r):
            text = str(value)
            if len(text) > widths[i]:
                text = text[:widths[i]-1] + "…"
            cells.append(text.ljust(widths[i]))
        print(" | ".join(cells))

def main():
    setup()

    print(f"CODE OUTPUT: {VERSION}")
    print("")
    print("CODE INPUTS")
    print_table([
        ("Project", PROJECT),
        ("Output folder", str(OUTPUT_DIR)),
        ("Targets", len(TARGETS)),
        ("Mode", "MAST JWST metadata query, no FITS download"),
    ], ["Field", "Value"])

    df, failures = query_jwst_metadata()
    df = enrich(df)

    metadata_path, summary_path = save_tables(df)
    png_paths = []
    png_paths.extend(save_sky_maps(df))
    png_paths.extend(save_summary_plots(df))

    print("")
    print("RESULTS")
    if df.empty:
        print("No JWST metadata rows were retrieved.")
    else:
        rows = []
        for target, g in df.groupby("target_group"):
            exposure_hours = g["t_exptime"].sum() / 3600.0 if "t_exptime" in g.columns else 0.0
            instruments = ", ".join(sorted(set(g["instrument_short"].astype(str))))[:42]
            rows.append((target, len(g), f"{exposure_hours:.6f}", instruments))
        print_table(rows, ["Target group", "Rows", "Exposure hours", "Instruments"])

    if failures:
        print("")
        print("QUERY NOTES")
        print_table(failures, ["Target group", "Status"])

    print("")
    print("OUTPUT SUMMARY")
    output_rows = []
    if metadata_path:
        output_rows.append(("metadata_csv", str(metadata_path)))
    if summary_path:
        output_rows.append(("summary_csv", str(summary_path)))
    for path in png_paths:
        output_rows.append(("plot_png", str(path)))
    print_table(output_rows, ["Type", "Path"])

    print("")
    print("COMMENTS")
    print("This script queries JWST observation metadata only.")
    print("It intentionally avoids bulk FITS downloads and product-list explosion.")
    print("Use the CSV outputs as the next launchpad for deeper target-specific analysis.")

    print("")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")

if __name__ == "__main__":
    main()
