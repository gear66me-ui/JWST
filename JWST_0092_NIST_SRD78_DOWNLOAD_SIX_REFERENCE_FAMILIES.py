#!/usr/bin/env python3
"""
JWST_0092_NIST_SRD78_DOWNLOAD_SIX_REFERENCE_FAMILIES.py

Download official query-specific CSV output from the NIST Atomic Spectra
Database, Standard Reference Database 78, for six rest-UV ionic families used
in the MoM-z14 redshift study. The script preserves each NIST response exactly,
creates cleaned machine-readable CSV files, and writes an Excel workbook with
one sheet per ion.

No synthetic spectra. No AI images. No plotting in this download stage.
"""
from __future__ import annotations

import importlib.util
import io
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def ensure_packages() -> None:
    required = {
        "requests": "requests",
        "pandas": "pandas",
        "openpyxl": "openpyxl",
    }
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION = "JWST_0092"
DATABASE = "NIST Standard Reference Database 78"
DATABASE_VERSION = "5.12"
ENDPOINT = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"
ROOT = Path("/content/JWST_OUTPUT")
CSV_DIR = ROOT / "CSV"
DATA_DIR = ROOT / "DATA" / VERSION
CSV_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

QUERIES = [
    {
        "key": "N_IV",
        "spectrum": "N IV",
        "family": "N IV]",
        "element": "Nitrogen",
        "ion_stage": "N3+",
        "low_nm": 147.5,
        "high_nm": 149.2,
        "target_rest_A": "1483.32;1486.50",
    },
    {
        "key": "C_IV",
        "spectrum": "C IV",
        "family": "C IV",
        "element": "Carbon",
        "ion_stage": "C3+",
        "low_nm": 153.8,
        "high_nm": 155.8,
        "target_rest_A": "1548.20;1550.77",
    },
    {
        "key": "He_II",
        "spectrum": "He II",
        "family": "He II",
        "element": "Helium",
        "ion_stage": "He+",
        "low_nm": 163.3,
        "high_nm": 164.8,
        "target_rest_A": "1640.42",
    },
    {
        "key": "O_III",
        "spectrum": "O III",
        "family": "O III]",
        "element": "Oxygen",
        "ion_stage": "O2+",
        "low_nm": 165.4,
        "high_nm": 167.2,
        "target_rest_A": "1660.81;1666.15",
    },
    {
        "key": "N_III",
        "spectrum": "N III",
        "family": "N III]",
        "element": "Nitrogen",
        "ion_stage": "N2+",
        "low_nm": 174.0,
        "high_nm": 176.0,
        "target_rest_A": "1746.82;1748.65;1749.67;1752.16;1753.99",
    },
    {
        "key": "C_III",
        "spectrum": "C III",
        "family": "C III]",
        "element": "Carbon",
        "ion_stage": "C2+",
        "low_nm": 189.7,
        "high_nm": 191.8,
        "target_rest_A": "1906.68;1908.73",
    },
]


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({
        "User-Agent": f"{VERSION} educational NIST-SRD78 CSV downloader",
        "Accept": "text/csv,text/plain;q=0.9,text/html;q=0.3,*/*;q=0.1",
    })
    return session


def query_parameters(item: dict) -> dict:
    return {
        "spectra": item["spectrum"],
        "limits_type": "0",
        "low_w": f"{item['low_nm']:.6f}",
        "upp_w": f"{item['high_nm']:.6f}",
        "unit": "1",
        "submit": "Retrieve Data",
        "de": "0",
        "format": "3",
        "line_out": "0",
        "remove_js": "on",
        "en_unit": "1",
        "output": "0",
        "bibrefs": "1",
        "page_size": "5000",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
        "show_diff_obs_calc": "1",
        "unc_out": "1",
        "order_out": "0",
        "show_av": "2",
        "A_out": "0",
        "f_out": "on",
        "S_out": "on",
        "loggf_out": "on",
        "intens_out": "on",
        "allowed_out": "1",
        "forbid_out": "1",
        "conf_out": "on",
        "term_out": "on",
        "enrg_out": "on",
        "J_out": "on",
        "g_out": "on",
    }


def strip_excel_formula(value):
    if value is None or pd.isna(value):
        return value
    text = str(value).strip()
    if text.startswith('="') and text.endswith('"'):
        return text[2:-1]
    if text.startswith("='") and text.endswith("'"):
        return text[2:-1]
    return text


def normalize_columns(columns) -> list[str]:
    output = []
    counts = {}
    for column in columns:
        name = re.sub(r"\s+", " ", str(column)).strip()
        name = name.replace("\ufeff", "")
        if not name:
            name = "unnamed"
        counts[name] = counts.get(name, 0) + 1
        output.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return output


def parse_official_csv(text: str) -> pd.DataFrame:
    stripped = text.lstrip().lower()
    if stripped.startswith("<!doctype") or stripped.startswith("<html"):
        raise RuntimeError("NIST returned HTML instead of CSV. The query may have failed.")
    frame = pd.read_csv(
        io.StringIO(text),
        dtype=str,
        engine="python",
        keep_default_na=False,
        on_bad_lines="skip",
    )
    frame.columns = normalize_columns(frame.columns)
    frame = frame.loc[:, [c for c in frame.columns if not c.lower().startswith("unnamed")]]
    for column in frame.columns:
        frame[column] = frame[column].map(strip_excel_formula)
    frame = frame.replace({"": pd.NA})
    frame = frame.dropna(how="all").reset_index(drop=True)
    return frame


def add_provenance_columns(frame: pd.DataFrame, item: dict, query_url: str) -> pd.DataFrame:
    output = frame.copy()
    output.insert(0, "selected_family", item["family"])
    output.insert(1, "nist_spectrum", item["spectrum"])
    output.insert(2, "element", item["element"])
    output.insert(3, "ion_stage", item["ion_stage"])
    output.insert(4, "query_low_nm", item["low_nm"])
    output.insert(5, "query_high_nm", item["high_nm"])
    output.insert(6, "study_target_rest_angstrom", item["target_rest_A"])
    output["nist_query_url"] = query_url
    output["database"] = DATABASE
    output["database_version"] = DATABASE_VERSION
    return output


def download_one(session: requests.Session, item: dict) -> tuple[pd.DataFrame, dict]:
    params = query_parameters(item)
    response = session.get(ENDPOINT, params=params, timeout=(30, 240))
    response.raise_for_status()
    text = response.text
    if len(text.strip()) < 40:
        raise RuntimeError(f"NIST returned an unexpectedly short response for {item['spectrum']}.")

    raw_path = DATA_DIR / f"{VERSION}_NIST_SRD78_{item['key']}_OFFICIAL_RAW.csv"
    raw_path.write_text(text, encoding="utf-8")

    clean = parse_official_csv(text)
    if clean.empty:
        raise RuntimeError(f"No tabular rows were parsed for {item['spectrum']}.")
    clean = add_provenance_columns(clean, item, response.url)

    clean_path = CSV_DIR / f"{VERSION}_NIST_SRD78_{item['key']}_CLEAN.csv"
    clean.to_csv(clean_path, index=False)

    record = {
        "key": item["key"],
        "family": item["family"],
        "nist_spectrum": item["spectrum"],
        "element": item["element"],
        "ion_stage": item["ion_stage"],
        "query_low_nm": item["low_nm"],
        "query_high_nm": item["high_nm"],
        "study_target_rest_angstrom": item["target_rest_A"],
        "returned_rows": len(clean),
        "official_raw_csv": str(raw_path),
        "clean_csv": str(clean_path),
        "nist_query_url": response.url,
        "http_status": response.status_code,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return clean, record


def write_excel(frames: dict[str, pd.DataFrame], manifest: pd.DataFrame) -> Path:
    path = CSV_DIR / f"{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        manifest.to_excel(writer, sheet_name="MANIFEST", index=False)
        for item in QUERIES:
            frames[item["key"]].to_excel(writer, sheet_name=item["key"][:31], index=False)
    return path


def write_archive(paths: list[Path], excel_path: Path, master_path: Path, manifest_path: Path) -> Path:
    archive_path = DATA_DIR / f"{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, arcname=path.name)
        archive.write(excel_path, arcname=excel_path.name)
        archive.write(master_path, arcname=master_path.name)
        archive.write(manifest_path, arcname=manifest_path.name)
    return archive_path


def main() -> None:
    session = make_session()
    frames = {}
    records = []
    raw_paths = []

    for index, item in enumerate(QUERIES, start=1):
        print(f"DOWNLOAD {index}/6  {item['spectrum']:<6}  {item['low_nm']:.1f}-{item['high_nm']:.1f} nm")
        frame, record = download_one(session, item)
        frames[item["key"]] = frame
        records.append(record)
        raw_paths.append(Path(record["official_raw_csv"]))
        if index < len(QUERIES):
            time.sleep(1.2)

    manifest = pd.DataFrame(records)
    manifest_path = CSV_DIR / f"{VERSION}_NIST_SRD78_QUERY_MANIFEST.csv"
    manifest.to_csv(manifest_path, index=False)

    master = pd.concat([frames[item["key"]] for item in QUERIES], ignore_index=True, sort=False)
    master_path = CSV_DIR / f"{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES_MASTER.csv"
    master.to_csv(master_path, index=False)

    excel_path = write_excel(frames, manifest)
    archive_path = write_archive(raw_paths, excel_path, master_path, manifest_path)

    print()
    print(f"CODE OUTPUT: {VERSION}")
    print(f"DATABASE        {DATABASE}")
    print(f"VERSION         {DATABASE_VERSION}")
    print("OUTPUT FORMAT   official NIST CSV + cleaned CSV + XLSX workbook")
    print("WAVELENGTHS     query ranges in nm; NIST vacuum convention applies below 200 nm")
    print()
    print(f"{'FAMILY':<10} {'ROWS':>6} {'RANGE [nm]':>17}")
    print("-" * 37)
    for record in records:
        bounds = f"{record['query_low_nm']:.1f}-{record['query_high_nm']:.1f}"
        print(f"{record['family']:<10} {record['returned_rows']:>6} {bounds:>17}")
    print()
    print(f"MASTER CSV      {master_path}")
    print(f"MANIFEST CSV    {manifest_path}")
    print(f"EXCEL WORKBOOK  {excel_path}")
    print(f"RAW ARCHIVE     {archive_path}")
    print(f"RAW DIRECTORY   {DATA_DIR}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
