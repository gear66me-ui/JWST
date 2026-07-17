#!/usr/bin/env python3
"""
JWST_0093_NIST_SRD78_DOWNLOAD_AND_UPLOAD_GITHUB.py

Download official NIST Atomic Spectra Database (SRD 78) line tables for six
rest-UV ionic families, save raw and cleaned copies, create a master CSV and
Excel workbook, and upload the data products to gear66me-ui/JWST.

The script installs missing libraries automatically in a fresh Colab notebook.
No synthetic spectra. No AI images. No plotting in this data-download stage.
"""
from __future__ import annotations

import base64
import getpass
import importlib.util
import io
import os
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
        "lxml": "lxml",
        "openpyxl": "openpyxl",
    }
    missing = [
        pip_name for module_name, pip_name in required.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )


ensure_packages()

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

VERSION = "JWST_0093"
DATABASE = "NIST Standard Reference Database 78"
DATABASE_VERSION = "5.12"
ENDPOINT = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl"

REPOSITORY = "gear66me-ui/JWST"
BRANCH = "main"
REPO_DATA_DIR = f"data/NIST_SRD78/{VERSION}"

ROOT = Path("/content/JWST_OUTPUT")
CSV_DIR = ROOT / "CSV"
DATA_DIR = ROOT / "DATA" / VERSION
for directory in (CSV_DIR, DATA_DIR):
    directory.mkdir(parents=True, exist_ok=True)

QUERIES = [
    dict(key="N_IV", spectrum="N IV", family="N IV]", element="Nitrogen",
         ion_stage="N3+", low_nm=147.5, high_nm=149.2,
         target_rest_A="1483.32;1486.50"),
    dict(key="C_IV", spectrum="C IV", family="C IV", element="Carbon",
         ion_stage="C3+", low_nm=153.8, high_nm=155.8,
         target_rest_A="1548.20;1550.77"),
    dict(key="He_II", spectrum="He II", family="He II", element="Helium",
         ion_stage="He+", low_nm=163.3, high_nm=164.8,
         target_rest_A="1640.42"),
    dict(key="O_III", spectrum="O III", family="O III]", element="Oxygen",
         ion_stage="O2+", low_nm=165.4, high_nm=167.2,
         target_rest_A="1660.81;1666.15"),
    dict(key="N_III", spectrum="N III", family="N III]", element="Nitrogen",
         ion_stage="N2+", low_nm=174.0, high_nm=176.0,
         target_rest_A="1746.82;1748.65;1749.67;1752.16;1753.99"),
    dict(key="C_III", spectrum="C III", family="C III]", element="Carbon",
         ion_stage="C2+", low_nm=189.7, high_nm=191.8,
         target_rest_A="1906.68;1908.73"),
]


def make_session() -> requests.Session:
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "PUT"}),
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": f"{VERSION} educational NIST-SRD78 client",
        "Accept": "text/csv,text/plain,text/html;q=0.9,*/*;q=0.5",
    })
    return session


def query_parameters(item: dict) -> dict:
    # Known-working NIST ASD lines-form parameter set. NIST sometimes returns
    # HTML even when CSV is requested, so the parser supports both.
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
        "en_unit": "1",
        "output": "0",
        "bibrefs": "1",
        "page_size": "500",
        "show_obs_wl": "1",
        "show_calc_wl": "1",
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


def normalize_columns(columns) -> list[str]:
    names = []
    counts = {}
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if str(part).lower() != "nan"]
            name = " ".join(parts)
        else:
            name = str(column)
        name = re.sub(r"\s+", " ", name).strip().replace("\ufeff", "")
        if not name:
            name = "unnamed"
        counts[name] = counts.get(name, 0) + 1
        names.append(name if counts[name] == 1 else f"{name}_{counts[name]}")
    return names


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output.columns = normalize_columns(output.columns)
    output = output.dropna(axis=1, how="all")
    output = output.loc[
        :, [column for column in output.columns
            if not column.lower().startswith("unnamed")]
    ]
    output = output.dropna(how="all").reset_index(drop=True)
    return output


def table_score(frame: pd.DataFrame) -> float:
    columns = " | ".join(str(c).lower() for c in frame.columns)
    wavelength_bonus = 10000 if (
        "wavelength" in columns or "ritz" in columns or "obs" in columns
    ) else 0
    return wavelength_bonus + frame.shape[0] * max(frame.shape[1], 1)


def parse_nist_response(text: str) -> tuple[pd.DataFrame, str]:
    stripped = text.lstrip()
    lower = stripped.lower()

    if lower.startswith("<!doctype") or lower.startswith("<html"):
        tables = pd.read_html(io.StringIO(text))
        tables = [normalize_frame(table) for table in tables if table.shape[1] >= 2]
        tables = [table for table in tables if not table.empty]
        if not tables:
            raise RuntimeError("NIST returned HTML without a usable tabular line table.")
        table = max(tables, key=table_score)
        return table, "HTML_TABLE"

    parsers = [
        ("CSV", dict(sep=",", engine="python")),
        ("TAB", dict(sep="\t", engine="python")),
        ("PIPE", dict(sep="|", engine="python")),
    ]
    errors = []
    for label, kwargs in parsers:
        try:
            frame = pd.read_csv(
                io.StringIO(text),
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
                **kwargs,
            )
            frame = normalize_frame(frame)
            if frame.shape[1] >= 2 and not frame.empty:
                return frame, label
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}")
    raise RuntimeError("Unable to parse NIST response: " + ", ".join(errors))


def numeric_value(value) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    text = str(value).strip()
    text = text.replace('="', "").replace('"', "").replace("−", "-")
    text = re.sub(r"\[[^\]]*\]", "", text)
    match = re.search(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?",
        text,
    )
    if not match:
        return float("nan")
    try:
        return float(match.group())
    except ValueError:
        return float("nan")


def find_column(columns, required: tuple[str, ...], rejected: tuple[str, ...] = ()):
    for column in columns:
        text = str(column).lower()
        if all(token in text for token in required) and not any(
            token in text for token in rejected
        ):
            return column
    return None


def canonicalize(raw: pd.DataFrame, item: dict, query_url: str) -> pd.DataFrame:
    columns = list(raw.columns)
    observed_col = (
        find_column(columns, ("obs", "wavelength"))
        or find_column(columns, ("observed",))
        or find_column(columns, ("obs", "wl"))
    )
    ritz_col = (
        find_column(columns, ("ritz", "wavelength"))
        or find_column(columns, ("ritz",))
        or find_column(columns, ("calc", "wavelength"))
    )

    wavelength = pd.Series(float("nan"), index=raw.index, dtype=float)
    wavelength_source = pd.Series("", index=raw.index, dtype=object)

    if observed_col is not None:
        observed = raw[observed_col].map(numeric_value)
        mask = observed.notna()
        wavelength.loc[mask] = observed.loc[mask]
        wavelength_source.loc[mask] = "observed"

    if ritz_col is not None:
        ritz = raw[ritz_col].map(numeric_value)
        mask = wavelength.isna() & ritz.notna()
        wavelength.loc[mask] = ritz.loc[mask]
        wavelength_source.loc[mask] = "Ritz"

    if wavelength.notna().sum() == 0:
        for column in columns:
            if "wave" not in str(column).lower():
                continue
            candidate = raw[column].map(numeric_value)
            mask = wavelength.isna() & candidate.notna()
            wavelength.loc[mask] = candidate.loc[mask]
            wavelength_source.loc[mask] = "listed"

    intensity_col = (
        find_column(columns, ("rel", "int"))
        or find_column(columns, ("intens",))
    )
    aki_col = (
        find_column(columns, ("aki",))
        or find_column(columns, ("a", "s-1"))
    )
    fik_col = (
        find_column(columns, ("fik",))
        or find_column(columns, ("osc",))
    )
    loggf_col = find_column(columns, ("log", "gf"))

    def numeric_column(column):
        if column is None:
            return pd.Series(float("nan"), index=raw.index, dtype=float)
        return raw[column].map(numeric_value)

    output = pd.DataFrame({
        "selected_family": item["family"],
        "nist_spectrum": item["spectrum"],
        "element": item["element"],
        "ion_stage": item["ion_stage"],
        "rest_wavelength_vacuum_nm": wavelength,
        "rest_wavelength_vacuum_angstrom": wavelength * 10.0,
        "wavelength_source": wavelength_source,
        "relative_intensity": numeric_column(intensity_col),
        "Aki_s^-1": numeric_column(aki_col),
        "fik": numeric_column(fik_col),
        "log_gf": numeric_column(loggf_col),
        "query_low_nm": item["low_nm"],
        "query_high_nm": item["high_nm"],
        "study_target_rest_angstrom": item["target_rest_A"],
        "nist_query_url": query_url,
        "database": DATABASE,
        "database_version": DATABASE_VERSION,
    })
    output = output[
        output["rest_wavelength_vacuum_nm"].notna()
        & output["rest_wavelength_vacuum_nm"].between(
            item["low_nm"], item["high_nm"]
        )
    ].copy()
    output = output.sort_values(
        "rest_wavelength_vacuum_nm"
    ).reset_index(drop=True)
    if output.empty:
        raise RuntimeError(
            f"No parseable {item['spectrum']} wavelength rows were found "
            f"in {item['low_nm']}-{item['high_nm']} nm."
        )
    return output


def download_one(session: requests.Session, item: dict):
    response = session.get(
        ENDPOINT,
        params=query_parameters(item),
        timeout=(30, 240),
    )
    response.raise_for_status()
    text = response.text
    if len(text.strip()) < 40:
        raise RuntimeError(
            f"NIST returned an unexpectedly short response for {item['spectrum']}."
        )

    suffix = ".html" if text.lstrip().lower().startswith(("<!doctype", "<html")) else ".txt"
    raw_path = DATA_DIR / f"{VERSION}_NIST_SRD78_{item['key']}_OFFICIAL_RAW{suffix}"
    raw_path.write_text(text, encoding="utf-8")

    parsed, parse_mode = parse_nist_response(text)
    parsed.insert(0, "selected_family", item["family"])
    parsed.insert(1, "nist_spectrum", item["spectrum"])
    parsed.insert(2, "element", item["element"])
    parsed.insert(3, "ion_stage", item["ion_stage"])
    parsed["nist_query_url"] = response.url
    parsed["database"] = DATABASE
    parsed["database_version"] = DATABASE_VERSION

    parsed_path = CSV_DIR / f"{VERSION}_NIST_SRD78_{item['key']}_FULL_TABLE.csv"
    parsed.to_csv(parsed_path, index=False)

    canonical = canonicalize(parsed, item, response.url)
    canonical_path = CSV_DIR / f"{VERSION}_NIST_SRD78_{item['key']}_CANONICAL.csv"
    canonical.to_csv(canonical_path, index=False)

    record = {
        "key": item["key"],
        "family": item["family"],
        "nist_spectrum": item["spectrum"],
        "query_low_nm": item["low_nm"],
        "query_high_nm": item["high_nm"],
        "target_rest_angstrom": item["target_rest_A"],
        "parsed_table_rows": len(parsed),
        "canonical_wavelength_rows": len(canonical),
        "parse_mode": parse_mode,
        "http_status": response.status_code,
        "query_url": response.url,
        "raw_response": str(raw_path),
        "full_table_csv": str(parsed_path),
        "canonical_csv": str(canonical_path),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return parsed, canonical, record


def write_excel(
    parsed_frames: dict[str, pd.DataFrame],
    canonical_frames: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
) -> Path:
    path = CSV_DIR / f"{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        manifest.to_excel(writer, sheet_name="MANIFEST", index=False)
        for item in QUERIES:
            key = item["key"]
            parsed_frames[key].to_excel(
                writer, sheet_name=f"{key}_FULL"[:31], index=False
            )
            canonical_frames[key].to_excel(
                writer, sheet_name=f"{key}_CANONICAL"[:31], index=False
            )
    return path


def write_raw_archive(raw_paths: list[Path]) -> Path:
    archive_path = DATA_DIR / f"{VERSION}_NIST_SRD78_RAW_RESPONSES.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in raw_paths:
            archive.write(path, arcname=path.name)
    return archive_path


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        if token:
            return str(token).strip()
    except Exception:
        pass
    print()
    print("GitHub write access is required to save the downloaded data.")
    print("Paste a fine-grained token with Contents: Read and write for gear66me-ui/JWST.")
    return getpass.getpass("GitHub token (hidden): ").strip()


def github_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": f"{VERSION}-data-uploader",
    }


def github_existing_sha(
    session: requests.Session,
    token: str,
    repo_path: str,
) -> str | None:
    url = f"https://api.github.com/repos/{REPOSITORY}/contents/{repo_path}"
    response = session.get(
        url,
        headers=github_headers(token),
        params={"ref": BRANCH},
        timeout=60,
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json().get("sha")


def github_upload_file(
    session: requests.Session,
    token: str,
    local_path: Path,
    repo_path: str,
) -> dict:
    sha = github_existing_sha(session, token, repo_path)
    payload = {
        "message": f"{VERSION}: save {local_path.name}",
        "branch": BRANCH,
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    url = f"https://api.github.com/repos/{REPOSITORY}/contents/{repo_path}"
    response = session.put(
        url,
        headers=github_headers(token),
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    return {
        "local_file": str(local_path),
        "repository_path": repo_path,
        "action": "updated" if sha else "created",
        "commit_sha": body.get("commit", {}).get("sha", ""),
        "html_url": body.get("content", {}).get("html_url", ""),
    }


def write_readme(manifest: pd.DataFrame) -> Path:
    lines = [
        f"# {VERSION} NIST SRD 78 data",
        "",
        "Official query-specific line tables downloaded from the NIST Atomic Spectra Database.",
        "",
        f"- Database: {DATABASE}",
        f"- Database version recorded by script: {DATABASE_VERSION}",
        "- Wavelength convention: vacuum below 200 nm",
        "- Six selected ionic families: N IV], C IV, He II, O III], N III], C III]",
        f"- Downloaded UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Files",
        "",
        "- `*_FULL_TABLE.csv`: complete parsed NIST response table",
        "- `*_CANONICAL.csv`: normalized wavelength and strength columns",
        f"- `{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES_MASTER.csv`: merged canonical table",
        f"- `{VERSION}_NIST_SRD78_QUERY_MANIFEST.csv`: query and provenance audit",
        f"- `{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES.xlsx`: workbook with all tables",
        f"- `{VERSION}_NIST_SRD78_RAW_RESPONSES.zip`: exact server responses",
        "",
        "## Row counts",
        "",
        "```text",
        manifest[
            ["family", "parsed_table_rows", "canonical_wavelength_rows", "parse_mode"]
        ].to_string(index=False),
        "```",
        "",
    ]
    path = DATA_DIR / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    session = make_session()
    parsed_frames = {}
    canonical_frames = {}
    records = []
    raw_paths = []
    data_paths = []

    for index, item in enumerate(QUERIES, start=1):
        print(
            f"DOWNLOAD {index}/6  {item['spectrum']:<6}  "
            f"{item['low_nm']:.1f}-{item['high_nm']:.1f} nm"
        )
        parsed, canonical, record = download_one(session, item)
        parsed_frames[item["key"]] = parsed
        canonical_frames[item["key"]] = canonical
        records.append(record)
        raw_paths.append(Path(record["raw_response"]))
        data_paths.extend([
            Path(record["full_table_csv"]),
            Path(record["canonical_csv"]),
        ])
        print(
            f"  parsed={len(parsed):>4}  canonical={len(canonical):>4}  "
            f"mode={record['parse_mode']}"
        )
        if index < len(QUERIES):
            time.sleep(1.2)

    manifest = pd.DataFrame(records)
    manifest_path = CSV_DIR / f"{VERSION}_NIST_SRD78_QUERY_MANIFEST.csv"
    manifest.to_csv(manifest_path, index=False)

    master = pd.concat(
        [canonical_frames[item["key"]] for item in QUERIES],
        ignore_index=True,
        sort=False,
    )
    master_path = CSV_DIR / f"{VERSION}_NIST_SRD78_SIX_REFERENCE_FAMILIES_MASTER.csv"
    master.to_csv(master_path, index=False)

    excel_path = write_excel(parsed_frames, canonical_frames, manifest)
    archive_path = write_raw_archive(raw_paths)
    readme_path = write_readme(manifest)

    upload_paths = data_paths + [
        manifest_path,
        master_path,
        excel_path,
        archive_path,
        readme_path,
    ]

    token = get_github_token()
    if not token:
        raise RuntimeError("No GitHub token was supplied; repository upload cannot continue.")

    print()
    print(f"UPLOADING {len(upload_paths)} files to {REPOSITORY}/{REPO_DATA_DIR}")
    upload_records = []
    for index, local_path in enumerate(upload_paths, start=1):
        repo_path = f"{REPO_DATA_DIR}/{local_path.name}"
        result = github_upload_file(
            session, token, local_path, repo_path
        )
        upload_records.append(result)
        print(
            f"UPLOAD {index:02d}/{len(upload_paths):02d}  "
            f"{result['action']:<7}  {repo_path}"
        )

    upload_manifest = pd.DataFrame(upload_records)
    upload_manifest_path = CSV_DIR / f"{VERSION}_GITHUB_UPLOAD_MANIFEST.csv"
    upload_manifest.to_csv(upload_manifest_path, index=False)
    upload_result = github_upload_file(
        session,
        token,
        upload_manifest_path,
        f"{REPO_DATA_DIR}/{upload_manifest_path.name}",
    )

    print()
    print(f"CODE OUTPUT: {VERSION}")
    print(f"DATABASE        {DATABASE}")
    print(f"NIST ENDPOINT   {ENDPOINT}")
    print("LIBRARIES       requests, pandas, lxml, openpyxl; auto-installed if missing")
    print(f"MASTER CSV      {master_path}")
    print(f"EXCEL WORKBOOK  {excel_path}")
    print(f"RAW ARCHIVE     {archive_path}")
    print(f"GITHUB REPO     {REPOSITORY}")
    print(f"GITHUB FOLDER   {REPO_DATA_DIR}")
    print(f"UPLOAD MANIFEST {upload_manifest_path}")
    print(f"FINAL COMMIT    {upload_result['commit_sha']}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
