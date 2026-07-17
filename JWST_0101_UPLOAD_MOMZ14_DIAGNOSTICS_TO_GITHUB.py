#!/usr/bin/env python3
"""
Upload the MoM-z14 diagnostic plots and the exact source CSVs used to create them
from the current Colab runtime to gear66me-ui/JWST for visual/data inspection.
"""
from __future__ import annotations

import base64
import getpass
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def ensure_requests() -> None:
    try:
        import requests  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])


ensure_requests()
import requests

VERSION = "JWST_0101"
OWNER = "gear66me-ui"
REPO = "JWST"
BRANCH = "main"
REMOTE_FOLDER = f"artifacts/MOMZ14/{VERSION}"
ROOT = Path("/content/JWST_OUTPUT")
PNG = ROOT / "PNG"
CSV = ROOT / "CSV"
RUN_0100 = Path("/content/JWST_0100_MOMZ14_SIX_RAW_REST_WINDOWS.py")
URL_0100 = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "JWST_0100_MOMZ14_SIX_RAW_REST_WINDOWS.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def github_token() -> str:
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    try:
        from google.colab import userdata
        token = str(userdata.get("GITHUB_TOKEN") or "").strip()
        if token:
            return token
    except Exception:
        pass
    return getpass.getpass(
        "GitHub token for gear66me-ui/JWST (Contents: read/write; input hidden): "
    ).strip()


def ensure_outputs() -> None:
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    pngs = sorted(PNG.glob("JWST_0100*.png"))
    csvs = sorted(CSV.glob("JWST_0100*.csv"))
    if pngs and csvs:
        return
    print("JWST_0100 outputs are missing; regenerating them from the existing JWST_0098 native data.")
    subprocess.check_call([
        "bash", "-lc",
        f"curl -fsSL -o {RUN_0100} {URL_0100} && python {RUN_0100}",
    ])


def choose_files() -> list[Path]:
    files: list[Path] = []
    files.extend(sorted(PNG.glob("JWST_0100*.png")))
    files.extend(sorted(CSV.glob("JWST_0100*.csv")))

    for name in (
        "JWST_0098_MOMZ14_MATCHED_X1D_CANDIDATES.csv",
        "JWST_0098_X1D_EXTENSION_AUDIT.csv",
        "JWST_0098_MAST_DOWNLOAD_MANIFEST.csv",
        "JWST_0098_MAST_MATCHED_OBSERVATIONS.csv",
        "JWST_0098_MAST_ALL_PRODUCTS.csv",
    ):
        path = CSV / name
        if path.exists():
            files.append(path)

    native = sorted(CSV.glob("JWST_0098_*_NATIVE_X1D.csv"))
    if native:
        files.extend(native)

    unique: list[Path] = []
    seen: set[str] = set()
    for path in files:
        key = str(path.resolve())
        if path.is_file() and key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise RuntimeError("No JWST_0098/JWST_0100 diagnostic files were found in /content/JWST_OUTPUT.")
    return unique


def upload_file(session: requests.Session, token: str, local: Path, remote: str) -> tuple[str, str]:
    api = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{remote}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    lookup = session.get(api, headers=headers, params={"ref": BRANCH}, timeout=60)
    existing_sha = None
    action = "created"
    if lookup.status_code == 200:
        existing_sha = lookup.json().get("sha")
        action = "updated"
    elif lookup.status_code != 404:
        raise RuntimeError(f"GitHub lookup failed for {remote}: {lookup.status_code} {lookup.text[:300]}")

    payload = {
        "message": f"Upload {VERSION} diagnostic file {local.name}",
        "content": base64.b64encode(local.read_bytes()).decode("ascii"),
        "branch": BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    response = session.put(api, headers=headers, data=json.dumps(payload), timeout=300)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"GitHub upload failed for {remote}: {response.status_code} {response.text[:500]}")
    commit_sha = response.json().get("commit", {}).get("sha", "")
    return action, commit_sha


def make_readme(files: list[Path]) -> Path:
    rows = []
    for path in files:
        rows.append(
            f"| `{path.name}` | {path.suffix.lower().lstrip('.')} | {path.stat().st_size:,} | `{sha256(path)}` |"
        )
    text = "\n".join([
        f"# {VERSION} MoM-z14 diagnostic upload",
        "",
        "Files uploaded directly from the active Google Colab runtime for visual and numerical inspection.",
        "",
        "| File | Type | Bytes | SHA-256 |",
        "|---|---:|---:|---|",
        *rows,
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
    ])
    path = ROOT / f"{VERSION}_README.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    ensure_outputs()
    files = choose_files()
    readme = make_readme(files)
    files.append(readme)

    token = github_token()
    if not token:
        raise RuntimeError("No GitHub token was supplied.")

    session = requests.Session()
    manifest = []
    last_commit = ""
    print(f"UPLOADING {len(files)} files to {OWNER}/{REPO}/{REMOTE_FOLDER}")
    for index, path in enumerate(files, 1):
        remote = f"{REMOTE_FOLDER}/{path.name}"
        action, commit_sha = upload_file(session, token, path, remote)
        last_commit = commit_sha or last_commit
        manifest.append({
            "local_path": str(path),
            "github_path": remote,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "action": action,
            "commit_sha": commit_sha,
        })
        print(f"UPLOAD {index:02d}/{len(files):02d}  {action:<7}  {remote}")

    manifest_path = CSV / f"{VERSION}_GITHUB_UPLOAD_MANIFEST.csv"
    import csv
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print()
    print(f"CODE OUTPUT: {VERSION}")
    print(f"GITHUB REPO     {OWNER}/{REPO}")
    print(f"GITHUB FOLDER   {REMOTE_FOLDER}")
    print(f"FILES UPLOADED  {len(files)}")
    print(f"MANIFEST        {manifest_path}")
    print(f"FINAL COMMIT    {last_commit}")
    print(f"Timestamp       {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
