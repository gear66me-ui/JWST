# JWST_0023
# Audit: Push JWST_0022 output images and CSV files from Colab to GitHub. No AI images. No plot generation.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import mimetypes
import subprocess
import importlib

VERSION = "JWST_0023"
PROJECT = "PUSH JWST_0022 OUTPUTS TO GITHUB"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0022_EINSTEIN_EQUATION_PLOTS"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

FILES = [
    (PNG / "JWST_0022_EINSTEIN_EQUATION_DASHBOARD.png", f"{DEST_ROOT}/PNG/JWST_0022_EINSTEIN_EQUATION_DASHBOARD.png"),
    (PNG / "JWST_0022_FRIEDMANN_REDSHIFT_DISTANCE_HORIZONS.png", f"{DEST_ROOT}/PNG/JWST_0022_FRIEDMANN_REDSHIFT_DISTANCE_HORIZONS.png"),
    (CSV / "JWST_0022_SCHWARZSCHILD_CURVES.csv", f"{DEST_ROOT}/CSV/JWST_0022_SCHWARZSCHILD_CURVES.csv"),
    (CSV / "JWST_0022_FRIEDMANN_TERMS.csv", f"{DEST_ROOT}/CSV/JWST_0022_FRIEDMANN_TERMS.csv"),
    (CSV / "JWST_0022_HORIZON_SCALES.csv", f"{DEST_ROOT}/CSV/JWST_0022_HORIZON_SCALES.csv"),
    (CSV / "JWST_0022_FRIEDMANN_DISTANCE_CURVE.csv", f"{DEST_ROOT}/CSV/JWST_0022_FRIEDMANN_DISTANCE_CURVE.csv"),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def get_token():
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
    import getpass
    return getpass.getpass("GitHub token, hidden input: ").strip()


def github_get_sha(token, repo, path):
    import requests
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    r = requests.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, params={"ref": BRANCH}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def github_put_file(token, repo, src, dest):
    import requests
    content_b64 = base64.b64encode(src.read_bytes()).decode("ascii")
    sha = github_get_sha(token, repo, dest)
    url = f"https://api.github.com/repos/{repo}/contents/{dest}"
    payload = {
        "message": f"{VERSION}: upload {src.name}",
        "content": content_b64,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, json=payload, timeout=60)
    r.raise_for_status()
    raw = f"https://raw.githubusercontent.com/{repo}/{BRANCH}/{dest}"
    return raw, "updated" if sha else "created"


def print_table(rows, heads):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(heads)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(heads)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(heads))))


def main():
    need("requests")
    token = get_token()
    link_rows = []
    missing = []
    for src, dest in FILES:
        if not src.exists():
            missing.append(str(src))
            continue
        raw, status = github_put_file(token, REPO, src, dest)
        size_mb = src.stat().st_size / (1024 * 1024)
        link_rows.append({
            "type": src.suffix.lower().replace(".", ""),
            "file": src.name,
            "status": status,
            "size_mb": round(size_mb, 4),
            "raw_url": raw,
        })
    CSV.mkdir(parents=True, exist_ok=True)
    links_csv = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    links_json = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"
    import pandas as pd
    pd.DataFrame(link_rows).to_csv(links_csv, index=False)
    links_json.write_text(json.dumps(link_rows, indent=2), encoding="utf-8")

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Repo", REPO),
        ("Destination", DEST_ROOT),
        ("Uploaded", len(link_rows)),
        ("Missing", len(missing)),
    ], ["Field", "Value"])
    if missing:
        print("\nMISSING FILES")
        print_table([(m,) for m in missing], ["Path"])
        print("\nRun JWST_0022 first, then rerun this exporter.")
    print("\nRAW LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"]) for r in link_rows], ["Type", "Status", "File", "Raw URL"])
    print("\nLOCAL LINK TABLES")
    print_table([("csv", str(links_csv)), ("json", str(links_json))], ["Type", "Path"])
    print("\nCOMMENTS")
    print("This uploads existing JWST_0022 output files from Colab to GitHub.")
    print("No token is printed or saved in the output tables.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
