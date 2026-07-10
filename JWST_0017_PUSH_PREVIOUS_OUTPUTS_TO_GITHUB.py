# JWST_0017_PUSH_PREVIOUS_OUTPUTS_TO_GITHUB.py
# Upload the previous JWST_0016 Colab output PNG/CSV files into the JWST GitHub repository.
# No AI images. No plotting. No FITS/image downloads.

from pathlib import Path
from datetime import datetime
import base64
import getpass
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

VERSION = "JWST_0017_PUSH_PREVIOUS_OUTPUTS_TO_GITHUB"
REPO_OWNER = "gear66me-ui"
REPO_NAME = "JWST"
BRANCH = "main"
REPO_FULL = f"{REPO_OWNER}/{REPO_NAME}"
API_ROOT = f"https://api.github.com/repos/{REPO_FULL}/contents"
RAW_ROOT = f"https://raw.githubusercontent.com/{REPO_OWNER}/{REPO_NAME}/{BRANCH}"

SOURCE_FILES = [
    "/content/JWST_OUTPUT/PNG/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK.png",
    "/content/JWST_OUTPUT/PNG/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK_DISTANCE_AGE_CURVES.png",
    "/content/JWST_OUTPUT/PNG/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK_SUMMARY_TABLE.png",
    "/content/JWST_OUTPUT/CSV/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK_REDSHIFT_SAMPLE.csv",
    "/content/JWST_OUTPUT/CSV/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK_REDSHIFT_CURVE.csv",
    "/content/JWST_OUTPUT/CSV/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK_COSMOLOGY_CURVE.csv",
    "/content/JWST_OUTPUT/CSV/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK_AUDIT.csv",
]

DEST_ROOT = "OUTPUTS/JWST_0016_FAST_REDSHIFT_CURVE_FALLBACK"
LOCAL_LINKS_CSV = Path("/content/JWST_OUTPUT/CSV/JWST_0017_GITHUB_OUTPUT_LINKS.csv")
LOCAL_LINKS_JSON = Path("/content/JWST_OUTPUT/CSV/JWST_0017_GITHUB_OUTPUT_LINKS.json")


def get_github_token():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    try:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        if token:
            return str(token).strip()
    except Exception:
        pass

    print("GitHub token not found in Colab Secrets as GITHUB_TOKEN or environment variable GITHUB_TOKEN.")
    print("Paste a GitHub fine-grained token with Contents: Read and write for gear66me-ui/JWST.")
    print("The token input is hidden by getpass and is not printed.")
    return getpass.getpass("GitHub token: ").strip()


def request_json(method, url, token, payload=None):
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": VERSION,
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            text = response.read().decode("utf-8")
            if not text:
                return {}
            return json.loads(text)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code == 404:
            return {"_not_found": True, "status": 404, "body": body}
        raise RuntimeError(f"GitHub API HTTP {exc.code}: {body[:500]}") from exc


def repo_path_for_local(path):
    p = Path(path)
    if "/PNG/" in path:
        return f"{DEST_ROOT}/PNG/{p.name}"
    if "/CSV/" in path:
        return f"{DEST_ROOT}/CSV/{p.name}"
    return f"{DEST_ROOT}/{p.name}"


def content_api_url(repo_path):
    quoted = urllib.parse.quote(repo_path, safe="/")
    return f"{API_ROOT}/{quoted}?ref={urllib.parse.quote(BRANCH)}"


def upload_file(local_path, repo_path, token):
    local = Path(local_path)
    if not local.exists():
        return {
            "local_path": str(local),
            "repo_path": repo_path,
            "status": "missing_local_file",
            "bytes": 0,
            "raw_url": "",
        }

    get_url = content_api_url(repo_path)
    existing = request_json("GET", get_url, token)
    sha = None if existing.get("_not_found") else existing.get("sha")

    content_b64 = base64.b64encode(local.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(str(local))[0] or "application/octet-stream"
    payload = {
        "message": f"Upload {VERSION} output {local.name}",
        "content": content_b64,
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha

    put_url = f"{API_ROOT}/{urllib.parse.quote(repo_path, safe='/')}"
    result = request_json("PUT", put_url, token, payload=payload)
    raw_url = f"{RAW_ROOT}/{urllib.parse.quote(repo_path, safe='/')}"
    return {
        "local_path": str(local),
        "repo_path": repo_path,
        "status": "updated" if sha else "created",
        "bytes": local.stat().st_size,
        "mime_type": mime_type,
        "raw_url": raw_url,
        "commit_sha": result.get("commit", {}).get("sha", ""),
    }


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


def write_link_files(results):
    import csv
    LOCAL_LINKS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["status", "bytes", "mime_type", "local_path", "repo_path", "raw_url", "commit_sha"]
    with LOCAL_LINKS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fields})
    LOCAL_LINKS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print()
    token = get_github_token()
    if not token:
        raise RuntimeError("No GitHub token provided.")

    results = []
    for source in SOURCE_FILES:
        repo_path = repo_path_for_local(source)
        print(f"Uploading: {source}")
        results.append(upload_file(source, repo_path, token))

    write_link_files(results)

    print_table(
        "UPLOAD SUMMARY",
        [(r.get("status", ""), r.get("bytes", 0), r.get("repo_path", "")) for r in results],
        ["status", "bytes", "repo_path"],
    )

    print_table(
        "RAW GITHUB LINKS",
        [(Path(r.get("repo_path", "")).name, r.get("raw_url", "")) for r in results if r.get("raw_url")],
        ["file", "raw_url"],
    )

    print_table(
        "LOCAL LINK FILES",
        [("csv", str(LOCAL_LINKS_CSV)), ("json", str(LOCAL_LINKS_JSON))],
        ["type", "path"],
    )

    print("Copy the raw PNG/CSV links into ChatGPT so we can discuss the charts directly.")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
