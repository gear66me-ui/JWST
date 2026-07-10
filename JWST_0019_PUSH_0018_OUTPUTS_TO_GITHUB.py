# JWST_0019
# Audit: Push JWST_0018 PNG/CSV outputs to GitHub. No image generation.

from pathlib import Path
from datetime import datetime, timezone
import base64
import json
import os
import urllib.error
import urllib.request

VERSION = "JWST_0019"
OWNER = "gear66me-ui"
REPO = "JWST"
BRANCH = "main"
LOCAL_ROOT = Path("/content/JWST_OUTPUT")
DEST_ROOT = "OUTPUTS/JWST_0018_SCHECHTER_NORMALIZATION"

FILES = [
    ("PNG/JWST_0018_SCHECHTER_NORMALIZATION_DASHBOARD.png", f"{DEST_ROOT}/PNG/JWST_0018_SCHECHTER_NORMALIZATION_DASHBOARD.png"),
    ("PNG/JWST_0018_HORIZON_R3_VOLUME_GROWTH.png", f"{DEST_ROOT}/PNG/JWST_0018_HORIZON_R3_VOLUME_GROWTH.png"),
    ("CSV/JWST_0018_NORMALIZATION_CURVE.csv", f"{DEST_ROOT}/CSV/JWST_0018_NORMALIZATION_CURVE.csv"),
    ("CSV/JWST_0018_ANCHOR_VALUES.csv", f"{DEST_ROOT}/CSV/JWST_0018_ANCHOR_VALUES.csv"),
    ("CSV/JWST_0018_MODEL_NOTES.csv", f"{DEST_ROOT}/CSV/JWST_0018_MODEL_NOTES.csv"),
]


def get_token():
    token = os.environ.get("GITHUB_TOKEN")
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
    print("GitHub token not found in Colab Secrets as GITHUB_TOKEN.")
    print("Paste a fine-grained token with Contents: Read and write for gear66me-ui/JWST.")
    return getpass.getpass("GitHub token: ").strip()


def api_request(method, url, token, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"message": body}
        return e.code, parsed


def existing_sha(token, repo_path):
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{repo_path}?ref={BRANCH}"
    status, payload = api_request("GET", url, token)
    if status == 200:
        return payload.get("sha")
    if status == 404:
        return None
    raise RuntimeError(f"GitHub GET failed for {repo_path}: {status} {payload}")


def upload_file(token, local_path, repo_path):
    data = local_path.read_bytes()
    sha = existing_sha(token, repo_path)
    payload = {
        "message": f"Upload {repo_path}",
        "content": base64.b64encode(data).decode("ascii"),
        "branch": BRANCH,
    }
    if sha:
        payload["sha"] = sha
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{repo_path}"
    status, reply = api_request("PUT", url, token, payload)
    if status not in (200, 201):
        raise RuntimeError(f"GitHub PUT failed for {repo_path}: {status} {reply}")
    return "updated" if status == 200 else "created", len(data)


def print_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths)-1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    print(f"CODE OUTPUT: {VERSION}\n")
    token = get_token()
    rows = []
    links = []
    missing = []

    for rel, dest in FILES:
        src = LOCAL_ROOT / rel
        if not src.exists():
            missing.append(str(src))
            continue
        print(f"Uploading: {src}")
        status, size = upload_file(token, src, dest)
        rows.append((status, size, dest))
        links.append((src.name, f"https://raw.githubusercontent.com/{OWNER}/{REPO}/{BRANCH}/{dest}"))

    print("\nUPLOAD SUMMARY")
    print_table(rows, ["status", "bytes", "repo_path"])

    if missing:
        print("\nMISSING LOCAL FILES")
        for item in missing:
            print(item)
        print("Run JWST_0018 first, then rerun this exporter.")

    print("\nRAW GITHUB LINKS")
    print_table(links, ["file", "raw_url"])

    out_csv = LOCAL_ROOT / "CSV" / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    out_json = LOCAL_ROOT / "CSV" / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv.write_text("file,raw_url\n" + "\n".join(f"{a},{b}" for a, b in links) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps([{"file": a, "raw_url": b} for a, b in links], indent=2), encoding="utf-8")

    print("\nLOCAL LINK FILES")
    print_table([("csv", out_csv), ("json", out_json)], ["type", "path"])
    print("\nPaste the raw PNG/CSV links into ChatGPT so we can discuss the dashboard directly.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
