# JWST_0030
# Audit: GitHub JWST repository harvest widget. Python/matplotlib only. No AI images. No FITS downloads.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import math
import time
import base64
import getpass
import subprocess
import importlib

VERSION = "JWST_0030"
PROJECT = "GITHUB JWST REPOSITORY HARVEST WIDGET"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0030_GITHUB_JWST_REPO_HARVEST"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

REPO_QUERIES = [
    ("official_pipeline", "JWST python"),
    ("official_pipeline", "org:spacetelescope jwst python"),
    ("notebooks", "JWST notebook"),
    ("notebooks", "JDAT notebooks JWST"),
    ("pipeline_notebooks", "JWST pipeline notebooks"),
    ("validation", "JWST validation notebooks"),
    ("mast_query", "JWST MAST query"),
    ("mast_query", "jwst_mast_query"),
    ("miri", "JWST MIRI notebook"),
    ("miri", "MIRI ExampleNB JWST"),
    ("nircam", "JWST NIRCam python"),
    ("nirspec", "JWST NIRSpec python"),
    ("niriss", "JWST NIRISS python"),
    ("psf", "JWST PSF webbpsf"),
    ("psf", "STPSF JWST"),
    ("psf", "pynrc JWST NIRCam"),
    ("exoplanet", "JWST exoplanet spectroscopy python"),
    ("exoplanet", "Eureka JWST"),
    ("nearby_galaxies", "PHANGS JWST python"),
    ("footprints", "JWST footprints"),
    ("templates", "JWST templates notebooks"),
    ("etc", "JWST ETC Pandeia"),
    ("imaging", "JWST image processing python"),
    ("spectra", "JWST spectral analysis python"),
]

CODE_QUERIES = [
    "astroquery.mast JWST language:Python",
    "Observations.query_criteria JWST language:Python",
    "jwst.datamodels language:Python",
    "jwst.pipeline language:Python",
    "Detector1Pipeline language:Python",
    "Image2Pipeline language:Python",
    "Spec2Pipeline language:Python",
    "instrument_name JWST language:Python",
]

OFFICIAL_OWNERS = {"spacetelescope", "STScI-MIRI", "JWST-Templates", "PhangsTeam"}


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["requests", "pandas", "numpy", "matplotlib"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def get_token(optional=True):
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
    if optional:
        return ""
    return getpass.getpass("GitHub token, hidden input: ").strip()


def gh_headers(token):
    h = {"Accept": "application/vnd.github+json", "User-Agent": f"{VERSION}-harvest"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def gh_get(url, token, params=None, tries=3):
    import requests
    headers = gh_headers(token)
    last = None
    for attempt in range(tries):
        r = requests.get(url, headers=headers, params=params or {}, timeout=45)
        last = r
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset = r.headers.get("X-RateLimit-Reset")
            if reset:
                wait = max(3, min(60, int(reset) - int(time.time()) + 2))
                time.sleep(wait)
                continue
        if r.status_code in (500, 502, 503, 504):
            time.sleep(2 + attempt)
            continue
        return r
    return last


def search_repositories(token):
    rows = {}
    errors = []
    for category, query in REPO_QUERIES:
        params = {"q": query, "per_page": 12, "page": 1}
        r = gh_get("https://api.github.com/search/repositories", token, params=params)
        if not r or r.status_code != 200:
            errors.append({"type": "repo", "query": query, "status": getattr(r, "status_code", None), "message": getattr(r, "text", "")[:220]})
            continue
        for item in r.json().get("items", []):
            full = item.get("full_name", "")
            if not full:
                continue
            owner = item.get("owner", {}).get("login", "")
            previous = rows.get(full, {})
            cats = set(str(previous.get("matched_categories", "")).split(";")) if previous else set()
            cats.discard("")
            cats.add(category)
            queries = set(str(previous.get("matched_queries", "")).split(" || ")) if previous else set()
            queries.discard("")
            queries.add(query)
            rows[full] = {
                "repository": full,
                "owner": owner,
                "name": item.get("name", ""),
                "description": (item.get("description") or "").replace("\n", " ")[:420],
                "html_url": item.get("html_url", ""),
                "clone_url": item.get("clone_url", ""),
                "default_branch": item.get("default_branch", ""),
                "language": item.get("language") or "",
                "stars": item.get("stargazers_count", 0) or 0,
                "forks": item.get("forks_count", 0) or 0,
                "watchers": item.get("watchers_count", 0) or 0,
                "open_issues": item.get("open_issues_count", 0) or 0,
                "size_kb": item.get("size", 0) or 0,
                "archived": item.get("archived", False),
                "updated_at": item.get("updated_at", ""),
                "created_at": item.get("created_at", ""),
                "matched_categories": ";".join(sorted(cats)),
                "matched_queries": " || ".join(sorted(queries)),
            }
        time.sleep(0.15)
    return list(rows.values()), errors


def search_code(token):
    rows = []
    errors = []
    if not token:
        errors.append({"type": "code", "query": "all", "status": "skipped", "message": "GitHub token not available; code search skipped."})
        return rows, errors
    seen = set()
    for query in CODE_QUERIES:
        params = {"q": query, "per_page": 10, "page": 1}
        r = gh_get("https://api.github.com/search/code", token, params=params)
        if not r or r.status_code != 200:
            errors.append({"type": "code", "query": query, "status": getattr(r, "status_code", None), "message": getattr(r, "text", "")[:240]})
            continue
        for item in r.json().get("items", []):
            repo = item.get("repository", {}).get("full_name", "")
            path = item.get("path", "")
            key = (repo, path)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "repository": repo,
                "path": path,
                "name": item.get("name", ""),
                "html_url": item.get("html_url", ""),
                "matched_query": query,
            })
        time.sleep(0.25)
    return rows, errors


def classify(row):
    full = row["repository"].lower()
    desc = row["description"].lower()
    cats = row["matched_categories"].lower()
    if row["repository"] == "spacetelescope/jwst":
        return "official pipeline"
    if "jdat_notebooks" in full:
        return "science notebooks"
    if "pipeline-notebooks" in full:
        return "pipeline notebooks"
    if "validation" in full:
        return "validation notebooks"
    if "eureka" in full:
        return "exoplanet spectra"
    if "phangs" in full or "pjpipe" in full:
        return "nearby galaxies"
    if "miri" in full or "miri" in desc or "miri" in cats:
        return "MIRI examples"
    if "nircam" in desc or "pynrc" in full:
        return "NIRCam / PSF"
    if "webbpsf" in full or "stpsf" in full or "psf" in desc:
        return "PSF modeling"
    if "mast" in full or "mast" in desc:
        return "MAST search"
    if "footprint" in full or "footprint" in desc:
        return "observing footprints"
    if "pandeia" in full or "etc" in full:
        return "exposure calculator"
    if "image" in full or "imaging" in full:
        return "image processing"
    return "general JWST"


def score(row):
    owner = row["owner"]
    stars = float(row.get("stars", 0))
    forks = float(row.get("forks", 0))
    size = float(row.get("size_kb", 0))
    full = row["repository"].lower()
    desc = row["description"].lower()
    cats = row["matched_categories"].lower()
    s = 0.0
    s += 7.0 * math.log10(stars + 1.0)
    s += 3.0 * math.log10(forks + 1.0)
    s += 1.0 * math.log10(size + 1.0)
    if owner in OFFICIAL_OWNERS:
        s += 12.0
    if "spacetelescope/jwst" == row["repository"]:
        s += 18.0
    if any(k in full for k in ["jdat", "pipeline-notebooks", "validation", "eureka", "webbpsf", "stpsf", "pynrc", "pjpipe", "jwst_mast_query"]):
        s += 8.0
    if any(k in desc for k in ["notebook", "pipeline", "jwst", "mast", "nircam", "miri", "spectroscopy"]):
        s += 4.0
    if "notebooks" in cats or "pipeline" in cats or "mast" in cats:
        s += 3.0
    if row.get("archived"):
        s -= 8.0
    return round(s, 3)


def enrich(repos):
    import pandas as pd
    df = pd.DataFrame(repos)
    if df.empty:
        return df
    df["category"] = df.apply(classify, axis=1)
    df["interest_score"] = df.apply(score, axis=1)
    df = df.sort_values(["interest_score", "stars", "forks"], ascending=[False, False, False]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def dark_axis(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, axis="x", color="#334155", linewidth=0.55, alpha=0.68)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def make_repo_bar(df):
    import matplotlib.pyplot as plt
    if df.empty:
        return None
    top = df.head(16).iloc[::-1].copy()
    labels = [r.replace("spacetelescope/", "stsci/").replace("PhangsTeam/", "PHANGS/") for r in top["repository"]]
    fig, ax = plt.subplots(figsize=(15.8, 9.2))
    dark_axis(fig, ax)
    ax.barh(labels, top["interest_score"])
    for y, (_, row) in enumerate(top.iterrows()):
        txt = f" {row['category']} | ★ {int(row['stars'])} | forks {int(row['forks'])}"
        ax.text(row["interest_score"] + 0.4, y, txt, color="#dbeafe", va="center", fontsize=8.4)
    ax.set_xlabel("interest score: official status + stars + forks + relevance")
    ax.set_title("JWST_0030 — GitHub JWST Python repositories worth inspecting first")
    fig.tight_layout()
    path = PNG / f"{VERSION}_TOP_GITHUB_REPOS.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def make_category_bar(df):
    import matplotlib.pyplot as plt
    if df.empty:
        return None
    counts = df.groupby("category", as_index=False).agg(repos=("repository", "count"), stars=("stars", "sum"), max_score=("interest_score", "max"))
    counts = counts.sort_values(["max_score", "repos"], ascending=[False, False]).head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    dark_axis(fig, ax)
    ax.barh(counts["category"], counts["repos"])
    for y, (_, row) in enumerate(counts.iterrows()):
        ax.text(row["repos"] + 0.05, y, f" {int(row['repos'])} repos | ★ total {int(row['stars'])}", color="#dbeafe", va="center", fontsize=8.6)
    ax.set_xlabel("repository count in harvest")
    ax.set_title("JWST_0030 — Harvest by useful science/programming category")
    fig.tight_layout()
    path = PNG / f"{VERSION}_CATEGORY_COUNTS.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def make_table_png(df):
    import matplotlib.pyplot as plt
    if df.empty:
        return None
    view = df.head(18)[["rank", "repository", "category", "stars", "forks", "interest_score", "updated_at"]].copy()
    view["updated_at"] = view["updated_at"].astype(str).str.replace("T", " ").str.replace("Z", "").str.slice(0, 10)
    view.columns = ["rank", "repository", "category", "stars", "forks", "score", "updated"]
    fig, ax = plt.subplots(figsize=(17.5, 8.2))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title("JWST_0030 — Top GitHub JWST Python targets", color="#f8fafc", fontsize=16, pad=14)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            repo = str(view.iloc[r - 1, 1])
            cell.set_facecolor("#020617" if r % 2 else "#0f172a")
            cell.get_text().set_color("#dbeafe")
            if repo.startswith("spacetelescope/") or repo.startswith("STScI-MIRI/"):
                cell.set_facecolor("#0c2d57")
                cell.get_text().set_color("#dbeafe")
            if repo.startswith("PhangsTeam/"):
                cell.set_facecolor("#052e16")
                cell.get_text().set_color("#bbf7d0")
    fig.tight_layout()
    path = PNG / f"{VERSION}_REPO_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def github_get_sha(token, path):
    import requests
    if not token:
        return None
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    r = requests.get(url, headers=gh_headers(token), params={"ref": BRANCH}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def github_put_file(token, local_path, repo_path):
    import requests
    if not token:
        return "", "skipped-no-token"
    sha = github_get_sha(token, repo_path)
    payload = {
        "message": f"{VERSION}: upload {local_path.name}",
        "branch": BRANCH,
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    url = f"https://api.github.com/repos/{REPO}/contents/{repo_path}"
    r = requests.put(url, headers=gh_headers(token), json=payload, timeout=90)
    r.raise_for_status()
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{repo_path}", "updated" if sha else "created"


def upload_outputs(paths, token):
    import pandas as pd
    rows = []
    for local_path in paths:
        if local_path is None:
            continue
        sub = "PNG" if local_path.suffix.lower() == ".png" else "CSV"
        repo_path = f"{DEST_ROOT}/{sub}/{local_path.name}"
        raw, status = github_put_file(token, local_path, repo_path)
        rows.append({"type": sub.lower(), "status": status, "file": local_path.name, "raw_url": raw})
    links_csv = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    links_json = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"
    pd.DataFrame(rows).to_csv(links_csv, index=False)
    links_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for p in [links_csv, links_json]:
        repo_path = f"{DEST_ROOT}/CSV/{p.name}"
        raw, status = github_put_file(token, p, repo_path)
        rows.append({"type": p.suffix.lower().replace(".", ""), "status": status, "file": p.name, "raw_url": raw})
    return rows


def display_links(rows):
    image_rows = [r for r in rows if r["type"] == "png"]
    print("\nIMAGES AVAILABLE HERE:")
    for r in image_rows:
        print(r["file"])
        print(r["raw_url"] if r["raw_url"] else "local only; no upload token available")
    try:
        from IPython.display import HTML, display
        links = "".join(
            f'<li><a href="{r["raw_url"]}" target="_blank">{r["file"]}</a><br><code>{r["raw_url"]}</code></li>'
            for r in image_rows if r["raw_url"]
        )
        previews = "".join(
            f'<h4 style="color:#e5e7eb">{r["file"]}</h4><a href="{r["raw_url"]}" target="_blank"><img src="{r["raw_url"]}" style="max-width:100%;border:1px solid #475569;border-radius:8px"></a>'
            for r in image_rows if r["raw_url"]
        )
        if links:
            display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Images available here</h2><p>GitHub JWST repo harvest dashboards.</p><ul>{links}</ul><hr style="border-color:#334155">{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def print_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    token = get_token(optional=True)
    print(f"CODE OUTPUT: {VERSION}")
    print("Searching GitHub repositories and Python code references...")

    repos, repo_errors = search_repositories(token)
    code_rows, code_errors = search_code(token)
    df = enrich(repos)

    repo_csv = CSV / f"{VERSION}_GITHUB_REPO_HARVEST.csv"
    code_csv = CSV / f"{VERSION}_GITHUB_CODE_HITS.csv"
    errors_csv = CSV / f"{VERSION}_GITHUB_SEARCH_ERRORS.csv"
    df.to_csv(repo_csv, index=False)
    import pandas as pd
    pd.DataFrame(code_rows).to_csv(code_csv, index=False)
    pd.DataFrame(repo_errors + code_errors).to_csv(errors_csv, index=False)

    png1 = make_repo_bar(df)
    png2 = make_category_bar(df)
    png3 = make_table_png(df)

    uploaded = upload_outputs([png1, png2, png3, repo_csv, code_csv, errors_csv], token)

    print("\nSEARCH SUMMARY")
    print_table([
        ("Repository queries", len(REPO_QUERIES)),
        ("Unique repositories", len(df)),
        ("Code queries", len(CODE_QUERIES)),
        ("Unique code hits", len(code_rows)),
        ("Repo CSV", str(repo_csv)),
        ("Code CSV", str(code_csv)),
        ("Errors CSV", str(errors_csv)),
    ], ["Field", "Value"])

    print("\nTOP REPOSITORIES")
    if df.empty:
        print("No repository results returned.")
    else:
        top_rows = []
        for _, row in df.head(12).iterrows():
            top_rows.append((int(row["rank"]), row["repository"], row["category"], int(row["stars"]), int(row["forks"]), f"{row['interest_score']:.2f}", row["html_url"]))
        print_table(top_rows, ["Rank", "Repository", "Category", "Stars", "Forks", "Score", "URL"])

    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"] or "local only") for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)

    print("\nCOMMENTS")
    print("This is a GitHub code/repository harvest only. It does not download FITS files or science images.")
    print("Best first targets are official STScI pipeline/notebook repos, MAST query helpers, MIRI examples, PHANGS/pjpipe, PSF tools, and exoplanet Eureka.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
