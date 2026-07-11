# JWST_0025
# Audit: Detailed Schwarzschild zero-axis plot. Matplotlib only. No AI images. Uploads outputs to GitHub.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import getpass
import subprocess
import importlib

VERSION = "JWST_0025"
PROJECT = "DETAILED SCHWARZSCHILD ZERO-AXIS PLOT"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0025_SCHWARZSCHILD_DETAIL"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "requests"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


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
    return getpass.getpass("GitHub token, hidden input: ").strip()


def github_get_sha(token, path):
    import requests
    url = f"https://api.github.com/repos/{REPO}/contents/{path}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(url, headers=headers, params={"ref": BRANCH}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def github_put_file(token, local_path, repo_path):
    import requests
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    sha = github_get_sha(token, repo_path)
    payload = {
        "message": f"{VERSION}: upload {local_path.name}",
        "branch": BRANCH,
        "content": base64.b64encode(local_path.read_bytes()).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    url = f"https://api.github.com/repos/{REPO}/contents/{repo_path}"
    r = requests.put(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    raw = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{repo_path}"
    return raw, "updated" if sha else "created"


def print_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def dark_axes(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.65)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def save_schwarzschild_data():
    import numpy as np
    import pandas as pd
    r = np.linspace(0.0, 15.0, 3001)
    outside = r >= 1.0
    metric = np.full_like(r, np.nan, dtype=float)
    clock = np.full_like(r, np.nan, dtype=float)
    tidal = np.full_like(r, np.nan, dtype=float)
    metric[outside] = 1.0 - 1.0 / r[outside]
    clock[outside] = np.sqrt(metric[outside])
    tidal[outside] = 1.0 / r[outside]**3
    metric[0] = np.nan
    clock[0] = np.nan
    tidal[0] = np.nan
    region = np.where(r == 0, "center singularity", np.where(r < 1, "inside horizon", np.where(r == 1, "event horizon", "outside horizon")))
    df = pd.DataFrame({
        "r_over_rs": r,
        "region": region,
        "clock_rate_fraction_sqrt_1_minus_rs_over_r": clock,
        "metric_factor_1_minus_rs_over_r": metric,
        "tidal_scale_rs_over_r_cubed": tidal,
    })
    csv = CSV / f"{VERSION}_SCHWARZSCHILD_DETAIL_CURVES.csv"
    df.to_csv(csv, index=False)
    return df, csv


def make_detail_plot(df):
    import numpy as np
    import matplotlib.pyplot as plt

    r = df["r_over_rs"].to_numpy()
    clock = df["clock_rate_fraction_sqrt_1_minus_rs_over_r"].to_numpy()
    metric = df["metric_factor_1_minus_rs_over_r"].to_numpy()
    tidal = df["tidal_scale_rs_over_r_cubed"].to_numpy()

    fig, ax = plt.subplots(figsize=(15.4, 8.4))
    dark_axes(fig, ax)

    ax.axvspan(0, 1, color="#7f1d1d", alpha=0.30, label="inside black-hole event horizon")
    ax.axvline(0, color="#e5e7eb", linewidth=1.4, linestyle=":", label="center r/rs = 0")
    ax.axvline(1, color="#facc15", linewidth=2.2, linestyle="--", label="event horizon r/rs = 1")
    ax.axhline(1, color="#38bdf8", linewidth=1.7, linestyle=(0, (8, 4)), label="clock-rate asymptote y = 1.000")
    ax.axhline(0, color="#94a3b8", linewidth=0.9, linestyle="--", alpha=0.8)

    ax.plot(r, clock, color="#38bdf8", linewidth=2.9, label=r"clock rate $\sqrt{1-r_s/r}$")
    ax.plot(r, metric, color="#f97316", linewidth=2.1, label=r"metric factor $1-r_s/r$")
    ax.plot(r, tidal, color="#f43f5e", linewidth=2.0, label=r"tidal scale $(r_s/r)^3$")
    ax.scatter([1], [0], s=42, color="#facc15", edgecolor="#050712", zorder=5)

    note = (
        "x-axis is normalized radius r/rs\n"
        "center: r/rs = 0\n"
        "event horizon: r/rs = 1\n"
        "exterior: r/rs > 1\n"
        "blue asymptote: y = 1.000"
    )
    ax.text(9.1, 0.23, note, color="#f8fafc", fontsize=10.2,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#020617", edgecolor="#475569", alpha=0.92))

    ax.annotate("center / singularity\nr/rs = 0", xy=(0, 0.72), xytext=(1.9, 0.82),
                color="#e5e7eb", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#e5e7eb", lw=1.1))
    ax.annotate("black-hole event horizon\nr/rs = 1\nclock rate = 0", xy=(1, 0), xytext=(2.25, 0.10),
                color="#facc15", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.3))
    ax.annotate("far from the black hole\nclock tends to normal rate\ny → 1.000", xy=(12.2, 1.0), xytext=(7.2, 0.92),
                color="#38bdf8", fontsize=10,
                arrowprops=dict(arrowstyle="->", color="#38bdf8", lw=1.2))

    ax.set_xlim(0, 15)
    ax.set_ylim(-0.045, 1.085)
    ax.set_xlabel(r"normalized Schwarzschild radius, $r/r_s$")
    ax.set_ylabel("normalized exterior quantity")
    ax.set_title("JWST_0025 — Schwarzschild solution with center, event horizon, and clock-rate asymptote")

    leg = ax.legend(loc="lower right", fontsize=8.4, facecolor="#020617", edgecolor="#475569")
    for text in leg.get_texts():
        text.set_color("#f8fafc")

    fig.tight_layout()
    png = PNG / f"{VERSION}_SCHWARZSCHILD_DETAIL_ZERO_AXIS.png"
    fig.savefig(png, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return png


def make_table_png():
    import pandas as pd
    import matplotlib.pyplot as plt

    rows = []
    for r in [0, 0.5, 1, 1.25, 2, 3, 10, 15]:
        if r == 0:
            region = "center"
            clock = metric = tidal = "undefined"
        elif r < 1:
            region = "inside horizon"
            clock = metric = tidal = "no static hover"
        else:
            region = "horizon" if r == 1 else "outside"
            metric_v = 1 - 1 / r
            clock_v = metric_v ** 0.5
            tidal_v = 1 / r ** 3
            clock = f"{clock_v:.6f}"
            metric = f"{metric_v:.6f}"
            tidal = f"{tidal_v:.6f}"
        rows.append([f"{r:.2f}", region, clock, metric, tidal])
    cols = ["r/rs", "region", "clock rate", "metric", "tidal scale"]
    df = pd.DataFrame(rows, columns=cols)
    csv = CSV / f"{VERSION}_SCHWARZSCHILD_KEY_VALUES.csv"
    df.to_csv(csv, index=False)

    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title("JWST_0025 — Schwarzschild key values", color="#f8fafc", fontsize=15, pad=14)
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.55)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#020617" if row % 2 else "#0f172a")
            cell.get_text().set_color("#dbeafe")
            if df.iloc[row-1, 1] == "horizon":
                cell.set_facecolor("#422006")
                cell.get_text().set_color("#fde68a")
    fig.tight_layout()
    png = PNG / f"{VERSION}_SCHWARZSCHILD_KEY_VALUES_TABLE.png"
    fig.savefig(png, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return png, csv


def upload_outputs(paths):
    token = get_token()
    rows = []
    for local_path in paths:
        sub = "PNG" if local_path.suffix.lower() == ".png" else "CSV"
        repo_path = f"{DEST_ROOT}/{sub}/{local_path.name}"
        raw, status = github_put_file(token, local_path, repo_path)
        rows.append({"type": sub.lower(), "status": status, "file": local_path.name, "raw_url": raw})
    links_csv = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    links_json = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"
    import pandas as pd
    pd.DataFrame(rows).to_csv(links_csv, index=False)
    links_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Upload the link tables too, but do not recurse endlessly.
    for p in [links_csv, links_json]:
        repo_path = f"{DEST_ROOT}/CSV/{p.name}"
        raw, status = github_put_file(token, p, repo_path)
        rows.append({"type": "csv" if p.suffix == ".csv" else "json", "status": status, "file": p.name, "raw_url": raw})
    return rows


def main():
    setup()
    df, curves_csv = save_schwarzschild_data()
    detail_png = make_detail_plot(df)
    table_png, values_csv = make_table_png()
    uploaded = upload_outputs([detail_png, table_png, curves_csv, values_csv])

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Plot type", "Python/matplotlib only"),
        ("x-axis starts at", "r/rs = 0, black-hole center"),
        ("vertical dashed line", "r/rs = 1, black-hole event horizon"),
        ("blue asymptote", "y = 1.000, far-away clock-rate limit"),
        ("GitHub destination", DEST_ROOT),
    ], ["Field", "Value"])
    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"]) for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    print("\nLOCAL FILES")
    print_table([
        ("png", str(detail_png)),
        ("png", str(table_png)),
        ("csv", str(curves_csv)),
        ("csv", str(values_csv)),
    ], ["Type", "Path"])
    print("\nCOMMENTS")
    print("The blue clock-rate curve is exterior-only. Inside r/rs < 1, a stationary hovering observer is not physically allowed.")
    print("The asymptote is y = 1.000, meaning the local clock approaches the far-away observer clock rate far from the black hole.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
