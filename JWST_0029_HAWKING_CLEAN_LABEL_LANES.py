# JWST_0029
# Audit: Hawking temperature clean label lanes. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import getpass
import subprocess
import importlib

VERSION = "JWST_0029"
PROJECT = "HAWKING TEMPERATURE CLEAN LABEL LANES"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0029_HAWKING_CLEAN_LABEL_LANES"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

G = 6.67430e-11
C = 299792458.0
HBAR = 1.054571817e-34
KB = 1.380649e-23
MSUN = 1.98847e30
MEARTH = 5.9722e24
AU_KM = 149597870.7
CMB_K = 2.7255
SGRA_MASS_MSUN = 4.10e6
PI = 3.141592653589793


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


def rs_m(mass_kg):
    return 2.0 * G * mass_kg / C**2


def th_k(mass_kg):
    return HBAR * C**3 / (8.0 * PI * G * mass_kg * KB)


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
    return f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{repo_path}", "updated" if sha else "created"


def print_table(rows, headers):
    widths = [max(len(str(h)), *(len(str(row[i])) for row in rows)) for i, h in enumerate(headers)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def dark_axis(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.66)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.2, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def build_data():
    import numpy as np
    import pandas as pd

    mass_msun = np.logspace(-20, 12, 2600)
    mass_kg = mass_msun * MSUN
    curve = pd.DataFrame({
        "mass_Msun": mass_msun,
        "mass_kg": mass_kg,
        "schwarzschild_radius_km": rs_m(mass_kg) / 1000.0,
        "hawking_temperature_K": th_k(mass_kg),
    })

    specs = [
        (1, "Asteroid-mass", 1e15, "theoretical tiny BH"),
        (2, "Earth-mass", MEARTH, "hypothetical BH"),
        (3, "1 solar mass", MSUN, "stellar reference"),
        (4, "10 solar masses", 10.0 * MSUN, "stellar BH"),
        (5, "Sagittarius A*", SGRA_MASS_MSUN * MSUN, "Milky Way center"),
        (6, "1 billion solar masses", 1.0e9 * MSUN, "supermassive BH"),
    ]
    rows = []
    for n, label, kg, note in specs:
        r_km = rs_m(kg) / 1000.0
        temp = th_k(kg)
        rows.append({
            "n": n,
            "label": label,
            "mass_kg": kg,
            "mass_Msun": kg / MSUN,
            "schwarzschild_radius_km": r_km,
            "schwarzschild_radius_AU": r_km / AU_KM,
            "hawking_temperature_K": temp,
            "log10_T_K": np.log10(temp),
            "T_over_CMB": temp / CMB_K,
            "note": note,
        })
    anchors = pd.DataFrame(rows)
    curve_csv = CSV / f"{VERSION}_TEMPERATURE_VS_RADIUS_CURVE.csv"
    anchors_csv = CSV / f"{VERSION}_ANCHOR_VALUES.csv"
    curve.to_csv(curve_csv, index=False)
    anchors.to_csv(anchors_csv, index=False)
    return curve, anchors, curve_csv, anchors_csv


def plot_clean_lanes(curve, anchors):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(17.6, 9.2))
    fig.patch.set_facecolor("#050712")
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.35, 1.15], wspace=0.04)
    ax = fig.add_subplot(gs[0, 0])
    lane = fig.add_subplot(gs[0, 1])
    dark_axis(fig, ax)
    lane.set_facecolor("#050712")
    lane.axis("off")

    ax.loglog(curve["schwarzschild_radius_km"], curve["hawking_temperature_K"],
              color="#67e8f9", linewidth=2.8, label=r"$T_H \propto 1/r_s$")
    ax.axhline(CMB_K, color="#facc15", linestyle="--", linewidth=1.55, label=f"CMB today = {CMB_K:.4f} K")

    colors = ["#f43f5e", "#f97316", "#a78bfa", "#38bdf8", "#22c55e", "#eab308"]
    for (_, row), color in zip(anchors.iterrows(), colors):
        x = row["schwarzschild_radius_km"]
        y = row["hawking_temperature_K"]
        ax.scatter([x], [y], s=96, color=color, edgecolor="#f8fafc", linewidth=0.8, zorder=6)
        ax.text(x, y, f" {int(row['n'])}", color="#f8fafc", fontsize=10.5, weight="bold",
                ha="left", va="bottom", zorder=7)

    ax.set_xlabel("Schwarzschild radius, km")
    ax.set_ylabel("Hawking temperature, K")
    ax.set_title("JWST_0029 — Clean Hawking temperature map: larger horizon radius means colder black hole")
    legend(ax, "upper right")

    lane.text(0.02, 0.97, "Numbered label lane", color="#f8fafc", fontsize=15, weight="bold", va="top")
    lane.text(0.02, 0.925, "Labels moved off the curve to keep the data clean.", color="#cbd5e1", fontsize=9.5, va="top")
    y0 = 0.84
    dy = 0.124
    for i, (_, row) in enumerate(anchors.iterrows()):
        color = colors[i]
        y = y0 - i * dy
        lane.text(0.02, y, f"{int(row['n'])}", color="#050712", fontsize=10.5, weight="bold",
                  bbox=dict(boxstyle="circle,pad=0.25", facecolor=color, edgecolor="#f8fafc", linewidth=0.8))
        lane.text(0.13, y + 0.018, row["label"], color=color, fontsize=11.2, weight="bold", va="center")
        lane.text(0.13, y - 0.018,
                  f"r_s={row['schwarzschild_radius_km']:.2e} km   T={row['hawking_temperature_K']:.2e} K",
                  color="#dbeafe", fontsize=8.6, va="center")
        lane.text(0.13, y - 0.050,
                  f"M={row['mass_Msun']:.2e} M_sun   {row['note']}",
                  color="#94a3b8", fontsize=8.0, va="center")

    lane.text(0.02, 0.055,
              "Read: T_H = ħc/(4πk_B r_s).\nDouble r_s → half the Hawking temperature.",
              color="#f8fafc", fontsize=10.2,
              bbox=dict(boxstyle="round,pad=0.45", facecolor="#020617", edgecolor="#475569", alpha=0.92))

    fig.tight_layout()
    path = PNG / f"{VERSION}_HAWKING_CLEAN_LABEL_LANES.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_mass_temperature(anchors):
    import matplotlib.pyplot as plt
    import numpy as np

    mass = np.logspace(-20, 12, 1800)
    temp = th_k(mass * MSUN)
    fig, ax = plt.subplots(figsize=(14.4, 8.0))
    dark_axis(fig, ax)
    ax.loglog(mass, temp, color="#38bdf8", linewidth=2.8, label=r"$T_H \propto 1/M$")
    ax.axhline(CMB_K, color="#facc15", linestyle="--", linewidth=1.55, label="CMB today")
    for _, row in anchors.iterrows():
        ax.scatter([row["mass_Msun"]], [row["hawking_temperature_K"]], s=74, color="#f97316", edgecolor="#f8fafc", zorder=5)
        ax.text(row["mass_Msun"], row["hawking_temperature_K"], f" {int(row['n'])}", color="#f8fafc", fontsize=10, weight="bold")
    ax.set_xlabel("Black-hole mass, solar masses")
    ax.set_ylabel("Hawking temperature, K")
    ax.set_title("Same physics by mass: more massive black holes are colder")
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_TEMPERATURE_VS_MASS_NUMBERED.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_table_png(anchors):
    import matplotlib.pyplot as plt

    view = anchors[["n", "label", "mass_Msun", "schwarzschild_radius_km", "schwarzschild_radius_AU", "hawking_temperature_K", "T_over_CMB"]].copy()
    view["n"] = view["n"].map(lambda x: f"{int(x)}")
    for col in ["mass_Msun", "schwarzschild_radius_km", "schwarzschild_radius_AU", "hawking_temperature_K", "T_over_CMB"]:
        view[col] = view[col].map(lambda x: f"{x:.3e}")
    view.columns = ["#", "object", "mass M_sun", "r_s km", "r_s AU", "T_H K", "T_H/CMB"]

    fig, ax = plt.subplots(figsize=(16.2, 5.0))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title("JWST_0029 — Hawking temperature anchor table", color="#f8fafc", fontsize=16, pad=14)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.0)
    table.scale(1, 1.58)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#475569")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor("#1e293b")
            cell.get_text().set_color("#f8fafc")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#020617" if r % 2 else "#0f172a")
            cell.get_text().set_color("#dbeafe")
            if view.iloc[r-1, 1] == "Sagittarius A*":
                cell.set_facecolor("#052e16")
                cell.get_text().set_color("#bbf7d0")
    fig.tight_layout()
    path = PNG / f"{VERSION}_CLEAN_LABEL_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def upload_outputs(paths):
    token = get_token()
    rows = []
    for local_path in paths:
        sub = "PNG" if local_path.suffix.lower() == ".png" else "CSV"
        repo_path = f"{DEST_ROOT}/{sub}/{local_path.name}"
        raw, status = github_put_file(token, local_path, repo_path)
        rows.append({"type": sub.lower(), "status": status, "file": local_path.name, "raw_url": raw})
    import pandas as pd
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
        print(r["raw_url"])
    try:
        from IPython.display import HTML, display
        links = "".join(f'<li><a href="{r["raw_url"]}" target="_blank">{r["file"]}</a><br><code>{r["raw_url"]}</code></li>' for r in image_rows)
        previews = "".join(f'<h4 style="color:#e5e7eb">{r["file"]}</h4><a href="{r["raw_url"]}" target="_blank"><img src="{r["raw_url"]}" style="max-width:100%;border:1px solid #475569;border-radius:8px"></a>' for r in image_rows)
        display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Images available here</h2><p>Clean numbered labels plus a right-side label lane.</p><ul>{links}</ul><hr style="border-color:#334155">{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    curve, anchors, curve_csv, anchors_csv = build_data()
    png1 = plot_clean_lanes(curve, anchors)
    png2 = plot_mass_temperature(anchors)
    png3 = plot_table_png(anchors)
    uploaded = upload_outputs([png1, png2, png3, curve_csv, anchors_csv])
    sgra = anchors[anchors["label"] == "Sagittarius A*"].iloc[0]
    earth = anchors[anchors["label"] == "Earth-mass"].iloc[0]

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Read", "larger black hole -> larger r_s -> colder Hawking temperature"),
        ("Formula", "T_H = hbar*c/(4*pi*kB*r_s) = hbar*c^3/(8*pi*G*M*kB)"),
        ("Sgr A* T_H K", f"{sgra['hawking_temperature_K']:.6e}"),
        ("Sgr A* r_s km", f"{sgra['schwarzschild_radius_km']:.6e}"),
        ("Earth-mass T_H K", f"{earth['hawking_temperature_K']:.6e}"),
        ("Earth-mass log10(T_H)", f"{earth['log10_T_K']:.6f}"),
    ], ["Field", "Value"])
    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"]) for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)
    print("\nCOMMENTS")
    print("Tiny black holes are hot; astrophysical black holes are cold.")
    print("The plot compares different black-hole sizes. It is not a temperature map from the horizon to the center of one black hole.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
