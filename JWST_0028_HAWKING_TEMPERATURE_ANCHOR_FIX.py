# JWST_0028
# Audit: Hawking temperature anchor fix. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import getpass
import subprocess
import importlib

VERSION = "JWST_0028"
PROJECT = "HAWKING TEMPERATURE ANCHOR FIX"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0028_HAWKING_TEMPERATURE_ANCHOR_FIX"
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


def th_from_rs_k(radius_m):
    return HBAR * C / (4.0 * PI * KB * radius_m)


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


def dark(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.68)
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
    radius_km = rs_m(mass_kg) / 1000.0
    curve = pd.DataFrame({
        "mass_Msun": mass_msun,
        "mass_kg": mass_kg,
        "schwarzschild_radius_km": radius_km,
        "hawking_temperature_K": th_k(mass_kg),
    })

    anchor_specs = [
        ("Asteroid-mass 1e15 kg", 1e15, "theoretical tiny black hole"),
        ("Earth-mass black hole", MEARTH, "hypothetical; r_s is millimeters"),
        ("1 solar-mass black hole", MSUN, "stellar reference"),
        ("10 solar-mass black hole", 10.0 * MSUN, "stellar black hole"),
        ("Sagittarius A*", SGRA_MASS_MSUN * MSUN, "Milky Way central black hole"),
        ("1 billion solar-mass black hole", 1e9 * MSUN, "supermassive scale"),
    ]
    rows = []
    for name, kg, note in anchor_specs:
        r_km = rs_m(kg) / 1000.0
        t = th_k(kg)
        rows.append({
            "object": name,
            "mass_kg": kg,
            "mass_Msun": kg / MSUN,
            "schwarzschild_radius_km": r_km,
            "schwarzschild_radius_m": r_km * 1000.0,
            "schwarzschild_radius_AU": r_km / AU_KM,
            "hawking_temperature_K": t,
            "log10_T_K": np.log10(t),
            "T_over_CMB": t / CMB_K,
            "note": note,
        })
    anchors = pd.DataFrame(rows)
    curve_csv = CSV / f"{VERSION}_TEMPERATURE_VS_RADIUS_CURVE.csv"
    anchors_csv = CSV / f"{VERSION}_ANCHOR_VALUES.csv"
    curve.to_csv(curve_csv, index=False)
    anchors.to_csv(anchors_csv, index=False)
    return curve, anchors, curve_csv, anchors_csv


def plot_radius_temperature(curve, anchors):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15.2, 8.4))
    dark(fig, ax)
    ax.loglog(curve["schwarzschild_radius_km"], curve["hawking_temperature_K"],
              color="#67e8f9", linewidth=2.8, label="Hawking temperature curve")
    ax.axhline(CMB_K, color="#facc15", linestyle="--", linewidth=1.5, label=f"CMB today = {CMB_K:.4f} K")

    offsets = {
        "Asteroid-mass 1e15 kg": (1.6, 1.4),
        "Earth-mass black hole": (1.7, 1.3),
        "1 solar-mass black hole": (1.5, 1.5),
        "10 solar-mass black hole": (1.5, 0.55),
        "Sagittarius A*": (0.000012, 50000.0),
        "1 billion solar-mass black hole": (0.00008, 90000.0),
    }
    colors = {
        "Asteroid-mass 1e15 kg": "#f43f5e",
        "Earth-mass black hole": "#f97316",
        "1 solar-mass black hole": "#a78bfa",
        "10 solar-mass black hole": "#38bdf8",
        "Sagittarius A*": "#22c55e",
        "1 billion solar-mass black hole": "#eab308",
    }

    for _, row in anchors.iterrows():
        name = row["object"]
        x = row["schwarzschild_radius_km"]
        y = row["hawking_temperature_K"]
        color = colors[name]
        ax.scatter([x], [y], s=70, color=color, edgecolor="#f8fafc", linewidth=0.7, zorder=5)
        dx, dy = offsets[name]
        label = f"{name}\nT={y:.2e} K\nr_s={x:.2e} km"
        ax.annotate(label, xy=(x, y), xytext=(x*dx, y*dy), color=color, fontsize=8.7,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#020617", edgecolor=color, alpha=0.84))

    ax.set_xlabel("Schwarzschild radius of the black hole, km")
    ax.set_ylabel("Hawking temperature, K")
    ax.set_title("JWST_0028 — Hawking temperature is hot only for tiny black holes; large black holes are cold")
    legend(ax, "upper right")
    fig.tight_layout()
    png = PNG / f"{VERSION}_HAWKING_TEMPERATURE_ANCHORS.png"
    fig.savefig(png, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return png


def plot_anchor_ladder(anchors):
    import matplotlib.pyplot as plt
    import numpy as np

    ordered = anchors.sort_values("mass_kg")
    fig, ax = plt.subplots(figsize=(14.8, 7.6))
    dark(fig, ax)
    y = np.arange(len(ordered))
    ax.barh(y, ordered["hawking_temperature_K"], color="#38bdf8", alpha=0.85)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["object"], color="#f8fafc")
    ax.axvline(CMB_K, color="#facc15", linestyle="--", linewidth=1.6, label="CMB today")
    for yi, (_, row) in zip(y, ordered.iterrows()):
        ax.text(row["hawking_temperature_K"]*1.25, yi, f"{row['hawking_temperature_K']:.3e} K", color="#f8fafc", va="center", fontsize=9)
    ax.set_xlabel("Hawking temperature, K")
    ax.set_title("Anchor comparison: Earth-mass, stellar-mass, and Sagittarius A* black holes")
    legend(ax, "lower right")
    fig.tight_layout()
    png = PNG / f"{VERSION}_ANCHOR_TEMPERATURE_LADDER.png"
    fig.savefig(png, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return png


def make_table_png(anchors):
    import matplotlib.pyplot as plt

    view = anchors[["object", "mass_Msun", "schwarzschild_radius_km", "schwarzschild_radius_AU", "hawking_temperature_K", "log10_T_K", "T_over_CMB"]].copy()
    for col in ["mass_Msun", "schwarzschild_radius_km", "schwarzschild_radius_AU", "hawking_temperature_K", "T_over_CMB"]:
        view[col] = view[col].map(lambda x: f"{x:.3e}")
    view["log10_T_K"] = view["log10_T_K"].map(lambda x: f"{x:.3f}")
    view.columns = ["object", "mass M_sun", "r_s km", "r_s AU", "T_H K", "log10(T_H)", "T_H/CMB"]

    fig, ax = plt.subplots(figsize=(16.6, 5.2))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title("JWST_0028 — Hawking temperature anchors", color="#f8fafc", fontsize=16, pad=14)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1, 1.55)
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
            name = str(view.iloc[r-1, 0])
            if "Earth" in name:
                cell.set_facecolor("#431407")
                cell.get_text().set_color("#fed7aa")
            if "Sagittarius" in name:
                cell.set_facecolor("#052e16")
                cell.get_text().set_color("#bbf7d0")
    fig.tight_layout()
    png = PNG / f"{VERSION}_ANCHOR_TABLE.png"
    fig.savefig(png, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return png


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
        display(HTML(f'<div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569"><h2>Images available here</h2><ul>{links}</ul><hr style="border-color:#334155">{previews}</div>'))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    curve, anchors, curve_csv, anchors_csv = build_data()
    png1 = plot_radius_temperature(curve, anchors)
    png2 = plot_anchor_ladder(anchors)
    png3 = make_table_png(anchors)
    uploaded = upload_outputs([png1, png2, png3, curve_csv, anchors_csv])

    earth = anchors[anchors["object"] == "Earth-mass black hole"].iloc[0]
    sgra = anchors[anchors["object"] == "Sagittarius A*"].iloc[0]

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Critical clarification", "x-axis is black-hole size across different black holes, not radius inside one black hole"),
        ("Formula", "T_H = hbar*c/(4*pi*kB*r_s)"),
        ("Earth-mass T_H K", f"{earth['hawking_temperature_K']:.6e}"),
        ("Earth-mass log10(T_H)", f"{earth['log10_T_K']:.6f}"),
        ("Earth-mass r_s", f"{earth['schwarzschild_radius_m']:.6e} m"),
        ("Sgr A* T_H K", f"{sgra['hawking_temperature_K']:.6e}"),
        ("Sgr A* r_s km", f"{sgra['schwarzschild_radius_km']:.6e}"),
    ], ["Field", "Value"])
    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"]) for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)
    print("\nCOMMENTS")
    print("Tiny black holes are hot. Large black holes are cold. The radius axis is not a travel path into Sagittarius A*.")
    print("Earth-mass black-hole Hawking temperature is about 2.06e-2 K, log10(T) about -1.686, not 10^-1.25 K.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
