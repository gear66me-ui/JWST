# JWST_0027
# Audit: Hawking radiation temperature vs Schwarzschild radius. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import getpass
import subprocess
import importlib

VERSION = "JWST_0027"
PROJECT = "HAWKING TEMPERATURE VS SCHWARZSCHILD RADIUS"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = "OUTPUTS/JWST_0027_HAWKING_TEMPERATURE_SGRA"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

# CODATA/SI exact or conventional constants where applicable.
G = 6.67430e-11
C = 299792458.0
HBAR = 1.054571817e-34
KB = 1.380649e-23
MSUN = 1.98847e30
AU_KM = 149597870.7
CMB_K = 2.7255
SGRA_MASS_MSUN = 4.10e6


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


def schwarzschild_radius_m(mass_kg):
    return 2.0 * G * mass_kg / C**2


def hawking_temperature_k(mass_kg):
    return HBAR * C**3 / (8.0 * 3.141592653589793 * G * mass_kg * KB)


def hawking_temperature_from_rs_k(rs_m):
    return HBAR * C / (4.0 * 3.141592653589793 * KB * rs_m)


def light_seconds_from_km(km):
    return km / (C / 1000.0)


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
    leg = ax.legend(loc=loc, fontsize=8.4, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")
    return leg


def build_tables():
    import numpy as np
    import pandas as pd

    mass_grid_msun = np.logspace(-18, 12, 2400)
    mass_grid_kg = mass_grid_msun * MSUN
    rs_m = schwarzschild_radius_m(mass_grid_kg)
    temp_k = hawking_temperature_k(mass_grid_kg)
    curve = pd.DataFrame({
        "mass_Msun": mass_grid_msun,
        "mass_kg": mass_grid_kg,
        "schwarzschild_radius_km": rs_m / 1000.0,
        "schwarzschild_radius_AU": (rs_m / 1000.0) / AU_KM,
        "hawking_temperature_K": temp_k,
        "temperature_relative_to_CMB": temp_k / CMB_K,
    })

    objects = [
        ("Primordial example: asteroid mass", 1e15 / MSUN, "theoretical tiny black hole scale"),
        ("Earth-mass black hole", 5.9722e24 / MSUN, "hypothetical; radius centimeter scale"),
        ("1 solar-mass black hole", 1.0, "reference scaling value"),
        ("10 solar-mass black hole", 10.0, "stellar black-hole scale"),
        ("Sagittarius A*", SGRA_MASS_MSUN, "Milky Way central black hole"),
        ("1 billion solar-mass black hole", 1.0e9, "quasar/supermassive scale"),
    ]
    rows = []
    for name, m_msun, note in objects:
        m_kg = m_msun * MSUN
        rs_km = schwarzschild_radius_m(m_kg) / 1000.0
        t_k = hawking_temperature_k(m_kg)
        rows.append({
            "object": name,
            "mass_Msun": m_msun,
            "rs_km": rs_km,
            "rs_AU": rs_km / AU_KM,
            "rs_light_seconds": light_seconds_from_km(rs_km),
            "hawking_temperature_K": t_k,
            "T_over_CMB": t_k / CMB_K,
            "note": note,
        })
    anchors = pd.DataFrame(rows)
    curve_csv = CSV / f"{VERSION}_HAWKING_TEMPERATURE_CURVE.csv"
    anchor_csv = CSV / f"{VERSION}_ANCHOR_VALUES.csv"
    curve.to_csv(curve_csv, index=False)
    anchors.to_csv(anchor_csv, index=False)
    return curve, anchors, curve_csv, anchor_csv


def make_main_plot(curve, anchors):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14.8, 8.2))
    dark(fig, ax)

    ax.loglog(curve["schwarzschild_radius_km"], curve["hawking_temperature_K"],
              color="#67e8f9", linewidth=2.8, label=r"$T_H = \hbar c/(4\pi k_B r_s)$")
    ax.axhline(CMB_K, color="#facc15", linewidth=1.6, linestyle="--", label=f"CMB today = {CMB_K:.4f} K")

    for _, row in anchors.iterrows():
        x = row["rs_km"]
        y = row["hawking_temperature_K"]
        if "Sagittarius" in row["object"]:
            ax.scatter([x], [y], s=96, color="#f43f5e", edgecolor="#f8fafc", linewidth=0.8, zorder=6, label="Sagittarius A*")
            ax.annotate(f"Sagittarius A*\nT = {y:.3e} K\nr_s = {x/1e6:.2f} million km",
                        xy=(x, y), xytext=(x/80, y*1e5), color="#fecaca", fontsize=10.2,
                        arrowprops=dict(arrowstyle="->", color="#fecaca", lw=1.2),
                        bbox=dict(boxstyle="round,pad=0.35", facecolor="#020617", edgecolor="#7f1d1d", alpha=0.92))
        elif row["object"] in ["1 solar-mass black hole", "10 solar-mass black hole", "Earth-mass black hole"]:
            ax.scatter([x], [y], s=38, color="#f97316", zorder=5)
            ax.text(x*1.25, y*1.15, row["object"].replace(" black hole", ""), color="#fed7aa", fontsize=8.8)

    ax.set_xlabel("Schwarzschild radius, km")
    ax.set_ylabel("Hawking temperature, K")
    ax.set_title("JWST_0027 — Hawking temperature falls as black-hole radius grows")
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HAWKING_TEMPERATURE_VS_RADIUS.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def make_sgra_zoom(anchors):
    import numpy as np
    import matplotlib.pyplot as plt

    sgra = anchors[anchors["object"] == "Sagittarius A*"].iloc[0]
    mass = np.logspace(0, 10, 1400)
    rs_km = schwarzschild_radius_m(mass * MSUN) / 1000.0
    t_k = hawking_temperature_k(mass * MSUN)

    fig, ax = plt.subplots(figsize=(13.8, 7.7))
    dark(fig, ax)
    ax.loglog(mass, t_k, color="#38bdf8", linewidth=2.7, label="black-hole temperature by mass")
    ax.axhline(CMB_K, color="#facc15", linestyle="--", linewidth=1.4, label="CMB today")
    ax.axvline(SGRA_MASS_MSUN, color="#f43f5e", linestyle="--", linewidth=1.8, label="Sagittarius A* mass")
    ax.scatter([SGRA_MASS_MSUN], [sgra["hawking_temperature_K"]], s=90, color="#f43f5e", edgecolor="#f8fafc", zorder=5)

    text = (
        f"Sagittarius A*\n"
        f"mass = {SGRA_MASS_MSUN:.2e} M_sun\n"
        f"r_s = {sgra['rs_km']/1e6:.3f} million km\n"
        f"r_s = {sgra['rs_AU']:.5f} AU\n"
        f"T_H = {sgra['hawking_temperature_K']:.3e} K"
    )
    ax.text(0.03, 0.08, text, transform=ax.transAxes, color="#f8fafc", fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#020617", edgecolor="#475569", alpha=0.92))
    ax.set_xlabel("black-hole mass, solar masses")
    ax.set_ylabel("Hawking temperature, K")
    ax.set_title("Sagittarius A* is much colder than the cosmic microwave background")
    legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_SGRA_HAWKING_TEMPERATURE_ZOOM.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def make_table_png(anchors):
    import pandas as pd
    import matplotlib.pyplot as plt

    view = anchors.copy()
    view["mass_Msun"] = view["mass_Msun"].map(lambda x: f"{x:.3e}")
    view["rs_km"] = view["rs_km"].map(lambda x: f"{x:.3e}")
    view["rs_AU"] = view["rs_AU"].map(lambda x: f"{x:.3e}")
    view["rs_light_seconds"] = view["rs_light_seconds"].map(lambda x: f"{x:.3e}")
    view["hawking_temperature_K"] = view["hawking_temperature_K"].map(lambda x: f"{x:.3e}")
    view["T_over_CMB"] = view["T_over_CMB"].map(lambda x: f"{x:.3e}")
    view = view[["object", "mass_Msun", "rs_km", "rs_AU", "rs_light_seconds", "hawking_temperature_K", "T_over_CMB"]]
    view.columns = ["object", "mass M_sun", "r_s km", "r_s AU", "r_s light-s", "T_H K", "T_H / CMB"]

    fig, ax = plt.subplots(figsize=(16.8, 5.2))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title("JWST_0027 — Hawking temperature anchor values", color="#f8fafc", fontsize=16, pad=14)
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
            if "Sagittarius" in str(view.iloc[r-1, 0]):
                cell.set_facecolor("#450a0a")
                cell.get_text().set_color("#fecaca")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HAWKING_TEMPERATURE_TABLE.png"
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


def display_links(uploaded):
    image_rows = [r for r in uploaded if r["type"] == "png"]
    print("\nIMAGES AVAILABLE HERE:")
    for r in image_rows:
        print(r["file"])
        print(r["raw_url"])
    try:
        from IPython.display import HTML, display
        links_html = "".join(
            f'<li><a href="{r["raw_url"]}" target="_blank">{r["file"]}</a><br><code>{r["raw_url"]}</code></li>'
            for r in image_rows
        )
        previews = "".join(
            f'<h4 style="color:#e5e7eb">{r["file"]}</h4>'
            f'<a href="{r["raw_url"]}" target="_blank"><img src="{r["raw_url"]}" style="max-width:100%;border:1px solid #475569;border-radius:8px"></a>'
            for r in image_rows
        )
        display(HTML(f"""
        <div style="background:#050712;color:#f8fafc;padding:16px;border-radius:10px;border:1px solid #475569">
          <h2 style="margin-top:0;color:#f8fafc">Images available here</h2>
          <p>Click any PNG name or preview to open the raw GitHub image in a new tab.</p>
          <ul>{links_html}</ul>
          <hr style="border-color:#334155">
          {previews}
        </div>
        """))
    except Exception as exc:
        print(f"HTML preview skipped: {exc}")


def main():
    setup()
    curve, anchors, curve_csv, anchor_csv = build_tables()
    plot1 = make_main_plot(curve, anchors)
    plot2 = make_sgra_zoom(anchors)
    table_png = make_table_png(anchors)
    uploaded = upload_outputs([plot1, plot2, table_png, curve_csv, anchor_csv])

    sgra = anchors[anchors["object"] == "Sagittarius A*"].iloc[0]
    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Formula", "T_H = hbar*c^3/(8*pi*G*M*kB) = hbar*c/(4*pi*kB*r_s)"),
        ("Sgr A mass M_sun", f"{SGRA_MASS_MSUN:.3e}"),
        ("Sgr A r_s km", f"{sgra['rs_km']:.6e}"),
        ("Sgr A r_s AU", f"{sgra['rs_AU']:.6e}"),
        ("Sgr A T_H K", f"{sgra['hawking_temperature_K']:.6e}"),
        ("CMB K", f"{CMB_K:.6f}"),
        ("T_H / CMB", f"{sgra['T_over_CMB']:.6e}"),
    ], ["Field", "Value"])

    print("\nOUTPUT HTTP LINKS")
    print_table([(r["type"], r["status"], r["file"], r["raw_url"]) for r in uploaded], ["Type", "Status", "File", "HTTP link"])
    display_links(uploaded)

    print("\nLOCAL FILES")
    print_table([
        ("png", str(plot1)),
        ("png", str(plot2)),
        ("png", str(table_png)),
        ("csv", str(curve_csv)),
        ("csv", str(anchor_csv)),
    ], ["Type", "Path"])
    print("\nCOMMENTS")
    print("Hawking radiation is thermal quantum emission associated with the event horizon.")
    print("Large black holes are colder: temperature is inversely proportional to mass and Schwarzschild radius.")
    print("Sagittarius A* is far colder than the current cosmic microwave background, so it absorbs more ambient radiation than it emits today.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
