# JWST_0025
# Audit: Correct horizon-line colors and auto-upload output links to GitHub. Matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import os
import sys
import json
import base64
import subprocess
import importlib

VERSION = "JWST_0025"
PROJECT = "COLOR-CODED FRIEDMANN HORIZONS WITH AUTO UPLOAD"
REPO = "gear66me-ui/JWST"
BRANCH = "main"
DEST_ROOT = f"OUTPUTS/{VERSION}_COLOR_HORIZONS_AUTO_UPLOAD"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

C_KM_S = 299792.458
MPC_TO_GLY = 3.261563777e-3
H0_KM_S_MPC = 67.4
OMEGA_M = 0.315
OMEGA_L = 0.685
OMEGA_R = 9.2e-5
OMEGA_K = 1.0 - OMEGA_M - OMEGA_L - OMEGA_R

COLORS = {
    "distance_curve": "#e2e8f0",
    "hubble": "#22d3ee",
    "event": "#f59e0b",
    "particle": "#e879f9",
    "grid": "#334155",
    "text": "#f8fafc",
    "muted": "#cbd5e1",
}


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


def print_table(rows, heads):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(heads)] if rows else [len(str(h)) for h in heads]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(heads)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(heads))))


def E_of_z(z):
    import numpy as np
    zp1 = 1.0 + z
    return np.sqrt(OMEGA_R * zp1**4 + OMEGA_M * zp1**3 + OMEGA_K * zp1**2 + OMEGA_L)


def E_of_a(a):
    import numpy as np
    return np.sqrt(OMEGA_R / a**4 + OMEGA_M / a**3 + OMEGA_K / a**2 + OMEGA_L)


def cumulative_trapezoid(y, x):
    import numpy as np
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def compute_curves():
    import numpy as np
    import pandas as pd

    hubble_radius_gly = (C_KM_S / H0_KM_S_MPC) * MPC_TO_GLY

    # Main curve: enough density to show the asymptote cleanly without slow runtime.
    z_plot = np.concatenate([[0.0], np.logspace(-3, 5, 5000)])
    dc_gly = hubble_radius_gly * cumulative_trapezoid(1.0 / E_of_z(z_plot), z_plot)

    # High-resolution horizon integrations.
    z_horizon = np.concatenate([[0.0], np.logspace(-6, 8, 170000)])
    particle_radius_gly = hubble_radius_gly * float(np.trapezoid(1.0 / E_of_z(z_horizon), z_horizon))

    a_future = np.logspace(0, 7, 170000)
    event_radius_gly = hubble_radius_gly * float(np.trapezoid(1.0 / (a_future**2 * E_of_a(a_future)), a_future))

    curve = pd.DataFrame({
        "redshift_z": z_plot,
        "E_z": E_of_z(z_plot),
        "comoving_distance_Gly": dc_gly,
    })
    horizons = pd.DataFrame({
        "boundary": ["Hubble radius", "Cosmic event horizon", "Particle horizon / observable radius"],
        "radius_Gly": [hubble_radius_gly, event_radius_gly, particle_radius_gly],
        "color": [COLORS["hubble"], COLORS["event"], COLORS["particle"]],
        "line_style": ["short dash", "long dash", "dash dot"],
        "meaning": [
            "recession speed equals c today; not a visibility wall",
            "light emitted today beyond this distance never reaches us",
            "present-day comoving radius of all light/signals that could have reached us",
        ],
    })
    return curve, horizons


def dark_ax(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color=COLORS["grid"], linewidth=0.55, alpha=0.70)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def plot_colored_horizons(curve, horizons):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(15.8, 8.7))
    dark_ax(fig, ax)

    ax.plot(
        curve["redshift_z"],
        curve["comoving_distance_Gly"],
        color=COLORS["distance_curve"],
        linewidth=3.0,
        label="Friedmann comoving distance curve",
        zorder=5,
    )

    dash_patterns = {
        "Hubble radius": (0, (4, 3)),
        "Cosmic event horizon": (0, (9, 4)),
        "Particle horizon / observable radius": (0, (3, 2, 1, 2)),
    }

    label_x_text = 1.35e4
    leader_x0 = 2.4e3
    leader_x1 = 1.1e4
    for _, row in horizons.iterrows():
        color = row["color"]
        radius = float(row["radius_Gly"])
        name = row["boundary"]
        ax.axhline(
            radius,
            color=color,
            linewidth=2.15,
            linestyle=dash_patterns[name],
            alpha=0.98,
            label=f"{name}: {radius:.2f} Gly",
            zorder=4,
        )
        ax.plot([leader_x0, leader_x1], [radius, radius], color=color, linewidth=2.2, alpha=0.96, zorder=6)
        ax.text(
            label_x_text,
            radius,
            f" {name}\n {radius:.2f} Gly",
            color=color,
            fontsize=10.2,
            va="center",
            ha="left",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "#020617", "edgecolor": color, "alpha": 0.92},
            zorder=7,
        )

    ax.text(
        0.0014,
        44.0,
        "Color key\ncyan = Hubble radius\norange = cosmic event horizon\nmagenta = particle horizon",
        color=COLORS["text"],
        fontsize=10.3,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.36", "facecolor": "#020617", "edgecolor": "#475569", "alpha": 0.92},
    )

    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e5)
    ax.set_ylim(0, max(horizons["radius_Gly"]) * 1.09)
    ax.set_xlabel("redshift z")
    ax.set_ylabel("model-inferred comoving distance, billion light-years")
    ax.set_title("JWST_0025 — Friedmann redshift-distance curve with distinct horizon colors")

    leg = ax.legend(loc="lower right", fontsize=8.4, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color(COLORS["text"])

    fig.tight_layout()
    path = PNG / f"{VERSION}_FRIEDMANN_HORIZONS_COLOR_FIXED.png"
    fig.savefig(path, dpi=270, facecolor=fig.get_facecolor())
    plt.show()
    return path


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
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.get(url, headers=headers, params={"ref": BRANCH}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def github_put_file(token, repo, src, dest):
    import requests
    url = f"https://api.github.com/repos/{repo}/contents/{dest}"
    sha = github_get_sha(token, repo, dest)
    payload = {
        "message": f"{VERSION}: upload {src.name}",
        "branch": BRANCH,
        "content": base64.b64encode(src.read_bytes()).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}
    r = requests.put(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    raw = f"https://raw.githubusercontent.com/{repo}/{BRANCH}/{dest}"
    return raw, "updated" if sha else "created"


def upload_outputs(files):
    token = get_token()
    rows = []
    for src in files:
        sub = "PNG" if src.suffix.lower() == ".png" else "CSV"
        dest = f"{DEST_ROOT}/{sub}/{src.name}"
        raw, status = github_put_file(token, REPO, src, dest)
        rows.append({
            "type": src.suffix.lower().replace(".", ""),
            "status": status,
            "file": src.name,
            "raw_url": raw,
            "size_mb": round(src.stat().st_size / (1024 * 1024), 5),
        })
    return rows


def main():
    setup()
    curve, horizons = compute_curves()

    curve_csv = CSV / f"{VERSION}_FRIEDMANN_DISTANCE_CURVE.csv"
    horizons_csv = CSV / f"{VERSION}_HORIZON_COLOR_KEY.csv"
    links_csv = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.csv"
    links_json = CSV / f"{VERSION}_GITHUB_OUTPUT_LINKS.json"

    curve.to_csv(curve_csv, index=False)
    horizons.to_csv(horizons_csv, index=False)
    plot_png = plot_colored_horizons(curve, horizons)

    upload_files = [plot_png, curve_csv, horizons_csv]
    link_rows = upload_outputs(upload_files)

    import pandas as pd
    pd.DataFrame(link_rows).to_csv(links_csv, index=False)
    links_json.write_text(json.dumps(link_rows, indent=2), encoding="utf-8")

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Plot corrected", "colored Hubble, event, and particle horizon lines"),
        ("Repository", REPO),
        ("GitHub destination", DEST_ROOT),
        ("No AI images", "matplotlib only"),
    ], ["Field", "Value"])

    print("\nHORIZON COLOR KEY")
    print_table(
        [(r["boundary"], f"{r['radius_Gly']:.3f}", r["color"], r["meaning"]) for _, r in horizons.iterrows()],
        ["Boundary", "Gly", "Color", "Meaning"],
    )

    print("\nRAW GITHUB LINKS")
    print_table(
        [(r["type"], r["status"], r["file"], r["raw_url"]) for r in link_rows],
        ["Type", "Status", "File", "Raw URL"],
    )

    print("\nLOCAL OUTPUTS")
    print_table([
        ("png", str(plot_png)),
        ("csv", str(curve_csv)),
        ("csv", str(horizons_csv)),
        ("csv", str(links_csv)),
        ("json", str(links_json)),
    ], ["Type", "Path"])

    print("\nCOMMENTS")
    print("The dashed horizon lines now use different colors, dash styles, and right-side label boxes.")
    print("Hubble radius is cyan, cosmic event horizon is orange, particle horizon is magenta.")
    print("Outputs are automatically pushed to GitHub and raw links are printed above.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
