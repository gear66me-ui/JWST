# JWST_0021
# Audit: 1769 Venus transit parallax tracks from JPL Horizons vectors. Matplotlib only. No AI images. No FITS/image downloads.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0021"
PROJECT = "1769 VENUS TRANSIT PARALLAX TRACKS FROM JPL HORIZONS"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
ARCSEC_PER_RAD = 206264.80624709636
AU_KM = 149597870.7
R_SUN_KM = 695700.0
R_VENUS_KM = 6051.8
START_UTC = "1769-06-03 18:00"
STOP_UTC = "1769-06-04 06:00"
STEP_MIN = 5

SITES = [
    {"site": "Tahiti / Point Venus", "short": "Tahiti", "lat_deg": -17.497, "lon_deg": -149.494, "elev_km": 0.005},
    {"site": "Vardo / Vardohus", "short": "Vardo", "lat_deg": 70.370, "lon_deg": 31.110, "elev_km": 0.010},
    {"site": "Philadelphia / Rittenhouse", "short": "Philadelphia", "lat_deg": 39.9526, "lon_deg": -75.1652, "elev_km": 0.020},
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg, imp in [("numpy", "numpy"), ("pandas", "pandas"), ("matplotlib", "matplotlib"), ("astropy", "astropy"), ("astroquery", "astroquery")]:
        need(pkg, imp)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def print_table(rows, heads):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(heads)]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(heads)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(heads))))


def make_times():
    import pandas as pd
    from astropy.time import Time
    times = pd.date_range(START_UTC, STOP_UTC, freq=f"{STEP_MIN}min")
    iso = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
    jd = Time(iso, scale="utc").jd
    return iso, jd


def unit(v):
    import numpy as np
    n = np.linalg.norm(v, axis=1)
    return v / n[:, None]


def tangent_basis(sun_unit):
    import numpy as np
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (sun_unit.shape[0], 1))
    north = ref - (ref * sun_unit).sum(axis=1)[:, None] * sun_unit
    bad = np.linalg.norm(north, axis=1) < 1e-12
    if bad.any():
        ref2 = np.tile(np.array([0.0, 1.0, 0.0]), (bad.sum(), 1))
        north[bad] = ref2 - (ref2 * sun_unit[bad]).sum(axis=1)[:, None] * sun_unit[bad]
    north = north / np.linalg.norm(north, axis=1)[:, None]
    east = np.cross(north, sun_unit)
    east = east / np.linalg.norm(east, axis=1)[:, None]
    return east, north


def horizons_vectors(target_id, site, jd):
    import numpy as np
    from astroquery.jplhorizons import Horizons
    loc = {"lon": float(site["lon_deg"]), "lat": float(site["lat_deg"]), "elevation": float(site["elev_km"]), "body": 399}
    obj = Horizons(id=target_id, id_type="majorbody", location=loc, epochs=list(map(float, jd)))
    tab = obj.vectors()
    return np.vstack([np.array(tab["x"], dtype=float), np.array(tab["y"], dtype=float), np.array(tab["z"], dtype=float)]).T


def compute_site(site, iso, jd):
    import numpy as np
    import pandas as pd
    sun = horizons_vectors("10", site, jd)
    ven = horizons_vectors("299", site, jd)
    sun_u = unit(sun)
    ven_u = unit(ven)
    east, north = tangent_basis(sun_u)
    delta = ven_u - sun_u
    x = (delta * east).sum(axis=1) * ARCSEC_PER_RAD
    y = (delta * north).sum(axis=1) * ARCSEC_PER_RAD
    sun_dist_km = np.linalg.norm(sun, axis=1) * AU_KM
    ven_dist_km = np.linalg.norm(ven, axis=1) * AU_KM
    sun_r = np.arcsin(R_SUN_KM / sun_dist_km) * ARCSEC_PER_RAD
    ven_r = np.arcsin(R_VENUS_KM / ven_dist_km) * ARCSEC_PER_RAD
    sep = np.sqrt(x*x + y*y)
    return pd.DataFrame({
        "utc": iso,
        "site": site["site"],
        "short": site["short"],
        "lat_deg": site["lat_deg"],
        "lon_deg": site["lon_deg"],
        "x_arcsec_east": x,
        "y_arcsec_north": y,
        "venus_sun_center_sep_arcsec": sep,
        "sun_radius_arcsec": sun_r,
        "venus_radius_arcsec": ven_r,
        "center_inside_sun": sep < sun_r,
        "venus_disk_fully_inside_sun": sep < (sun_r - ven_r),
    })


def fallback_tracks(iso):
    import numpy as np
    import pandas as pd
    t = np.linspace(-1, 1, len(iso))
    rows = []
    params = [
        ("Tahiti / Point Venus", "Tahiti", -17.497, -149.494, 210, 160),
        ("Vardo / Vardohus", "Vardo", 70.370, 31.110, 245, 145),
        ("Philadelphia / Rittenhouse", "Philadelphia", 39.9526, -75.1652, 230, 152),
    ]
    for site, short, lat, lon, y0, slope in params:
        x = 920.0 * t
        y = slope * t + y0
        sep = np.sqrt(x*x + y*y)
        for i, u in enumerate(iso):
            rows.append({
                "utc": u,
                "site": site,
                "short": short,
                "lat_deg": lat,
                "lon_deg": lon,
                "x_arcsec_east": x[i],
                "y_arcsec_north": y[i],
                "venus_sun_center_sep_arcsec": sep[i],
                "sun_radius_arcsec": 946.0,
                "venus_radius_arcsec": 29.0,
                "center_inside_sun": bool(sep[i] < 946.0),
                "venus_disk_fully_inside_sun": bool(sep[i] < 917.0),
            })
    return pd.DataFrame(rows)


def threshold_brackets(df):
    rows = []
    for site, g in df.groupby("site", sort=False):
        gg = g.reset_index(drop=True)
        for label, threshold in [
            ("external contact threshold", gg["sun_radius_arcsec"] + gg["venus_radius_arcsec"]),
            ("internal contact threshold", gg["sun_radius_arcsec"] - gg["venus_radius_arcsec"]),
        ]:
            f = gg["venus_sun_center_sep_arcsec"].to_numpy() - threshold.to_numpy()
            utc = list(gg["utc"])
            for i in range(len(f) - 1):
                if f[i] == 0 or f[i] * f[i+1] < 0:
                    rows.append({"site": site, "event": label, "bracket_utc": f"{utc[i]} to {utc[i+1]}"})
    return rows


def dark_axis(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.tick_params(colors="#dbeafe")
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def plot_tracks(df, zoom=False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig, ax = plt.subplots(figsize=(13.0, 8.0) if zoom else (10.5, 10.5))
    dark_axis(fig, ax)
    sun_r = float(df["sun_radius_arcsec"].median())
    ax.add_patch(Circle((0, 0), sun_r, fill=False, color="#fde68a", linewidth=1.5, alpha=0.95))
    ax.axhline(0, color="#475569", linewidth=0.75)
    ax.axvline(0, color="#475569", linewidth=0.75)
    colors = {"Tahiti": "#38bdf8", "Vardo": "#fb7185", "Philadelphia": "#a7f3d0"}
    for short, g in df.groupby("short", sort=False):
        ax.plot(g["x_arcsec_east"], g["y_arcsec_north"], color=colors.get(short), linewidth=2.1, label=short)
        every = max(1, len(g) // 9)
        ax.scatter(g["x_arcsec_east"].iloc[::every], g["y_arcsec_north"].iloc[::every], s=20, color=colors.get(short), edgecolor="white", linewidth=0.35, zorder=5)
        close = g.iloc[g["venus_sun_center_sep_arcsec"].to_numpy().argmin()]
        ax.scatter([close["x_arcsec_east"]], [close["y_arcsec_north"]], s=80, color=colors.get(short), edgecolor="white", linewidth=0.8, zorder=6)
        ax.text(close["x_arcsec_east"] + 18, close["y_arcsec_north"] + 12, f"{short}\nmin {close['venus_sun_center_sep_arcsec']:.1f}\"", color="#f8fafc", fontsize=8.2)
    if zoom:
        ax.set_xlim(-1020, 1020)
        ax.set_ylim(df["y_arcsec_north"].min() - 120, df["y_arcsec_north"].max() + 120)
        title = "1769 Venus transit parallax microscope — shifted tracks from Earth stations"
        path = PNG / f"{VERSION}_ZOOM_PARALLAX_TRACKS.png"
    else:
        ax.set_xlim(-1120, 1120)
        ax.set_ylim(-1120, 1120)
        title = "1769 Venus transit tracks on the solar disk"
        path = PNG / f"{VERSION}_FULL_SOLAR_DISK_TRACKS.png"
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Apparent offset from Sun center, east arcsec")
    ax.set_ylabel("Apparent offset from Sun center, north arcsec")
    ax.set_title(title)
    leg = ax.legend(loc="upper right", facecolor="#020617", edgecolor="#475569", fontsize=9)
    for t in leg.get_texts():
        t.set_color("#f8fafc")
    ax.text(0.012, 0.014, "JPL Horizons topocentric vectors if online; fallback demo only if network query fails.", transform=ax.transAxes, color="#cbd5e1", fontsize=8.2)
    fig.tight_layout()
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_track_separation(df):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    px = df.pivot_table(index="utc", columns="short", values="x_arcsec_east")
    py = df.pivot_table(index="utc", columns="short", values="y_arcsec_north")
    if "Tahiti" not in px.columns or "Vardo" not in px.columns:
        return None, None
    sep = np.sqrt((px["Tahiti"] - px["Vardo"])**2 + (py["Tahiti"] - py["Vardo"])**2)
    out = pd.DataFrame({"utc": sep.index, "tahiti_vardo_track_separation_arcsec": sep.values})
    out_csv = CSV / f"{VERSION}_TAHITI_VARDO_TRACK_SEPARATION.csv"
    out.to_csv(out_csv, index=False)
    fig, ax = plt.subplots(figsize=(13.2, 6.8))
    dark_axis(fig, ax)
    ax.plot(range(len(out)), out["tahiti_vardo_track_separation_arcsec"], color="#fbbf24", linewidth=2.2)
    step = max(1, len(out) // 10)
    ax.scatter(range(0, len(out), step), out["tahiti_vardo_track_separation_arcsec"].iloc[::step], s=28, color="#fbbf24", edgecolor="white", linewidth=0.4)
    ticks = list(range(0, len(out), max(1, len(out) // 6)))
    ax.set_xticks(ticks)
    ax.set_xticklabels([out["utc"].iloc[i][11:16] for i in ticks])
    ax.set_xlabel("UTC time on 1769-06-03/04")
    ax.set_ylabel("Tahiti-Vardo apparent track separation, arcsec")
    ax.set_title("Parallax signal: same Venus, different station, shifted apparent track")
    fig.tight_layout()
    path = PNG / f"{VERSION}_TAHITI_VARDO_PARALLAX_SEPARATION.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path, out_csv


def main():
    import pandas as pd
    setup()
    iso, jd = make_times()
    mode = "JPL_HORIZONS"
    try:
        df = pd.concat([compute_site(site, iso, jd) for site in SITES], ignore_index=True)
    except Exception as exc:
        mode = "EMBEDDED_FALLBACK_DEMO_NOT_HISTORICAL"
        print(f"WARNING: JPL Horizons query failed; using fallback demo only. {exc}")
        df = fallback_tracks(iso)
    master_csv = CSV / f"{VERSION}_1769_VENUS_TRANSIT_TRACKS.csv"
    df.to_csv(master_csv, index=False)
    contacts = threshold_brackets(df)
    contacts_csv = CSV / f"{VERSION}_CONTACT_THRESHOLD_BRACKETS.csv"
    pd.DataFrame(contacts).to_csv(contacts_csv, index=False)
    full_png = plot_tracks(df, zoom=False)
    zoom_png = plot_tracks(df, zoom=True)
    sep_png, sep_csv = plot_track_separation(df)
    close_rows = []
    for site, g in df.groupby("site", sort=False):
        r = g.iloc[g["venus_sun_center_sep_arcsec"].to_numpy().argmin()]
        close_rows.append((r["short"], r["utc"], f"{r['venus_sun_center_sep_arcsec']:.3f}", f"{r['x_arcsec_east']:.3f}", f"{r['y_arcsec_north']:.3f}"))
    outputs = [("png", str(full_png)), ("png", str(zoom_png)), ("csv", str(master_csv)), ("csv", str(contacts_csv))]
    if sep_png:
        outputs.append(("png", str(sep_png)))
    if sep_csv:
        outputs.append(("csv", str(sep_csv)))
    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Project", PROJECT),
        ("Mode", mode),
        ("Time range UTC", f"{START_UTC} to {STOP_UTC}"),
        ("Step", f"{STEP_MIN} minutes"),
        ("Sites", ", ".join([s["short"] for s in SITES])),
        ("Method", "observer-to-Sun and observer-to-Venus vectors projected onto local Sun tangent plane"),
    ], ["Field", "Value"])
    print("\nCLOSEST CENTER SEPARATION")
    print_table(close_rows, ["Site", "UTC", "sep arcsec", "x arcsec", "y arcsec"])
    print("\nOUTPUTS")
    print_table(outputs, ["Type", "Path"])
    print("\nCOMMENTS")
    print("Matplotlib only. No AI images. No FITS/image downloads.")
    print("If Mode is JPL_HORIZONS, tracks are computed from online JPL Horizons topocentric vectors.")
    print("If Mode is fallback, the plot is only a geometry demonstration and not historical data.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
