# JWST_0036
# Audit: High-redshift wavelength-stretch plots for MoM-z14 and z=13..15. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import sys, subprocess, importlib

VERSION = "JWST_0036"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

# Reported frontier objects / comparison redshifts.
# MoM-z14 is currently used here as the highest reported JWST spectroscopic frontier point.
REDSHIFTS = [
    ("z = 13.00 comparison", 13.00, "theoretical comparison point"),
    ("JADES-GS-z14-1", 13.90, "reported JWST/NIRSpec luminous galaxy"),
    ("JADES-GS-z14-0", 14.1793, "ALMA [O III] 88um precision redshift; JWST-selected"),
    ("MoM-z14", 14.44, "reported JWST/NIRSpec spectroscopic frontier galaxy"),
    ("z = 15.00 comparison", 15.00, "theoretical comparison point"),
]

# Classic H-alpha + [N II] triplet used in the previous JWST/NIRSpec z~2.5 example.
# At z~14 these move to ~10 microns, i.e. MIRI/MRS territory rather than NIRSpec.
OPTICAL_TRIPLET = [
    ("[N II] 6548", 0.654805, "ionized nitrogen"),
    ("H-alpha 6563", 0.656281, "hydrogen Balmer-alpha"),
    ("[N II] 6583", 0.658345, "ionized nitrogen"),
]

# Three common rest-UV lines that land in NIRSpec wavelength range at z~14.
# These are plotted as a practical high-z reference set, not as a claim of measured fluxes in this script.
UV_REFERENCE_LINES = [
    ("N IV] 1486", 0.1486, "ionized nitrogen UV"),
    ("He II 1640", 0.1640, "helium UV"),
    ("C III] 1908", 0.1908, "carbon UV doublet blend"),
]


def need(pkg, imp=None):
    imp = imp or pkg
    try:
        importlib.import_module(imp)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def setup():
    for pkg in ["numpy", "pandas", "matplotlib", "astropy"]:
        need(pkg)
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)


def dark_axis(fig, ax):
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def style_legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.1, facecolor="#020617", edgecolor="#475569")
    for text in leg.get_texts():
        text.set_color("#f8fafc")


def observed_um(rest_um, z):
    return rest_um * (1.0 + z)


def build_tables():
    import pandas as pd
    rows = []
    for object_name, z, status in REDSHIFTS:
        for line, rest, species in OPTICAL_TRIPLET:
            rows.append({
                "object_or_case": object_name,
                "z": z,
                "status": status,
                "line_set": "H-alpha plus [N II] optical triplet",
                "line": line,
                "species": species,
                "rest_um": rest,
                "observed_um": observed_um(rest, z),
                "stretch_factor_1_plus_z": 1 + z,
                "instrument_note": "JWST/MIRI MRS range for z~13-15",
            })
        for line, rest, species in UV_REFERENCE_LINES:
            rows.append({
                "object_or_case": object_name,
                "z": z,
                "status": status,
                "line_set": "rest-UV high-z reference lines",
                "line": line,
                "species": species,
                "rest_um": rest,
                "observed_um": observed_um(rest, z),
                "stretch_factor_1_plus_z": 1 + z,
                "instrument_note": "JWST/NIRSpec-accessible at z~13-15",
            })
    df = pd.DataFrame(rows)
    return df


def add_instrument_bands(ax, yloc=0.02):
    # Approximate wavelength domains for visual context only.
    bands = [
        (0.6, 5.3, "NIRSpec approx."),
        (5.0, 12.0, "MIRI MRS approx. ch1-2"),
    ]
    ymin, ymax = ax.get_ylim()
    height = ymax - ymin
    for x0, x1, label in bands:
        ax.axvspan(x0, x1, color="#64748b", alpha=0.08)
        ax.text((x0 + x1) / 2, ymin + yloc * height, label, color="#cbd5e1", fontsize=8.4, ha="center", va="bottom")


def plot_linearity_vs_z(df):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16.8, 9.2))
    dark_axis(fig, ax)
    zgrid = np.linspace(0, 15.5, 400)
    line_styles = ["-", "--", ":"]
    for idx, (line, rest, species) in enumerate(OPTICAL_TRIPLET):
        y = rest * (1 + zgrid)
        ax.plot(zgrid, y, linewidth=1.55, linestyle=line_styles[idx], label=f"{line}: λobs = {rest:.6f}(1+z)")
    for name, z, status in REDSHIFTS:
        if "MoM" in name or "JADES" in name:
            ax.axvline(z, linewidth=0.9, alpha=0.55)
            ax.text(z, 0.8, name.replace("JADES-GS-", "JADES "), color="#e0f2fe", fontsize=8.2, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Observed wavelength, micron")
    ax.set_title(f"{VERSION} — the H-alpha/[N II] triplet shifts linearly in wavelength: λobs = λrest(1+z)")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 11.2)
    add_instrument_bands(ax, yloc=0.025)
    style_legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HALPHA_NII_OBSERVED_WAVELENGTH_VS_Z.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_frontier_triplet_positions(df):
    import matplotlib.pyplot as plt
    sub = df[df["line_set"] == "H-alpha plus [N II] optical triplet"].copy()
    pivot = sub.pivot(index="object_or_case", columns="line", values="observed_um").reset_index()
    order = [r[0] for r in REDSHIFTS]
    sub["object_or_case"] = pd.Categorical(sub["object_or_case"], categories=order, ordered=True)
    sub = sub.sort_values(["object_or_case", "rest_um"])
    fig, ax = plt.subplots(figsize=(16.6, 8.4))
    dark_axis(fig, ax)
    y_positions = {name: i for i, name in enumerate(order)}
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for idx, (line, rest, species) in enumerate(OPTICAL_TRIPLET):
        s = sub[sub["line"] == line]
        ax.plot(s["observed_um"], [y_positions[v] for v in s["object_or_case"]], marker="o", linewidth=1.25, markersize=5.2, color=colors[idx], label=line)
        for _, row in s.iterrows():
            if "MoM" in row["object_or_case"]:
                ax.text(row["observed_um"], y_positions[row["object_or_case"]] + 0.10, f"{row['observed_um']:.3f} µm", color=colors[idx], fontsize=8.5, ha="center")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color="#dbeafe")
    ax.set_xlabel("Predicted observed wavelength, micron")
    ax.set_ylabel("Galaxy / comparison case")
    ax.set_title(f"{VERSION} — three optical triplet lines at z≈13–15; MoM-z14 puts H-alpha near 10.13 µm")
    ax.set_xlim(9.0, 10.65)
    style_legend(ax, "lower right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HIGH_Z_HALPHA_NII_THREE_LINE_POSITIONS.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_uv_reference_lines(df):
    import matplotlib.pyplot as plt
    sub = df[(df["line_set"] == "rest-UV high-z reference lines") & (df["object_or_case"] == "MoM-z14")].copy()
    fig, ax = plt.subplots(figsize=(15.4, 7.6))
    dark_axis(fig, ax)
    ax.set_xlim(2.1, 3.2)
    ax.set_ylim(0, 1)
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for idx, (_, row) in enumerate(sub.sort_values("rest_um").iterrows()):
        x = row["observed_um"]
        ax.axvline(x, color=colors[idx], linewidth=1.7, alpha=0.90, label=f"{row['line']} → {x:.3f} µm")
        ax.text(x, 0.90, f"{row['line']}\n{x:.3f} µm", color=colors[idx], fontsize=10, ha="center", va="top", rotation=90)
    ax.axvspan(0.6, 5.3, color="#64748b", alpha=0.08)
    ax.set_xlabel("Observed wavelength, micron")
    ax.set_yticks([])
    ax.set_title(f"{VERSION} — three rest-UV reference lines for MoM-z14 at z=14.44 land inside NIRSpec")
    style_legend(ax, "upper right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_MOM_Z14_UV_REFERENCE_THREE_LINES.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_table(df):
    import matplotlib.pyplot as plt
    table_df = df[(df["object_or_case"] == "MoM-z14")].copy()
    view = table_df[["line_set", "line", "rest_um", "observed_um", "stretch_factor_1_plus_z", "instrument_note"]].copy()
    view["rest_um"] = view["rest_um"].map(lambda x: f"{x:.6f}")
    view["observed_um"] = view["observed_um"].map(lambda x: f"{x:.6f}")
    view["stretch_factor_1_plus_z"] = view["stretch_factor_1_plus_z"].map(lambda x: f"{x:.6f}")
    view.columns = ["line set", "line", "rest λ µm", "observed λ µm", "1+z", "note"]
    fig, ax = plt.subplots(figsize=(18.4, 6.8))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — MoM-z14 z=14.44 wavelength-stretch table", color="#f8fafc", fontsize=15, pad=14)
    table = ax.table(cellText=view.values, colLabels=view.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1, 1.52)
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
            if "H-alpha" in str(view.iloc[r - 1, 1]):
                cell.set_facecolor("#450a0a")
                cell.get_text().set_color("#fecaca")
    fig.tight_layout()
    path = PNG / f"{VERSION}_MOM_Z14_WAVELENGTH_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


def cosmology_table(df):
    import pandas as pd
    try:
        from astropy.cosmology import Planck18 as cosmo
        import astropy.units as u
        rows = []
        for name, z, status in REDSHIFTS:
            rows.append({
                "object_or_case": name,
                "z": z,
                "status": status,
                "lookback_time_Gyr": cosmo.lookback_time(z).to(u.Gyr).value,
                "universe_age_Myr": cosmo.age(z).to(u.Myr).value,
                "comoving_distance_Gly": cosmo.comoving_distance(z).to(u.Glyr).value,
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        return pd.DataFrame([{"error": str(exc)}])


def print_table(rows, headers):
    widths = [len(str(h)) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(str(row[i]))) for i in range(len(headers))]
    print(" | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main():
    setup()
    df = build_tables()
    cosdf = cosmology_table(df)
    csv1 = CSV / f"{VERSION}_HIGH_Z_LINE_WAVELENGTHS.csv"
    csv2 = CSV / f"{VERSION}_HIGH_Z_COSMOLOGY_TABLE.csv"
    df.to_csv(csv1, index=False)
    cosdf.to_csv(csv2, index=False)

    p1 = plot_linearity_vs_z(df)
    p2 = plot_frontier_triplet_positions(df)
    p3 = plot_uv_reference_lines(df)
    p4 = plot_table(df)

    mom = df[(df["object_or_case"] == "MoM-z14") & (df["line_set"] == "H-alpha plus [N II] optical triplet")]
    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Concept", "lambda_observed = lambda_rest * (1 + z)"),
        ("Linear part", "observed wavelength is linear in z for any one rest-frame spectral line"),
        ("Nonlinear part", "cosmic age, lookback time, velocity interpretation, and distance are not linear in z"),
        ("Frontier target used", "MoM-z14, z_spec = 14.44"),
        ("H-alpha at z=14.44", f"{observed_um(0.656281, 14.44):.6f} micron"),
        ("H-alpha/[N II] visibility", "around 10.11-10.17 micron, MIRI/MRS territory"),
        ("Rest-UV reference visibility", "around 2.29-2.95 micron, NIRSpec territory"),
        ("CSV wavelengths", str(csv1)),
        ("CSV cosmology", str(csv2)),
        ("Plot 1", str(p1)),
        ("Plot 2", str(p2)),
        ("Plot 3", str(p3)),
        ("Table PNG", str(p4)),
    ], ["Field", "Value"])

    print("\nMoM-z14 H-alpha / [N II] predicted positions")
    rows = []
    for _, r in mom.iterrows():
        rows.append((r["line"], f"{r['rest_um']:.6f}", f"{r['observed_um']:.6f}", f"{r['stretch_factor_1_plus_z']:.6f}"))
    print_table(rows, ["Line", "Rest um", "Observed um", "1+z"])
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
