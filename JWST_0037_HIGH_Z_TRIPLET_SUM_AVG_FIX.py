# JWST_0037
# Audit: fixed high-z triplet plot with explicit sum and average table. Python/matplotlib only. No AI images.

from pathlib import Path
from datetime import datetime, timezone
import sys
import subprocess
import importlib

VERSION = "JWST_0037"
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"

TRIPLET = [
    (1, "[N II] 6548", 0.654805, "ionized nitrogen"),
    (2, "H-alpha 6563", 0.656281, "hydrogen Balmer-alpha"),
    (3, "[N II] 6583", 0.658345, "ionized nitrogen"),
]

REDSHIFTS = [
    ("z=13 comparison", 13.0000, "theoretical comparison"),
    ("JADES-GS-z14-1", 13.9000, "reported high-z JWST galaxy"),
    ("JADES-GS-z14-0", 14.1793, "reported high-z JWST galaxy / ALMA precision z"),
    ("MoM-z14", 14.4400, "reported JWST/NIRSpec frontier galaxy"),
    ("z=15 comparison", 15.0000, "theoretical comparison"),
]

UV_REFERENCE = [
    (1, "N IV] 1486", 0.148600, "rest-UV reference line"),
    (2, "He II 1640", 0.164000, "rest-UV reference line"),
    (3, "C III] 1908", 0.190800, "rest-UV reference line"),
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


def obs(rest_um, z):
    return rest_um * (1.0 + z)


def build_tables():
    import pandas as pd
    rows = []
    summary = []
    for name, z, status in REDSHIFTS:
        obs_values = []
        rest_values = []
        for n, line, rest, species in TRIPLET:
            observed = obs(rest, z)
            obs_values.append(observed)
            rest_values.append(rest)
            rows.append({
                "object_or_case": name,
                "z": z,
                "status": status,
                "line_set": "H-alpha plus [N II] optical triplet",
                "line_number": n,
                "line": line,
                "species": species,
                "rest_um": rest,
                "observed_um": observed,
                "slope_1_plus_z": 1.0 + z,
                "instrument_note": "At z~13-15 this optical triplet is around 9-10.5 um, MIRI/MRS territory.",
            })
        sum_obs = sum(obs_values)
        avg_obs = sum_obs / len(obs_values)
        sum_rest = sum(rest_values)
        avg_rest = sum_rest / len(rest_values)
        summary.append({
            "object_or_case": name,
            "z": z,
            "status": status,
            "triplet_observed_sum_um": sum_obs,
            "triplet_observed_average_um": avg_obs,
            "triplet_rest_sum_um": sum_rest,
            "triplet_rest_average_um": avg_rest,
            "average_stretch_check": avg_obs / avg_rest,
            "z_from_average_wavelengths": avg_obs / avg_rest - 1.0,
            "line_count": len(obs_values),
        })
        for n, line, rest, species in UV_REFERENCE:
            rows.append({
                "object_or_case": name,
                "z": z,
                "status": status,
                "line_set": "rest-UV reference triplet",
                "line_number": n,
                "line": line,
                "species": species,
                "rest_um": rest,
                "observed_um": obs(rest, z),
                "slope_1_plus_z": 1.0 + z,
                "instrument_note": "At z~13-15 these rest-UV lines are around 2-3 um, NIRSpec territory.",
            })
    return pd.DataFrame(rows), pd.DataFrame(summary)


def add_dark(ax):
    fig = ax.figure
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.grid(True, color="#334155", linewidth=0.55, alpha=0.70)
    ax.tick_params(colors="#dbeafe", labelsize=9)
    ax.xaxis.label.set_color("#f8fafc")
    ax.yaxis.label.set_color("#f8fafc")
    ax.title.set_color("#f8fafc")
    for spine in ax.spines.values():
        spine.set_color("#94a3b8")


def legend(ax, loc="best"):
    leg = ax.legend(loc=loc, fontsize=8.1, facecolor="#020617", edgecolor="#475569")
    for t in leg.get_texts():
        t.set_color("#f8fafc")


def plot_triplet_vs_z(df):
    import numpy as np
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16.6, 9.0))
    add_dark(ax)
    zgrid = np.linspace(0, 15.5, 500)
    styles = ["-", "--", ":"]
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    for idx, (n, line, rest, species) in enumerate(TRIPLET):
        ax.plot(zgrid, rest * (1.0 + zgrid), styles[idx], color=colors[idx], linewidth=1.65,
                label=f"{n}: {line}, λobs={rest:.6f}(1+z)")
    rest_avg = sum(x[2] for x in TRIPLET) / 3.0
    ax.plot(zgrid, rest_avg * (1.0 + zgrid), color="#f97316", linewidth=2.15,
            label=f"orange average: mean rest λ={rest_avg:.6f} µm times (1+z)")
    for name, z, status in REDSHIFTS:
        ax.axvline(z, linewidth=0.80, alpha=0.50, color="#cbd5e1")
        ax.text(z, 0.55, name, color="#e0f2fe", fontsize=8.0, rotation=90, ha="right", va="bottom")
    ax.axvspan(0.6, 5.3, color="#64748b", alpha=0.07)
    ax.axvspan(5.0, 12.0, color="#64748b", alpha=0.10)
    ax.text(2.9, 10.7, "NIRSpec wavelength range context", color="#94a3b8", fontsize=8.4, ha="center")
    ax.text(8.5, 10.7, "MIRI/MRS wavelength range context", color="#94a3b8", fontsize=8.4, ha="center")
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 11.2)
    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Observed wavelength, micron")
    ax.set_title(f"{VERSION} — three triplet curves plus orange average curve")
    legend(ax, "upper left")
    fig.tight_layout()
    path = PNG / f"{VERSION}_THREE_TRIPLET_CURVES_PLUS_AVERAGE.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_high_z_positions(df, summary):
    import pandas as pd
    import matplotlib.pyplot as plt
    sub = df[df["line_set"] == "H-alpha plus [N II] optical triplet"].copy()
    order = [r[0] for r in REDSHIFTS]
    sub["object_or_case"] = pd.Categorical(sub["object_or_case"], categories=order, ordered=True)
    sub = sub.sort_values(["object_or_case", "line_number"])
    summary = summary.copy()
    summary["object_or_case"] = pd.Categorical(summary["object_or_case"], categories=order, ordered=True)
    summary = summary.sort_values("object_or_case")
    ypos = {name: i for i, name in enumerate(order)}
    colors = ["#38bdf8", "#fb7185", "#a78bfa"]
    fig, ax = plt.subplots(figsize=(16.8, 8.8))
    add_dark(ax)
    for idx, (n, line, rest, species) in enumerate(TRIPLET):
        s = sub[sub["line_number"] == n]
        ax.plot(s["observed_um"], [ypos[x] for x in s["object_or_case"]], marker="o",
                linewidth=1.20, markersize=5.6, color=colors[idx], label=f"{n}: {line}")
    ax.plot(summary["triplet_observed_average_um"], [ypos[x] for x in summary["object_or_case"]],
            marker="D", markersize=6.6, linewidth=1.8, color="#f97316", label="orange diamonds: numerical average of three observed wavelengths")
    for _, row in summary.iterrows():
        y = ypos[row["object_or_case"]]
        ax.text(row["triplet_observed_average_um"], y + 0.14,
                f"sum={row['triplet_observed_sum_um']:.6f}\navg={row['triplet_observed_average_um']:.6f}",
                color="#fed7aa", fontsize=8.0, ha="center", va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, color="#dbeafe")
    ax.set_xlabel("Predicted observed wavelength, micron")
    ax.set_ylabel("Galaxy / comparison case")
    ax.set_title(f"{VERSION} — high-z triplet positions with explicit sum and average markers")
    ax.set_xlim(9.0, 10.65)
    legend(ax, "lower right")
    fig.tight_layout()
    path = PNG / f"{VERSION}_HIGH_Z_TRIPLET_SUM_AVERAGE_POSITIONS.png"
    fig.savefig(path, dpi=280, facecolor=fig.get_facecolor())
    plt.show()
    return path


def plot_mom_z14_table(df, summary):
    import matplotlib.pyplot as plt
    mom_lines = df[(df["object_or_case"] == "MoM-z14") & (df["line_set"] == "H-alpha plus [N II] optical triplet")].copy()
    mom_summary = summary[summary["object_or_case"] == "MoM-z14"].iloc[0]
    rows = []
    for _, row in mom_lines.sort_values("line_number").iterrows():
        rows.append([
            int(row["line_number"]),
            row["line"],
            f"{row['rest_um']:.6f}",
            f"{row['observed_um']:.6f}",
            f"{row['slope_1_plus_z']:.6f}",
        ])
    rows.append(["Σ", "SUM OF 3", f"{mom_summary['triplet_rest_sum_um']:.6f}", f"{mom_summary['triplet_observed_sum_um']:.6f}", ""])
    rows.append(["μ", "AVERAGE OF 3", f"{mom_summary['triplet_rest_average_um']:.6f}", f"{mom_summary['triplet_observed_average_um']:.6f}", f"{mom_summary['average_stretch_check']:.6f}"])
    fig, ax = plt.subplots(figsize=(16.2, 5.4))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.axis("off")
    ax.set_title(f"{VERSION} — MoM-z14 triplet sum and average check", color="#f8fafc", fontsize=14.2, pad=14)
    table = ax.table(cellText=rows, colLabels=["#", "line", "rest λ µm", "observed λ µm", "1+z / stretch"], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
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
            if r >= 4:
                cell.set_facecolor("#431407")
                cell.get_text().set_color("#fed7aa")
                cell.get_text().set_weight("bold")
    fig.tight_layout()
    path = PNG / f"{VERSION}_MOM_Z14_SUM_AVERAGE_TABLE.png"
    fig.savefig(path, dpi=260, facecolor=fig.get_facecolor())
    plt.show()
    return path


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
    df, summary = build_tables()
    line_csv = CSV / f"{VERSION}_LINE_WAVELENGTHS.csv"
    summary_csv = CSV / f"{VERSION}_TRIPLET_SUM_AVERAGE_SUMMARY.csv"
    df.to_csv(line_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    p1 = plot_triplet_vs_z(df)
    p2 = plot_high_z_positions(df, summary)
    p3 = plot_mom_z14_table(df, summary)

    mom_lines = df[(df["object_or_case"] == "MoM-z14") & (df["line_set"] == "H-alpha plus [N II] optical triplet")].sort_values("line_number")
    mom = summary[summary["object_or_case"] == "MoM-z14"].iloc[0]

    print(f"CODE OUTPUT: {VERSION}\n")
    print_table([
        ("Formula", "lambda_observed = lambda_rest * (1 + z)"),
        ("Fix", "pd import/name error removed; pandas imported where used"),
        ("Orange average curve", "mean of the three observed wavelength curves at each z"),
        ("MoM-z14 z", f"{mom['z']:.6f}"),
        ("MoM-z14 sum observed um", f"{mom['triplet_observed_sum_um']:.6f}"),
        ("MoM-z14 average observed um", f"{mom['triplet_observed_average_um']:.6f}"),
        ("MoM-z14 sum rest um", f"{mom['triplet_rest_sum_um']:.6f}"),
        ("MoM-z14 average rest um", f"{mom['triplet_rest_average_um']:.6f}"),
        ("Average stretch check", f"{mom['average_stretch_check']:.6f}"),
        ("z from average wavelengths", f"{mom['z_from_average_wavelengths']:.6f}"),
        ("Line CSV", str(line_csv)),
        ("Summary CSV", str(summary_csv)),
        ("Plot 1", str(p1)),
        ("Plot 2", str(p2)),
        ("Table PNG", str(p3)),
    ], ["Field", "Value"])

    print("\nMoM-z14 LINE SUM AND AVERAGE")
    rows = []
    for _, r in mom_lines.iterrows():
        rows.append((int(r["line_number"]), r["line"], f"{r['rest_um']:.6f}", f"{r['observed_um']:.6f}", f"{r['slope_1_plus_z']:.6f}"))
    rows.append(("Σ", "SUM", f"{mom['triplet_rest_sum_um']:.6f}", f"{mom['triplet_observed_sum_um']:.6f}", ""))
    rows.append(("μ", "AVERAGE", f"{mom['triplet_rest_average_um']:.6f}", f"{mom['triplet_observed_average_um']:.6f}", f"{mom['average_stretch_check']:.6f}"))
    print_table(rows, ["#", "Line", "Rest um", "Observed um", "1+z"])

    print("\nCOMMENTS")
    print("The orange average curve is not a separate spectral feature; it is the arithmetic mean of the three predicted observed wavelengths at each z.")
    print("For MoM-z14, the table prints the three observed wavelengths, their sum, and their average.")
    print("Python/matplotlib only. No AI images.")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
