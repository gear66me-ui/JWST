from pathlib import Path
from datetime import datetime, timezone, timedelta
import runpy
import urllib.request
import pandas as pd
from IPython.display import display
import ipywidgets as widgets

VERSION="JWST_0244"
SOURCE_VERSION="JWST_0243"
ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
OUT_PNG=Path("/content/JWST_OUTPUT/PNG")
OUT_CSV=Path("/content/JWST_OUTPUT/CSV")

RAW=OUT_PNG/f"{SOURCE_VERSION}_MODEL_D_MOMENT0_MASTER_RAW.png"
MASKED=OUT_PNG/f"{SOURCE_VERSION}_MODEL_D_MOMENT0_SELECTIVE_MASK.png"
CSV=OUT_CSV/f"{SOURCE_VERSION}_SELECTIVE_MASK_SUMMARY.csv"
SOURCE_LOCAL=Path(f"/content/{SOURCE_VERSION}_MODEL_E_SELECTIVE_MASK_0242_STYLE_WIDGET.py")
SOURCE_URL=f"https://raw.githubusercontent.com/gear66me-ui/JWST/main/{SOURCE_VERSION}_MODEL_E_SELECTIVE_MASK_0242_STYLE_WIDGET.py"

def ensure_outputs():
    if RAW.exists() and MASKED.exists() and CSV.exists():
        return
    print("JWST_0243 outputs not found; rerunning the full FITS analysis...")
    if not SOURCE_LOCAL.exists():
        urllib.request.urlretrieve(SOURCE_URL,SOURCE_LOCAL)
    runpy.run_path(str(SOURCE_LOCAL),run_name="__main__")
    missing=[str(p) for p in (RAW,MASKED,CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError("Expected JWST_0243 outputs are missing: "+", ".join(missing))

def scientific_html(row):
    target=float(row["target_frequency_GHz"])
    centroid=float(row["centroid_GHz"])
    peak=float(row["peak_frequency_GHz"])
    values=[
        ("Mask area",f"{int(row['mask_pixels'])}","pixels","Selective binary aperture"),
        ("Integrated line statistic",f"{float(row['target_line_flux']):.8f}","aperture units","Sum over selected grouped channels"),
        ("Robust null sigma",f"{float(row['null_sigma']):.8f}","aperture units","MAD-based null dispersion"),
        ("Integrated S/N",f"{float(row['integrated_SNR']):.6f}","sigma","Target statistic divided by null sigma"),
        ("Positive-weight centroid",f"{centroid:.9f}","GHz",f"Offset {(centroid-target)*1000:+.6f} MHz"),
        ("Strongest grouped channel",f"{peak:.9f}","GHz",f"Offset {(peak-target)*1000:+.6f} MHz"),
        ("Peak channel amplitude",f"{float(row['peak_amplitude']):.8f}","aperture units","Largest grouped-channel response"),
        ("Null percentile",f"{float(row['null_percentile']):.3f}","%","Controls below the target"),
        ("Null exceedances",f"{int(row['null_exceedances'])} / {int(row['control_positions'])}","positions","Controls at or above the target"),
        ("False-alarm fraction",f"{float(row['false_alarm_fraction']):.6f}","fraction","Empirical sampled-control rate"),
        ("Velocity window",f"{float(row['velocity_min_kms']):.3f} to {float(row['velocity_max_kms']):.3f}","km/s","Fixed Model D line window"),
        ("Line channels",f"{int(row['line_channels'])}","grouped channels","Approximately 5 MHz each")
    ]
    rows="".join(
        f"<tr><td>{name}</td><td class='v'>{value}</td><td>{unit}</td><td>{note}</td></tr>"
        for name,value,unit,note in values
    )
    return f"""
    <style>
      .jwstbox{{background:#071019;color:#eaf2ff;border:1px solid #29445f;border-radius:12px;padding:16px;font-family:Arial,sans-serif}}
      .jwstbox table{{border-collapse:collapse;width:100%;font-size:14px}}
      .jwstbox th{{background:#102235;padding:9px;text-align:left;border-bottom:1px solid #29445f}}
      .jwstbox td{{padding:8px;border-bottom:1px solid #1b3044}}
      .jwstbox .v{{text-align:right;font-weight:700;color:#ffffff}}
    </style>
    <div class='jwstbox'>
      <div style='font-size:22px;font-weight:700;margin-bottom:10px'>JWST_0243 — Selective Mask Scientific Results</div>
      <table>
        <thead><tr><th>Quantity</th><th style='text-align:right'>Value</th><th>Unit</th><th>Interpretation</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """

def main():
    ensure_outputs()
    df=pd.read_csv(CSV)
    if df.empty:
        raise RuntimeError("Summary CSV is empty")
    row=df.iloc[0]

    raw_bytes=RAW.read_bytes()
    masked_bytes=MASKED.read_bytes()
    image=widgets.Image(value=raw_bytes,format="png")
    toggle=widgets.ToggleButtons(
        options=[("Raw master","raw"),("Selective mask","mask")],
        value="raw",description="Compare:"
    )
    def switch(change):
        image.value=raw_bytes if change["new"]=="raw" else masked_bytes
    toggle.observe(switch,names="value")

    display(widgets.HTML("<h3>JWST_0244 — recovered JWST_0243 images and math</h3>"))
    display(toggle)
    display(image)
    display(widgets.HTML(scientific_html(row)))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"SOURCE RESULTS: {SOURCE_VERSION}")
    print(f"MASK PIXELS: {int(row['mask_pixels'])}")
    print(f"INTEGRATED LINE STATISTIC: {float(row['target_line_flux']):.9f}")
    print(f"ROBUST NULL SIGMA: {float(row['null_sigma']):.9f}")
    print(f"INTEGRATED S/N: {float(row['integrated_SNR']):.6f}")
    print(f"CENTROID GHz: {float(row['centroid_GHz']):.9f}")
    print(f"PEAK GHz: {float(row['peak_frequency_GHz']):.9f}")
    print(f"NULL PERCENTILE: {float(row['null_percentile']):.3f}%")
    print(f"NULL EXCEEDANCES: {int(row['null_exceedances'])}/{int(row['control_positions'])}")
    print(f"FALSE-ALARM FRACTION: {float(row['false_alarm_fraction']):.6f}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")

if __name__=="__main__":
    main()
