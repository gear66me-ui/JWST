from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import warnings

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from IPython.display import display
import ipywidgets as widgets
from PIL import Image

VERSION="JWST_0242"
MODEL="MODEL_D_MOMENT0_MASTER_EXPORT_WIDGET"
TARGET_GHZ=279.901
RA_DEG=53.1647632
DEC_DEG=-27.7746223
C_KMS=299792.458
OUTPUT_PIXELS=1024
MIN_BYTES=2_500_000

ROOT=Path("/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE")
CUBE=ROOT/"SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits"
DRIVE_MASTER=ROOT/"PNG"/f"{VERSION}_MODEL_D_MOMENT0_MASTER_RAW_1024.png"
LOCAL_MASTER=Path("/content")/DRIVE_MASTER.name

for p in (DRIVE_MASTER.parent, LOCAL_MASTER.parent):
    p.mkdir(parents=True,exist_ok=True)


def spec_axis(header):
    for axis in range(1,int(header["NAXIS"])+1):
        if "FREQ" in str(header.get(f"CTYPE{axis}","")).upper():
            return axis
    raise RuntimeError("No frequency axis found in FITS header")


def axis_values_ghz(header,axis):
    n=int(header[f"NAXIS{axis}"])
    pixel=np.arange(n,dtype=float)+1.0
    step=float(header.get(f"CDELT{axis}",header.get(f"CD{axis}_{axis}")))
    return (float(header[f"CRVAL{axis}"])+(pixel-float(header[f"CRPIX{axis}"]))*step)/1e9


def collapse_to_spectral_cube(array,np_spec):
    array=np.moveaxis(array,np_spec,0)
    while array.ndim>3:
        array=np.nanmean(array,axis=1)
    return array


def main():
    warnings.filterwarnings("ignore")
    if not CUBE.exists():
        raise FileNotFoundError(CUBE)

    with fits.open(CUBE,memmap=True,do_not_scale_image_data=True) as hdul:
        header=hdul[0].header
        data=hdul[0].data
        frequency_ghz=axis_values_ghz(header,spec_axis(header))
        fits_frequency_axis=spec_axis(header)
        numpy_frequency_axis=data.ndim-fits_frequency_axis

        target_x,target_y=WCS(header).celestial.world_to_pixel_values(RA_DEG,DEC_DEG)
        target_x=int(np.rint(target_x))
        target_y=int(np.rint(target_y))
        half=64
        y_slice=slice(target_y-half,target_y+half)
        x_slice=slice(target_x-half,target_x+half)

        native_mhz=float(np.nanmedian(np.abs(np.diff(frequency_ghz)))*1000.0)
        channels_per_group=max(1,int(round(5.0/native_mhz)))
        nearest=int(np.argmin(np.abs(frequency_ghz-TARGET_GHZ)))
        lower=max(0,nearest-14*channels_per_group)
        upper=min(len(frequency_ghz),nearest+14*channels_per_group)
        groups=[np.arange(i,min(i+channels_per_group,upper)) for i in range(lower,upper,channels_per_group)]
        centers_ghz=np.array([np.mean(frequency_ghz[group]) for group in groups])
        grouped_cube=np.empty((len(groups),128,128),dtype=np.float32)

        for index,group in enumerate(groups):
            selection=[slice(None)]*data.ndim
            selection[numpy_frequency_axis]=group
            selection[-2]=y_slice
            selection[-1]=x_slice
            grouped_cube[index]=np.nanmean(
                collapse_to_spectral_cube(np.asarray(data[tuple(selection)],dtype=np.float32),numpy_frequency_axis),
                axis=0,
            )
            if (index+1)%5==0 or index==len(groups)-1:
                print(f"5 MHz cube: {index+1:3d}/{len(groups):3d}")

        del data
        gc.collect()

    velocity_kms=C_KMS*(TARGET_GHZ-centers_ghz)/TARGET_GHZ
    line_selection=(velocity_kms>=-14.0)&(velocity_kms<=29.0)
    if int(line_selection.sum())<5:
        nearest_groups=np.argsort(np.abs(centers_ghz-TARGET_GHZ))[:8]
        line_selection=np.zeros(len(centers_ghz),dtype=bool)
        line_selection[nearest_groups]=True

    moment0=np.sum(grouped_cube[line_selection],axis=0)
    finite=moment0[np.isfinite(moment0)]
    if finite.size==0:
        raise RuntimeError("Moment-0 contains no finite pixels")

    vmin=float(np.nanmin(finite))
    vmax=float(np.nanmax(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax<=vmin:
        raise RuntimeError("Moment-0 display range is invalid")

    normalized=np.clip((moment0-vmin)/(vmax-vmin),0.0,1.0)
    normalized=np.nan_to_num(normalized,nan=0.0,posinf=1.0,neginf=0.0)
    rgb=(plt.get_cmap("viridis")(normalized)[...,:3]*255.0).astype(np.uint8)

    master=Image.fromarray(rgb,mode="RGB").resize(
        (OUTPUT_PIXELS,OUTPUT_PIXELS),
        Image.Resampling.NEAREST,
    )
    master.save(LOCAL_MASTER,format="PNG",compress_level=0,optimize=False)
    DRIVE_MASTER.write_bytes(LOCAL_MASTER.read_bytes())

    width,height=Image.open(LOCAL_MASTER).size
    byte_count=LOCAL_MASTER.stat().st_size
    decimal_mb=byte_count/1_000_000.0
    binary_mib=byte_count/(1024.0**2)
    size_pass=byte_count>=MIN_BYTES
    if not size_pass:
        raise RuntimeError(
            f"Master PNG is only {decimal_mb:.3f} MB; expected at least {MIN_BYTES/1_000_000:.3f} MB"
        )

    selected_centers=centers_ghz[line_selection]
    selected_velocities=velocity_kms[line_selection]

    report_html=f"""
    <div style='background:#071019;color:#eaf2ff;border:1px solid #29445f;border-radius:10px;padding:14px;font-family:Arial,sans-serif;max-width:920px'>
      <div style='font-size:20px;font-weight:700;margin-bottom:10px'>Model D Moment-0 Master Export</div>
      <table style='border-collapse:collapse;width:100%;font-size:14px'>
        <tr><td style='padding:6px'>Source FITS crop</td><td style='padding:6px;text-align:right;font-weight:700'>128 × 128 scientific pixels</td></tr>
        <tr><td style='padding:6px'>Export dimensions</td><td style='padding:6px;text-align:right;font-weight:700'>{width:,} × {height:,} pixels</td></tr>
        <tr><td style='padding:6px'>Grouped line channels</td><td style='padding:6px;text-align:right;font-weight:700'>{int(line_selection.sum())}</td></tr>
        <tr><td style='padding:6px'>Frequency span</td><td style='padding:6px;text-align:right;font-weight:700'>{selected_centers.min():.6f}–{selected_centers.max():.6f} GHz</td></tr>
        <tr><td style='padding:6px'>Velocity span</td><td style='padding:6px;text-align:right;font-weight:700'>{selected_velocities.min():.3f}–{selected_velocities.max():.3f} km/s</td></tr>
        <tr><td style='padding:6px'>PNG size</td><td style='padding:6px;text-align:right;font-weight:700'>{byte_count:,} bytes · {decimal_mb:.3f} MB · {binary_mib:.3f} MiB</td></tr>
        <tr><td style='padding:6px'>PNG compression</td><td style='padding:6px;text-align:right;font-weight:700'>Disabled · level 0</td></tr>
        <tr><td style='padding:6px'>Google Drive master</td><td style='padding:6px;text-align:right;font-weight:700'>{DRIVE_MASTER}</td></tr>
      </table>
      <div style='margin-top:10px;color:#a9c2dc'>This is the untouched Model D line-window moment-0 image: no mask, marker, axes, labels, or colorbar. The 1024×1024 export preserves the 128×128 scientific data grid by nearest-neighbor replication; it does not invent additional spatial information.</div>
    </div>
    """

    image_widget=widgets.Image(
        value=LOCAL_MASTER.read_bytes(),
        format="png",
        layout=widgets.Layout(width="900px",max_width="100%"),
    )
    download_button=widgets.Button(
        description="Download master PNG to device",
        button_style="success",
        icon="download",
        layout=widgets.Layout(width="280px"),
    )
    status=widgets.HTML("<span style='color:#9fb5cc'>The Google Drive copy is already saved. Press the button for a device download.</span>")

    def download_master(_):
        try:
            from google.colab import files
            status.value="<b style='color:#7ee787'>Starting device download…</b>"
            files.download(str(LOCAL_MASTER))
        except Exception as exc:
            status.value=f"<b style='color:#ff8f8f'>Download could not start: {exc}</b>"

    download_button.on_click(download_master)

    display(widgets.VBox([
        widgets.HTML(report_html),
        image_widget,
        widgets.HBox([download_button]),
        status,
    ]))

    print(f"CODE OUTPUT: {VERSION}")
    print(f"MODEL: {MODEL}")
    print(f"MASTER PNG: {DRIVE_MASTER}")
    print(f"FILE SIZE: {byte_count:,} bytes | {decimal_mb:.3f} MB | {binary_mib:.3f} MiB")
    print(f"LINE CHANNELS: {int(line_selection.sum())}")
    print(f"FREQUENCY RANGE GHz: {selected_centers.min():.6f} to {selected_centers.max():.6f}")
    print(f"VELOCITY RANGE km/s: {selected_velocities.min():.3f} to {selected_velocities.max():.3f}")
    print("Timestamp Colombia:",datetime.now(timezone(timedelta(hours=-5))).strftime("%Y-%m-%d %H:%M:%S %z"))
    print(f"# {VERSION}")


if __name__=="__main__":
    main()
