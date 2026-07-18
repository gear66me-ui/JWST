# JWST_0196
import os, tarfile, shutil, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from google.colab import drive

warnings.filterwarnings('ignore')

VERSION = 'JWST_0196'
TARGET_GHZ = 279.901
BIN_MHZ = 5.0
WINDOW_GHZ = 0.18

DRIVE_ROOT = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S')
ARCHIVE = DRIVE_ROOT / 'X362B_SCIENCE/ARCHIVE/2023.1.00336.S_uid___A001_X362b_Xae6_001_of_001.tar'
OUT_PNG = DRIVE_ROOT / 'X362B_SCIENCE/PNG'
OUT_CSV = DRIVE_ROOT / 'X362B_SCIENCE/CSV'
WORK = Path('/content/JWST_0196_WORK')


def fmt_bytes(n):
    for u in ['B','KB','MB','GB','TB']:
        if n < 1024 or u == 'TB':
            return f'{n:.2f} {u}'
        n /= 1024


def locate_member(tf):
    exact = [m for m in tf.getmembers() if m.isfile() and
             'JADES-GS-z11-0_sci.spw22.cube.I.pbcor.fits' in m.name and
             not m.name.endswith('.gz')]
    if not exact:
        exact = [m for m in tf.getmembers() if m.isfile() and
                 'JADES-GS-z11-0_sci.spw22.cube.I.pbcor.fits' in m.name]
    if not exact:
        raise RuntimeError('Exact SPW22 primary-beam-corrected science cube not found in archive.')
    return sorted(exact, key=lambda m: m.size, reverse=True)[0]


def extract_cube(tf, member):
    WORK.mkdir(parents=True, exist_ok=True)
    dst = WORK / Path(member.name).name
    with tf.extractfile(member) as src, open(dst, 'wb') as out:
        shutil.copyfileobj(src, out, length=16 * 1024 * 1024)
    if dst.stat().st_size != member.size:
        raise RuntimeError('Extracted FITS size does not match TAR member size.')
    return dst


def spectral_cube_and_frequency(path):
    with fits.open(path, memmap=True) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        hdr = hdu.header.copy()
        data = np.asarray(hdu.data)

    ndim = data.ndim
    spec_fits_axis = None
    for i in range(1, int(hdr.get('NAXIS', ndim)) + 1):
        ctype = str(hdr.get(f'CTYPE{i}', '')).upper()
        if any(k in ctype for k in ['FREQ', 'VRAD', 'VELO', 'VOPT']):
            spec_fits_axis = i
            break
    if spec_fits_axis is None:
        raise RuntimeError('No spectral WCS axis found in FITS header.')

    spec_numpy_axis = ndim - spec_fits_axis
    axis_records = []
    for np_axis, size in enumerate(data.shape):
        fits_axis = ndim - np_axis
        axis_records.append((np_axis, fits_axis, size, str(hdr.get(f'CTYPE{fits_axis}', ''))))

    slicer = []
    kept_original_axes = []
    for np_axis, fits_axis, size, ctype in axis_records:
        if np_axis != spec_numpy_axis and size == 1:
            slicer.append(0)
        else:
            slicer.append(slice(None))
            kept_original_axes.append(np_axis)
    data = data[tuple(slicer)]
    spec_axis_after = kept_original_axes.index(spec_numpy_axis)
    data = np.moveaxis(data, spec_axis_after, 0)

    while data.ndim > 3:
        data = np.nanmean(data, axis=1)
    if data.ndim != 3:
        raise RuntimeError(f'Expected spectral cube after WCS-aware axis handling; got {data.shape}.')

    nchan = data.shape[0]
    crval = float(hdr[f'CRVAL{spec_fits_axis}'])
    crpix = float(hdr.get(f'CRPIX{spec_fits_axis}', 1.0))
    cdelt = float(hdr[f'CDELT{spec_fits_axis}'])
    cunit = str(hdr.get(f'CUNIT{spec_fits_axis}', 'Hz')).strip().lower()
    pix = np.arange(nchan, dtype=float) + 1.0
    world = crval + (pix - crpix) * cdelt

    if 'ghz' in cunit:
        freq_ghz = world
    elif 'mhz' in cunit:
        freq_ghz = world / 1e3
    elif 'khz' in cunit:
        freq_ghz = world / 1e6
    else:
        freq_ghz = world / 1e9

    order = np.argsort(freq_ghz)
    return data[order], freq_ghz[order], hdr, spec_fits_axis, axis_records


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))


def choose_position(cube, freq):
    line = np.abs(freq - TARGET_GHZ) <= 0.045
    side = ((np.abs(freq - TARGET_GHZ) >= 0.08) &
            (np.abs(freq - TARGET_GHZ) <= WINDOW_GHZ))
    if line.sum() < 2:
        line = np.argsort(np.abs(freq - TARGET_GHZ))[:5]
    line_map = np.nanmean(cube[line], axis=0)
    baseline_map = np.nanmedian(cube[side], axis=0) if np.any(side) else np.nanmedian(cube, axis=0)
    signal_map = line_map - baseline_map

    ny, nx = signal_map.shape
    border = max(4, min(ny, nx) // 12)
    valid = signal_map.copy()
    valid[:border, :] = np.nan
    valid[-border:, :] = np.nan
    valid[:, :border] = np.nan
    valid[:, -border:] = np.nan
    y, x = np.unravel_index(np.nanargmax(valid), valid.shape)
    return int(y), int(x), signal_map


def aperture_spectrum(cube, y, x, radius=2):
    yy, xx = np.ogrid[:cube.shape[1], :cube.shape[2]]
    mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
    return np.nansum(cube[:, mask], axis=1), int(mask.sum())


def subtract_baseline(freq, flux):
    fitmask = ((np.abs(freq - TARGET_GHZ) >= 0.08) &
               (np.abs(freq - TARGET_GHZ) <= WINDOW_GHZ) &
               np.isfinite(flux))
    if fitmask.sum() >= 4:
        p = np.polyfit(freq[fitmask] - TARGET_GHZ, flux[fitmask], 1)
        baseline = np.polyval(p, freq - TARGET_GHZ)
    else:
        baseline = np.full_like(flux, np.nanmedian(flux))
    return flux - baseline, baseline, fitmask


def rebin_5mhz(freq, flux):
    step = BIN_MHZ / 1000.0
    lo = np.floor(freq.min() / step) * step
    hi = np.ceil(freq.max() / step) * step
    edges = np.arange(lo, hi + step * 1.01, step)
    idx = np.digitize(freq, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        m = idx == i
        if np.any(m):
            rows.append((0.5 * (edges[i] + edges[i + 1]), np.nanmean(flux[m]), int(m.sum())))
    return pd.DataFrame(rows, columns=['frequency_GHz', 'flux_baseline_subtracted', 'native_channels'])


def centroid(freq, flux):
    m = (np.abs(freq - TARGET_GHZ) <= 0.06) & np.isfinite(flux)
    local = flux[m]
    nu = freq[m]
    if len(local) < 3:
        raise RuntimeError('Too few bins in centroid window.')
    sigma = robust_sigma(flux[(np.abs(freq - TARGET_GHZ) >= 0.09) &
                              (np.abs(freq - TARGET_GHZ) <= WINDOW_GHZ)])
    weights = np.clip(local, 0, None)
    threshold = 1.0 * sigma if np.isfinite(sigma) else 0.0
    strong = local > threshold
    if strong.sum() >= 2:
        weights = np.where(strong, weights, 0.0)
    if np.nansum(weights) <= 0:
        raise RuntimeError('No positive line weight found for centroid.')
    cen = np.nansum(nu * weights) / np.nansum(weights)
    peak = nu[np.nanargmax(local)]
    return cen, peak, sigma, int(np.count_nonzero(weights > 0))


def main():
    print(f'CODE OUTPUT: {VERSION}')
    drive.mount('/content/drive', force_remount=False)
    if not ARCHIVE.exists():
        raise FileNotFoundError(f'Archive missing: {ARCHIVE}')
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    OUT_CSV.mkdir(parents=True, exist_ok=True)

    print(f'Archive: {ARCHIVE}')
    print(f'Archive size: {fmt_bytes(ARCHIVE.stat().st_size)}')
    with tarfile.open(ARCHIVE, 'r:*') as tf:
        member = locate_member(tf)
        print(f'Cube member: {member.name}')
        print(f'Cube member size: {fmt_bytes(member.size)}')
        cube_path = extract_cube(tf, member)

    cube, freq, hdr, spec_axis, axes = spectral_cube_and_frequency(cube_path)
    print(f'WCS spectral FITS axis: {spec_axis}')
    print(f'Cube shape [channel,y,x]: {cube.shape}')
    print(f'Frequency coverage: {freq.min():.6f} to {freq.max():.6f} GHz')
    print(f'Native channel spacing: {np.nanmedian(np.diff(freq))*1000:.6f} MHz')
    if not (freq.min() <= TARGET_GHZ <= freq.max()):
        raise RuntimeError(f'Target {TARGET_GHZ:.6f} GHz is outside cube coverage.')

    y, x, signal_map = choose_position(cube, freq)
    raw, npix = aperture_spectrum(cube, y, x, radius=2)
    corrected, baseline, fitmask = subtract_baseline(freq, raw)
    native = pd.DataFrame({
        'frequency_GHz': freq,
        'aperture_flux_raw': raw,
        'baseline': baseline,
        'flux_baseline_subtracted': corrected,
    })
    rebinned = rebin_5mhz(freq, corrected)
    cen, peak, sigma, weighted_bins = centroid(
        rebinned['frequency_GHz'].to_numpy(),
        rebinned['flux_baseline_subtracted'].to_numpy())

    z_oiii = 3393.006244 / cen - 1.0
    native_csv = OUT_CSV / f'{VERSION}_SPW22_NATIVE_SPECTRUM.csv'
    bin_csv = OUT_CSV / f'{VERSION}_SPW22_5MHZ_SPECTRUM.csv'
    result_csv = OUT_CSV / f'{VERSION}_CENTROID_RESULT.csv'
    native.to_csv(native_csv, index=False)
    rebinned.to_csv(bin_csv, index=False)
    pd.DataFrame([{
        'target': 'JADES-GS-z11-0',
        'observed_cube': Path(member.name).name,
        'aperture_center_x_pixel': x,
        'aperture_center_y_pixel': y,
        'aperture_pixels': npix,
        'bin_width_MHz': BIN_MHZ,
        'flux_weighted_centroid_GHz': cen,
        'peak_5MHz_bin_GHz': peak,
        'centroid_minus_279.901_MHz': (cen - TARGET_GHZ) * 1000.0,
        'OIII88_rest_frequency_GHz': 3393.006244,
        'derived_redshift_from_centroid': z_oiii,
        'off_line_robust_sigma': sigma,
        'weighted_bins': weighted_bins,
    }]).to_csv(result_csv, index=False)

    plt.figure(figsize=(12, 7), facecolor='#0b0f14')
    ax = plt.gca()
    ax.set_facecolor('#0b0f14')
    ax.plot(rebinned['frequency_GHz'], rebinned['flux_baseline_subtracted'],
            lw=1.2, marker='o', ms=3, label='Observed ALMA SPW22, 5 MHz bins')
    ax.axvline(TARGET_GHZ, lw=1.0, ls='--', label='Reference 279.901 GHz')
    ax.axvline(cen, lw=1.5, label=f'Flux-weighted centroid {cen:.6f} GHz')
    ax.set_xlim(TARGET_GHZ - WINDOW_GHZ, TARGET_GHZ + WINDOW_GHZ)
    ax.set_xlabel('Observed frequency (GHz)')
    ax.set_ylabel('Baseline-subtracted aperture flux (native cube units)')
    ax.set_title('JADES-GS-z11-0 ALMA [O III] 88 µm — SPW22 spectrum')
    ax.grid(alpha=0.22)
    ax.legend(loc='best')
    for spine in ax.spines.values(): spine.set_alpha(0.5)
    plt.tight_layout()
    plot_path = OUT_PNG / f'{VERSION}_SPW22_5MHZ_CENTROID.png'
    plt.savefig(plot_path, dpi=180, facecolor='#0b0f14')
    plt.show()
    plt.close()

    print('')
    print('RESULT')
    print(f'Flux-weighted centroid: {cen:.6f} GHz')
    print(f'Peak 5 MHz bin:         {peak:.6f} GHz')
    print(f'Offset from 279.901:    {(cen-TARGET_GHZ)*1000:+.3f} MHz')
    print(f'Derived [O III] z:      {z_oiii:.6f}')
    print(f'Aperture center:        x={x}, y={y}, pixels={npix}')
    print(f'PNG: {plot_path}')
    print(f'CSV: {bin_csv}')
    print(f'CSV: {result_csv}')
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')


main()
