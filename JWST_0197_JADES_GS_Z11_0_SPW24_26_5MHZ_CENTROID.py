# JWST_0197
import os, tarfile, shutil, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

VERSION = 'JWST_0197'
TARGET_GHZ = 279.901000
ARCHIVE = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X362B_SCIENCE/ARCHIVE/2023.1.00336.S_uid___A001_X362b_Xae6_001_of_001.tar')
OUT_DRIVE = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/RESULTS/JWST_0197')
OUT_PNG = Path('/content/JWST_OUTPUT/PNG')
OUT_CSV = Path('/content/JWST_OUTPUT/CSV')
TMP = Path('/content/JWST_0197_TEMP')


def spectral_axis(header):
    naxis = int(header.get('NAXIS', 0))
    for i in range(1, naxis + 1):
        ctype = str(header.get(f'CTYPE{i}', '')).upper()
        if any(k in ctype for k in ('FREQ', 'VRAD', 'VELO', 'VOPT')):
            return i
    raise RuntimeError('No spectral FITS axis found.')


def frequency_grid_ghz(header, fits_axis):
    n = int(header[f'NAXIS{fits_axis}'])
    crval = float(header[f'CRVAL{fits_axis}'])
    crpix = float(header.get(f'CRPIX{fits_axis}', 1.0))
    cdelt = float(header[f'CDELT{fits_axis}'])
    unit = str(header.get(f'CUNIT{fits_axis}', 'Hz')).strip().lower()
    pix = np.arange(n, dtype=float) + 1.0
    values = crval + (pix - crpix) * cdelt
    if unit == 'hz': values /= 1e9
    elif unit == 'khz': values /= 1e6
    elif unit == 'mhz': values /= 1e3
    elif unit == 'ghz': pass
    else:
        ctype = str(header.get(f'CTYPE{fits_axis}', '')).upper()
        if 'FREQ' not in ctype:
            raise RuntimeError(f'Unsupported spectral unit {unit!r} for {ctype}.')
        values /= 1e9
    return values


def candidate_members(tf):
    members = []
    for m in tf.getmembers():
        name = m.name
        low = name.lower()
        if ('jades-gs-z11-0_sci.spw24.cube.i.pbcor.fits' in low or
            'jades-gs-z11-0_sci.spw26.cube.i.pbcor.fits' in low):
            members.append(m)
    return sorted(members, key=lambda m: m.name)


def extract_member(tf, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tf.extractfile(member) as src, open(destination, 'wb') as dst:
        shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
    return destination


def inspect_cube(path):
    with fits.open(path, memmap=True) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        hdr = hdu.header.copy()
        shape = tuple(hdu.data.shape)
    fax = spectral_axis(hdr)
    freq = frequency_grid_ghz(hdr, fax)
    return hdr, shape, fax, freq


def cube_channel_first(data, hdr, fits_axis):
    arr = np.asarray(data)
    numpy_axis = arr.ndim - fits_axis
    arr = np.moveaxis(arr, numpy_axis, 0)
    while arr.ndim > 3:
        singleton = next((i for i in range(1, arr.ndim) if arr.shape[i] == 1), None)
        if singleton is None:
            arr = arr[:, 0]
        else:
            arr = np.take(arr, 0, axis=singleton)
    if arr.ndim != 3:
        raise RuntimeError(f'Unexpected cube shape after axis handling: {arr.shape}')
    return arr


def robust_continuum(cube, freq):
    side = (np.abs(freq - TARGET_GHZ) >= 0.12) & (np.abs(freq - TARGET_GHZ) <= 0.45)
    if side.sum() < 20:
        side = np.ones(freq.size, dtype=bool)
    return np.nanmedian(cube[side], axis=0)


def source_position(cube, freq):
    cont = robust_continuum(cube, freq)
    line_sel = np.abs(freq - TARGET_GHZ) <= 0.10
    residual = cube[line_sel] - cont
    moment = np.nansum(np.clip(residual, 0, None), axis=0)
    edge = max(8, int(min(moment.shape) * 0.08))
    work = moment.copy()
    work[:edge, :] = np.nan; work[-edge:, :] = np.nan
    work[:, :edge] = np.nan; work[:, -edge:] = np.nan
    y, x = np.unravel_index(np.nanargmax(work), work.shape)
    return int(y), int(x), cont, moment


def beam_radius_pixels(hdr):
    pix_deg = abs(float(hdr.get('CDELT1', 1/3600)))
    bmaj = abs(float(hdr.get('BMAJ', 0)))
    if bmaj > 0 and pix_deg > 0:
        return float(np.clip(0.65 * bmaj / pix_deg, 2.0, 10.0))
    return 4.0


def aperture_spectrum(cube, cont, y0, x0, radius):
    yy, xx = np.indices(cube.shape[1:])
    rr = np.hypot(xx - x0, yy - y0)
    aperture = rr <= radius
    annulus = (rr >= radius * 1.8) & (rr <= radius * 2.8)
    residual = cube - cont
    src = np.nansum(residual[:, aperture], axis=1)
    if annulus.sum() > 10:
        bg = np.nanmedian(residual[:, annulus], axis=1) * aperture.sum()
        src = src - bg
    return src, aperture


def rebin_5mhz(freq, flux):
    order = np.argsort(freq)
    f = freq[order]; s = flux[order]
    step = 0.005
    lo = np.floor(f.min() / step) * step
    hi = np.ceil(f.max() / step) * step
    edges = np.arange(lo, hi + step * 1.01, step)
    idx = np.digitize(f, edges) - 1
    rows = []
    for i in range(len(edges) - 1):
        use = idx == i
        if not np.any(use): continue
        rows.append((0.5 * (edges[i] + edges[i+1]), np.nanmean(s[use]), int(use.sum())))
    return pd.DataFrame(rows, columns=['frequency_GHz', 'flux_sum_native_units', 'native_channels'])


def centroid_from_bins(df):
    search = df[np.abs(df.frequency_GHz - TARGET_GHZ) <= 0.20].copy()
    if search.empty: raise RuntimeError('No 5 MHz bins near target frequency.')
    outer = search[np.abs(search.frequency_GHz - TARGET_GHZ) >= 0.10]
    baseline = float(np.nanmedian(outer.flux_sum_native_units)) if len(outer) else 0.0
    search['line_flux'] = search.flux_sum_native_units - baseline
    peak_i = search.line_flux.idxmax()
    peak_f = float(search.loc[peak_i, 'frequency_GHz'])
    line = search[np.abs(search.frequency_GHz - peak_f) <= 0.050].copy()
    weights = np.clip(line.line_flux.to_numpy(float), 0, None)
    if not np.any(weights > 0): raise RuntimeError('No positive line signal found for centroid.')
    centroid = float(np.sum(line.frequency_GHz.to_numpy(float) * weights) / np.sum(weights))
    return centroid, peak_f, baseline, search, line


def main():
    print(f'CODE OUTPUT: {VERSION}')
    if not ARCHIVE.exists(): raise FileNotFoundError(f'Archive not found: {ARCHIVE}')
    if not Path('/content/drive/MyDrive').exists(): raise RuntimeError('Google Drive is not mounted. Mount it once, then rerun.')
    OUT_DRIVE.mkdir(parents=True, exist_ok=True)
    OUT_PNG.mkdir(parents=True, exist_ok=True)
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)

    inventory = []
    selected = None
    selected_path = None
    with tarfile.open(ARCHIVE, 'r') as tf:
        members = candidate_members(tf)
        if not members: raise RuntimeError('SPW24/SPW26 pbcor cubes not found in archive.')
        for member in members:
            spw = 24 if 'spw24' in member.name.lower() else 26
            tmp_path = TMP / f'spw{spw}.pbcor.fits'
            print(f'Inspecting SPW{spw}: {member.size/1024**2:.2f} MB')
            extract_member(tf, member, tmp_path)
            hdr, shape, fax, freq = inspect_cube(tmp_path)
            covers = bool(freq.min() <= TARGET_GHZ <= freq.max())
            inventory.append({'spw': spw, 'member': member.name, 'size_MB': member.size/1024**2,
                              'shape': str(shape), 'spectral_FITS_axis': fax,
                              'frequency_min_GHz': float(freq.min()), 'frequency_max_GHz': float(freq.max()),
                              'native_spacing_MHz': float(np.nanmedian(np.abs(np.diff(freq))) * 1000),
                              'covers_279_901_GHz': covers})
            print(f'  coverage {freq.min():.6f} to {freq.max():.6f} GHz | covers target: {covers}')
            if covers and selected is None:
                selected = (member, spw, hdr, fax, freq)
                selected_path = tmp_path
            else:
                tmp_path.unlink(missing_ok=True)

    inv = pd.DataFrame(inventory)
    inv_csv = OUT_DRIVE / f'{VERSION}_SPW_COVERAGE.csv'
    inv.to_csv(inv_csv, index=False)
    inv.to_csv(OUT_CSV / inv_csv.name, index=False)
    print(inv[['spw','frequency_min_GHz','frequency_max_GHz','native_spacing_MHz','covers_279_901_GHz']].to_string(index=False))
    if selected is None: raise RuntimeError('Neither SPW24 nor SPW26 covers 279.901 GHz.')

    member, spw, hdr, fax, freq = selected
    with fits.open(selected_path, memmap=True) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        cube = cube_channel_first(hdu.data, hdr, fax)
        if cube.shape[0] != freq.size:
            raise RuntimeError(f'Spectral length mismatch: cube={cube.shape[0]}, WCS={freq.size}')
        y0, x0, cont, moment = source_position(cube, freq)
        radius = beam_radius_pixels(hdr)
        flux, aperture = aperture_spectrum(cube, cont, y0, x0, radius)
        bins = rebin_5mhz(freq, flux)
        centroid, peak_f, baseline, search, line = centroid_from_bins(bins)

    result = pd.DataFrame([{
        'target': 'JADES-GS-z11-0', 'selected_spw': spw,
        'requested_frequency_GHz': TARGET_GHZ, 'peak_5MHz_bin_GHz': peak_f,
        'flux_weighted_centroid_GHz': centroid,
        'centroid_minus_requested_MHz': (centroid - TARGET_GHZ) * 1000,
        'source_x_pixel': x0, 'source_y_pixel': y0, 'aperture_radius_pixels': radius,
        'method': 'continuum-subtracted aperture spectrum; 5 MHz mean bins; positive-flux centroid within ±50 MHz of local peak'
    }])

    bins_csv = OUT_DRIVE / f'{VERSION}_SPW{spw}_5MHZ_SPECTRUM.csv'
    result_csv = OUT_DRIVE / f'{VERSION}_CENTROID_RESULT.csv'
    bins.to_csv(bins_csv, index=False); result.to_csv(result_csv, index=False)
    bins.to_csv(OUT_CSV / bins_csv.name, index=False); result.to_csv(OUT_CSV / result_csv.name, index=False)

    plot_df = bins[np.abs(bins.frequency_GHz - TARGET_GHZ) <= 0.30]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(plot_df.frequency_GHz, plot_df.flux_sum_native_units, lw=1.2)
    ax.axvline(TARGET_GHZ, ls='--', lw=1.0, label=f'Reference {TARGET_GHZ:.6f} GHz')
    ax.axvline(centroid, ls='-', lw=1.4, label=f'Flux-weighted centroid {centroid:.6f} GHz')
    ax.set_title(f'JADES-GS-z11-0 ALMA SPW{spw} — 5 MHz Spectrum')
    ax.set_xlabel('Observed frequency (GHz)')
    ax.set_ylabel('Continuum-subtracted aperture flux (native summed units)')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png = OUT_DRIVE / f'{VERSION}_SPW{spw}_5MHZ_CENTROID.png'
    fig.savefig(png, dpi=180, bbox_inches='tight')
    fig.savefig(OUT_PNG / png.name, dpi=180, bbox_inches='tight')
    plt.show()

    selected_path.unlink(missing_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f'Selected spectral window: SPW{spw}')
    print(f'Peak 5 MHz bin: {peak_f:.6f} GHz')
    print(f'Flux-weighted centroid: {centroid:.6f} GHz')
    print(f'Offset from 279.901000 GHz: {(centroid-TARGET_GHZ)*1000:+.3f} MHz')
    print(f'CSV: {bins_csv}')
    print(f'Result: {result_csv}')
    print(f'Plot: {png}')
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')


main()
