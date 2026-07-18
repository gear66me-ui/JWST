from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

VERSION = "JWST_0204"
TARGET_GHZ = 279.901
BIN_MHZ = 5.0
N_MC = 1000
N_RANDOM = 500
ROOT = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE = ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
MASK_CSV = ROOT/'CSV/JWST_0203_EXTRACTION_MASK_PIXELS.csv'
OUT_PNG = Path('/content/JWST_OUTPUT/PNG')
OUT_CSV = Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG = ROOT/'PNG'
DRIVE_CSV = ROOT/'CSV'
for p in [OUT_PNG, OUT_CSV, DRIVE_PNG, DRIVE_CSV]:
    p.mkdir(parents=True, exist_ok=True)


def axis_world(header, fits_axis):
    n = int(header[f'NAXIS{fits_axis}'])
    pix = np.arange(n, dtype=float) + 1.0
    crval = float(header[f'CRVAL{fits_axis}'])
    crpix = float(header[f'CRPIX{fits_axis}'])
    cdelt = float(header.get(f'CDELT{fits_axis}', header.get(f'CD{fits_axis}_{fits_axis}')))
    vals = crval + (pix-crpix)*cdelt
    unit = str(header.get(f'CUNIT{fits_axis}','Hz')).lower()
    if unit == 'hz': vals /= 1e9
    elif unit == 'khz': vals /= 1e6
    elif unit == 'mhz': vals /= 1e3
    return vals


def find_spectral_axis(header):
    for ax in range(1, int(header['NAXIS'])+1):
        if 'FREQ' in str(header.get(f'CTYPE{ax}','')).upper():
            return ax
    raise RuntimeError('No frequency axis found')


def make_bins(freq, lo=279.84, hi=279.96):
    step = BIN_MHZ/1000.0
    edges = np.arange(lo, hi+step, step)
    centers = 0.5*(edges[:-1]+edges[1:])
    groups = [np.where((freq>=a)&(freq<b))[0] for a,b in zip(edges[:-1],edges[1:])]
    valid = [i for i,g in enumerate(groups) if len(g)]
    return centers[valid], [groups[i] for i in valid]


def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    med = np.median(x)
    mad = np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)


def main():
    if not CUBE.exists(): raise FileNotFoundError(CUBE)
    if not MASK_CSV.exists(): raise FileNotFoundError(MASK_CSV)

    mask_df = pd.read_csv(MASK_CSV)
    y_full = mask_df['y_full'].to_numpy(int)
    x_full = mask_df['x_full'].to_numpy(int)
    y0, x0 = int(np.round(np.mean(y_full))), int(np.round(np.mean(x_full)))
    dy = y_full-y0
    dx = x_full-x0

    with fits.open(CUBE, memmap=True, do_not_scale_image_data=True) as hdul:
        hdr = hdul[0].header
        data = hdul[0].data
        fax = find_spectral_axis(hdr)
        freq = axis_world(hdr, fax)
        np_spec = data.ndim-fax
        centers, groups = make_bins(freq)
        ny, nx = data.shape[-2], data.shape[-1]

        pad = 120
        ys = slice(max(0,y0-pad), min(ny,y0+pad+1))
        xs = slice(max(0,x0-pad), min(nx,x0+pad+1))
        cube5 = np.empty((len(groups), ys.stop-ys.start, xs.stop-xs.start), np.float32)
        for j,g in enumerate(groups):
            sl = [slice(None)]*data.ndim
            sl[np_spec] = g
            sl[-2] = ys
            sl[-1] = xs
            arr = np.asarray(data[tuple(sl)], dtype=np.float32)
            arr = np.moveaxis(arr, np_spec, 0)
            while arr.ndim > 3:
                arr = np.nanmean(arr, axis=1)
            cube5[j] = np.nanmean(arr, axis=0)
            if (j+1)%5==0 or j+1==len(groups):
                print(f'5 MHz line-cube build: {j+1:3d}/{len(groups):3d}')
        del data
        gc.collect()

    sy = y_full-ys.start
    sx = x_full-xs.start
    source_spec = np.array([np.nansum(im[sy,sx]) for im in cube5])

    rng = np.random.default_rng(20260717)
    random_specs = []
    attempts = 0
    while len(random_specs) < N_RANDOM and attempts < N_RANDOM*50:
        attempts += 1
        cy = rng.integers(max(-dy.min(),5), cube5.shape[1]-max(dy.max(),5))
        cx = rng.integers(max(-dx.min(),5), cube5.shape[2]-max(dx.max(),5))
        if np.hypot(cy-(y0-ys.start), cx-(x0-xs.start)) < 25:
            continue
        yy = cy+dy
        xx = cx+dx
        if yy.min()<0 or yy.max()>=cube5.shape[1] or xx.min()<0 or xx.max()>=cube5.shape[2]:
            continue
        random_specs.append([np.nansum(im[yy,xx]) for im in cube5])
    random_specs = np.asarray(random_specs, float)
    if len(random_specs) < 100:
        raise RuntimeError('Too few valid random apertures')

    sigma_chan = np.array([robust_sigma(random_specs[:,i]) for i in range(len(centers))])

    # Paper-like nine contiguous 5 MHz bins nearest the reference frequency.
    i0 = int(np.argmin(np.abs(centers-TARGET_GHZ)))
    lo = max(0, i0-4)
    hi = min(len(centers), lo+9)
    lo = max(0, hi-9)
    line_sel = np.zeros(len(centers), bool)
    line_sel[lo:hi] = True

    fline = centers[line_sel]
    sline = source_spec[line_sel]
    eline = sigma_chan[line_sel]
    denom = np.sum(sline)
    centroid = np.sum(fline*sline)/denom if denom != 0 else np.nan

    mc = np.empty(N_MC)
    for i in range(N_MC):
        sm = sline + rng.normal(0.0, eline)
        d = np.sum(sm)
        mc[i] = np.sum(fline*sm)/d if d != 0 else np.nan
    cent_err = np.nanstd(mc)

    random_int = np.sum(random_specs[:,line_sel], axis=1)
    source_int = np.sum(sline)
    int_sigma = robust_sigma(random_int)
    snr = source_int/int_sigma if int_sigma>0 else np.nan
    false_alarm = (np.sum(np.abs(random_int) >= abs(source_int)) + 1)/(len(random_int)+1)

    out = pd.DataFrame({
        'frequency_GHz': centers,
        'source_flux_sum_Jy_per_beam': source_spec,
        'empirical_sigma_Jy_per_beam': sigma_chan,
        'is_nine_bin_line_window': line_sel
    })
    spec_csv = OUT_CSV/f'{VERSION}_EMPIRICAL_NOISE_SPECTRUM.csv'
    out.to_csv(spec_csv,index=False)
    out.to_csv(DRIVE_CSV/spec_csv.name,index=False)

    mc_csv = OUT_CSV/f'{VERSION}_MONTE_CARLO_CENTROIDS.csv'
    pd.DataFrame({'centroid_GHz':mc}).to_csv(mc_csv,index=False)
    pd.read_csv(mc_csv).to_csv(DRIVE_CSV/mc_csv.name,index=False)

    null_csv = OUT_CSV/f'{VERSION}_RANDOM_APERTURE_NULL.csv'
    pd.DataFrame({'integrated_line_flux_Jy_per_beam':random_int}).to_csv(null_csv,index=False)
    pd.read_csv(null_csv).to_csv(DRIVE_CSV/null_csv.name,index=False)

    plt.figure(figsize=(14,7.5))
    plt.plot(centers, source_spec*1000, lw=1.5, label='Observed iterative-mask spectrum')
    plt.fill_between(centers,-sigma_chan*1000,sigma_chan*1000,alpha=.18,label='Empirical ±1σ from random apertures')
    plt.axvspan(fline.min()-0.0025,fline.max()+0.0025,alpha=.10,label='Nine-bin paper-like window')
    plt.axvline(TARGET_GHZ,ls='--',lw=1.4,label='Paper centroid 279.901 GHz')
    if np.isfinite(centroid):
        plt.axvline(centroid,ls=':',lw=1.4,label=f'Observed centroid {centroid:.6f} GHz')
    plt.xlabel('Observed frequency (GHz)')
    plt.ylabel('Integrated aperture flux (mJy beam⁻¹)')
    plt.title('JADES-GS-z11-0 — empirical-noise Monte Carlo centroid audit')
    plt.grid(alpha=.2)
    plt.legend()
    plt.tight_layout()
    png = OUT_PNG/f'{VERSION}_EMPIRICAL_NOISE_MONTE_CARLO.png'
    plt.savefig(png,dpi=180)
    plt.close()
    (DRIVE_PNG/png.name).write_bytes(png.read_bytes())

    print(f'CODE OUTPUT: {VERSION}')
    print(f'Random matched apertures: {len(random_specs)}')
    print(f'Line window: {fline.min():.6f} to {fline.max():.6f} GHz | bins={line_sel.sum()}')
    print(f'Observed flux-weighted centroid: {centroid:.6f} ± {cent_err:.6f} GHz')
    print(f'Paper value: 279.901000 ± 0.014000 GHz')
    print(f'Centroid offset: {(centroid-TARGET_GHZ)*1000:.3f} MHz')
    print(f'Empirical integrated-line S/N: {snr:.3f}')
    print(f'Random-aperture two-sided false-alarm fraction: {false_alarm:.6f}')
    print(f'PNG: {png}')
    print(f'CSV: {spec_csv}')
    print(f'CSV: {mc_csv}')
    print(f'CSV: {null_csv}')
    print('Status: observed centroid with empirical matched-aperture noise and 1000 Monte Carlo realizations.')
    print('Timestamp Colombia:', datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')


if __name__ == '__main__':
    main()
