from pathlib import Path
from datetime import datetime, timezone, timedelta
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import label

VERSION = "JWST_0203"
TARGET_GHZ = 279.901
LINE_HALF_WIDTH_GHZ = 0.035
BIN_MHZ = 5.0
N_MC = 1000
ROOT = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
CUBE = ROOT/'SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
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


def bin_indices(freq):
    lo = TARGET_GHZ - 0.45
    hi = TARGET_GHZ + 0.45
    step = BIN_MHZ/1000.0
    edges = np.arange(lo, hi + step, step)
    centers = 0.5*(edges[:-1]+edges[1:])
    groups=[]
    for a,b in zip(edges[:-1], edges[1:]):
        groups.append(np.where((freq>=a)&(freq<b))[0])
    return centers, groups


def robust_rms(x):
    x=np.asarray(x)
    x=x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med=np.median(x)
    mad=np.median(np.abs(x-med))
    return 1.4826*mad if mad>0 else np.std(x)


def collapse_group_to_image(arr, spectral_axis):
    arr = np.moveaxis(arr, spectral_axis, 0)
    while arr.ndim > 3:
        if arr.shape[1] == 1:
            arr = arr[:, 0, ...]
        else:
            arr = np.nanmean(arr, axis=1)
    if arr.ndim != 3:
        raise RuntimeError(f'Unexpected grouped cube shape after collapse: {arr.shape}')
    return np.nanmean(arr, axis=0, dtype=np.float64).astype(np.float32)


def main():
    if not CUBE.exists():
        raise FileNotFoundError(CUBE)

    with fits.open(CUBE, memmap=True, do_not_scale_image_data=True) as hdul:
        hdu=hdul[0]
        hdr=hdu.header
        data=hdu.data
        fax=find_spectral_axis(hdr)
        freq=axis_world(hdr,fax)
        np_spec=data.ndim-fax
        shape=data.shape
        cy,cx=shape[-2]//2, shape[-1]//2
        cut=96
        ys=slice(cy-cut//2, cy+cut//2)
        xs=slice(cx-cut//2, cx+cut//2)
        centers, groups=bin_indices(freq)
        valid=[i for i,g in enumerate(groups) if len(g)]
        centers=centers[valid]
        groups=[groups[i] for i in valid]
        nb=len(groups)
        ny=ys.stop-ys.start
        nx=xs.stop-xs.start
        cube5=np.empty((nb,ny,nx),dtype=np.float32)

        for j,g in enumerate(groups):
            sl=[slice(None)]*data.ndim
            sl[np_spec]=g
            sl[-2]=ys
            sl[-1]=xs
            arr=np.asarray(data[tuple(sl)],dtype=np.float32)
            cube5[j]=collapse_group_to_image(arr,np_spec)
            del arr
            if (j+1)%25==0 or j+1==nb:
                print(f'5 MHz cube build: {j+1:4d}/{nb:4d}')
        del data
        gc.collect()

    yy,xx=np.indices((ny,nx))
    rr=np.hypot(xx-(nx-1)/2.0,yy-(ny-1)/2.0)
    offmask=rr>20
    rms_chan=np.array([robust_rms(im[offmask]) for im in cube5])
    line_sel=np.abs(centers-TARGET_GHZ)<=LINE_HALF_WIDTH_GHZ
    if line_sel.sum()<3:
        raise RuntimeError('Insufficient line bins')

    weights=np.ones(line_sel.sum(),dtype=float)
    mask=np.zeros((ny,nx),dtype=bool)
    diagnostics=[]
    for it in range(3):
        sub=cube5[line_sel]
        norm=np.sum(np.abs(weights))
        if not np.isfinite(norm) or norm == 0:
            weights=np.ones_like(weights)
            norm=np.sum(weights)
        sb=np.tensordot(weights,sub,axes=(0,0))/norm
        map_rms=robust_rms(sb[offmask])
        sn=sb/map_rms
        cand=(sn>=2.0)&(rr<16)
        labs,nlab=label(cand)
        if nlab:
            components=[]
            for k in range(1,nlab+1):
                m=labs==k
                if m.any():
                    components.append((float(np.nansum(sn[m])),m))
            mask=max(components,key=lambda t:t[0])[1]
        if not mask.any():
            mask=rr<=4
        spec=np.array([np.nansum(im[mask]) for im in cube5])
        weights=spec[line_sel].copy()
        if not np.any(np.isfinite(weights)) or np.allclose(np.nan_to_num(weights),0):
            weights=np.ones_like(weights)
        diagnostics.append((it+1,int(mask.sum()),float(np.nanmax(sn)),float(map_rms)))

    spec=np.array([np.nansum(im[mask]) for im in cube5])
    nbeam=max(mask.sum(),1)
    err=rms_chan*np.sqrt(nbeam)
    fline=centers[line_sel]
    sline=spec[line_sel]
    eline=err[line_sel]
    denom=np.sum(sline)
    centroid=np.sum(fline*sline)/denom if np.isfinite(denom) and denom!=0 else np.nan

    rng=np.random.default_rng(20260717)
    mc=np.empty(N_MC)
    for i in range(N_MC):
        sm=sline+rng.normal(0,eline)
        d=np.sum(sm)
        mc[i]=np.sum(fline*sm)/d if np.isfinite(d) and d!=0 else np.nan
    cent_err=np.nanstd(mc)
    int_flux=np.sum(sline)
    int_err=np.sqrt(np.sum(eline**2))
    snr=int_flux/int_err if int_err>0 else np.nan

    df=pd.DataFrame({
        'frequency_GHz':centers,
        'flux_sum_mJy_beam':spec*1000,
        'sigma_mJy_beam':err*1000,
        'is_line_window':line_sel
    })
    csv=OUT_CSV/f'{VERSION}_PAPER_METHOD_SPECTRUM.csv'
    df.to_csv(csv,index=False)
    df.to_csv(DRIVE_CSV/csv.name,index=False)

    my,mx=np.where(mask)
    mask_df=pd.DataFrame({
        'y_cutout':my,
        'x_cutout':mx,
        'y_full':my+ys.start,
        'x_full':mx+xs.start
    })
    mask_csv=OUT_CSV/f'{VERSION}_EXTRACTION_MASK_PIXELS.csv'
    mask_df.to_csv(mask_csv,index=False)
    mask_df.to_csv(DRIVE_CSV/mask_csv.name,index=False)

    plt.figure(figsize=(14,7.5))
    plt.plot(centers,spec*1000,lw=1.4,label='Iterative weighted extraction')
    plt.fill_between(centers,-err*1000,err*1000,alpha=.18,label='±1σ')
    plt.axvline(TARGET_GHZ,ls='--',lw=1.4,label='279.901 GHz reference')
    if np.isfinite(centroid):
        plt.axvline(centroid,ls=':',lw=1.4,label=f'Centroid {centroid:.6f} GHz')
    plt.axvspan(TARGET_GHZ-LINE_HALF_WIDTH_GHZ,TARGET_GHZ+LINE_HALF_WIDTH_GHZ,alpha=.08)
    plt.xlabel('Observed frequency (GHz)')
    plt.ylabel('Integrated aperture flux (mJy beam⁻¹)')
    plt.title('JADES-GS-z11-0 — paper-style iterative ALMA extraction')
    plt.grid(alpha=.2)
    plt.legend()
    plt.tight_layout()
    png=OUT_PNG/f'{VERSION}_PAPER_METHOD_SPECTRUM.png'
    plt.savefig(png,dpi=180)
    plt.close()
    (DRIVE_PNG/png.name).write_bytes(png.read_bytes())

    img=np.sum(cube5[line_sel],axis=0)
    plt.figure(figsize=(8,7))
    plt.imshow(img,origin='lower')
    plt.contour(mask.astype(float),levels=[0.5],linewidths=1.2)
    plt.title('Line-window surface-brightness map and extraction mask')
    plt.xlabel('Cutout x pixel')
    plt.ylabel('Cutout y pixel')
    plt.tight_layout()
    mpng=OUT_PNG/f'{VERSION}_LINE_MAP_MASK.png'
    plt.savefig(mpng,dpi=180)
    plt.close()
    (DRIVE_PNG/mpng.name).write_bytes(mpng.read_bytes())

    print(f'CODE OUTPUT: {VERSION}')
    print(f'Cube: {CUBE}')
    print(f'Cube shape: {shape}')
    print(f'5 MHz bins analyzed: {nb}')
    for it,npix,peak,mrms in diagnostics:
        print(f'Iteration {it}: mask pixels={npix:4d} | peak S/N map={peak:7.3f} | map RMS={mrms:.6e}')
    print(f'Observed flux-weighted centroid: {centroid:.6f} ± {cent_err:.6f} GHz')
    print(f'Integrated line-window S/N: {snr:.3f}')
    print(f'Reference offset: {(centroid-TARGET_GHZ)*1000:.3f} MHz')
    print(f'PNG: {png}')
    print(f'PNG: {mpng}')
    print(f'CSV: {csv}')
    print(f'CSV: {mask_csv}')
    print('Status: observed extraction; significance must be judged from measured S/N and spatial coincidence.')
    print('Timestamp Colombia:', datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')


if __name__=='__main__':
    main()
