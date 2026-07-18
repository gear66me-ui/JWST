from pathlib import Path
from datetime import datetime, timezone, timedelta
import os, sys, json, hashlib, shutil, subprocess, warnings

VERSION = 'JWST_0214'
UID = 'uid://A001/X3667/Xe0'
TARGET_TOKEN = 'JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
DRIVE = Path('/content/drive/MyDrive')
ROOT = DRIVE / 'JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE'
SELECTED = ROOT / 'SELECTED_CUBE'
CACHE = ROOT / 'ALMA_ARCHIVE_CACHE'
AUDIT = ROOT / 'AUDIT'
OUT_CSV = Path('/content/JWST_OUTPUT/CSV')
OUT_PNG = Path('/content/JWST_OUTPUT/PNG')
for p in (SELECTED, CACHE, AUDIT, OUT_CSV, OUT_PNG):
    p.mkdir(parents=True, exist_ok=True)


def ensure_drive():
    if DRIVE.exists():
        return
    from google.colab import drive
    drive.mount('/content/drive')
    if not DRIVE.exists():
        raise RuntimeError('Google Drive mount failed.')


def ensure_packages():
    try:
        import astroquery  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'astroquery'])


def sha256(path, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def verify_fits(path):
    from astropy.io import fits
    with fits.open(path, memmap=True, do_not_scale_image_data=True) as hdul:
        h = hdul[0].header
        shape = tuple(int(x) for x in hdul[0].data.shape)
        freq_axis = None
        for ax in range(1, int(h['NAXIS']) + 1):
            if 'FREQ' in str(h.get(f'CTYPE{ax}', '')).upper():
                freq_axis = ax
                break
        if freq_axis is None:
            raise RuntimeError('Recovered FITS has no frequency axis.')
        n = int(h[f'NAXIS{freq_axis}'])
        crval = float(h[f'CRVAL{freq_axis}'])
        crpix = float(h[f'CRPIX{freq_axis}'])
        cdelt = float(h.get(f'CDELT{freq_axis}', h.get(f'CD{freq_axis}_{freq_axis}')))
        f0 = (crval + (1 - crpix) * cdelt) / 1e9
        f1 = (crval + (n - crpix) * cdelt) / 1e9
        lo, hi = sorted((f0, f1))
    if path.stat().st_size < 1_000_000_000:
        raise RuntimeError('Recovered FITS is unexpectedly small.')
    if not (lo < 279.901 < hi):
        raise RuntimeError(f'Recovered cube does not cover 279.901 GHz: {lo:.6f}-{hi:.6f}')
    return shape, lo, hi


def locate_existing():
    exact = SELECTED / f'member.uid___A001_X3667_Xe0.{TARGET_TOKEN}'
    if exact.exists() and exact.stat().st_size > 1_000_000_000:
        return exact
    candidates = [p for p in ROOT.rglob(f'*{TARGET_TOKEN}') if p.is_file() and p.stat().st_size > 1_000_000_000]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def safe_text(value):
    try:
        if value is None:
            return ''
        return str(value)
    except Exception:
        return ''


def recover_direct_fits():
    from astroquery.alma import Alma
    alma = Alma()
    alma.TIMEOUT = 600
    alma.cache_location = str(CACHE)
    print('Querying ALMA archive for individual QA2 products...')

    # NumPy 2 compatibility workaround: request the unfiltered table so astroquery
    # does not call np.char.find on an object-dtype semantics column.
    info = alma.get_data_info(
        UID,
        expand_tarfiles=True,
        with_auxiliary=True,
        with_rawdata=True,
    )
    if info is None or len(info) == 0:
        raise RuntimeError('ALMA archive returned no downloadable products.')

    rows = []
    cols = set(info.colnames)
    for row in info:
        url = safe_text(row['access_url']) if 'access_url' in cols else ''
        desc = safe_text(row['description']) if 'description' in cols else ''
        sem = safe_text(row['semantics']).lower() if 'semantics' in cols else ''
        ctype = safe_text(row['content_type']).lower() if 'content_type' in cols else ''
        size_raw = row['content_length'] if 'content_length' in cols else 0
        try:
            size = int(size_raw) if size_raw is not None else 0
        except Exception:
            size = 0
        text = (url + ' ' + desc).lower()
        if '#aux' in sem or '#raw' in sem:
            continue
        if 'fits' not in text and 'fits' not in ctype:
            continue
        rows.append((url, size, desc, sem, ctype))

    matches = [r for r in rows if TARGET_TOKEN.lower() in (r[0] + ' ' + r[2]).lower()]
    if not matches:
        matches = [r for r in rows if 'spw23' in (r[0] + ' ' + r[2]).lower() and '.pbcor.fits' in (r[0] + ' ' + r[2]).lower()]
    if not matches:
        preview = '\n'.join((u + ' | ' + d)[-260:] for u, _, d, _, _ in rows[:40])
        raise RuntimeError('Direct SPW23 pbcor FITS product not found in expanded archive listing.\n' + preview)

    matches.sort(key=lambda r: r[1], reverse=True)
    url, expected, desc, _, _ = matches[0]
    print(f'Selected ALMA product: {desc or url.split("/")[-1]}')
    print(f'Expected size: {expected / 1024**3:.3f} GiB' if expected else 'Expected size: archive did not report it')

    downloaded = alma.download_files([url], savedir=str(SELECTED), cache=True, continuation=True, verify_only=False)
    if not downloaded:
        raise RuntimeError('ALMA download returned no file.')
    source = Path(str(downloaded[0]))
    if not source.exists():
        raise RuntimeError(f'Download path does not exist: {source}')

    final = SELECTED / f'member.uid___A001_X3667_Xe0.{TARGET_TOKEN}'
    if source.resolve() != final.resolve():
        tmp = final.with_suffix(final.suffix + '.part')
        if tmp.exists():
            tmp.unlink()
        shutil.copy2(source, tmp)
        os.replace(tmp, final)
    else:
        final = source
    return final, expected, url


def main():
    warnings.filterwarnings('ignore')
    ensure_drive()
    ensure_packages()
    usage = shutil.disk_usage(DRIVE)
    print(f'Google Drive free space: {usage.free / 1024**3:.2f} GiB')
    if usage.free < 4 * 1024**3:
        raise RuntimeError('At least 4 GiB of free Google Drive space is required.')

    cube = locate_existing()
    source_url = 'existing Drive file'
    expected = cube.stat().st_size if cube else 0
    if cube is None:
        cube, expected, source_url = recover_direct_fits()

    print('Verifying FITS structure and science frequency coverage...')
    shape, flo, fhi = verify_fits(cube)
    digest = sha256(cube)
    os.sync()
    persisted_size = cube.stat().st_size
    with open(cube, 'rb') as f:
        first_block = hashlib.sha256(f.read(1024 * 1024)).hexdigest()
    if persisted_size < 1_000_000_000:
        raise RuntimeError('Drive persistence verification failed: file size is too small.')

    manifest = {
        'version': VERSION,
        'uid': UID,
        'cube_path': str(cube),
        'size_bytes': persisted_size,
        'size_GiB': persisted_size / 1024**3,
        'expected_size_bytes': int(expected or 0),
        'sha256': digest,
        'first_1MiB_sha256': first_block,
        'shape': list(shape),
        'frequency_min_GHz': flo,
        'frequency_max_GHz': fhi,
        'target_frequency_GHz': 279.901,
        'source_url': source_url,
        'verified_on_google_drive': str(cube).startswith('/content/drive/MyDrive/'),
        'timestamp_colombia': datetime.now(timezone(timedelta(hours=-5))).isoformat(),
    }
    manifest_json = AUDIT / f'{VERSION}_CUBE_PERSISTENCE_MANIFEST.json'
    manifest_json.write_text(json.dumps(manifest, indent=2))

    import pandas as pd
    df = pd.DataFrame([{
        'status': 'PASS',
        'cube_path': str(cube),
        'size_GiB': manifest['size_GiB'],
        'shape': 'x'.join(map(str, shape)),
        'frequency_min_GHz': flo,
        'frequency_max_GHz': fhi,
        'sha256': digest,
        'verified_on_google_drive': True,
    }])
    csv = AUDIT / f'{VERSION}_CUBE_PERSISTENCE_AUDIT.csv'
    df.to_csv(csv, index=False)
    shutil.copy2(csv, OUT_CSV / csv.name)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d','text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8','xtick.color':'#c7d4e5','ytick.color':'#c7d4e5'})
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.axis('off')
    lines = [
        'GOOGLE DRIVE PERSISTENCE VERIFIED',
        f'Cube: {cube.name}',
        f'Size: {manifest["size_GiB"]:.3f} GiB',
        f'Shape: {shape}',
        f'Frequency coverage: {flo:.6f}–{fhi:.6f} GHz',
        'Target 279.901 GHz: COVERED',
        f'SHA-256: {digest[:24]}…',
        f'Drive path: {cube}',
    ]
    y = 0.88
    for i, line in enumerate(lines):
        ax.text(0.04, y, line, transform=ax.transAxes, fontsize=20 if i == 0 else 12.5,
                color='#4dd0e1' if i == 0 else '#e8f1ff', weight='bold' if i == 0 else 'normal')
        y -= 0.105
    png = AUDIT / f'{VERSION}_CUBE_PERSISTENCE_AUDIT.png'
    fig.savefig(png, dpi=180, bbox_inches='tight')
    plt.close(fig)
    shutil.copy2(png, OUT_PNG / png.name)

    print(f'CODE OUTPUT: {VERSION}')
    print(df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print(f'Manifest: {manifest_json}')
    print(f'Drive CSV: {csv}')
    print(f'Drive PNG: {png}')
    print('Persistence verification: PASS')
    print('Timestamp Colombia:', datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')


if __name__ == '__main__':
    main()
