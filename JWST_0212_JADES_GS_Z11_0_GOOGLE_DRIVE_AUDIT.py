from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import tarfile
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

VERSION = 'JWST_0212'
DRIVE_ROOT = Path('/content/drive/MyDrive')
PROJECT_REL = Path('JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S')
PROJECT = DRIVE_ROOT / PROJECT_REL
EXPECTED_CUBE = PROJECT / 'X3667_SCIENCE/SELECTED_CUBE/member.uid___A001_X3667_Xe0.JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits'
EXPECTED_TAR = PROJECT / 'X3667_SCIENCE/ARCHIVE/2023.1.00336.S_uid___A001_X3667_Xe0_001_of_001.tar'
OUT_CSV = Path('/content/JWST_OUTPUT/CSV')
OUT_PNG = Path('/content/JWST_OUTPUT/PNG')
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_PNG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'figure.facecolor':'#05080d','axes.facecolor':'#05080d','savefig.facecolor':'#05080d',
    'text.color':'#e8f1ff','axes.labelcolor':'#e8f1ff','axes.edgecolor':'#8aa0b8',
    'xtick.color':'#c7d4e5','ytick.color':'#c7d4e5','grid.color':'#33485f'
})


def gib(path):
    try:
        return path.stat().st_size / 1024**3
    except Exception:
        return 0.0


def sha256_head(path, nbytes=1024*1024):
    try:
        h = hashlib.sha256()
        with path.open('rb') as f:
            h.update(f.read(nbytes))
        return h.hexdigest()[:16]
    except Exception:
        return ''


def ensure_drive():
    if DRIVE_ROOT.exists() and any(DRIVE_ROOT.iterdir()):
        return 'already mounted'
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        return 'mounted now'
    except Exception as exc:
        return f'mount failed: {exc}'


def locate_files():
    roots = [PROJECT]
    if not PROJECT.exists() and DRIVE_ROOT.exists():
        roots = [DRIVE_ROOT]
    fits_hits, tar_hits, result_hits = [], [], []
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob('*'):
                if not p.is_file():
                    continue
                low = p.name.lower()
                if low.endswith('.fits') and ('spw23' in low or 'jades-gs-z11-0' in low):
                    fits_hits.append(p)
                elif low.endswith('.tar') and ('x3667' in str(p).lower() or '2023.1.00336.s' in low):
                    tar_hits.append(p)
                elif low.startswith('jwst_02') and low.endswith(('.csv','.png')):
                    result_hits.append(p)
        except Exception:
            pass
    return fits_hits, tar_hits, result_hits


def inspect_tar(path):
    if not path.exists():
        return False, '', 0
    try:
        with tarfile.open(path, 'r:*') as tf:
            names = tf.getnames()
        matches = [n for n in names if n.endswith('JADES-GS-z11-0_sci.spw23.cube.I.pbcor.fits')]
        return bool(matches), (matches[0] if matches else ''), len(names)
    except Exception as exc:
        return False, f'ERROR: {exc}', 0


def main():
    mount_status = ensure_drive()
    fits_hits, tar_hits, result_hits = locate_files()

    cube_exists = EXPECTED_CUBE.exists()
    tar_exists = EXPECTED_TAR.exists()
    tar_member_found, tar_member, tar_count = inspect_tar(EXPECTED_TAR) if tar_exists else (False, '', 0)

    rows = []
    rows.append({'item':'Google Drive mount','status':'PASS' if DRIVE_ROOT.exists() else 'FAIL','size_GiB':0.0,'path':str(DRIVE_ROOT),'detail':mount_status})
    rows.append({'item':'JWST project folder','status':'PASS' if PROJECT.exists() else 'MISSING','size_GiB':0.0,'path':str(PROJECT),'detail':'Expected project root'})
    rows.append({'item':'Selected SPW23 cube','status':'PASS' if cube_exists else 'MISSING','size_GiB':gib(EXPECTED_CUBE),'path':str(EXPECTED_CUBE),'detail':f'head_sha256={sha256_head(EXPECTED_CUBE)}' if cube_exists else 'Not found at expected persistent path'})
    rows.append({'item':'X3667 archive','status':'PASS' if tar_exists else 'MISSING','size_GiB':gib(EXPECTED_TAR),'path':str(EXPECTED_TAR),'detail':f'members={tar_count}; selected_cube_inside={tar_member_found}' if tar_exists else 'Archive not found'})
    rows.append({'item':'Recoverability','status':'PASS' if (cube_exists or tar_member_found) else 'FAIL','size_GiB':0.0,'path':tar_member,'detail':'Cube available directly' if cube_exists else ('Cube can be re-extracted from archive' if tar_member_found else 'Neither cube nor recoverable archive member found')})
    rows.append({'item':'Candidate FITS files','status':'INFO','size_GiB':sum(gib(p) for p in fits_hits),'path':f'{len(fits_hits)} files','detail':'Largest: '+(str(max(fits_hits,key=gib)) if fits_hits else 'none')})
    rows.append({'item':'Candidate TAR files','status':'INFO','size_GiB':sum(gib(p) for p in tar_hits),'path':f'{len(tar_hits)} files','detail':'Largest: '+(str(max(tar_hits,key=gib)) if tar_hits else 'none')})
    rows.append({'item':'Saved JWST result files','status':'INFO','size_GiB':sum(gib(p) for p in result_hits),'path':f'{len(result_hits)} PNG/CSV files','detail':'Drive-side diagnostic outputs located'})

    df = pd.DataFrame(rows)
    csv = OUT_CSV / f'{VERSION}_GOOGLE_DRIVE_AUDIT.csv'
    df.to_csv(csv, index=False)

    if PROJECT.exists():
        audit_dir = PROJECT / 'AUDIT'
        audit_dir.mkdir(parents=True, exist_ok=True)
        drive_csv = audit_dir / csv.name
        df.to_csv(drive_csv, index=False)
    else:
        drive_csv = None

    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.axis('off')
    table_df = df[['item','status','size_GiB','detail']].copy()
    table_df['size_GiB'] = table_df['size_GiB'].map(lambda x: f'{x:.3f}')
    tbl = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc='center', cellLoc='left', colLoc='left', colWidths=[0.22,0.10,0.11,0.57])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1,1.65)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#33485f')
        cell.set_facecolor('#101923' if r else '#1b2a3a')
        cell.get_text().set_color('#e8f1ff')
    ax.set_title('JADES-GS-z11-0 — Google Drive persistence audit', fontsize=16, pad=20)
    png = OUT_PNG / f'{VERSION}_GOOGLE_DRIVE_AUDIT.png'
    fig.tight_layout(); fig.savefig(png, dpi=190, bbox_inches='tight'); plt.close(fig)
    if PROJECT.exists():
        drive_png = PROJECT / 'AUDIT' / png.name
        drive_png.write_bytes(png.read_bytes())
    else:
        drive_png = None

    print(f'CODE OUTPUT: {VERSION}')
    print(df[['item','status','size_GiB','detail']].to_string(index=False, float_format=lambda x:f'{x:.3f}'))
    print(f'Expected cube exists: {cube_exists}')
    print(f'Expected archive exists: {tar_exists}')
    print(f'Cube member recoverable from archive: {tar_member_found}')
    if tar_member_found:
        print(f'Archive member: {tar_member}')
    print(f'CSV: {csv}')
    print(f'PNG: {png}')
    if drive_csv: print(f'Drive CSV: {drive_csv}')
    if drive_png: print(f'Drive PNG: {drive_png}')
    print('Timestamp Colombia:', datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')


if __name__ == '__main__':
    main()
