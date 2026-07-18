# JWST_0198
import io
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

for pkg in ['requests','astropy','pandas']:
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import requests
import pandas as pd
from astropy.io import votable

VERSION='JWST_0198'
TARGET_GHZ=279.901
UID='uid://A001/X3667/Xe0'
DATALINK='https://almascience.eso.org/datalink/sync'
DRIVE_ROOT=Path('/content/drive/MyDrive/JWST')
OUT_DIR=DRIVE_ROOT/'ALMA'/'JADES_GS_Z11_0_2023.1.00336.S'/'X3667_METADATA'


def parse_votable(content):
    return votable.parse_single_table(io.BytesIO(content)).to_table().to_pandas()


def sval(row,*names):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            value=row[name]
            if isinstance(value,bytes):
                value=value.decode(errors='ignore')
            return str(value)
    return ''


def main():
    print(f'CODE OUTPUT: {VERSION}')
    if not DRIVE_ROOT.exists():
        raise RuntimeError('Google Drive is not mounted at /content/drive. Mount it once, then rerun.')
    OUT_DIR.mkdir(parents=True,exist_ok=True)

    r=requests.get(DATALINK,params={'ID':UID},timeout=180)
    r.raise_for_status()
    df=parse_votable(r.content)

    rows=[]
    readme_url=''
    readme_name=''
    for _,row in df.iterrows():
        url=sval(row,'access_url','accessURL')
        desc=sval(row,'description')
        ctype=sval(row,'content_type','contentType')
        length=sval(row,'content_length','contentLength')
        name=url.split('?')[0].rstrip('/').split('/')[-1]
        try:
            size=int(float(length))
        except Exception:
            size=None
        rows.append({'filename':name,'size_bytes':size,'description':desc,'content_type':ctype,'access_url':url})
        if 'readme' in (name+' '+desc).lower() and not readme_url:
            readme_url=url
            readme_name=name or 'member.uid___A001_X3667_Xe0.README.txt'

    inv=pd.DataFrame(rows)
    inv_csv=OUT_DIR/f'{VERSION}_X3667_DATALINK.csv'
    inv.to_csv(inv_csv,index=False)
    print(f'DataLink rows: {len(inv)}')

    if not readme_url:
        raise RuntimeError('X3667 README URL was not found in DataLink metadata.')

    rr=requests.get(readme_url,timeout=180)
    rr.raise_for_status()
    text=rr.text
    readme_path=OUT_DIR/readme_name
    readme_path.write_text(text,encoding='utf-8')
    print(f'README saved: {readme_path}')

    lines=[]
    patterns=[r'spw\s*\d+',r'GHz',r'frequency',r'spectral',r'channel',r'JADES-GS-z11-0']
    for i,line in enumerate(text.splitlines(),1):
        low=line.lower()
        if any(re.search(p,low,re.I) for p in patterns):
            lines.append({'line_number':i,'text':line.strip()})

    audit=pd.DataFrame(lines)
    audit_csv=OUT_DIR/f'{VERSION}_X3667_README_MATCHES.csv'
    audit.to_csv(audit_csv,index=False)

    print('\nRelevant README lines:')
    if len(audit):
        print(audit.head(120).to_string(index=False,max_colwidth=150))
    else:
        print('No explicit SPW/frequency lines found in README.')

    archive_rows=inv[inv['filename'].str.contains('_001_of_001.tar',case=False,na=False)]
    if len(archive_rows):
        size=archive_rows.iloc[0]['size_bytes']
        print(f'X3667 science archive size: {size/1024**3:.2f} GB' if pd.notna(size) else 'X3667 science archive size: unknown')

    print(f'Target under investigation: {TARGET_GHZ:.6f} GHz')
    print(f'Inventory CSV: {inv_csv}')
    print(f'README audit CSV: {audit_csv}')
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')


main()
