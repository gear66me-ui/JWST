# JWST_0192
import io, os, sys, subprocess, warnings
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
warnings.filterwarnings('ignore')

for pkg in ['requests','astropy','pandas','numpy']:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable,'-m','pip','install','-q',pkg])

import requests
import numpy as np
import pandas as pd
from astropy.io import votable
from google.colab import drive

VERSION='JWST_0192'
PROJECT='2023.1.00336.S'
TARGET='JADES-GS-z11-0'
TARGET_GHZ=279.901
DRIVE_DIR=Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/METADATA')
MIRRORS=[
    ('ESO','https://almascience.eso.org/tap','https://almascience.eso.org/datalink/sync'),
    ('NAOJ','https://almascience.nao.ac.jp/tap','https://almascience.nao.ac.jp/datalink/sync'),
    ('NRAO','https://almascience.nrao.edu/tap','https://almascience.nrao.edu/datalink/sync'),
]


def parse_votable(content):
    return votable.parse_single_table(io.BytesIO(content)).to_table().to_pandas()


def tap_rows(tap):
    adql=f"SELECT * FROM ivoa.obscore WHERE proposal_id='{PROJECT}'"
    r=requests.get(tap+'/sync',params={'REQUEST':'doQuery','LANG':'ADQL','FORMAT':'votable','QUERY':adql},timeout=180)
    r.raise_for_status()
    return parse_votable(r.content)


def sval(row,*names):
    for n in names:
        if n in row.index and pd.notna(row[n]):
            v=row[n]
            if isinstance(v,bytes): v=v.decode(errors='ignore')
            return str(v)
    return ''


def uid_candidates(rows):
    out=[]
    for _,r in rows.iterrows():
        uid=sval(r,'member_ous_uid','obs_publisher_did','obs_id')
        text=' '.join(sval(r,n) for n in ['target_name','source_name','obs_title','proposal_title']).lower()
        score=0
        if 'jades-gs-z11-0' in text: score+=200
        elif 'jades' in text: score+=100
        if uid.startswith('uid://'): score+=20
        if uid: out.append((score,uid,text))
    seen={}
    for score,uid,text in out:
        if uid not in seen or score>seen[uid][0]: seen[uid]=(score,text)
    return sorted([(s,u,t) for u,(s,t) in seen.items()],reverse=True)


def classify(name,desc,mime,semantics,size):
    text=' '.join([name,desc,mime,semantics]).lower()
    cls=[]
    if '.asdm.sdm.tar' in text: cls.append('RAW_ASDM')
    if 'auxiliary' in text: cls.append('AUXILIARY')
    if any(k in text for k in ['calibrated','calibration']): cls.append('CALIBRATED')
    if any(k in text for k in ['image','cube','pbcor','contsub']): cls.append('IMAGE_OR_CUBE')
    if any(k in text for k in ['fits','application/fits']): cls.append('FITS')
    if any(k in text for k in ['tar','tgz','zip']): cls.append('ARCHIVE')
    if 'readme' in text: cls.append('README')
    if not cls: cls=['UNCLASSIFIED']
    return '|'.join(cls)


def product_score(name,desc,mime,semantics,size):
    text=' '.join([name,desc,mime,semantics]).lower()
    score=0
    if any(k in text for k in ['image.fits','image.pbcor.fits','cube.fits','contsub.fits']): score+=500
    if 'fits' in text: score+=200
    if any(k in text for k in ['cube','image','pbcor','contsub']): score+=150
    if TARGET.lower() in text: score+=100
    if any(k in text for k in ['spw22','spw.22','279.901']): score+=80
    if '.asdm.sdm.tar' in text: score-=1000
    if 'auxiliary' in text: score-=800
    if 'readme' in text: score-=500
    if np.isfinite(size):
        gb=size/1024**3
        if gb<5: score+=30
        if gb>15: score-=100
    return score


def inventory_mirror(label,tap,dl):
    print(f'Querying {label} TAP: {tap}')
    rows=tap_rows(tap)
    print(f'  TAP rows: {len(rows)}')
    uids=uid_candidates(rows)
    print(f'  Candidate UIDs: {len(uids)}')
    recs=[]
    for uscore,uid,_ in uids[:12]:
        print(f'  DataLink {uid} | uid_score={uscore}')
        try:
            r=requests.get(dl,params={'ID':uid},timeout=180)
            r.raise_for_status()
            products=parse_votable(r.content)
        except Exception as e:
            recs.append({'mirror':label,'uid':uid,'uid_score':uscore,'error':str(e)})
            continue
        for _,p in products.iterrows():
            url=sval(p,'access_url','accessURL')
            name=url.split('?')[0].rstrip('/').split('/')[-1]
            desc=sval(p,'description')
            mime=sval(p,'content_type','contentType')
            semantics=sval(p,'semantics')
            qualifier=sval(p,'content_qualifier')
            length=sval(p,'content_length','contentLength')
            try: size=float(length)
            except: size=np.nan
            recs.append({
                'mirror':label,'uid':uid,'uid_score':uscore,'filename':name,
                'size_bytes':size,'size_GB':size/1024**3 if np.isfinite(size) else np.nan,
                'description':desc,'content_type':mime,'semantics':semantics,
                'content_qualifier':qualifier,'access_url':url,
                'classification':classify(name,desc,mime,semantics,size),
                'science_score':product_score(name,desc,mime,semantics,size),
                'error':''
            })
    return recs


def main():
    print(f'CODE OUTPUT: {VERSION}')
    print('Metadata-only ALMA inventory: no science archive will be downloaded.')
    drive.mount('/content/drive',force_remount=False)
    DRIVE_DIR.mkdir(parents=True,exist_ok=True)
    all_records=[]
    for mirror in MIRRORS:
        try: all_records.extend(inventory_mirror(*mirror))
        except Exception as e: all_records.append({'mirror':mirror[0],'error':str(e)})
    df=pd.DataFrame(all_records)
    if 'science_score' not in df: df['science_score']=np.nan
    df=df.sort_values(['science_score','size_bytes'],ascending=[False,True],na_position='last')
    csv=DRIVE_DIR/f'{VERSION}_ALMA_PRODUCT_INVENTORY.csv'
    df.to_csv(csv,index=False)
    print(f'Inventory rows: {len(df)}')
    good=df[df.get('science_score',pd.Series(dtype=float)).fillna(-999)>0]
    print(f'Positive-score products: {len(good)}')
    cols=[c for c in ['mirror','science_score','size_GB','classification','filename','description'] if c in df]
    if len(df): print(df[cols].head(30).to_string(index=False,max_colwidth=90))
    print(f'CSV: {csv}')
    print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
    print(f'# {VERSION}')

main()
