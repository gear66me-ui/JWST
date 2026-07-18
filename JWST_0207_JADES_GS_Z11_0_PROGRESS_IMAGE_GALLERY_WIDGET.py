from pathlib import Path
from datetime import datetime, timezone, timedelta
import base64
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import ipywidgets as widgets

VERSION = 'JWST_0207'
TARGET_GHZ = 279.901
ROOT = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
OUT_PNG = Path('/content/JWST_OUTPUT/PNG')
OUT_CSV = Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG = ROOT / 'PNG'
DRIVE_CSV = ROOT / 'CSV'
for p in (OUT_PNG, OUT_CSV, DRIVE_PNG, DRIVE_CSV):
    p.mkdir(parents=True, exist_ok=True)


def latest_csv(pattern):
    files = sorted(list(OUT_CSV.glob(pattern)) + list(DRIVE_CSV.glob(pattern)))
    return files[-1] if files else None


def centroid_from_csv(path):
    if path is None or not path.exists():
        return np.nan
    try:
        df = pd.read_csv(path)
        fcol = next(c for c in df.columns if 'frequency' in c.lower())
        flux_candidates = [c for c in df.columns if 'flux' in c.lower()]
        if not flux_candidates:
            return np.nan
        scol = flux_candidates[0]
        if 'line_window' in df.columns:
            m = df['line_window'].astype(bool).to_numpy()
        elif 'is_line_window' in df.columns:
            m = df['is_line_window'].astype(bool).to_numpy()
        else:
            m = np.abs(df[fcol].to_numpy() - TARGET_GHZ) <= 0.035
        f = df.loc[m, fcol].to_numpy(float)
        s = df.loc[m, scol].to_numpy(float)
        d = np.nansum(s)
        return np.nansum(f * s) / d if np.isfinite(d) and d != 0 else np.nan
    except Exception:
        return np.nan


def build_progress_dashboard():
    rows = [
        ('JWST_0203 iterative mask', 279.903856, 0.001077, 'Pixel-noise model; S/N inflated'),
        ('JWST_0204 empirical apertures', 279.908108, 0.097345, 'Matched-aperture null; broad MC tail'),
        ('JWST_0206 paper channel/WCS', 279.902420, 0.000129, 'Centroid close; uncertainty and S/N inflated'),
        ('Paper reported', 279.901000, 0.014000, 'Published reference'),
    ]
    df = pd.DataFrame(rows, columns=['method','centroid_GHz','sigma_GHz','status'])
    csv = OUT_CSV / f'{VERSION}_CENTROID_PROGRESS.csv'
    df.to_csv(csv, index=False)
    df.to_csv(DRIVE_CSV / csv.name, index=False)

    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    y = np.arange(len(df))
    ax.errorbar(df['centroid_GHz'], y, xerr=df['sigma_GHz'], fmt='o', capsize=5, lw=1.5)
    ax.axvline(TARGET_GHZ, ls='--', lw=1.4, label='Paper centroid 279.901 GHz')
    ax.set_yticks(y, df['method'])
    ax.invert_yaxis()
    ax.set_xlabel('Observed centroid frequency (GHz)')
    ax.set_title('JADES-GS-z11-0 — extraction progress and current diagnostic status')
    ax.grid(alpha=0.22)
    ax.legend(loc='lower right')
    xmin = min(df['centroid_GHz'] - np.minimum(df['sigma_GHz'], 0.02)) - 0.003
    xmax = max(df['centroid_GHz'] + np.minimum(df['sigma_GHz'], 0.02)) + 0.003
    ax.set_xlim(xmin, xmax)
    for i, row in df.iterrows():
        ax.text(xmax - 0.0005, i, row['status'], ha='right', va='center', fontsize=9)
    fig.tight_layout()
    png = OUT_PNG / f'{VERSION}_CENTROID_PROGRESS_DASHBOARD.png'
    fig.savefig(png, dpi=190)
    plt.close(fig)
    (DRIVE_PNG / png.name).write_bytes(png.read_bytes())
    return png, csv


def collect_images(extra):
    paths = []
    for folder in (OUT_PNG, DRIVE_PNG):
        if folder.exists():
            paths.extend(folder.glob('JWST_02*.png'))
    paths.append(extra)
    unique = {}
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            unique[p.name] = p
    preferred = [
        'JWST_0207_CENTROID_PROGRESS_DASHBOARD.png',
        'JWST_0206_PAPER_EXACT_SPECTRUM.png',
        'JWST_0206_MOMENT0_MASK.png',
        'JWST_0204_EMPIRICAL_NOISE_MONTE_CARLO.png',
        'JWST_0203_PAPER_METHOD_SPECTRUM.png',
        'JWST_0203_LINE_MAP_MASK.png',
        'JWST_0201_DIAGNOSTIC_SPECTRUM.png',
    ]
    ordered = []
    for name in preferred:
        if name in unique:
            ordered.append(unique.pop(name))
    ordered.extend(sorted(unique.values(), key=lambda p: p.name, reverse=True))
    return ordered


def main():
    dashboard, csv = build_progress_dashboard()
    images = collect_images(dashboard)
    if not images:
        raise RuntimeError('No diagnostic PNG files were found.')

    options = [(p.name.replace('_', ' '), str(p)) for p in images]
    dropdown = widgets.Dropdown(
        options=options,
        value=str(images[0]),
        description='Image:',
        layout=widgets.Layout(width='96%'),
        style={'description_width': '70px'},
    )
    image_widget = widgets.Image(
        value=images[0].read_bytes(),
        format='png',
        layout=widgets.Layout(width='100%', max_width='1100px', height='auto'),
    )
    caption = widgets.HTML(value=f'<b>{images[0].name}</b>')

    def update(change):
        p = Path(change['new'])
        image_widget.value = p.read_bytes()
        caption.value = f'<b>{p.name}</b><br><span style="color:#777">{p}</span>'

    dropdown.observe(update, names='value')
    title = widgets.HTML(
        '<h3 style="margin:0 0 4px 0">JADES-GS-z11-0 diagnostic gallery</h3>'
        '<div style="margin-bottom:8px">One display window; select any saved diagnostic from the menu.</div>'
    )
    box = widgets.VBox([title, dropdown, caption, image_widget], layout=widgets.Layout(width='100%'))
    display(box)

    print(f'CODE OUTPUT: {VERSION}')
    print(f'Images available in menu: {len(images)}')
    print(f'Progress dashboard: {dashboard}')
    print(f'Progress CSV: {csv}')
    print('Current update: centroid is within 1.420 MHz of the paper value, but the noise normalization and extraction-mask independence still need correction before matching the published 4.5 sigma and ±0.014 GHz uncertainty.')
    print('Timestamp Colombia:', datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')


if __name__ == '__main__':
    main()
