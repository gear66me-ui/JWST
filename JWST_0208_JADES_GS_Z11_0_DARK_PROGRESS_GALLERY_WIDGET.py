from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

VERSION = 'JWST_0208'
TARGET_GHZ = 279.901
ROOT = Path('/content/drive/MyDrive/JWST/ALMA/JADES_GS_Z11_0_2023.1.00336.S/X3667_SCIENCE')
OUT_PNG = Path('/content/JWST_OUTPUT/PNG')
OUT_CSV = Path('/content/JWST_OUTPUT/CSV')
DRIVE_PNG = ROOT / 'PNG'
DRIVE_CSV = ROOT / 'CSV'
for p in (OUT_PNG, OUT_CSV, DRIVE_PNG, DRIVE_CSV):
    p.mkdir(parents=True, exist_ok=True)

BG = '#05070b'
PANEL = '#0b111a'
TEXT = '#f2f6fb'
MUTED = '#9fb0c3'
GRID = '#263445'
CYAN = '#4dd9ff'
BLUE = '#4c8dff'
ORANGE = '#ff9f43'
RED = '#ff5d5d'
MAGENTA = '#d77cff'


def dark_axes(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(color=GRID, alpha=0.55, linewidth=0.7)


def save_dark_progress():
    rows = [
        ('JWST_0203 iterative mask', 279.903856, 0.001077, 'Centroid close; pixel-noise S/N inflated'),
        ('JWST_0204 empirical apertures', 279.908108, 0.097345, 'Empirical null; broad Monte Carlo tail'),
        ('JWST_0206 paper WCS/channels', 279.902420, 0.000129, '1.420 MHz offset; uncertainty inflated'),
        ('Paper reported', 279.901000, 0.014000, 'Published 279.901 ± 0.014 GHz'),
    ]
    df = pd.DataFrame(rows, columns=['method','centroid_GHz','sigma_GHz','status'])
    csv = OUT_CSV / f'{VERSION}_CENTROID_PROGRESS.csv'
    df.to_csv(csv, index=False)
    df.to_csv(DRIVE_CSV / csv.name, index=False)

    fig, ax = plt.subplots(figsize=(14, 7.8), facecolor=BG)
    dark_axes(ax)
    y = np.arange(len(df))
    colors = [CYAN, ORANGE, MAGENTA, RED]
    for i, row in df.iterrows():
        ax.errorbar(row.centroid_GHz, i, xerr=row.sigma_GHz, fmt='o', ms=8,
                    capsize=5, lw=1.7, color=colors[i], ecolor=colors[i])
    ax.axvline(TARGET_GHZ, color=ORANGE, ls='--', lw=1.5, label='Paper centroid 279.901 GHz')
    ax.set_yticks(y, df['method'], color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel('Observed centroid frequency (GHz)')
    ax.set_title('JADES-GS-z11-0 — extraction progress')
    xmin = min(df.centroid_GHz - np.minimum(df.sigma_GHz, 0.02)) - 0.003
    xmax = max(df.centroid_GHz + np.minimum(df.sigma_GHz, 0.02)) + 0.003
    ax.set_xlim(xmin, xmax)
    for i, row in df.iterrows():
        ax.text(xmax - 0.0005, i, row.status, ha='right', va='center', color=MUTED, fontsize=9)
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    png = OUT_PNG / f'{VERSION}_DARK_CENTROID_PROGRESS.png'
    fig.savefig(png, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    (DRIVE_PNG / png.name).write_bytes(png.read_bytes())
    return png, csv


def find_csv(name):
    for p in (OUT_CSV / name, DRIVE_CSV / name):
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def save_dark_spectrum():
    src = find_csv('JWST_0206_PAPER_EXACT_SPECTRUM.csv')
    if src is None:
        return None
    df = pd.read_csv(src)
    f = df['frequency_GHz'].to_numpy(float)
    s = df['flux_sum_Jy_beam'].to_numpy(float) * 1000.0
    e = df['sigma_Jy_beam'].to_numpy(float) * 1000.0
    line = df['line_window'].astype(bool).to_numpy()
    centroid = np.nansum(f[line] * s[line]) / np.nansum(s[line])

    fig, ax = plt.subplots(figsize=(14, 7.8), facecolor=BG)
    dark_axes(ax)
    ax.plot(f, s, marker='o', ms=4, lw=1.35, color=CYAN, label='Paper-style extracted spectrum')
    ax.fill_between(f, s-e, s+e, color=BLUE, alpha=0.20, label='±1σ model')
    ax.axvspan(f[line].min(), f[line].max(), color=MAGENTA, alpha=0.10, label='Paper velocity window')
    ax.axvline(TARGET_GHZ, color=ORANGE, ls='--', lw=1.5, label='Paper 279.901 GHz')
    ax.axvline(centroid, color=RED, ls=':', lw=1.6, label=f'Observed centroid {centroid:.6f} GHz')
    ax.axhline(0, color=MUTED, lw=0.8, alpha=0.7)
    ax.set_xlabel('Observed frequency (GHz)')
    ax.set_ylabel('Integrated flux (mJy beam⁻¹)')
    ax.set_title('JADES-GS-z11-0 — 5 MHz paper-channel extraction')
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    png = OUT_PNG / f'{VERSION}_DARK_PAPER_EXACT_SPECTRUM.png'
    fig.savefig(png, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    (DRIVE_PNG / png.name).write_bytes(png.read_bytes())
    return png


def save_dark_mc():
    src = find_csv('JWST_0204_MONTE_CARLO_CENTROIDS.csv')
    if src is None:
        return None
    df = pd.read_csv(src)
    col = next((c for c in df.columns if 'centroid' in c.lower()), None)
    if col is None:
        return None
    x = df[col].to_numpy(float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    fig, ax = plt.subplots(figsize=(14, 7.8), facecolor=BG)
    dark_axes(ax)
    ax.hist(x, bins=55, color=BLUE, alpha=0.78, edgecolor=CYAN, linewidth=0.5)
    ax.axvline(TARGET_GHZ, color=ORANGE, ls='--', lw=1.6, label='Paper 279.901 GHz')
    ax.axvline(np.nanmedian(x), color=RED, ls=':', lw=1.6, label=f'MC median {np.nanmedian(x):.6f} GHz')
    ax.set_xlabel('Monte Carlo centroid (GHz)')
    ax.set_ylabel('Realizations')
    ax.set_title('JADES-GS-z11-0 — empirical-noise Monte Carlo centroid distribution')
    leg = ax.legend(facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(TEXT)
    fig.tight_layout()
    png = OUT_PNG / f'{VERSION}_DARK_MONTE_CARLO_CENTROIDS.png'
    fig.savefig(png, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    (DRIVE_PNG / png.name).write_bytes(png.read_bytes())
    return png


def collect_images(generated):
    preferred = [p for p in generated if p is not None and p.exists()]
    names = {
        'JWST_0206_MOMENT0_MASK.png',
        'JWST_0204_EMPIRICAL_NOISE_MONTE_CARLO.png',
        'JWST_0203_LINE_MAP_MASK.png',
        'JWST_0201_DIAGNOSTIC_SPECTRUM.png',
    }
    extra = []
    for folder in (OUT_PNG, DRIVE_PNG):
        for name in names:
            p = folder / name
            if p.exists() and p.stat().st_size > 0:
                extra.append(p)
    unique = {}
    for p in preferred + extra:
        unique[p.name] = p
    return list(unique.values())


def main():
    progress, csv = save_dark_progress()
    spectrum = save_dark_spectrum()
    mc = save_dark_mc()
    images = collect_images([progress, spectrum, mc])
    if not images:
        raise RuntimeError('No diagnostic images found.')

    labels = [p.name.replace('_', ' ') for p in images]
    state = {'index': 0}
    dropdown = widgets.Dropdown(options=list(zip(labels, range(len(images)))), value=0,
                                description='Image:', layout=widgets.Layout(width='78%'),
                                style={'description_width': '60px'})
    previous = widgets.Button(description='◀ Previous', button_style='', layout=widgets.Layout(width='110px'))
    next_button = widgets.Button(description='Next ▶', button_style='', layout=widgets.Layout(width='110px'))
    counter = widgets.HTML()
    caption = widgets.HTML()
    image = widgets.Image(format='png', layout=widgets.Layout(width='100%', max_width='1150px', height='auto'))

    def show(index):
        index %= len(images)
        state['index'] = index
        dropdown.value = index
        p = images[index]
        image.value = p.read_bytes()
        counter.value = f'<span style="color:{MUTED}"><b>{index+1} / {len(images)}</b></span>'
        caption.value = f'<b style="color:{TEXT}">{p.name}</b><br><span style="color:{MUTED}">{p}</span>'

    def on_dropdown(change):
        if change['name'] == 'value' and change['new'] is not None:
            show(int(change['new']))

    def on_previous(_):
        show(state['index'] - 1)

    def on_next(_):
        show(state['index'] + 1)

    dropdown.observe(on_dropdown, names='value')
    previous.on_click(on_previous)
    next_button.on_click(on_next)

    header = widgets.HTML(
        f'<div style="background:{BG};color:{TEXT};padding:12px 14px;border:1px solid {GRID};border-radius:8px">'
        '<h3 style="margin:0 0 4px 0">JADES-GS-z11-0 diagnostic gallery</h3>'
        f'<div style="color:{MUTED}">Black-background scientific plots · one image window · dropdown plus Previous/Next controls</div></div>'
    )
    controls = widgets.HBox([previous, next_button, dropdown, counter],
                            layout=widgets.Layout(width='100%', align_items='center'))
    panel = widgets.VBox([header, controls, caption, image],
                         layout=widgets.Layout(width='100%', padding='8px', background_color=BG))
    show(0)
    display(panel)

    print(f'CODE OUTPUT: {VERSION}')
    print(f'Images available: {len(images)}')
    print(f'Dark progress PNG: {progress}')
    if spectrum: print(f'Dark spectrum PNG: {spectrum}')
    if mc: print(f'Dark Monte Carlo PNG: {mc}')
    print(f'Progress CSV: {csv}')
    print('Timestamp Colombia:', datetime.now(timezone(timedelta(hours=-5))).strftime('%Y-%m-%d %H:%M:%S %z'))
    print(f'# {VERSION}')


if __name__ == '__main__':
    main()
