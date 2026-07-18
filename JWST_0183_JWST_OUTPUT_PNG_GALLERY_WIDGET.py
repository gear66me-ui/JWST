# JWST_0183
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import ipywidgets as widgets
from IPython.display import display, clear_output, Image, HTML

VERSION = 'JWST_0183'
PNG_DIR = Path('/content/JWST_OUTPUT/PNG')
SUPPORTED = {'.png', '.jpg', '.jpeg', '.webp'}


def image_files():
    if not PNG_DIR.exists():
        return []
    return sorted(
        [p for p in PNG_DIR.rglob('*') if p.is_file() and p.suffix.lower() in SUPPORTED],
        key=lambda p: (p.stat().st_mtime, p.name),
        reverse=True,
    )

files = image_files()
index = {'value': 0}

selector = widgets.Dropdown(
    options=[(f'{i+1:02d} — {p.name}', i) for i, p in enumerate(files)],
    description='Image:',
    layout=widgets.Layout(width='780px'),
    style={'description_width': '60px'},
)
prev_btn = widgets.Button(description='◀ Previous', layout=widgets.Layout(width='130px'))
next_btn = widgets.Button(description='Next ▶', layout=widgets.Layout(width='130px'))
refresh_btn = widgets.Button(description='Refresh folder', button_style='info', layout=widgets.Layout(width='145px'))
all_btn = widgets.Button(description='Show all', button_style='success', layout=widgets.Layout(width='120px'))
out = widgets.Output()


def show_one(i):
    global files
    with out:
        clear_output(wait=True)
        if not files:
            print(f'No image files found in {PNG_DIR}')
            print('Run the earlier JWST widgets first, then press Refresh folder.')
            return
        i = max(0, min(int(i), len(files)-1))
        index['value'] = i
        p = files[i]
        size_mb = p.stat().st_size / 1_048_576
        stamp = datetime.fromtimestamp(p.stat().st_mtime, ZoneInfo('America/Bogota')).strftime('%Y-%m-%d %H:%M:%S %Z')
        display(HTML(
            f"<div style='background:#0d1117;color:#f0f6fc;padding:12px 16px;border:1px solid #30363d;border-radius:8px;margin-bottom:10px'>"
            f"<b>{i+1} of {len(files)}</b><br><span style='font-size:16px'>{p.name}</span><br>"
            f"<span style='color:#9da7b3'>{size_mb:.2f} MB · modified {stamp}</span></div>"
        ))
        display(Image(filename=str(p)))
        print(f'FILE: {p}')


def on_select(change):
    if change.get('name') == 'value' and change.get('new') is not None:
        show_one(change['new'])


def on_prev(_):
    if files:
        selector.value = (index['value'] - 1) % len(files)


def on_next(_):
    if files:
        selector.value = (index['value'] + 1) % len(files)


def on_refresh(_):
    global files
    files = image_files()
    selector.options = [(f'{i+1:02d} — {p.name}', i) for i, p in enumerate(files)]
    selector.value = 0 if files else None
    show_one(0)


def on_all(_):
    with out:
        clear_output(wait=True)
        if not files:
            print(f'No image files found in {PNG_DIR}')
            return
        display(HTML(f"<h3 style='color:#f0f6fc;background:#0d1117;padding:10px'>JWST PNG gallery — {len(files)} files</h3>"))
        for i, p in enumerate(files, 1):
            display(HTML(f"<hr><h4>{i:02d} — {p.name}</h4>"))
            display(Image(filename=str(p), width=1100))

selector.observe(on_select, names='value')
prev_btn.on_click(on_prev)
next_btn.on_click(on_next)
refresh_btn.on_click(on_refresh)
all_btn.on_click(on_all)

print(f'CODE OUTPUT: {VERSION}')
print(f'Image folder: {PNG_DIR}')
print(f'Images found: {len(files)}')
print(f'Timestamp Colombia: {datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")}')
print(f'# {VERSION}')
display(widgets.VBox([selector, widgets.HBox([prev_btn, next_btn, refresh_btn, all_btn]), out]))
show_one(0)
