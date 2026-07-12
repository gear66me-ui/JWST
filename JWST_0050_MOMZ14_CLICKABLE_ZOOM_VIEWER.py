#!/usr/bin/env python3
"""
JWST_0050_MOMZ14_CLICKABLE_ZOOM_VIEWER.py

Download genuine calibrated JWST/NIRCam FITS cutouts of MoM-z14, create
large lossless PNG views, start a local Colab image server, and display
clickable hyperlinks that open each image in a new browser tab.

No AI-generated imagery is used. Enlarged PNGs improve viewing only; they do
not create additional scientific resolution beyond the FITS data.
"""

from __future__ import annotations

import html
import importlib.util
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "JWST_0048_MOMZ14_4FILTER_FITS.py"
)
BASE_PATH = Path("/content/JWST_0048_MOMZ14_4FILTER_FITS.py")
VERSION = "JWST_0050"
PORT = 8877
ROOT = Path("/content/JWST_OUTPUT")
PNG_DIR = ROOT / "PNG"
FITS_DIR = ROOT / "FITS"
CSV_DIR = ROOT / "CSV"
for directory in (PNG_DIR, FITS_DIR, CSV_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def load_base_module():
    urllib.request.urlretrieve(BASE_URL, BASE_PATH)
    spec = importlib.util.spec_from_file_location("jwst0048", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load JWST_0048 base module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    return module


def add_marker(ax, data, pixel_scale: float, wide: bool) -> None:
    from matplotlib.patches import Circle

    ny, nx = data.shape
    cx, cy = nx / 2.0, ny / 2.0
    radius_arcsec = 0.55 if wide else 0.28
    radius_pix = radius_arcsec / pixel_scale

    ax.add_patch(Circle((cx, cy), radius_pix, fill=False,
                        edgecolor="black", linewidth=7.0, zorder=8))
    ax.add_patch(Circle((cx, cy), radius_pix, fill=False,
                        edgecolor="red", linewidth=3.2, zorder=9))

    gap = radius_pix * 1.25
    arm = radius_pix * 0.90
    segments = [
        ([cx - gap - arm, cx - gap], [cy, cy]),
        ([cx + gap, cx + gap + arm], [cy, cy]),
        ([cx, cx], [cy - gap - arm, cy - gap]),
        ([cx, cx], [cy + gap, cy + gap + arm]),
    ]
    for xs, ys in segments:
        ax.plot(xs, ys, color="black", linewidth=7.0, zorder=8)
        ax.plot(xs, ys, color="red", linewidth=3.2, zorder=9)

    ax.annotate(
        "MoM-z14",
        xy=(cx + radius_pix * 0.72, cy + radius_pix * 0.72),
        xytext=(20, 20),
        textcoords="offset points",
        color="red",
        fontsize=15,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 2.4},
        zorder=10,
    )


def save_single(module, record: dict, zoom: bool) -> Path:
    import matplotlib.pyplot as plt

    entry = record["entry"]
    data = record["data"]
    pixel_scale = record["pixel_scale_arcsec"]
    if zoom:
        shown = module.central_crop(data, module.ZOOM_ARCSEC, pixel_scale)
        field_arcsec = module.ZOOM_ARCSEC
        label = "ZOOM"
        wide = False
    else:
        shown = data
        field_arcsec = module.CUTOUT_ARCSEC
        label = "FULL"
        wide = True

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(12, 12), constrained_layout=True)
    ax.imshow(
        shown,
        origin="lower",
        cmap="gray",
        norm=module.image_norm(shown),
        interpolation="nearest",
    )
    add_marker(ax, shown, pixel_scale, wide)
    ax.set_title(
        f"MoM-z14 | {entry['name']} | {entry['lambda_um']:.2f} µm | {field_arcsec:.1f} arcsec field",
        fontsize=20,
        pad=18,
    )
    ax.set_xlabel(
        f"Pixel scale: {pixel_scale:.5f} arcsec/pixel | native FITS data, nearest-neighbor display",
        fontsize=12,
        labelpad=12,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("0.55")

    path = PNG_DIR / f"{VERSION}_MOMZ14_{entry['name']}_{label}_ZOOMABLE.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def save_dashboard(module, records: list[dict]) -> Path:
    import matplotlib.pyplot as plt

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), constrained_layout=True)

    for col, record in enumerate(records):
        entry = record["entry"]
        data = record["data"]
        pixel_scale = record["pixel_scale_arcsec"]
        zoom = module.central_crop(data, module.ZOOM_ARCSEC, pixel_scale)

        for row, shown, field_arcsec, wide in [
            (0, data, module.CUTOUT_ARCSEC, True),
            (1, zoom, module.ZOOM_ARCSEC, False),
        ]:
            ax = axes[row, col]
            ax.imshow(shown, origin="lower", cmap="gray",
                      norm=module.image_norm(shown), interpolation="nearest")
            add_marker(ax, shown, pixel_scale, wide)
            if row == 0:
                ax.set_title(f"{entry['name']} | {entry['lambda_um']:.2f} µm",
                             fontsize=15, pad=10)
            else:
                ax.set_title(f"center zoom | {field_arcsec:.1f} arcsec",
                             fontsize=13, pad=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "MoM-z14 — clickable high-resolution JWST/NIRCam channel dashboard",
        fontsize=24,
    )
    path = PNG_DIR / f"{VERSION}_MOMZ14_4FILTER_ZOOMABLE_DASHBOARD.png"
    fig.savefig(path, dpi=260, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server() -> None:
    if port_is_open(PORT):
        return
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(PORT),
         "--directory", str(ROOT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    for _ in range(30):
        if port_is_open(PORT):
            return
        time.sleep(0.1)
    raise RuntimeError(f"Local image server did not start on port {PORT}")


def get_proxy_url() -> str | None:
    try:
        from google.colab import output
        value = output.eval_js(f"google.colab.kernel.proxyPort({PORT})")
        return str(value).rstrip("/")
    except Exception:
        return None


def display_link_panel(proxy_url: str | None, files: list[Path]) -> None:
    from IPython.display import FileLink, HTML, display

    if not proxy_url:
        print("Clickable Colab proxy unavailable; using notebook file links instead.")
        for path in files:
            display(FileLink(str(path)))
        return

    buttons = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        url = f"{proxy_url}/{relative}"
        title = path.stem.replace("_", " ")
        buttons.append(
            f'<a href="{html.escape(url)}" target="_blank" rel="noopener" '
            'style="display:inline-block;margin:6px;padding:11px 15px;'
            'background:#18344f;color:white;text-decoration:none;border-radius:7px;'
            'font-family:Arial,sans-serif;font-weight:600">'
            f'Open {html.escape(title)}</a>'
        )

    panel = (
        '<div style="padding:14px;border:1px solid #536878;border-radius:10px;'
        'background:#0e1621;color:#eef5ff">'
        '<div style="font-size:18px;font-weight:700;margin-bottom:7px">'
        'MoM-z14 full-resolution viewer</div>'
        '<div style="margin-bottom:8px">Click any button. It opens the PNG in a new tab, '
        'where browser zoom can be pushed as far as desired.</div>'
        + "".join(buttons)
        + '<div style="font-size:12px;margin-top:10px;color:#b9c7d5">'
        'The PNGs are enlarged with nearest-neighbor display. No pixels are invented; '
        'the underlying scientific resolution remains that of the FITS mosaic.</div></div>'
    )
    display(HTML(panel))


def main() -> None:
    module = load_base_module()
    session = module.build_session()
    records = []

    for entry in module.FILTERS:
        path, source_url, byte_count = module.download_channel(session, entry)
        data, header = module.load_image(path)
        pixel_scale = module.estimate_pixel_scale_arcsec(header)
        records.append({
            "entry": entry,
            "path": path,
            "data": data,
            "pixel_scale_arcsec": pixel_scale,
            "source_url": source_url,
            "download_bytes": byte_count,
        })

    dashboard = save_dashboard(module, records)
    outputs = [dashboard]
    for record in records:
        outputs.append(save_single(module, record, zoom=False))
        outputs.append(save_single(module, record, zoom=True))

    start_server()
    proxy_url = get_proxy_url()
    display_link_panel(proxy_url, outputs)

    print(f"CODE OUTPUT: {VERSION}")
    print("Target       MoM-z14")
    print("Filters      F115W  F150W  F277W  F444W")
    print(f"Dashboard    {dashboard}")
    print(f"PNG count    {len(outputs)}")
    print(f"Viewer URL   {proxy_url or 'FileLink fallback'}")
    print(f"Timestamp    {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
