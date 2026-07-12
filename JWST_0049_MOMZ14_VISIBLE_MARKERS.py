#!/usr/bin/env python3
"""
JWST_0049_MOMZ14_VISIBLE_MARKERS.py

Re-run the four genuine JWST/NIRCam MoM-z14 channel cutouts with large,
high-contrast catalog markers that remain visible in a 2x4 Colab dashboard.
No AI-generated imagery is used.
"""

from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path

BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    "JWST_0048_MOMZ14_4FILTER_FITS.py"
)
BASE_PATH = Path("/content/JWST_0048_MOMZ14_4FILTER_FITS.py")
VERSION = "JWST_0049"


def load_base_module():
    urllib.request.urlretrieve(BASE_URL, BASE_PATH)
    spec = importlib.util.spec_from_file_location("jwst0048", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load JWST_0048 base module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_visible_marker(module) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    def decorate_axis(ax: plt.Axes, title: str, width_arcsec: float, data,
                      pixel_scale: float, circle_arcsec: float = 0.30) -> None:
        ax.imshow(
            data,
            origin="lower",
            cmap="gray",
            norm=module.image_norm(data),
            interpolation="nearest",
        )

        ny, nx = data.shape
        cx, cy = nx / 2.0, ny / 2.0

        marker_radius_arcsec = 0.55 if width_arcsec >= 4.0 else 0.30
        radius_pix = marker_radius_arcsec / pixel_scale

        # Thick dark underlay plus bright red overlay keeps the circle visible
        # against both bright objects and the dark sky background.
        ax.add_patch(Circle(
            (cx, cy), radius_pix, fill=False,
            edgecolor="black", linewidth=5.0, zorder=8,
        ))
        ax.add_patch(Circle(
            (cx, cy), radius_pix, fill=False,
            edgecolor="red", linewidth=2.8, zorder=9,
        ))

        gap = radius_pix * 1.25
        arm = radius_pix * 0.85
        segments = [
            ([cx - gap - arm, cx - gap], [cy, cy]),
            ([cx + gap, cx + gap + arm], [cy, cy]),
            ([cx, cx], [cy - gap - arm, cy - gap]),
            ([cx, cx], [cy + gap, cy + gap + arm]),
        ]
        for xs, ys in segments:
            ax.plot(xs, ys, color="black", linewidth=5.0, zorder=8)
            ax.plot(xs, ys, color="red", linewidth=2.8, zorder=9)

        ax.annotate(
            "MoM-z14",
            xy=(cx + radius_pix * 0.72, cy + radius_pix * 0.72),
            xytext=(12, 12),
            textcoords="offset points",
            color="red",
            fontsize=9,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "red", "lw": 1.8},
            zorder=10,
        )

        ax.set_title(title, fontsize=11, pad=7)
        ax.text(
            0.03, 0.04, f'{width_arcsec:.1f}" field',
            transform=ax.transAxes, fontsize=8, ha="left", va="bottom",
            bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_edgecolor("0.55")

    module.decorate_axis = decorate_axis


def main() -> None:
    module = load_base_module()
    module.VERSION = VERSION
    install_visible_marker(module)
    module.main()


if __name__ == "__main__":
    main()
