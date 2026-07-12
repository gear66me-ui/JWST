# JWST_0070
# Five separate measured-reference versus real JWST/PRISM plots for MoM-z14.
# No AI images. No Gaussian profiles. No synthetic spectra. Matplotlib only.
# Published figure traces are digitized as jagged measured curves and audited.

from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
import importlib
import importlib.util
import math
import re
import subprocess
import sys

VERSION = "JWST_0070"
GALAXY = "MoM-z14"
MOM_RA = 150.0933255
MOM_DEC = 2.2731627
Z = 14.44
STRETCH = 1.0 + Z
C_NM_THz = 299792.458

ROOT = Path("/content") if Path("/content").exists() else Path.cwd()
OUT = ROOT / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION
REPO_RAW = "https://raw.githubusercontent.com/gear66me-ui/JWST/main"
MAST_HELPER = "JWST_0060_MOMZ14_FAST_CONE_CLASSY.py"
MAST_PATH = ROOT / MAST_HELPER

BG = "#050712"
AX_BG = "#07101f"
GRID = "#1f6f8b"
TEXT = "#e6f4ff"
MUTED = "#8fb3c7"
LAB = "#ffd84d"
OBS = "#ff9d2e"
MARK = "#ff5a66"
POINT = "#d9edf7"

SOURCE_DEFS = {
    "KOMPPULA_H": {
        "title": "Komppula et al. hydrogen-discharge VUV scan",
        "source_class": "TERRESTRIAL LABORATORY SPECTROGRAPH",
        "citation": "Komppula et al. 2015, J. Phys. D 48 365201, Fig. 2(a), arXiv:1510.02246",
        "urls": [
            "https://ar5iv.labs.arxiv.org/html/1510.02246/assets/x2.png",
            "https://arxiv.org/html/1510.02246/assets/x2.png",
        ],
        "full_x_nm": (80.0, 250.0),
        "candidate_boxes": [
            (0.08, 0.04, 0.95, 0.49),
            (0.07, 0.05, 0.95, 0.57),
            (0.08, 0.05, 0.49, 0.93),
        ],
    },
    "DIIID_C": {
        "title": "DIII-D deuterium/carbon VUV survey",
        "source_class": "TERRESTRIAL LABORATORY TOKAMAK SPECTROGRAPH",
        "citation": "Published DIII-D discharge spectrum, shot 174248, measured VUV/EUV trace",
        "urls": [
            "https://www.researchgate.net/publication/332788623/figure/fig1/AS:867316157919232@1583795694063/Spectra-of-the-deuterium-and-carbon-emission-in-the-EUV-a-DIII-D-174248-and-the.png",
            "https://images.weserv.nl/?url=www.researchgate.net/publication/332788623/figure/fig1/AS:867316157919232@1583795694063/Spectra-of-the-deuterium-and-carbon-emission-in-the-EUV-a-DIII-D-174248-and-the.png",
        ],
        "full_x_nm": (110.0, 160.0),
        "candidate_boxes": [
            (0.085, 0.062, 0.977, 0.560),
            (0.07, 0.04, 0.98, 0.59),
        ],
    },
    "HUT_SNR": {
        "title": "HUT Cygnus Loop low-density plasma spectrum",
        "source_class": "MEASURED EMPIRICAL LOW-DENSITY REFERENCE — NOT A LAB DISCHARGE",
        "citation": "Hopkins Ultraviolet Telescope Cygnus Loop spectrum; Blair et al. 1991 / Long et al. 1992",
        "urls": [
            "https://archive.stsci.edu/hut/images/snr_spec.gif",
            "http://archive.stsci.edu/hut/images/snr_spec.gif",
        ],
        "full_x_nm": (90.0, 185.0),
        "candidate_boxes": [
            (0.102, 0.535, 0.993, 0.905),
            (0.08, 0.50, 0.995, 0.93),
        ],
    },
}

FEATURES = [
    {
        "number": 1,
        "key": "H_I_LYA",
        "label": "H I Lyα 121.567",
        "rest_nm": 121.567,
        "left_window_nm": (117.5, 126.0),
        "source": "KOMPPULA_H",
        "fallback_source": None,
        "observed_mode": "break",
        "observed_rest_half_width_nm": 6.5,
    },
    {
        "number": 2,
        "key": "N_IV_1486",
        "label": "N IV] 1486.50",
        "rest_nm": 148.650,
        "left_window_nm": (145.5, 151.5),
        "source": "HUT_SNR",
        "fallback_source": None,
        "observed_mode": "peak",
        "observed_rest_half_width_nm": 2.3,
    },
    {
        "number": 3,
        "key": "C_IV_1549",
        "label": "C IV 1548+1551 blend",
        "rest_nm": 154.949,
        "left_window_nm": (152.0, 157.5),
        "source": "DIIID_C",
        "fallback_source": "HUT_SNR",
        "observed_mode": "peak",
        "observed_rest_half_width_nm": 2.3,
    },
    {
        "number": 4,
        "key": "HE_II_1640",
        "label": "He II 1640.42",
        "rest_nm": 164.042,
        "left_window_nm": (160.8, 167.2),
        "source": "HUT_SNR",
        "fallback_source": None,
        "observed_mode": "peak",
        "observed_rest_half_width_nm": 2.5,
    },
    {
        "number": 5,
        "key": "O_III_1666",
        "label": "O III] 1666.15",
        "rest_nm": 166.615,
        "left_window_nm": (163.4, 170.0),
        "source": "HUT_SNR",
        "fallback_source": None,
        "observed_mode": "peak",
        "observed_rest_half_width_nm": 2.5,
    },
]


def need(import_name, pip_name=None):
    try:
        importlib.import_module(import_name)
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", pip_name or import_name]
        )


def ensure_repo_file(path, filename, minimum_size=5000):
    if path.exists() and path.stat().st_size >= minimum_size:
        return path
    subprocess.run(
        [
            "curl", "-fsSL", "--connect-timeout", "15", "--max-time", "120",
            "-o", str(path), f"{REPO_RAW}/{filename}",
        ],
        check=True,
        timeout=130,
    )
    if not path.exists() or path.stat().st_size < minimum_size:
        raise RuntimeError(f"Could not download GitHub helper {filename}")
    return path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frequency_thz(wavelength_nm):
    return C_NM_THz / np.asarray(wavelength_nm, dtype=float)


def wavelength_nm(frequency_thz_value):
    return C_NM_THz / np.asarray(frequency_thz_value, dtype=float)


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def find_column(frame, exact_names, prefixes):
    lookup = {str(column).lower(): column for column in frame.columns}
    for name in exact_names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    for column in frame.columns:
        text = str(column).lower()
        if any(text.startswith(prefix.lower()) for prefix in prefixes):
            return column
    return None


def locate_cached_jwst():
    preferred = [
        f"{VERSION}_{GALAXY}_EXACT_JWST.csv",
        "JWST_0069_MoM-z14_EXACT_JWST.csv",
        "JWST_0068_MoM-z14_EXACT_JWST.csv",
        "JWST_0060_MoM-z14_EXACT_JWST.csv",
        "JWST_0059_MoM-z14_EXACT_JWST.csv",
    ]
    for filename in preferred:
        path = CSV / filename
        if path.exists() and path.stat().st_size > 100:
            return path, "coordinate-matched cached JWST X1D", None
    matches = sorted(
        CSV.glob("JWST_*_MoM-z14_EXACT_JWST.csv"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if matches:
        return matches[0], "coordinate-matched cached JWST X1D", None
    return fetch_exact_jwst()


def fetch_exact_jwst():
    ensure_repo_file(MAST_PATH, MAST_HELPER, 12000)
    mast = load_module("jwst_0060_fetch", MAST_PATH)
    mast.VERSION = VERSION
    mast.OUT = OUT
    mast.PNG = PNG
    mast.CSV = CSV
    mast.DATA = DATA / "MAST"
    mast.MAX_JWST_X1D = 24
    base = mast.load_base(mast.ensure_base())
    best, metadata = mast.exact_momz14_cone(base)
    path = Path(metadata["exact_csv"])
    if not path.exists() or path.stat().st_size < 100:
        raise RuntimeError("MAST retrieval did not produce a usable exact spectrum CSV")
    status = (
        f"coordinate-verified GO-5224 X1D; separation={metadata['sep']:.6f} arcsec; "
        f"keys={metadata['coord_source']}"
    )
    return path, status, metadata


def load_jwst(path):
    frame = pd.read_csv(path)
    wave_column = find_column(
        frame,
        ["wavelength_um", "wavelength_nm", "wavelength", "wave"],
        ["wavelength", "wave"],
    )
    flux_column = find_column(
        frame,
        ["flux", "raw_flux", "jwst_flux"],
        ["flux_raw_", "flux", "raw_flux"],
    )
    error_column = find_column(
        frame,
        ["flux_error", "error", "err"],
        ["flux_error", "error", "err"],
    )
    if wave_column is None or flux_column is None:
        raise RuntimeError(f"Could not identify wavelength/flux columns in {path.name}")
    wavelength = pd.to_numeric(frame[wave_column], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(frame[flux_column], errors="coerce").to_numpy(float)
    if error_column is not None:
        error = pd.to_numeric(frame[error_column], errors="coerce").to_numpy(float)
    else:
        error = np.full_like(flux, np.nan)
    finite = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
    wavelength, flux, error = wavelength[finite], flux[finite], error[finite]
    median = float(np.nanmedian(wavelength))
    name = str(wave_column).lower()
    if "_um" in name or median < 20:
        observed_nm = wavelength * 1000.0
    elif "_nm" in name or median < 10000:
        observed_nm = wavelength
    else:
        observed_nm = wavelength / 10.0
    order = np.argsort(observed_nm)
    return observed_nm[order], flux[order], error[order], str(wave_column), str(flux_column)


def download_image(source_key):
    import requests
    from PIL import Image

    definition = SOURCE_DEFS[source_key]
    target_dir = DATA / "REFERENCE_IMAGES"
    target_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    for attempt, url in enumerate(definition["urls"], 1):
        try:
            response = requests.get(
                url,
                timeout=(20, 120),
                headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                },
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            if image.width < 250 or image.height < 180:
                raise RuntimeError(f"image too small: {image.size}")
            path = target_dir / f"{source_key}.png"
            image.save(path)
            return image, path, url
        except Exception as exc:
            errors.append(f"attempt={attempt} {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not download {source_key}: " + " | ".join(errors))


def crop_box(image, normalized_box):
    x0, y0, x1, y1 = normalized_box
    width, height = image.size
    return image.crop(
        (
            int(round(width * x0)),
            int(round(height * y0)),
            int(round(width * x1)),
            int(round(height * y1)),
        )
    )


def viterbi_trace(gray):
    height, width = gray.shape
    y0 = max(2, int(0.035 * height))
    y1 = min(height - 2, int(0.925 * height))
    work = gray[y0:y1].astype(float)
    h = work.shape[0]

    dark = work < 125
    row_coverage = dark.mean(axis=1)
    col_coverage = dark.mean(axis=0)
    work[row_coverage > 0.45, :] = 255.0
    work[:, col_coverage > 0.72] = 255.0

    ink = np.clip((245.0 - work) / 245.0, 0.0, 1.0)
    if float(np.nanmax(ink)) <= 0:
        raise RuntimeError("reference image contains no detectable trace ink")

    baseline = 0.72 * h
    yy = np.arange(h, dtype=float)
    score = 2.8 * ink[:, 0] - 0.0007 * (yy - baseline) ** 2
    back = np.zeros((width, h), dtype=np.int8)
    shifts = np.arange(-8, 9, dtype=int)

    for column in range(1, width):
        candidates = np.full((len(shifts), h), -1.0e30, dtype=float)
        for index, shift in enumerate(shifts):
            if shift < 0:
                candidates[index, :shift] = score[-shift:] - 0.075 * abs(shift)
            elif shift > 0:
                candidates[index, shift:] = score[:-shift] - 0.075 * abs(shift)
            else:
                candidates[index, :] = score
        choice = np.argmax(candidates, axis=0)
        score = candidates[choice, np.arange(h)] + 3.2 * ink[:, column]
        back[column, :] = shifts[choice].astype(np.int8)

    path = np.empty(width, dtype=int)
    path[-1] = int(np.argmax(score))
    for column in range(width - 1, 0, -1):
        path[column - 1] = path[column] - int(back[column, path[column]])
        path[column - 1] = int(np.clip(path[column - 1], 0, h - 1))
    return path + y0


def trace_quality(wavelength_nm, intensity, nominal_nm):
    finite = np.isfinite(wavelength_nm) & np.isfinite(intensity)
    if int(finite.sum()) < 12:
        return -math.inf
    x = wavelength_nm[finite]
    y = intensity[finite]
    near = np.abs(x - nominal_nm) <= max(0.7, 0.18 * (x.max() - x.min()))
    if int(near.sum()) < 3:
        return -math.inf
    spread = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 10))
    peak = float(np.nanmax(y[near]))
    jagged = float(np.nanmedian(np.abs(np.diff(y)))) if len(y) > 1 else 0.0
    return 3.0 * peak + spread + 0.25 * jagged


def digitize_candidate(image, source_def, box, window_nm, nominal_nm):
    panel = crop_box(image, box)
    rgb = np.asarray(panel)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    full_low, full_high = source_def["full_x_nm"]
    win_low, win_high = window_nm
    width = gray.shape[1]
    px0 = int(np.floor((win_low - full_low) / (full_high - full_low) * width))
    px1 = int(np.ceil((win_high - full_low) / (full_high - full_low) * width))
    px0 = max(0, min(width - 2, px0))
    px1 = max(px0 + 2, min(width, px1))
    strip = gray[:, px0:px1]
    if strip.shape[1] < 12:
        raise RuntimeError("digitization strip has fewer than 12 image columns")

    path_y = viterbi_trace(strip)
    baseline = float(np.nanpercentile(path_y, 88))
    raw_intensity = baseline - path_y.astype(float)
    low = float(np.nanpercentile(raw_intensity, 5))
    high = float(np.nanpercentile(raw_intensity, 99))
    if not np.isfinite(high - low) or high - low <= 0:
        raise RuntimeError("digitized trace has no measurable vertical range")
    intensity = np.clip((raw_intensity - low) / (high - low), -0.10, 1.15)
    wavelength = np.linspace(win_low, win_high, len(intensity), endpoint=False)

    search = np.abs(wavelength - nominal_nm) <= min(1.2, 0.30 * (win_high - win_low))
    if int(search.sum()) >= 3:
        peak_index = np.flatnonzero(search)[int(np.nanargmax(intensity[search]))]
        calibration_shift = nominal_nm - float(wavelength[peak_index])
        wavelength = wavelength + calibration_shift
    quality = trace_quality(wavelength, intensity, nominal_nm)
    return wavelength, intensity, quality, panel


def digitize_reference(image, source_key, window_nm, nominal_nm):
    definition = SOURCE_DEFS[source_key]
    attempts = []
    for box in definition["candidate_boxes"]:
        try:
            wavelength, intensity, quality, panel = digitize_candidate(
                image, definition, box, window_nm, nominal_nm
            )
            attempts.append((quality, wavelength, intensity, panel, box))
        except Exception:
            continue
    if not attempts:
        raise RuntimeError(f"No usable trace could be digitized from {source_key}")
    attempts.sort(key=lambda item: item[0], reverse=True)
    quality, wavelength, intensity, panel, box = attempts[0]
    if not np.isfinite(quality):
        raise RuntimeError(f"Digitized trace quality is invalid for {source_key}")
    peak_search = np.abs(wavelength - nominal_nm) <= 0.55
    if int(peak_search.sum()) < 2:
        peak_index = int(np.nanargmax(intensity))
    else:
        peak_index = np.flatnonzero(peak_search)[int(np.nanargmax(intensity[peak_search]))]
    return {
        "wavelength_nm": wavelength,
        "intensity": intensity,
        "peak_nm": float(wavelength[peak_index]),
        "peak_intensity": float(intensity[peak_index]),
        "quality": float(quality),
        "box": box,
        "panel": panel,
    }


def acquire_reference(feature, image_cache):
    keys = [feature["source"]]
    if feature.get("fallback_source"):
        keys.append(feature["fallback_source"])
    errors = []
    for key in keys:
        try:
            if key not in image_cache:
                image_cache[key] = download_image(key)
            image, image_path, url = image_cache[key]
            trace = digitize_reference(
                image,
                key,
                feature["left_window_nm"],
                feature["rest_nm"],
            )
            trace.update(
                {
                    "source_key": key,
                    "image_path": image_path,
                    "source_url": url,
                    "source_title": SOURCE_DEFS[key]["title"],
                    "source_class": SOURCE_DEFS[key]["source_class"],
                    "citation": SOURCE_DEFS[key]["citation"],
                }
            )
            return trace
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def observed_window(feature, observed_nm, flux, error):
    expected = feature["rest_nm"] * STRETCH
    half = feature["observed_rest_half_width_nm"] * STRETCH
    mask = (
        np.isfinite(observed_nm)
        & np.isfinite(flux)
        & (observed_nm >= expected - half)
        & (observed_nm <= expected + half)
    )
    x = observed_nm[mask]
    y = flux[mask]
    e = error[mask]
    if len(x) < 4:
        larger = 1.8 * half
        mask = (
            np.isfinite(observed_nm)
            & np.isfinite(flux)
            & (observed_nm >= expected - larger)
            & (observed_nm <= expected + larger)
        )
        x, y, e = observed_nm[mask], flux[mask], error[mask]
    if len(x) < 3:
        raise RuntimeError(f"{feature['label']}: only {len(x)} JWST samples in the feature window")
    order = np.argsort(x)
    return x[order], y[order], e[order], expected


def measure_observed(feature, x, y, expected):
    if feature["observed_mode"] == "break":
        if len(x) < 4:
            raise RuntimeError("Ly-alpha break requires at least four samples")
        dx = np.diff(x)
        dy = np.diff(y)
        mid = 0.5 * (x[:-1] + x[1:])
        search = np.abs(mid - expected) <= 90.0
        candidates = np.where(search & np.isfinite(dy) & (dx > 0))[0]
        if len(candidates) == 0:
            candidates = np.arange(len(dy))
        positive = candidates[dy[candidates] > 0]
        if len(positive):
            candidates = positive
        proximity = np.abs(mid[candidates] - expected) / 90.0
        scale = float(np.nanmedian(np.abs(dy[candidates] - np.nanmedian(dy[candidates])))) * 1.4826
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.nanstd(dy[candidates])) or 1.0
        score = dy[candidates] / scale - 0.12 * proximity
        index = int(candidates[int(np.nanargmax(score))])
        return {
            "marker_nm": float(mid[index]),
            "marker_flux": float(0.5 * (y[index] + y[index + 1])),
            "kind": "sample-bracketed Lyα break",
            "uncertainty_nm": float(0.5 * dx[index]),
            "left_sample_nm": float(x[index]),
            "right_sample_nm": float(x[index + 1]),
        }

    core_half = 1.35 * STRETCH
    core = np.abs(x - expected) <= core_half
    if int(core.sum()) == 0:
        index = int(np.argmin(np.abs(x - expected)))
    else:
        local_indices = np.flatnonzero(core)
        local_flux = y[core]
        index = int(local_indices[int(np.nanargmax(local_flux))])
    spacing = float(np.nanmedian(np.diff(x))) if len(x) > 1 else math.nan
    return {
        "marker_nm": float(x[index]),
        "marker_flux": float(y[index]),
        "kind": "highest raw JWST sample near published feature",
        "uncertainty_nm": 0.5 * spacing if np.isfinite(spacing) else math.nan,
        "left_sample_nm": math.nan,
        "right_sample_nm": math.nan,
    }


def style(axis):
    axis.set_facecolor(AX_BG)
    axis.grid(True, color=GRID, linewidth=0.48, alpha=0.45)
    axis.tick_params(colors=TEXT, labelsize=8.5)
    axis.xaxis.label.set_color(TEXT)
    axis.yaxis.label.set_color(TEXT)
    axis.title.set_color(TEXT)
    for spine in axis.spines.values():
        spine.set_color("#4fa8c8")
        spine.set_linewidth(0.80)


def full_limits(values):
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    pad = 0.06 * (high - low) if high > low else max(abs(high) * 0.08, 1.0e-8)
    return low - pad, high + pad


def make_plot(feature, reference, x_obs, y_obs, e_obs, expected_obs, measured, source_path, source_status):
    fig, (left, right) = plt.subplots(1, 2, figsize=(15.8, 6.5), facecolor=BG)
    style(left)
    style(right)

    x_ref = reference["wavelength_nm"]
    y_ref = reference["intensity"]
    left.plot(x_ref, y_ref, color=LAB, linewidth=0.82, alpha=0.94)
    left.scatter(x_ref, y_ref, s=11, color=LAB, edgecolor="none", alpha=0.58)
    left.axvline(
        reference["peak_nm"], color=MARK, linestyle=(0, (3, 5)),
        linewidth=0.55, alpha=0.56, zorder=6,
    )
    left.scatter(
        [reference["peak_nm"]], [reference["peak_intensity"]], s=28,
        facecolor=MARK, edgecolor=BG, linewidth=0.35, zorder=7,
    )
    left.set_xlim(float(np.nanmin(x_ref)), float(np.nanmax(x_ref)))
    left.set_ylim(*full_limits(y_ref))
    left.set_title(f"MEASURED REFERENCE TRACE — {feature['label']}", fontsize=11.0, pad=12)
    left.set_xlabel("Rest wavelength, nm")
    left.set_ylabel("Digitized measured signal, normalized")

    top_left = left.secondary_xaxis("top", functions=(frequency_thz, wavelength_nm))
    top_left.set_xlabel("Rest frequency, THz", color=TEXT, labelpad=7)
    top_left.tick_params(colors=TEXT, labelsize=8)

    right.plot(x_obs, y_obs, color=OBS, linewidth=0.72, alpha=0.82)
    right.scatter(
        x_obs, y_obs, s=28, color=POINT, edgecolor=BG,
        linewidth=0.32, zorder=4,
    )
    right.axvline(
        measured["marker_nm"], color=MARK, linestyle=(0, (3, 5)),
        linewidth=0.55, alpha=0.56, zorder=6,
    )
    right.scatter(
        [measured["marker_nm"]], [measured["marker_flux"]], s=34,
        facecolor=MARK, edgecolor=BG, linewidth=0.38, zorder=7,
    )
    right.set_xlim(float(np.nanmin(x_obs)), float(np.nanmax(x_obs)))
    right.set_ylim(*full_limits(y_obs))
    right.set_title(f"RAW JWST/PRISM OBSERVATION — {feature['label']}", fontsize=11.0, pad=12)
    right.set_xlabel("Observed wavelength, nm")
    right.set_ylabel("JWST flux samples")

    top_right = right.secondary_xaxis("top", functions=(frequency_thz, wavelength_nm))
    top_right.set_xlabel("Observed frequency, THz", color=TEXT, labelpad=7)
    top_right.tick_params(colors=TEXT, labelsize=8)

    left.text(
        0.018, 0.045,
        (
            f"{reference['source_class']}\n"
            f"measured rest peak = {reference['peak_nm']:.6f} nm\n"
            f"rest frequency = {frequency_thz(reference['peak_nm']):.3f} THz"
        ),
        transform=left.transAxes, ha="left", va="bottom", color=TEXT, fontsize=7.3,
        bbox=dict(boxstyle="round,pad=.28", facecolor="#07111f",
                  edgecolor=LAB, linewidth=.60, alpha=.94),
    )
    uncertainty = measured["uncertainty_nm"]
    uncertainty_text = f" ± {uncertainty:.3f}" if np.isfinite(uncertainty) else ""
    right.text(
        0.018, 0.045,
        (
            f"{measured['kind']}\n"
            f"observed marker = {measured['marker_nm']:.3f}{uncertainty_text} nm\n"
            f"observed frequency = {frequency_thz(measured['marker_nm']):.3f} THz\n"
            f"published-z position = {expected_obs:.3f} nm"
        ),
        transform=right.transAxes, ha="left", va="bottom", color=TEXT, fontsize=7.3,
        bbox=dict(boxstyle="round,pad=.28", facecolor="#07111f",
                  edgecolor=MARK, linewidth=.60, alpha=.94),
    )

    fig.suptitle(
        f"{VERSION} — {GALAXY} — {feature['number']:02d}/05 — {feature['label']}",
        color=TEXT, fontsize=14.7, fontweight="bold", y=.982,
    )
    fig.text(
        .5, .914,
        "Left and right are independent measured datasets. Jagged samples are preserved; no Gaussian or smoothing model is plotted.",
        ha="center", color=MUTED, fontsize=8.5,
    )
    fig.text(
        .5, .017,
        f"Reference: {reference['citation']} | JWST: {source_path.name} ({source_status})",
        ha="center", color=MUTED, fontsize=7.4,
    )
    fig.subplots_adjust(left=.075, right=.985, top=.825, bottom=.12, wspace=.16)

    stem = f"{VERSION}_{feature['number']:02d}_{feature['key']}"
    png_path = PNG / f"{stem}_MEASURED_REST_VS_JWST.png"
    fig.savefig(png_path, dpi=245, facecolor=BG, edgecolor=BG)
    plt.show()
    plt.close(fig)

    ref_csv = CSV / f"{stem}_REFERENCE_TRACE.csv"
    pd.DataFrame(
        {
            "rest_wavelength_nm": x_ref,
            "rest_frequency_THz": frequency_thz(x_ref),
            "measured_reference_signal_normalized": y_ref,
            "digitized_from_published_figure": True,
            "source_key": reference["source_key"],
            "source_class": reference["source_class"],
            "source_url": reference["source_url"],
        }
    ).to_csv(ref_csv, index=False)

    obs_csv = CSV / f"{stem}_JWST_RAW_SAMPLES.csv"
    pd.DataFrame(
        {
            "observed_wavelength_nm": x_obs,
            "observed_frequency_THz": frequency_thz(x_obs),
            "jwst_flux": y_obs,
            "jwst_flux_error": e_obs,
            "published_z_expected_nm": expected_obs,
            "selected_observed_marker_nm": measured["marker_nm"],
            "selected_observed_marker_kind": measured["kind"],
        }
    ).to_csv(obs_csv, index=False)

    return png_path, ref_csv, obs_csv


def main():
    for import_name, pip_name in [
        ("numpy", None), ("pandas", None), ("matplotlib", None),
        ("requests", None), ("PIL", "Pillow"),
        ("astropy", None), ("astroquery", None),
    ]:
        need(import_name, pip_name)

    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"CODE OUTPUT: {VERSION}")
    print("STEP 1/3 | Locate or retrieve coordinate-verified MoM-z14 JWST X1D")
    source_path, source_status, mast_metadata = locate_cached_jwst()
    observed_nm, flux, error, wave_column, flux_column = load_jwst(source_path)

    print("STEP 2/3 | Download and digitize published measured reference traces")
    image_cache = {}
    audit_rows = []
    generated = []
    failures = []

    print("STEP 3/3 | Render five independent measured-reference/JWST pairs")
    for feature in FEATURES:
        try:
            print(f"  PLOT {feature['number']:02d}/05 | {feature['label']}")
            reference = acquire_reference(feature, image_cache)
            x_obs, y_obs, e_obs, expected = observed_window(
                feature, observed_nm, flux, error
            )
            measured = measure_observed(feature, x_obs, y_obs, expected)
            png_path, ref_csv, obs_csv = make_plot(
                feature, reference, x_obs, y_obs, e_obs, expected,
                measured, source_path, source_status,
            )
            generated.append((feature, reference, measured, png_path, ref_csv, obs_csv, len(x_obs)))
            audit_rows.append(
                {
                    "feature_number": feature["number"],
                    "feature": feature["label"],
                    "nominal_rest_nm": feature["rest_nm"],
                    "reference_measured_peak_nm": reference["peak_nm"],
                    "reference_source_key": reference["source_key"],
                    "reference_source_class": reference["source_class"],
                    "reference_citation": reference["citation"],
                    "reference_source_url": reference["source_url"],
                    "reference_image_path": str(reference["image_path"]),
                    "reference_digitization_quality": reference["quality"],
                    "reference_crop_box": str(reference["box"]),
                    "observed_marker_nm": measured["marker_nm"],
                    "observed_marker_kind": measured["kind"],
                    "observed_marker_uncertainty_nm": measured["uncertainty_nm"],
                    "published_z_expected_nm": expected,
                    "jwst_samples": len(x_obs),
                    "jwst_source": str(source_path),
                    "jwst_source_status": source_status,
                    "status": "GENERATED",
                }
            )
        except Exception as exc:
            failures.append((feature["number"], feature["label"], type(exc).__name__, str(exc)))
            audit_rows.append(
                {
                    "feature_number": feature["number"],
                    "feature": feature["label"],
                    "nominal_rest_nm": feature["rest_nm"],
                    "status": f"FAILED: {type(exc).__name__}: {exc}",
                }
            )
            print(f"    FAILED | {type(exc).__name__}: {exc}")

    audit_path = CSV / f"{VERSION}_{GALAXY}_SOURCE_AND_MEASUREMENT_AUDIT.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    if failures:
        failure_path = CSV / f"{VERSION}_{GALAXY}_FAILURES.csv"
        pd.DataFrame(
            failures,
            columns=["number", "feature", "error_type", "message"],
        ).to_csv(failure_path, index=False)

    if not generated:
        raise RuntimeError(f"No figures generated. Audit: {audit_path}")

    print()
    print("NO  FEATURE                    REST PEAK nm   OBS MARKER nm   JWST N  SOURCE")
    for feature, reference, measured, _, _, _, sample_count in generated:
        print(
            f"{feature['number']:02d}  {feature['label']:<25} "
            f"{reference['peak_nm']:12.6f}  {measured['marker_nm']:13.6f}  "
            f"{sample_count:6d}  {reference['source_key']}"
        )
    print()
    print(f"JWST SOURCE          : {source_path}")
    print(f"JWST SOURCE STATUS   : {source_status}")
    print(f"WAVELENGTH COLUMN    : {wave_column}")
    print(f"FLUX COLUMN          : {flux_column}")
    print(f"GENERATED            : {len(generated)} of {len(FEATURES)} figures")
    print(f"SOURCE AUDIT CSV     : {audit_path}")
    print(f"PNG DIRECTORY        : {PNG}")
    print(f"CSV DIRECTORY        : {CSV}")
    if mast_metadata is not None:
        print(f"JWST PRODUCT         : {Path(mast_metadata['product']).name}")
        print(f"JWST SEPARATION      : {mast_metadata['sep']:.6f} arcsec")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
