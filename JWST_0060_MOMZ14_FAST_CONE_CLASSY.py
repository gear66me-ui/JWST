# JWST_0060
# Fast, verified replacement for JWST_0059.
# - Cone-searches MAST around the published MoM-z14 coordinates first.
# - Inspects only X1D products from the matching NIRSpec pointing.
# - Uses a compact CLASSY HLSP/HST candidate set instead of 30 SIMBAD calls.
# - Reuses the real-data plotting and preflight logic from JWST_0059.
# No AI images. Matplotlib only.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import math
import subprocess
import sys

VERSION = "JWST_0060"
BASE_FILENAME = "JWST_0059_MOMZ14_REAL_CLASSY_VS_EXACT_JWST.py"
BASE_URL = "https://raw.githubusercontent.com/gear66me-ui/JWST/main/" + BASE_FILENAME
BASE_PATH = Path("/content") / BASE_FILENAME if Path("/content").exists() else Path.cwd() / BASE_FILENAME
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
PNG = OUT / "PNG"
CSV = OUT / "CSV"
DATA = OUT / "DATA" / VERSION
MAX_JWST_X1D = 18
MAX_CLASSY_PRODUCTS = 16


def ensure_base():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 12000:
        return BASE_PATH
    subprocess.run(
        [
            "curl",
            "-fsSL",
            "--connect-timeout",
            "15",
            "--max-time",
            "90",
            "-o",
            str(BASE_PATH),
            BASE_URL,
        ],
        check=True,
        timeout=100,
    )
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size < 12000:
        raise RuntimeError("Could not download JWST_0059 helper script.")
    return BASE_PATH


def load_base(path):
    spec = importlib.util.spec_from_file_location("jwst_0059_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load JWST_0059 helper module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.VERSION = VERSION
    module.DATA = DATA
    module.PNG = PNG
    module.CSV = CSV
    return module


def coords_fast(headers):
    pairs = [
        ("SRCRA", "SRCDEC"),
        ("RA_OBJ", "DEC_OBJ"),
        ("OBJ_RA", "OBJ_DEC"),
        ("SLIT_RA", "SLIT_DEC"),
        ("MSA_RA", "MSA_DEC"),
        ("SHUT_RA", "SHUT_DEC"),
        ("TARG_RA", "TARG_DEC"),
        ("RA_TARG", "DEC_TARG"),
        ("RA_REF", "DEC_REF"),
    ]
    for header in headers:
        for ra_key, dec_key in pairs:
            ra = fnum(header.get(ra_key))
            dec = fnum(header.get(dec_key))
            if ra is not None and dec is not None:
                return ra, dec, f"{ra_key}/{dec_key}"
    return None, None, "NONE"


def fnum(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except Exception:
        return None


def exact_momz14_cone(base):
    import astropy.units as u
    import pandas as pd
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Observations

    print("STEP 1/4 | MAST cone search centered on published MoM-z14 coordinates")
    target = SkyCoord(base.MOM_RA * u.deg, base.MOM_DEC * u.deg)
    selected = None

    for radius_arcsec in [3.0, 10.0, 30.0]:
        table = Observations.query_region(target, radius=radius_arcsec * u.arcsec)
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
        if frame.empty:
            print(f"  cone radius {radius_arcsec:.0f} arcsec | observations=0")
            continue

        proposal_col = base.col(frame, ["proposal_id", "proposalid"])
        collection_col = base.col(frame, ["obs_collection"])
        instrument_col = base.col(frame, ["instrument_name", "instrument"])
        data_col = base.col(frame, ["dataproduct_type"])

        mask = pd.Series(True, index=frame.index)
        if proposal_col is not None:
            mask &= frame[proposal_col].astype(str).str.replace(".0", "", regex=False).eq(base.JWST_PID)
        if collection_col is not None:
            mask &= frame[collection_col].astype(str).str.upper().eq("JWST")
        if instrument_col is not None:
            mask &= frame[instrument_col].astype(str).str.contains("NIRSPEC", case=False, na=False)
        if data_col is not None:
            spectrum = frame[data_col].astype(str).str.contains("spectrum", case=False, na=False)
            if spectrum.any():
                mask &= spectrum

        candidate = frame[mask].copy()
        print(
            f"  cone radius {radius_arcsec:.0f} arcsec | all={len(frame)} | "
            f"GO-5224 NIRSpec={len(candidate)}"
        )
        if not candidate.empty:
            selected = candidate
            break

    if selected is None or selected.empty:
        raise RuntimeError("No GO-5224 NIRSpec observation footprint intersects MoM-z14 coordinates.")

    obsid_col = base.col(selected, ["obsid", "obs_id"])
    if obsid_col is None:
        raise RuntimeError("Cone-search results do not contain observation IDs.")
    obsids = selected[obsid_col].dropna().astype(str).drop_duplicates().tolist()

    print(f"STEP 2/4 | Fetching products for {len(obsids)} coordinate-matched observation(s)")
    products = base.query_products(obsids)
    x1d, filename_col = base.x1d_products(products)
    if x1d.empty:
        raise RuntimeError("No X1D products found for the coordinate-matched GO-5224 observation.")

    size_col = base.col(x1d, ["size"])
    if size_col is not None:
        x1d = x1d.sort_values(size_col, ascending=False)
    x1d = x1d.drop_duplicates(subset=[filename_col]).head(MAX_JWST_X1D)
    print(f"  X1D candidates to inspect={len(x1d)}")

    base.coords = coords_fast
    all_extensions = []
    audit_rows = []

    print("STEP 3/4 | Downloading/reading only the matched-pointing X1D products")
    for index, record in enumerate(x1d.to_dict("records"), 1):
        product_name = str(record.get(filename_col, "UNKNOWN"))
        try:
            path = base.download(record, "JWST_GO5224_CONE")
            extensions = base.read_jwst(path)
            all_extensions.extend(extensions)
            nearest = min((entry["sep"] for entry in extensions), default=math.inf)
            print(
                f"  {index:02d}/{len(x1d):02d} | {path.name} | "
                f"extensions={len(extensions)} | nearest={nearest:.3f} arcsec"
            )
            audit_rows.append((path.name, "OK", len(extensions), nearest))
        except Exception as exc:
            print(f"  {index:02d}/{len(x1d):02d} | {product_name} | {type(exc).__name__}")
            audit_rows.append((product_name, type(exc).__name__, 0, math.nan))

    audit_path = CSV / f"{VERSION}_JWST_CONE_X1D_AUDIT.csv"
    pd.DataFrame(
        audit_rows,
        columns=["product", "status", "extensions", "nearest_arcsec"],
    ).to_csv(audit_path, index=False)

    valid = [
        entry
        for entry in all_extensions
        if math.isfinite(entry["sep"])
        and entry["sep"] <= 1.5
        and len(entry["wave"]) >= 5
    ]
    if not valid:
        nearest = min((entry["sep"] for entry in all_extensions), default=math.inf)
        raise RuntimeError(
            f"No source extraction within 1.5 arcsec after cone filtering. "
            f"Nearest={nearest:.3f} arcsec. Audit: {audit_path}"
        )

    required_low = min(item[3][0] for item in base.COMPLEXES) * (1.0 + base.MOM_Z) * 1.0e-4
    required_high = max(item[3][1] for item in base.COMPLEXES) * (1.0 + base.MOM_Z) * 1.0e-4
    covering = [
        entry
        for entry in valid
        if float(entry["wave"].min()) <= required_low
        and float(entry["wave"].max()) >= required_high
    ]
    if not covering:
        raise RuntimeError(
            "Coordinate-matched X1D extension was found, but it does not cover all five UV complexes."
        )

    covering.sort(key=lambda entry: (entry["sep"], -len(entry["wave"])))
    best = covering[0]
    exact_csv = CSV / f"{VERSION}_{base.GALAXY}_EXACT_JWST.csv"
    pd.DataFrame(
        {
            "wavelength_um": best["wave"],
            "flux": best["flux"],
            "flux_error": best["err"],
        }
    ).to_csv(exact_csv, index=False)

    metadata = {
        "exact_csv": str(exact_csv),
        "product": str(best["path"]),
        "hdu": best["hdu"],
        "sep": best["sep"],
        "source_id": best["source_id"],
        "coord_source": best["coord_source"],
        "audit": str(audit_path),
    }
    print(
        f"  ACCEPTED | separation={best['sep']:.6f} arcsec | "
        f"coordinate keys={best['coord_source']} | HDU={best['hdu']}"
    )
    return best, metadata


def query_classy_observations(base):
    import pandas as pd
    from astroquery.mast import Observations

    attempts = [
        {"obs_collection": "HLSP", "provenance_name": "CLASSY"},
        {"obs_collection": "HST", "provenance_name": "CLASSY"},
        {"project": "CLASSY"},
    ]
    for criteria in attempts:
        try:
            table = Observations.query_criteria(**criteria)
            frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
            if not frame.empty:
                print(f"  CLASSY HLSP query {criteria} | observations={len(frame)}")
                return frame, "HLSP"
        except Exception as exc:
            print(f"  CLASSY HLSP query {criteria} | {type(exc).__name__}")

    print("  HLSP query unavailable; using HST Program 15840 fallback")
    return base.query_obs("HST", base.HST_PID), "PROGRAM_15840"


def product_target_map(base, observations):
    obsid_col = base.col(observations, ["obsid", "obs_id"])
    target_col = base.col(observations, ["target_name", "target"])
    ra_col = base.col(observations, ["s_ra", "ra"])
    dec_col = base.col(observations, ["s_dec", "dec"])
    exposure_col = base.col(observations, ["t_exptime", "exptime"])
    if obsid_col is None:
        raise RuntimeError("CLASSY observation table lacks observation IDs.")

    mapping = {}
    for _, row in observations.iterrows():
        obsid = str(row[obsid_col])
        mapping[obsid] = {
            "target": str(row[target_col]) if target_col is not None else obsid,
            "ra": fnum(row[ra_col]) if ra_col is not None else None,
            "dec": fnum(row[dec_col]) if dec_col is not None else None,
            "exposure": fnum(row[exposure_col]) if exposure_col is not None else 0.0,
        }
    return mapping, obsid_col


def header_redshift(path):
    from astropy.io import fits

    keys = ["REDSHIFT", "Z", "SPEC_Z", "ZSPEC", "TARG_Z", "TARGZ"]
    try:
        with fits.open(path, memmap=False) as hdul:
            headers = [hdu.header for hdu in hdul]
            for header in headers:
                for key in keys:
                    value = fnum(header.get(key))
                    if value is not None and 0.0 <= value < 0.3:
                        return value, f"FITS:{key}"
    except Exception:
        pass
    return None, "NONE"


def classy_curves_fast(base):
    import pandas as pd

    print("STEP 4/4 | Selecting a compact set of real CLASSY HST/COS spectra")
    observations, query_mode = query_classy_observations(base)
    mapping, obsid_col = product_target_map(base, observations)

    instrument_col = base.col(observations, ["instrument_name", "instrument"])
    if instrument_col is not None:
        cos_mask = observations[instrument_col].astype(str).str.contains("COS", case=False, na=False)
        if cos_mask.any():
            observations = observations[cos_mask].copy()

    target_col = base.col(observations, ["target_name", "target"])
    exposure_col = base.col(observations, ["t_exptime", "exptime"])
    if target_col is not None:
        observations["_exp"] = (
            pd.to_numeric(observations[exposure_col], errors="coerce").fillna(0.0)
            if exposure_col is not None
            else 0.0
        )
        preferred_targets = (
            observations.groupby(target_col)["_exp"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .index.astype(str)
        )
        observations = observations[
            observations[target_col].astype(str).isin(set(preferred_targets))
        ].copy()

    obsids = observations[obsid_col].dropna().astype(str).drop_duplicates().tolist()
    if not obsids:
        raise RuntimeError("No CLASSY observation IDs remained after filtering.")

    print(f"  CLASSY query mode={query_mode} | selected observation IDs={len(obsids)}")
    products = base.query_products(obsids)
    filename_col = base.col(products, ["productFilename", "productfilename"])
    if filename_col is None:
        raise RuntimeError("CLASSY products lack product filenames.")

    names = products[filename_col].astype(str).str.lower()
    fits_mask = names.str.endswith(".fits", na=False)
    preferred_mask = (
        names.str.contains("classy", na=False)
        | names.str.contains("coadd", na=False)
        | names.str.contains("x1d", na=False)
        | names.str.contains("spec", na=False)
    )
    candidates = products[fits_mask & preferred_mask].copy()
    if candidates.empty:
        candidates = products[fits_mask].copy()
    if candidates.empty:
        raise RuntimeError("No CLASSY/HST FITS spectral products were found.")

    size_col = base.col(candidates, ["size"])
    if size_col is not None:
        candidates = candidates.sort_values(size_col, ascending=False)
    candidates = candidates.drop_duplicates(subset=[filename_col]).head(MAX_CLASSY_PRODUCTS)

    product_obsid_col = base.col(candidates, ["obsID", "obsid", "obs_id", "parent_obsid"])
    spectra = []
    audit_rows = []
    resolved_redshifts = {}

    for index, record in enumerate(candidates.to_dict("records"), 1):
        product_name = str(record.get(filename_col, "UNKNOWN"))
        product_obsid = str(record.get(product_obsid_col, "")) if product_obsid_col else ""
        meta = mapping.get(product_obsid, {})
        target_name = str(meta.get("target", product_name))
        try:
            path = base.download(record, "HST_CLASSY_FAST")
            spectrum = base.read_hst(path)
            if spectrum is None:
                audit_rows.append((path.name, target_name, "NO_SPECTRUM", 0, math.nan, "NONE"))
                continue

            z_value, z_method = header_redshift(path)
            if z_value is None:
                cache_key = target_name
                if cache_key not in resolved_redshifts:
                    ra = meta.get("ra")
                    dec = meta.get("dec")
                    if ra is not None and dec is not None:
                        resolved_redshifts[cache_key] = base.redshift(target_name, ra, dec)
                    else:
                        resolved_redshifts[cache_key] = (None, "NO_COORDINATES")
                z_value, z_method = resolved_redshifts[cache_key]

            if z_value is None:
                audit_rows.append((path.name, target_name, "NO_REDSHIFT", len(spectrum["wave"]), math.nan, z_method))
                continue

            spectrum.update(name=target_name, z=float(z_value), z_method=z_method)
            spectra.append(spectrum)
            audit_rows.append((path.name, target_name, "OK", len(spectrum["wave"]), z_value, z_method))
            print(
                f"  {index:02d}/{len(candidates):02d} | {path.name} | "
                f"target={target_name} | z={z_value:.6f} | samples={len(spectrum['wave'])}"
            )
        except Exception as exc:
            print(f"  {index:02d}/{len(candidates):02d} | {product_name} | {type(exc).__name__}")
            audit_rows.append((product_name, target_name, type(exc).__name__, 0, math.nan, "NONE"))

    audit_path = CSV / f"{VERSION}_CLASSY_FAST_AUDIT.csv"
    pd.DataFrame(
        audit_rows,
        columns=["product", "target", "status", "samples", "redshift", "redshift_method"],
    ).to_csv(audit_path, index=False)

    if not spectra:
        raise RuntimeError(f"No usable real CLASSY spectra were resolved. Audit: {audit_path}")

    best = {}
    selection_rows = []
    for number, name, components, window in base.COMPLEXES:
        ranked = []
        for spectrum in spectra:
            score_value, peaks = base.score(
                spectrum,
                spectrum["z"],
                components,
                window,
            )
            ranked.append((score_value, peaks, spectrum))
            selection_rows.append(
                (
                    number,
                    name,
                    spectrum["name"],
                    spectrum["z"],
                    spectrum["path"].name,
                    score_value,
                    str(peaks),
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen = ranked[0]
        if not math.isfinite(chosen[0]):
            raise RuntimeError(f"No real CLASSY product covers {name}. Audit: {audit_path}")
        best[number] = {"score": chosen[0], "peaks": chosen[1], "spec": chosen[2]}
        print(
            f"  SELECTED | {number} {name} | analog={chosen[2]['name']} | "
            f"score={chosen[0]:.3f}"
        )

    selection_path = CSV / f"{VERSION}_CLASSY_SELECTION.csv"
    pd.DataFrame(
        selection_rows,
        columns=[
            "n",
            "complex",
            "target",
            "z",
            "product",
            "score",
            "component_peak_sigma",
        ],
    ).to_csv(selection_path, index=False)
    return best, audit_path, selection_path


def main():
    PNG.mkdir(parents=True, exist_ok=True)
    CSV.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    base = load_base(ensure_base())
    print(f"CODE OUTPUT: {VERSION}")
    print("MODE       : coordinate-cone JWST selection + compact CLASSY search")
    print()

    jwst_spectrum, jwst_meta = exact_momz14_cone(base)
    classy_refs, classy_audit, classy_selection = classy_curves_fast(base)
    plot_path, components_path, preflight_path = base.plot(
        jwst_spectrum,
        jwst_meta,
        classy_refs,
    )

    print()
    print(f"JWST separation arcsec : {jwst_meta['sep']:.6f}")
    print(f"JWST coordinate keys   : {jwst_meta['coord_source']}")
    print(f"JWST product           : {Path(jwst_meta['product']).name} HDU={jwst_meta['hdu']}")
    print(f"Plot PNG               : {plot_path}")
    print(f"Components CSV         : {components_path}")
    print(f"Preflight audit        : {preflight_path}")
    print(f"JWST X1D audit         : {jwst_meta['audit']}")
    print(f"CLASSY audit           : {classy_audit}")
    print(f"CLASSY selection       : {classy_selection}")
    print(datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"))
    print(f"# {VERSION}")


if __name__ == "__main__":
    main()
