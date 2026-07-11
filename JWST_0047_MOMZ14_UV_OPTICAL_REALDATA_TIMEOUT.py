# JWST_0047
# MoM-z14 real-data workflow reliability patch for JWST_0046.
# Uses bounded MAST queries, hard process timeouts, retries, and timed downloads.
# No AI images. Scientific plots remain matplotlib-only and use real MAST data.

from pathlib import Path
from datetime import datetime, timezone
import importlib.util
import multiprocessing as mp
import subprocess
import time
import warnings

VERSION = "JWST_0047"
BASE_FILENAME = "JWST_0046_MOMZ14_UV_OPTICAL_REALDATA.py"
BASE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/JWST/main/"
    + BASE_FILENAME
)
BASE_PATH = Path("/content") / BASE_FILENAME if Path("/content").exists() else Path.cwd() / BASE_FILENAME
OUT = Path("/content/JWST_OUTPUT") if Path("/content").exists() else Path.cwd() / "JWST_OUTPUT"
CSV = OUT / "CSV"
DATA = OUT / "DATA"
QUERY_TIMEOUT_S = 50
QUERY_RETRIES = 2
BATCH_SIZE = 6
MAX_OBSERVATIONS = 36
DOWNLOAD_CONNECT_TIMEOUT_S = 15
DOWNLOAD_READ_TIMEOUT_S = 120


def _ensure_base_script():
    if BASE_PATH.exists() and BASE_PATH.stat().st_size > 10000:
        return BASE_PATH
    BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fsSL", "--connect-timeout", "15", "--max-time", "60", "-o", str(BASE_PATH), BASE_URL],
        check=True,
        timeout=70,
    )
    if not BASE_PATH.exists() or BASE_PATH.stat().st_size < 10000:
        raise RuntimeError(f"Base script download failed: {BASE_PATH}")
    return BASE_PATH


def _load_base_module(path):
    spec = importlib.util.spec_from_file_location("jwst_0046_base", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load base script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _child_observation_query(program_id, out_csv, error_txt):
    try:
        import pandas as pd
        from astroquery.mast import Observations

        table = Observations.query_criteria(obs_collection="JWST", proposal_id=str(program_id))
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
        frame.to_csv(out_csv, index=False)
    except BaseException as exc:
        Path(error_txt).write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def _child_product_query(obsids, out_csv, error_txt):
    try:
        import pandas as pd
        from astroquery.mast import Observations

        table = Observations.get_product_list([str(x) for x in obsids])
        frame = table.to_pandas() if hasattr(table, "to_pandas") else pd.DataFrame(table)
        frame.to_csv(out_csv, index=False)
    except BaseException as exc:
        Path(error_txt).write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise


def _run_bounded(target, args, timeout_s):
    ctx = mp.get_context("fork")
    process = ctx.Process(target=target, args=args)
    process.start()
    process.join(timeout_s)
    if process.is_alive():
        process.terminate()
        process.join(5)
        if process.is_alive():
            process.kill()
            process.join(2)
        return False, "TIMEOUT"
    if process.exitcode != 0:
        return False, f"EXIT_{process.exitcode}"
    return True, "OK"


def _find_column(frame, names):
    lower = {str(c).lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def _prioritize_observations(obs_df):
    selected = obs_df.copy()
    instrument_col = _find_column(selected, ["instrument_name", "instrument"])
    target_col = _find_column(selected, ["target_name", "target"])

    if instrument_col is not None:
        nirspec = selected[instrument_col].astype(str).str.contains("NIRSPEC", case=False, na=False)
        if nirspec.any():
            selected = selected[nirspec].copy()

    if target_col is not None:
        target_mask = selected[target_col].astype(str).str.contains(
            "MOM|Z14|JADES|GS-Z14", case=False, na=False, regex=True
        )
        if target_mask.any():
            selected = selected[target_mask].copy()

    obsid_col = _find_column(selected, ["obsid", "obs_id"])
    if obsid_col is None:
        return selected.iloc[0:0].copy(), None

    selected = selected.dropna(subset=[obsid_col]).copy()
    selected[obsid_col] = selected[obsid_col].astype(str)
    selected = selected.drop_duplicates(subset=[obsid_col]).head(MAX_OBSERVATIONS)
    return selected, obsid_col


def query_mast_products_safe(base):
    import pandas as pd

    CSV.mkdir(parents=True, exist_ok=True)
    query_dir = CSV / f"{VERSION}_MAST_QUERY_PARTS"
    query_dir.mkdir(parents=True, exist_ok=True)

    obs_path = CSV / f"{VERSION}_PROGRAM_{base.PROGRAM_ID}_OBSERVATIONS.csv"
    obs_error = query_dir / "observations_error.txt"
    obs_temp = query_dir / "observations.csv"

    ok = False
    status = "NOT_RUN"
    for attempt in range(1, QUERY_RETRIES + 1):
        print(f"MAST observation query | attempt {attempt}/{QUERY_RETRIES} | timeout {QUERY_TIMEOUT_S}s")
        ok, status = _run_bounded(
            _child_observation_query,
            (base.PROGRAM_ID, str(obs_temp), str(obs_error)),
            QUERY_TIMEOUT_S,
        )
        if ok and obs_temp.exists():
            break
        time.sleep(2)

    if not ok or not obs_temp.exists():
        detail = obs_error.read_text(encoding="utf-8").strip() if obs_error.exists() else status
        warnings.warn(f"MAST observation query failed safely: {detail}")
        pd.DataFrame([{"query_status": status, "detail": detail}]).to_csv(obs_path, index=False)
        return [], obs_path, None

    obs_df = pd.read_csv(obs_temp)
    obs_df.to_csv(obs_path, index=False)
    if obs_df.empty:
        return [], obs_path, None

    selected, obsid_col = _prioritize_observations(obs_df)
    selected_path = CSV / f"{VERSION}_PROGRAM_{base.PROGRAM_ID}_SELECTED_OBSERVATIONS.csv"
    selected.to_csv(selected_path, index=False)
    if obsid_col is None or selected.empty:
        warnings.warn("No usable MAST observation IDs were found after filtering.")
        return [], obs_path, selected_path

    obsids = selected[obsid_col].astype(str).tolist()
    product_frames = []
    audit_rows = []

    for start in range(0, len(obsids), BATCH_SIZE):
        batch = obsids[start : start + BATCH_SIZE]
        batch_number = start // BATCH_SIZE + 1
        part_csv = query_dir / f"products_batch_{batch_number:02d}.csv"
        error_txt = query_dir / f"products_batch_{batch_number:02d}_error.txt"
        batch_ok = False
        batch_status = "NOT_RUN"

        for attempt in range(1, QUERY_RETRIES + 1):
            print(
                f"MAST product batch {batch_number:02d} | obs={len(batch)} | "
                f"attempt {attempt}/{QUERY_RETRIES} | timeout {QUERY_TIMEOUT_S}s"
            )
            batch_ok, batch_status = _run_bounded(
                _child_product_query,
                (batch, str(part_csv), str(error_txt)),
                QUERY_TIMEOUT_S,
            )
            if batch_ok and part_csv.exists():
                break
            time.sleep(2)

        detail = error_txt.read_text(encoding="utf-8").strip() if error_txt.exists() else ""
        audit_rows.append(
            {
                "batch": batch_number,
                "observation_count": len(batch),
                "status": "OK" if batch_ok else batch_status,
                "detail": detail,
            }
        )
        if batch_ok and part_csv.exists():
            frame = pd.read_csv(part_csv)
            if not frame.empty:
                product_frames.append(frame)

    audit_path = CSV / f"{VERSION}_PROGRAM_{base.PROGRAM_ID}_QUERY_AUDIT.csv"
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    if not product_frames:
        warnings.warn(f"No MAST product batches completed. Audit: {audit_path}")
        return [], obs_path, audit_path

    prod_df = pd.concat(product_frames, ignore_index=True, sort=False).drop_duplicates()
    prod_path = CSV / f"{VERSION}_PROGRAM_{base.PROGRAM_ID}_PRODUCTS.csv"
    prod_df.to_csv(prod_path, index=False)

    fn_col = _find_column(prod_df, ["productFilename", "productfilename"])
    subgroup = _find_column(prod_df, ["productSubGroupDescription"])
    mask = pd.Series(False, index=prod_df.index)
    if fn_col is not None:
        names = prod_df[fn_col].astype(str).str.lower()
        mask |= names.str.contains("x1d", na=False) & names.str.endswith(".fits", na=False)
        mask |= names.str.contains("s2d", na=False) & names.str.endswith(".fits", na=False)
    if subgroup is not None:
        mask |= prod_df[subgroup].astype(str).str.upper().isin(["X1D", "S2D"])

    candidates = prod_df[mask].copy()
    if candidates.empty and fn_col is not None:
        candidates = prod_df[
            prod_df[fn_col].astype(str).str.lower().str.endswith(".fits", na=False)
        ].copy()

    cand_path = CSV / f"{VERSION}_PROGRAM_{base.PROGRAM_ID}_SPECTRUM_CANDIDATES.csv"
    candidates.to_csv(cand_path, index=False)
    print(f"MAST bounded query complete | products={len(prod_df)} | candidates={len(candidates)}")
    return candidates.to_dict("records"), obs_path, cand_path


def download_product_safe(record):
    import requests

    DATA.mkdir(parents=True, exist_ok=True)
    uri = record.get("dataURI") or record.get("dataUri") or record.get("uri")
    filename = record.get("productFilename") or record.get("productfilename") or "mast_product.fits"
    local = DATA / str(filename)
    partial = local.with_suffix(local.suffix + ".part")

    if local.exists() and local.stat().st_size > 100000:
        return local, "cached"
    if not uri:
        raise RuntimeError("No dataURI in MAST product record.")

    endpoint = "https://mast.stsci.edu/api/v0.1/Download/file"
    last_error = None
    for attempt in range(1, 4):
        try:
            print(f"MAST product download | {filename} | attempt {attempt}/3")
            with requests.get(
                endpoint,
                params={"uri": str(uri)},
                stream=True,
                timeout=(DOWNLOAD_CONNECT_TIMEOUT_S, DOWNLOAD_READ_TIMEOUT_S),
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            if partial.stat().st_size < 100000:
                raise RuntimeError(f"Downloaded file too small: {partial.stat().st_size} bytes")
            partial.replace(local)
            return local, "downloaded-from-mast-timed"
        except Exception as exc:
            last_error = exc
            partial.unlink(missing_ok=True)
            time.sleep(2 * attempt)

    raise RuntimeError(f"Timed MAST download failed for {filename}: {last_error}")


def main():
    print(f"CODE OUTPUT: {VERSION}")
    print("MAST MODE  : bounded batches + hard timeouts + retries")
    print()

    base_path = _ensure_base_script()
    base = _load_base_module(base_path)
    base.VERSION = VERSION
    base.query_mast_products = lambda: query_mast_products_safe(base)
    base.download_product = download_product_safe
    base.main()


if __name__ == "__main__":
    main()
