#!/usr/bin/env python3
"""
09_prepare_n100_dataset.py
==========================

Prepare the v2 pipeline to run on the 100-case VitalDB cohort (or any subset).

What this script does
---------------------
1. Reads ``03_data/case_catalogue.csv`` (the canonical record of downloaded cases).
2. Selects the first N cases (default 100) where ``qc_notes == 'ok'``.
3. For each case i = 1..N:
     * Loads ``03_data/vitaldb_raw/<vitaldb_XXXX.mat>``.
     * Converts BIS NaN -> -1 (v2's ``BIS_MISSING_VALUE`` sentinel).
     * Writes to ``04_pipeline_v2/dataset_n<N>/case<i>.mat`` with the same
       HDF5 dataset names (``EEG`` and ``bis``) that v2's loader expects.
4. Saves a traceability map at ``dataset_n<N>/case_id_map.csv``.

Why this exists
---------------
v2 was written to load files named ``case1.mat``, ``case2.mat``, ... using
``BIS_MISSING_VALUE = -1``. New VitalDB downloads use the real VitalDB
caseid in the filename and preserve BIS NaN as-is. This script bridges
those two conventions *without modifying v2 code* — the v2 pipeline is
frozen; only its configuration and data layout are parameterised.

Usage
-----

::

    # default: 100 cases, reads 03_data/vitaldb_raw, writes 04_pipeline_v2/dataset_n100
    python 04_pipeline_v2/scripts/09_prepare_n100_dataset.py

    # different cohort size
    python 04_pipeline_v2/scripts/09_prepare_n100_dataset.py --n 50

    # force rebuild
    python 04_pipeline_v2/scripts/09_prepare_n100_dataset.py --force

The script is idempotent by default: if ``case<i>.mat`` already exists in
the target directory, it is skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = V2_DIR.parent


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare v2-compatible case1..caseN.mat files from the VitalDB cohort.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of cases to prepare (case1..caseN)",
    )
    p.add_argument(
        "--source-dir",
        type=Path,
        default=PROJECT_ROOT / "03_data" / "vitaldb_raw",
        help="Directory containing raw VitalDB .mat files",
    )
    p.add_argument(
        "--catalogue",
        type=Path,
        default=PROJECT_ROOT / "03_data" / "case_catalogue.csv",
        help="Path to case_catalogue.csv",
    )
    p.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Target directory (default: 04_pipeline_v2/dataset_n<N>)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every case even if case<i>.mat already exists",
    )
    p.add_argument(
        "--bis-missing",
        type=float,
        default=-1.0,
        help="Sentinel value v2 treats as missing BIS",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_case_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read EEG and BIS arrays from a VitalDB-downloaded .mat (HDF5 v7.3)."""
    with h5py.File(path, "r") as f:
        if "EEG" in f:
            eeg = np.array(f["EEG"]).flatten().astype(np.float32)
        elif "eeg" in f:
            eeg = np.array(f["eeg"]).flatten().astype(np.float32)
        else:
            raise KeyError(f"No 'EEG' or 'eeg' dataset in {path.name}")

        if "bis" in f:
            bis = np.array(f["bis"]).flatten().astype(np.float32)
        elif "BIS" in f:
            bis = np.array(f["BIS"]).flatten().astype(np.float32)
        else:
            raise KeyError(f"No 'bis' or 'BIS' dataset in {path.name}")
    return eeg, bis


def write_case_mat(path: Path, eeg: np.ndarray, bis: np.ndarray) -> None:
    """Write EEG + BIS in v2's expected HDF5 layout (datasets 'EEG' and 'bis', both (1, N))."""
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "EEG",
            data=eeg.reshape(1, -1).astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "bis",
            data=bis.reshape(1, -1).astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def select_cases(catalogue: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return the first `n` qc-passed cases, sorted by vitaldb_caseid."""
    ok = catalogue[
        (catalogue["status"] == "downloaded") & (catalogue["qc_notes"] == "ok")
    ].copy()
    if len(ok) < n:
        raise ValueError(
            f"Catalogue has only {len(ok)} QC-passed cases; requested {n}."
        )
    ok = ok.sort_values("vitaldb_caseid").reset_index(drop=True)
    selected = ok.head(n).copy()
    selected["v2_case_num"] = range(1, n + 1)
    selected["v2_filename"] = [f"case{i}.mat" for i in selected["v2_case_num"]]
    return selected


def prepare_case(
    source: Path,
    target: Path,
    bis_missing_sentinel: float,
) -> dict:
    """Read source .mat, translate BIS NaN -> sentinel, write target .mat."""
    eeg, bis = read_case_mat(source)
    bis_nan_count = int(np.isnan(bis).sum())
    bis_clean = np.where(np.isnan(bis), bis_missing_sentinel, bis).astype(np.float32)
    write_case_mat(target, eeg, bis_clean)
    return {
        "n_eeg": eeg.size,
        "n_bis": bis.size,
        "bis_nan_converted": bis_nan_count,
        "eeg_nan": int(np.isnan(eeg).sum()),
    }


def main() -> int:
    args = parse_args()

    target_dir = args.target_dir or (V2_DIR / f"dataset_n{args.n}")

    # Validation
    if not args.catalogue.exists():
        print(f"[err] catalogue not found: {args.catalogue}", file=sys.stderr)
        return 1
    if not args.source_dir.exists():
        print(f"[err] source dir not found: {args.source_dir}", file=sys.stderr)
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] cohort size      : {args.n}")
    print(f"[cfg] source dir       : {args.source_dir}")
    print(f"[cfg] target dir       : {target_dir}")
    print(f"[cfg] catalogue        : {args.catalogue}")
    print(f"[cfg] bis_missing      : {args.bis_missing}")
    print(f"[cfg] force rebuild    : {args.force}")
    print()

    # Pick cases
    catalogue = pd.read_csv(args.catalogue)
    try:
        selected = select_cases(catalogue, args.n)
    except ValueError as e:
        print(f"[err] {e}", file=sys.stderr)
        return 1

    # Save map up-front so traceability exists even on partial runs
    map_path = target_dir / "case_id_map.csv"
    selected[["v2_case_num", "v2_filename", "vitaldb_caseid", "filename"]].to_csv(
        map_path, index=False
    )
    print(f"[map] wrote {map_path}")
    print()

    # Process each case
    t0 = time.time()
    processed = 0
    skipped = 0
    failed = 0

    for row in selected.itertuples():
        src = args.source_dir / row.filename
        dst = target_dir / row.v2_filename

        if dst.exists() and not args.force:
            skipped += 1
            continue
        if not src.exists():
            print(f"[warn] missing source: {src.name} (case {row.v2_case_num})")
            failed += 1
            continue

        try:
            info = prepare_case(src, dst, args.bis_missing)
            processed += 1
            if processed % 10 == 0 or processed == args.n - skipped:
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 1e-6)
                eta = (args.n - skipped - processed) / max(rate, 1e-6)
                print(
                    f"[prep] {processed:3d}/{args.n - skipped} "
                    f"| case{row.v2_case_num:<3d} <- {row.filename:<20s} "
                    f"| bis_nan_fixed={info['bis_nan_converted']:3d} "
                    f"| {elapsed/60:4.1f}m elapsed, ~{eta/60:4.1f}m remaining"
                )
        except Exception as e:  # noqa: BLE001
            print(f"[err ] case{row.v2_case_num} ({row.filename}): {e}")
            failed += 1
            continue

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed/60:.1f} minutes.")
    print(f"  processed : {processed}")
    print(f"  skipped   : {skipped} (already existed; re-run with --force to rebuild)")
    print(f"  failed    : {failed}")
    print(f"  target    : {target_dir}")
    print(f"  mapping   : {map_path}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Set cohort size:   $env:V2_NUM_CASES = {args.n}")
    print(f"  2. Preprocess:        python 04_pipeline_v2/scripts/02_preprocess.py")
    print(f"  3. Extract features:  python 04_pipeline_v2/scripts/03_extract_features.py")
    print(f"  4. Run LOPO:          python 04_pipeline_v2/scripts/05_train_lopo.py")
    print()

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
