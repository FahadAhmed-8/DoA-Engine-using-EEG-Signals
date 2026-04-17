#!/usr/bin/env python3
"""
CLI entry point for the VitalDB downloader.

Run from the project root:

    python 05_pipeline_v3/scripts/download_vitaldb.py --n 100

Common flags:

    --n INT                    how many successful cases to collect (default 100)
    --output-dir PATH          where .mat files land (default 03_data/vitaldb_raw)
    --catalogue PATH           catalogue CSV path (default 03_data/case_catalogue.csv)
    --seed INT                 RNG seed for case selection (default 42)
    --sleep SECONDS            pause between downloads (default 0.0)
    --dry-run                  print the case IDs the selector would pick, then exit
    --rebuild-catalogue        scan output dir, rebuild catalogue from files on disk
    --min-duration-min FLOAT   reject cases shorter than this (default 20.0)

Resumability
------------
The catalogue CSV is updated after every case, so Ctrl-C loses at most one
in-flight case. Re-running the command picks up where it left off.
"""
from __future__ import annotations

import argparse
import os
import sys

# Resolve paths so this script is runnable from the project root.
THIS_FILE = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(THIS_FILE)
V3_ROOT = os.path.dirname(SCRIPTS_DIR)                     # .../05_pipeline_v3
PROJECT_ROOT = os.path.dirname(V3_ROOT)                    # .../Mini Project 2

# Ensure `from data.vitaldb_downloader import ...` works.
sys.path.insert(0, V3_ROOT)

from data.vitaldb_downloader import (  # noqa: E402
    QCThresholds,
    SelectionSpec,
    build_catalogue,
    run_download,
    select_case_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100, help="target number of successful cases")
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "03_data", "vitaldb_raw"),
    )
    p.add_argument(
        "--catalogue",
        type=str,
        default=os.path.join(PROJECT_ROOT, "03_data", "case_catalogue.csv"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.0, help="seconds between downloads")
    p.add_argument("--dry-run", action="store_true", help="print first N case IDs and exit")
    p.add_argument(
        "--rebuild-catalogue",
        action="store_true",
        help="scan output dir, rebuild catalogue from files on disk",
    )
    p.add_argument("--min-duration-min", type=float, default=20.0)
    p.add_argument("--max-bis-nan-pct", type=float, default=50.0)
    p.add_argument("--max-eeg-nan-pct", type=float, default=20.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.rebuild_catalogue:
        print(f"Scanning {args.output_dir} and rebuilding {args.catalogue}...")
        df = build_catalogue(args.output_dir, args.catalogue)
        print(f"Rebuilt catalogue with {len(df)} entries.")
        return 0

    spec = SelectionSpec(n_target=args.n, seed=args.seed)

    if args.dry_run:
        ids = select_case_ids(spec)
        print(f"Would attempt up to {len(ids)} cases (target={args.n}, seed={args.seed}).")
        print("First 30 case IDs:", ids[:30])
        return 0

    thresholds = QCThresholds(
        min_duration_min=args.min_duration_min,
        max_bis_nan_pct=args.max_bis_nan_pct,
        max_eeg_nan_pct=args.max_eeg_nan_pct,
    )

    df = run_download(
        n_target=args.n,
        output_dir=args.output_dir,
        catalogue_path=args.catalogue,
        spec=spec,
        thresholds=thresholds,
        sleep_seconds=args.sleep,
    )

    success = (df["status"] == "downloaded").sum()
    print(f"\nCatalogue: {args.catalogue}")
    print(f"Files dir: {args.output_dir}")
    print(f"Success  : {success} / {len(df)} attempts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
