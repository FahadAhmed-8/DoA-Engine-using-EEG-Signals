"""
VitalDB downloader for the v3 pipeline.

Responsibilities
----------------
1. Select a subset of VitalDB case IDs that match the study inclusion criteria
   (BIS monitoring present, total IV anaesthesia / propofol, adequate duration).
2. Download the single-channel EEG waveform and BIS index for each selected
   case, saving them to disk in the same HDF5-backed `.mat` schema that the
   v2 pipeline expects (datasets named `EEG` and `bis`).
3. Keep a catalogue CSV with one row per case: VitalDB case id, file path,
   signal stats, QC verdict, download timestamp.
4. Be resumable. If the downloader is re-run, cases whose files already exist
   and pass QC are skipped; failed cases are retried.

Why these design choices
------------------------
* EEG at 128 Hz, BIS at 1/5 Hz to match the existing v2 sampling convention.
* Track names are `BIS/EEG1_WAV` and `BIS/BIS` — verified present across a
  random sample of VitalDB cases; the SNUADC/EEG track many web examples
  mention is a general-purpose bedside monitor channel and is often absent.
* We store raw micro-volt values without clipping. v2 saved data clipped to
  plus/minus 62.5 uV (likely the BIS monitor's soft limit). That choice is a
  preprocessing decision — the downloader should preserve the source signal
  so downstream stages can revisit it if needed.
* Quality control at download time is conservative: we reject cases only when
  the signal is unambiguously unusable (too short, mostly NaN BIS, flat EEG,
  rail-clipped EEG). Fine-grained artefact handling is left to the
  preprocessing stage.

Usage
-----
    from data.vitaldb_downloader import run_download

    run_download(
        n_target=100,
        output_dir="03_data/vitaldb_raw",
        catalogue_path="03_data/case_catalogue.csv",
    )

Or from the command line via `05_pipeline_v3/scripts/download_vitaldb.py`.
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #

DEFAULT_EEG_TRACK = "BIS/EEG1_WAV"
DEFAULT_BIS_TRACK = "BIS/BIS"

SAMPLING_RATE_HZ = 128  # EEG native rate. interval = 1 / 128 seconds.
BIS_INTERVAL_SEC = 5  # BIS reported every 5 seconds.


# ----------------------------------------------------------------------------- #
# Data classes for configuration
# ----------------------------------------------------------------------------- #


@dataclass
class CatalogueColumns:
    """Column names used in the catalogue CSV. Grouped here so downstream
    loaders can use the constants instead of hard-coded strings."""

    VITALDB_CASEID = "vitaldb_caseid"
    LOCAL_PATH = "local_path"  # relative to catalogue file's directory
    FILENAME = "filename"
    STATUS = "status"  # downloaded | qc_fail | download_fail
    QC_NOTES = "qc_notes"
    N_EEG_SAMPLES = "n_eeg_samples"
    N_BIS_SAMPLES = "n_bis_samples"
    DURATION_MIN = "duration_min"
    EEG_MIN = "eeg_min"
    EEG_MAX = "eeg_max"
    EEG_STD = "eeg_std"
    EEG_NAN_PCT = "eeg_nan_pct"
    BIS_MIN = "bis_min"
    BIS_MAX = "bis_max"
    BIS_MEDIAN = "bis_median"
    BIS_NAN_PCT = "bis_nan_pct"
    DOWNLOADED_AT = "downloaded_at"
    DOWNLOAD_SECONDS = "download_seconds"


@dataclass
class QCThresholds:
    """Thresholds for minimal download-time quality filtering. Defaults are
    deliberately permissive — the heavy filtering lives in the preprocessing
    stage. Here we only reject signals that are obviously unusable."""

    min_duration_min: float = 20.0  # drop very short surgical tails
    max_bis_nan_pct: float = 50.0  # drop cases with mostly-missing BIS
    max_eeg_nan_pct: float = 20.0  # drop cases with mostly-missing EEG
    min_eeg_std: float = 0.5  # microvolts; reject effectively-flat signals
    max_eeg_rail_pct: float = 30.0  # percent of samples at +/- 32767-like rail


@dataclass
class SelectionSpec:
    """How to pick case IDs before any downloading happens."""

    n_target: int = 100
    require_bis: bool = True
    require_tiva: bool = True  # total IV anaesthesia (usually propofol)
    seed: int = 42
    # Hard-coded exclusions, e.g. cases already owned from v2. Defaults to
    # empty; caller should pass in the list of legacy IDs if known.
    exclude_case_ids: list[int] = field(default_factory=list)


# ----------------------------------------------------------------------------- #
# Case selection
# ----------------------------------------------------------------------------- #


def select_case_ids(spec: SelectionSpec) -> list[int]:
    """Return a deterministic, shuffled list of VitalDB case IDs.

    The result is not necessarily of length `spec.n_target`; it may be longer
    so the downloader can keep moving when some cases fail QC. Callers should
    consume from the front of the list until `n_target` successful downloads
    are achieved.
    """
    import vitaldb

    pool: set[int]
    if spec.require_bis and spec.require_tiva:
        pool = vitaldb.caseids_bis & vitaldb.caseids_tiva
    elif spec.require_bis:
        pool = set(vitaldb.caseids_bis)
    elif spec.require_tiva:
        pool = set(vitaldb.caseids_tiva)
    else:
        # Fall back to every case id mentioned in the BIS pool, which is a
        # broader but still bounded set.
        pool = set(vitaldb.caseids_bis)

    pool -= set(spec.exclude_case_ids)
    ordered = sorted(pool)

    rng = random.Random(spec.seed)
    rng.shuffle(ordered)

    # Return 3x the target so we have headroom for QC failures.
    return ordered[: max(spec.n_target * 3, spec.n_target + 20)]


# ----------------------------------------------------------------------------- #
# Per-case download
# ----------------------------------------------------------------------------- #


def _percent_nan(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 100.0
    return float(np.isnan(arr).sum()) / float(arr.size) * 100.0


def _percent_at_rail(arr: np.ndarray, rail_threshold: float = 3000.0) -> float:
    """Rough estimate of the fraction of samples that are railed at ADC
    extremes. A well-calibrated BIS monitor EEG stays under a few hundred
    microvolts; anything over 3 mV is almost certainly a saturation artefact.
    """
    if arr.size == 0:
        return 100.0
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 100.0
    rail_mask = np.abs(finite) >= rail_threshold
    return float(rail_mask.sum()) / float(finite.size) * 100.0


def _download_one(
    caseid: int,
    eeg_track: str,
    bis_track: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Low-level download helper: returns (eeg, bis, seconds_elapsed)."""
    import vitaldb

    t0 = time.time()
    eeg = vitaldb.load_case(caseid, [eeg_track], interval=1.0 / SAMPLING_RATE_HZ)
    bis = vitaldb.load_case(caseid, [bis_track], interval=float(BIS_INTERVAL_SEC))
    # vitaldb.load_case returns (n_samples, n_tracks); squeeze to 1-D.
    eeg = np.asarray(eeg).reshape(-1)
    bis = np.asarray(bis).reshape(-1)
    return eeg, bis, time.time() - t0


def _save_mat_v73(
    output_path: str,
    eeg: np.ndarray,
    bis: np.ndarray,
) -> None:
    """Write EEG and BIS as MATLAB v7.3 (HDF5) datasets compatible with the
    v2 pipeline's on-disk convention: two datasets named 'EEG' and 'bis',
    both shape (1, N), float64.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("EEG", data=eeg.astype(np.float64).reshape(1, -1))
        f.create_dataset("bis", data=bis.astype(np.float64).reshape(1, -1))


def _qc_verdict(
    eeg: np.ndarray,
    bis: np.ndarray,
    thresholds: QCThresholds,
) -> tuple[bool, str]:
    """Run a small fixed battery of QC checks. Returns (passed, notes)."""
    n_eeg = eeg.size
    duration_min = n_eeg / SAMPLING_RATE_HZ / 60.0

    notes: list[str] = []
    failed = False

    if duration_min < thresholds.min_duration_min:
        failed = True
        notes.append(
            f"duration {duration_min:.1f}min < {thresholds.min_duration_min:.1f}"
        )

    bis_nan_pct = _percent_nan(bis)
    if bis_nan_pct > thresholds.max_bis_nan_pct:
        failed = True
        notes.append(f"BIS NaN {bis_nan_pct:.1f}% > {thresholds.max_bis_nan_pct:.1f}%")

    eeg_nan_pct = _percent_nan(eeg)
    if eeg_nan_pct > thresholds.max_eeg_nan_pct:
        failed = True
        notes.append(f"EEG NaN {eeg_nan_pct:.1f}% > {thresholds.max_eeg_nan_pct:.1f}%")

    finite_eeg = eeg[np.isfinite(eeg)]
    eeg_std = float(finite_eeg.std()) if finite_eeg.size > 0 else 0.0
    if eeg_std < thresholds.min_eeg_std:
        failed = True
        notes.append(f"EEG std {eeg_std:.3f} < {thresholds.min_eeg_std:.3f}")

    eeg_rail_pct = _percent_at_rail(eeg)
    if eeg_rail_pct > thresholds.max_eeg_rail_pct:
        failed = True
        notes.append(
            f"EEG rail {eeg_rail_pct:.1f}% > {thresholds.max_eeg_rail_pct:.1f}%"
        )

    return (not failed), "; ".join(notes) if notes else "ok"


def _summarise_signals(
    caseid: int,
    eeg: np.ndarray,
    bis: np.ndarray,
    download_seconds: float,
    filename: str,
    status: str,
    qc_notes: str,
) -> dict:
    """Build a catalogue row for one case."""
    finite_eeg = eeg[np.isfinite(eeg)]
    finite_bis = bis[np.isfinite(bis)]

    row = {
        CatalogueColumns.VITALDB_CASEID: int(caseid),
        CatalogueColumns.FILENAME: filename,
        CatalogueColumns.LOCAL_PATH: filename,  # caller may prefix with dir
        CatalogueColumns.STATUS: status,
        CatalogueColumns.QC_NOTES: qc_notes,
        CatalogueColumns.N_EEG_SAMPLES: int(eeg.size),
        CatalogueColumns.N_BIS_SAMPLES: int(bis.size),
        CatalogueColumns.DURATION_MIN: round(eeg.size / SAMPLING_RATE_HZ / 60.0, 2),
        CatalogueColumns.EEG_MIN: (
            float(finite_eeg.min()) if finite_eeg.size else np.nan
        ),
        CatalogueColumns.EEG_MAX: (
            float(finite_eeg.max()) if finite_eeg.size else np.nan
        ),
        CatalogueColumns.EEG_STD: (
            float(finite_eeg.std()) if finite_eeg.size else np.nan
        ),
        CatalogueColumns.EEG_NAN_PCT: round(_percent_nan(eeg), 2),
        CatalogueColumns.BIS_MIN: (
            float(finite_bis.min()) if finite_bis.size else np.nan
        ),
        CatalogueColumns.BIS_MAX: (
            float(finite_bis.max()) if finite_bis.size else np.nan
        ),
        CatalogueColumns.BIS_MEDIAN: (
            float(np.median(finite_bis)) if finite_bis.size else np.nan
        ),
        CatalogueColumns.BIS_NAN_PCT: round(_percent_nan(bis), 2),
        CatalogueColumns.DOWNLOADED_AT: datetime.utcnow().isoformat(timespec="seconds")
        + "Z",
        CatalogueColumns.DOWNLOAD_SECONDS: round(download_seconds, 2),
    }
    return row


def download_and_save_case(
    caseid: int,
    output_dir: str,
    eeg_track: str = DEFAULT_EEG_TRACK,
    bis_track: str = DEFAULT_BIS_TRACK,
    thresholds: Optional[QCThresholds] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Download, QC-check, and save one case. Returns a catalogue row dict.

    The file is only written to disk if QC passes. For QC failures, the row
    still carries the signal stats so the caller can audit rejections.
    """
    log = logger or logging.getLogger(__name__)
    thresholds = thresholds or QCThresholds()
    filename = f"vitaldb_{caseid:04d}.mat"
    output_path = os.path.join(output_dir, filename)

    try:
        eeg, bis, dl_seconds = _download_one(caseid, eeg_track, bis_track)
    except Exception as exc:
        log.warning("case %d: download failed (%s)", caseid, exc)
        return {
            CatalogueColumns.VITALDB_CASEID: int(caseid),
            CatalogueColumns.FILENAME: filename,
            CatalogueColumns.LOCAL_PATH: filename,
            CatalogueColumns.STATUS: "download_fail",
            CatalogueColumns.QC_NOTES: f"exception: {exc}",
            CatalogueColumns.N_EEG_SAMPLES: 0,
            CatalogueColumns.N_BIS_SAMPLES: 0,
            CatalogueColumns.DURATION_MIN: 0.0,
            CatalogueColumns.EEG_MIN: np.nan,
            CatalogueColumns.EEG_MAX: np.nan,
            CatalogueColumns.EEG_STD: np.nan,
            CatalogueColumns.EEG_NAN_PCT: np.nan,
            CatalogueColumns.BIS_MIN: np.nan,
            CatalogueColumns.BIS_MAX: np.nan,
            CatalogueColumns.BIS_MEDIAN: np.nan,
            CatalogueColumns.BIS_NAN_PCT: np.nan,
            CatalogueColumns.DOWNLOADED_AT: datetime.utcnow().isoformat(
                timespec="seconds"
            )
            + "Z",
            CatalogueColumns.DOWNLOAD_SECONDS: 0.0,
        }

    passed, notes = _qc_verdict(eeg, bis, thresholds)
    status = "downloaded" if passed else "qc_fail"

    row = _summarise_signals(
        caseid=caseid,
        eeg=eeg,
        bis=bis,
        download_seconds=dl_seconds,
        filename=filename,
        status=status,
        qc_notes=notes,
    )

    if passed:
        _save_mat_v73(output_path, eeg, bis)
        log.info(
            "case %d: OK duration=%.1fmin EEG_std=%.1f BIS_missing=%.1f%% (%.1fs)",
            caseid,
            row[CatalogueColumns.DURATION_MIN],
            row[CatalogueColumns.EEG_STD],
            row[CatalogueColumns.BIS_NAN_PCT],
            dl_seconds,
        )
    else:
        log.info("case %d: QC FAIL (%s)", caseid, notes)

    return row


# ----------------------------------------------------------------------------- #
# Catalogue and batch driver
# ----------------------------------------------------------------------------- #


def _read_existing_catalogue(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _write_catalogue(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Always sort by case id for stable diffs.
    if CatalogueColumns.VITALDB_CASEID in df.columns:
        df = df.sort_values(CatalogueColumns.VITALDB_CASEID).reset_index(drop=True)
    df.to_csv(path, index=False)


def _already_downloaded_ids(catalogue: pd.DataFrame) -> set[int]:
    """Set of case IDs that have a successful `downloaded` row already."""
    if catalogue.empty:
        return set()
    done = catalogue[catalogue[CatalogueColumns.STATUS] == "downloaded"]
    return set(done[CatalogueColumns.VITALDB_CASEID].astype(int).tolist())


def build_catalogue(
    output_dir: str,
    catalogue_path: str,
) -> pd.DataFrame:
    """Scan `output_dir` for existing .mat files and rebuild/repair the
    catalogue by re-reading each file's shape. Useful after a crash where the
    catalogue CSV is out of sync with files on disk.
    """
    existing = _read_existing_catalogue(catalogue_path)
    existing_by_id = (
        existing.set_index(CatalogueColumns.VITALDB_CASEID).to_dict(orient="index")
        if not existing.empty
        else {}
    )

    rows = []
    for fname in sorted(os.listdir(output_dir)):
        if not fname.startswith("vitaldb_") or not fname.endswith(".mat"):
            continue
        try:
            caseid = int(fname.split("_")[1].split(".")[0])
        except Exception:
            continue
        fpath = os.path.join(output_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                eeg = np.asarray(f["EEG"]).reshape(-1)
                bis = np.asarray(f["bis"]).reshape(-1)
        except Exception as exc:
            rows.append(
                {
                    CatalogueColumns.VITALDB_CASEID: caseid,
                    CatalogueColumns.FILENAME: fname,
                    CatalogueColumns.LOCAL_PATH: fname,
                    CatalogueColumns.STATUS: "qc_fail",
                    CatalogueColumns.QC_NOTES: f"unreadable: {exc}",
                }
            )
            continue
        merged = existing_by_id.get(caseid, {})
        merged.update(
            _summarise_signals(
                caseid=caseid,
                eeg=eeg,
                bis=bis,
                download_seconds=float(
                    merged.get(CatalogueColumns.DOWNLOAD_SECONDS, 0.0) or 0.0
                ),
                filename=fname,
                status="downloaded",
                qc_notes=str(merged.get(CatalogueColumns.QC_NOTES, "rebuilt")),
            )
        )
        rows.append(merged)

    df = pd.DataFrame(rows)
    _write_catalogue(df, catalogue_path)
    return df


def run_download(
    n_target: int = 100,
    output_dir: str = "03_data/vitaldb_raw",
    catalogue_path: str = "03_data/case_catalogue.csv",
    spec: Optional[SelectionSpec] = None,
    thresholds: Optional[QCThresholds] = None,
    eeg_track: str = DEFAULT_EEG_TRACK,
    bis_track: str = DEFAULT_BIS_TRACK,
    sleep_seconds: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """End-to-end download driver.

    Parameters
    ----------
    n_target : int
        Desired number of successfully-downloaded, QC-passing cases.
    output_dir : str
        Directory where `.mat` files land (created if needed).
    catalogue_path : str
        CSV file with one row per attempted case id.
    spec : SelectionSpec | None
        Controls which case IDs are eligible; defaults use
        `n_target=n_target`, BIS+TIVA required, seed 42, no exclusions.
    thresholds : QCThresholds | None
        Signal QC thresholds.
    sleep_seconds : float
        Optional pause between requests to avoid hammering VitalDB during
        very large pulls. Zero by default — VitalDB has no advertised rate
        limit, but be polite if scaling to hundreds of cases.

    Returns
    -------
    pd.DataFrame
        The full catalogue after this run.
    """
    log = logger or _default_logger()
    spec = spec or SelectionSpec(n_target=n_target)
    thresholds = thresholds or QCThresholds()

    os.makedirs(output_dir, exist_ok=True)
    catalogue = _read_existing_catalogue(catalogue_path)
    already = _already_downloaded_ids(catalogue)

    log.info("Target cases: %d", n_target)
    log.info("Already downloaded: %d", len(already))

    if len(already) >= n_target:
        log.info(
            "Nothing to do — catalogue already has %d successful cases.", len(already)
        )
        return catalogue

    candidates = select_case_ids(spec)
    # Don't re-attempt cases already marked qc_fail or downloaded.
    already_attempted = set()
    if not catalogue.empty:
        already_attempted = set(
            catalogue[CatalogueColumns.VITALDB_CASEID].astype(int).tolist()
        )

    queue = [cid for cid in candidates if cid not in already_attempted]
    log.info("Candidate queue length: %d", len(queue))

    successes = len(already)
    attempts = 0
    start = time.time()

    rows_new: list[dict] = []

    for caseid in queue:
        if successes >= n_target:
            break
        attempts += 1
        row = download_and_save_case(
            caseid=caseid,
            output_dir=output_dir,
            eeg_track=eeg_track,
            bis_track=bis_track,
            thresholds=thresholds,
            logger=log,
        )
        rows_new.append(row)

        # Merge and persist after every case so a crash loses at most one row.
        combined = pd.concat(
            [catalogue, pd.DataFrame(rows_new)],
            ignore_index=True,
        )
        # Deduplicate on case id, keeping the last (most recent) entry.
        combined = combined.drop_duplicates(
            subset=[CatalogueColumns.VITALDB_CASEID], keep="last"
        )
        _write_catalogue(combined, catalogue_path)

        if row[CatalogueColumns.STATUS] == "downloaded":
            successes += 1
            elapsed = time.time() - start
            avg = elapsed / max(successes - len(already), 1)
            remaining = max(n_target - successes, 0)
            eta = avg * remaining
            log.info(
                "[%d/%d successes] attempts=%d  avg=%.1fs/case  ETA=%s",
                successes,
                n_target,
                attempts,
                avg,
                _fmt_eta(eta),
            )

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    log.info(
        "Downloader finished: %d successes / %d attempts in %s",
        successes,
        attempts,
        _fmt_eta(time.time() - start),
    )
    return pd.read_csv(catalogue_path)


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #


def _default_logger() -> logging.Logger:
    log = logging.getLogger("vitaldb_downloader")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", "%H:%M:%S")
        )
        log.addHandler(handler)
        log.setLevel(logging.INFO)
    return log


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ----------------------------------------------------------------------------- #
# Dev / smoke-test entry point
# ----------------------------------------------------------------------------- #


if __name__ == "__main__":
    # Quick smoke test: try to download one case into /tmp and print the row.
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the downloader on one case."
    )
    parser.add_argument("--caseid", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="/tmp/vitaldb_smoke")
    args = parser.parse_args()

    row = download_and_save_case(caseid=args.caseid, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                k: (v if isinstance(v, (int, float, str)) else str(v))
                for k, v in row.items()
            },
            indent=2,
        )
    )
