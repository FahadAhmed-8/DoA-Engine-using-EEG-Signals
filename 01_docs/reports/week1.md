# Week 1 Report — v2 closeout + v3 foundations

**Period:** Week 1 of the 10-12 week IEEE Transactions push
**Prepared:** 17 April 2026
**Author:** Fahad Ahmed
**Repo:** [FahadAhmed-8/DoA-Engine-using-EEG-Signals](https://github.com/FahadAhmed-8/DoA-Engine-using-EEG-Signals)

---

## 1. Executive summary

Three deliverables landed this week. v2's LOPO evaluation is now complete across all 28 feature×model configurations, giving us a published-quality baseline to beat. The VitalDB cohort has been expanded from 24 to 100 cases with zero quality-control failures, ready for v3. The GitHub repository has been restructured to mirror the local `01_docs / 02_literature / 03_data / 04_pipeline_v2 / 05_pipeline_v3` layout, with MIT license, citation metadata, CI, issue/PR templates, and a v3 scaffold in place.

**Headline v2 LOPO result:** `All (SampEn+ApEn+PE) + SVR` → RMSE 11.47 ± 2.11, MAE 7.92 ± 1.64, r = 0.709 ± 0.10, across 24 leave-one-patient-out folds. This is the number v3's HEED-Net has to improve on.

---

## 2. v2 LOPO — final 7×4 grid (24 patients, LOPO)

All values are the mean ± std across 24 LOPO folds. Bold = best in column / overall.

| Feature combo | RF | XGB | ANN | SVR |
|---|---:|---:|---:|---:|
| SampEn       | 15.06 ± 3.28 | 14.63 ± 3.30 | 14.64 ± 3.27 | 15.32 ± 3.97 |
| ApEn         | 15.10 ± 3.09 | 14.69 ± 3.14 | 14.72 ± 3.14 | 15.31 ± 3.70 |
| PE           | 13.31 ± 2.24 | 12.86 ± 2.17 | 12.91 ± 2.19 | 12.97 ± 2.22 |
| SampEn+ApEn  | 14.62 ± 3.11 | 14.53 ± 3.20 | 14.68 ± 3.31 | 15.17 ± 3.86 |
| SampEn+PE    | 12.64 ± 2.26 | 12.66 ± 2.34 | 12.80 ± 2.35 | **12.58 ± 2.42** |
| ApEn+PE      | 12.92 ± 2.24 | 12.78 ± 2.20 | 12.66 ± 2.03 | 12.76 ± 2.29 |
| **All**      | 11.53 ± 2.13 | 11.63 ± 2.04 | 11.89 ± 2.04 | **11.47 ± 2.11** |

*All numbers are LOPO RMSE of BIS prediction. Full metrics (MAE, Pearson r) and per-fold breakdowns are in `04_pipeline_v2/outputs/results/lopo_results.csv` and `lopo_fold_details.csv`.*

### 2.1 What the numbers say

**Permutation Entropy is the workhorse.** Every feature combination *containing PE* lands in the RMSE 11.5–13.3 band. Every combination *without PE* lands in 14.5–15.3. That's a near-2-RMSE gap driven entirely by a single feature family. SampEn and ApEn individually — and even fused — don't break RMSE 14.5.

**Fusion helps, but with diminishing returns.** Going PE alone (12.86) → PE + one entropy (12.58–12.66) buys ~0.2 RMSE. Going two-entropy+PE → all-three (11.47) buys another ~1.1. That's a real gain, but we're in the regime where feature fusion has mostly saturated on this 24-case cohort. The ceiling for classical entropy-feature pipelines is approximately RMSE 11.5.

**Model choice is secondary once features are fixed.** Within the All-feature block, the spread from SVR to ANN is only 0.42 RMSE. Within PE-only, the spread is 0.45. The feature design contributes roughly 3× what the model choice contributes.

### 2.2 Per-fold variance

Standard deviations across LOPO folds are ~2.0 for PE-containing configs, rising to ~3.3 for SampEn/ApEn-only configs. Two things follow:

1. A handful of patients are systematically harder than the rest. (Per-fold inspection via `lopo_fold_details.csv` is a Week 2 action — it will tell us whether the variance is driven by signal quality, surgical phase distribution, or a real biological outlier.)
2. The 100-case rerun should shrink these stds meaningfully by washing out per-patient idiosyncrasies. This is the primary statistical reason to rerun v2 on the larger cohort before v3 comparisons.

### 2.3 Claims we can safely make in the paper

From this table alone, with appropriate caveats about cohort size:

- Classical entropy fusion with an RBF-SVR achieves subject-independent DoA estimation at RMSE 11.5 on the VitalDB cohort (LOPO, n=24).
- Permutation Entropy is the dominant single feature family for this task; its inclusion improves RMSE by ~2 over pipelines restricted to amplitude-complexity entropies.
- Within the tested classical model space, model choice explains <1 RMSE of variation; feature design explains ~3.

These framings are honest and they give v3 a clear numerical target.

---

## 3. VitalDB cohort expansion

### 3.1 Numbers

| Quantity | Value |
|---|---:|
| Target cases | 100 |
| Downloaded (post-QC) | 100 |
| Attempts required | 100 (no QC rejections) |
| Total EEG duration | 324.8 hours |
| Mean case duration | 3.25 hours |
| Download wall time | 3.9 minutes |
| Pool sampled from | `caseids_bis ∩ caseids_tiva` = 2,605 candidates |
| RNG seed | 42 (deterministic selection) |

### 3.2 Signal layout

- EEG track: `BIS/EEG1_WAV` @ 128 Hz, stored as `EEG` dataset in the `.mat` v7.3 (HDF5) file
- BIS track: `BIS/BIS` @ 0.2 Hz (one sample per 5 s), stored as `bis` dataset
- Schema-compatible with v2's loader (same keys, same shape convention)
- Amplitude range: ±1500 µV (raw); v2's pre-clipped ±62.5 µV range is **not** applied at download — clipping is deferred to the preprocessing layer so v3 can experiment with alternatives

### 3.3 Catalogue

`03_data/case_catalogue.csv` tracks every case with columns: `case_id, duration_min, eeg_samples, bis_samples, eeg_nan_pct, bis_nan_pct, eeg_std_uv, eeg_rail_clip_pct, qc_passed, qc_notes, downloaded_at, file_size_mb`. The catalogue is git-tracked; the `.mat` files themselves are gitignored under `03_data/vitaldb_raw/`.

### 3.4 What this unlocks

- A 100-case v2 rerun (Week 2, priority 1) that tightens LOPO confidence intervals and tells us whether the RMSE-11.5 ceiling holds at cohort scale.
- A subject-stratified split for v3: 70 train / 15 val / 15 test, held out at patient level, still with a comfortable fold budget.
- Enough statistical power (n=100) to credibly claim "subject-independent" in the paper.

---

## 4. Repository state

### 4.1 Layout on `origin/main`

```
DoA-Engine-using-EEG-Signals/
├── 01_docs/                    research roadmap, weekly reports
├── 02_literature/              baseline, DL, NIRS, reference PDFs
├── 03_data/                    case_catalogue.csv (raw .mat files gitignored)
├── 04_pipeline_v2/             frozen — DO NOT MODIFY
│   ├── config/ data/ features/ models/ validation/ analysis/ scripts/ tests/
│   ├── outputs/results/        baseline, LOPO, channel analyses
│   └── requirements.txt
├── 05_pipeline_v3/             in progress
│   ├── data/                   vitaldb_downloader.py (working)
│   ├── config/ preprocessing/ features/ models/ training/
│   ├── validation/ smoothing/ interpret/ results/ paper/ tests/
│   ├── scripts/download_vitaldb.py
│   └── requirements.txt
├── .github/                    CI, issue templates, PR template
├── .gitignore  CITATION.cff  CONTRIBUTING.md  LICENSE (MIT)  README.md
```

### 4.2 History preservation

The migration moved `.git` up from `04_pipeline_v2/` to the project root and let git's rename detector catch the 36 path changes at commit time. `git log --follow -- 04_pipeline_v2/<any_v2_file>` traces back through the rename to the original `Initial commit` (`fa2891f`). No history was rewritten; the push was a clean fast-forward.

Commits on `main` after Week 1:

1. `fa2891f` Initial commit (original v2 flat layout)
2. `56e5fdc` Remove outputs from tracking, add to gitignore
3. `f058a30` Remove outputs/preprocessing from tracking
4. `07e1bb5` Add output subfolders
5. `247c9b2` v2 LOPO completion (best: All+SVR, RMSE 11.47)
6. `d98fd8d` Reorganize to v2/v3 layout
7. *(pending)* Delete MIGRATION.md

### 4.3 CI

`.github/workflows/ci.yml` defines three jobs on every push and PR to `main`:

- **lint** — `ruff` + `black --check` across both pipelines
- **tests-v3** — pytest against `05_pipeline_v3/tests/` (currently no-op, will activate as tests land)
- **downloader-smoke** — runs `download_vitaldb.py --dry-run --n 3` to confirm case selection logic doesn't regress

### 4.4 Reproduction entry points

| Task | Command |
|---|---|
| Reproduce v2 LOPO table | `cd 04_pipeline_v2 && python scripts/05_train_lopo.py` |
| Complete any missing LOPO configs | `cd 04_pipeline_v2 && python scripts/08_complete_missing_lopo.py` |
| Download 100-case VitalDB cohort | `python 05_pipeline_v3/scripts/download_vitaldb.py --n 100 --output-dir 03_data/vitaldb_raw` |
| Install v2 deps | `pip install -r 04_pipeline_v2/requirements.txt` |
| Install v3 deps (incl. torch) | `pip install -r 05_pipeline_v3/requirements.txt` |

---

## 5. Week 2 plan

Four priorities in decreasing order of criticality. Each has a concrete deliverable.

### P1 — v2 rerun on the 100-case cohort

**Why first:** without a 100-case v2 number, every v3 result is confounded — we won't know if improvements come from the architecture or from more training data. This must be the apples-to-apples baseline.

**Deliverable:** `04_pipeline_v2/outputs/results/lopo_results_n100.csv` with the same 7×4 grid, LOPO over 100 folds.

**Effort estimate:** ~4× the 24-case wall time extrapolated from this week's run, i.e. 5-6 hours overnight. The existing resumable `08_complete_missing_lopo.py` script handles interruption gracefully, so it can be run in phases.

**Code changes needed:** point `04_pipeline_v2/config/config.py::DATASET_DIR` at `03_data/vitaldb_raw/` (already done) and update `CASE_IDS` to the 100-case list from `03_data/case_catalogue.csv`. I'll write the one-line config patch when we start P1.

### P2 — v3 preprocessing port

**Why second:** everything downstream depends on it. Need to decide whether v3 pre-processing reuses v2's loader/segmenter/preprocessor as-is, reuses with the amplitude-range adjustment (±1500 µV rather than ±62.5 µV), or takes a clean-slate approach for the dual-stream architecture.

**Deliverable:** `05_pipeline_v3/preprocessing/` with a `VitalDBCase` class, a segmenter that produces both (a) raw time-domain windows for the CNN stream and (b) pre-computed wavelet coefficients for the time-frequency stream, plus a QC-pass filter that honours `case_catalogue.csv`.

**Design question to resolve:** window length. v2 uses 10-second windows aligned with BIS updates. HEED-Net papers typically use 30-60 s. Suggest 30 s as a starting default, which is long enough for lower-frequency content and still produces hundreds of windows per case.

### P3 — v3 baseline CNN stub

**Why third:** we need *something* training end-to-end before the dual-stream complexity lands. A single-stream 1D-CNN on raw EEG, LOPO on the 100-case cohort, with the same metrics as v2. If it beats RMSE 11.5 without domain-adversarial loss, we know the deep-learning signal is real.

**Deliverable:** `05_pipeline_v3/models/heednet_timestream.py` + `05_pipeline_v3/training/train_timestream.py` + one numerical result.

**Risk:** overfitting on a 100-case cohort with a deep CNN is likely. Mitigations to put in place early: aggressive dropout, label smoothing, early stopping on held-out subjects, no data augmentation until we see the baseline failure mode.

### P4 — Minimum-viable tests

**Why fourth but important:** CI's `tests-v3` job is currently a no-op. Every v3 module should land with at least one unit test. Target: 5-10 tests covering VitalDB loader roundtrip, segmenter shape invariants, preprocessing determinism with fixed seed, and metric computation (RMSE, Pearson r) against hand-computed ground truth.

**Deliverable:** `05_pipeline_v3/tests/test_data.py`, `test_preprocessing.py`, `test_metrics.py`.

---

## 6. Open questions for the Week 2 planning conversation

1. Stick with 30-second windows for v3 or revisit with a small sensitivity study?
2. Do we commit the 100 `.mat` files anywhere (a private S3 bucket, a Zenodo release, or just the case-ID list + reproducibility script)? My default is the last option, which is what the current gitignore enforces.
3. HEED-Net reference implementation — is there one we can fork, or are we writing it from the paper?
4. Domain-adversarial head — gradient reversal layer on the second-to-last representation, trained with a patient-ID classifier? That's the cleanest formulation to cite.

---

## 7. Risks / watch-list

- **Cohort size ceiling.** 100 cases is comfortable for classical pipelines but tight for a dual-stream CNN with domain-adversarial training. If P3 overfits badly, scaling to 200-300 cases (another 8 minutes of download, same QC pipeline) is the first lever.
- **BIS as ground truth.** BIS itself has ~10% inter-subject disagreement with anaesthetist-rated depth. Any RMSE below ~8 starts hitting this noise floor. The paper should acknowledge this up-front.
- **VitalDB API stability.** The downloader's CI smoke test will catch outright API breakage, but silent schema drift (e.g. new optional fields) won't trip it. Re-verify track names if we expand the cohort substantially later.

---

## 8. Hand-off summary

v2 is frozen with a published-quality result. v3 has data, scaffolding, CI, licensing, and a four-item priority list for Week 2. Next pairing session starts at P1: patch the v2 config to the 100-case list, kick off the LOPO rerun, and while that runs, begin the v3 preprocessing port.
