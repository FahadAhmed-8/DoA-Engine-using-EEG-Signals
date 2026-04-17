# Depth-of-Anesthesia Estimation from EEG — Project Root

A two-phase research project on subject-independent Depth-of-Anesthesia (DoA)
estimation from single-channel EEG on the VitalDB cohort.

**Target venue:** IEEE Transactions (TBME / TNSRE / JBHI)
**Author:** Fahad Ahmed
**Current phase:** Transitioning from v2 (mini-project, done) to v3 (journal paper, 12 weeks)

---

## Folder map

| Folder | Purpose |
|---|---|
| `01_docs/` | Planning documents. Current roadmap lives here. Older plans are in `archive/`. |
| `02_literature/` | Reference papers, organised by topic (baseline, deep_learning, nirs, reference). |
| `03_data/vitaldb_raw/` | Raw EEG + BIS recordings. 24 cases at the moment; v3 Week 1 expands this to ≥100. |
| `04_pipeline_v2/` | The mini-project codebase. Frozen. Git repository intact. Reproduces the v2 results: baseline RMSE 10.65, LOPO RMSE 11.53 ± 2.13. |
| `05_pipeline_v3/` | The new pipeline scaffold. Empty skeleton matching the roadmap's module layout. Week 1 of the v3 plan starts here. |

## Where to start

1. Read `01_docs/DoA_Research_Roadmap_v3.docx` end to end — it is the single source of truth for the plan, the architecture, and the 12-week schedule.
2. Section 13 of the roadmap is a 48-hour unblock checklist for Week 1.
3. The v2 results that v3 must beat are summarised in Section 2 of the roadmap and reproducible from `04_pipeline_v2/`.

## Reproducing v2 (sanity check)

```bash
cd 04_pipeline_v2
pip install -r requirements.txt       # if not yet installed
python scripts/main_pipeline.py       # runs inspect → preprocess → features → baseline → LOPO → results
```

Configuration automatically resolves `03_data/vitaldb_raw/` after the v3 reorg, with the old `Dataset/` path kept as a fallback.

## Headline v2 results (from `04_pipeline_v2/outputs/results/`)

| Metric | Value | Comparison |
|---|---|---|
| Best baseline RMSE (80/20) | 10.65 | Beats Rani 2020 target of 11.73 |
| Best baseline MAE | 6.97 | — |
| Best baseline Pearson r | 0.777 | — |
| Best LOPO RMSE | 11.53 ± 2.13 | Cross-patient validated |
| Best LOPO Pearson r | 0.700 | — |
