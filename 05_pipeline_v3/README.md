# Pipeline v3 — Scaffold

Empty skeleton for the v3 journal-paper pipeline. Every subfolder maps to a
stage in the architecture diagram of `../01_docs/DoA_Research_Roadmap_v3.docx`.
Nothing here is implemented yet — Week 1 of the roadmap fills in `data/`,
`preprocessing/`, and `features/`.

## Layout

| Folder | What goes here | Roadmap section |
|---|---|---|
| `config/` | Hydra / YAML configs for data, preprocessing, features, models, training. One file per concern. | §5, §8.2 |
| `data/` | VitalDB downloader, case catalogue, train/val/test split manifests, LOPO fold definitions. | §7.1, §13 |
| `preprocessing/` | Notch + bandpass, wICA artifact removal, NA-MEMD decomposition, QC gating. | §5.2 |
| `features/` | 45-D multi-view extractor: entropy (SampEn/ApEn/PE/MSE/SpEn), spectral (bands, SEF, BSR), wavelet (db4 × 5), Hjorth. Windowing aligned to BIS also lives here. | §5.3–§5.4 |
| `models/` | `heednet.py` (1D-ResNet + BiLSTM + cross-attention), `dann.py` (gradient-reversal head), baseline wrappers (SVR/RF/XGB/ANN). | §5.5 |
| `training/` | Training driver, multi-task loss (regression + 4-state classification), domain-adversarial schedule, checkpointing. | §7.2 |
| `validation/` | Nested LOPO-CV, inner 5-fold HP selection, bootstrap BCa CIs, paired Wilcoxon + BH-FDR, Bland–Altman, ICC(3,1). | §6 |
| `smoothing/` | Post-hoc temporal smoothers over per-window BIS estimates (Kalman filter, HMM). | §5.6 |
| `interpret/` | SHAP on features, Grad-CAM on 1D-ResNet, attention-weight export and visualisation. | §9 |
| `results/` | Experiment outputs: run JSONs, metrics CSVs, figures, LaTeX tables. Git-ignored bulk; commit only summary artefacts. | §7 |
| `paper/` | IEEE Transactions manuscript sources (LaTeX), supplementary materials, rebuttal drafts. | §11 |
| `scripts/` | One-off utilities: dataset stats, figure rendering, LaTeX table emission, reproducibility checks. | — |
| `tests/` | Unit + integration tests (`pytest`). Features, preprocessing, and LOPO splitter must be covered before modelling starts. | — |

## How v3 differs from v2

1. **Dataset** — ≥100 VitalDB cases (v2: 24).
2. **Features** — 45-D multi-view vector (v2: 3 entropy features).
3. **Model** — HEED-Net dual-stream with cross-attention + DANN head (v2: classical SVR / RF / XGB / ANN).
4. **Validation** — Nested LOPO-CV with inner 5-fold HP selection (v2: plain LOPO, no inner selection).
5. **Reporting** — Bootstrap CIs, paired tests with FDR, Bland–Altman, ICC(3,1) (v2: point estimates only).

## Bringing the scaffold to life

Work top-down per the 12-week schedule in §7 of the roadmap:

- **Week 1** — `data/` downloader, fix preprocessing EMD issue, reproduce v2 LOPO on all 24 cases as a sanity check.
- **Weeks 2–3** — `preprocessing/` + `features/` at parity with literature; freeze the 45-D feature spec.
- **Weeks 4–6** — `models/heednet.py`, `training/` with multi-task + DANN.
- **Weeks 7–8** — `validation/` protocol, `smoothing/` post-processing, ablations.
- **Weeks 9–10** — `interpret/` figures, finalise result tables.
- **Weeks 11–12** — `paper/` draft, rebuttal-ready supplementary.

The v2 repo is frozen at `../04_pipeline_v2/` as the numerical baseline that v3 must beat (best LOPO RMSE 11.53 ± 2.13).
