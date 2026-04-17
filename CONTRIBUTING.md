# Contributing

This is a single-author research repository. The guidance below is mostly for
future-me (Fahad) when picking the project back up after time away, and for
reviewers who want to reproduce results.

## Project layout

```
Mini Project 2/
├── 01_docs/                    Roadmap + archived plans
├── 02_literature/              Reference papers (baseline, deep_learning, nirs)
├── 03_data/vitaldb_raw/        Raw EEG + BIS .mat files (gitignored)
├── 03_data/case_catalogue.csv  Per-case metadata (tracked)
├── 04_pipeline_v2/             Frozen mini-project code — do not modify
└── 05_pipeline_v3/             Journal paper pipeline — active development
```

## Working rules

1. **Never modify 04_pipeline_v2/.** It is the numerical baseline that v3 must
   beat. If you need to reproduce its numbers years from now, those files
   should be byte-identical to the state that produced the v2 result table.
2. **v3 work happens in 05_pipeline_v3/.** Every subfolder maps to a stage in
   the roadmap's architecture diagram; see `05_pipeline_v3/README.md`.
3. **Commit data manifests, not data.** The catalogue CSV is tracked. Raw
   `.mat` files are gitignored — anyone cloning the repo re-runs the
   downloader to populate `03_data/vitaldb_raw/`.
4. **One concern per commit.** Reorgs, scaffold additions, feature code, and
   results updates go in separate commits so `git bisect` stays useful.
5. **Tests before model changes.** Unit tests for the feature extractor and
   LOPO splitter land before HEED-Net so refactors don't silently change the
   numbers.

## Local setup

```bash
# One-time
python -m venv .venv
. .venv/bin/activate                # macOS/Linux
# .venv\Scripts\activate            # Windows PowerShell

# Minimum: one of the two pipelines
pip install -r 04_pipeline_v2/requirements.txt
pip install -r 05_pipeline_v3/requirements.txt

# Dev tooling (optional)
pip install ruff black mypy pytest pytest-cov
```

## Reproducing v2 (sanity check)

```bash
cd 04_pipeline_v2
python scripts/main_pipeline.py
```

This runs inspect → preprocess → features → baseline → LOPO → results end to
end. Expected best baseline RMSE 10.65, best LOPO RMSE 11.53 ± 2.13 (once the
`08_complete_missing_lopo.py` fill-in run has been executed).

## v3 workflow

Work top-down through the 12-week schedule in
`01_docs/DoA_Research_Roadmap_v3.docx`. The `05_pipeline_v3/README.md` table
shows which Week fills in which subfolder.

## Commit message style

- `chore: reorganize folder layout for v3`
- `feat(v3/data): VitalDB downloader with QC catalogue`
- `fix(v2/scripts): complete missing LOPO configs with resumable save`
- `docs: Week 1 report`

## Before pushing

```bash
# Quick static checks (if tools installed)
ruff check .
black --check .

# Run fast unit tests
pytest 05_pipeline_v3/tests -q
```

The CI workflow in `.github/workflows/ci.yml` runs the same checks on push
and on pull requests into `main`.
