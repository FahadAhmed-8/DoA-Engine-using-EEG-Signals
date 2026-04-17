#!/usr/bin/env python3
"""
Master Pipeline: Orchestrate all 7 steps in sequence.
Run from the DoA_Pipeline directory.
"""
import subprocess
import sys
import os
import time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)

PIPELINE_STEPS = [
    ('01_inspect_data.py',      'Dataset Inspection'),
    ('02_preprocess.py',        'Signal Preprocessing'),
    ('03_extract_features.py',  'Feature Extraction'),
    ('04_train_baseline.py',    'Baseline Training (80/20)'),
    ('05_train_lopo.py',        'LOPO Cross-Validation'),
    ('06_channel_analysis.py',  'Multi-Channel Analysis'),
    ('07_generate_results.py',  'Results & Visualization'),
]


def main():
    print("=" * 60)
    print("DoA EEG Pipeline — Full Execution")
    print("=" * 60)

    total_start = time.time()
    step_times = {}

    for i, (script, description) in enumerate(PIPELINE_STEPS):
        print(f"\n{'#' * 60}")
        print(f"# Step {i+1}/{len(PIPELINE_STEPS)}: {description}")
        print(f"# Script: {script}")
        print(f"{'#' * 60}\n")

        script_path = os.path.join(SCRIPTS_DIR, script)
        step_start = time.time()

        result = subprocess.run(
            [sys.executable, script_path],
            cwd=PROJECT_DIR
        )

        step_elapsed = time.time() - step_start
        step_times[description] = step_elapsed

        if result.returncode != 0:
            print(f"\nERROR in Step {i+1}: {script}")
            print("Pipeline stopped.")
            break

        print(f"\n[Step {i+1} completed in {step_elapsed:.1f}s]")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"Pipeline Execution Summary")
    print(f"{'=' * 60}")
    for desc, t in step_times.items():
        print(f"  {desc}: {t:.1f}s")
    print(f"  {'─' * 40}")
    print(f"  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
