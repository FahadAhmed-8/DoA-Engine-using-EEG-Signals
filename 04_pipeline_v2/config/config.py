"""
Centralized configuration for the DoA EEG Pipeline.
All hyperparameters, paths, and constants in one place.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
# BASE_DIR = .../Mini Project 2/04_pipeline_v2
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Cohort size is parameterised via the V2_NUM_CASES env var so v2 can be
# rerun on different cohorts (e.g. the 100-case VitalDB expansion) without
# any change to algorithms or hyperparameters. Defaults to the original 24.
_N_env = os.environ.get("V2_NUM_CASES")

if _N_env is not None:
    # ---- Cohort-specific mode (e.g. V2_NUM_CASES=100) ---------------------
    # Data lives in 04_pipeline_v2/dataset_n<N>/ (prepared by
    # scripts/09_prepare_n100_dataset.py), outputs go to outputs_n<N>/ to
    # preserve the original n=24 runs.
    NUM_CASES = int(_N_env)
    DATASET_DIR = os.path.join(BASE_DIR, f"dataset_n{NUM_CASES}")
    OUTPUT_DIR = os.path.join(BASE_DIR, f"outputs_n{NUM_CASES}")
else:
    # ---- Default mode (original n=24) -------------------------------------
    # The old "<parent>/Dataset" path is kept for reproducibility of the
    # original runs; 03_data/vitaldb_raw is available as a fallback for
    # tooling that expects the v3 data layout.
    NUM_CASES = 24
    _NEW_DATASET = os.path.join(_PROJECT_ROOT, "03_data", "vitaldb_raw")
    _OLD_DATASET = os.path.join(_PROJECT_ROOT, "Dataset")
    DATASET_DIR = _OLD_DATASET if os.path.isdir(_OLD_DATASET) else _NEW_DATASET
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocessing")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# ── Dataset ────────────────────────────────────────────────────────
SAMPLING_RATE = 128          # Hz
BIS_INTERVAL = 5             # seconds
WINDOW_SIZE_SEC = 5          # seconds
SAMPLES_PER_WINDOW = SAMPLING_RATE * WINDOW_SIZE_SEC  # 640

# ── Preprocessing ──────────────────────────────────────────────────
NOTCH_FREQ = 60.0            # Hz (power line interference)
NOTCH_Q = 30.0               # Quality factor for notch filter
BUTTERWORTH_ORDER = 4
ARTIFACT_THRESHOLD_STD = 5   # Samples > this × std are artifacts
ARTIFACT_FILL_WINDOW = 1000  # Surrounding samples for artifact replacement
BIS_MISSING_VALUE = -1       # BIS missing marker in dataset
BIS_FILL_WINDOW = 10         # Surrounding BIS values for gap filling
BANDPASS_LOW = 0.5           # Hz (EMD fallback)
BANDPASS_HIGH = 30.0         # Hz (EMD fallback)
IMF_INDICES = [1, 2]         # IMF2 + IMF3 (0-indexed)

# ── Feature Extraction ─────────────────────────────────────────────
SAMPLE_ENTROPY_ORDER = 2
SAMPLE_ENTROPY_R_RATIO = 0.2  # r = 0.2 × std(signal)
APP_ENTROPY_ORDER = 2
PERM_ENTROPY_ORDER = 3
PERM_ENTROPY_NORMALIZE = True

# All 7 feature combinations
FEATURE_COMBOS = {
    'SampEn':       ['SampEn'],
    'ApEn':         ['ApEn'],
    'PE':           ['PE'],
    'SampEn_ApEn':  ['SampEn', 'ApEn'],
    'SampEn_PE':    ['SampEn', 'PE'],
    'ApEn_PE':      ['ApEn', 'PE'],
    'All':          ['SampEn', 'ApEn', 'PE'],
}

# ── Model Hyperparameters ──────────────────────────────────────────
ANN_CONFIG = {
    'hidden_layer_sizes': (10, 8, 9),
    'max_iter': 1000,
    'activation': 'relu',
    'solver': 'adam',
    'early_stopping': True,
    'validation_fraction': 0.1,
}

RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
}

XGB_CONFIG = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
}

SVR_CONFIG = {
    'kernel': 'rbf',
    'C': 100.0,
    'gamma': 'scale',
    'epsilon': 0.1,
}

# ── Validation ─────────────────────────────────────────────────────
TRAIN_TEST_RATIO = 0.8
RANDOM_SEED = 42

# ── Targets (from Rani & Maheshwari 2020) ──────────────────────────
TARGET_RMSE = 11.73
TARGET_MAE = 5.75
