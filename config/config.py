"""
Centralized configuration for the DoA EEG Pipeline.
All hyperparameters, paths, and constants in one place.
"""
import os

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(os.path.dirname(BASE_DIR), "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

PREPROCESS_DIR = os.path.join(OUTPUT_DIR, "preprocessing")
FEATURES_DIR = os.path.join(OUTPUT_DIR, "features")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# ── Dataset ────────────────────────────────────────────────────────
NUM_CASES = 24
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
