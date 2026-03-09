"""
Unified model factory: Create, train, and predict with all 4 ML models.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import xgboost as xgb

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import ANN_CONFIG, RF_CONFIG, XGB_CONFIG, SVR_CONFIG, RANDOM_SEED


MODEL_CONFIGS = {
    'ANN': (MLPRegressor, {**ANN_CONFIG, 'random_state': RANDOM_SEED}),
    'RF':  (RandomForestRegressor, {**RF_CONFIG, 'random_state': RANDOM_SEED, 'n_jobs': -1}),
    'XGB': (xgb.XGBRegressor, {**XGB_CONFIG, 'random_state': RANDOM_SEED, 'n_jobs': -1,
                                 'verbosity': 0}),
    'SVR': (SVR, SVR_CONFIG),
}


def create_model(model_name: str):
    """Create a fresh model instance."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODEL_CONFIGS.keys())}")

    cls, params = MODEL_CONFIGS[model_name]
    return cls(**params)


def train_model(model, X_train: np.ndarray, y_train: np.ndarray):
    """Train model in-place, return the model."""
    model.fit(X_train, y_train)
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """Get predictions, clipped to valid BIS range [0, 100]."""
    preds = model.predict(X)
    return np.clip(preds, 0, 100)


def get_model_names() -> list:
    """Return list of all available model names."""
    return list(MODEL_CONFIGS.keys())
