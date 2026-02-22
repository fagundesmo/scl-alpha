"""
Model training and prediction module.

Provides a uniform interface for three regressors:
    1. Ridge Regression
    2. Random Forest
    3. XGBoost

All models are wrapped so they can be swapped with a single string argument.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.config import (
    RIDGE_ALPHA,
    RF_PARAMS,
    XGB_PARAMS,
    MODELS_DIR,
    RANDOM_SEED,
)
from src.features import FEATURE_COLUMNS  # single source of truth


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(name: str, params: dict | None = None) -> Pipeline:
    """
    Create a sklearn Pipeline with StandardScaler + regressor.

    Parameters
    ----------
    name : str
        One of 'ridge', 'rf', 'xgboost'.
    params : dict | None
        Optional model-specific parameter overrides.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready to .fit(X, y).

    Notes
    -----
    The scaler is fit INSIDE the pipeline, which means it only ever sees
    training data, preventing data leakage.
    """
    name = name.lower().strip()
    params = params or {}

    if name == "ridge":
        ridge_params = {"alpha": RIDGE_ALPHA, "random_state": RANDOM_SEED}
        ridge_params.update(params)
        reg = Ridge(**ridge_params)

    elif name == "rf":
        rf_params = dict(RF_PARAMS)
        rf_params.update(params)
        reg = RandomForestRegressor(**rf_params)

    elif name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for model_name='xgboost'. It is included in the default "
                "project dependencies; reinstall with `pip install .` or `pip install -r requirements.txt`."
            ) from exc

        xgb_params = dict(XGB_PARAMS)
        xgb_params.update(params)
        xgb_params.pop("early_stopping_rounds", None)
        reg = XGBRegressor(**xgb_params)

    else:
        raise ValueError(f"Unknown model name: {name!r}.  Choose 'ridge', 'rf', or 'xgboost'.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", reg),
    ])


# ---------------------------------------------------------------------------
# Train / Predict helpers
# ---------------------------------------------------------------------------

def train_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> Pipeline:
    """
    Fit the model pipeline on training data.

    If the regressor is XGBoost and validation data is provided,
    early stopping is used automatically.
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in X_train.columns]
    X = X_train[feature_cols].values
    y = y_train.values

    regressor_name = type(model.named_steps["regressor"]).__name__

    if regressor_name == "XGBRegressor" and X_val is not None and y_val is not None:
        scaler = model.named_steps["scaler"]
        X_scaled = scaler.fit_transform(X)
        X_val_scaled = scaler.transform(X_val[feature_cols].values)

        model.named_steps["regressor"].fit(
            X_scaled,
            y,
            eval_set=[(X_val_scaled, y_val.values)],
            verbose=False,
        )
    else:
        model.fit(X, y)

    return model


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions from a fitted pipeline.

    Returns
    -------
    np.ndarray
        Predicted 5-day forward returns (%).
    """
    feature_cols = [c for c in FEATURE_COLUMNS if c in X.columns]
    return model.predict(X[feature_cols].values)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model: Pipeline, feature_names: list[str] | None = None) -> pd.Series:
    """
    Extract feature importances / coefficients from the fitted model.

    Returns a pd.Series sorted by absolute value (descending).
    """
    reg = model.named_steps["regressor"]
    feature_names = feature_names or FEATURE_COLUMNS

    if hasattr(reg, "feature_importances_"):
        imp = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        imp = reg.coef_
    else:
        return pd.Series(dtype=float)

    s = pd.Series(imp, index=feature_names[: len(imp)], name="importance")
    return s.reindex(s.abs().sort_values(ascending=False).index)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: Pipeline, name: str, suffix: str = "") -> Path:
    """Save a fitted model to disk."""
    fname = f"{name}{suffix}.joblib"
    path = MODELS_DIR / fname
    joblib.dump(model, path)
    print(f"[model] Saved -> {path}")
    return path


def load_model(name: str, suffix: str = "") -> Pipeline:
    """Load a previously saved model."""
    fname = f"{name}{suffix}.joblib"
    path = MODELS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)
