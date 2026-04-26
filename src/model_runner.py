"""
Model training and evaluation for the Sandbox Models page.

Supports Ridge, Random Forest, and XGBoost. All models receive
StandardScaler-normalized inputs. The target is always TARGET_COL
(target_fwd_1d = next-day log return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import TARGET_COL
from src.metrics import compute_ml_metrics


# ============================================================
# Helpers
# ============================================================

def _feature_cols(df: pd.DataFrame) -> list[str]:
    """All numeric columns except metadata and the target."""
    exclude = {"Date", "Ticker", TARGET_COL, "index",
               "company_close", "sector_etf_close", "spy_close", "atr_14"}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def time_split(
    df: pd.DataFrame,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a features DataFrame into train / validation / test by date.

    The split is date-ordered: all rows for early dates go to train,
    middle dates to val, later dates to test. This prevents data leakage
    across the time series.
    """
    dates = pd.to_datetime(df["Date"]).sort_values().unique()
    n = len(dates)
    i_val = int(n * (1 - val_frac - test_frac))
    i_test = int(n * (1 - test_frac))

    d = pd.to_datetime(df["Date"])
    return (
        df[d.isin(dates[:i_val])].copy(),
        df[d.isin(dates[i_val:i_test])].copy(),
        df[d.isin(dates[i_test:])].copy(),
    )


# ============================================================
# Model runner
# ============================================================

def run_model(
    features: pd.DataFrame,
    model_name: str,
    model_params: dict,
    val_frac: float = 0.20,
    test_frac: float = 0.20,
) -> dict:
    """
    Train and evaluate one model on the feature matrix.

    Parameters
    ----------
    features : pd.DataFrame
        Output of build_feature_matrix (includes Date, Ticker, target_fwd_1d).
    model_name : str
        One of 'ridge', 'random_forest', 'xgboost'.
    model_params : dict
        Model hyperparameters (override defaults).
    val_frac, test_frac : float
        Fraction of unique dates allocated to validation and test.

    Returns
    -------
    dict with keys:
        model_name, model_params, feature_cols,
        train_metrics, val_metrics, test_metrics,
        feature_importance, test_predictions,
        train_size, val_size, test_size
    """
    cols = _feature_cols(features)
    train, val, test = time_split(features, val_frac, test_frac)

    # Impute with training-set medians (prevents leakage into val/test)
    medians = train[cols].median()

    X_train = train[cols].fillna(medians).values.astype(float)
    y_train = train[TARGET_COL].values.astype(float)
    X_val = val[cols].fillna(medians).values.astype(float)
    y_val = val[TARGET_COL].values.astype(float)
    X_test = test[cols].fillna(medians).values.astype(float)
    y_test = test[TARGET_COL].values.astype(float)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    if model_name == "ridge":
        reg = Ridge(alpha=float(model_params.get("alpha", 1.0)))
        reg.fit(X_train, y_train)

    elif model_name == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=int(model_params.get("n_estimators", 100)),
            max_depth=int(model_params.get("max_depth", 6)),
            min_samples_leaf=int(model_params.get("min_samples_leaf", 20)),
            random_state=42,
            n_jobs=-1,
        )
        reg.fit(X_train, y_train)

    elif model_name == "xgboost":
        from xgboost import XGBRegressor

        early_stop = int(model_params.get("early_stopping_rounds", 20))
        reg = XGBRegressor(
            n_estimators=int(model_params.get("n_estimators", 200)),
            max_depth=int(model_params.get("max_depth", 4)),
            learning_rate=float(model_params.get("learning_rate", 0.05)),
            subsample=float(model_params.get("subsample", 0.8)),
            colsample_bytree=float(model_params.get("colsample_bytree", 0.8)),
            reg_alpha=float(model_params.get("reg_alpha", 0.0)),
            reg_lambda=float(model_params.get("reg_lambda", 1.0)),
            objective=str(model_params.get("objective", "reg:squarederror")),
            random_state=42,
            early_stopping_rounds=early_stop if len(y_val) > 0 else None,
            verbosity=0,
        )
        if len(y_val) > 0:
            reg.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            reg.fit(X_train, y_train)

    else:
        raise ValueError(
            f"Unknown model_name {model_name!r}. "
            "Choose 'ridge', 'random_forest', or 'xgboost'."
        )

    def _metrics(X: np.ndarray, y: np.ndarray) -> dict:
        return compute_ml_metrics(y, reg.predict(X)) if len(y) > 0 else {}

    # Feature importance
    if hasattr(reg, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": cols, "importance": reg.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
    elif hasattr(reg, "coef_"):
        imp = pd.DataFrame(
            {"feature": cols, "importance": np.abs(reg.coef_)}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        imp = pd.DataFrame(columns=["feature", "importance"])

    # Test-set predictions
    test_preds = test[["Date", "Ticker"]].copy().reset_index(drop=True)
    test_preds["y_true"] = y_test
    test_preds["y_pred"] = reg.predict(X_test)
    test_preds["residual"] = test_preds["y_true"] - test_preds["y_pred"]

    return {
        "model_name": model_name,
        "model_params": model_params,
        "feature_cols": cols,
        "train_metrics": _metrics(X_train, y_train),
        "val_metrics": _metrics(X_val, y_val),
        "test_metrics": _metrics(X_test, y_test),
        "feature_importance": imp,
        "test_predictions": test_preds,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
