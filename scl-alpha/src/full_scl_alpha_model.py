"""Standardized reporting pipeline for the Full SCL-alpha model page.

This module focuses on beginner-friendly, comparable outputs across:
- Ridge Regression
- Random Forest Regressor
- XGBoost Regressor

The outputs are tabular pandas DataFrames so they are easy to inspect,
compare, and export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import inspect
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_TARGET_NAME = "predicted_return_5d"


def _to_frame(X: pd.DataFrame | np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Convert X into a DataFrame with deterministic feature ordering."""
    if isinstance(X, pd.DataFrame):
        out = X.copy()
        missing = [c for c in feature_names if c not in out.columns]
        if missing:
            raise ValueError(f"X is missing expected feature columns: {missing}")
        return out[feature_names]

    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features; got shape={arr.shape}")
    if arr.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch. X has {arr.shape[1]} columns; "
            f"feature_names has {len(feature_names)}."
        )
    return pd.DataFrame(arr, columns=feature_names)


def _to_series(y: pd.Series | np.ndarray, name: str = "target") -> pd.Series:
    """Convert y into a numeric Series."""
    if isinstance(y, pd.Series):
        out = y.copy()
        out.name = name
        return pd.to_numeric(out, errors="coerce")
    arr = np.asarray(y).reshape(-1)
    return pd.Series(arr, name=name, dtype=float)


def _coerce_metadata(
    metadata: pd.DataFrame | None,
    n_rows: int,
    split_name: str,
) -> pd.DataFrame:
    """Ensure metadata has required columns and expected length."""
    if metadata is None:
        return pd.DataFrame(
            {
                "date": pd.NaT,
                "ticker": "UNKNOWN",
            },
            index=np.arange(n_rows),
        )

    out = metadata.reset_index(drop=True).copy()
    if len(out) != n_rows:
        raise ValueError(
            f"{split_name} metadata rows ({len(out)}) do not match sample rows ({n_rows})."
        )
    if "date" not in out.columns:
        out["date"] = pd.NaT
    if "ticker" not in out.columns:
        out["ticker"] = "UNKNOWN"

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str)
    return out


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    """Correlation helper that handles constant/insufficient vectors."""
    valid = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(valid) < 2:
        return float("nan")
    if valid["a"].nunique() < 2 or valid["b"].nunique() < 2:
        return float("nan")
    return float(valid["a"].corr(valid["b"], method=method))


def _cross_sectional_ic(df: pd.DataFrame, method: str) -> float:
    """Mean cross-sectional IC by date; falls back to overall correlation."""
    required = {"date", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        return _safe_corr(df.get("y_true", pd.Series(dtype=float)), df.get("y_pred", pd.Series(dtype=float)), method)

    by_date = []
    for _dt, grp in df.groupby("date"):
        corr = _safe_corr(grp["y_true"], grp["y_pred"], method)
        if pd.notna(corr):
            by_date.append(corr)

    if by_date:
        return float(np.mean(by_date))

    return _safe_corr(df["y_true"], df["y_pred"], method)


def _date_range(meta: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return min/max date strings for summary tables."""
    if not isinstance(meta, pd.DataFrame) or meta.empty or "date" not in meta.columns:
        return None, None
    d = pd.to_datetime(meta["date"], errors="coerce").dropna()
    if d.empty:
        return None, None
    return d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d")


def _rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Small wrapper for RMSE."""
    valid = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if valid.empty:
        return float("nan")
    return float(np.sqrt(mean_squared_error(valid["y_true"], valid["y_pred"])))


def _directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of rows where prediction sign matches realized sign."""
    valid = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if valid.empty:
        return float("nan")
    return float((np.sign(valid["y_true"]) == np.sign(valid["y_pred"])).mean())


def _build_split_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Create a per-row prediction table used across metrics and diagnostics."""
    out = metadata.copy()
    out["y_true"] = pd.to_numeric(y_true, errors="coerce").values
    out["y_pred"] = pd.to_numeric(pd.Series(y_pred), errors="coerce").values
    return out


def fit_ridge_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    alpha: float = 1.0,
    fit_intercept: bool = True,
    max_iter: int = 1000,
    solver: str = "auto",
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit Ridge model and return model + split predictions + runtime."""
    x_tr = _to_frame(X_train, feature_names)
    x_va = _to_frame(X_val, feature_names)
    x_te = _to_frame(X_test, feature_names)
    y_tr = _to_series(y_train)

    start = time.perf_counter()
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                Ridge(
                    alpha=alpha,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    solver=solver,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(x_tr, y_tr)
    runtime = time.perf_counter() - start

    return {
        "model": model,
        "pred_train": model.predict(x_tr),
        "pred_val": model.predict(x_va),
        "pred_test": model.predict(x_te),
        "runtime_seconds": runtime,
    }


def fit_random_forest_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    n_estimators: int = 300,
    max_depth: int | None = 8,
    min_samples_split: int = 2,
    min_samples_leaf: int = 5,
    max_features: str | float | int = "sqrt",
    bootstrap: bool = True,
    oob_score_enabled: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Fit Random Forest model and return model + split predictions + runtime."""
    x_tr = _to_frame(X_train, feature_names)
    x_va = _to_frame(X_val, feature_names)
    x_te = _to_frame(X_test, feature_names)
    y_tr = _to_series(y_train)

    start = time.perf_counter()
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=bool(bootstrap),
        oob_score=bool(oob_score_enabled and bootstrap),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    model.fit(x_tr, y_tr)
    runtime = time.perf_counter() - start

    return {
        "model": model,
        "pred_train": model.predict(x_tr),
        "pred_val": model.predict(x_va),
        "pred_test": model.predict(x_te),
        "runtime_seconds": runtime,
    }


def fit_xgboost_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    objective: str = "reg:squarederror",
    random_state: int = 42,
    early_stopping_rounds: int | None = 30,
) -> dict[str, Any]:
    """Fit XGBoost model and return model + split predictions + runtime.

    Handles API differences across xgboost versions by only sending supported
    fit kwargs, and falling back to constructor-level params when needed.
    """
    try:
        from xgboost import XGBRegressor
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError(f"xgboost is unavailable in this environment: {exc}") from exc

    x_tr = _to_frame(X_train, feature_names)
    x_va = _to_frame(X_val, feature_names)
    x_te = _to_frame(X_test, feature_names)
    y_tr = _to_series(y_train)
    y_va = _to_series(y_val)

    start = time.perf_counter()
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        subsample=float(subsample),
        colsample_bytree=float(colsample_bytree),
        reg_alpha=float(reg_alpha),
        reg_lambda=float(reg_lambda),
        objective=objective,
        random_state=int(random_state),
    )

    fit_signature = inspect.signature(model.fit)
    fit_params = set(fit_signature.parameters.keys())
    has_val = len(x_va) > 0 and len(y_va) > 0

    fit_kwargs: dict[str, Any] = {}
    if "verbose" in fit_params:
        fit_kwargs["verbose"] = False

    if has_val and "eval_set" in fit_params:
        fit_kwargs["eval_set"] = [(x_tr, y_tr), (x_va, y_va)]

    if "eval_metric" in fit_params:
        fit_kwargs["eval_metric"] = "rmse"
    else:
        try:
            model.set_params(eval_metric="rmse")
        except Exception:
            pass

    if early_stopping_rounds is not None and has_val:
        if "early_stopping_rounds" in fit_params:
            fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        else:
            try:
                model.set_params(early_stopping_rounds=int(early_stopping_rounds))
            except Exception:
                pass

    try:
        model.fit(x_tr, y_tr, **fit_kwargs)
    except TypeError as exc:
        # Last-resort fallback for unexpected kwargs in mixed environments.
        msg = str(exc)
        fallback_kwargs = dict(fit_kwargs)
        for key in ("eval_metric", "early_stopping_rounds", "verbose"):
            if key in msg and key in fallback_kwargs:
                fallback_kwargs.pop(key, None)
        model.fit(x_tr, y_tr, **fallback_kwargs)

    runtime = time.perf_counter() - start

    evals_result = None
    if hasattr(model, "evals_result"):
        try:
            evals_result = model.evals_result()
        except Exception:
            evals_result = None

    return {
        "model": model,
        "pred_train": model.predict(x_tr),
        "pred_val": model.predict(x_va),
        "pred_test": model.predict(x_te),
        "runtime_seconds": runtime,
        "evals_result": evals_result,
    }

def compute_common_predictions_table(
    model_name: str,
    y_test: pd.Series | np.ndarray,
    y_pred_test: np.ndarray,
    test_metadata: pd.DataFrame,
    transaction_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Build standardized test predictions table for a model."""
    y_te = _to_series(y_test, name="y_true")
    meta = _coerce_metadata(test_metadata, len(y_te), "test")

    pred = meta.copy()
    pred["y_true"] = y_te.values
    pred["y_pred"] = pd.to_numeric(pd.Series(y_pred_test), errors="coerce").values
    pred["residual"] = pred["y_true"] - pred["y_pred"]
    pred["abs_error"] = pred["residual"].abs()
    pred["signal"] = np.where(pred["y_pred"] > 0, "LONG", "FLAT")

    if pred["date"].notna().any():
        pred["pred_rank_daily"] = pred.groupby("date")["y_pred"].rank(ascending=False, method="first")
        n_by_day = pred.groupby("date")["y_pred"].transform("count")
    else:
        pred["pred_rank_daily"] = pred["y_pred"].rank(ascending=False, method="first")
        n_by_day = pd.Series(len(pred), index=pred.index)

    denom = (n_by_day - 1).replace(0, np.nan)
    pred["confidence_score"] = 1.0 - ((pred["pred_rank_daily"] - 1.0) / denom)
    pred.loc[denom.isna(), "confidence_score"] = 1.0
    pred["confidence_score"] = pred["confidence_score"].clip(0, 1)

    if transaction_cost_bps > 0:
        pred["estimated_net_pred_after_cost"] = pred["y_pred"] - (transaction_cost_bps / 100.0)

    pred["model_name"] = model_name
    return pred[
        [
            "date",
            "ticker",
            "y_true",
            "y_pred",
            "residual",
            "abs_error",
            "signal",
            "pred_rank_daily",
            "confidence_score",
            *(["estimated_net_pred_after_cost"] if "estimated_net_pred_after_cost" in pred.columns else []),
            "model_name",
        ]
    ]


def compute_performance_metrics(
    model_name: str,
    split: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute standardized performance metrics for one split."""
    y_t = _to_series(y_true, name="y_true")
    y_p = _to_series(y_pred, name="y_pred")

    base = pd.DataFrame({"y_true": y_t, "y_pred": y_p}).dropna()
    if base.empty:
        return pd.DataFrame(
            [
                {
                    "model_name": model_name,
                    "split": split,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "directional_accuracy": np.nan,
                    "information_coefficient_pearson": np.nan,
                    "information_coefficient_spearman": np.nan,
                    "prediction_mean": np.nan,
                    "prediction_std": np.nan,
                    "actual_mean": np.nan,
                    "actual_std": np.nan,
                    "n_obs": 0,
                }
            ]
        )

    if metadata is None:
        meta = pd.DataFrame({"date": pd.NaT}, index=base.index)
    else:
        meta = _coerce_metadata(metadata, len(y_t), split).loc[base.index]

    ic_input = base.copy()
    ic_input["date"] = meta["date"].values

    row = {
        "model_name": model_name,
        "split": split,
        "mae": float(mean_absolute_error(base["y_true"], base["y_pred"])),
        "rmse": float(np.sqrt(mean_squared_error(base["y_true"], base["y_pred"]))),
        "r2": float(r2_score(base["y_true"], base["y_pred"])),
        "directional_accuracy": _directional_accuracy(base["y_true"], base["y_pred"]),
        "information_coefficient_pearson": _cross_sectional_ic(ic_input, method="pearson"),
        "information_coefficient_spearman": _cross_sectional_ic(ic_input, method="spearman"),
        "prediction_mean": float(base["y_pred"].mean()),
        "prediction_std": float(base["y_pred"].std(ddof=0)),
        "actual_mean": float(base["y_true"].mean()),
        "actual_std": float(base["y_true"].std(ddof=0)),
        "n_obs": int(len(base)),
    }
    return pd.DataFrame([row])


def compute_daily_signal_summary(predictions_test: pd.DataFrame) -> pd.DataFrame:
    """Daily summary of signal breadth and top predictions."""
    if not isinstance(predictions_test, pd.DataFrame) or predictions_test.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_universe",
                "n_long_signals",
                "avg_predicted_return",
                "top_ticker_1",
                "top_pred_1",
                "top_ticker_2",
                "top_pred_2",
                "top_ticker_3",
                "top_pred_3",
                "model_name",
            ]
        )

    rows: list[dict[str, Any]] = []
    mname = str(predictions_test["model_name"].iloc[0]) if "model_name" in predictions_test.columns else "unknown"
    for dt, grp in predictions_test.groupby("date", dropna=False):
        ranked = grp.sort_values("y_pred", ascending=False).reset_index(drop=True)
        row: dict[str, Any] = {
            "date": dt,
            "n_universe": int(len(ranked)),
            "n_long_signals": int((ranked["signal"] == "LONG").sum()),
            "avg_predicted_return": float(ranked["y_pred"].mean()) if len(ranked) else np.nan,
            "model_name": mname,
        }
        for i in (1, 2, 3):
            if len(ranked) >= i:
                row[f"top_ticker_{i}"] = str(ranked.loc[i - 1, "ticker"])
                row[f"top_pred_{i}"] = float(ranked.loc[i - 1, "y_pred"])
            else:
                row[f"top_ticker_{i}"] = None
                row[f"top_pred_{i}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def run_model_health_checks(
    model_name: str,
    predictions_test: pd.DataFrame,
    performance_metrics: pd.DataFrame,
    feature_count_expected: int,
    feature_count_actual: int,
    runtime_seconds: float,
    train_prediction_mean: float,
) -> pd.DataFrame:
    """Run beginner-friendly health checks and return PASS/WARN/FAIL table."""
    checks: list[dict[str, Any]] = []

    pred = predictions_test.copy() if isinstance(predictions_test, pd.DataFrame) else pd.DataFrame()
    perf = performance_metrics.copy() if isinstance(performance_metrics, pd.DataFrame) else pd.DataFrame()

    missing_preds = int(pred["y_pred"].isna().sum()) if "y_pred" in pred.columns else -1
    checks.append(
        {
            "model_name": model_name,
            "check_name": "missing_predictions",
            "check_status": "PASS" if missing_preds == 0 else "FAIL",
            "check_value": missing_preds,
            "threshold": "0",
            "comment": "Predictions should exist for every test row.",
        }
    )

    pred_std = float(pred["y_pred"].std(ddof=0)) if "y_pred" in pred.columns and len(pred) else np.nan
    checks.append(
        {
            "model_name": model_name,
            "check_name": "constant_predictions",
            "check_status": "PASS" if pd.notna(pred_std) and pred_std > 1e-8 else "FAIL",
            "check_value": pred_std,
            "threshold": "> 1e-8",
            "comment": "A constant prediction often means training/data issues.",
        }
    )

    outlier_ratio = np.nan
    if "y_pred" in pred.columns and pred["y_pred"].notna().sum() >= 5:
        z = (pred["y_pred"] - pred["y_pred"].mean()) / pred["y_pred"].std(ddof=0)
        outlier_ratio = float((z.abs() > 4).mean())
    checks.append(
        {
            "model_name": model_name,
            "check_name": "extreme_prediction_outliers",
            "check_status": "PASS" if pd.notna(outlier_ratio) and outlier_ratio <= 0.01 else "WARN",
            "check_value": outlier_ratio,
            "threshold": "<= 1% (|z| > 4)",
            "comment": "Very high outlier ratio can indicate unstable model behavior.",
        }
    )

    train_rmse = float(perf.loc[perf["split"] == "train", "rmse"].iloc[0]) if "split" in perf.columns and (perf["split"] == "train").any() else np.nan
    val_rmse = float(perf.loc[perf["split"] == "val", "rmse"].iloc[0]) if "split" in perf.columns and (perf["split"] == "val").any() else np.nan
    rmse_gap = val_rmse - train_rmse if pd.notna(train_rmse) and pd.notna(val_rmse) else np.nan
    checks.append(
        {
            "model_name": model_name,
            "check_name": "train_vs_val_rmse_gap",
            "check_status": "PASS" if pd.notna(rmse_gap) and rmse_gap <= max(0.50, 0.5 * max(train_rmse, 1e-9)) else "WARN",
            "check_value": rmse_gap,
            "threshold": "<= max(0.50, 50% of train RMSE)",
            "comment": "Large positive gap can suggest overfitting.",
        }
    )

    test_pred_mean = float(pred["y_pred"].mean()) if "y_pred" in pred.columns and len(pred) else np.nan
    drift = abs(test_pred_mean - train_prediction_mean) if pd.notna(test_pred_mean) and pd.notna(train_prediction_mean) else np.nan
    checks.append(
        {
            "model_name": model_name,
            "check_name": "prediction_drift_vs_train_mean",
            "check_status": "PASS" if pd.notna(drift) and drift <= 1.0 else "WARN",
            "check_value": drift,
            "threshold": "<= 1.0 return points",
            "comment": "Large drift can indicate data regime shift.",
        }
    )

    feature_ok = int(feature_count_expected) == int(feature_count_actual)
    checks.append(
        {
            "model_name": model_name,
            "check_name": "feature_count_match",
            "check_status": "PASS" if feature_ok else "FAIL",
            "check_value": f"expected={feature_count_expected}, actual={feature_count_actual}",
            "threshold": "exact match",
            "comment": "Feature mismatch can break comparability and reliability.",
        }
    )

    checks.append(
        {
            "model_name": model_name,
            "check_name": "runtime_seconds",
            "check_status": "PASS" if pd.notna(runtime_seconds) and runtime_seconds > 0 else "WARN",
            "check_value": float(runtime_seconds) if pd.notna(runtime_seconds) else np.nan,
            "threshold": "> 0",
            "comment": "Tracks operational performance of each run.",
        }
    )

    return pd.DataFrame(checks)


def extract_ridge_outputs(
    model_name: str,
    ridge_pipeline: Pipeline,
    feature_names: list[str],
    y_pred_test: np.ndarray,
) -> dict[str, pd.DataFrame]:
    """Extract Ridge-specific diagnostics and explainability tables."""
    reg: Ridge = ridge_pipeline.named_steps["model"]
    coef = pd.Series(reg.coef_, index=feature_names, name="coefficient")
    coef_df = (
        coef.rename_axis("feature")
        .reset_index()
        .assign(
            abs_coefficient=lambda d: d["coefficient"].abs(),
            sign=lambda d: np.where(d["coefficient"] > 0, "POS", np.where(d["coefficient"] < 0, "NEG", "ZERO")),
            standardized_input_assumed=True,
            model_name=model_name,
        )
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )

    hyper = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "alpha": reg.alpha,
                "fit_intercept": reg.fit_intercept,
                "max_iter": reg.max_iter,
                "solver": reg.solver,
                "random_state": reg.random_state,
            }
        ]
    )

    diag = pd.DataFrame(
        [
            {"model_name": model_name, "metric": "intercept", "value": float(reg.intercept_)},
            {"model_name": model_name, "metric": "coefficient_l2_norm", "value": float(np.linalg.norm(reg.coef_))},
            {
                "model_name": model_name,
                "metric": "number_nonzero_coefficients",
                "value": int((np.abs(reg.coef_) > 1e-12).sum()),
            },
            {"model_name": model_name, "metric": "prediction_range_min", "value": float(np.min(y_pred_test))},
            {"model_name": model_name, "metric": "prediction_range_max", "value": float(np.max(y_pred_test))},
        ]
    )

    return {
        "ridge_coefficients": coef_df,
        "ridge_hyperparams": hyper,
        "ridge_diagnostics": diag,
    }


def extract_rf_outputs(
    model_name: str,
    rf_model: RandomForestRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Extract Random Forest-specific diagnostics and explainability tables."""
    imp = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gini_or_mse_decrease": rf_model.feature_importances_,
        }
    )
    imp = imp.sort_values("importance_gini_or_mse_decrease", ascending=False).reset_index(drop=True)
    imp["importance_rank"] = np.arange(1, len(imp) + 1)
    imp["model_name"] = model_name

    perm_df = pd.DataFrame(
        columns=["feature", "perm_importance_mean", "perm_importance_std", "importance_rank", "model_name"]
    )
    try:
        if len(X_val) > 0 and len(y_val) > 0:
            perm = permutation_importance(
                rf_model,
                X_val,
                y_val,
                n_repeats=8,
                random_state=42,
                scoring="neg_mean_squared_error",
                n_jobs=-1,
            )
            perm_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "perm_importance_mean": perm.importances_mean,
                    "perm_importance_std": perm.importances_std,
                }
            ).sort_values("perm_importance_mean", ascending=False)
            perm_df = perm_df.reset_index(drop=True)
            perm_df["importance_rank"] = np.arange(1, len(perm_df) + 1)
            perm_df["model_name"] = model_name
    except Exception:
        perm_df = pd.DataFrame(
            columns=["feature", "perm_importance_mean", "perm_importance_std", "importance_rank", "model_name"]
        )

    hyper = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n_estimators": rf_model.n_estimators,
                "max_depth": rf_model.max_depth,
                "min_samples_split": rf_model.min_samples_split,
                "min_samples_leaf": rf_model.min_samples_leaf,
                "max_features": rf_model.max_features,
                "bootstrap": rf_model.bootstrap,
                "oob_score_enabled": rf_model.oob_score,
                "random_state": rf_model.random_state,
            }
        ]
    )

    depths = [est.tree_.max_depth for est in rf_model.estimators_] if hasattr(rf_model, "estimators_") else []
    train_rmse = _rmse(y_train, pd.Series(y_pred_train))
    test_rmse = _rmse(y_test, pd.Series(y_pred_test))

    diag = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "metric": "oob_score",
                "value": float(getattr(rf_model, "oob_score_", np.nan)),
            },
            {
                "model_name": model_name,
                "metric": "avg_tree_depth",
                "value": float(np.mean(depths)) if depths else np.nan,
            },
            {
                "model_name": model_name,
                "metric": "max_tree_depth_observed",
                "value": float(np.max(depths)) if depths else np.nan,
            },
            {
                "model_name": model_name,
                "metric": "prediction_variance",
                "value": float(np.var(y_pred_test)),
            },
            {
                "model_name": model_name,
                "metric": "train_test_rmse_gap",
                "value": float(test_rmse - train_rmse) if pd.notna(train_rmse) and pd.notna(test_rmse) else np.nan,
            },
        ]
    )

    return {
        "rf_feature_importance": imp,
        "rf_permutation_importance": perm_df,
        "rf_hyperparams": hyper,
        "rf_diagnostics": diag,
    }


def _xgb_importance_table(
    model_name: str,
    booster: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Build unified XGBoost feature importance table across types."""
    rows: list[dict[str, Any]] = []
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)}

    for imp_type in ("gain", "weight", "cover"):
        score_map = booster.get_score(importance_type=imp_type)
        typed = []
        for raw_key, value in score_map.items():
            feat = fmap.get(raw_key, raw_key)
            typed.append((feat, float(value)))

        typed_df = pd.DataFrame(typed, columns=["feature", "importance_value"]) if typed else pd.DataFrame(columns=["feature", "importance_value"])
        if typed_df.empty:
            continue
        typed_df = typed_df.sort_values("importance_value", ascending=False).reset_index(drop=True)
        typed_df["importance_rank_within_type"] = np.arange(1, len(typed_df) + 1)
        typed_df["importance_type"] = imp_type
        typed_df["model_name"] = model_name
        rows.extend(typed_df.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(columns=["feature", "importance_type", "importance_value", "importance_rank_within_type", "model_name"])
    return pd.DataFrame(rows)


def _xgb_shap_summary(
    model_name: str,
    xgb_model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    max_samples: int = 400,
) -> tuple[pd.DataFrame, str]:
    """Compute SHAP summary if shap is installed; otherwise return empty table."""
    try:
        import shap  # type: ignore
    except Exception:
        return (
            pd.DataFrame(columns=["feature", "mean_abs_shap", "shap_rank", "model_name"]),
            "shap not installed",
        )

    try:
        sample = X_test.copy()
        if len(sample) > max_samples:
            sample = sample.sample(max_samples, random_state=42)

        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        out = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        out["shap_rank"] = np.arange(1, len(out) + 1)
        out["model_name"] = model_name
        return out, "computed"
    except Exception as exc:
        return (
            pd.DataFrame(columns=["feature", "mean_abs_shap", "shap_rank", "model_name"]),
            f"shap failed: {exc}",
        )


def _xgb_training_history_table(model_name: str, evals_result: dict[str, Any] | None) -> pd.DataFrame:
    """Normalize xgboost eval history into a standardized table."""
    if not evals_result:
        return pd.DataFrame(
            columns=[
                "iteration",
                "train_metric_name",
                "train_metric_value",
                "val_metric_name",
                "val_metric_value",
                "model_name",
            ]
        )

    train_key = "validation_0" if "validation_0" in evals_result else None
    val_key = "validation_1" if "validation_1" in evals_result else None

    if train_key is None:
        return pd.DataFrame(
            columns=[
                "iteration",
                "train_metric_name",
                "train_metric_value",
                "val_metric_name",
                "val_metric_value",
                "model_name",
            ]
        )

    train_metrics = evals_result.get(train_key, {})
    val_metrics = evals_result.get(val_key, {}) if val_key else {}
    metric_name = next(iter(train_metrics.keys()), None)
    if metric_name is None:
        return pd.DataFrame(
            columns=[
                "iteration",
                "train_metric_name",
                "train_metric_value",
                "val_metric_name",
                "val_metric_value",
                "model_name",
            ]
        )

    train_vals = train_metrics.get(metric_name, [])
    val_name = next(iter(val_metrics.keys()), None)
    val_vals = val_metrics.get(val_name, []) if val_name else []

    rows = []
    for i in range(len(train_vals)):
        rows.append(
            {
                "iteration": i,
                "train_metric_name": metric_name,
                "train_metric_value": float(train_vals[i]),
                "val_metric_name": val_name,
                "val_metric_value": float(val_vals[i]) if i < len(val_vals) else np.nan,
                "model_name": model_name,
            }
        )

    return pd.DataFrame(rows)


def extract_xgb_outputs(
    model_name: str,
    xgb_model: Any,
    X_test: pd.DataFrame,
    feature_names: list[str],
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    evals_result: dict[str, Any] | None = None,
    include_shap: bool = True,
) -> dict[str, pd.DataFrame]:
    """Extract XGBoost-specific diagnostics and explainability tables."""
    booster = xgb_model.get_booster()
    importance = _xgb_importance_table(model_name, booster, feature_names)

    shap_df = pd.DataFrame(columns=["feature", "mean_abs_shap", "shap_rank", "model_name"])
    shap_note = "skipped"
    if include_shap:
        shap_df, shap_note = _xgb_shap_summary(model_name, xgb_model, X_test, feature_names)

    params = xgb_model.get_params()
    hyper = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "n_estimators": params.get("n_estimators"),
                "max_depth": params.get("max_depth"),
                "learning_rate": params.get("learning_rate"),
                "subsample": params.get("subsample"),
                "colsample_bytree": params.get("colsample_bytree"),
                "reg_alpha": params.get("reg_alpha"),
                "reg_lambda": params.get("reg_lambda"),
                "objective": params.get("objective"),
                "random_state": params.get("random_state"),
                "early_stopping_rounds": params.get("early_stopping_rounds"),
                "best_iteration": getattr(xgb_model, "best_iteration", np.nan),
            }
        ]
    )

    history = _xgb_training_history_table(model_name, evals_result)
    train_rmse = _rmse(y_train, pd.Series(y_pred_train))
    test_rmse = _rmse(y_test, pd.Series(y_pred_test))

    final_train_score = np.nan
    final_val_score = np.nan
    if not history.empty:
        final_train_score = float(history["train_metric_value"].iloc[-1])
        if history["val_metric_value"].notna().any():
            final_val_score = float(history["val_metric_value"].dropna().iloc[-1])

    diag = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "metric": "best_iteration",
                "value": float(getattr(xgb_model, "best_iteration", np.nan)),
            },
            {
                "model_name": model_name,
                "metric": "final_train_score",
                "value": final_train_score,
            },
            {
                "model_name": model_name,
                "metric": "final_val_score",
                "value": final_val_score,
            },
            {
                "model_name": model_name,
                "metric": "train_val_gap",
                "value": final_val_score - final_train_score if pd.notna(final_val_score) and pd.notna(final_train_score) else np.nan,
            },
            {
                "model_name": model_name,
                "metric": "prediction_variance",
                "value": float(np.var(y_pred_test)),
            },
            {
                "model_name": model_name,
                "metric": "train_test_rmse_gap",
                "value": float(test_rmse - train_rmse) if pd.notna(train_rmse) and pd.notna(test_rmse) else np.nan,
            },
            {
                "model_name": model_name,
                "metric": "shap_status",
                "value": shap_note,
            },
        ]
    )

    return {
        "xgb_feature_importance": importance,
        "xgb_shap_summary": shap_df,
        "xgb_hyperparams": hyper,
        "xgb_training_history": history,
        "xgb_diagnostics": diag,
    }


def build_model_report(
    model_name: str,
    model_type: str,
    target_name: str,
    fit_result: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    feature_names: list[str],
    transaction_cost_bps: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Build common output tables for a single model run."""
    pred_train = _build_split_predictions(y_train, fit_result["pred_train"], train_metadata)
    pred_val = _build_split_predictions(y_val, fit_result["pred_val"], val_metadata)
    pred_test = compute_common_predictions_table(
        model_name=model_name,
        y_test=y_test,
        y_pred_test=fit_result["pred_test"],
        test_metadata=test_metadata,
        transaction_cost_bps=transaction_cost_bps,
    )

    perf = pd.concat(
        [
            compute_performance_metrics(model_name, "train", y_train, fit_result["pred_train"], train_metadata),
            compute_performance_metrics(model_name, "val", y_val, fit_result["pred_val"], val_metadata),
            compute_performance_metrics(model_name, "test", y_test, fit_result["pred_test"], test_metadata),
        ],
        ignore_index=True,
    )

    daily = compute_daily_signal_summary(pred_test)
    train_pred_mean = float(pred_train["y_pred"].mean()) if len(pred_train) else np.nan
    health = run_model_health_checks(
        model_name=model_name,
        predictions_test=pred_test,
        performance_metrics=perf,
        feature_count_expected=len(feature_names),
        feature_count_actual=X_test.shape[1],
        runtime_seconds=float(fit_result.get("runtime_seconds", np.nan)),
        train_prediction_mean=train_pred_mean,
    )

    train_start, train_end = _date_range(train_metadata)
    test_start, test_end = _date_range(test_metadata)

    fail_count = int((health["check_status"] == "FAIL").sum())
    warn_count = int((health["check_status"] == "WARN").sum())
    status = "FAIL" if fail_count > 0 else "PASS"
    notes = f"health_fail={fail_count}, health_warn={warn_count}"

    summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "model_type": model_type,
                "target_name": target_name,
                "train_rows": int(len(X_train)),
                "val_rows": int(len(X_val)),
                "test_rows": int(len(X_test)),
                "train_start_date": train_start,
                "train_end_date": train_end,
                "test_start_date": test_start,
                "test_end_date": test_end,
                "run_timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "status": status,
                "notes": notes,
            }
        ]
    )

    return {
        "model_summary": summary,
        "predictions_test": pred_test,
        "performance_metrics": perf,
        "daily_signal_summary": daily,
        "model_health": health,
    }


def _empty_model_package(model_name: str, model_type: str, target_name: str, note: str) -> dict[str, pd.DataFrame]:
    """Fallback package when a model run fails."""
    summary = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "model_type": model_type,
                "target_name": target_name,
                "train_rows": 0,
                "val_rows": 0,
                "test_rows": 0,
                "train_start_date": None,
                "train_end_date": None,
                "test_start_date": None,
                "test_end_date": None,
                "run_timestamp": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "FAIL",
                "notes": note,
            }
        ]
    )

    health = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "check_name": "runtime_exception",
                "check_status": "FAIL",
                "check_value": note,
                "threshold": "No exception",
                "comment": "Model run did not complete.",
            }
        ]
    )

    return {
        "model_summary": summary,
        "predictions_test": pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "y_true",
                "y_pred",
                "residual",
                "abs_error",
                "signal",
                "pred_rank_daily",
                "confidence_score",
                "model_name",
            ]
        ),
        "performance_metrics": pd.DataFrame(
            columns=[
                "model_name",
                "split",
                "mae",
                "rmse",
                "r2",
                "directional_accuracy",
                "information_coefficient_pearson",
                "information_coefficient_spearman",
                "prediction_mean",
                "prediction_std",
                "actual_mean",
                "actual_std",
                "n_obs",
            ]
        ),
        "daily_signal_summary": pd.DataFrame(
            columns=[
                "date",
                "n_universe",
                "n_long_signals",
                "avg_predicted_return",
                "top_ticker_1",
                "top_pred_1",
                "top_ticker_2",
                "top_pred_2",
                "top_ticker_3",
                "top_pred_3",
                "model_name",
            ]
        ),
        "model_health": health,
    }


def build_full_comparison_report(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_val: pd.DataFrame | np.ndarray,
    y_val: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    feature_names: list[str],
    test_metadata: pd.DataFrame,
    train_metadata: pd.DataFrame | None = None,
    val_metadata: pd.DataFrame | None = None,
    target_name: str = DEFAULT_TARGET_NAME,
    transaction_cost_bps: float = 0.0,
    ridge_params: dict[str, Any] | None = None,
    rf_params: dict[str, Any] | None = None,
    xgb_params: dict[str, Any] | None = None,
    include_shap: bool = True,
    selected_models: list[str] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Run selected models and return nested standardized report outputs."""
    ridge_params = ridge_params or {}
    rf_params = rf_params or {}
    xgb_params = xgb_params or {}

    x_tr = _to_frame(X_train, feature_names)
    x_va = _to_frame(X_val, feature_names)
    x_te = _to_frame(X_test, feature_names)

    y_tr = _to_series(y_train, name=target_name)
    y_va = _to_series(y_val, name=target_name)
    y_te = _to_series(y_test, name=target_name)

    meta_tr = _coerce_metadata(train_metadata, len(x_tr), "train")
    meta_va = _coerce_metadata(val_metadata, len(x_va), "val")
    meta_te = _coerce_metadata(test_metadata, len(x_te), "test")

    reports: dict[str, dict[str, pd.DataFrame]] = {}
    comparison_rows: list[dict[str, Any]] = []

    model_specs = {
        "ridge": ("Ridge Regression", "coefficients"),
        "random_forest": ("Random Forest Regressor", "feature_importance"),
        "xgboost": ("XGBoost Regressor", "SHAP/feature_importance"),
    }
    default_order = ["ridge", "random_forest", "xgboost"]

    if selected_models is None:
        model_order = default_order
    else:
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in selected_models:
            key = str(raw).strip().lower()
            if key in model_specs and key not in seen:
                normalized.append(key)
                seen.add(key)
        model_order = normalized

    if not model_order:
        raise ValueError("selected_models must include at least one of ridge/random_forest/xgboost")

    for model_key in model_order:
        model_label, explain = model_specs[model_key]
        try:
            if model_key == "ridge":
                fit = fit_ridge_model(
                    x_tr,
                    y_tr,
                    x_va,
                    y_va,
                    x_te,
                    feature_names,
                    **ridge_params,
                )
            elif model_key == "random_forest":
                fit = fit_random_forest_model(
                    x_tr,
                    y_tr,
                    x_va,
                    x_te,
                    feature_names,
                    **rf_params,
                )
            else:
                fit = fit_xgboost_model(
                    x_tr,
                    y_tr,
                    x_va,
                    y_va,
                    x_te,
                    feature_names,
                    **xgb_params,
                )

            common = build_model_report(
                model_name=model_key,
                model_type=model_label,
                target_name=target_name,
                fit_result=fit,
                X_train=x_tr,
                y_train=y_tr,
                X_val=x_va,
                y_val=y_va,
                X_test=x_te,
                y_test=y_te,
                train_metadata=meta_tr,
                val_metadata=meta_va,
                test_metadata=meta_te,
                feature_names=feature_names,
                transaction_cost_bps=transaction_cost_bps,
            )

            if model_key == "ridge":
                specific = extract_ridge_outputs(model_key, fit["model"], feature_names, fit["pred_test"])
            elif model_key == "random_forest":
                specific = extract_rf_outputs(
                    model_key,
                    fit["model"],
                    x_va,
                    y_va,
                    feature_names,
                    fit["pred_train"],
                    fit["pred_test"],
                    y_tr,
                    y_te,
                )
            else:
                specific = extract_xgb_outputs(
                    model_key,
                    fit["model"],
                    x_te,
                    feature_names,
                    fit["pred_train"],
                    fit["pred_test"],
                    y_tr,
                    y_te,
                    evals_result=fit.get("evals_result"),
                    include_shap=include_shap,
                )

            package = {**common, **specific}
            reports[model_key] = package

            perf_test = package["performance_metrics"]
            perf_test = perf_test[perf_test["split"] == "test"]
            if perf_test.empty:
                test_row = {}
            else:
                test_row = perf_test.iloc[0].to_dict()

            signal = package["daily_signal_summary"]
            avg_longs = float(signal["n_long_signals"].mean()) if isinstance(signal, pd.DataFrame) and not signal.empty else np.nan
            status = str(package["model_summary"].iloc[0].get("status", "FAIL"))
            notes = str(package["model_summary"].iloc[0].get("notes", ""))

            comparison_rows.append(
                {
                    "model_name": model_key,
                    "test_mae": test_row.get("mae", np.nan),
                    "test_rmse": test_row.get("rmse", np.nan),
                    "test_r2": test_row.get("r2", np.nan),
                    "test_directional_accuracy": test_row.get("directional_accuracy", np.nan),
                    "test_ic_pearson": test_row.get("information_coefficient_pearson", np.nan),
                    "test_ic_spearman": test_row.get("information_coefficient_spearman", np.nan),
                    "prediction_std": test_row.get("prediction_std", np.nan),
                    "n_long_signals_avg_per_day": avg_longs,
                    "status": status,
                    "primary_explainability_output": explain,
                    "notes": notes,
                }
            )

        except Exception as exc:
            fail_note = str(exc)
            package = _empty_model_package(model_key, model_label, target_name, fail_note)
            reports[model_key] = package
            comparison_rows.append(
                {
                    "model_name": model_key,
                    "test_mae": np.nan,
                    "test_rmse": np.nan,
                    "test_r2": np.nan,
                    "test_directional_accuracy": np.nan,
                    "test_ic_pearson": np.nan,
                    "test_ic_spearman": np.nan,
                    "prediction_std": np.nan,
                    "n_long_signals_avg_per_day": np.nan,
                    "status": "FAIL",
                    "primary_explainability_output": explain,
                    "notes": fail_note,
                }
            )

    model_comparison = pd.DataFrame(comparison_rows)
    if not model_comparison.empty:
        model_comparison = model_comparison.sort_values(
            ["test_ic_spearman", "test_rmse"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    reports["comparison"] = {"model_comparison": model_comparison}
    return reports

def export_reports_to_csv(
    reports: dict[str, dict[str, pd.DataFrame]],
    output_root: str | Path = "outputs",
) -> dict[str, Path]:
    """Export all report tables to model-specific CSV folders."""
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    folder_map = {
        "ridge": "ridge",
        "random_forest": "random_forest",
        "xgboost": "xgboost",
        "comparison": "comparison",
    }

    for section, tables in reports.items():
        folder = out_root / folder_map.get(section, section)
        folder.mkdir(parents=True, exist_ok=True)

        for table_name, df in tables.items():
            if not isinstance(df, pd.DataFrame):
                continue
            path = folder / f"{table_name}.csv"
            df.to_csv(path, index=False)
            written[f"{section}.{table_name}"] = path

    return written


def create_synthetic_example_dataset(
    feature_names: list[str],
    tickers: list[str],
    n_days: int = 260,
    target_name: str = DEFAULT_TARGET_NAME,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a synthetic ticker-date dataset for sandbox demos."""
    rng = np.random.default_rng(random_state)
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    rows = []
    n_feat = len(feature_names)
    w = rng.normal(0, 0.3, n_feat)
    if n_feat > 0:
        w[: min(5, n_feat)] += np.array([0.8, -0.6, 0.5, 0.4, -0.3])[: min(5, n_feat)]

    for ticker in tickers:
        ticker_bias = rng.normal(0, 0.25)
        trend = np.linspace(-0.2, 0.2, n_days)
        latent = rng.normal(0, 1.0, (n_days, n_feat))
        latent[:, 0] += trend
        latent[:, 1] += np.sin(np.linspace(0, 8, n_days))

        noise = rng.normal(0, 0.7, n_days)
        target = latent @ w + ticker_bias + noise

        for i, dt in enumerate(dates):
            rec = {"date": dt, "ticker": ticker}
            for j, name in enumerate(feature_names):
                rec[name] = float(latent[i, j])
            rec[target_name] = float(target[i])
            rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["date", "ticker"]).reset_index(drop=True)
    return out


def time_based_split(
    df: pd.DataFrame,
    date_col: str = "date",
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date buckets to avoid time leakage."""
    if not (0 < val_fraction < 0.5) or not (0 < test_fraction < 0.5):
        raise ValueError("val_fraction and test_fraction must be between 0 and 0.5")
    if val_fraction + test_fraction >= 0.8:
        raise ValueError("val_fraction + test_fraction must be < 0.8")

    if date_col not in df.columns:
        raise ValueError(f"date column '{date_col}' not found")

    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)

    unique_dates = work[date_col].drop_duplicates().sort_values().to_list()
    n_dates = len(unique_dates)
    if n_dates < 30:
        raise ValueError("Need at least 30 unique dates for train/val/test split.")

    n_test = max(1, int(round(n_dates * test_fraction)))
    n_val = max(1, int(round(n_dates * val_fraction)))
    n_train = n_dates - n_val - n_test
    if n_train < 10:
        raise ValueError("Training split too small after date-based split.")

    train_dates = set(unique_dates[:n_train])
    val_dates = set(unique_dates[n_train : n_train + n_val])
    test_dates = set(unique_dates[n_train + n_val :])

    train_df = work[work[date_col].isin(train_dates)].copy()
    val_df = work[work[date_col].isin(val_dates)].copy()
    test_df = work[work[date_col].isin(test_dates)].copy()

    return train_df, val_df, test_df
