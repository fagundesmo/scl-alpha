"""
Runtime utilities for interactive Streamlit model runs.
"""

from __future__ import annotations

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.backtest import buy_and_hold_baseline, run_backtest
from src.config import TICKERS
from src.data_pull import pull_all
from src.features import FEATURE_COLUMNS, build_features
from src.metrics import compute_ml_metrics, compute_trading_metrics
from src.model import get_feature_importance, make_model, predict, train_model

MIN_FEATURE_NON_NULL_RATIO = 0.05
MIN_FEATURE_COUNT = 4
MIN_TRAIN_ROWS = 100
MIN_FOLD_ROWS = 60


def load_feature_matrix(refresh_data: bool = False) -> pd.DataFrame:
    """Download/merge data and compute features for app runs."""
    panel = pull_all(cache=not refresh_data)
    return build_features(panel.copy())


def _prepare_features(tradable: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Keep feature columns with minimum data coverage, then impute remaining gaps.
    """
    if not feature_cols:
        raise ValueError("No feature columns found in the dataset.")

    coverage = tradable[feature_cols].notna().mean()
    usable_cols = [c for c in feature_cols if coverage.get(c, 0.0) >= MIN_FEATURE_NON_NULL_RATIO]

    if len(usable_cols) < MIN_FEATURE_COUNT:
        top_cov = coverage.sort_values(ascending=False).head(10).to_dict()
        raise ValueError(
            "Insufficient feature coverage after data prep. "
            f"Usable features: {len(usable_cols)}. Coverage snapshot: {top_cov}"
        )

    prepared = tradable.copy()
    for col in usable_cols:
        s = prepared[col]
        s = s.groupby(level="ticker").ffill()
        s = s.groupby(level="ticker").bfill()
        if s.isna().all():
            continue
        s = s.fillna(float(s.median()))
        prepared[col] = s

    return prepared, usable_cols


def _trainable_data(df: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    tradable = df[df.index.get_level_values("ticker").isin(tickers)].copy()

    if tradable.empty:
        raise ValueError("No tradable ticker rows found in the prepared dataset.")

    prepared, usable_cols = _prepare_features(tradable, feature_cols)

    target_non_null = int(prepared["target_ret_5d_fwd"].notna().sum())
    if target_non_null == 0:
        raise ValueError(
            "No valid target values found for training. "
            "This usually means price history is too short or unavailable from upstream APIs."
        )

    trainable = prepared.dropna(subset=["target_ret_5d_fwd"])
    if len(trainable) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Only {len(trainable)} trainable rows available; need at least {MIN_TRAIN_ROWS}."
        )

    return trainable.sort_index(), usable_cols, prepared.sort_index()


def _maybe_sample_trainable(
    trainable: pd.DataFrame,
    sample_fraction: float,
    max_rows_per_ticker: int = 220,
) -> pd.DataFrame:
    """Sample the most recent rows per ticker for faster parameter search."""
    sample_fraction = float(sample_fraction)
    if sample_fraction >= 0.999:
        return trainable

    groups = []
    for _, grp in trainable.groupby(level="ticker"):
        desired_n = max(int(len(grp) * sample_fraction), 80)
        keep_n = min(desired_n, max_rows_per_ticker)
        groups.append(grp.tail(keep_n))

    sampled = pd.concat(groups).sort_index()
    if len(sampled) < MIN_TRAIN_ROWS:
        return trainable
    return sampled


def _latest_snapshot(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    dates = df.index.get_level_values("date").unique().sort_values(ascending=False)
    for dt in dates:
        snap = df[df.index.get_level_values("date") == dt].dropna(subset=feature_cols)
        if not snap.empty:
            return snap
    raise ValueError("No valid snapshot date found with complete feature rows.")


def _split_train_test(data: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(data) < 2:
        raise ValueError("Need at least 2 rows to split train/test.")

    split_idx = int(len(data) * (1.0 - test_fraction))
    split_idx = max(split_idx, 1)
    split_idx = min(split_idx, len(data) - 1)
    return data.iloc[:split_idx], data.iloc[split_idx:]


def score_model_params(
    feature_df: pd.DataFrame,
    model_name: str,
    model_params: dict,
    tickers: list[str] | None = None,
    sample_fraction: float = 0.35,
    cv_splits: int = 3,
    max_rows_per_ticker: int = 220,
) -> dict:
    """
    Fast CV-based ML-only evaluation used during grid search (no backtest).
    """
    tickers = tickers or TICKERS
    trainable, feature_cols, _prepared_df = _trainable_data(feature_df, tickers)
    search_df = _maybe_sample_trainable(
        trainable,
        sample_fraction=sample_fraction,
        max_rows_per_ticker=max_rows_per_ticker,
    )

    cv_splits = max(2, int(cv_splits))
    max_allowed_splits = max(2, len(search_df) // MIN_FOLD_ROWS)
    cv_splits = min(cv_splits, max_allowed_splits)

    fold_metrics: list[dict] = []
    if cv_splits >= 2:
        splitter = TimeSeriesSplit(n_splits=cv_splits)
        for train_idx, test_idx in splitter.split(search_df):
            train_data = search_df.iloc[train_idx]
            test_data = search_df.iloc[test_idx]
            if len(train_data) < MIN_TRAIN_ROWS or len(test_data) < MIN_FOLD_ROWS:
                continue

            model_eval = make_model(model_name, params=model_params)
            model_eval = train_model(
                model_eval,
                train_data[feature_cols],
                train_data["target_ret_5d_fwd"],
                X_val=test_data[feature_cols],
                y_val=test_data["target_ret_5d_fwd"],
            )
            y_pred = predict(model_eval, test_data[feature_cols], feature_cols=feature_cols)
            fold_metrics.append(compute_ml_metrics(test_data["target_ret_5d_fwd"].values, y_pred))

    if not fold_metrics:
        train_data, test_data = _split_train_test(search_df)
        model_eval = make_model(model_name, params=model_params)
        model_eval = train_model(
            model_eval,
            train_data[feature_cols],
            train_data["target_ret_5d_fwd"],
            X_val=test_data[feature_cols],
            y_val=test_data["target_ret_5d_fwd"],
        )
        y_pred = predict(model_eval, test_data[feature_cols], feature_cols=feature_cols)
        fold_metrics.append(compute_ml_metrics(test_data["target_ret_5d_fwd"].values, y_pred))

    ml_metrics = (
        pd.DataFrame(fold_metrics)
        .replace([np.inf, -np.inf], np.nan)
        .mean(numeric_only=True)
        .to_dict()
    )

    return {
        "ml_metrics": ml_metrics,
        "feature_cols": feature_cols,
        "n_rows": len(search_df),
        "cv_splits": len(fold_metrics),
    }


def run_single_model(
    feature_df: pd.DataFrame,
    model_name: str,
    model_params: dict,
    top_k: int,
    cost_bps: float,
    slippage_bps: float,
    retrain_every: int,
    initial_train_years: int,
    tickers: list[str] | None = None,
) -> dict:
    """Train/evaluate one model and return all artifacts used by app pages."""
    tickers = tickers or TICKERS
    start_ts = time.time()

    trainable, feature_cols, prepared_df = _trainable_data(feature_df, tickers)

    train_data, test_data = _split_train_test(trainable)
    X_train = train_data[feature_cols]
    y_train = train_data["target_ret_5d_fwd"]
    X_test = test_data[feature_cols]
    y_test = test_data["target_ret_5d_fwd"]

    model_eval = make_model(model_name, params=model_params)
    model_eval = train_model(model_eval, X_train, y_train, X_val=X_test, y_val=y_test)
    y_pred = predict(model_eval, X_test, feature_cols=feature_cols)
    ml_metrics = compute_ml_metrics(y_test.values, y_pred)

    model_full = make_model(model_name, params=model_params)
    model_full = train_model(
        model_full,
        trainable[feature_cols],
        trainable["target_ret_5d_fwd"],
    )

    latest = _latest_snapshot(prepared_df, feature_cols)
    latest_pred = latest.reset_index()[["date", "ticker"]].copy()
    latest_pred["predicted_ret"] = predict(model_full, latest[feature_cols], feature_cols=feature_cols)
    latest_pred = latest_pred.sort_values("predicted_ret", ascending=False)

    importance = get_feature_importance(model_full, feature_cols)

    backtest_df = run_backtest(
        prepared_df,
        model_name=model_name,
        top_k=top_k,
        cost_bps=cost_bps,
        slippage_bps=slippage_bps,
        retrain_every=retrain_every,
        initial_train_years=initial_train_years,
        tickers=tickers,
        model_params=model_params,
        feature_columns=feature_cols,
    )

    if backtest_df.empty:
        trading_metrics = {
            "CAGR": np.nan,
            "Sharpe Ratio": np.nan,
            "Max Drawdown": np.nan,
            "Profit Factor": np.nan,
            "Avg Turnover": np.nan,
            "Total Periods": 0,
            "Positive Periods": 0,
            "Negative Periods": 0,
        }
        benchmark = pd.DataFrame(columns=["cumulative_return"])
    else:
        trading_metrics = compute_trading_metrics(backtest_df)
        try:
            benchmark = buy_and_hold_baseline(
                feature_df,
                ticker="IYT",
                start=backtest_df.index.min(),
                end=backtest_df.index.max(),
            )
        except KeyError:
            benchmark = pd.DataFrame(columns=["cumulative_return"])

    elapsed = time.time() - start_ts

    return {
        "model_name": model_name,
        "model_params": dict(model_params),
        "model": model_full,
        "features_used": feature_cols,
        "ml_metrics": ml_metrics,
        "trading_metrics": trading_metrics,
        "backtest": backtest_df,
        "benchmark": benchmark,
        "latest_predictions": latest_pred,
        "feature_importance": importance,
        "elapsed_seconds": elapsed,
    }


def summarize_runs(runs: dict[str, dict]) -> pd.DataFrame:
    """Build a compact comparison table for multiple runs."""
    rows = []
    for model_name, run in runs.items():
        t = run["trading_metrics"]
        m = run["ml_metrics"]
        rows.append(
            {
                "Model": model_name,
                "MAE": m.get("MAE", np.nan),
                "RMSE": m.get("RMSE", np.nan),
                "IC": m.get("IC", np.nan),
                "Hit Rate": m.get("Hit Rate", np.nan),
                "CAGR": t.get("CAGR", np.nan),
                "Sharpe Ratio": t.get("Sharpe Ratio", np.nan),
                "Max Drawdown": t.get("Max Drawdown", np.nan),
                "Profit Factor": t.get("Profit Factor", np.nan),
                "Avg Turnover": t.get("Avg Turnover", np.nan),
                "Total Periods": t.get("Total Periods", 0),
                "Features Used": len(run.get("features_used", [])),
                "Run Time (s)": run.get("elapsed_seconds", np.nan),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Sharpe Ratio", "CAGR"], ascending=False).reset_index(drop=True)


def persist_run_outputs(cache_dir: Path, run: dict, feature_matrix: pd.DataFrame) -> None:
    """Persist selected run outputs so existing app pages can load without rerun."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    feature_matrix.to_parquet(cache_dir / "feature_matrix.parquet")

    run["latest_predictions"].to_parquet(
        cache_dir / "latest_predictions.parquet",
        index=False,
    )

    if not run["backtest"].empty:
        run["backtest"].to_parquet(cache_dir / "backtest_results.parquet")

    if not run["benchmark"].empty:
        run["benchmark"].to_parquet(cache_dir / "benchmark_iyt.parquet")

    importance = run["feature_importance"]
    if isinstance(importance, pd.Series) and not importance.empty:
        imp_df = (
            importance.rename_axis("feature")
            .reset_index(name="importance")
            .sort_values("importance", key=lambda s: s.abs(), ascending=False)
        )
        imp_df.to_parquet(cache_dir / "feature_importance.parquet", index=False)

    joblib.dump(run["model"], cache_dir / "best_model.joblib")
