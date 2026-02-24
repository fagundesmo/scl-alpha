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

    oos_predictions = test_data.reset_index()[["date", "ticker"]].copy()
    oos_predictions["actual_ret"] = y_test.values
    oos_predictions["predicted_ret"] = y_pred

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
        "oos_predictions": oos_predictions,
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




def _as_holdings(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, (tuple, set, np.ndarray, pd.Series)):
        return [str(x) for x in list(raw)]
    return []


def _status_row(name: str, passed: bool | None, detail: str) -> dict[str, str]:
    if passed is True:
        status = "PASS"
    elif passed is False:
        status = "WARN"
    else:
        status = "INFO"
    return {"Check": name, "Status": status, "Details": detail}


def build_daily_run_outputs(
    run: dict,
    feature_matrix: pd.DataFrame,
    top_k: int,
    cost_bps: float,
    slippage_bps: float,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build detailed daily/operational outputs for Streamlit model monitoring."""
    tickers = tickers or TICKERS
    latest_predictions = run.get("latest_predictions", pd.DataFrame())
    backtest = run.get("backtest", pd.DataFrame())
    benchmark = run.get("benchmark", pd.DataFrame())
    feature_cols = list(run.get("features_used", []))

    if not isinstance(latest_predictions, pd.DataFrame):
        latest_predictions = pd.DataFrame()
    if not isinstance(backtest, pd.DataFrame):
        backtest = pd.DataFrame()
    if not isinstance(benchmark, pd.DataFrame):
        benchmark = pd.DataFrame()
    if not isinstance(feature_matrix, pd.DataFrame):
        feature_matrix = pd.DataFrame()

    pred_map: dict[str, float] = {}
    latest_date = pd.NaT
    if not latest_predictions.empty:
        ranked = latest_predictions.copy().sort_values("predicted_ret", ascending=False)
        ranked = ranked[ranked["ticker"].isin(tickers)]
        pred_map = dict(zip(ranked["ticker"], ranked["predicted_ret"]))
        latest_date = ranked["date"].max() if "date" in ranked.columns else pd.NaT
        target_holdings = (
            ranked[ranked["predicted_ret"] > 0]
            .head(max(int(top_k), 1))["ticker"]
            .astype(str)
            .tolist()
        )
    else:
        ranked = pd.DataFrame(columns=["date", "ticker", "predicted_ret"])
        target_holdings = []

    prev_holdings: list[str] = []
    if not backtest.empty and "holdings" in backtest.columns:
        prev_holdings = _as_holdings(backtest["holdings"].iloc[-1])

    signal_rows = []
    for idx, ticker in enumerate(target_holdings, start=1):
        signal_rows.append(
            {
                "Rank": idx,
                "Ticker": ticker,
                "Predicted Return (%)": pred_map.get(ticker, np.nan),
                "Signal": "LONG",
                "Change vs prior": "Keep" if ticker in prev_holdings else "New",
            }
        )

    dropped = sorted(set(prev_holdings) - set(target_holdings))
    for ticker in dropped:
        signal_rows.append(
            {
                "Rank": np.nan,
                "Ticker": ticker,
                "Predicted Return (%)": pred_map.get(ticker, np.nan),
                "Signal": "DROP",
                "Change vs prior": "Removed",
            }
        )
    signal_summary = pd.DataFrame(signal_rows)

    price_map: dict[str, float] = {}
    data_latest_date = pd.NaT
    if not feature_matrix.empty and "adj_close" in feature_matrix.columns:
        data_latest_date = feature_matrix.index.get_level_values("date").max()
        latest_slice = feature_matrix[
            feature_matrix.index.get_level_values("date") == data_latest_date
        ].reset_index()
        if "ticker" in latest_slice.columns:
            latest_slice = latest_slice[latest_slice["ticker"].isin(tickers)]
            if "adj_close" in latest_slice.columns:
                price_map = dict(zip(latest_slice["ticker"], latest_slice["adj_close"]))

    universe = sorted(set(prev_holdings) | set(target_holdings))
    curr_w = 1.0 / len(prev_holdings) if prev_holdings else 0.0
    targ_w = 1.0 / len(target_holdings) if target_holdings else 0.0
    total_cost_bps = float(cost_bps) + float(slippage_bps)

    order_rows = []
    for ticker in universe:
        current_weight = curr_w if ticker in prev_holdings else 0.0
        target_weight = targ_w if ticker in target_holdings else 0.0
        delta = target_weight - current_weight
        if delta > 1e-12:
            action = "BUY"
        elif delta < -1e-12:
            action = "SELL"
        else:
            action = "HOLD"
        order_rows.append(
            {
                "Ticker": ticker,
                "Action": action,
                "Current Weight": current_weight,
                "Target Weight": target_weight,
                "Weight Change": delta,
                "Signal Score (%)": pred_map.get(ticker, np.nan),
                "Reference Price": price_map.get(ticker, np.nan),
                "Estimated Cost (bps)": abs(delta) * total_cost_bps,
                "Estimated Cost (%)": abs(delta) * total_cost_bps / 100.0,
            }
        )

    orders_generated = pd.DataFrame(order_rows)
    if not orders_generated.empty:
        orders_generated = orders_generated.sort_values(
            ["Action", "Weight Change"],
            ascending=[True, False],
            key=lambda s: s.abs() if s.name == "Weight Change" else s,
        ).reset_index(drop=True)

    executed_trades = orders_generated[orders_generated.get("Action", pd.Series(dtype=str)).isin(["BUY", "SELL"])].copy()
    if not executed_trades.empty:
        executed_trades["Execution Status"] = "Simulated fill"
        executed_trades["Fill Price"] = executed_trades["Reference Price"]

    positions_rows = []
    for ticker in target_holdings:
        positions_rows.append(
            {
                "Ticker": ticker,
                "Weight": targ_w,
                "Predicted Return (%)": pred_map.get(ticker, np.nan),
                "Last Price": price_map.get(ticker, np.nan),
            }
        )
    positions_exposure = pd.DataFrame(positions_rows)

    hhi = float(np.sum(np.square([targ_w] * len(target_holdings)))) if target_holdings else 0.0
    effective_n = (1.0 / hhi) if hhi > 0 else 0.0
    turnover_est = float(np.sum(np.abs(orders_generated["Weight Change"]))) if not orders_generated.empty else 0.0
    exposure_summary = pd.DataFrame(
        [
            {"Metric": "Long positions", "Value": len(target_holdings)},
            {"Metric": "Previous positions", "Value": len(prev_holdings)},
            {"Metric": "Gross exposure", "Value": float(np.sum(np.abs([targ_w] * len(target_holdings)))) if target_holdings else 0.0},
            {"Metric": "Net exposure", "Value": float(np.sum([targ_w] * len(target_holdings))) if target_holdings else 0.0},
            {"Metric": "Max position weight", "Value": targ_w if target_holdings else 0.0},
            {"Metric": "Effective N", "Value": effective_n},
            {"Metric": "Herfindahl index", "Value": hhi},
            {"Metric": "Estimated turnover", "Value": turnover_est},
        ]
    )

    pnl_rows = []
    cost_detail = pd.DataFrame()
    cost_summary = pd.DataFrame()
    risk_metrics = pd.DataFrame()

    if not backtest.empty:
        net = backtest["net_return_pct"]
        gross = backtest["gross_return_pct"]
        latest_net = float(net.iloc[-1])
        latest_gross = float(gross.iloc[-1])
        latest_cost = float(backtest["cost_pct"].iloc[-1])

        gross_cum = float(((1 + gross / 100.0).prod() - 1) * 100.0)
        net_cum = float((backtest["cumulative_return"].iloc[-1] - 1.0) * 100.0)
        cost_cum = float(backtest["cost_pct"].sum())

        bench_last = np.nan
        bench_cum = np.nan
        if not benchmark.empty and "cumulative_return" in benchmark.columns:
            bench_cum = float((benchmark["cumulative_return"].iloc[-1] - 1.0) * 100.0)
            bench_ret = benchmark["cumulative_return"].pct_change().dropna()
            if not bench_ret.empty:
                bench_last = float(bench_ret.iloc[-1] * 100.0)

        pnl_rows.extend(
            [
                {
                    "Component": "Signal return (gross)",
                    "Last period (%)": latest_gross,
                    "Cumulative (%)": gross_cum,
                },
                {
                    "Component": "Transaction + slippage drag",
                    "Last period (%)": -latest_cost,
                    "Cumulative (%)": -cost_cum,
                },
                {
                    "Component": "Net portfolio return",
                    "Last period (%)": latest_net,
                    "Cumulative (%)": net_cum,
                },
            ]
        )

        if pd.notna(bench_cum):
            pnl_rows.append(
                {
                    "Component": "Benchmark return (IYT)",
                    "Last period (%)": bench_last,
                    "Cumulative (%)": bench_cum,
                }
            )
            pnl_rows.append(
                {
                    "Component": "Net alpha vs benchmark",
                    "Last period (%)": latest_net - bench_last if pd.notna(bench_last) else np.nan,
                    "Cumulative (%)": net_cum - bench_cum,
                }
            )

        commission_drag = backtest["turnover_frac"] * (float(cost_bps) / 100.0)
        slippage_drag = backtest["turnover_frac"] * (float(slippage_bps) / 100.0)
        cost_summary = pd.DataFrame(
            [
                {"Metric": "Avg weekly turnover", "Value": float(backtest["turnover_frac"].mean())},
                {"Metric": "Avg weekly cost (%)", "Value": float(backtest["cost_pct"].mean())},
                {"Metric": "Total cost (%)", "Value": float(backtest["cost_pct"].sum())},
                {"Metric": "Commission drag (%)", "Value": float(commission_drag.sum())},
                {"Metric": "Slippage drag (%)", "Value": float(slippage_drag.sum())},
            ]
        )

        cost_detail = backtest[["turnover_frac", "cost_pct"]].copy()
        cost_detail["commission_pct"] = commission_drag
        cost_detail["slippage_pct"] = slippage_drag
        cost_detail = cost_detail.tail(20).reset_index()
        cost_detail.rename(
            columns={
                "date": "Date",
                "turnover_frac": "Turnover",
                "cost_pct": "Total Cost (%)",
                "commission_pct": "Commission (%)",
                "slippage_pct": "Slippage (%)",
            },
            inplace=True,
        )

        net_dec = net / 100.0
        var95 = float(np.quantile(net_dec, 0.05)) if len(net_dec) else np.nan
        es95 = float(net_dec[net_dec <= var95].mean()) if len(net_dec) else np.nan
        dd_series = (backtest["cumulative_return"] / backtest["cumulative_return"].cummax()) - 1.0

        beta = np.nan
        if not benchmark.empty and "cumulative_return" in benchmark.columns:
            bench_weekly = benchmark["cumulative_return"].resample("W-FRI").last().pct_change().dropna()
            aligned = pd.concat([net_dec.rename("portfolio"), bench_weekly.rename("benchmark")], axis=1).dropna()
            if len(aligned) >= 3 and aligned["benchmark"].var() > 0:
                beta = float(aligned["portfolio"].cov(aligned["benchmark"]) / aligned["benchmark"].var())

        risk_metrics = pd.DataFrame(
            [
                {"Metric": "Annualized volatility", "Value": float(net_dec.std(ddof=0) * np.sqrt(52.0))},
                {"Metric": "VaR 95% (weekly)", "Value": var95},
                {"Metric": "Expected shortfall 95%", "Value": es95},
                {"Metric": "Max drawdown", "Value": float(-dd_series.min())},
                {"Metric": "Current drawdown", "Value": float(-dd_series.iloc[-1])},
                {"Metric": "Beta vs IYT", "Value": beta},
            ]
        )

    pnl_attribution = pd.DataFrame(pnl_rows)

    constraint_checks = []
    if not backtest.empty:
        constraint_checks.append(
            _status_row(
                "Position cap (n_holdings <= top_k)",
                bool((backtest["n_holdings"] <= max(top_k, 1)).all()),
                f"max observed={int(backtest['n_holdings'].max())}, limit={max(top_k, 1)}",
            )
        )
        constraint_checks.append(
            _status_row(
                "Turnover cap (<= 100%)",
                bool((backtest["turnover_frac"] <= 1.0 + 1e-9).all()),
                f"max observed={backtest['turnover_frac'].max():.2%}",
            )
        )
        constraint_checks.append(
            _status_row(
                "Cost non-negative",
                bool((backtest["cost_pct"] >= 0).all()),
                f"min observed={backtest['cost_pct'].min():.4f}%",
            )
        )
    else:
        constraint_checks.append(_status_row("Backtest availability", False, "No backtest periods available."))

    constraint_checks.append(
        _status_row(
            "Long-only signal",
            True,
            "All generated orders are long-only (BUY/SELL between long basket and cash).",
        )
    )
    constraint_checks.append(
        _status_row(
            "Borrow/short availability",
            None,
            "Not applicable for long-only configuration.",
        )
    )
    constraint_checks.append(
        _status_row(
            "Liquidity cap",
            None,
            "No ADV/liquidity model is currently configured in this app.",
        )
    )
    constraint_checks_df = pd.DataFrame(constraint_checks)

    data_quality_rows = []
    if not feature_matrix.empty and feature_cols:
        tradable = feature_matrix[feature_matrix.index.get_level_values("ticker").isin(tickers)].copy()
        if not tradable.empty:
            fm_latest = tradable.index.get_level_values("date").max()
            latest_slice = tradable[tradable.index.get_level_values("date") == fm_latest]

            stale_days = int((pd.Timestamp.utcnow().normalize() - pd.Timestamp(fm_latest).normalize()).days)
            feature_missing = float(latest_slice[feature_cols].isna().mean().mean()) if feature_cols else np.nan
            price_missing = float(latest_slice["adj_close"].isna().mean()) if "adj_close" in latest_slice.columns else np.nan
            target_cov = (
                float(tradable["target_ret_5d_fwd"].notna().mean())
                if "target_ret_5d_fwd" in tradable.columns
                else np.nan
            )
            outlier_ratio = np.nan
            if "target_ret_5d_fwd" in tradable.columns and tradable["target_ret_5d_fwd"].notna().any():
                targ = tradable["target_ret_5d_fwd"].dropna().abs()
                cutoff = float(targ.quantile(0.995)) if len(targ) else np.inf
                outlier_ratio = float((targ >= cutoff).mean()) if np.isfinite(cutoff) else np.nan

            data_quality_rows.extend(
                [
                    _status_row("Data freshness", stale_days <= 5, f"latest date={fm_latest.date()}, stale={stale_days} day(s)"),
                    _status_row("Latest feature completeness", feature_missing <= 0.05, f"missing={feature_missing:.2%}"),
                    _status_row("Latest price completeness", price_missing <= 0.01 if pd.notna(price_missing) else None, f"missing={price_missing:.2%}" if pd.notna(price_missing) else "adj_close unavailable"),
                    _status_row("Target coverage", target_cov >= 0.30 if pd.notna(target_cov) else None, f"coverage={target_cov:.2%}" if pd.notna(target_cov) else "target_ret_5d_fwd unavailable"),
                    _status_row("Extreme target outliers", outlier_ratio <= 0.02 if pd.notna(outlier_ratio) else None, f"ratio={outlier_ratio:.2%}" if pd.notna(outlier_ratio) else "insufficient data"),
                ]
            )

    if not data_quality_rows:
        data_quality_rows.append(_status_row("Data quality availability", False, "No feature matrix available for diagnostics."))
    data_quality_checks = pd.DataFrame(data_quality_rows)

    ml = run.get("ml_metrics", {}) or {}
    search_ml = run.get("search_ml_metrics", {}) or {}
    pred_dispersion = float(ranked["predicted_ret"].std()) if not ranked.empty else np.nan
    top_bottom_spread = (
        float(ranked["predicted_ret"].iloc[0] - ranked["predicted_ret"].iloc[-1])
        if len(ranked) > 1
        else np.nan
    )

    oos = run.get("oos_predictions", pd.DataFrame())
    recent_ic = np.nan
    recent_hit = np.nan
    if isinstance(oos, pd.DataFrame) and not oos.empty:
        recent = oos.tail(min(200, len(oos)))
        if recent["actual_ret"].notna().sum() >= 3:
            recent_ic = float(recent["actual_ret"].corr(recent["predicted_ret"], method="spearman"))
            recent_hit = float((np.sign(recent["actual_ret"]) == np.sign(recent["predicted_ret"])).mean())

    full_ic = ml.get("IC", np.nan)
    search_ic = search_ml.get("IC", np.nan)
    ic_delta = full_ic - search_ic if pd.notna(full_ic) and pd.notna(search_ic) else np.nan

    model_health = pd.DataFrame(
        [
            {
                "Metric": "IC (out-of-sample)",
                "Value": full_ic,
                "Status": "PASS" if pd.notna(full_ic) and full_ic > 0 else "WARN",
            },
            {
                "Metric": "Hit rate",
                "Value": ml.get("Hit Rate", np.nan),
                "Status": "PASS" if pd.notna(ml.get("Hit Rate", np.nan)) and ml.get("Hit Rate", 0) >= 0.50 else "WARN",
            },
            {
                "Metric": "RMSE",
                "Value": ml.get("RMSE", np.nan),
                "Status": "INFO",
            },
            {
                "Metric": "Recent IC (last <=200 preds)",
                "Value": recent_ic,
                "Status": "PASS" if pd.notna(recent_ic) and recent_ic > 0 else "WARN" if pd.notna(recent_ic) else "INFO",
            },
            {
                "Metric": "Recent hit rate",
                "Value": recent_hit,
                "Status": "PASS" if pd.notna(recent_hit) and recent_hit >= 0.50 else "WARN" if pd.notna(recent_hit) else "INFO",
            },
            {
                "Metric": "Prediction dispersion (%)",
                "Value": pred_dispersion,
                "Status": "PASS" if pd.notna(pred_dispersion) and pred_dispersion > 0 else "WARN",
            },
            {
                "Metric": "Top-bottom spread (%)",
                "Value": top_bottom_spread,
                "Status": "PASS" if pd.notna(top_bottom_spread) and top_bottom_spread > 0 else "WARN",
            },
            {
                "Metric": "IC drift vs search",
                "Value": ic_delta,
                "Status": "PASS" if pd.notna(ic_delta) and abs(ic_delta) <= 0.05 else "WARN" if pd.notna(ic_delta) else "INFO",
            },
        ]
    )

    system_health_rows = [
        {
            "Metric": "Run duration (s)",
            "Value": float(run.get("elapsed_seconds", np.nan)),
            "Status": "PASS" if pd.notna(run.get("elapsed_seconds", np.nan)) else "INFO",
        },
        {
            "Metric": "Trials evaluated",
            "Value": int(run.get("trial_count", 1)),
            "Status": "INFO",
        },
        {
            "Metric": "Features used",
            "Value": len(feature_cols),
            "Status": "PASS" if len(feature_cols) >= MIN_FEATURE_COUNT else "WARN",
        },
        {
            "Metric": "Backtest periods",
            "Value": int(len(backtest)),
            "Status": "PASS" if len(backtest) >= 20 else "WARN",
        },
        {
            "Metric": "Latest prediction date",
            "Value": latest_date,
            "Status": "PASS" if pd.notna(latest_date) else "WARN",
        },
        {
            "Metric": "Latest data date",
            "Value": data_latest_date,
            "Status": "PASS" if pd.notna(data_latest_date) else "WARN",
        },
        {
            "Metric": "Completed UTC",
            "Value": run.get("completed_utc", "n/a"),
            "Status": "INFO",
        },
    ]
    system_health = pd.DataFrame(system_health_rows)

    return {
        "signal_summary": signal_summary,
        "orders_generated": orders_generated,
        "executed_trades": executed_trades,
        "positions_exposure": positions_exposure,
        "exposure_summary": exposure_summary,
        "pnl_attribution": pnl_attribution,
        "transaction_costs_summary": cost_summary,
        "transaction_costs_detail": cost_detail,
        "risk_metrics": risk_metrics,
        "constraint_checks": constraint_checks_df,
        "data_quality_checks": data_quality_checks,
        "model_health": model_health,
        "system_health": system_health,
    }
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
