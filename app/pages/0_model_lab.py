"""
Interactive model runner for the Streamlit app.

Lets users:
- choose one model or multiple models
- tune model/backtest parameters (single value or range)
- run and compare results side-by-side
- publish one run to other app pages
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.app_runner import (
    load_feature_matrix,
    persist_run_outputs,
    run_single_model,
    summarize_runs,
)
from src.config import (
    INITIAL_TRAIN_YEARS,
    RETRAIN_EVERY_WEEKS,
    RF_PARAMS,
    RIDGE_ALPHA,
    SLIPPAGE_BPS,
    TICKERS,
    TOP_K,
    TRANSACTION_COST_BPS,
    XGB_PARAMS,
)
from src.plots import plot_feature_importance

MAX_GRID_TRIALS_PER_MODEL = 18


def _float_grid(min_val: float, max_val: float, n_values: int) -> list[float]:
    if n_values <= 1:
        return [float(min_val)]
    vals = np.linspace(float(min_val), float(max_val), int(n_values))
    return [float(v) for v in vals]


def _int_grid(min_val: int, max_val: int, n_values: int) -> list[int]:
    if n_values <= 1:
        return [int(min_val)]
    vals = np.linspace(int(min_val), int(max_val), int(n_values))
    out = sorted(set(int(round(v)) for v in vals))
    return out if out else [int(min_val)]


def _downsample_trials(candidates: list[dict], limit: int = MAX_GRID_TRIALS_PER_MODEL) -> list[dict]:
    if len(candidates) <= limit:
        return candidates

    idx = np.linspace(0, len(candidates) - 1, limit, dtype=int)
    return [candidates[int(i)] for i in idx]


def _score_run(run: dict) -> tuple[float, float]:
    trade = run.get("trading_metrics", {})
    sharpe = trade.get("Sharpe Ratio", np.nan)
    cagr = trade.get("CAGR", np.nan)

    sharpe_score = float(sharpe) if pd.notna(sharpe) else -1e9
    cagr_score = float(cagr) if pd.notna(cagr) else -1e9
    return sharpe_score, cagr_score


st.set_page_config(page_title="Model Lab", page_icon="🧪", layout="wide")
st.title("🧪 Model Lab")
st.caption(
    "Select one or more models, configure parameters, run experiments, and compare results."
)

model_options = ["ridge", "rf", "xgboost"]

selected_models = st.multiselect(
    "Models to run",
    options=model_options,
    default=model_options,
    help="Pick one model or multiple models for side-by-side comparison.",
)

with st.expander("Portfolio and backtest settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    top_k = c1.slider("portfolio.top_k (number of long positions)", min_value=1, max_value=len(TICKERS), value=TOP_K)
    initial_train_years = c2.slider(
        "backtest.initial_train_years",
        min_value=1,
        max_value=6,
        value=INITIAL_TRAIN_YEARS,
    )
    retrain_every = c3.slider(
        "backtest.retrain_every_weeks",
        min_value=4,
        max_value=26,
        value=RETRAIN_EVERY_WEEKS,
    )

    c4, c5 = st.columns(2)
    cost_bps = c4.number_input(
        "backtest.transaction_cost_bps",
        min_value=0.0,
        max_value=100.0,
        value=float(TRANSACTION_COST_BPS),
        step=1.0,
    )
    slippage_bps = c5.number_input(
        "backtest.slippage_bps",
        min_value=0.0,
        max_value=100.0,
        value=float(SLIPPAGE_BPS),
        step=1.0,
    )

refresh_data = st.checkbox(
    "Refresh market/macro data before run",
    value=False,
    help="Turn on to force API refresh instead of using cached parquet files.",
)

range_mode = st.checkbox(
    "Use parameter ranges (grid search)",
    value=False,
    help="When enabled, each selected model runs multiple parameter combinations and keeps the best one.",
)

with st.expander("Model parameters", expanded=True):
    st.markdown("#### Ridge parameters")
    if range_mode:
        rc1, rc2, rc3 = st.columns(3)
        ridge_alpha_min = rc1.number_input("ridge.alpha min (L2 strength)", min_value=0.001, max_value=100.0, value=0.1, step=0.1, format="%.3f")
        ridge_alpha_max = rc2.number_input("ridge.alpha max (L2 strength)", min_value=0.001, max_value=100.0, value=5.0, step=0.1, format="%.3f")
        ridge_alpha_n = rc3.slider("ridge.alpha #values", min_value=1, max_value=8, value=4)
    else:
        ridge_alpha = st.number_input(
            "ridge.alpha (L2 strength)",
            min_value=0.001,
            max_value=100.0,
            value=float(RIDGE_ALPHA),
            step=0.1,
            format="%.3f",
        )

    st.markdown("#### Random Forest parameters")
    if range_mode:
        rfc1, rfc2, rfc3 = st.columns(3)
        rf_n_est_min = rfc1.slider("rf.n_estimators min (trees)", min_value=50, max_value=800, value=100, step=10)
        rf_n_est_max = rfc2.slider("rf.n_estimators max (trees)", min_value=50, max_value=800, value=300, step=10)
        rf_n_est_n = rfc3.slider("rf.n_estimators #values", min_value=1, max_value=6, value=3)

        rfd1, rfd2, rfd3 = st.columns(3)
        rf_depth_min = rfd1.slider("rf.max_depth min", min_value=2, max_value=20, value=4)
        rf_depth_max = rfd2.slider("rf.max_depth max", min_value=2, max_value=20, value=10)
        rf_depth_n = rfd3.slider("rf.max_depth #values", min_value=1, max_value=6, value=3)

        rfl1, rfl2, rfl3 = st.columns(3)
        rf_leaf_min = rfl1.slider("rf.min_samples_leaf min", min_value=1, max_value=50, value=5)
        rf_leaf_max = rfl2.slider("rf.min_samples_leaf max", min_value=1, max_value=50, value=25)
        rf_leaf_n = rfl3.slider("rf.min_samples_leaf #values", min_value=1, max_value=6, value=3)
    else:
        rf_c1, rf_c2, rf_c3 = st.columns(3)
        rf_n_estimators = rf_c1.slider(
            "rf.n_estimators (number of trees)",
            min_value=50,
            max_value=800,
            value=int(RF_PARAMS["n_estimators"]),
            step=10,
        )
        rf_max_depth = rf_c2.slider(
            "rf.max_depth (tree depth)",
            min_value=2,
            max_value=20,
            value=int(RF_PARAMS["max_depth"]),
        )
        rf_min_leaf = rf_c3.slider(
            "rf.min_samples_leaf (leaf size)",
            min_value=1,
            max_value=50,
            value=int(RF_PARAMS["min_samples_leaf"]),
        )

    st.markdown("#### XGBoost parameters")
    if range_mode:
        x1, x2, x3 = st.columns(3)
        xgb_n_est_min = x1.slider("xgboost.n_estimators min (trees)", min_value=50, max_value=1000, value=100, step=25)
        xgb_n_est_max = x2.slider("xgboost.n_estimators max (trees)", min_value=50, max_value=1000, value=400, step=25)
        xgb_n_est_n = x3.slider("xgboost.n_estimators #values", min_value=1, max_value=6, value=3)

        xd1, xd2, xd3 = st.columns(3)
        xgb_depth_min = xd1.slider("xgboost.max_depth min", min_value=2, max_value=10, value=3)
        xgb_depth_max = xd2.slider("xgboost.max_depth max", min_value=2, max_value=10, value=6)
        xgb_depth_n = xd3.slider("xgboost.max_depth #values", min_value=1, max_value=6, value=3)

        xl1, xl2, xl3 = st.columns(3)
        xgb_lr_min = xl1.number_input("xgboost.learning_rate min", min_value=0.01, max_value=0.5, value=0.03, step=0.01, format="%.2f")
        xgb_lr_max = xl2.number_input("xgboost.learning_rate max", min_value=0.01, max_value=0.5, value=0.15, step=0.01, format="%.2f")
        xgb_lr_n = xl3.slider("xgboost.learning_rate #values", min_value=1, max_value=6, value=3)
    else:
        xgb_c1, xgb_c2, xgb_c3 = st.columns(3)
        xgb_n_estimators = xgb_c1.slider(
            "xgboost.n_estimators (number of trees)",
            min_value=50,
            max_value=1000,
            value=int(XGB_PARAMS["n_estimators"]),
            step=25,
        )
        xgb_max_depth = xgb_c2.slider(
            "xgboost.max_depth (tree depth)",
            min_value=2,
            max_value=10,
            value=int(XGB_PARAMS["max_depth"]),
        )
        xgb_lr = xgb_c3.number_input(
            "xgboost.learning_rate",
            min_value=0.01,
            max_value=0.5,
            value=float(XGB_PARAMS["learning_rate"]),
            step=0.01,
            format="%.2f",
        )

if range_mode:
    ridge_candidates = [{"alpha": a} for a in _float_grid(ridge_alpha_min, ridge_alpha_max, ridge_alpha_n)]

    rf_candidates = [
        {
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_leaf": leaf,
        }
        for n_est, depth, leaf in product(
            _int_grid(rf_n_est_min, rf_n_est_max, rf_n_est_n),
            _int_grid(rf_depth_min, rf_depth_max, rf_depth_n),
            _int_grid(rf_leaf_min, rf_leaf_max, rf_leaf_n),
        )
    ]

    xgb_candidates = [
        {
            "n_estimators": n_est,
            "max_depth": depth,
            "learning_rate": lr,
        }
        for n_est, depth, lr in product(
            _int_grid(xgb_n_est_min, xgb_n_est_max, xgb_n_est_n),
            _int_grid(xgb_depth_min, xgb_depth_max, xgb_depth_n),
            _float_grid(xgb_lr_min, xgb_lr_max, xgb_lr_n),
        )
    ]

    param_candidates = {
        "ridge": _downsample_trials(ridge_candidates),
        "rf": _downsample_trials(rf_candidates),
        "xgboost": _downsample_trials(xgb_candidates),
    }
else:
    param_candidates = {
        "ridge": [{"alpha": ridge_alpha}],
        "rf": [{
            "n_estimators": rf_n_estimators,
            "max_depth": rf_max_depth,
            "min_samples_leaf": rf_min_leaf,
        }],
        "xgboost": [{
            "n_estimators": xgb_n_estimators,
            "max_depth": xgb_max_depth,
            "learning_rate": xgb_lr,
        }],
    }

if range_mode:
    st.caption(
        f"Grid mode enabled. Max {MAX_GRID_TRIALS_PER_MODEL} parameter trials per model after downsampling."
    )

if st.button("Run selected models", type="primary", use_container_width=True):
    if not selected_models:
        st.warning("Select at least one model.")
    else:
        errors = {}
        runs = {}
        timeline_rows = []

        with st.spinner("Preparing data and running models..."):
            try:
                feature_matrix = load_feature_matrix(refresh_data=refresh_data)
            except Exception as exc:
                st.error(f"Data preparation failed: {exc}")
                st.stop()

            progress = st.progress(0)
            status_box = st.empty()
            status_lines: list[str] = []

            for i, model_name in enumerate(selected_models):
                model_start = pd.Timestamp.utcnow()
                candidates = param_candidates[model_name]
                status_lines.append(
                    f"{model_start.strftime('%Y-%m-%d %H:%M:%S')} UTC - {model_name} started ({len(candidates)} trial(s))"
                )
                status_box.info("\n".join(status_lines[-8:]))

                best_run = None
                best_score = (-1e12, -1e12)
                trial_errors = []

                for trial_idx, params in enumerate(candidates, start=1):
                    try:
                        run = run_single_model(
                            feature_df=feature_matrix,
                            model_name=model_name,
                            model_params=params,
                            top_k=top_k,
                            cost_bps=cost_bps,
                            slippage_bps=slippage_bps,
                            retrain_every=retrain_every,
                            initial_train_years=initial_train_years,
                            tickers=TICKERS,
                        )
                        run["trial_count"] = len(candidates)
                        run["trial_index"] = trial_idx

                        score = _score_run(run)
                        if score > best_score:
                            best_score = score
                            best_run = run
                    except Exception as exc:
                        trial_errors.append(f"trial {trial_idx}: {exc}")

                model_end = pd.Timestamp.utcnow()
                duration = (model_end - model_start).total_seconds()

                if best_run is None:
                    errors[model_name] = "; ".join(trial_errors[:3]) if trial_errors else "All trials failed."
                    timeline_rows.append(
                        {
                            "Model": model_name,
                            "Started (UTC)": model_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "Finished (UTC)": model_end.strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration (s)": duration,
                            "Trials": len(candidates),
                            "Status": "failed",
                        }
                    )
                else:
                    runs[model_name] = best_run
                    timeline_rows.append(
                        {
                            "Model": model_name,
                            "Started (UTC)": model_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "Finished (UTC)": model_end.strftime("%Y-%m-%d %H:%M:%S"),
                            "Duration (s)": duration,
                            "Trials": len(candidates),
                            "Status": "success",
                        }
                    )

                status_lines.append(
                    f"{model_end.strftime('%Y-%m-%d %H:%M:%S')} UTC - {model_name} finished in {duration:.1f}s"
                )
                status_box.info("\n".join(status_lines[-8:]))
                progress.progress((i + 1) / len(selected_models))

        st.session_state["run_timeline"] = pd.DataFrame(timeline_rows)

        if errors:
            for model_name, msg in errors.items():
                st.error(f"{model_name} failed: {msg}")

        if runs:
            summary = summarize_runs(runs)
            st.session_state["model_runs"] = runs
            st.session_state["model_summary"] = summary
            st.session_state["feature_matrix"] = feature_matrix

            best_model = summary.iloc[0]["Model"]
            st.session_state["active_model"] = best_model

            cache_dir = Path(__file__).resolve().parent.parent / "cache"
            persist_run_outputs(cache_dir, runs[best_model], feature_matrix)
            st.success(f"Run complete. Published '{best_model}' to app cache for other pages.")

runs = st.session_state.get("model_runs", {})
summary = st.session_state.get("model_summary")
feature_matrix = st.session_state.get("feature_matrix")
run_timeline = st.session_state.get("run_timeline")

if isinstance(run_timeline, pd.DataFrame) and not run_timeline.empty:
    st.subheader("Run timeline")
    st.dataframe(
        run_timeline.style.format({"Duration (s)": "{:.1f}"}),
        use_container_width=True,
        hide_index=True,
    )

if not runs:
    st.info("Run at least one model to see comparison results.")
    st.stop()

st.subheader("Model comparison")
if isinstance(summary, pd.DataFrame) and not summary.empty:
    st.dataframe(
        summary.style.format(
            {
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "IC": "{:.3f}",
                "Hit Rate": "{:.1%}",
                "CAGR": "{:.1%}",
                "Sharpe Ratio": "{:.2f}",
                "Max Drawdown": "{:.1%}",
                "Profit Factor": "{:.2f}",
                "Avg Turnover": "{:.1%}",
                "Run Time (s)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

if len(runs) > 1:
    st.subheader("Equity curve comparison")
    curves = {}
    for model_name, run in runs.items():
        bt = run.get("backtest")
        if bt is not None and not bt.empty:
            curves[model_name] = bt["cumulative_return"]

    if curves:
        curve_df = pd.concat(curves, axis=1)
        st.line_chart(curve_df)

model_names = list(runs.keys())
active_model = st.session_state.get("active_model")
default_idx = model_names.index(active_model) if active_model in model_names else 0
inspect_model = st.selectbox("Inspect model", model_names, index=default_idx)
run = runs[inspect_model]

if st.button("Use this model in other pages", use_container_width=True):
    st.session_state["active_model"] = inspect_model
    if feature_matrix is not None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
        persist_run_outputs(cache_dir, run, feature_matrix)
    st.success(f"'{inspect_model}' is now active for Signals/Backtest/Explain/Data pages.")

c1, c2, c3, c4 = st.columns(4)
trade = run["trading_metrics"]
c1.metric("CAGR", f"{trade['CAGR']:.1%}" if pd.notna(trade["CAGR"]) else "n/a")
c2.metric("Sharpe", f"{trade['Sharpe Ratio']:.2f}" if pd.notna(trade["Sharpe Ratio"]) else "n/a")
c3.metric("Max Drawdown", f"{trade['Max Drawdown']:.1%}" if pd.notna(trade["Max Drawdown"]) else "n/a")
c4.metric("Profit Factor", f"{trade['Profit Factor']:.2f}" if pd.notna(trade["Profit Factor"]) else "n/a")

st.subheader("Latest predicted returns")
st.dataframe(
    run["latest_predictions"].style.format({"predicted_ret": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Feature importance")
importance = run.get("feature_importance")
if importance is not None and not importance.empty:
    fig = plot_feature_importance(importance, top_n=15, title=f"Feature Importance - {inspect_model}")
    st.pyplot(fig)
else:
    st.info("No feature importance available for this model run.")

with st.expander("Parameters used"):
    st.json(run.get("model_params", {}))
    st.caption(f"Trial {run.get('trial_index', 1)} of {run.get('trial_count', 1)}")
