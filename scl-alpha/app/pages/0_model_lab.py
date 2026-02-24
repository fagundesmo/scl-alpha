"""Interactive model runner for the Streamlit app."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st

from src.app_runner import (
    build_daily_run_outputs,
    load_feature_matrix,
    persist_run_outputs,
    run_single_model,
    score_model_params,
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

MODEL_LABELS = {
    "ridge": "Ridge",
    "rf": "Random Forest",
    "xgboost": "XGBoost",
}

PARAM_INFO = {
    "ridge": {
        "alpha": {
            "label": "Alpha (L2 strength)",
            "help": "Higher alpha increases regularization and makes predictions more conservative.",
        },
    },
    "rf": {
        "n_estimators": {
            "label": "Trees",
            "help": "Number of trees. More trees can improve stability but increase runtime.",
        },
        "max_depth": {
            "label": "Max depth",
            "help": "Maximum tree depth. Higher values capture more complexity but can overfit.",
        },
        "min_samples_leaf": {
            "label": "Min leaf size",
            "help": "Minimum samples in each leaf. Higher values smooth predictions.",
        },
    },
    "xgboost": {
        "n_estimators": {
            "label": "Trees",
            "help": "Number of boosting rounds. More rounds can improve fit but increase runtime.",
        },
        "max_depth": {
            "label": "Max depth",
            "help": "Maximum tree depth per boosting round.",
        },
        "learning_rate": {
            "label": "Learning rate",
            "help": "Step size per boosting round. Lower values are more stable but often need more rounds.",
        },
    },
}

SEARCH_PRESETS = {
    "Fast": {
        "sample_fraction": 0.25,
        "cv_folds": 2,
        "max_trials": 4,
        "time_limit_s": 35,
        "rows_per_ticker": 160,
    },
    "Balanced": {
        "sample_fraction": 0.35,
        "cv_folds": 3,
        "max_trials": 6,
        "time_limit_s": 60,
        "rows_per_ticker": 220,
    },
    "Thorough": {
        "sample_fraction": 0.55,
        "cv_folds": 4,
        "max_trials": 10,
        "time_limit_s": 120,
        "rows_per_ticker": 320,
    },
}


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


def _downsample_trials(candidates: list[dict], limit: int) -> list[dict]:
    if len(candidates) <= limit:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, limit, dtype=int)
    return [candidates[int(i)] for i in idx]


def _score_metrics(ml_metrics: dict) -> tuple[float, float, float]:
    ic = ml_metrics.get("IC", np.nan)
    hit = ml_metrics.get("Hit Rate", np.nan)
    rmse = ml_metrics.get("RMSE", np.nan)

    ic_score = float(ic) if pd.notna(ic) else -1e9
    hit_score = float(hit) if pd.notna(hit) else -1e9
    rmse_score = -float(rmse) if pd.notna(rmse) else -1e9
    return ic_score, hit_score, rmse_score


def _model_label(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def _params_table(model_name: str, params: dict) -> pd.DataFrame:
    info_map = PARAM_INFO.get(model_name, {})
    rows = []
    for key, value in params.items():
        meta = info_map.get(key, {})
        rows.append(
            {
                "Parameter": meta.get("label", key),
                "Value": value,
                "What it does": meta.get("help", ""),
            }
        )
    return pd.DataFrame(rows)


def _render_table(df: pd.DataFrame, formats: dict | None = None, hide_index: bool = True) -> None:
    if isinstance(df, pd.DataFrame) and not df.empty:
        style = df.style.format(formats or {}) if formats else df
        st.dataframe(style, use_container_width=True, hide_index=hide_index)
    else:
        st.info("No data available for this section yet.")


st.set_page_config(page_title="Model Lab", page_icon="🧪", layout="wide")
st.title("🧪 Model Lab")
st.caption("Choose models, tune settings, run experiments, and compare results.")

model_options = ["ridge", "rf", "xgboost"]
selected_models = st.multiselect(
    "Models",
    options=model_options,
    default=model_options,
    format_func=_model_label,
    help="Pick one or more models to run and compare.",
)

with st.expander("Portfolio & backtest", expanded=True):
    c1, c2, c3 = st.columns(3)
    top_k = c1.slider(
        "Long positions",
        min_value=1,
        max_value=len(TICKERS),
        value=TOP_K,
        help="How many stocks to hold each rebalance period.",
    )
    initial_train_years = c2.slider(
        "Train years",
        min_value=1,
        max_value=6,
        value=INITIAL_TRAIN_YEARS,
        help="History length used before first live prediction.",
    )
    retrain_every = c3.slider(
        "Retrain (weeks)",
        min_value=4,
        max_value=26,
        value=RETRAIN_EVERY_WEEKS,
        help="How often the model retrains during walk-forward backtest.",
    )

    c4, c5 = st.columns(2)
    cost_bps = c4.number_input(
        "Trading cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=float(TRANSACTION_COST_BPS),
        step=1.0,
        help="One-way trading cost in basis points (1 bps = 0.01%).",
    )
    slippage_bps = c5.number_input(
        "Slippage (bps)",
        min_value=0.0,
        max_value=100.0,
        value=float(SLIPPAGE_BPS),
        step=1.0,
        help="Extra execution slippage in basis points.",
    )

refresh_data = st.checkbox(
    "Refresh data",
    value=False,
    help="Enable to pull fresh API data. Disable to use local cache and run faster.",
)

range_mode = st.checkbox(
    "Use parameter ranges",
    value=False,
    help="Test multiple parameter combinations per model and keep the best one.",
)

if range_mode:
    preset_name = st.selectbox(
        "Search preset",
        options=list(SEARCH_PRESETS.keys()),
        index=1,
        help="Fast = quickest, Balanced = recommended default, Thorough = slower but more exhaustive.",
    )
    preset = SEARCH_PRESETS[preset_name]

    gs1, gs2, gs3, gs4 = st.columns(4)
    grid_sample_fraction = gs1.slider(
        "Data sample",
        min_value=0.20,
        max_value=1.00,
        value=float(preset["sample_fraction"]),
        step=0.05,
        help="Fraction of recent data used during search scoring. Lower is faster.",
    )
    grid_cv_folds = gs2.slider(
        "CV folds",
        min_value=2,
        max_value=5,
        value=int(preset["cv_folds"]),
        help="Time-series cross-validation folds used in parameter search.",
    )
    grid_max_trials = gs3.slider(
        "Max trials/model",
        min_value=3,
        max_value=16,
        value=int(preset["max_trials"]),
        help="Maximum parameter combinations evaluated per model.",
    )
    grid_time_limit_s = gs4.slider(
        "Time limit/model (s)",
        min_value=15,
        max_value=240,
        value=int(preset["time_limit_s"]),
        step=5,
        help="Stops search early when this time limit is reached for a model.",
    )

    grid_rows_per_ticker = st.slider(
        "Rows per ticker",
        min_value=100,
        max_value=500,
        value=int(preset["rows_per_ticker"]),
        step=20,
        help="Maximum recent rows per ticker used during search scoring.",
    )

    st.caption(
        f"Preset: {preset_name} | sample={grid_sample_fraction:.2f}, CV={grid_cv_folds}, trials={grid_max_trials}, limit={grid_time_limit_s}s"
    )
else:
    grid_sample_fraction = 1.0
    grid_cv_folds = 3
    grid_max_trials = 1
    grid_time_limit_s = 0
    grid_rows_per_ticker = 300

with st.expander("Model parameters", expanded=True):
    st.markdown("#### Ridge")
    if range_mode:
        rc1, rc2, rc3 = st.columns(3)
        ridge_alpha_min = rc1.number_input(
            "Alpha min",
            min_value=0.001,
            max_value=100.0,
            value=0.1,
            step=0.1,
            format="%.3f",
            help=PARAM_INFO["ridge"]["alpha"]["help"],
        )
        ridge_alpha_max = rc2.number_input(
            "Alpha max",
            min_value=0.001,
            max_value=100.0,
            value=5.0,
            step=0.1,
            format="%.3f",
            help=PARAM_INFO["ridge"]["alpha"]["help"],
        )
        ridge_alpha_n = rc3.slider(
            "Alpha points",
            min_value=1,
            max_value=8,
            value=4,
            help="How many alpha values to test between min and max.",
        )
    else:
        ridge_alpha = st.number_input(
            "Alpha",
            min_value=0.001,
            max_value=100.0,
            value=float(RIDGE_ALPHA),
            step=0.1,
            format="%.3f",
            help=PARAM_INFO["ridge"]["alpha"]["help"],
        )

    st.markdown("#### Random Forest")
    if range_mode:
        rfc1, rfc2, rfc3 = st.columns(3)
        rf_n_est_min = rfc1.slider(
            "Trees min",
            min_value=50,
            max_value=800,
            value=100,
            step=10,
            help=PARAM_INFO["rf"]["n_estimators"]["help"],
        )
        rf_n_est_max = rfc2.slider(
            "Trees max",
            min_value=50,
            max_value=800,
            value=300,
            step=10,
            help=PARAM_INFO["rf"]["n_estimators"]["help"],
        )
        rf_n_est_n = rfc3.slider(
            "Trees points",
            min_value=1,
            max_value=6,
            value=3,
            help="How many tree-count values to test.",
            key="rf_trees_points",
        )

        rfd1, rfd2, rfd3 = st.columns(3)
        rf_depth_min = rfd1.slider(
            "Depth min",
            min_value=2,
            max_value=20,
            value=4,
            help=PARAM_INFO["rf"]["max_depth"]["help"],
        )
        rf_depth_max = rfd2.slider(
            "Depth max",
            min_value=2,
            max_value=20,
            value=10,
            help=PARAM_INFO["rf"]["max_depth"]["help"],
        )
        rf_depth_n = rfd3.slider(
            "Depth points",
            min_value=1,
            max_value=6,
            value=3,
            help="How many depth values to test.",
            key="rf_depth_points",
        )

        rfl1, rfl2, rfl3 = st.columns(3)
        rf_leaf_min = rfl1.slider(
            "Leaf min",
            min_value=1,
            max_value=50,
            value=5,
            help=PARAM_INFO["rf"]["min_samples_leaf"]["help"],
        )
        rf_leaf_max = rfl2.slider(
            "Leaf max",
            min_value=1,
            max_value=50,
            value=25,
            help=PARAM_INFO["rf"]["min_samples_leaf"]["help"],
        )
        rf_leaf_n = rfl3.slider(
            "Leaf points",
            min_value=1,
            max_value=6,
            value=3,
            help="How many leaf-size values to test.",
        )
    else:
        rf_c1, rf_c2, rf_c3 = st.columns(3)
        rf_n_estimators = rf_c1.slider(
            "Trees",
            min_value=50,
            max_value=800,
            value=int(RF_PARAMS["n_estimators"]),
            step=10,
            help=PARAM_INFO["rf"]["n_estimators"]["help"],
        )
        rf_max_depth = rf_c2.slider(
            "Max depth",
            min_value=2,
            max_value=20,
            value=int(RF_PARAMS["max_depth"]),
            help=PARAM_INFO["rf"]["max_depth"]["help"],
        )
        rf_min_leaf = rf_c3.slider(
            "Min leaf size",
            min_value=1,
            max_value=50,
            value=int(RF_PARAMS["min_samples_leaf"]),
            help=PARAM_INFO["rf"]["min_samples_leaf"]["help"],
        )

    st.markdown("#### XGBoost")
    if range_mode:
        x1, x2, x3 = st.columns(3)
        xgb_n_est_min = x1.slider(
            "Trees min",
            min_value=50,
            max_value=1000,
            value=100,
            step=25,
            help=PARAM_INFO["xgboost"]["n_estimators"]["help"],
        )
        xgb_n_est_max = x2.slider(
            "Trees max",
            min_value=50,
            max_value=1000,
            value=400,
            step=25,
            help=PARAM_INFO["xgboost"]["n_estimators"]["help"],
        )
        xgb_n_est_n = x3.slider(
            "Trees points",
            min_value=1,
            max_value=6,
            value=3,
            help="How many tree-count values to test.",
            key="xgb_trees_points",
        )

        xd1, xd2, xd3 = st.columns(3)
        xgb_depth_min = xd1.slider(
            "Depth min",
            min_value=2,
            max_value=10,
            value=3,
            help=PARAM_INFO["xgboost"]["max_depth"]["help"],
        )
        xgb_depth_max = xd2.slider(
            "Depth max",
            min_value=2,
            max_value=10,
            value=6,
            help=PARAM_INFO["xgboost"]["max_depth"]["help"],
        )
        xgb_depth_n = xd3.slider(
            "Depth points",
            min_value=1,
            max_value=6,
            value=3,
            help="How many depth values to test.",
            key="xgb_depth_points",
        )

        xl1, xl2, xl3 = st.columns(3)
        xgb_lr_min = xl1.number_input(
            "Rate min",
            min_value=0.01,
            max_value=0.5,
            value=0.03,
            step=0.01,
            format="%.2f",
            help=PARAM_INFO["xgboost"]["learning_rate"]["help"],
        )
        xgb_lr_max = xl2.number_input(
            "Rate max",
            min_value=0.01,
            max_value=0.5,
            value=0.15,
            step=0.01,
            format="%.2f",
            help=PARAM_INFO["xgboost"]["learning_rate"]["help"],
        )
        xgb_lr_n = xl3.slider(
            "Rate points",
            min_value=1,
            max_value=6,
            value=3,
            help="How many learning-rate values to test.",
        )
    else:
        xgb_c1, xgb_c2, xgb_c3 = st.columns(3)
        xgb_n_estimators = xgb_c1.slider(
            "Trees",
            min_value=50,
            max_value=1000,
            value=int(XGB_PARAMS["n_estimators"]),
            step=25,
            help=PARAM_INFO["xgboost"]["n_estimators"]["help"],
        )
        xgb_max_depth = xgb_c2.slider(
            "Max depth",
            min_value=2,
            max_value=10,
            value=int(XGB_PARAMS["max_depth"]),
            help=PARAM_INFO["xgboost"]["max_depth"]["help"],
        )
        xgb_lr = xgb_c3.number_input(
            "Learning rate",
            min_value=0.01,
            max_value=0.5,
            value=float(XGB_PARAMS["learning_rate"]),
            step=0.01,
            format="%.2f",
            help=PARAM_INFO["xgboost"]["learning_rate"]["help"],
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
        "ridge": _downsample_trials(ridge_candidates, grid_max_trials),
        "rf": _downsample_trials(rf_candidates, grid_max_trials),
        "xgboost": _downsample_trials(xgb_candidates, grid_max_trials),
    }
else:
    param_candidates = {
        "ridge": [{"alpha": ridge_alpha}],
        "rf": [
            {
                "n_estimators": rf_n_estimators,
                "max_depth": rf_max_depth,
                "min_samples_leaf": rf_min_leaf,
            }
        ],
        "xgboost": [
            {
                "n_estimators": xgb_n_estimators,
                "max_depth": xgb_max_depth,
                "learning_rate": xgb_lr,
            }
        ],
    }

if range_mode:
    st.caption(
        "Search mode: each model uses time-series CV for fast scoring, then runs one full backtest with the best settings."
    )

if st.button("Run selected models", type="primary", use_container_width=True):
    if not selected_models:
        st.warning("Select at least one model.")
    else:
        errors: dict[str, str] = {}
        runs: dict[str, dict] = {}
        timeline_rows: list[dict] = []

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
                model_title = _model_label(model_name)
                candidates = param_candidates[model_name]
                status_lines.append(
                    f"{model_start.strftime('%Y-%m-%d %H:%M:%S')} UTC - {model_title} started ({len(candidates)} trial(s))"
                )
                status_box.info("\n".join(status_lines[-8:]))

                best_params = None
                best_score = (-1e12, -1e12, -1e12)
                best_quick_metrics = None
                trial_errors = []
                tried = 0

                search_start = time.time()
                if range_mode and len(candidates) > 1:
                    for trial_idx, params in enumerate(candidates, start=1):
                        if (time.time() - search_start) > grid_time_limit_s and trial_idx > 2:
                            break
                        tried += 1
                        try:
                            quick = score_model_params(
                                feature_df=feature_matrix,
                                model_name=model_name,
                                model_params=params,
                                tickers=TICKERS,
                                sample_fraction=grid_sample_fraction,
                                cv_splits=grid_cv_folds,
                                max_rows_per_ticker=grid_rows_per_ticker,
                            )
                            score = _score_metrics(quick["ml_metrics"])
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_quick_metrics = quick["ml_metrics"]
                        except Exception as exc:
                            trial_errors.append(f"trial {trial_idx}: {exc}")
                else:
                    best_params = candidates[0]
                    tried = 1
                search_duration = time.time() - search_start

                full_start = time.time()
                if best_params is None:
                    errors[model_name] = "; ".join(trial_errors[:3]) if trial_errors else "All trials failed."
                    full_duration = 0.0
                    model_end = pd.Timestamp.utcnow()
                    timeline_rows.append(
                        {
                            "Model": model_title,
                            "Started (UTC)": model_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "Finished (UTC)": model_end.strftime("%Y-%m-%d %H:%M:%S"),
                            "Search (s)": search_duration,
                            "Full run (s)": full_duration,
                            "Total (s)": search_duration + full_duration,
                            "Trials": tried,
                            "Status": "failed",
                        }
                    )
                else:
                    try:
                        run = run_single_model(
                            feature_df=feature_matrix,
                            model_name=model_name,
                            model_params=best_params,
                            top_k=top_k,
                            cost_bps=cost_bps,
                            slippage_bps=slippage_bps,
                            retrain_every=retrain_every,
                            initial_train_years=initial_train_years,
                            tickers=TICKERS,
                        )
                        run["trial_count"] = tried
                        run["search_ml_metrics"] = best_quick_metrics
                        run["search_cv_folds"] = grid_cv_folds if range_mode else 1
                        run["search_sample_fraction"] = grid_sample_fraction
                        run["search_rows_per_ticker"] = grid_rows_per_ticker
                        run["search_preset"] = preset_name if range_mode else "single"
                        run["completed_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                        runs[model_name] = run
                    except Exception as exc:
                        errors[model_name] = str(exc)

                    full_duration = time.time() - full_start
                    model_end = pd.Timestamp.utcnow()
                    timeline_rows.append(
                        {
                            "Model": model_title,
                            "Started (UTC)": model_start.strftime("%Y-%m-%d %H:%M:%S"),
                            "Finished (UTC)": model_end.strftime("%Y-%m-%d %H:%M:%S"),
                            "Search (s)": search_duration,
                            "Full run (s)": full_duration,
                            "Total (s)": search_duration + full_duration,
                            "Trials": tried,
                            "Status": "success" if model_name in runs else "failed",
                        }
                    )

                status_lines.append(
                    f"{model_end.strftime('%Y-%m-%d %H:%M:%S')} UTC - {model_title} finished"
                )
                status_box.info("\n".join(status_lines[-8:]))
                progress.progress((i + 1) / len(selected_models))

        st.session_state["run_timeline"] = pd.DataFrame(timeline_rows)

        if errors:
            for model_name, msg in errors.items():
                st.error(f"{_model_label(model_name)} failed: {msg}")

        if runs:
            summary = summarize_runs(runs)
            st.session_state["model_runs"] = runs
            st.session_state["model_summary"] = summary
            st.session_state["feature_matrix"] = feature_matrix

            best_model = summary.iloc[0]["Model"]
            st.session_state["active_model"] = best_model

            cache_dir = Path(__file__).resolve().parent.parent / "cache"
            persist_run_outputs(cache_dir, runs[best_model], feature_matrix)
            st.success(f"Run complete. Published '{_model_label(best_model)}' to app cache for other pages.")

runs = st.session_state.get("model_runs", {})
summary = st.session_state.get("model_summary")
feature_matrix = st.session_state.get("feature_matrix")
run_timeline = st.session_state.get("run_timeline")

if isinstance(run_timeline, pd.DataFrame) and not run_timeline.empty:
    st.subheader("Run timeline")
    st.dataframe(
        run_timeline.style.format(
            {
                "Search (s)": "{:.1f}",
                "Full run (s)": "{:.1f}",
                "Total (s)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

if not runs:
    st.info("Run at least one model to see comparison results.")
    st.stop()

st.subheader("Model comparison")
if isinstance(summary, pd.DataFrame) and not summary.empty:
    summary_view = summary.copy()
    summary_view["Model"] = summary_view["Model"].map(_model_label)
    st.dataframe(
        summary_view.style.format(
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
    curves: dict[str, pd.Series] = {}
    for model_name, run in runs.items():
        bt = run.get("backtest")
        if bt is not None and not bt.empty:
            curves[_model_label(model_name)] = bt["cumulative_return"]

    if curves:
        curve_df = pd.concat(curves, axis=1)
        st.line_chart(curve_df)

model_names = list(runs.keys())
active_model = st.session_state.get("active_model")
default_idx = model_names.index(active_model) if active_model in model_names else 0
inspect_model = st.selectbox(
    "Inspect model",
    model_names,
    index=default_idx,
    format_func=_model_label,
    help="Choose which completed model run to inspect in detail.",
)
run = runs[inspect_model]

if st.button("Use this model in other pages", use_container_width=True):
    st.session_state["active_model"] = inspect_model
    if feature_matrix is not None:
        cache_dir = Path(__file__).resolve().parent.parent / "cache"
        persist_run_outputs(cache_dir, run, feature_matrix)
    st.success(f"'{_model_label(inspect_model)}' is now active for Signals/Backtest/Explain/Data pages.")

c1, c2, c3, c4 = st.columns(4)
trade = run["trading_metrics"]
c1.metric("CAGR", f"{trade['CAGR']:.1%}" if pd.notna(trade["CAGR"]) else "n/a")
c2.metric("Sharpe", f"{trade['Sharpe Ratio']:.2f}" if pd.notna(trade["Sharpe Ratio"]) else "n/a")
c3.metric("Max drawdown", f"{trade['Max Drawdown']:.1%}" if pd.notna(trade["Max Drawdown"]) else "n/a")
c4.metric("Profit factor", f"{trade['Profit Factor']:.2f}" if pd.notna(trade["Profit Factor"]) else "n/a")

st.subheader("Daily trading run outputs")
st.caption(
    "Operational diagnostics for daily workflow: signals, orders, risk, costs, constraints, data quality, and system health."
)
daily_outputs = build_daily_run_outputs(
    run=run,
    feature_matrix=feature_matrix if isinstance(feature_matrix, pd.DataFrame) else pd.DataFrame(),
    top_k=top_k,
    cost_bps=cost_bps,
    slippage_bps=slippage_bps,
    tickers=TICKERS,
)

tab_signal, tab_orders, tab_exec, tab_pos, tab_pnl, tab_cost, tab_risk, tab_constraints, tab_data, tab_model, tab_system = st.tabs(
    [
        "Signal summary",
        "Orders generated",
        "Executed trades",
        "Positions & exposure",
        "P&L attribution",
        "Transaction costs",
        "Risk metrics",
        "Constraint checks",
        "Data quality",
        "Model health",
        "System health",
    ]
)

with tab_signal:
    _render_table(daily_outputs.get("signal_summary", pd.DataFrame()), {"Predicted Return (%)": "{:.2f}"})

with tab_orders:
    st.caption("Generated from latest signal vs prior holdings (pre-trade allocation plan).")
    _render_table(
        daily_outputs.get("orders_generated", pd.DataFrame()),
        {
            "Current Weight": "{:.1%}",
            "Target Weight": "{:.1%}",
            "Weight Change": "{:+.1%}",
            "Signal Score (%)": "{:.2f}",
            "Reference Price": "{:.2f}",
            "Estimated Cost (bps)": "{:.2f}",
            "Estimated Cost (%)": "{:.4f}",
        },
    )

with tab_exec:
    st.caption("Execution log is simulated using latest reference prices (no broker integration yet).")
    _render_table(
        daily_outputs.get("executed_trades", pd.DataFrame()),
        {
            "Current Weight": "{:.1%}",
            "Target Weight": "{:.1%}",
            "Weight Change": "{:+.1%}",
            "Fill Price": "{:.2f}",
            "Estimated Cost (bps)": "{:.2f}",
            "Estimated Cost (%)": "{:.4f}",
        },
    )

with tab_pos:
    st.markdown("**Current target positions**")
    _render_table(
        daily_outputs.get("positions_exposure", pd.DataFrame()),
        {"Weight": "{:.1%}", "Predicted Return (%)": "{:.2f}", "Last Price": "{:.2f}"},
    )
    st.markdown("**Exposure summary**")
    _render_table(
        daily_outputs.get("exposure_summary", pd.DataFrame()),
        {
            "Value": "{:.4f}",
        },
    )

with tab_pnl:
    _render_table(
        daily_outputs.get("pnl_attribution", pd.DataFrame()),
        {"Last period (%)": "{:.2f}", "Cumulative (%)": "{:.2f}"},
    )

with tab_cost:
    st.markdown("**Cost summary**")
    _render_table(daily_outputs.get("transaction_costs_summary", pd.DataFrame()), {"Value": "{:.4f}"})
    st.markdown("**Recent cost detail**")
    _render_table(
        daily_outputs.get("transaction_costs_detail", pd.DataFrame()),
        {
            "Turnover": "{:.1%}",
            "Total Cost (%)": "{:.4f}",
            "Commission (%)": "{:.4f}",
            "Slippage (%)": "{:.4f}",
        },
    )

with tab_risk:
    _render_table(daily_outputs.get("risk_metrics", pd.DataFrame()), {"Value": "{:.4f}"})

with tab_constraints:
    _render_table(daily_outputs.get("constraint_checks", pd.DataFrame()))

with tab_data:
    _render_table(daily_outputs.get("data_quality_checks", pd.DataFrame()))

with tab_model:
    _render_table(daily_outputs.get("model_health", pd.DataFrame()), {"Value": "{:.4f}"})

with tab_system:
    _render_table(daily_outputs.get("system_health", pd.DataFrame()))

st.subheader("Latest predicted returns")
st.dataframe(
    run["latest_predictions"].style.format({"predicted_ret": "{:.2f}"}),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Feature importance")
importance = run.get("feature_importance")
if importance is not None and not importance.empty:
    fig = plot_feature_importance(importance, top_n=15, title=f"Feature Importance - {_model_label(inspect_model)}")
    st.pyplot(fig)
else:
    st.info("No feature importance available for this model run.")

with st.expander("Parameters used"):
    st.dataframe(_params_table(inspect_model, run.get("model_params", {})), use_container_width=True, hide_index=True)
    st.caption(f"Trials evaluated: {run.get('trial_count', 1)}")
    if run.get("search_ml_metrics"):
        st.caption(
            f"Search mode: {run.get('search_preset', 'custom')} | folds={run.get('search_cv_folds', 1)}, "
            f"sample={run.get('search_sample_fraction', 1.0):.2f}, rows/ticker={run.get('search_rows_per_ticker', 0)}"
        )
