"""
Interactive model runner for the Streamlit app.

Lets users:
- choose one model or multiple models
- tune model/backtest parameters
- run and compare results side-by-side
- publish one run to other app pages
"""

from pathlib import Path

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

st.set_page_config(page_title="Model Lab", page_icon="🧪", layout="wide")
st.title("🧪 Model Lab")
st.caption("Run one or more models, compare performance, then publish a selected run to the rest of the app.")

model_options = ["ridge", "rf", "xgboost"]

selected_models = st.multiselect(
    "Models to run",
    options=model_options,
    default=model_options,
    help="Pick one model or multiple models for side-by-side comparison.",
)

with st.expander("Portfolio and backtest settings", expanded=True):
    c1, c2, c3 = st.columns(3)
    top_k = c1.slider("Top K long positions", min_value=1, max_value=len(TICKERS), value=TOP_K)
    initial_train_years = c2.slider(
        "Initial training window (years)",
        min_value=1,
        max_value=6,
        value=INITIAL_TRAIN_YEARS,
    )
    retrain_every = c3.slider(
        "Retrain every N weeks",
        min_value=4,
        max_value=26,
        value=RETRAIN_EVERY_WEEKS,
    )

    c4, c5 = st.columns(2)
    cost_bps = c4.number_input(
        "Transaction cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=float(TRANSACTION_COST_BPS),
        step=1.0,
    )
    slippage_bps = c5.number_input(
        "Slippage (bps)",
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

with st.expander("Model parameters", expanded=True):
    ridge_alpha = st.number_input(
        "Ridge alpha",
        min_value=0.001,
        max_value=100.0,
        value=float(RIDGE_ALPHA),
        step=0.1,
        format="%.3f",
    )

    rf_c1, rf_c2, rf_c3 = st.columns(3)
    rf_n_estimators = rf_c1.slider(
        "RF n_estimators",
        min_value=50,
        max_value=600,
        value=int(RF_PARAMS["n_estimators"]),
        step=10,
    )
    rf_max_depth = rf_c2.slider(
        "RF max_depth",
        min_value=2,
        max_value=20,
        value=int(RF_PARAMS["max_depth"]),
    )
    rf_min_leaf = rf_c3.slider(
        "RF min_samples_leaf",
        min_value=1,
        max_value=50,
        value=int(RF_PARAMS["min_samples_leaf"]),
    )

    xgb_c1, xgb_c2, xgb_c3 = st.columns(3)
    xgb_n_estimators = xgb_c1.slider(
        "XGB n_estimators",
        min_value=50,
        max_value=800,
        value=int(XGB_PARAMS["n_estimators"]),
        step=25,
    )
    xgb_max_depth = xgb_c2.slider(
        "XGB max_depth",
        min_value=2,
        max_value=10,
        value=int(XGB_PARAMS["max_depth"]),
    )
    xgb_lr = xgb_c3.number_input(
        "XGB learning_rate",
        min_value=0.01,
        max_value=0.5,
        value=float(XGB_PARAMS["learning_rate"]),
        step=0.01,
        format="%.2f",
    )

param_map = {
    "ridge": {"alpha": ridge_alpha},
    "rf": {
        "n_estimators": rf_n_estimators,
        "max_depth": rf_max_depth,
        "min_samples_leaf": rf_min_leaf,
    },
    "xgboost": {
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "learning_rate": xgb_lr,
    },
}

if st.button("Run selected models", type="primary", use_container_width=True):
    if not selected_models:
        st.warning("Select at least one model.")
    else:
        errors = {}
        runs = {}

        with st.spinner("Preparing data and running models..."):
            try:
                feature_matrix = load_feature_matrix(refresh_data=refresh_data)
            except Exception as exc:
                st.error(f"Data preparation failed: {exc}")
                st.stop()

            progress = st.progress(0)
            for i, model_name in enumerate(selected_models):
                try:
                    runs[model_name] = run_single_model(
                        feature_df=feature_matrix,
                        model_name=model_name,
                        model_params=param_map[model_name],
                        top_k=top_k,
                        cost_bps=cost_bps,
                        slippage_bps=slippage_bps,
                        retrain_every=retrain_every,
                        initial_train_years=initial_train_years,
                        tickers=TICKERS,
                    )
                except Exception as exc:
                    errors[model_name] = str(exc)
                progress.progress((i + 1) / len(selected_models))

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
