"""Predictions: signal analysis, time-series plots, and simple strategy simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
st.title("🔮 Predictions & Signal Analysis")
st.caption(
    "Explore model predictions over time, signal quality, and a simple long/short strategy simulation. "
    "Run **Sandbox Models** first to populate this page."
)

# -----------------------------------------------------------------------
# Guard
# -----------------------------------------------------------------------
results: dict | None = st.session_state.get("sandbox_results")
if not results:
    st.warning("No model results found. Run at least one model in **Sandbox Models** first.")
    st.stop()

MODEL_LABELS = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# -----------------------------------------------------------------------
# Model selector
# -----------------------------------------------------------------------
model_choice = st.selectbox(
    "Select model to analyse",
    options=list(results.keys()),
    format_func=lambda k: MODEL_LABELS.get(k, k),
)
res = results[model_choice]
all_preds: pd.DataFrame = res.get("all_predictions", pd.DataFrame())
test_preds: pd.DataFrame = res.get("test_predictions", pd.DataFrame())

if all_preds.empty:
    st.error("No predictions found for this model.")
    st.stop()

all_preds = all_preds.copy()
all_preds["Date"] = pd.to_datetime(all_preds["Date"])

tickers = sorted(all_preds["Ticker"].unique())

# -----------------------------------------------------------------------
# Section 1: Predicted vs actual time series (per ticker)
# -----------------------------------------------------------------------
st.header("1 · Predicted vs Actual Returns Over Time")
st.caption(
    "Shows predicted and realized next-day log return for the selected ticker across all splits. "
    "Closer alignment in the test region (2026) = better out-of-sample fit."
)

ticker_choice = st.selectbox("Select ticker", options=tickers, key="pred_ts_ticker")
split_filter = st.multiselect(
    "Show splits",
    options=["train", "val", "test"],
    default=["val", "test"],
    key="pred_ts_splits",
)

ts_df = all_preds[
    (all_preds["Ticker"] == ticker_choice) & (all_preds["split"].isin(split_filter))
].sort_values("Date")

if ts_df.empty:
    st.info("No data for this selection.")
else:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["Date"], y=ts_df["y_true"],
        name="Actual return", line=dict(color="#1f77b4", width=1.5), opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        x=ts_df["Date"], y=ts_df["y_pred"],
        name="Predicted return", line=dict(color="#ff7f0e", width=1.5, dash="dot"), opacity=0.9,
    ))
    # shade test region
    test_min = all_preds.loc[all_preds["split"] == "test", "Date"].min()
    if pd.notna(test_min):
        fig.add_vrect(
            x0=test_min, x1=all_preds["Date"].max(),
            fillcolor="lightgreen", opacity=0.08, line_width=0,
            annotation_text="Test (2026)", annotation_position="top left",
        )
    fig.update_layout(
        title=f"Predicted vs Actual Next-Day Log Return — {ticker_choice}",
        xaxis_title="Date", yaxis_title="Log Return",
        height=440, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 2: Rolling 21-day IC (signal consistency)
# -----------------------------------------------------------------------
st.header("2 · Rolling 21-Day Information Coefficient (IC)")
st.caption(
    "IC = Spearman rank correlation between predicted and actual returns on that day, "
    "computed across all tickers. Rolling 21-day window smooths noise. "
    "IC > 0.05 consistently = practically useful signal."
)

roll_splits = st.multiselect(
    "Include splits in IC chart",
    options=["train", "val", "test"],
    default=["val", "test"],
    key="ic_splits",
)

ic_df = all_preds[all_preds["split"].isin(roll_splits)].sort_values("Date")

if len(ic_df) < 30:
    st.info("Not enough data for rolling IC — include more splits.")
else:
    from scipy import stats as scipy_stats

    daily_ic = (
        ic_df.groupby("Date")
        .apply(
            lambda g: scipy_stats.spearmanr(g["y_true"], g["y_pred"]).statistic
            if len(g) >= 3 else np.nan,
            include_groups=False,
        )
        .rename("IC")
        .reset_index()
    )
    daily_ic["IC_21d"] = daily_ic["IC"].rolling(21, min_periods=10).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_ic["Date"], y=daily_ic["IC"],
        name="Daily IC", line=dict(color="#aec7e8", width=0.8), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=daily_ic["Date"], y=daily_ic["IC_21d"],
        name="21-day rolling IC", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.6)
    fig.add_hline(y=0.05, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="IC=0.05 (practical threshold)", annotation_position="top right")
    test_min = ic_df.loc[ic_df["split"] == "test", "Date"].min() if "test" in roll_splits else None
    if pd.notna(test_min) if test_min is not None else False:
        fig.add_vrect(
            x0=test_min, x1=ic_df["Date"].max(),
            fillcolor="lightgreen", opacity=0.08, line_width=0,
            annotation_text="Test (2026)", annotation_position="top left",
        )
    fig.update_layout(
        title=f"Rolling 21-Day IC — {MODEL_LABELS.get(model_choice, model_choice)}",
        xaxis_title="Date", yaxis_title="IC (Spearman)",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 3: Per-ticker IC table (test set)
# -----------------------------------------------------------------------
st.header("3 · Per-Ticker Signal Quality — Test Set (2026)")
st.caption(
    "IC and Hit Rate computed separately for each ticker on the test set. "
    "Shows which companies the model predicts best."
)

if test_preds.empty:
    st.info("No test-set predictions available.")
else:
    from scipy import stats as scipy_stats2

    ticker_metrics = []
    for t in sorted(test_preds["Ticker"].unique()):
        sub = test_preds[test_preds["Ticker"] == t]
        if len(sub) < 5:
            continue
        ic_val = scipy_stats2.spearmanr(sub["y_true"], sub["y_pred"]).statistic
        hit = float((np.sign(sub["y_pred"]) == np.sign(sub["y_true"])).mean())
        mae_val = float(np.mean(np.abs(sub["y_true"] - sub["y_pred"])))
        strategy = np.sign(sub["y_pred"].values) * sub["y_true"].values
        sharpe = float(strategy.mean() / strategy.std() * np.sqrt(252)) if strategy.std() > 0 else np.nan
        ticker_metrics.append({
            "Ticker": t,
            "IC": ic_val,
            "Hit Rate": hit,
            "MAE": mae_val,
            "Pred Sharpe": sharpe,
            "N days": len(sub),
        })

    tm_df = pd.DataFrame(ticker_metrics).sort_values("IC", ascending=False)
    st.dataframe(
        tm_df.style.format({
            "IC": "{:.3f}", "Hit Rate": "{:.1%}",
            "MAE": "{:.4f}", "Pred Sharpe": "{:.2f}",
        }).background_gradient(subset=["IC"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

# -----------------------------------------------------------------------
# Section 4: Long / Short strategy simulation (test set)
# -----------------------------------------------------------------------
st.header("4 · Long / Short Strategy Simulation — Test Set")
st.caption(
    "Each day, rank all tickers by predicted return. Go **long** the top-K and **short** the bottom-K "
    "(equal weight within each leg). Portfolio daily return = avg(long actual) − avg(short actual). "
    "Compare vs equal-weight long-only (buy-and-hold) benchmark."
)

if test_preds.empty:
    st.info("No test-set predictions available.")
else:
    k_long = st.slider("K — number of long/short positions per day", min_value=1, max_value=5, value=3)

    test_sorted = test_preds.copy()
    test_sorted["Date"] = pd.to_datetime(test_sorted["Date"])

    def daily_strategy(group: pd.DataFrame, k: int) -> float:
        ranked = group.sort_values("y_pred", ascending=False)
        long_ret  = ranked.head(k)["y_true"].mean()
        short_ret = ranked.tail(k)["y_true"].mean()
        return float(long_ret - short_ret)

    daily_ls = (
        test_sorted.groupby("Date")
        .apply(lambda g: daily_strategy(g, k_long), include_groups=False)
        .rename("ls_return")
        .reset_index()
    )
    daily_bh = (
        test_sorted.groupby("Date")["y_true"]
        .mean()
        .rename("bh_return")
        .reset_index()
    )
    strat_df = daily_ls.merge(daily_bh, on="Date").sort_values("Date")

    # Cumulative log returns
    strat_df["cum_ls"] = np.exp(strat_df["ls_return"].cumsum())
    strat_df["cum_bh"] = np.exp(strat_df["bh_return"].cumsum())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strat_df["Date"], y=strat_df["cum_ls"],
        name=f"Long top-{k_long} / Short bottom-{k_long}",
        line=dict(color="#2ca02c", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=strat_df["Date"], y=strat_df["cum_bh"],
        name="Equal-weight long only (benchmark)",
        line=dict(color="#d62728", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(
        title=f"Cumulative Return — Long/Short vs Benchmark (Test Set 2026)",
        xaxis_title="Date", yaxis_title="Growth of $1",
        height=430, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    ls_daily = strat_df["ls_return"]
    bh_daily = strat_df["bh_return"]

    def _sharpe(r: pd.Series) -> float:
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan

    def _maxdd(cum: pd.Series) -> float:
        peak = cum.cummax()
        return float(((cum - peak) / peak).min())

    summary = pd.DataFrame([
        {
            "Strategy": f"L/S top-{k_long}/bottom-{k_long}",
            "Total return": f"{(strat_df['cum_ls'].iloc[-1] - 1)*100:.1f}%",
            "Ann. Sharpe": f"{_sharpe(ls_daily):.2f}",
            "Max Drawdown": f"{_maxdd(strat_df['cum_ls'])*100:.1f}%",
            "Hit Rate": f"{(ls_daily > 0).mean()*100:.1f}%",
        },
        {
            "Strategy": "Equal-weight long only",
            "Total return": f"{(strat_df['cum_bh'].iloc[-1] - 1)*100:.1f}%",
            "Ann. Sharpe": f"{_sharpe(bh_daily):.2f}",
            "Max Drawdown": f"{_maxdd(strat_df['cum_bh'])*100:.1f}%",
            "Hit Rate": f"{(bh_daily > 0).mean()*100:.1f}%",
        },
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ This simulation ignores transaction costs, slippage, and bid-ask spread. "
        "Real trading returns would be lower. This is for educational purposes only."
    )

# -----------------------------------------------------------------------
# Section 5: Download all predictions
# -----------------------------------------------------------------------
st.header("5 · Download Predictions")

col1, col2 = st.columns(2)
with col1:
    if not all_preds.empty:
        st.download_button(
            label="Download ALL predictions CSV (train + val + test)",
            data=all_preds.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_choice}_all_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
with col2:
    if not test_preds.empty:
        st.download_button(
            label="Download TEST predictions CSV (2026 only)",
            data=test_preds.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_choice}_test_predictions_2026.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.caption(
    "Columns: Date, Ticker, y_true (actual log return), y_pred (predicted log return), "
    "residual (y_true − y_pred), split (train/val/test)."
)
