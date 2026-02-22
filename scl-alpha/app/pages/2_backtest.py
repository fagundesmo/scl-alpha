"""
Backtest page for historical strategy performance.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.metrics import compute_trading_metrics

st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")
st.title("📈 Backtest Results")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
BACKTEST_PATH = CACHE_DIR / "backtest_results.parquet"
BENCHMARK_PATH = CACHE_DIR / "benchmark_iyt.parquet"


@st.cache_data(ttl=3600)
def load_backtest() -> pd.DataFrame | None:
    if BACKTEST_PATH.exists():
        return pd.read_parquet(BACKTEST_PATH)
    return None


@st.cache_data(ttl=3600)
def load_benchmark() -> pd.DataFrame | None:
    if BENCHMARK_PATH.exists():
        return pd.read_parquet(BENCHMARK_PATH)
    return None


runs = st.session_state.get("model_runs", {})
active_model = st.session_state.get("active_model")
bt = None
bench = None

if runs:
    model_names = list(runs.keys())
    idx = model_names.index(active_model) if active_model in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=idx, key="backtest_model")
    st.session_state["active_model"] = selected_model
    bt = runs[selected_model].get("backtest")
    bench = runs[selected_model].get("benchmark")
    st.caption(f"Using in-memory run output for model: {selected_model}")

if bt is None:
    bt = load_backtest()
if bench is None:
    bench = load_benchmark()

if bt is None or bt.empty:
    st.warning(
        "No backtest results found. Open 'model lab' and run at least one model, "
        "or upload app/cache/backtest_results.parquet."
    )
    st.stop()

bt = bt.copy()
if not isinstance(bt.index, pd.DatetimeIndex):
    bt.index = pd.to_datetime(bt.index)

min_date = bt.index.min().date()
max_date = bt.index.max().date()

c_start, c_end = st.columns(2)
with c_start:
    start = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with c_end:
    end = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

bt_filtered = bt[(bt.index >= pd.Timestamp(start)) & (bt.index <= pd.Timestamp(end))]

if len(bt_filtered) < 2:
    st.info("Not enough data in the selected range.")
    st.stop()

bt_filtered = bt_filtered.copy()
bt_filtered["cumulative_return"] = bt_filtered["cumulative_return"] / bt_filtered["cumulative_return"].iloc[0]

metrics = compute_trading_metrics(bt_filtered)

st.subheader("Performance Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CAGR", f"{metrics['CAGR']:.1%}")
c2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
c3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.1%}")
c4.metric("Profit Factor", f"{metrics['Profit Factor']:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Total Weeks", f"{metrics['Total Periods']}")
c6.metric("Winning Weeks", f"{metrics['Positive Periods']}")
c7.metric("Losing Weeks", f"{metrics['Negative Periods']}")
c8.metric("Avg Turnover", f"{metrics['Avg Turnover']:.0%}")

st.subheader("Equity Curve")

try:
    from src.plots import plotly_equity_curve

    bench_cum = None
    if bench is not None and not bench.empty:
        bench = bench.copy()
        if not isinstance(bench.index, pd.DatetimeIndex):
            bench.index = pd.to_datetime(bench.index)
        bench_f = bench[(bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))]
        if len(bench_f) > 1:
            bench_cum = bench_f["cumulative_return"]

    fig = plotly_equity_curve(bt_filtered["cumulative_return"], bench_cum)
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.line_chart(bt_filtered["cumulative_return"])

with st.expander("Weekly returns detail"):
    detail = bt_filtered[["gross_return_pct", "net_return_pct", "n_holdings", "cost_pct"]].copy()
    detail.columns = ["Gross Return (%)", "Net Return (%)", "# Holdings", "Cost (%)"]
    st.dataframe(detail.style.format("{:.2f}"), use_container_width=True)
