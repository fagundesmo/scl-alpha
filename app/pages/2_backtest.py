"""
Backtest page — historical strategy performance.

Displays:
    - Equity curve (strategy vs IYT buy-and-hold).
    - Metric summary cards (CAGR, Sharpe, Max Drawdown, Hit Rate).
    - Date range selector.

All data is pre-computed and loaded from cache.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Backtest", page_icon="📈", layout="wide")
st.title("📈 Backtest Results")

# ---------------------------------------------------------------------------
# Load cached backtest results
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
BACKTEST_PATH = CACHE_DIR / "backtest_results.parquet"
BENCHMARK_PATH = CACHE_DIR / "benchmark_iyt.parquet"


@st.cache_data(ttl=3600)
def load_backtest():
    if BACKTEST_PATH.exists():
        return pd.read_parquet(BACKTEST_PATH)
    return None


@st.cache_data(ttl=3600)
def load_benchmark():
    if BENCHMARK_PATH.exists():
        return pd.read_parquet(BENCHMARK_PATH)
    return None


bt = load_backtest()
bench = load_benchmark()

if bt is None:
    st.warning(
        "No cached backtest results found. "
        "Run the notebook first, then copy outputs to `app/cache/`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Date range filter
# ---------------------------------------------------------------------------
min_date = bt.index.min().date()
max_date = bt.index.max().date()

col_start, col_end = st.columns(2)
with col_start:
    start = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with col_end:
    end = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

bt_filtered = bt[(bt.index >= pd.Timestamp(start)) & (bt.index <= pd.Timestamp(end))]

if len(bt_filtered) < 2:
    st.info("Not enough data in the selected range.")
    st.stop()

# Re-base cumulative return to start of filtered range
bt_filtered = bt_filtered.copy()
bt_filtered["cumulative_return"] = (
    bt_filtered["cumulative_return"] / bt_filtered["cumulative_return"].iloc[0]
)

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
from src.metrics import compute_trading_metrics

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

# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------
st.subheader("Equity Curve")

try:
    from src.plots import plotly_equity_curve

    bench_cum = None
    if bench is not None:
        bench_f = bench[
            (bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))
        ]
        if len(bench_f) > 1:
            bench_cum = bench_f["cumulative_return"]

    fig = plotly_equity_curve(bt_filtered["cumulative_return"], bench_cum)
    st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.line_chart(bt_filtered["cumulative_return"])

# ---------------------------------------------------------------------------
# Weekly returns table (expandable)
# ---------------------------------------------------------------------------
with st.expander("Weekly returns detail"):
    detail = bt_filtered[["gross_return_pct", "net_return_pct", "n_holdings", "cost_pct"]].copy()
    detail.columns = ["Gross Return (%)", "Net Return (%)", "# Holdings", "Cost (%)"]
    st.dataframe(detail.style.format("{:.2f}"), use_container_width=True)
