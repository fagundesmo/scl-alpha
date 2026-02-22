"""
Data page — summary statistics and feature correlations.

Entirely static: computed once from the cached feature matrix.
Helps users understand the raw data before interpreting model outputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Data", page_icon="📋", layout="wide")
st.title("📋 Data Explorer")

# ---------------------------------------------------------------------------
# Load cached feature matrix
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
FEATURES_PATH = CACHE_DIR / "feature_matrix.parquet"


@st.cache_data(ttl=3600)
def load_features():
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    return None


df = load_features()

if df is None:
    st.warning(
        "No cached feature matrix found. "
        "Run the notebook first, then copy `feature_matrix.parquet` to `app/cache/`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
st.subheader("Summary Statistics")

# Pick numeric feature columns only
feature_cols = [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64")]

stats = df[feature_cols].describe().T
stats["missing_%"] = (df[feature_cols].isna().sum() / len(df) * 100).round(1)

st.dataframe(
    stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_%"]]
    .style.format("{:.2f}"),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------
st.subheader("Feature Correlations")

# Let user select which features to include
from src.features import FEATURE_COLUMNS

available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
selected = st.multiselect(
    "Select features for correlation matrix",
    options=available_features,
    default=available_features[:10],
)

if len(selected) >= 2:
    try:
        from src.plots import plot_correlation_heatmap
        fig = plot_correlation_heatmap(df.reset_index(drop=True), columns=selected)
        st.pyplot(fig)
    except Exception:
        corr = df[selected].corr()
        st.dataframe(corr.style.format("{:.2f}"), use_container_width=True)
else:
    st.info("Select at least 2 features to display the correlation matrix.")

# ---------------------------------------------------------------------------
# Ticker filter
# ---------------------------------------------------------------------------
st.subheader("Per-Ticker View")

if "ticker" in df.index.names:
    tickers = df.index.get_level_values("ticker").unique().tolist()
elif "ticker" in df.columns:
    tickers = df["ticker"].unique().tolist()
else:
    tickers = []

if tickers:
    sel_ticker = st.selectbox("Ticker", tickers)
    if "ticker" in df.index.names:
        ticker_data = df.xs(sel_ticker, level="ticker")
    else:
        ticker_data = df[df["ticker"] == sel_ticker]

    st.write(f"**{sel_ticker}** — {len(ticker_data)} observations")
    st.dataframe(
        ticker_data[available_features].describe().T.style.format("{:.2f}"),
        use_container_width=True,
    )
