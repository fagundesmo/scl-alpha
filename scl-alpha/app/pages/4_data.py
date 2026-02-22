"""
Data page for summary statistics and feature correlations.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Data", page_icon="📋", layout="wide")
st.title("📋 Data Explorer")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
FEATURES_PATH = CACHE_DIR / "feature_matrix.parquet"


@st.cache_data(ttl=3600)
def load_features():
    if FEATURES_PATH.exists():
        return pd.read_parquet(FEATURES_PATH)
    return None


df = st.session_state.get("feature_matrix")
if df is None:
    df = load_features()

if df is None or df.empty:
    st.warning(
        "No feature matrix found. Run one or more models in 'model lab' "
        "or upload app/cache/feature_matrix.parquet."
    )
    st.stop()

st.subheader("Summary Statistics")

numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
stats = df[numeric_cols].describe().T
stats["missing_%"] = (df[numeric_cols].isna().sum() / len(df) * 100).round(1)

st.dataframe(
    stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing_%"]]
    .style.format("{:.2f}"),
    use_container_width=True,
)

st.subheader("Feature Correlations")
from src.features import FEATURE_COLUMNS

available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
selected = st.multiselect(
    "Select features for correlation matrix",
    options=available_features,
    default=available_features[: min(10, len(available_features))],
)

if len(selected) >= 2:
    try:
        from src.plots import plot_correlation_heatmap

        fig = plot_correlation_heatmap(df.reset_index(drop=True), columns=selected)
        st.pyplot(fig)
    except Exception:
        st.dataframe(df[selected].corr().style.format("{:.2f}"), use_container_width=True)
else:
    st.info("Select at least 2 features to display the correlation matrix.")

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

    st.write(f"**{sel_ticker}** - {len(ticker_data)} observations")
    st.dataframe(
        ticker_data[available_features].describe().T.style.format("{:.2f}"),
        use_container_width=True,
    )
