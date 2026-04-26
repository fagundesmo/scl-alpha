"""Output: all exploratory and diagnostic plots from the notebook pipeline."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Output", page_icon="📊", layout="wide")
st.title("📊 Output — Data & Feature Plots")
st.caption(
    "All plots are generated from the raw and engineered data produced in Pre-data. "
    "Run Pre-data first to populate this page."
)

# -----------------------------------------------------------------------
# Guard: require Pre-data to have been run
# -----------------------------------------------------------------------
raw: dict | None = st.session_state.get("raw_data")
features: pd.DataFrame | None = st.session_state.get("features")
aligned_data: dict | None = st.session_state.get("aligned")

if raw is None or features is None or aligned_data is None:
    st.warning("No data found in session. Please complete all three steps in **Pre-data** first.")
    st.stop()

companies_ohlcv = raw["companies_ohlcv"]
vix_close = raw["vix_close"]
fred_daily = raw["fred_daily"]
sector_ml = aligned_data["sector_ml"]
spy_ml = aligned_data["spy_ml"]

all_tickers = sorted(features["Ticker"].unique().tolist())
sample_tickers_default = [t for t in ["CHRW", "FDX", "UPS"] if t in all_tickers][:3]

# -----------------------------------------------------------------------
# Section 1: Raw prices
# -----------------------------------------------------------------------
st.header("1 · Raw Company Close Prices")

selected_raw = st.multiselect(
    "Select tickers to plot",
    options=all_tickers,
    default=sample_tickers_default,
    key="raw_price_tickers",
)

if selected_raw:
    plot_df = companies_ohlcv[companies_ohlcv["Ticker"].isin(selected_raw)].copy()
    fig = px.line(
        plot_df,
        x="Date",
        y="Close",
        color="Ticker",
        title="Raw Company Close Prices",
    )
    fig.update_layout(height=480, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select at least one ticker above.")

# -----------------------------------------------------------------------
# Section 2: Rebased to 100
# -----------------------------------------------------------------------
st.header("2 · Companies Rebased to 100")
st.caption("Normalises each ticker to 100 at its earliest date for apples-to-apples comparison.")

selected_rebased = st.multiselect(
    "Select tickers",
    options=all_tickers,
    default=sample_tickers_default,
    key="rebased_tickers",
)

if selected_rebased:
    plot_df = companies_ohlcv[companies_ohlcv["Ticker"].isin(selected_rebased)].copy()
    plot_df["Close_100"] = plot_df.groupby("Ticker")["Close"].transform(
        lambda s: 100 * s / s.iloc[0]
    )
    fig = px.line(
        plot_df,
        x="Date",
        y="Close_100",
        color="Ticker",
        title="Company Close Prices — Rebased to 100",
        labels={"Close_100": "Price (rebased to 100)"},
    )
    fig.update_layout(height=480, xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 3: Sector ETF vs SPY
# -----------------------------------------------------------------------
st.header("3 · Sector ETF (IYT) vs SPY — Rebased to 100")

sector_close = sector_ml["Close"]
if isinstance(sector_close, pd.DataFrame):
    sector_close = sector_close.iloc[:, 0]
spy_close = spy_ml["Close"]
if isinstance(spy_close, pd.DataFrame):
    spy_close = spy_close.iloc[:, 0]

compare_df = pd.DataFrame(
    {
        "Date": sector_close.index,
        "Sector ETF (IYT)": (100 * sector_close / sector_close.iloc[0]).values,
        "SPY": (100 * spy_close / spy_close.iloc[0]).values,
    }
).melt(id_vars="Date", var_name="Series", value_name="Value")

fig = px.line(
    compare_df,
    x="Date",
    y="Value",
    color="Series",
    title="Sector ETF (IYT) vs SPY — Rebased to 100",
    labels={"Value": "Price (rebased to 100)"},
)
fig.update_layout(height=420, xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 4: VIX
# -----------------------------------------------------------------------
st.header("4 · VIX Daily Close")

vix_series = vix_close
if isinstance(vix_series, pd.DataFrame):
    vix_series = vix_series.iloc[:, 0]

fig = px.line(
    x=vix_series.index,
    y=vix_series.values,
    labels={"x": "Date", "y": "VIX"},
    title="VIX Daily Close",
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 5: FRED macro series
# -----------------------------------------------------------------------
st.header("5 · FRED Macro Series")
st.caption("Each panel uses its own Y axis. Data is forward-filled between releases.")

fred_long = fred_daily.reset_index().melt(
    id_vars="Date", var_name="Series", value_name="Value"
)
fig = px.line(
    fred_long,
    x="Date",
    y="Value",
    facet_row="Series",
    facet_row_spacing=0.03,
    height=900,
    title="FRED Macro Series (forward-filled daily)",
)
fig.update_yaxes(matches=None)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 6: Missing values
# -----------------------------------------------------------------------
st.header("6 · Missing Values by Dataset")

vix_aligned = aligned_data["vix_ml"]
if isinstance(vix_aligned, pd.DataFrame):
    vix_aligned = vix_aligned.iloc[:, 0]

missing_raw = pd.Series(
    {
        "companies_ohlcv": companies_ohlcv.isna().sum().sum(),
        "sector_ml": sector_ml.isna().sum().sum(),
        "spy_ml": spy_ml.isna().sum().sum(),
        "vix_ml": int(vix_aligned.isna().sum()),
        "fred_ml": aligned_data["fred_ml"].isna().sum().sum(),
        "features": features.drop(columns=["Date", "Ticker"]).isna().sum().sum(),
    }
).reset_index()
missing_raw.columns = ["Dataset", "Missing Values"]

fig = px.bar(
    missing_raw,
    x="Dataset",
    y="Missing Values",
    title="Missing Values by Dataset",
    text_auto=True,
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 7: Engineered feature time series (one ticker)
# -----------------------------------------------------------------------
st.header("7 · Engineered Feature Time Series")

ticker_view = st.selectbox(
    "Select ticker",
    options=all_tickers,
    index=all_tickers.index("UPS") if "UPS" in all_tickers else 0,
    key="feature_ts_ticker",
)

feature_ts_cols = [
    "mom_1m", "vol_1m", "beta_spy_63d", "beta_sector_63d",
    "vix_z_20", "target_fwd_1d",
]
available_ts_cols = [c for c in feature_ts_cols if c in features.columns]

feat_one = features.loc[
    features["Ticker"] == ticker_view,
    ["Date", "Ticker"] + available_ts_cols,
].copy()
feat_long = feat_one.melt(
    id_vars=["Date", "Ticker"], var_name="Feature", value_name="Value"
)

fig = px.line(
    feat_long,
    x="Date",
    y="Value",
    facet_row="Feature",
    facet_row_spacing=0.02,
    height=1000,
    title=f"Engineered Features — {ticker_view}",
)
fig.update_yaxes(matches=None)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 8: Feature distributions
# -----------------------------------------------------------------------
st.header("8 · Feature Distributions")

dist_cols = [
    "mom_1m", "vol_1m", "beta_spy_63d", "atr_pct_14",
    "volume_z_20", "target_fwd_1d",
]
available_dist = [c for c in dist_cols if c in features.columns]

cols_per_row = 3
rows = [
    available_dist[i : i + cols_per_row]
    for i in range(0, len(available_dist), cols_per_row)
]
for row_cols in rows:
    st_cols = st.columns(len(row_cols))
    for col_widget, feat_col in zip(st_cols, row_cols):
        fig = px.histogram(
            features,
            x=feat_col,
            nbins=60,
            title=feat_col,
        )
        fig.update_layout(height=300, margin=dict(t=40, b=20))
        col_widget.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 9: Box plots by ticker
# -----------------------------------------------------------------------
st.header("9 · Distribution by Ticker")

box_specs = [
    ("target_fwd_1d", "Next-Day Return Target (target_fwd_1d) by Ticker"),
    ("vol_1m", "1-Month Volatility (vol_1m) by Ticker"),
    ("beta_spy_63d", "63-Day Beta vs SPY (beta_spy_63d) by Ticker"),
]

for col_name, title in box_specs:
    if col_name not in features.columns:
        continue
    fig = px.box(
        features,
        x="Ticker",
        y=col_name,
        title=title,
        points=False,
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 10: Correlation heatmap
# -----------------------------------------------------------------------
st.header("10 · Feature Correlation Heatmap")
st.caption(
    "Pearson correlation across all numeric features. "
    "Red = positive, Blue = negative. |r| > 0.7 between predictors may indicate redundancy."
)

corr_cols = [
    "mom_1m", "rel_sector_1m", "rel_spy_1m", "vol_1m",
    "beta_spy_63d", "beta_sector_63d", "corr_spy_63d",
    "atr_pct_14", "volume_z_20",
    "sector_etf_ret_1d", "vix_ret_1d", "vix_z_20",
    "dgs10_chg_5d", "claims_yoy", "diesel_1m",
    "target_fwd_1d",
]
available_corr = [c for c in corr_cols if c in features.columns]
corr = features[available_corr].corr()

fig = px.imshow(
    corr,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Feature Correlation Heatmap",
    aspect="auto",
)
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 11: Feature coverage by ticker
# -----------------------------------------------------------------------
st.header("11 · Feature Coverage by Ticker")
st.caption("Number of trading days with complete feature data per ticker.")

coverage = (
    features.groupby("Ticker")["Date"].nunique().sort_values().reset_index()
)
coverage.columns = ["Ticker", "n_dates"]

fig = px.bar(
    coverage,
    x="Ticker",
    y="n_dates",
    title="Feature Coverage by Ticker",
    labels={"n_dates": "Number of Dates"},
    text_auto=True,
)
fig.update_layout(height=380)
st.plotly_chart(fig, use_container_width=True)
