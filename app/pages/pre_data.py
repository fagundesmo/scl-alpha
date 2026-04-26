"""Pre-data: step-by-step data download and feature engineering pipeline."""

import pandas as pd
import streamlit as st

from src.config import BENCHMARK_ETF, DATA_START, MARKET_ETF, TICKERS
from src.data_pull import pull_all
from src.features import align_by_timeframe, build_feature_matrix, validate_trading_data

st.set_page_config(page_title="Pre-data", page_icon="📥", layout="wide")
st.title("📥 Pre-data Pipeline")
st.caption(
    "Walk through the data preparation steps in order before running models. "
    "Each step stores results in session state so the Output and Sandbox pages can use them."
)

st.markdown(
    """
**Pipeline steps**
1. **Download** OHLCV prices for supply-chain companies, IYT, and SPY from Yahoo Finance;
   VIX from Yahoo Finance; and macro series (VIX, diesel, 10Y yield, jobless claims) from FRED.
2. **Align** all datasets to a shared trading calendar and validate data quality.
3. **Build features** — momentum, volatility, beta, macro transforms, and `target_fwd_1d` (next-day log return).
"""
)

st.divider()

# -----------------------------------------------------------------------
# Step 1: Download raw data
# -----------------------------------------------------------------------
st.subheader("Step 1 — Download raw data")
st.caption(
    f"Universe: {', '.join(TICKERS)} + {BENCHMARK_ETF} (sector ETF) + {MARKET_ETF}. "
    f"Start date: {DATA_START}."
)

refresh = st.checkbox(
    "Force fresh download (clears all cached files)",
    value=False,
    help="Enable to ignore cached parquet files and re-download everything from APIs.",
)

if st.button("Pull raw data", type="primary", use_container_width=True):
    with st.spinner("Downloading prices, VIX, and FRED macro series..."):
        try:
            raw = pull_all(cache=True, refresh=refresh)
            st.session_state["raw_data"] = raw
            st.success("Raw data downloaded successfully.")
        except Exception as exc:
            st.error(f"Download failed: {exc}")
            st.stop()

raw: dict | None = st.session_state.get("raw_data")

if raw:
    companies_ohlcv = raw["companies_ohlcv"]
    sector_etf = raw["sector_etf"]
    spy_ohlcv = raw["spy_ohlcv"]
    vix_close = raw["vix_close"]
    fred_daily = raw["fred_daily"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Company rows", f"{len(companies_ohlcv):,}")
    c2.metric("Sector ETF rows", f"{len(sector_etf):,}")
    c3.metric("SPY rows", f"{len(spy_ohlcv):,}")
    c4.metric("VIX rows", f"{len(vix_close):,}")
    c5.metric("FRED rows", f"{len(fred_daily):,}")

    with st.expander("Company OHLCV sample (first 20 rows)"):
        st.dataframe(companies_ohlcv.head(20), use_container_width=True, hide_index=True)

    with st.expander("FRED macro sample"):
        st.dataframe(fred_daily.head(10), use_container_width=True)
else:
    st.info("Click **Pull raw data** to begin.")

st.divider()

# -----------------------------------------------------------------------
# Step 2: Align & validate
# -----------------------------------------------------------------------
st.subheader("Step 2 — Align to trading calendar & validate")
st.caption(
    "Datasets are aligned to the intersection of trading days shared by IYT, SPY, and VIX. "
    "FRED data is forward-filled (most recent known value) to that calendar. "
    "Early rows with no FRED data are then dropped."
)

if raw is None:
    st.info("Complete Step 1 first.")
elif st.button("Align and validate", type="primary", use_container_width=True):
    with st.spinner("Aligning datasets..."):
        try:
            aligned = align_by_timeframe(
                fred_daily=fred_daily,
                vix_close=vix_close,
                sector_etf=sector_etf,
                spy_ohlcv=spy_ohlcv,
                companies_ohlcv=companies_ohlcv,
                strict_panel=False,
            )

            fred_ml = aligned["fred_daily"]
            vix_ml = aligned["vix_close"]
            sector_ml = aligned["sector_etf"]
            spy_ml = aligned["spy_ohlcv"]
            companies_ml = aligned["companies_ohlcv"]
            coverage = aligned["coverage"]

            # Drop the first few trading days where FRED data is not yet available
            first_full_date = fred_ml.dropna().index.min()
            fred_ml = fred_ml.loc[first_full_date:]
            vix_ml = vix_ml.loc[first_full_date:]
            sector_ml = sector_ml.loc[first_full_date:]
            spy_ml = spy_ml.loc[first_full_date:]
            companies_ml = companies_ml[
                companies_ml["Date"] >= first_full_date
            ].copy()

            st.session_state["aligned"] = {
                "fred_ml": fred_ml,
                "vix_ml": vix_ml,
                "sector_ml": sector_ml,
                "spy_ml": spy_ml,
                "companies_ml": companies_ml,
            }

            checks = validate_trading_data(
                fred_ml, vix_ml, sector_ml, spy_ml, companies_ml
            )

            bool_checks = {k: v for k, v in checks.items() if isinstance(v, bool)}
            all_pass = all(bool_checks.values())
            if all_pass:
                st.success("All boolean checks passed.")
            else:
                failed = [k for k, v in bool_checks.items() if not v]
                st.warning(f"Failed checks: {failed}")

            check_df = pd.DataFrame(
                [{"Check": k, "Result": str(v)} for k, v in checks.items()]
            )
            st.dataframe(check_df, use_container_width=True, hide_index=True)

            st.markdown("**Coverage by ticker** (master_dates = trading days after filtering)")
            st.dataframe(coverage.sort_values("missing_vs_master", ascending=False), use_container_width=True)

        except Exception as exc:
            st.error(f"Alignment failed: {exc}")

aligned_data: dict | None = st.session_state.get("aligned")
if aligned_data and raw:
    st.caption(
        f"Aligned master calendar: {aligned_data['sector_ml'].index.min().date()} "
        f"→ {aligned_data['sector_ml'].index.max().date()} "
        f"({len(aligned_data['sector_ml']):,} trading days)"
    )

st.divider()

# -----------------------------------------------------------------------
# Step 3: Build features
# -----------------------------------------------------------------------
st.subheader("Step 3 — Build feature matrix")
st.caption(
    "Computes momentum, volatility, beta, ATR, volume z-score, VIX, and lagged macro features "
    "for every ticker. Target column: `target_fwd_1d` (next-day log return). "
    "Expect ~30–60 seconds."
)

if aligned_data is None:
    st.info("Complete Step 2 first.")
elif st.button("Build features", type="primary", use_container_width=True):
    with st.spinner(f"Building features for {len(TICKERS)} tickers..."):
        try:
            features = build_feature_matrix(
                companies_ml=aligned_data["companies_ml"],
                sector_ml=aligned_data["sector_ml"],
                spy_ml=aligned_data["spy_ml"],
                vix_ml=aligned_data["vix_ml"],
                fred_ml=aligned_data["fred_ml"],
            )
            st.session_state["features"] = features
            st.success(
                f"Feature matrix built: {features.shape[0]:,} rows × {features.shape[1]} columns."
            )
        except Exception as exc:
            st.error(f"Feature building failed: {exc}")

features: pd.DataFrame | None = st.session_state.get("features")
if isinstance(features, pd.DataFrame) and not features.empty:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(features):,}")
    c2.metric("Columns", features.shape[1])
    c3.metric("Tickers", features["Ticker"].nunique())
    c4.metric(
        "Date range",
        f"{pd.to_datetime(features['Date']).min().date()} → {pd.to_datetime(features['Date']).max().date()}",
    )

    missing = int(
        features.drop(columns=["Date", "Ticker"]).isna().sum().sum()
    )
    if missing > 0:
        st.warning(f"Feature matrix has {missing:,} missing values. Models will impute with training-set medians.")
    else:
        st.success("No missing values in the feature matrix.")

    with st.expander("Feature matrix preview (first 30 rows)"):
        st.dataframe(features.head(30), use_container_width=True, hide_index=True)

    st.info(
        "Data is ready. Go to **Output** to explore plots or **Sandbox Models** to run models."
    )
