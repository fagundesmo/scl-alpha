"""
Feature engineering module.

All features use only information available on or before the given date
(no look-ahead bias). The target variable (target_fwd_1d) is computed
separately and kept out of FEATURE_COLUMNS.

Public API
----------
align_by_timeframe(...)    -> dict
validate_trading_data(...) -> dict of check results
build_features_for_ticker(...) -> pd.DataFrame  (one company)
build_feature_matrix(...)  -> pd.DataFrame  (all companies stacked)
FEATURE_COLUMNS            -> list[str]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BETA_WIN, TARGET_COL


# ============================================================
# Internal datetime helpers
# ============================================================

def _normalize_index(obj: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Strip timezone from index and remove duplicate dates (keep last)."""
    x = obj.copy()
    idx = pd.to_datetime(x.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    x.index = idx
    x = x[~x.index.duplicated(keep="last")].sort_index()
    return x


def _normalize_date_col(df: pd.DataFrame, col: str = "Date") -> pd.DataFrame:
    """Strip timezone from a Date column and sort."""
    x = df.copy()
    d = pd.to_datetime(x[col])
    if getattr(d.dt, "tz", None) is not None:
        d = d.dt.tz_localize(None)
    x[col] = d
    return x.sort_values(col)


def _get_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Extract a single column, squeezing MultiIndex DataFrames if needed."""
    c = df[col]
    if isinstance(c, pd.DataFrame):
        c = c.iloc[:, 0]
    return c


# ============================================================
# Alignment
# ============================================================

def align_by_timeframe(
    fred_daily: pd.DataFrame,
    vix_close: pd.Series | pd.DataFrame,
    sector_etf: pd.DataFrame,
    spy_ohlcv: pd.DataFrame,
    companies_ohlcv: pd.DataFrame,
    strict_panel: bool = False,
) -> dict:
    """
    Align all datasets to a shared trading calendar.

    The master calendar is the intersection of dates present in the sector
    ETF, SPY, and VIX series. FRED data is forward-filled to that calendar.
    Company rows outside the master calendar are dropped.

    Parameters
    ----------
    fred_daily : DataFrame indexed by Date
    vix_close  : Series or 1-column DataFrame indexed by Date
    sector_etf : OHLCV DataFrame indexed by Date
    spy_ohlcv  : OHLCV DataFrame indexed by Date
    companies_ohlcv : long panel with columns [Date, Ticker, Open, High, Low, Close, Volume]
    strict_panel : if True, keep only dates where every ticker has a row

    Returns
    -------
    dict with keys: fred_daily, vix_close, sector_etf, spy_ohlcv,
                    companies_ohlcv, coverage, master_index
    """
    fred_daily = _normalize_index(fred_daily)
    sector_etf = _normalize_index(sector_etf)
    spy_ohlcv = _normalize_index(spy_ohlcv)
    companies_ohlcv = _normalize_date_col(companies_ohlcv, "Date")

    if isinstance(vix_close, pd.DataFrame):
        if vix_close.shape[1] != 1:
            raise ValueError("vix_close must be a Series or 1-column DataFrame.")
        vix_close = vix_close.iloc[:, 0]
    vix_close = _normalize_index(vix_close).rename("VIX")

    master_idx = (
        sector_etf.index
        .intersection(spy_ohlcv.index)
        .intersection(vix_close.index)
        .sort_values()
    )

    sector_aligned = sector_etf.reindex(master_idx)
    spy_aligned = spy_ohlcv.reindex(master_idx)
    vix_aligned = vix_close.reindex(master_idx).ffill()
    fred_aligned = fred_daily.reindex(master_idx).ffill()

    companies_aligned = companies_ohlcv[
        companies_ohlcv["Date"].isin(master_idx)
    ].copy()

    if strict_panel:
        n_tickers = companies_aligned["Ticker"].nunique()
        valid_dates = (
            companies_aligned.groupby("Date")["Ticker"].nunique()
            .loc[lambda s: s == n_tickers]
            .index
            .sort_values()
        )
        master_idx = pd.DatetimeIndex(valid_dates)
        sector_aligned = sector_aligned.reindex(master_idx)
        spy_aligned = spy_aligned.reindex(master_idx)
        vix_aligned = vix_aligned.reindex(master_idx).ffill()
        fred_aligned = fred_aligned.reindex(master_idx).ffill()
        companies_aligned = companies_aligned[
            companies_aligned["Date"].isin(master_idx)
        ].copy()

    coverage = (
        companies_aligned.groupby("Ticker")["Date"].nunique()
        .rename("n_dates")
        .to_frame()
    )
    coverage["master_dates"] = len(master_idx)
    coverage["missing_vs_master"] = coverage["master_dates"] - coverage["n_dates"]

    return {
        "fred_daily": fred_aligned,
        "vix_close": vix_aligned,
        "sector_etf": sector_aligned,
        "spy_ohlcv": spy_aligned,
        "companies_ohlcv": companies_aligned,
        "coverage": coverage,
        "master_index": master_idx,
    }


# ============================================================
# Validation
# ============================================================

def validate_trading_data(
    fred_ml: pd.DataFrame,
    vix_ml: pd.Series,
    sector_ml: pd.DataFrame,
    spy_ml: pd.DataFrame,
    companies_ml: pd.DataFrame,
) -> dict:
    """Run basic data-quality checks. Returns a dict of check results."""
    checks: dict = {}

    checks["sector_spy_same_index"] = sector_ml.index.equals(spy_ml.index)
    checks["sector_vix_same_index"] = sector_ml.index.equals(vix_ml.index)
    checks["sector_fred_same_index"] = sector_ml.index.equals(fred_ml.index)

    checks["sector_sorted_unique"] = (
        sector_ml.index.is_monotonic_increasing and sector_ml.index.is_unique
    )
    checks["spy_sorted_unique"] = (
        spy_ml.index.is_monotonic_increasing and spy_ml.index.is_unique
    )
    checks["fred_sorted_unique"] = (
        fred_ml.index.is_monotonic_increasing and fred_ml.index.is_unique
    )

    checks["company_has_required_cols"] = all(
        c in companies_ml.columns
        for c in ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    )
    checks["company_no_dupes"] = not companies_ml.duplicated(
        subset=["Date", "Ticker"]
    ).any()

    sector_close = _get_col(sector_ml, "Close")
    spy_close = _get_col(spy_ml, "Close")
    checks["sector_missing_close"] = int(sector_close.isna().sum())
    checks["spy_missing_close"] = int(spy_close.isna().sum())
    checks["vix_missing"] = int(vix_ml.isna().sum())
    checks["fred_missing_total"] = int(fred_ml.isna().sum().sum())
    checks["company_missing_close"] = int(companies_ml["Close"].isna().sum())

    coverage = companies_ml.groupby("Ticker")["Date"].nunique()
    checks["company_min_dates"] = int(coverage.min())
    checks["company_max_dates"] = int(coverage.max())
    checks["master_dates"] = int(len(sector_ml.index))

    return checks


# ============================================================
# Per-ticker feature builder
# ============================================================

def build_features_for_ticker(
    company_ohlcv: pd.DataFrame,
    sector_etf_ohlcv: pd.DataFrame,
    spy_ohlcv: pd.DataFrame,
    vix_close: pd.Series | None = None,
    fred_daily: pd.DataFrame | None = None,
    beta_win: int = BETA_WIN,
    min_history: int | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix for ONE company on a daily time series.

    Target variable: target_fwd_1d = next-day company log return.

    Parameters
    ----------
    company_ohlcv : DataFrame with index=DatetimeIndex, columns=[Open,High,Low,Close,Volume]
    sector_etf_ohlcv : OHLCV DataFrame for the sector ETF (e.g. IYT)
    spy_ohlcv : OHLCV DataFrame for SPY
    vix_close : daily VIX close (Series). If None and VIXCLS in fred_daily, uses that.
    fred_daily : daily macro DataFrame (VIXCLS, GASDESW, DGS10, ICNSA). Already
                 aligned to the trading calendar — do NOT pre-lag before passing.
    beta_win : rolling beta / correlation window (trading days)
    min_history : warmup rows to drop from the start

    Returns
    -------
    pd.DataFrame with feature columns + target_fwd_1d
    """
    company_ohlcv = company_ohlcv.copy().sort_index()
    sector_etf_ohlcv = sector_etf_ohlcv.copy().sort_index()
    spy_ohlcv = spy_ohlcv.copy().sort_index()

    if fred_daily is None:
        fred_daily = pd.DataFrame(index=company_ohlcv.index)
    else:
        fred_daily = fred_daily.copy().sort_index()

    # Resolve VIX source: prefer explicit Series, fall back to FRED VIXCLS
    if vix_close is None and "VIXCLS" in fred_daily.columns:
        vix_close = fred_daily["VIXCLS"].copy()

    # Remove VIXCLS from macro table so it appears only once
    if "VIXCLS" in fred_daily.columns:
        fred_daily = fred_daily.drop(columns=["VIXCLS"])

    f = pd.DataFrame(index=company_ohlcv.index).sort_index()

    # Core price levels
    f["company_close"] = company_ohlcv["Close"]
    f["sector_etf_close"] = _get_col(sector_etf_ohlcv, "Close").reindex(f.index)
    f["spy_close"] = _get_col(spy_ohlcv, "Close").reindex(f.index)

    if vix_close is not None:
        f["vix_close"] = pd.Series(vix_close).reindex(f.index)

    # Daily log returns
    f["r_company"] = np.log(f["company_close"]).diff()
    f["r_sector_etf"] = np.log(f["sector_etf_close"]).diff()
    f["r_spy"] = np.log(f["spy_close"]).diff()

    # Multi-horizon momentum, relative momentum, and rolling volatility
    for n, label in [(5, "1w"), (21, "1m"), (63, "3m"), (126, "6m")]:
        f[f"mom_{label}"] = f["company_close"].pct_change(n)
        f[f"rel_sector_{label}"] = (
            f["company_close"].pct_change(n) - f["sector_etf_close"].pct_change(n)
        )
        f[f"rel_spy_{label}"] = (
            f["company_close"].pct_change(n) - f["spy_close"].pct_change(n)
        )
        f[f"vol_{label}"] = f["r_company"].rolling(n).std() * np.sqrt(252)

    # Rolling beta and correlation vs SPY and sector ETF
    f[f"beta_spy_{beta_win}d"] = (
        f["r_company"].rolling(beta_win).cov(f["r_spy"])
        / f["r_spy"].rolling(beta_win).var()
    )
    f[f"beta_sector_{beta_win}d"] = (
        f["r_company"].rolling(beta_win).cov(f["r_sector_etf"])
        / f["r_sector_etf"].rolling(beta_win).var()
    )
    f[f"corr_spy_{beta_win}d"] = (
        f["r_company"].rolling(beta_win).corr(f["r_spy"])
    )

    # ATR and volume z-score from company OHLCV
    tr = pd.concat(
        [
            company_ohlcv["High"] - company_ohlcv["Low"],
            (company_ohlcv["High"] - company_ohlcv["Close"].shift()).abs(),
            (company_ohlcv["Low"] - company_ohlcv["Close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    f["atr_14"] = tr.rolling(14).mean()
    f["atr_pct_14"] = f["atr_14"] / f["company_close"]

    v20 = company_ohlcv["Volume"].rolling(20)
    f["volume_z_20"] = (company_ohlcv["Volume"] - v20.mean()) / v20.std()

    # Sector ETF features
    f["sector_etf_ret_1d"] = f["sector_etf_close"].pct_change()
    f["sector_etf_mom_1m"] = f["sector_etf_close"].pct_change(21)

    # VIX-derived features
    if "vix_close" in f.columns:
        f["vix_ret_1d"] = f["vix_close"].pct_change()
        f["vix_z_20"] = (
            (f["vix_close"] - f["vix_close"].rolling(20).mean())
            / f["vix_close"].rolling(20).std()
        )

    # Macro features — aligned to trading calendar and lagged 1 day here.
    # fred_daily must NOT be pre-lagged before being passed to this function.
    m = fred_daily.reindex(f.index).ffill().copy()
    if "DGS10" in m.columns:
        m["dgs10_chg_5d"] = m["DGS10"].diff(5)
    if "ICNSA" in m.columns:
        m["claims_yoy"] = m["ICNSA"].pct_change(252)
    if "GASDESW" in m.columns:
        m["diesel_1m"] = m["GASDESW"].pct_change(21)
    if "NAPM_lag1" in m.columns:
        m["pmi_gap_50"] = m["NAPM_lag1"] - 50

    # Shift by 1 trading day to prevent look-ahead bias
    f = f.join(m.shift(1), how="left")

    # Target: next-day log return
    f[TARGET_COL] = f["r_company"].shift(-1)

    f = f.replace([np.inf, -np.inf], np.nan)

    if min_history is None:
        min_history = max(126, beta_win, 20, 14)
    f = f.iloc[min_history:].copy()

    required = [
        "r_company",
        "r_sector_etf",
        "r_spy",
        f"beta_spy_{beta_win}d",
        f"beta_sector_{beta_win}d",
        TARGET_COL,
    ]
    if "vix_close" in f.columns:
        required.append("vix_close")

    f = f.dropna(subset=[c for c in required if c in f.columns])
    return f


# ============================================================
# Full panel builder
# ============================================================

FEATURE_COLUMNS: list[str] = [
    # Returns
    "r_company", "r_sector_etf", "r_spy",
    # Momentum
    "mom_1w", "mom_1m", "mom_3m", "mom_6m",
    # Relative momentum vs sector ETF
    "rel_sector_1w", "rel_sector_1m", "rel_sector_3m", "rel_sector_6m",
    # Relative momentum vs SPY
    "rel_spy_1w", "rel_spy_1m", "rel_spy_3m", "rel_spy_6m",
    # Volatility
    "vol_1w", "vol_1m", "vol_3m", "vol_6m",
    # Beta / correlation
    f"beta_spy_{BETA_WIN}d", f"beta_sector_{BETA_WIN}d", f"corr_spy_{BETA_WIN}d",
    # OHLCV-derived
    "atr_pct_14", "volume_z_20",
    # Sector ETF
    "sector_etf_ret_1d", "sector_etf_mom_1m",
    # VIX
    "vix_close", "vix_ret_1d", "vix_z_20",
    # Macro levels (lagged)
    "GASDESW", "DGS10", "ICNSA",
    # Macro transforms (lagged)
    "dgs10_chg_5d", "claims_yoy", "diesel_1m",
]


def build_feature_matrix(
    companies_ml: pd.DataFrame,
    sector_ml: pd.DataFrame,
    spy_ml: pd.DataFrame,
    vix_ml: pd.Series,
    fred_ml: pd.DataFrame,
    beta_win: int = BETA_WIN,
) -> pd.DataFrame:
    """
    Build the feature matrix for all companies and stack into one DataFrame.

    Parameters
    ----------
    companies_ml  : long panel (Date, Ticker, Open, High, Low, Close, Volume)
    sector_ml     : aligned sector ETF OHLCV (index = trading calendar)
    spy_ml        : aligned SPY OHLCV
    vix_ml        : aligned VIX close Series
    fred_ml       : aligned FRED macro DataFrame (NOT pre-lagged)
    beta_win      : rolling beta window

    Returns
    -------
    pd.DataFrame with columns [Date, Ticker, <features>, target_fwd_1d]
    """
    all_feat = []
    for ticker, g in companies_ml.groupby("Ticker"):
        company_ohlcv = (
            g.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            .sort_index()
        )
        feat = build_features_for_ticker(
            company_ohlcv=company_ohlcv,
            sector_etf_ohlcv=sector_ml,
            spy_ohlcv=spy_ml,
            vix_close=vix_ml,
            fred_daily=fred_ml,
            beta_win=beta_win,
        )
        feat["Ticker"] = ticker
        all_feat.append(feat)

    features = (
        pd.concat(all_feat)
        .reset_index()
        .rename(columns={"index": "Date"})
    )
    return features
