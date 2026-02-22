"""
Feature engineering module.

Every function computes features using ONLY data available on or before the
given date.  This is the single most important invariant in the project.

Convention:
    - Input: a DataFrame with MultiIndex (date, ticker) and at least
      [open, high, low, close, adj_close, volume] plus macro columns.
    - Output: same DataFrame with new feature columns appended.

The target variable (forward return) is computed in a separate function
and is NEVER added to the feature matrix.
"""

import numpy as np
import pandas as pd
from src.config import FORWARD_RETURN_DAYS, FEATURE_BURN_IN_DAYS


# ===================================================================
# Helper: per-ticker rolling computation
# ===================================================================

def _per_ticker(df: pd.DataFrame, func, col_name: str) -> pd.DataFrame:
    """Apply *func* to each ticker's time series and assign the result
    as a new column *col_name*."""
    results = []
    for ticker, grp in df.groupby(level="ticker"):
        s = func(grp)
        s.name = col_name
        results.append(s)
    df[col_name] = pd.concat(results)
    return df


# ===================================================================
# Core features (price / volume / returns)
# ===================================================================

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """1-day, 5-day, and 20-day historical returns (log-style, in %)."""
    for window in [1, 5, 20]:
        col = f"ret_{window}d"
        df[col] = df.groupby(level="ticker")["adj_close"].pct_change(window) * 100.0
    return df


def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling annualised volatility (std of daily returns * sqrt(252))."""
    daily_ret = df.groupby(level="ticker")["adj_close"].pct_change()
    df["vol_20d"] = (
        daily_ret.groupby(level="ticker")
        .rolling(window)
        .std()
        .droplevel(0) * np.sqrt(252) * 100.0
    )
    return df


def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume today / 20-day average volume.  Measures unusual activity."""
    avg_vol = (
        df.groupby(level="ticker")["volume"]
        .rolling(window)
        .mean()
        .droplevel(0)
    )
    df["volume_ratio_20d"] = df["volume"] / avg_vol.replace(0, np.nan)
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Relative Strength Index (0-100)."""
    delta = df.groupby(level="ticker")["adj_close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(level="ticker").rolling(window).mean().droplevel(0)
    avg_loss = loss.groupby(level="ticker").rolling(window).mean().droplevel(0)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    return df


def add_relative_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    5-day return of each ticker minus the 5-day return of IYT and SPY.

    Requires that IYT and SPY are present in the panel as ticker rows.
    Alignment strategy: pivot bench return to a date → value mapping, then
    map across the (date, ticker) index to broadcast safely.
    """
    ret5 = df.groupby(level="ticker")["adj_close"].pct_change(5) * 100.0

    for bench in ["IYT", "SPY"]:
        col = f"ret_vs_{bench.lower()}_5d"
        if bench not in df.index.get_level_values("ticker").unique():
            df[col] = np.nan
            continue

        # bench_ret: Series indexed by date only
        bench_ret: pd.Series = ret5.xs(bench, level="ticker")

        # Create a broadcast Series: map each row's date to the benchmark value.
        # Using merge on date is the safest alignment approach.
        ret5_df = ret5.rename("ticker_ret5").reset_index()
        bench_df = bench_ret.rename("bench_ret5").reset_index()

        merged = ret5_df.merge(bench_df[["date", "bench_ret5"]], on="date", how="left")
        merged[col] = merged["ticker_ret5"] - merged["bench_ret5"]
        merged = merged.set_index(["date", "ticker"])
        df[col] = merged[col]

    return df


# ===================================================================
# Macro / "tuning" features
# ===================================================================

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive change-based features from FRED macro columns.
    Raw levels (VIXCLS, DGS10, NAPM) are kept as-is; they were forward-filled
    during the merge step, so they reflect the most recently published value.
    """
    if "VIXCLS" in df.columns:
        df["vix_level"] = df["VIXCLS"]
        # 5-day change in VIX (first difference within each ticker is the same
        # because VIX is cross-sectional, but groupby keeps alignment clean)
        df["vix_change_5d"] = df.groupby(level="ticker")["VIXCLS"].diff(5)

    if "GASDESW" in df.columns:
        df["diesel_change_4w"] = (
            df.groupby(level="ticker")["GASDESW"].pct_change(20) * 100.0
        )

    if "NAPM" in df.columns:
        df["ism_pmi"] = df["NAPM"]

    if "DGS10" in df.columns:
        df["dgs10"] = df["DGS10"]

    if "ICNSA" in df.columns:
        df["claims_change_4w"] = df.groupby(level="ticker")["ICNSA"].diff(20)

    return df


# ===================================================================
# "Professional" features
# ===================================================================

def add_rolling_beta(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Rolling OLS beta of each ticker vs SPY over *window* days.
    Beta = Cov(r_i, r_m) / Var(r_m).
    """
    daily_ret = df.groupby(level="ticker")["adj_close"].pct_change()
    if "SPY" not in df.index.get_level_values("ticker").unique():
        df["rolling_beta_60d"] = np.nan
        return df

    spy_ret = daily_ret.xs("SPY", level="ticker")

    betas = []
    for ticker, grp in daily_ret.groupby(level="ticker"):
        if ticker in ("SPY", "IYT"):
            continue
        aligned = pd.DataFrame({"stock": grp.droplevel("ticker"), "market": spy_ret}).dropna()
        cov = aligned["stock"].rolling(window).cov(aligned["market"])
        var = aligned["market"].rolling(window).var()
        beta = (cov / var).rename(f"beta")
        beta = pd.DataFrame(beta).assign(ticker=ticker).set_index("ticker", append=True)
        beta.index.names = ["date", "ticker"]
        betas.append(beta.rename(columns={"beta": "rolling_beta_60d"}))

    if betas:
        beta_df = pd.concat(betas)
        df = df.join(beta_df, how="left")
    else:
        df["rolling_beta_60d"] = np.nan

    return df


def add_vol_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Binary flag: 1 if current vol_20d is above its own 80th percentile
    over the trailing 252 days."""
    if "vol_20d" not in df.columns:
        df["vol_regime"] = np.nan
        return df

    pct80 = (
        df.groupby(level="ticker")["vol_20d"]
        .rolling(252)
        .quantile(0.8)
        .droplevel(0)
    )
    pct80 = pct80.reindex(df.index)
    df["vol_regime"] = (df["vol_20d"] > pct80).astype(float)
    return df


def add_momentum_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank of 20-day return within the universe (1 = best)."""
    if "ret_20d" not in df.columns:
        df["momentum_rank"] = np.nan
        return df

    df["momentum_rank"] = (
        df.groupby(level="date")["ret_20d"]
        .rank(ascending=False, method="min")
    )
    return df


def add_mean_reversion(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score of 5-day return vs its own rolling 60-day distribution."""
    if "ret_5d" not in df.columns:
        df["mean_reversion_5d"] = np.nan
        return df

    roll_mean = df.groupby(level="ticker")["ret_5d"].rolling(60).mean().droplevel(0)
    roll_std = df.groupby(level="ticker")["ret_5d"].rolling(60).std().droplevel(0)
    df["mean_reversion_5d"] = (df["ret_5d"] - roll_mean) / roll_std.replace(0, np.nan)
    return df


# ===================================================================
# Target variable  (kept separate from features!)
# ===================================================================

def add_target(df: pd.DataFrame, horizon: int = FORWARD_RETURN_DAYS) -> pd.DataFrame:
    """
    Compute the forward return used as the regression target.

    **WARNING:** This column must NEVER be used as a feature.
    It contains future information by definition.
    """
    df["target_ret_5d_fwd"] = (
        df.groupby(level="ticker")["adj_close"]
        .pct_change(horizon)
        .shift(-horizon) * 100.0
    )
    return df


# ===================================================================
# Master pipeline
# ===================================================================

FEATURE_COLUMNS = [
    # Core
    "ret_1d", "ret_5d", "ret_20d", "vol_20d",
    "volume_ratio_20d", "rsi_14",
    "ret_vs_iyt_5d", "ret_vs_spy_5d",
    # Macro
    "vix_level", "vix_change_5d", "diesel_change_4w",
    "ism_pmi", "dgs10", "claims_change_4w",
    # Professional
    "rolling_beta_60d", "vol_regime", "momentum_rank", "mean_reversion_5d",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Merged panel with index (date, ticker).

    Returns
    -------
    pd.DataFrame
        Same panel with feature columns appended + target column.
    """
    print("[features] Building core features ...")
    df = add_return_features(df)
    df = add_volatility(df)
    df = add_volume_ratio(df)
    df = add_rsi(df)
    df = add_relative_strength(df)

    print("[features] Building macro features ...")
    df = add_macro_features(df)

    print("[features] Building professional features ...")
    df = add_rolling_beta(df)
    df = add_vol_regime(df)
    df = add_momentum_rank(df)
    df = add_mean_reversion(df)

    print("[features] Computing target variable ...")
    df = add_target(df)

    # Drop burn-in rows where rolling features are NaN
    initial_len = len(df)
    df = df.dropna(subset=["vol_20d", "ret_20d"])  # Core features that take longest
    print(f"[features] Dropped {initial_len - len(df)} burn-in rows.  Final shape: {df.shape}")

    return df
