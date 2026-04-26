"""
Data collection module.

Pulls OHLCV data from Yahoo Finance and macro series from FRED via
pandas-datareader (no API key required). Results are cached as parquet
files in data/raw/ so APIs are only hit once (or when refresh=True).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

from src.config import (
    ALL_SYMBOLS,
    BENCHMARK_ETF,
    DATA_RAW,
    DATA_START,
    FRED_SERIES,
    MARKET_ETF,
    TICKERS,
)


def _yesterday() -> str:
    return (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns from yfinance to simple field names."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------------------
# Company OHLCV  (long format: Date, Ticker, Open, High, Low, Close, Volume)
# ---------------------------------------------------------------------------

def fetch_company_ohlcv(
    tickers: list[str] | None = None,
    start: str = DATA_START,
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV for supply-chain tickers and return a long panel.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Ticker, Open, High, Low, Close, Volume
    """
    tickers = tickers or TICKERS
    end = end or _yesterday()
    cache_path = DATA_RAW / "companies_ohlcv.parquet"

    if cache and cache_path.exists():
        print("[data_pull] Loading cached company OHLCV")
        return pd.read_parquet(cache_path)

    print(f"[data_pull] Downloading OHLCV for {len(tickers)} tickers ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
    )

    # Convert wide MultiIndex → long format
    if isinstance(raw.columns, pd.MultiIndex):
        fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        lvl0 = set(raw.columns.get_level_values(0))
        if lvl0 & fields:
            # columns are (Field, Ticker)
            df = raw.stack(1).rename_axis(["Date", "Ticker"]).reset_index()
        else:
            # columns are (Ticker, Field)
            df = raw.stack(0).rename_axis(["Date", "Ticker"]).reset_index()
    else:
        df = raw.reset_index()
        df["Ticker"] = tickers[0]

    df["Date"] = pd.to_datetime(df["Date"])
    if getattr(df["Date"].dt, "tz", None) is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    keep = [c for c in ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].sort_values(["Ticker", "Date"]).reset_index(drop=True)

    if cache:
        df.to_parquet(cache_path)
        print(f"[data_pull] Cached company OHLCV -> {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Single-symbol ETF OHLCV (simple column names after normalization)
# ---------------------------------------------------------------------------

def fetch_etf_ohlcv(
    symbol: str,
    start: str = DATA_START,
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV for a single ETF symbol.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex, columns = [Open, High, Low, Close, Volume]
    """
    end = end or _yesterday()
    cache_path = DATA_RAW / f"{symbol.lower()}_ohlcv.parquet"

    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    print(f"[data_pull] Downloading OHLCV for {symbol} ...")
    raw = yf.download(
        [symbol],
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    ).dropna()

    raw = _normalize_ohlcv_columns(raw)
    raw.index = pd.to_datetime(raw.index)
    if getattr(raw.index, "tz", None) is not None:
        raw.index = raw.index.tz_localize(None)
    raw.index.name = "Date"

    if cache:
        raw.to_parquet(cache_path)

    return raw


# ---------------------------------------------------------------------------
# VIX  (daily close, Series)
# ---------------------------------------------------------------------------

def fetch_vix(
    start: str = DATA_START,
    end: str | None = None,
    cache: bool = True,
) -> pd.Series:
    """Download VIX daily close from Yahoo Finance."""
    end = end or _yesterday()
    cache_path = DATA_RAW / "vix_close.parquet"

    if cache and cache_path.exists():
        s = pd.read_parquet(cache_path).iloc[:, 0]
        s.name = "VIX"
        return s

    print("[data_pull] Downloading VIX ...")
    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna()
    close.name = "VIX"
    close.index = pd.to_datetime(close.index)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    close.index.name = "Date"

    if cache:
        close.to_frame().to_parquet(cache_path)

    return close


# ---------------------------------------------------------------------------
# FRED macro series
# ---------------------------------------------------------------------------

def fetch_fred(
    series: list[str] | None = None,
    start: str = DATA_START,
    end: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download macro series from FRED via pandas-datareader (no API key needed).

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex (daily, forward-filled), one column per series.
        Rows with all-NaN are dropped.
    """
    series = series or FRED_SERIES
    end = end or date.today().strftime("%Y-%m-%d")
    cache_path = DATA_RAW / "fred.parquet"

    if cache and cache_path.exists():
        print("[data_pull] Loading cached FRED data")
        return pd.read_parquet(cache_path)

    print(f"[data_pull] Downloading FRED series: {series} ...")
    idx = pd.date_range(start=start, end=end, freq="D")
    fred = pd.DataFrame(index=idx)

    for sid in series:
        try:
            s = pdr.DataReader(sid, "fred", start, end)
            fred = fred.join(s, how="left")
        except Exception as exc:
            print(f"[data_pull] WARNING: could not fetch {sid}: {exc}")

    fred = fred.ffill()
    fred_daily = fred.dropna(how="all")
    fred_daily.index.name = "Date"

    if cache:
        fred_daily.to_parquet(cache_path)
        print(f"[data_pull] Cached FRED data -> {cache_path}")

    return fred_daily


# ---------------------------------------------------------------------------
# Convenience: pull everything in one call
# ---------------------------------------------------------------------------

def pull_all(cache: bool = True, refresh: bool = False) -> dict:
    """
    Download all raw data and return a dict of DataFrames.

    Parameters
    ----------
    cache : bool
        Use cached parquet files when they exist.
    refresh : bool
        Delete all cached files before downloading (forces fresh pull).

    Returns
    -------
    dict with keys:
        companies_ohlcv, sector_etf, spy_ohlcv, vix_close, fred_daily
    """
    if refresh:
        for f in DATA_RAW.glob("*.parquet"):
            f.unlink()
            print(f"[data_pull] Removed cache: {f.name}")

    return {
        "companies_ohlcv": fetch_company_ohlcv(cache=cache),
        "sector_etf": fetch_etf_ohlcv(BENCHMARK_ETF, cache=cache),
        "spy_ohlcv": fetch_etf_ohlcv(MARKET_ETF, cache=cache),
        "vix_close": fetch_vix(cache=cache),
        "fred_daily": fetch_fred(cache=cache),
    }
