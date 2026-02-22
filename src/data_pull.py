"""
Data collection module.

Pulls price data from Yahoo Finance and macro data from FRED.
Caches everything as parquet files in data/raw/ so we only hit the
APIs once (or when you explicitly ask for a refresh).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import pandas as pd
import yfinance as yf
from fredapi import Fred

from src.config import (
    ALL_SYMBOLS,
    DATA_END,
    DATA_PROCESSED,
    DATA_RAW,
    DATA_START,
    FRED_API_KEY,
    FRED_SERIES,
)


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

def fetch_prices(
    symbols: list[str] | None = None,
    start: str = DATA_START,
    end: str = DATA_END,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for *symbols* from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (field, ticker) where field is one of
        Open, High, Low, Close, Adj Close, Volume.
        Index is DatetimeIndex (daily, business days).
    """
    symbols = symbols or ALL_SYMBOLS
    cache_path = DATA_RAW / "prices.parquet"

    if cache and cache_path.exists():
        print(f"[data_pull] Loading cached prices from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"[data_pull] Downloading prices for {len(symbols)} symbols ...")
    df = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=False,   # keep both Close and Adj Close
        threads=True,
    )

    # yfinance >=0.2.36 returns MultiIndex columns: (field, ticker)
    # For a single ticker it may flatten — guard against that.
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, symbols])

    if cache:
        df.to_parquet(cache_path)
        print(f"[data_pull] Cached prices -> {cache_path}")

    return df


def prices_to_long(prices_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide MultiIndex price DataFrame into a long panel:
    index = (date, ticker), columns = [open, high, low, close, adj_close, volume].

    This is the canonical format used by all downstream code.
    """
    adj = prices_wide.stack(level=1, future_stack=True)
    adj.index.names = ["date", "ticker"]

    # Normalize column names
    adj.columns = [c.lower().replace(" ", "_") for c in adj.columns]
    adj = adj.sort_index()
    return adj


# ---------------------------------------------------------------------------
# FRED
# ---------------------------------------------------------------------------

def fetch_fred(
    series: dict[str, str] | None = None,
    start: str = DATA_START,
    end: str = DATA_END,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download macro series from FRED.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex, one column per series ID (e.g. VIXCLS).
    """
    series = series or FRED_SERIES
    cache_path = DATA_RAW / "fred.parquet"

    if cache and cache_path.exists():
        print(f"[data_pull] Loading cached FRED data from {cache_path}")
        return pd.read_parquet(cache_path)

    if not FRED_API_KEY:
        raise ValueError(
            "FRED_API_KEY is missing. Set it in your environment or .env file at PROJECT_ROOT/.env "
            "(example: FRED_API_KEY=your_key). Get a free key at "
            "https://fred.stlouisfed.org/docs/api/api_key.html."
        )

    fred = Fred(api_key=FRED_API_KEY)
    frames: dict[str, pd.Series] = {}
    skipped: list[str] = []
    for sid, label in series.items():
        print(f"[data_pull] Fetching FRED series {sid} ({label}) ...")
        try:
            s = fred.get_series(sid, observation_start=start, observation_end=end)
            frames[sid] = s
        except Exception as exc:
            skipped.append(sid)
            print(f"[data_pull] WARNING: skipping series {sid} ({label}) due to: {exc}")

    if not frames:
        raise ValueError(
            "No FRED series could be downloaded. Verify your FRED_API_KEY and series IDs."
        )

    if skipped:
        print(f"[data_pull] Loaded {len(frames)} series; skipped {len(skipped)} unavailable series: {skipped}")

    df = pd.DataFrame(frames)
    df.index.name = "date"

    if cache:
        df.to_parquet(cache_path)
        print(f"[data_pull] Cached FRED data -> {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Fama-French Factors (Kenneth French Data Library)
# ---------------------------------------------------------------------------

def _download_ff_csv_from_zip(url: str) -> pd.DataFrame:
    """Robust fallback downloader for Kenneth French zipped CSV files."""
    with urllib.request.urlopen(url, timeout=60) as response:
        raw = response.read()

    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("No CSV file found in Fama-French archive.")

        # The archive contains a single CSV in standard releases.
        with zf.open(csv_names[0]) as fh:
            data = pd.read_csv(fh, skiprows=3)

    first_col = data.columns[0]
    data = data.rename(columns={first_col: "date"})
    data["date"] = pd.to_numeric(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])

    # Keep rows that look like YYYYMMDD dates.
    data["date"] = data["date"].astype("int64")
    data = data[data["date"] >= 19000101]
    data["date"] = pd.to_datetime(data["date"].astype(str), format="%Y%m%d", errors="coerce")
    data = data.dropna(subset=["date"]).set_index("date")

    data.columns = [str(c).strip() for c in data.columns]
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML", "RF"] if c in data.columns]
    if not factor_cols:
        raise ValueError("Expected FF factor columns not found in CSV archive.")

    data = data[factor_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
    return data / 100.0  # Convert percent to decimal


def fetch_fama_french(cache: bool = True) -> pd.DataFrame:
    """
    Download daily Fama-French 3-factor data (Mkt-RF, SMB, HML, RF).

    Uses pandas-datareader when available, and falls back to a robust
    direct zip download parser if needed.

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex (daily), columns = [Mkt-RF, SMB, HML, RF].
        Values are in decimal (e.g. 0.01 = 1%).
    """
    cache_path = DATA_RAW / "fama_french.parquet"

    if cache and cache_path.exists():
        print(f"[data_pull] Loading cached Fama-French data from {cache_path}")
        return pd.read_parquet(cache_path)

    try:
        import pandas_datareader.data as web

        ff = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            start=DATA_START,
        )
        df = ff[0] / 100.0  # percent -> decimal
    except Exception as exc:
        url = "https://mba.tufts.edu/~kfrench/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        print(
            f"[data_pull] pandas-datareader path failed ({exc}); "
            "falling back to direct Kenneth French download."
        )
        df = _download_ff_csv_from_zip(url)

    df.index.name = "date"

    if cache:
        df.to_parquet(cache_path)
        print(f"[data_pull] Cached Fama-French data -> {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------

def merge_all_data(
    prices_long: pd.DataFrame,
    fred_df: pd.DataFrame,
    ff_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge the long-format price panel with FRED macro data and
    (optionally) Fama-French factors.

    FRED and FF data are forward-filled to daily frequency before merging,
    which is the standard approach: the most recent known value is used
    until a new release arrives.

    Returns
    -------
    pd.DataFrame
        index = (date, ticker), columns include price fields + macro fields.
    """
    bdays = pd.bdate_range(start=fred_df.index.min(), end=fred_df.index.max())
    fred_daily = fred_df.reindex(bdays).ffill()
    fred_daily.index.name = "date"

    merged = prices_long.reset_index()
    merged["date"] = pd.to_datetime(merged["date"])
    fred_daily = fred_daily.reset_index().rename(columns={"index": "date"})
    merged = merged.merge(fred_daily, on="date", how="left")

    if ff_df is not None:
        ff_daily = ff_df.reindex(bdays).ffill().reset_index().rename(
            columns={"index": "date"}
        )
        merged = merged.merge(ff_daily, on="date", how="left")

    merged = merged.set_index(["date", "ticker"]).sort_index()
    return merged


# ---------------------------------------------------------------------------
# Convenience: pull everything in one call
# ---------------------------------------------------------------------------

def pull_all(cache: bool = True) -> pd.DataFrame:
    """
    One-liner to download + merge all data into a single panel DataFrame.
    """
    prices_wide = fetch_prices(cache=cache)
    prices_long = prices_to_long(prices_wide)
    try:
        fred_df = fetch_fred(cache=cache)
    except Exception as exc:
        print(f"[data_pull] FRED download failed ({exc}); proceeding without FRED macro series.")
        fallback_dates = prices_long.index.get_level_values("date").unique().sort_values()
        fred_df = pd.DataFrame(index=fallback_dates)
        fred_df.index.name = "date"

    try:
        ff_df = fetch_fama_french(cache=cache)
    except Exception as exc:
        print(f"[data_pull] Fama-French download failed ({exc}); proceeding without it.")
        ff_df = None

    merged = merge_all_data(prices_long, fred_df, ff_df)

    out_path = DATA_PROCESSED / "panel.parquet"
    merged.to_parquet(out_path)
    print(f"[data_pull] Merged panel saved -> {out_path}  shape={merged.shape}")

    return merged
