"""
Walk-forward backtesting engine.

Implements the loop described in the blueprint:
    signal (Friday close) → positions (top-K) → returns (Monday open → Friday close) → costs

Key design decisions:
    - Rebalance weekly (every Friday signal, execute Monday open).
    - Expanding training window, retrained every 13 weeks.
    - Transaction costs and slippage applied on every trade.
"""

import numpy as np
import pandas as pd
from datetime import timedelta

from src.config import (
    TICKERS, TOP_K, FORWARD_RETURN_DAYS,
    TRANSACTION_COST_BPS, SLIPPAGE_BPS,
    RETRAIN_EVERY_WEEKS, INITIAL_TRAIN_YEARS,
    RANDOM_SEED,
)
from src.features import FEATURE_COLUMNS
from src.model import make_model, train_model, predict


# ---------------------------------------------------------------------------
# Identify rebalance dates
# ---------------------------------------------------------------------------

def get_rebalance_dates(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Return every Friday in the dataset's date range.

    Only dates that actually appear in the data index are returned,
    so market holidays are skipped.
    """
    dates = df.index.get_level_values("date").unique().sort_values()
    fridays = dates[dates.dayofweek == 4]  # 0=Mon … 4=Fri
    return fridays


# ---------------------------------------------------------------------------
# Single-period portfolio return
# ---------------------------------------------------------------------------

def _period_return(
    df: pd.DataFrame,
    signal_date: pd.Timestamp,
    holdings: list[str],
) -> float:
    """
    Compute the equal-weight gross return (%) of *holdings* from the Monday
    open after *signal_date* to the following Friday close.

    Returns
    -------
    float
        Gross return in percent, or np.nan if insufficient future data.

    Notes
    -----
    Turnover is NOT computed here — it is computed in run_backtest() based
    on the change in holdings between consecutive periods, which is the
    correct place because it requires knowledge of the previous holdings.
    """
    # Next Monday = signal_date + 3 calendar days (Fri → Mon)
    exec_date = signal_date + timedelta(days=3)
    # Find the actual next trading day on or after exec_date
    dates = df.index.get_level_values("date").unique().sort_values()
    future_dates = dates[dates >= exec_date]

    # Need at least FORWARD_RETURN_DAYS trading days to compute the return
    if len(future_dates) < FORWARD_RETURN_DAYS:
        return np.nan

    entry_date = future_dates[0]
    exit_date = future_dates[FORWARD_RETURN_DAYS - 1]  # e.g. index 4 for 5-day hold

    returns = []
    for ticker in holdings:
        try:
            entry_price = df.loc[(entry_date, ticker), "open"]
            exit_price = df.loc[(exit_date, ticker), "adj_close"]
            ret = (exit_price / entry_price) - 1.0
            returns.append(ret)
        except KeyError:
            continue

    if not returns:
        return 0.0

    return np.mean(returns) * 100.0  # equal-weight, in %


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    df: pd.DataFrame,
    model_name: str = "ridge",
    top_k: int = TOP_K,
    cost_bps: float = TRANSACTION_COST_BPS,
    slippage_bps: float = SLIPPAGE_BPS,
    retrain_every: int = RETRAIN_EVERY_WEEKS,
    initial_train_years: int = INITIAL_TRAIN_YEARS,
    tickers: list[str] | None = None,
    model_params: dict | None = None,
) -> pd.DataFrame:
    """
    Run the full walk-forward backtest.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched panel with (date, ticker) index.
        Must contain FEATURE_COLUMNS and 'target_ret_5d_fwd'.
    model_name : str
        One of 'ridge', 'rf', 'xgboost'.
    top_k : int
        Number of stocks to hold long.
    cost_bps, slippage_bps : float
        Transaction cost and slippage in basis points.
    retrain_every : int
        Retrain the model every N rebalance dates.
    initial_train_years : int
        Minimum years of data before first prediction.
    model_params : dict | None
        Optional model-specific parameter overrides passed to make_model().

    Returns
    -------
    pd.DataFrame
        One row per rebalance date with columns:
        [date, holdings, predicted_returns, gross_return, net_return,
         cumulative_return, turnover].
    """
    tickers = tickers or TICKERS
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Filter to tradable tickers only (exclude benchmarks)
    trade_df = df[df.index.get_level_values("ticker").isin(tickers)].copy()

    rebal_dates = get_rebalance_dates(trade_df)
    all_dates = trade_df.index.get_level_values("date").unique().sort_values()

    # Determine first valid signal date (after burn-in)
    min_date = all_dates[0]
    first_signal_date = min_date + pd.DateOffset(years=initial_train_years)
    rebal_dates = rebal_dates[rebal_dates >= first_signal_date]

    if len(rebal_dates) == 0:
        raise ValueError("No rebalance dates after the initial training period.")

    print(f"[backtest] {len(rebal_dates)} rebalance dates from "
          f"{rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    model = None
    results = []
    prev_holdings: list[str] = []

    for i, sig_date in enumerate(rebal_dates):
        # -----------------------------------------------------------
        # (Re)train model if needed
        # -----------------------------------------------------------
        if model is None or i % retrain_every == 0:
            train_mask = trade_df.index.get_level_values("date") < sig_date
            train_data = trade_df[train_mask].dropna(subset=feature_cols + ["target_ret_5d_fwd"])

            if len(train_data) < 100:
                continue  # Not enough data yet

            X_train = train_data[feature_cols]
            y_train = train_data["target_ret_5d_fwd"]

            model = make_model(model_name, params=model_params)
            model = train_model(model, X_train, y_train)
            print(f"  [backtest] Retrained {model_name} at {sig_date.date()} "
                  f"(train size: {len(X_train)})")

        # -----------------------------------------------------------
        # Generate predictions for signal date
        # -----------------------------------------------------------
        sig_mask = trade_df.index.get_level_values("date") == sig_date
        sig_data = trade_df[sig_mask].dropna(subset=feature_cols)

        if len(sig_data) == 0:
            continue

        preds = predict(model, sig_data)
        sig_data = sig_data.copy()
        sig_data["predicted_ret"] = preds

        # -----------------------------------------------------------
        # Select top-K with predicted return > 0
        # -----------------------------------------------------------
        ranked = (
            sig_data.reset_index()
            .sort_values("predicted_ret", ascending=False)
        )
        longs = ranked[ranked["predicted_ret"] > 0].head(top_k)["ticker"].tolist()

        # -----------------------------------------------------------
        # Compute period return
        # -----------------------------------------------------------
        if longs:
            gross_ret = _period_return(df, sig_date, longs)
        else:
            gross_ret = 0.0  # All cash

        # Transaction cost: proportional to turnover
        changed = set(longs) ^ set(prev_holdings)
        trade_frac = len(changed) / max(top_k, 1)
        cost = trade_frac * (cost_bps + slippage_bps) / 100.0  # in %
        net_ret = gross_ret - cost if not np.isnan(gross_ret) else np.nan

        results.append({
            "date": sig_date,
            "holdings": longs,
            "n_holdings": len(longs),
            "gross_return_pct": gross_ret,
            "net_return_pct": net_ret,
            "turnover_frac": trade_frac,
            "cost_pct": cost,
        })

        prev_holdings = longs

    # -----------------------------------------------------------
    # Assemble results
    # -----------------------------------------------------------
    res_df = pd.DataFrame(results)

    if len(res_df) == 0:
        print("[backtest] WARNING: No trades generated.")
        return res_df

    res_df = res_df.dropna(subset=["net_return_pct"])
    res_df["cumulative_return"] = (1 + res_df["net_return_pct"] / 100).cumprod()
    res_df = res_df.set_index("date")

    print(f"[backtest] Done.  {len(res_df)} periods, "
          f"final cumulative return: {res_df['cumulative_return'].iloc[-1]:.3f}x")

    return res_df


# ---------------------------------------------------------------------------
# Baseline: buy-and-hold benchmark
# ---------------------------------------------------------------------------

def buy_and_hold_baseline(
    df: pd.DataFrame,
    ticker: str = "IYT",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Simple buy-and-hold equity curve for a single ticker.
    Returns a DataFrame with columns [date, cumulative_return].
    """
    bench = df.xs(ticker, level="ticker")["adj_close"].sort_index()
    if start:
        bench = bench[bench.index >= start]
    if end:
        bench = bench[bench.index <= end]

    bench = bench / bench.iloc[0]
    return bench.to_frame("cumulative_return")
