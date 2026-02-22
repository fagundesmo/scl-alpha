"""
Evaluation metrics — both ML and trading.

ML metrics operate on (y_true, y_pred) arrays.
Trading metrics operate on the backtest results DataFrame.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ===================================================================
# ML Metrics
# ===================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error (in same units as target, i.e. %)."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Spearman rank correlation between predicted and realized returns.
    IC > 0.05 is meaningful in practice.
    """
    if len(y_true) < 3:
        return np.nan
    corr, _ = stats.spearmanr(y_true, y_pred)
    return float(corr)


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions where sign(predicted) == sign(realized)."""
    correct = np.sign(y_pred) == np.sign(y_true)
    return float(np.mean(correct))


def compute_ml_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute all ML metrics and return as a dict."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "IC": information_coefficient(y_true, y_pred),
        "Hit Rate": hit_rate(y_true, y_pred),
    }


# ===================================================================
# Trading Metrics
# ===================================================================

def cagr(cum_returns: pd.Series, periods_per_year: float = 52.0) -> float:
    """
    Compound Annual Growth Rate.

    Parameters
    ----------
    cum_returns : pd.Series
        Cumulative return series (starting at 1.0).
    periods_per_year : float
        52 for weekly rebalancing.
    """
    n_periods = len(cum_returns)
    if n_periods < 2:
        return 0.0
    total_return = cum_returns.iloc[-1] / cum_returns.iloc[0]
    years = n_periods / periods_per_year
    return float(total_return ** (1 / years) - 1)


def sharpe_ratio(
    period_returns_pct: pd.Series,
    risk_free_annual: float = 0.0,
    periods_per_year: float = 52.0,
) -> float:
    """
    Annualised Sharpe ratio.

    Parameters
    ----------
    period_returns_pct : pd.Series
        Per-period returns in percent (e.g. 1.5 means +1.5%).
    """
    if len(period_returns_pct) < 2:
        return 0.0
    r = period_returns_pct / 100.0  # Convert to decimal
    rf_per_period = risk_free_annual / periods_per_year
    excess = r - rf_per_period
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods_per_year))


def max_drawdown(cum_returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (as a positive fraction, e.g. 0.15 = 15%)."""
    peak = cum_returns.cummax()
    dd = (cum_returns - peak) / peak
    return float(-dd.min())


def profit_factor(period_returns_pct: pd.Series) -> float:
    """Gross profit / gross loss.  > 1 means profitable overall."""
    gains = period_returns_pct[period_returns_pct > 0].sum()
    losses = -period_returns_pct[period_returns_pct < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def avg_turnover(backtest_df: pd.DataFrame) -> float:
    """Average weekly turnover fraction."""
    if "turnover_frac" in backtest_df.columns:
        return float(backtest_df["turnover_frac"].mean())
    return np.nan


def compute_trading_metrics(backtest_df: pd.DataFrame) -> dict[str, float]:
    """
    Compute all trading metrics from a backtest results DataFrame.

    Expects columns: 'net_return_pct', 'cumulative_return', 'turnover_frac'.
    """
    cum = backtest_df["cumulative_return"]
    net = backtest_df["net_return_pct"]

    return {
        "CAGR": cagr(cum),
        "Sharpe Ratio": sharpe_ratio(net),
        "Max Drawdown": max_drawdown(cum),
        "Profit Factor": profit_factor(net),
        "Avg Turnover": avg_turnover(backtest_df),
        "Total Periods": len(backtest_df),
        "Positive Periods": int((net > 0).sum()),
        "Negative Periods": int((net < 0).sum()),
    }
