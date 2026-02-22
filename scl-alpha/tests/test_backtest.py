"""
Smoke tests for the backtesting engine.

These tests use synthetic data and verify that the backtest machinery
runs without errors and produces sensible output shapes.
They do NOT test profitability (that would be overfitting to the test).
"""

import numpy as np
import pandas as pd
import pytest


def _make_synthetic_panel(n_days: int = 400) -> pd.DataFrame:
    """Create a synthetic feature-enriched panel for backtesting."""
    np.random.seed(42)
    tickers = ["AAA", "BBB", "CCC"]
    dates = pd.bdate_range(start="2019-01-01", periods=n_days)

    rows = []
    for ticker in tickers:
        price = 100.0
        for d in dates:
            ret = np.random.normal(0.0003, 0.015)
            price *= (1 + ret)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * 0.999,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "adj_close": price,
                "volume": int(np.random.uniform(1e6, 3e6)),
                # Fake features
                "ret_1d": np.random.normal(0, 2),
                "ret_5d": np.random.normal(0, 4),
                "ret_20d": np.random.normal(0, 8),
                "vol_20d": abs(np.random.normal(20, 5)),
                "volume_ratio_20d": abs(np.random.normal(1, 0.3)),
                "rsi_14": np.random.uniform(30, 70),
                "ret_vs_iyt_5d": np.random.normal(0, 3),
                "ret_vs_spy_5d": np.random.normal(0, 3),
                "vix_level": 20 + np.random.normal(0, 3),
                "vix_change_5d": np.random.normal(0, 2),
                "diesel_change_4w": np.random.normal(0, 3),
                "ism_pmi": 50 + np.random.normal(0, 3),
                "dgs10": 4 + np.random.normal(0, 0.5),
                "claims_change_4w": np.random.normal(0, 10000),
                "rolling_beta_60d": np.random.normal(1, 0.3),
                "vol_regime": float(np.random.choice([0, 1])),
                "momentum_rank": float(np.random.choice([1, 2, 3])),
                "mean_reversion_5d": np.random.normal(0, 1),
                # Target
                "target_ret_5d_fwd": np.random.normal(0, 4),
            })

    return pd.DataFrame(rows).set_index(["date", "ticker"])


class TestBacktestSmoke:
    """Basic smoke tests: does the backtest run and produce valid output?"""

    def test_backtest_runs(self):
        from src.backtest import run_backtest

        df = _make_synthetic_panel(400)
        result = run_backtest(
            df,
            model_name="ridge",
            top_k=2,
            tickers=["AAA", "BBB", "CCC"],
            initial_train_years=1,
            retrain_every=13,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert "cumulative_return" in result.columns
        assert "net_return_pct" in result.columns

    def test_cumulative_return_starts_near_one(self):
        from src.backtest import run_backtest

        df = _make_synthetic_panel(400)
        result = run_backtest(
            df,
            model_name="ridge",
            top_k=2,
            tickers=["AAA", "BBB", "CCC"],
            initial_train_years=1,
        )

        # First cumulative return = 1 * (1 + first_period_net_return / 100).
        # Synthetic data uses N(0.0003, 0.015) daily returns → weekly returns
        # are typically within ±7%.  The bound of ±15% gives a comfortable
        # margin without being so wide it passes nonsensical values.
        first_cum = result["cumulative_return"].iloc[0]
        assert 0.85 < first_cum < 1.15, (
            f"First cumulative return {first_cum:.4f} is outside expected range "
            "[0.85, 1.15]; check the backtest cost or return calculation."
        )

    def test_costs_are_applied(self):
        from src.backtest import run_backtest

        df = _make_synthetic_panel(400)

        # High cost vs zero cost
        res_high = run_backtest(
            df, model_name="ridge", top_k=2,
            tickers=["AAA", "BBB", "CCC"],
            initial_train_years=1,
            cost_bps=50, slippage_bps=50,
        )
        res_zero = run_backtest(
            df, model_name="ridge", top_k=2,
            tickers=["AAA", "BBB", "CCC"],
            initial_train_years=1,
            cost_bps=0, slippage_bps=0,
        )

        # Net returns with high costs should be lower on average
        assert res_high["net_return_pct"].mean() <= res_zero["net_return_pct"].mean()


class TestMetrics:
    """Test that metric computations are correct on known inputs."""

    def test_sharpe_zero_vol(self):
        from src.metrics import sharpe_ratio
        # Constant returns → zero std → Sharpe = 0
        returns = pd.Series([1.0, 1.0, 1.0, 1.0])
        assert sharpe_ratio(returns) == 0.0

    def test_max_drawdown_no_loss(self):
        from src.metrics import max_drawdown
        # Monotonically increasing → 0 drawdown
        cum = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert max_drawdown(cum) == 0.0

    def test_max_drawdown_known(self):
        from src.metrics import max_drawdown
        # Peak at 2.0, trough at 1.0 → 50% drawdown
        cum = pd.Series([1.0, 2.0, 1.0, 1.5])
        assert abs(max_drawdown(cum) - 0.5) < 1e-10

    def test_hit_rate_perfect(self):
        from src.metrics import hit_rate
        y_true = np.array([1, -1, 1, -1])
        y_pred = np.array([2, -3, 0.5, -0.1])
        assert hit_rate(y_true, y_pred) == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
