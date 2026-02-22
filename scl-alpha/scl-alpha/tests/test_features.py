"""
Unit tests for feature engineering.

The most critical test: NO LOOK-AHEAD BIAS.
We verify that adding future data does not change features computed at time t.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers: create a small synthetic dataset
# ---------------------------------------------------------------------------

def _make_synthetic_panel(n_days: int = 100, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    Generate a synthetic price panel for testing.
    Prices follow a random walk; volume is random.
    """
    tickers = tickers or ["AAA", "BBB"]
    np.random.seed(42)

    dates = pd.bdate_range(start="2020-01-01", periods=n_days)
    rows = []
    for ticker in tickers:
        price = 100.0
        for d in dates:
            ret = np.random.normal(0.0005, 0.02)
            price *= (1 + ret)
            rows.append({
                "date": d,
                "ticker": ticker,
                "open": price * (1 + np.random.normal(0, 0.005)),
                "high": price * (1 + abs(np.random.normal(0, 0.01))),
                "low": price * (1 - abs(np.random.normal(0, 0.01))),
                "close": price,
                "adj_close": price,
                "volume": int(np.random.uniform(1e6, 5e6)),
            })

    df = pd.DataFrame(rows).set_index(["date", "ticker"])

    # Add dummy macro columns (constant for simplicity)
    df["VIXCLS"] = 20.0
    df["GASDESW"] = 3.50
    df["NAPM"] = 52.0
    df["DGS10"] = 4.0
    df["ICNSA"] = 200_000

    return df


# ---------------------------------------------------------------------------
# Test: features don't change when future data is appended
# ---------------------------------------------------------------------------

class TestLookAheadBias:
    """Verify that features at date t are identical whether or not
    data after t exists in the DataFrame."""

    def test_returns_no_lookahead(self):
        from src.features import add_return_features

        df_full = _make_synthetic_panel(100)
        df_truncated = df_full.loc[df_full.index.get_level_values("date") <= df_full.index.get_level_values("date").unique()[59]]

        df_full = add_return_features(df_full.copy())
        df_truncated = add_return_features(df_truncated.copy())

        # Compare at the last date of the truncated set
        last_date = df_truncated.index.get_level_values("date").max()
        for col in ["ret_1d", "ret_5d", "ret_20d"]:
            full_val = df_full.loc[last_date][col]
            trunc_val = df_truncated.loc[last_date][col]
            pd.testing.assert_series_equal(full_val, trunc_val, check_names=False)

    def test_volatility_no_lookahead(self):
        from src.features import add_return_features, add_volatility

        df_full = _make_synthetic_panel(100)
        df_truncated = df_full.loc[df_full.index.get_level_values("date") <= df_full.index.get_level_values("date").unique()[59]]

        df_full = add_volatility(add_return_features(df_full.copy()))
        df_truncated = add_volatility(add_return_features(df_truncated.copy()))

        last_date = df_truncated.index.get_level_values("date").max()
        full_val = df_full.loc[last_date]["vol_20d"]
        trunc_val = df_truncated.loc[last_date]["vol_20d"]
        pd.testing.assert_series_equal(full_val, trunc_val, check_names=False)

    def test_rsi_no_lookahead(self):
        from src.features import add_rsi

        df_full = _make_synthetic_panel(100)
        df_truncated = df_full.loc[df_full.index.get_level_values("date") <= df_full.index.get_level_values("date").unique()[59]]

        df_full = add_rsi(df_full.copy())
        df_truncated = add_rsi(df_truncated.copy())

        last_date = df_truncated.index.get_level_values("date").max()
        full_val = df_full.loc[last_date]["rsi_14"]
        trunc_val = df_truncated.loc[last_date]["rsi_14"]
        pd.testing.assert_series_equal(full_val, trunc_val, check_names=False)


# ---------------------------------------------------------------------------
# Test: target variable uses future data (by design)
# ---------------------------------------------------------------------------

class TestTarget:
    """The target IS supposed to use future data — verify it does."""

    def test_target_is_forward_looking(self):
        from src.features import add_target

        df = _make_synthetic_panel(50)
        df = add_target(df.copy(), horizon=5)

        # The last `horizon` rows per ticker must have NaN target because
        # there is no future data to compute the forward return from.
        # Use pd.isna() — the "x != x" trick works for IEEE 754 floats but
        # is fragile with pandas types and harder to read.
        for ticker in ["AAA", "BBB"]:
            ticker_data = df.xs(ticker, level="ticker")
            assert pd.isna(ticker_data["target_ret_5d_fwd"].iloc[-1]), (
                f"Expected NaN target for last row of {ticker}, "
                f"got {ticker_data['target_ret_5d_fwd'].iloc[-1]}"
            )

    def test_target_not_in_feature_list(self):
        from src.features import FEATURE_COLUMNS
        assert "target_ret_5d_fwd" not in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Test: feature column list is consistent
# ---------------------------------------------------------------------------

class TestFeatureConsistency:
    """Ensure the declared feature list matches what build_features produces."""

    def test_build_features_produces_expected_columns(self):
        from src.features import build_features, FEATURE_COLUMNS

        df = _make_synthetic_panel(300)  # Need enough for burn-in
        # Add SPY and IYT as dummy tickers for relative strength
        df_spy = _make_synthetic_panel(300, tickers=["SPY", "IYT"])
        df = pd.concat([df, df_spy])

        result = build_features(df)

        for col in FEATURE_COLUMNS:
            # Some columns may be NaN if lookback is longer than data,
            # but the column itself should exist
            assert col in result.columns, f"Missing feature column: {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
