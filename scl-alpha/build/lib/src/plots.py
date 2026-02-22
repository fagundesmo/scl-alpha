"""
Visualization module.

All plots are designed for two audiences:
    1. The quant researcher (detailed, in the notebook).
    2. A non-quant stakeholder (simple, in the Streamlit app).

Uses matplotlib for static notebook charts and plotly for interactive app charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Try plotly for interactive charts (used in the Streamlit app)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ===================================================================
# Color palette
# ===================================================================
COLORS = {
    "strategy": "#1f77b4",
    "benchmark": "#ff7f0e",
    "equal_weight": "#2ca02c",
    "cash": "#d3d3d3",
    "positive": "#2ca02c",
    "negative": "#d62728",
}


# ===================================================================
# 1. Equity Curve Comparison (Stakeholder Visual #1)
# ===================================================================

def plot_equity_curve(
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series | None = None,
    title: str = "Strategy vs Benchmark — Growth of $10,000",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Line chart: cumulative returns scaled to $10,000 starting value.
    Shaded drawdown region for the strategy.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scale to $10k
    strat = strategy_cum / strategy_cum.iloc[0] * 10_000
    ax.plot(strat.index, strat.values, label="ML Strategy",
            color=COLORS["strategy"], linewidth=2)

    # Drawdown shading
    peak = strat.cummax()
    ax.fill_between(strat.index, strat, peak,
                    alpha=0.15, color=COLORS["negative"], label="Drawdown")

    if benchmark_cum is not None:
        bench = benchmark_cum / benchmark_cum.iloc[0] * 10_000
        ax.plot(bench.index, bench.values, label="Buy & Hold IYT",
                color=COLORS["benchmark"], linewidth=1.5, linestyle="--")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    return fig


# ===================================================================
# 2. Weekly Signal Heatmap (Stakeholder Visual #2)
# ===================================================================

def plot_signal_heatmap(
    predictions: pd.DataFrame,
    title: str = "Weekly Predicted Returns (%)",
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """
    Heatmap: tickers (y-axis) x weeks (x-axis), color = predicted return.
    Green = positive, red = negative, intensity = magnitude.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must have columns ['date', 'ticker', 'predicted_ret'].
    """
    pivot = predictions.pivot_table(
        index="ticker", columns="date", values="predicted_ret"
    )

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 3)

    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Predicted 5-day Return (%)"},
        ax=ax,
    )
    # Simplify x-axis labels (show every 4th date).
    # Seaborn heatmap tick positions are floats centred at integer column indices.
    # Using round() rather than int() avoids off-by-one misalignment when
    # matplotlib places ticks at e.g. 3.5 instead of 4.
    xticks = ax.get_xticks()
    xlabels = [
        pivot.columns[round(t)].strftime("%Y-%m")
        if 0 <= round(t) < len(pivot.columns)
        else ""
        for t in xticks
    ]
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    ax.set_xlabel("")
    fig.tight_layout()
    return fig


# ===================================================================
# 3. Correlation Heatmap
# ===================================================================

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Return Correlations",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Correlation matrix of daily returns (or any numeric columns)."""
    if columns:
        df = df[columns]
    corr = df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ===================================================================
# 4. Feature Importance (Waterfall-style)
# ===================================================================

def plot_feature_importance(
    importance: pd.Series,
    top_n: int = 15,
    title: str = "Feature Importance",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Horizontal bar chart of feature importances / coefficients."""
    imp = importance.head(top_n).sort_values()

    fig, ax = plt.subplots(figsize=figsize)
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in imp]
    ax.barh(imp.index, imp.values, color=colors, edgecolor="white")
    ax.set_xlabel("Importance / Coefficient")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    sns.despine()
    fig.tight_layout()
    return fig


# ===================================================================
# 5. Cumulative Returns (Multiple Tickers)
# ===================================================================

def plot_cumulative_returns(
    prices_long: pd.DataFrame,
    tickers: list[str],
    title: str = "Cumulative Returns",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Line chart of cumulative returns for multiple tickers."""
    fig, ax = plt.subplots(figsize=figsize)

    for ticker in tickers:
        try:
            series = prices_long.xs(ticker, level="ticker")["adj_close"].sort_index()
            cum = series / series.iloc[0]
            ax.plot(cum.index, cum.values, label=ticker, linewidth=1.2)
        except KeyError:
            continue

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Growth of $1")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    sns.despine()
    fig.tight_layout()
    return fig


# ===================================================================
# 6. Distribution of Target Variable
# ===================================================================

def plot_target_distribution(
    target: pd.Series,
    title: str = "Distribution of 5-Day Forward Returns",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Histogram + KDE of the target variable."""
    fig, ax = plt.subplots(figsize=figsize)
    target_clean = target.dropna()
    ax.hist(target_clean, bins=80, density=True, alpha=0.5,
            color=COLORS["strategy"], edgecolor="white")
    target_clean.plot.kde(ax=ax, color=COLORS["strategy"], linewidth=2)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("5-Day Forward Return (%)")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=14, fontweight="bold")
    sns.despine()
    fig.tight_layout()
    return fig


# ===================================================================
# Plotly (interactive, for Streamlit)
# ===================================================================

def plotly_equity_curve(
    strategy_cum: pd.Series,
    benchmark_cum: pd.Series | None = None,
) -> "go.Figure":
    """Interactive equity curve for the Streamlit app."""
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive charts.")

    fig = go.Figure()

    strat = strategy_cum / strategy_cum.iloc[0] * 10_000
    fig.add_trace(go.Scatter(
        x=strat.index, y=strat.values,
        mode="lines", name="ML Strategy",
        line=dict(color=COLORS["strategy"], width=2),
    ))

    if benchmark_cum is not None:
        bench = benchmark_cum / benchmark_cum.iloc[0] * 10_000
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench.values,
            mode="lines", name="Buy & Hold IYT",
            line=dict(color=COLORS["benchmark"], width=1.5, dash="dash"),
        ))

    fig.update_layout(
        title="Growth of $10,000",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
        height=450,
    )
    return fig


def plotly_signal_bars(predictions_row: pd.DataFrame) -> "go.Figure":
    """
    Horizontal bar chart of predicted returns for a single date.
    Used on the Signals page of the app.

    Parameters
    ----------
    predictions_row : pd.DataFrame
        Columns: ['ticker', 'predicted_ret'], one row per ticker.
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive charts.")

    df = predictions_row.sort_values("predicted_ret")
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"]
              for v in df["predicted_ret"]]

    fig = go.Figure(go.Bar(
        x=df["predicted_ret"],
        y=df["ticker"],
        orientation="h",
        marker_color=colors,
        text=df["predicted_ret"].round(2).astype(str) + "%",
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted 5-Day Returns (%)",
        xaxis_title="Predicted Return (%)",
        template="plotly_white",
        height=400,
    )
    return fig
