"""Home page for scl-alpha."""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "scl-alpha requires Python 3.11 or newer. "
        f"Detected {sys.version.split()[0]}."
    )

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(ttl=3600, show_spinner=False)
def load_latest_prices(symbols: tuple[str, ...]) -> tuple[str | None, dict[str, float], dict[str, float | None], str | None]:
    """Load latest daily close prices and 1-day change for display."""
    try:
        df = yf.download(
            list(symbols),
            period="7d",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        if df.empty:
            return None, {}, {}, "No market data returned from Yahoo Finance."

        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df[["Close"]].copy()
            close.columns = [symbols[0]]

        close = close.dropna(how="all")
        if close.empty:
            return None, {}, {}, "Close prices were empty."

        latest_date = close.index.max()
        latest_row = close.loc[latest_date]
        if not isinstance(latest_row, pd.Series):
            latest_row = pd.Series({symbols[0]: float(latest_row)})

        prices: dict[str, float] = {}
        for s in symbols:
            v = latest_row.get(s)
            prices[s] = float(v) if pd.notna(v) else float("nan")

        previous_rows = close.loc[close.index < latest_date]
        changes: dict[str, float | None] = {s: None for s in symbols}
        if len(previous_rows) > 0:
            prev_row = previous_rows.iloc[-1]
            for s in symbols:
                p = prices.get(s, float("nan"))
                prev = prev_row.get(s)
                if pd.notna(p) and pd.notna(prev) and float(prev) != 0.0:
                    changes[s] = (float(p) / float(prev) - 1.0) * 100.0

        return latest_date.strftime("%Y-%m-%d"), prices, changes, None

    except Exception as exc:
        return None, {}, {}, str(exc)


@st.cache_data(ttl=86400, show_spinner=False)
def load_analyst_consensus(symbols: tuple[str, ...]) -> dict[str, str]:
    """Load analyst consensus per ticker (Strong Buy/Buy/Hold/Sell/Strong Sell)."""
    out: dict[str, str] = {}
    for symbol in symbols:
        try:
            t = yf.Ticker(symbol)
            summary = getattr(t, "recommendations_summary", None)

            if not isinstance(summary, pd.DataFrame) or summary.empty:
                out[symbol] = "n/a"
                continue

            row = summary.iloc[0]
            counts = {
                "Strong Buy": float(row.get("strongBuy", 0) or 0),
                "Buy": float(row.get("buy", 0) or 0),
                "Hold": float(row.get("hold", 0) or 0),
                "Sell": float(row.get("sell", 0) or 0),
                "Strong Sell": float(row.get("strongSell", 0) or 0),
            }

            total = sum(counts.values())
            out[symbol] = max(counts, key=counts.get) if total > 0 else "n/a"
        except Exception:
            out[symbol] = "n/a"

    return out


st.set_page_config(
    page_title="Home - SCL-Alpha",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

tickers = ["UPS", "FDX", "XPO", "CHRW", "JBHT", "UNP", "CSX", "MATX", "GXO", "EXPD"]
as_of, prices, changes, price_error = load_latest_prices(tuple(tickers))
analyst_consensus = load_analyst_consensus(tuple(tickers))

st.title("SCL-Alpha: ML-Driven Supply-Chain Trading Signals")

st.subheader("Tracked Companies")
cols = st.columns(5)
for i, t in enumerate(tickers):
    p = prices.get(t)
    chg = changes.get(t)
    rec = analyst_consensus.get(t, "n/a")

    if p is None or pd.isna(p):
        price_text = "n/a"
    elif chg is None or pd.isna(chg):
        price_text = f"${p:,.2f}"
    else:
        price_text = f"${p:,.2f} ({chg:+.2f}%)"

    cols[i % 5].markdown(
        f"**{t}**  \n"
        f"Price: {price_text}  \n"
        f"Analyst: {rec}"
    )

left, center, right = st.columns([1, 2, 1])
with center:
    st.markdown("### How to Use the App")
    st.markdown(
        "1. Open **Model Lab** to select one or more models.\n"
        "2. Set portfolio settings and model parameters (single values or ranges).\n"
        "3. Click **Run selected models** and compare results.\n"
        "4. Publish your preferred model to use it in **Signals**, **Backtest**, **Explain**, and **Data** pages."
    )

st.markdown("---")
st.caption(
    "This is a research prototype for educational purposes. It is not investment advice."
)

# Sidebar info
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "**scl-alpha** v0.1.0\n\n"
    "ML-driven trading research for\n"
    "supply-chain & logistics stocks."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Ticker Universe")
for t in tickers:
    p = prices.get(t)
    rec = analyst_consensus.get(t, "n/a")

    if p is None or pd.isna(p):
        st.sidebar.text(f"{t}: n/a | Analyst: {rec}")
        continue

    chg = changes.get(t)
    if chg is None or pd.isna(chg):
        st.sidebar.text(f"{t}: ${p:,.2f} | Analyst: {rec}")
    else:
        st.sidebar.text(f"{t}: ${p:,.2f} ({chg:+.2f}%) | Analyst: {rec}")

if as_of:
    st.sidebar.caption(f"Latest close as of {as_of} (prices refresh hourly)")

st.sidebar.caption("Analyst consensus refreshes daily.")

if price_error:
    st.sidebar.caption("Price feed unavailable right now; showing cached/empty values.")
