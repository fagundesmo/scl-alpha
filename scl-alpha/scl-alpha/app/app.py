"""
Streamlit app entrypoint for scl-alpha.

Run with:
    streamlit run app/app.py

Or via Procfile on Railway:
    web: streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
"""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "scl-alpha requires Python 3.11 or newer. "
        f"Detected {sys.version.split()[0]}."
    )

import streamlit as st

st.set_page_config(
    page_title="SCL-Alpha - ML Trading Signals",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("SCL-Alpha: ML-Driven Supply-Chain Trading Signals")

st.markdown("""
Welcome to **SCL-Alpha**, a sandbox project that applies machine learning
to quantitative trading research for publicly traded supply-chain and
logistics companies.

### How to use this app

Use the sidebar to navigate between pages:

- **Model Lab** - Choose one or more models, tune parameters, run backtests, and compare results.
- **Signals** - View this week's model predictions and rankings.
- **Backtest** - Explore historical strategy performance vs. benchmarks.
- **Explain** - See which features drove a specific prediction.
- **Data** - Browse summary statistics and feature correlations.

---

*This is a research prototype for educational purposes.
It is not investment advice.*
""")

# Sidebar info
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "**scl-alpha** v0.1.0\n\n"
    "ML-driven trading research for\n"
    "supply-chain & logistics stocks.\n\n"
    "[GitHub Repo](#) · [Blueprint](#)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Ticker Universe")
tickers = ["UPS", "FDX", "XPO", "CHRW", "JBHT", "UNP", "CSX", "MATX", "GXO", "EXPD"]
for t in tickers:
    st.sidebar.text(f"  {t}")
