"""
Signals page for current model predictions and rankings.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Signals", page_icon="📊", layout="wide")
st.title("📊 Weekly Trading Signals")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
PREDICTIONS_PATH = CACHE_DIR / "latest_predictions.parquet"


@st.cache_data(ttl=3600)
def load_predictions() -> pd.DataFrame | None:
    if PREDICTIONS_PATH.exists():
        return pd.read_parquet(PREDICTIONS_PATH)
    return None


runs = st.session_state.get("model_runs", {})
active_model = st.session_state.get("active_model")
predictions = None

if runs:
    model_names = list(runs.keys())
    idx = model_names.index(active_model) if active_model in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=idx, key="signals_model")
    st.session_state["active_model"] = selected_model
    predictions = runs[selected_model].get("latest_predictions")
    st.caption(f"Using in-memory run output for model: {selected_model}")

if predictions is None:
    predictions = load_predictions()

if predictions is None or predictions.empty:
    st.warning(
        "No predictions found. Open 'model lab' and run one or more models, "
        "or upload app/cache/latest_predictions.parquet."
    )
    st.stop()

predictions = predictions.copy()
predictions["date"] = pd.to_datetime(predictions["date"])

available_dates = sorted(predictions["date"].unique(), reverse=True)
selected_date = st.selectbox(
    "Signal date",
    options=available_dates,
    format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d (%A)"),
)

current = predictions[predictions["date"] == selected_date].copy()
current = current.sort_values("predicted_ret", ascending=False).reset_index(drop=True)

current["rank"] = range(1, len(current) + 1)
current["signal"] = np.where(current["predicted_ret"] > 0, "LONG", "FLAT")

display_df = current[["rank", "ticker", "predicted_ret", "signal"]].rename(
    columns={
        "rank": "Rank",
        "ticker": "Ticker",
        "predicted_ret": "Predicted 5d Return (%)",
        "signal": "Signal",
    }
)

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Signal Table")
    st.dataframe(
        display_df.style.format({"Predicted 5d Return (%)": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )

    n_long = int((current["signal"] == "LONG").sum())
    st.metric("Holdings this week", f"{n_long} / {len(current)} stocks")

with col2:
    st.subheader("Predicted Returns")
    try:
        from src.plots import plotly_signal_bars

        fig = plotly_signal_bars(current[["ticker", "predicted_ret"]])
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        chart_data = current.set_index("ticker")["predicted_ret"].sort_values()
        st.bar_chart(chart_data)

latest = pd.Timestamp(available_dates[0])
age_days = (pd.Timestamp.now(tz=None) - latest).days
if age_days > 7:
    st.warning(
        f"Data may be stale. Last signal is from {latest.strftime('%Y-%m-%d')} "
        f"({age_days} days ago)."
    )
