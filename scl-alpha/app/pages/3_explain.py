"""
Explainability page for feature importance.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Explain", page_icon="🔍", layout="wide")
st.title("🔍 Prediction Explainability")

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
MODEL_PATH = CACHE_DIR / "best_model.joblib"
IMPORTANCE_PATH = CACHE_DIR / "feature_importance.parquet"


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data(ttl=3600)
def load_importance():
    if IMPORTANCE_PATH.exists():
        return pd.read_parquet(IMPORTANCE_PATH)
    return None


runs = st.session_state.get("model_runs", {})
active_model = st.session_state.get("active_model")
model = None
importance_df = None

if runs:
    model_names = list(runs.keys())
    idx = model_names.index(active_model) if active_model in model_names else 0
    selected_model = st.selectbox("Model", model_names, index=idx, key="explain_model")
    st.session_state["active_model"] = selected_model

    run = runs[selected_model]
    model = run.get("model")
    importance = run.get("feature_importance")
    if importance is not None and not importance.empty:
        importance_df = (
            importance.rename_axis("feature")
            .reset_index(name="importance")
            .sort_values("importance", key=lambda s: s.abs(), ascending=False)
        )
    st.caption(f"Using in-memory run output for model: {selected_model}")

if model is None:
    model = load_model()
if importance_df is None:
    importance_df = load_importance()

st.subheader("Global Feature Importance")
st.markdown(
    "This chart shows which features the model relies on most across predictions. "
    "Larger absolute bars indicate stronger influence."
)

if importance_df is not None and not importance_df.empty:
    try:
        from src.plots import plot_feature_importance

        imp_series = importance_df.set_index("feature")["importance"]
        fig = plot_feature_importance(imp_series, top_n=15)
        st.pyplot(fig)
    except Exception:
        st.dataframe(importance_df, use_container_width=True)

elif model is not None:
    try:
        from src.model import get_feature_importance
        from src.plots import plot_feature_importance

        imp = get_feature_importance(model)
        fig = plot_feature_importance(imp, top_n=15)
        st.pyplot(fig)
    except Exception as exc:
        st.error(f"Could not extract feature importance: {exc}")
else:
    st.info(
        "No model or importance found. Run one or more models from 'model lab'."
    )

st.subheader("Feature Glossary")

glossary = {
    "ret_1d": "1-day return, most recent daily price change.",
    "ret_5d": "5-day return, weekly momentum.",
    "ret_20d": "20-day return, monthly momentum.",
    "vol_20d": "20-day rolling volatility.",
    "volume_ratio_20d": "Today's volume vs 20-day average volume.",
    "rsi_14": "Relative Strength Index (overbought/oversold signal).",
    "ret_vs_iyt_5d": "Stock return minus sector ETF return.",
    "ret_vs_spy_5d": "Stock return minus S&P 500 return.",
    "vix_level": "VIX level as market stress proxy.",
    "vix_change_5d": "5-day VIX change.",
    "diesel_change_4w": "4-week diesel price change.",
    "ism_pmi": "ISM Manufacturing PMI.",
    "dgs10": "10-year Treasury yield.",
    "claims_change_4w": "4-week change in jobless claims.",
    "rolling_beta_60d": "60-day beta vs S&P 500.",
    "vol_regime": "High-volatility regime flag.",
    "momentum_rank": "Cross-sectional momentum rank.",
    "mean_reversion_5d": "Z-score of 5-day return stretch.",
}

for feature, desc in glossary.items():
    st.markdown(f"**`{feature}`** - {desc}")
