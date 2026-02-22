"""
Explain page — feature importance for individual predictions.

Displays a horizontal bar chart showing which features contributed most
to a specific prediction.  Helps users build intuition about what drives
the model's output.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Explain", page_icon="🔍", layout="wide")
st.title("🔍 Prediction Explainability")

# ---------------------------------------------------------------------------
# Load model and feature importance
# ---------------------------------------------------------------------------
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


model = load_model()
importance_df = load_importance()

# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------
st.subheader("Global Feature Importance")
st.markdown(
    "This chart shows which features the model relies on most across all "
    "predictions.  Larger bars = more influence on the prediction."
)

if importance_df is not None:
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
        imp = get_feature_importance(model)
        from src.plots import plot_feature_importance
        fig = plot_feature_importance(imp, top_n=15)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not extract feature importance: {e}")
else:
    st.info(
        "No model or pre-computed importance found in `app/cache/`. "
        "Run the notebook first."
    )

# ---------------------------------------------------------------------------
# Feature glossary
# ---------------------------------------------------------------------------
st.subheader("Feature Glossary")

glossary = {
    "ret_1d": "1-day return — most recent daily price change.",
    "ret_5d": "5-day return — weekly momentum.",
    "ret_20d": "20-day return — monthly momentum.",
    "vol_20d": "20-day rolling volatility — how much the stock has been bouncing.",
    "volume_ratio_20d": "Today's volume vs 20-day average — unusual activity detector.",
    "rsi_14": "Relative Strength Index — overbought (>70) or oversold (<30).",
    "ret_vs_iyt_5d": "Stock return minus sector ETF return — relative strength.",
    "ret_vs_spy_5d": "Stock return minus S&P 500 return — market-relative strength.",
    "vix_level": "VIX — market fear gauge.",
    "vix_change_5d": "5-day change in VIX — fear increasing or decreasing.",
    "diesel_change_4w": "4-week change in diesel prices — direct cost driver.",
    "ism_pmi": "ISM Manufacturing PMI — factory activity indicator.",
    "dgs10": "10-Year Treasury yield — interest rate environment.",
    "claims_change_4w": "4-week change in jobless claims — labor market signal.",
    "rolling_beta_60d": "60-day beta vs S&P 500 — market sensitivity.",
    "vol_regime": "High-volatility flag — 1 when vol is in top 20%.",
    "momentum_rank": "Rank 1–10 by 20-day return — cross-sectional momentum.",
    "mean_reversion_5d": "Z-score of 5-day return — how stretched is the stock.",
}

for feature, desc in glossary.items():
    st.markdown(f"**`{feature}`** — {desc}")
