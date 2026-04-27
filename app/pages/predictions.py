"""Predictions: signal analysis, time-series plots, simple strategy simulation, and AI Tutor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
st.title("🔮 Predictions & Signal Analysis")
st.caption(
    "Explore model predictions over time, signal quality, and a simple long/short strategy simulation. "
    "Run **Sandbox Models** first to populate this page."
)

# -----------------------------------------------------------------------
# Sidebar: AI Tutor API key setup
# -----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 AI Tutor Setup")
    st.caption(
        "Get a **free** API key and let an LLM explain your results in plain English.  \n"
        "Choose one provider:"
    )

    ai_provider = st.radio(
        "LLM provider",
        options=["Gemini Flash (Google)", "Groq / Llama 3 (Meta)"],
        index=0,
        help=(
            "Gemini Flash: free at aistudio.google.com — 1,500 req/day.  \n"
            "Groq: free at console.groq.com — very fast Llama 3."
        ),
    )

    ai_api_key = st.text_input(
        "Paste your free API key here",
        type="password",
        placeholder="AIza... or gsk_...",
        help=(
            "Gemini: get key at https://aistudio.google.com/app/apikey  \n"
            "Groq: get key at https://console.groq.com"
        ),
    )

    # Also check Streamlit secrets as fallback (for instructor deployment)
    if not ai_api_key:
        if "GEMINI_API_KEY" in st.secrets and "Gemini" in ai_provider:
            ai_api_key = st.secrets["GEMINI_API_KEY"]
        elif "GROQ_API_KEY" in st.secrets and "Groq" in ai_provider:
            ai_api_key = st.secrets["GROQ_API_KEY"]

    st.divider()


# -----------------------------------------------------------------------
# AI Tutor helper
# -----------------------------------------------------------------------

def _call_llm(prompt: str, provider: str, api_key: str) -> str:
    """Call Gemini or Groq and return the text response."""
    if "Gemini" in provider:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        return resp.text
    else:
        from groq import Groq
        client = Groq(api_key=api_key)
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        return chat.choices[0].message.content


def _build_tutor_prompt(model_label: str, res: dict, ticker_metrics: list[dict]) -> str:
    """Build a structured prompt from model results for an undergrad audience."""
    tm = res.get("test_metrics", {})
    vm = res.get("val_metrics", {})
    tr = res.get("train_metrics", {})

    ticker_summary = "\n".join(
        f"  - {r['Ticker']}: IC={r['IC']:.3f}, Hit Rate={r['Hit Rate']:.1%}, Pred Sharpe={r['Pred Sharpe']:.2f}"
        for r in sorted(ticker_metrics, key=lambda x: -x["IC"])[:8]
    ) if ticker_metrics else "  (not available)"

    return f"""
You are a teaching assistant for an undergraduate finance course on machine learning in trading.
A student just ran a {model_label} model to predict next-day stock returns for supply-chain companies.
Explain the results below in plain English — no jargon, use analogies, keep it engaging.
Structure your response with these sections:

1. **What the model learned** (2-3 sentences — what is it trying to do?)
2. **Is the signal any good?** (explain IC, Hit Rate, Pred Sharpe in plain terms with the actual numbers)
3. **How does it do by stock?** (highlight the best and worst tickers from the per-ticker table)
4. **Training vs Validation vs Test — what do the gaps tell us?** (overfitting? generalisation?)
5. **What this means for trading** (can you actually use this? what would you need to be careful about?)
6. **One key takeaway** (1 sentence the student should remember)

Here are the actual results:

MODEL: {model_label}

Test set metrics (2026 — true out-of-sample):
  - MAE: {tm.get('MAE', float('nan')):.4f}
  - RMSE: {tm.get('RMSE', float('nan')):.4f}
  - IC (Spearman): {tm.get('IC', float('nan')):.3f}
  - Hit Rate: {tm.get('Hit Rate', float('nan')):.1%}
  - Pred Sharpe: {tm.get('Pred Sharpe', float('nan')):.2f}

Validation set metrics (2025):
  - IC: {vm.get('IC', float('nan')):.3f}, Hit Rate: {vm.get('Hit Rate', float('nan')):.1%}, Pred Sharpe: {vm.get('Pred Sharpe', float('nan')):.2f}

Training set metrics (2018–2024):
  - IC: {tr.get('IC', float('nan')):.3f}, Hit Rate: {tr.get('Hit Rate', float('nan')):.1%}, Pred Sharpe: {tr.get('Pred Sharpe', float('nan')):.2f}

Split sizes: Train={res.get('train_size',0):,} rows | Val={res.get('val_size',0):,} | Test={res.get('test_size',0):,}
Test date range: {res.get('test_date_range', 'N/A')}

Per-ticker IC on test set (top/bottom performers):
{ticker_summary}

Context clues:
- IC > 0.05 is considered meaningful in practice; IC > 0.10 is strong.
- Hit Rate > 52% is useful if you trade at scale; 50% = pure luck.
- Pred Sharpe > 0.5 annualised is respectable for a daily signal.
- These are real supply-chain/logistics stocks (UPS, FDX, JBHT, etc.).
- The target variable is the next-day log return (so tiny numbers like 0.003 = +0.3% return).

Keep the explanation under 500 words, friendly, and educational.
""".strip()

# -----------------------------------------------------------------------
# Guard
# -----------------------------------------------------------------------
results: dict | None = st.session_state.get("sandbox_results")
if not results:
    st.warning("No model results found. Run at least one model in **Sandbox Models** first.")
    st.stop()

MODEL_LABELS = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# -----------------------------------------------------------------------
# Model selector
# -----------------------------------------------------------------------
model_choice = st.selectbox(
    "Select model to analyse",
    options=list(results.keys()),
    format_func=lambda k: MODEL_LABELS.get(k, k),
)
res = results[model_choice]
all_preds: pd.DataFrame = res.get("all_predictions", pd.DataFrame())
test_preds: pd.DataFrame = res.get("test_predictions", pd.DataFrame())

if all_preds.empty:
    st.error("No predictions found for this model.")
    st.stop()

all_preds = all_preds.copy()
all_preds["Date"] = pd.to_datetime(all_preds["Date"])

tickers = sorted(all_preds["Ticker"].unique())

# -----------------------------------------------------------------------
# Section 1: Predicted vs actual time series (per ticker)
# -----------------------------------------------------------------------
st.header("1 · Predicted vs Actual Returns Over Time")

st.info(
    "**Why does the prediction look flat?**  \n"
    "Next-day returns are near-random noise — even top quant models predict magnitudes of 0.001–0.005, "
    "while actual daily moves can be ±5–15% on news days. The model is being honest about uncertainty. "
    "**What matters is the sign (direction), not the magnitude.** "
    "A 53% hit rate on direction is genuinely valuable in practice.  \n"
    "The chart below normalises both series to the same scale so you can see direction agreement.",
    icon="ℹ️",
)

ticker_choice = st.selectbox("Select ticker", options=tickers, key="pred_ts_ticker")
split_filter = st.multiselect(
    "Show splits",
    options=["train", "val", "test"],
    default=["val", "test"],
    key="pred_ts_splits",
)

ts_df = all_preds[
    (all_preds["Ticker"] == ticker_choice) & (all_preds["split"].isin(split_filter))
].sort_values("Date")

if ts_df.empty:
    st.info("No data for this selection.")
else:
    # ---- Panel A: z-score normalised so both series are on the same scale ----
    def _zscore(s: pd.Series) -> pd.Series:
        std = s.std()
        return (s - s.mean()) / std if std > 0 else s * 0

    ts_plot = ts_df.copy()
    ts_plot["actual_z"]    = _zscore(ts_plot["y_true"])
    ts_plot["predicted_z"] = _zscore(ts_plot["y_pred"])

    test_min = all_preds.loc[all_preds["split"] == "test", "Date"].min()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_plot["Date"], y=ts_plot["actual_z"],
        name="Actual return (z-scored)", line=dict(color="#1f77b4", width=1.2), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=ts_plot["Date"], y=ts_plot["predicted_z"],
        name="Predicted return (z-scored)", line=dict(color="#ff7f0e", width=2), opacity=0.9,
    ))
    if pd.notna(test_min):
        fig.add_vrect(
            x0=test_min, x1=all_preds["Date"].max(),
            fillcolor="lightgreen", opacity=0.08, line_width=0,
            annotation_text="Test (2026)", annotation_position="top left",
        )
    fig.update_layout(
        title=f"Signal Direction — Normalised Predicted vs Actual — {ticker_choice}",
        xaxis_title="Date", yaxis_title="Z-score (same scale)",
        height=380, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Panel B: correct-direction highlighting ----
    st.caption("**Direction correctness** — green bar = model predicted the correct sign (up/down) that day.")
    ts_plot["correct"] = np.sign(ts_plot["y_pred"]) == np.sign(ts_plot["y_true"])
    ts_plot["bar_color"] = ts_plot["correct"].map({True: "Correct", False: "Wrong"})

    fig2 = px.bar(
        ts_plot, x="Date", y="y_true",
        color="bar_color",
        color_discrete_map={"Correct": "#2ca02c", "Wrong": "#d62728"},
        labels={"y_true": "Actual return", "bar_color": "Direction"},
        title=f"Actual Daily Return — Coloured by Prediction Direction Accuracy — {ticker_choice}",
    )
    fig2.update_layout(height=300, hovermode="x unified", bargap=0)
    st.plotly_chart(fig2, use_container_width=True)

    hit = ts_plot["correct"].mean()
    st.caption(
        f"Hit rate for selected period: **{hit:.1%}** "
        f"({'above' if hit > 0.50 else 'below'} the 50% random baseline). "
        f"Days shown: {len(ts_plot):,}."
    )

# -----------------------------------------------------------------------
# Section 2: Rolling 21-day IC (signal consistency)
# -----------------------------------------------------------------------
st.header("2 · Rolling 21-Day Information Coefficient (IC)")
st.caption(
    "IC = Spearman rank correlation between predicted and actual returns on that day, "
    "computed across all tickers. Rolling 21-day window smooths noise. "
    "IC > 0.05 consistently = practically useful signal."
)

roll_splits = st.multiselect(
    "Include splits in IC chart",
    options=["train", "val", "test"],
    default=["val", "test"],
    key="ic_splits",
)

ic_df = all_preds[all_preds["split"].isin(roll_splits)].sort_values("Date")

if len(ic_df) < 30:
    st.info("Not enough data for rolling IC — include more splits.")
else:
    from scipy import stats as scipy_stats

    daily_ic = (
        ic_df.groupby("Date")
        .apply(
            lambda g: scipy_stats.spearmanr(g["y_true"], g["y_pred"]).statistic
            if len(g) >= 3 else np.nan,
            include_groups=False,
        )
        .rename("IC")
        .reset_index()
    )
    daily_ic["IC_21d"] = daily_ic["IC"].rolling(21, min_periods=10).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_ic["Date"], y=daily_ic["IC"],
        name="Daily IC", line=dict(color="#aec7e8", width=0.8), opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=daily_ic["Date"], y=daily_ic["IC_21d"],
        name="21-day rolling IC", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.6)
    fig.add_hline(y=0.05, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="IC=0.05 (practical threshold)", annotation_position="top right")
    test_min = ic_df.loc[ic_df["split"] == "test", "Date"].min() if "test" in roll_splits else None
    if pd.notna(test_min) if test_min is not None else False:
        fig.add_vrect(
            x0=test_min, x1=ic_df["Date"].max(),
            fillcolor="lightgreen", opacity=0.08, line_width=0,
            annotation_text="Test (2026)", annotation_position="top left",
        )
    fig.update_layout(
        title=f"Rolling 21-Day IC — {MODEL_LABELS.get(model_choice, model_choice)}",
        xaxis_title="Date", yaxis_title="IC (Spearman)",
        height=400, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------
# Section 3: Per-ticker IC table (test set)
# -----------------------------------------------------------------------
st.header("3 · Per-Ticker Signal Quality — Test Set (2026)")
st.caption(
    "IC and Hit Rate computed separately for each ticker on the test set. "
    "Shows which companies the model predicts best."
)

ticker_metrics: list[dict] = []

if test_preds.empty:
    st.info("No test-set predictions available.")
else:
    from scipy import stats as scipy_stats2
    for t in sorted(test_preds["Ticker"].unique()):
        sub = test_preds[test_preds["Ticker"] == t]
        if len(sub) < 5:
            continue
        ic_val = scipy_stats2.spearmanr(sub["y_true"], sub["y_pred"]).statistic
        hit = float((np.sign(sub["y_pred"]) == np.sign(sub["y_true"])).mean())
        mae_val = float(np.mean(np.abs(sub["y_true"] - sub["y_pred"])))
        strategy = np.sign(sub["y_pred"].values) * sub["y_true"].values
        sharpe = float(strategy.mean() / strategy.std() * np.sqrt(252)) if strategy.std() > 0 else np.nan
        ticker_metrics.append({
            "Ticker": t,
            "IC": ic_val,
            "Hit Rate": hit,
            "MAE": mae_val,
            "Pred Sharpe": sharpe,
            "N days": len(sub),
        })

    tm_df = pd.DataFrame(ticker_metrics).sort_values("IC", ascending=False)
    st.dataframe(
        tm_df.style.format({
            "IC": "{:.3f}", "Hit Rate": "{:.1%}",
            "MAE": "{:.4f}", "Pred Sharpe": "{:.2f}",
        }).background_gradient(subset=["IC"], cmap="RdYlGn"),
        use_container_width=True, hide_index=True,
    )

# -----------------------------------------------------------------------
# Section 4 · AI Tutor — explain results in plain English
# -----------------------------------------------------------------------
st.header("4 · AI Tutor — Explain My Results")
st.caption(
    "Paste a free API key in the sidebar, then click the button below. "
    "The AI will read your actual numbers and explain what they mean in plain English."
)

if not ai_api_key:
    st.info(
        "No API key found. Get a **free** key from one of these:  \n"
        "• **Gemini Flash (recommended):** https://aistudio.google.com/app/apikey — 1,500 free calls/day  \n"
        "• **Groq / Llama 3:** https://console.groq.com — also free, very fast  \n\n"
        "Paste it in the **AI Tutor Setup** panel in the sidebar.",
        icon="🔑",
    )
else:
    if st.button("Explain my results with AI", type="primary", use_container_width=True):
        prompt = _build_tutor_prompt(
            model_label=MODEL_LABELS.get(model_choice, model_choice),
            res=res,
            ticker_metrics=ticker_metrics,
        )
        with st.spinner("Thinking..."):
            try:
                explanation = _call_llm(prompt, ai_provider, ai_api_key)
                st.session_state["ai_explanation"] = explanation
            except Exception as e:
                st.error(f"LLM call failed: {e}")

    explanation = st.session_state.get("ai_explanation")
    if explanation:
        st.markdown("---")
        st.markdown(explanation)
        st.markdown("---")
        st.caption(
            "Generated by AI based on your actual model results. "
            "Always verify key numbers against the tables above."
        )
        st.download_button(
            label="Download explanation as text",
            data=explanation.encode("utf-8"),
            file_name=f"{model_choice}_ai_explanation.txt",
            mime="text/plain",
        )

# -----------------------------------------------------------------------
# Section 5: Long / Short strategy simulation (test set)
# -----------------------------------------------------------------------
st.header("5 · Long / Short Strategy Simulation — Test Set")
st.caption(
    "Each day, rank all tickers by predicted return. Go **long** the top-K and **short** the bottom-K "
    "(equal weight within each leg). Portfolio daily return = avg(long actual) − avg(short actual). "
    "Compare vs equal-weight long-only (buy-and-hold) benchmark."
)

if test_preds.empty:
    st.info("No test-set predictions available.")
else:
    k_long = st.slider("K — number of long/short positions per day", min_value=1, max_value=5, value=3)

    test_sorted = test_preds.copy()
    test_sorted["Date"] = pd.to_datetime(test_sorted["Date"])

    def daily_strategy(group: pd.DataFrame, k: int) -> float:
        ranked = group.sort_values("y_pred", ascending=False)
        long_ret  = ranked.head(k)["y_true"].mean()
        short_ret = ranked.tail(k)["y_true"].mean()
        return float(long_ret - short_ret)

    daily_ls = (
        test_sorted.groupby("Date")
        .apply(lambda g: daily_strategy(g, k_long), include_groups=False)
        .rename("ls_return")
        .reset_index()
    )
    daily_bh = (
        test_sorted.groupby("Date")["y_true"]
        .mean()
        .rename("bh_return")
        .reset_index()
    )
    strat_df = daily_ls.merge(daily_bh, on="Date").sort_values("Date")

    # Cumulative log returns
    strat_df["cum_ls"] = np.exp(strat_df["ls_return"].cumsum())
    strat_df["cum_bh"] = np.exp(strat_df["bh_return"].cumsum())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strat_df["Date"], y=strat_df["cum_ls"],
        name=f"Long top-{k_long} / Short bottom-{k_long}",
        line=dict(color="#2ca02c", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=strat_df["Date"], y=strat_df["cum_bh"],
        name="Equal-weight long only (benchmark)",
        line=dict(color="#d62728", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", opacity=0.5)
    fig.update_layout(
        title=f"Cumulative Return — Long/Short vs Benchmark (Test Set 2026)",
        xaxis_title="Date", yaxis_title="Growth of $1",
        height=430, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    ls_daily = strat_df["ls_return"]
    bh_daily = strat_df["bh_return"]

    def _sharpe(r: pd.Series) -> float:
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan

    def _maxdd(cum: pd.Series) -> float:
        peak = cum.cummax()
        return float(((cum - peak) / peak).min())

    summary = pd.DataFrame([
        {
            "Strategy": f"L/S top-{k_long}/bottom-{k_long}",
            "Total return": f"{(strat_df['cum_ls'].iloc[-1] - 1)*100:.1f}%",
            "Ann. Sharpe": f"{_sharpe(ls_daily):.2f}",
            "Max Drawdown": f"{_maxdd(strat_df['cum_ls'])*100:.1f}%",
            "Hit Rate": f"{(ls_daily > 0).mean()*100:.1f}%",
        },
        {
            "Strategy": "Equal-weight long only",
            "Total return": f"{(strat_df['cum_bh'].iloc[-1] - 1)*100:.1f}%",
            "Ann. Sharpe": f"{_sharpe(bh_daily):.2f}",
            "Max Drawdown": f"{_maxdd(strat_df['cum_bh'])*100:.1f}%",
            "Hit Rate": f"{(bh_daily > 0).mean()*100:.1f}%",
        },
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ This simulation ignores transaction costs, slippage, and bid-ask spread. "
        "Real trading returns would be lower. This is for educational purposes only."
    )

# -----------------------------------------------------------------------
# Section 6: Download all predictions
# -----------------------------------------------------------------------
st.header("6 · Download Predictions")

col1, col2 = st.columns(2)
with col1:
    if not all_preds.empty:
        st.download_button(
            label="Download ALL predictions CSV (train + val + test)",
            data=all_preds.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_choice}_all_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
with col2:
    if not test_preds.empty:
        st.download_button(
            label="Download TEST predictions CSV (2026 only)",
            data=test_preds.to_csv(index=False).encode("utf-8"),
            file_name=f"{model_choice}_test_predictions_2026.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.caption(
    "Columns: Date, Ticker, y_true (actual log return), y_pred (predicted log return), "
    "residual (y_true − y_pred), split (train/val/test)."
)
