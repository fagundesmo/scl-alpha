"""Sandbox Models: train Ridge, Random Forest, and XGBoost on the feature matrix."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import RF_PARAMS, RIDGE_ALPHA, TARGET_COL, TICKERS, XGB_PARAMS
from src.model_runner import run_model

st.set_page_config(page_title="Sandbox Models", page_icon="🧪", layout="wide")
st.title("🧪 Sandbox Models")
st.caption(
    f"Train and compare Ridge, Random Forest, and XGBoost to predict `{TARGET_COL}` "
    "(next-day log return). Run **Pre-data** first to build the feature matrix."
)

# -----------------------------------------------------------------------
# Guard: require Pre-data features
# -----------------------------------------------------------------------
features: pd.DataFrame | None = st.session_state.get("features")

if features is None or not isinstance(features, pd.DataFrame) or features.empty:
    st.warning("No feature matrix found. Please complete all steps in **Pre-data** first.")
    st.stop()

st.caption(
    f"Feature matrix loaded: {len(features):,} rows × {features.shape[1]} columns | "
    f"Tickers: {', '.join(sorted(features['Ticker'].unique()))}"
)

MODEL_OPTIONS = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# -----------------------------------------------------------------------
# Run settings
# -----------------------------------------------------------------------
with st.expander("Run settings", expanded=True):
    selected_models = st.multiselect(
        "Models to run",
        options=list(MODEL_OPTIONS.keys()),
        default=list(MODEL_OPTIONS.keys()),
        format_func=lambda k: MODEL_OPTIONS[k],
        help="Select one or more models to train and compare.",
    )

    c1, c2 = st.columns(2)
    val_frac = c1.slider(
        "Validation fraction",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        help="Fraction of unique dates held out for validation (date-ordered split).",
    )
    test_frac = c2.slider(
        "Test fraction",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        help="Fraction of unique dates held out for final evaluation.",
    )

    if val_frac + test_frac >= 0.8:
        st.error("Validation + test fractions must total less than 0.8.")
        st.stop()

# -----------------------------------------------------------------------
# Model parameters
# -----------------------------------------------------------------------
with st.expander("Model parameters", expanded=True):
    st.markdown("#### Ridge Regression")
    ridge_alpha = st.slider(
        "Alpha (L2 strength)",
        min_value=0.001,
        max_value=100.0,
        value=float(RIDGE_ALPHA),
        step=0.1,
        help="Higher alpha → stronger regularization → more conservative predictions.",
    )

    st.markdown("#### Random Forest")
    rf1, rf2, rf3 = st.columns(3)
    rf_n_estimators = rf1.slider(
        "Trees", min_value=50, max_value=800,
        value=int(RF_PARAMS.get("n_estimators", 100)), step=10,
    )
    rf_max_depth = rf2.slider(
        "Max depth", min_value=2, max_value=20,
        value=int(RF_PARAMS.get("max_depth", 6)),
        help="Maximum tree depth. Higher → more complex model, risk of overfitting.",
    )
    rf_min_leaf = rf3.slider(
        "Min leaf size", min_value=1, max_value=50,
        value=int(RF_PARAMS.get("min_samples_leaf", 20)),
        help="Minimum samples per leaf. Higher → smoother predictions.",
    )

    st.markdown("#### XGBoost")
    x1, x2, x3, x4 = st.columns(4)
    xgb_n_estimators = x1.slider(
        "Trees", min_value=50, max_value=1000,
        value=int(XGB_PARAMS.get("n_estimators", 200)), step=25,
    )
    xgb_max_depth = x2.slider(
        "Max depth", min_value=2, max_value=10,
        value=int(XGB_PARAMS.get("max_depth", 4)),
    )
    xgb_learning_rate = x3.number_input(
        "Learning rate", min_value=0.01, max_value=0.5,
        value=float(XGB_PARAMS.get("learning_rate", 0.05)),
        step=0.01, format="%.2f",
    )
    xgb_early_stopping = x4.slider(
        "Early stopping rounds", min_value=5, max_value=100,
        value=int(XGB_PARAMS.get("early_stopping_rounds", 20)),
        help="Stop training if validation metric does not improve.",
    )

    x5, x6, x7, x8 = st.columns(4)
    xgb_subsample = x5.slider(
        "Subsample", min_value=0.4, max_value=1.0,
        value=float(XGB_PARAMS.get("subsample", 0.8)), step=0.05,
    )
    xgb_colsample = x6.slider(
        "Colsample by tree", min_value=0.4, max_value=1.0,
        value=float(XGB_PARAMS.get("colsample_bytree", 0.8)), step=0.05,
    )
    xgb_reg_alpha = x7.slider(
        "Reg alpha (L1)", min_value=0.0, max_value=10.0,
        value=float(XGB_PARAMS.get("reg_alpha", 0.0)), step=0.1,
    )
    xgb_reg_lambda = x8.slider(
        "Reg lambda (L2)", min_value=0.0, max_value=20.0,
        value=float(XGB_PARAMS.get("reg_lambda", 1.0)), step=0.1,
    )

    xgb_obj_options = ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]
    xgb_objective = st.selectbox(
        "Objective", options=xgb_obj_options,
        help="Loss function for XGBoost training.",
    )

# -----------------------------------------------------------------------
# Run button
# -----------------------------------------------------------------------
if st.button("Run selected models", type="primary", use_container_width=True):
    if not selected_models:
        st.warning("Select at least one model.")
        st.stop()

    param_map = {
        "ridge": {"alpha": float(ridge_alpha)},
        "random_forest": {
            "n_estimators": int(rf_n_estimators),
            "max_depth": int(rf_max_depth),
            "min_samples_leaf": int(rf_min_leaf),
        },
        "xgboost": {
            "n_estimators": int(xgb_n_estimators),
            "max_depth": int(xgb_max_depth),
            "learning_rate": float(xgb_learning_rate),
            "subsample": float(xgb_subsample),
            "colsample_bytree": float(xgb_colsample),
            "reg_alpha": float(xgb_reg_alpha),
            "reg_lambda": float(xgb_reg_lambda),
            "objective": xgb_objective,
            "early_stopping_rounds": int(xgb_early_stopping),
        },
    }

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    progress = st.progress(0)
    status = st.empty()

    for i, model_name in enumerate(selected_models):
        label = MODEL_OPTIONS[model_name]
        status.info(f"Training {label}...")
        try:
            result = run_model(
                features=features,
                model_name=model_name,
                model_params=param_map[model_name],
                val_frac=float(val_frac),
                test_frac=float(test_frac),
            )
            results[model_name] = result
        except Exception as exc:
            errors[model_name] = str(exc)
        progress.progress((i + 1) / len(selected_models))

    status.empty()
    progress.empty()

    if errors:
        for model_name, msg in errors.items():
            st.error(f"{MODEL_OPTIONS[model_name]} failed: {msg}")

    if results:
        st.session_state["sandbox_results"] = results
        st.success(f"Completed {len(results)} model(s).")

# -----------------------------------------------------------------------
# Display results
# -----------------------------------------------------------------------
results: dict[str, dict] | None = st.session_state.get("sandbox_results")

if not results:
    st.info("Run at least one model to see results.")
    st.stop()

# Comparison table
st.subheader("Model Comparison — Test-Set Metrics")

metric_labels = {
    "MAE": "MAE",
    "RMSE": "RMSE",
    "IC": "IC (Spearman)",
    "Hit Rate": "Hit Rate",
}
rows = []
for model_name, res in results.items():
    row = {"Model": MODEL_OPTIONS[model_name]}
    for key, label in metric_labels.items():
        v = res["test_metrics"].get(key, float("nan"))
        row[label] = v
    row["Train rows"] = res["train_size"]
    row["Val rows"] = res["val_size"]
    row["Test rows"] = res["test_size"]
    rows.append(row)

comparison_df = pd.DataFrame(rows)
st.dataframe(
    comparison_df.style.format(
        {
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "IC (Spearman)": "{:.3f}",
            "Hit Rate": "{:.1%}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "IC = Information Coefficient (Spearman rank correlation between predicted and realized returns). "
    "Hit Rate = fraction of predictions with the correct sign."
)

# Per-split metrics
st.subheader("Train / Val / Test Metrics by Model")
split_rows = []
for model_name, res in results.items():
    for split_name, split_metrics in [
        ("Train", res["train_metrics"]),
        ("Val", res["val_metrics"]),
        ("Test", res["test_metrics"]),
    ]:
        row = {
            "Model": MODEL_OPTIONS[model_name],
            "Split": split_name,
        }
        row.update(split_metrics)
        split_rows.append(row)

split_df = pd.DataFrame(split_rows)
st.dataframe(
    split_df.style.format(
        {"MAE": "{:.4f}", "RMSE": "{:.4f}", "IC": "{:.3f}", "Hit Rate": "{:.1%}"}
    ),
    use_container_width=True,
    hide_index=True,
)

# Individual model detail tabs
tab_names = [MODEL_OPTIONS[m] for m in results]
tabs = st.tabs(tab_names)

for tab, (model_name, res) in zip(tabs, results.items()):
    with tab:
        st.markdown(f"**Parameters used:** `{res['model_params']}`")
        st.caption(
            f"Train: {res['train_size']:,} rows | "
            f"Val: {res['val_size']:,} rows | "
            f"Test: {res['test_size']:,} rows"
        )

        # Feature importance
        imp = res.get("feature_importance")
        if isinstance(imp, pd.DataFrame) and not imp.empty:
            st.markdown("#### Feature Importance (top 20)")
            top_imp = imp.head(20)
            fig = px.bar(
                top_imp,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Feature Importance — {MODEL_OPTIONS[model_name]}",
            )
            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=500,
                margin=dict(l=10, r=10, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Test predictions scatter
        preds = res.get("test_predictions")
        if isinstance(preds, pd.DataFrame) and not preds.empty:
            st.markdown("#### Predicted vs Actual (Test Set)")
            fig = px.scatter(
                preds,
                x="y_true",
                y="y_pred",
                color="Ticker",
                opacity=0.4,
                title="Predicted vs Actual Next-Day Log Return",
                labels={"y_true": "Actual return", "y_pred": "Predicted return"},
                trendline="ols",
                trendline_scope="overall",
            )
            fig.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.add_vline(x=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig.update_layout(height=480)
            st.plotly_chart(fig, use_container_width=True)

            # Residual distribution
            st.markdown("#### Residual Distribution (Test Set)")
            fig = px.histogram(
                preds,
                x="residual",
                nbins=60,
                title="Residuals (y_true − y_pred)",
                color="Ticker",
                barmode="overlay",
                opacity=0.6,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View test predictions table"):
                st.dataframe(
                    preds.style.format(
                        {
                            "y_true": "{:.4f}",
                            "y_pred": "{:.4f}",
                            "residual": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    label="Download test predictions CSV",
                    data=preds.to_csv(index=False).encode("utf-8"),
                    file_name=f"{model_name}_test_predictions.csv",
                    mime="text/csv",
                    key=f"dl_preds_{model_name}",
                )
