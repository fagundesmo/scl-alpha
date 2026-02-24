"""Full SCL-alpha model page.

This page runs a standardized reporting pipeline across Ridge, Random Forest,
and XGBoost, then shows beginner-friendly comparison tables.
"""

from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.app_runner import load_feature_matrix
from src.config import RF_PARAMS, RIDGE_ALPHA, TICKERS, XGB_PARAMS
from src.features import FEATURE_COLUMNS
from src.full_scl_alpha_model import (
    build_full_comparison_report,
    create_synthetic_example_dataset,
    export_reports_to_csv,
    time_based_split,
)

TARGET_COL = "predicted_return_5d"


def _prepare_real_dataset(feature_matrix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build a clean learning dataset from project features for the same ticker universe."""
    if not isinstance(feature_matrix.index, pd.MultiIndex):
        raise ValueError("Expected feature matrix with MultiIndex (date, ticker).")

    if "target_ret_5d_fwd" not in feature_matrix.columns:
        raise ValueError("Feature matrix is missing target_ret_5d_fwd.")

    feature_cols = [c for c in FEATURE_COLUMNS if c in feature_matrix.columns]
    if not feature_cols:
        raise ValueError("No expected feature columns were found in the feature matrix.")

    base = feature_matrix[feature_matrix.index.get_level_values("ticker").isin(TICKERS)].copy()
    if base.empty:
        raise ValueError("No rows found for configured supply-chain ticker universe.")

    # Fill gaps per ticker first, then fallback to global median for robustness.
    for col in feature_cols:
        s = base[col].groupby(level="ticker").ffill()
        s = s.groupby(level="ticker").bfill()
        median = float(s.median()) if s.notna().any() else 0.0
        base[col] = s.fillna(median)

    base = base.dropna(subset=["target_ret_5d_fwd"])
    if base.empty:
        raise ValueError("No trainable rows after dropping missing targets.")

    out = base.reset_index()[["date", "ticker", *feature_cols, "target_ret_5d_fwd"]].copy()
    out = out.rename(columns={"target_ret_5d_fwd": TARGET_COL})
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    return out, feature_cols


def _show_df(df: pd.DataFrame, formats: dict | None = None, max_rows: int = 500) -> None:
    """Display a DataFrame consistently across sections."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No rows available for this section.")
        return

    view = df.head(max_rows)
    if formats:
        st.dataframe(view.style.format(formats), use_container_width=True, hide_index=True)
    else:
        st.dataframe(view, use_container_width=True, hide_index=True)


def _download_df(df: pd.DataFrame, file_name_stem: str, key: str) -> None:
    """Render a one-click CSV download button for a table."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    st.download_button(
        label=f"Download {file_name_stem}.csv",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{file_name_stem}.csv",
        mime="text/csv",
        key=key,
        use_container_width=True,
    )


def _show_with_download(
    title: str,
    df: pd.DataFrame,
    file_name_stem: str,
    key: str,
    formats: dict | None = None,
    max_rows: int = 500,
) -> None:
    """Convenience helper: show table then place download button below it."""
    st.markdown(f"#### {title}")
    _show_df(df, formats=formats, max_rows=max_rows)
    _download_df(df, file_name_stem=file_name_stem, key=key)


def _markdown_cell(source: str) -> dict:
    """Build a Jupyter markdown cell payload."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\\n" for line in source.strip().splitlines()],
    }


def _code_cell(source: str) -> dict:
    """Build a Jupyter code cell payload."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\\n" for line in source.strip().splitlines()],
    }


def _build_notebook_bytes(config: dict) -> bytes:
    """Generate a runnable notebook that reproduces this page's workflow."""
    setup_cell = (
        f"DATA_SOURCE = {config['data_source']!r}\\n"
        f"REFRESH_DATA = {bool(config['refresh_data'])!r}\\n"
        f"VAL_FRACTION = {float(config['val_fraction'])!r}\\n"
        f"TEST_FRACTION = {float(config['test_fraction'])!r}\\n"
        f"TRANSACTION_COST_BPS = {float(config['transaction_cost_bps'])!r}\\n"
        f"INCLUDE_SHAP = {bool(config['include_shap'])!r}\\n"
        f"EXPORT_CSV = {bool(config['export_csv'])!r}\\n"
        f"OUTPUT_DIR = {config['output_dir']!r}\\n\\n"
        f"RIDGE_PARAMS = {{'alpha': {float(config['ridge_alpha'])!r}}}\\n"
        "RF_PARAMS = {\\n"
        f"    'n_estimators': {int(config['rf_n_estimators'])!r},\\n"
        f"    'max_depth': {int(config['rf_max_depth'])!r},\\n"
        f"    'min_samples_leaf': {int(config['rf_min_leaf'])!r},\\n"
        f"    'max_features': {config['rf_max_features']!r},\\n"
        "    'bootstrap': True,\\n"
        "    'oob_score_enabled': True,\\n"
        "    'random_state': 42,\\n"
        "}\\n"
        "XGB_PARAMS = {\\n"
        f"    'n_estimators': {int(config['xgb_n_estimators'])!r},\\n"
        f"    'max_depth': {int(config['xgb_max_depth'])!r},\\n"
        f"    'learning_rate': {float(config['xgb_learning_rate'])!r},\\n"
        f"    'subsample': {float(config['xgb_subsample'])!r},\\n"
        f"    'colsample_bytree': {float(config['xgb_colsample'])!r},\\n"
        f"    'reg_alpha': {float(config['xgb_reg_alpha'])!r},\\n"
        f"    'reg_lambda': {float(config['xgb_reg_lambda'])!r},\\n"
        "    'random_state': 42,\\n"
        f"    'early_stopping_rounds': {int(config['xgb_early_stopping'])!r},\\n"
        "}"
    )

    notebook = {
        "cells": [
            _markdown_cell(
                "# Full SCL-alpha model\\n"
                "This notebook was generated from the Streamlit page settings."
            ),
            _code_cell(
                "import pandas as pd\\n"
                "from src.app_runner import load_feature_matrix\\n"
                "from src.config import TICKERS\\n"
                "from src.features import FEATURE_COLUMNS\\n"
                "from src.full_scl_alpha_model import (\\n"
                "    build_full_comparison_report,\\n"
                "    create_synthetic_example_dataset,\\n"
                "    export_reports_to_csv,\\n"
                "    time_based_split,\\n"
                ")\\n\\n"
                "TARGET_COL = 'predicted_return_5d'"
            ),
            _code_cell(
                "def prepare_real_dataset(feature_matrix: pd.DataFrame):\\n"
                "    if not isinstance(feature_matrix.index, pd.MultiIndex):\\n"
                "        raise ValueError('Expected feature matrix with MultiIndex (date, ticker).')\\n"
                "    if 'target_ret_5d_fwd' not in feature_matrix.columns:\\n"
                "        raise ValueError('Feature matrix is missing target_ret_5d_fwd.')\\n\\n"
                "    feature_cols = [c for c in FEATURE_COLUMNS if c in feature_matrix.columns]\\n"
                "    if not feature_cols:\\n"
                "        raise ValueError('No expected feature columns found.')\\n\\n"
                "    base = feature_matrix[feature_matrix.index.get_level_values('ticker').isin(TICKERS)].copy()\\n"
                "    if base.empty:\\n"
                "        raise ValueError('No rows found for configured supply-chain ticker universe.')\\n\\n"
                "    for col in feature_cols:\\n"
                "        s = base[col].groupby(level='ticker').ffill()\\n"
                "        s = s.groupby(level='ticker').bfill()\\n"
                "        median = float(s.median()) if s.notna().any() else 0.0\\n"
                "        base[col] = s.fillna(median)\\n\\n"
                "    base = base.dropna(subset=['target_ret_5d_fwd'])\\n"
                "    out = base.reset_index()[['date', 'ticker', *feature_cols, 'target_ret_5d_fwd']].copy()\\n"
                "    out = out.rename(columns={'target_ret_5d_fwd': TARGET_COL})\\n"
                "    out = out.sort_values(['date', 'ticker']).reset_index(drop=True)\\n"
                "    return out, feature_cols"
            ),
            _code_cell(setup_cell),
            _code_cell(
                "if DATA_SOURCE == 'Use project supply-chain data':\\n"
                "    feature_matrix = load_feature_matrix(refresh_data=REFRESH_DATA)\\n"
                "    dataset, feature_cols = prepare_real_dataset(feature_matrix)\\n"
                "else:\\n"
                "    feature_cols = [c for c in FEATURE_COLUMNS]\\n"
                "    dataset = create_synthetic_example_dataset(\\n"
                "        feature_names=feature_cols,\\n"
                "        tickers=TICKERS,\\n"
                "        n_days=320,\\n"
                "        target_name=TARGET_COL,\\n"
                "    )\\n\\n"
                "train_df, val_df, test_df = time_based_split(\\n"
                "    dataset,\\n"
                "    date_col='date',\\n"
                "    val_fraction=VAL_FRACTION,\\n"
                "    test_fraction=TEST_FRACTION,\\n"
                ")\\n\\n"
                "report = build_full_comparison_report(\\n"
                "    X_train=train_df[feature_cols],\\n"
                "    y_train=train_df[TARGET_COL],\\n"
                "    X_val=val_df[feature_cols],\\n"
                "    y_val=val_df[TARGET_COL],\\n"
                "    X_test=test_df[feature_cols],\\n"
                "    y_test=test_df[TARGET_COL],\\n"
                "    feature_names=feature_cols,\\n"
                "    test_metadata=test_df[['date', 'ticker']],\\n"
                "    train_metadata=train_df[['date', 'ticker']],\\n"
                "    val_metadata=val_df[['date', 'ticker']],\\n"
                "    target_name=TARGET_COL,\\n"
                "    transaction_cost_bps=TRANSACTION_COST_BPS,\\n"
                "    ridge_params=RIDGE_PARAMS,\\n"
                "    rf_params=RF_PARAMS,\\n"
                "    xgb_params=XGB_PARAMS,\\n"
                "    include_shap=INCLUDE_SHAP,\\n"
                ")\\n\\n"
                "comparison = report['comparison']['model_comparison']\\n"
                "comparison"
            ),
            _code_cell(
                "if EXPORT_CSV:\\n"
                "    written = export_reports_to_csv(report, output_root=OUTPUT_DIR)\\n"
                "    pd.DataFrame([{'table': k, 'path': str(v)} for k, v in written.items()])\\n"
                "else:\\n"
                "    print('CSV export disabled. Set EXPORT_CSV = True to save tables.')"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=2).encode("utf-8")


st.set_page_config(page_title="Full SCL-alpha model", page_icon="🧾", layout="wide")
st.title("🧾 Full SCL-alpha model")
st.caption(
    "Standardized model output report across Ridge, Random Forest, and XGBoost "
    "for the same supply-chain universe."
)
st.caption(f"Universe: {', '.join(TICKERS)}")

with st.expander("Data & run settings", expanded=True):
    data_source = st.radio(
        "Data source",
        options=["Use project supply-chain data", "Use synthetic demo data"],
        index=0,
        horizontal=True,
        help="Project data uses your real feature pipeline. Synthetic data is useful for quick demos.",
    )
    refresh_data = st.checkbox(
        "Refresh data",
        value=False,
        help="Only used for project data. Disabled = cached data for faster runs.",
    )

    c1, c2, c3 = st.columns(3)
    val_fraction = c1.slider(
        "Validation fraction",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        help="Date-based split fraction for validation set.",
    )
    test_fraction = c2.slider(
        "Test fraction",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        help="Date-based split fraction for final test set.",
    )
    transaction_cost_bps = c3.number_input(
        "Transaction cost (bps)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
        help="Used for simple net-prediction estimate columns in report outputs.",
    )

    c4, c5, c6 = st.columns(3)
    include_shap = c4.checkbox(
        "Compute SHAP for XGBoost",
        value=False,
        help="Optional and slower. If SHAP package is missing, report will skip gracefully.",
    )
    export_csv = c5.checkbox(
        "Export all tables to CSV",
        value=False,
        help="Writes all generated report tables to model folders for download/review.",
    )
    output_dir = c6.text_input(
        "Output folder",
        value="outputs/full_scl_alpha_model",
        help="Folder where exported CSV tables will be saved.",
    )

with st.expander("Model parameters", expanded=True):
    st.markdown("#### Ridge")
    ridge_alpha = st.number_input(
        "Alpha",
        min_value=0.001,
        max_value=100.0,
        value=float(RIDGE_ALPHA),
        step=0.1,
        format="%.3f",
        help="L2 regularization strength.",
    )

    st.markdown("#### Random Forest")
    rf1, rf2, rf3, rf4 = st.columns(4)
    rf_n_estimators = rf1.slider(
        "Trees",
        min_value=50,
        max_value=1000,
        value=int(RF_PARAMS.get("n_estimators", 300)),
        step=25,
        help="Number of trees.",
    )
    rf_max_depth = rf2.slider(
        "Max depth",
        min_value=2,
        max_value=20,
        value=int(RF_PARAMS.get("max_depth", 8)),
        help="Maximum depth of trees.",
    )
    rf_min_leaf = rf3.slider(
        "Min leaf size",
        min_value=1,
        max_value=50,
        value=int(RF_PARAMS.get("min_samples_leaf", 5)),
        help="Minimum samples per leaf.",
    )
    rf_max_features = rf4.selectbox(
        "Max features",
        options=["sqrt", "log2", "None"],
        index=0,
        help="Feature sampling strategy per split.",
    )

    st.markdown("#### XGBoost")
    x1, x2, x3, x4 = st.columns(4)
    xgb_n_estimators = x1.slider(
        "Trees",
        min_value=100,
        max_value=1200,
        value=int(XGB_PARAMS.get("n_estimators", 400)),
        step=25,
        help="Number of boosting rounds.",
    )
    xgb_max_depth = x2.slider(
        "Max depth",
        min_value=2,
        max_value=10,
        value=int(XGB_PARAMS.get("max_depth", 4)),
        help="Maximum tree depth per boosting round.",
    )
    xgb_learning_rate = x3.number_input(
        "Learning rate",
        min_value=0.01,
        max_value=0.5,
        value=float(XGB_PARAMS.get("learning_rate", 0.05)),
        step=0.01,
        format="%.2f",
        help="Smaller values are usually more stable.",
    )
    xgb_early_stopping = x4.slider(
        "Early stopping rounds",
        min_value=5,
        max_value=100,
        value=int(XGB_PARAMS.get("early_stopping_rounds", 20)),
        help="Stops training if validation metric does not improve.",
    )

    x5, x6, x7, x8 = st.columns(4)
    xgb_subsample = x5.slider(
        "Subsample",
        min_value=0.4,
        max_value=1.0,
        value=float(XGB_PARAMS.get("subsample", 0.9)),
        step=0.05,
    )
    xgb_colsample = x6.slider(
        "Colsample by tree",
        min_value=0.4,
        max_value=1.0,
        value=float(XGB_PARAMS.get("colsample_bytree", 0.9)),
        step=0.05,
    )
    xgb_reg_alpha = x7.number_input(
        "Reg alpha",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.1,
    )
    xgb_reg_lambda = x8.number_input(
        "Reg lambda",
        min_value=0.0,
        max_value=20.0,
        value=1.0,
        step=0.1,
    )

if st.button("Run Full SCL-alpha model", type="primary", use_container_width=True):
    if val_fraction + test_fraction >= 0.8:
        st.error("Validation + test fractions must be less than 0.8.")
        st.stop()

    with st.spinner("Preparing dataset and generating standardized model reports..."):
        try:
            if data_source == "Use project supply-chain data":
                feature_matrix = load_feature_matrix(refresh_data=refresh_data)
                dataset, feature_cols = _prepare_real_dataset(feature_matrix)
                source_note = "project"
            else:
                # Synthetic fallback keeps the same ticker universe and feature label style.
                feature_cols = [c for c in FEATURE_COLUMNS]
                dataset = create_synthetic_example_dataset(
                    feature_names=feature_cols,
                    tickers=TICKERS,
                    n_days=320,
                    target_name=TARGET_COL,
                )
                source_note = "synthetic"

            train_df, val_df, test_df = time_based_split(
                dataset,
                date_col="date",
                val_fraction=float(val_fraction),
                test_fraction=float(test_fraction),
            )

            report = build_full_comparison_report(
                X_train=train_df[feature_cols],
                y_train=train_df[TARGET_COL],
                X_val=val_df[feature_cols],
                y_val=val_df[TARGET_COL],
                X_test=test_df[feature_cols],
                y_test=test_df[TARGET_COL],
                feature_names=feature_cols,
                test_metadata=test_df[["date", "ticker"]],
                train_metadata=train_df[["date", "ticker"]],
                val_metadata=val_df[["date", "ticker"]],
                target_name=TARGET_COL,
                transaction_cost_bps=float(transaction_cost_bps),
                ridge_params={"alpha": float(ridge_alpha)},
                rf_params={
                    "n_estimators": int(rf_n_estimators),
                    "max_depth": int(rf_max_depth),
                    "min_samples_leaf": int(rf_min_leaf),
                    "max_features": None if rf_max_features == "None" else rf_max_features,
                    "bootstrap": True,
                    "oob_score_enabled": True,
                    "random_state": 42,
                },
                xgb_params={
                    "n_estimators": int(xgb_n_estimators),
                    "max_depth": int(xgb_max_depth),
                    "learning_rate": float(xgb_learning_rate),
                    "subsample": float(xgb_subsample),
                    "colsample_bytree": float(xgb_colsample),
                    "reg_alpha": float(xgb_reg_alpha),
                    "reg_lambda": float(xgb_reg_lambda),
                    "random_state": 42,
                    "early_stopping_rounds": int(xgb_early_stopping),
                },
                include_shap=bool(include_shap),
            )

            st.session_state["full_scl_alpha_report"] = report
            st.session_state["full_scl_alpha_source"] = source_note
            st.session_state["full_scl_alpha_feature_count"] = len(feature_cols)

            if export_csv:
                written = export_reports_to_csv(report, output_root=output_dir)
                st.session_state["full_scl_alpha_exports"] = written
                st.success(f"Run complete and exported {len(written)} CSV table(s) to {output_dir}.")
            else:
                st.success("Run complete.")

        except Exception as exc:
            st.error(f"Full report run failed: {exc}")

report = st.session_state.get("full_scl_alpha_report")
if not report:
    st.info("Run the model to generate the full standardized report tables.")
    st.stop()

st.subheader("Model comparison")
comparison_df = report.get("comparison", {}).get("model_comparison", pd.DataFrame())
_show_df(
    comparison_df,
    {
        "test_mae": "{:.3f}",
        "test_rmse": "{:.3f}",
        "test_r2": "{:.3f}",
        "test_directional_accuracy": "{:.1%}",
        "test_ic_pearson": "{:.3f}",
        "test_ic_spearman": "{:.3f}",
        "prediction_std": "{:.3f}",
        "n_long_signals_avg_per_day": "{:.2f}",
    },
)
_download_df(comparison_df, file_name_stem="model_comparison", key="dl_model_comparison")

source_note = st.session_state.get("full_scl_alpha_source", "project")
feature_count = st.session_state.get("full_scl_alpha_feature_count", "n/a")
st.caption(f"Data source: {source_note} | Features used: {feature_count}")
notebook_config = {
    "data_source": data_source,
    "refresh_data": bool(refresh_data),
    "val_fraction": float(val_fraction),
    "test_fraction": float(test_fraction),
    "transaction_cost_bps": float(transaction_cost_bps),
    "include_shap": bool(include_shap),
    "export_csv": bool(export_csv),
    "output_dir": output_dir,
    "ridge_alpha": float(ridge_alpha),
    "rf_n_estimators": int(rf_n_estimators),
    "rf_max_depth": int(rf_max_depth),
    "rf_min_leaf": int(rf_min_leaf),
    "rf_max_features": None if rf_max_features == "None" else rf_max_features,
    "xgb_n_estimators": int(xgb_n_estimators),
    "xgb_max_depth": int(xgb_max_depth),
    "xgb_learning_rate": float(xgb_learning_rate),
    "xgb_early_stopping": int(xgb_early_stopping),
    "xgb_subsample": float(xgb_subsample),
    "xgb_colsample": float(xgb_colsample),
    "xgb_reg_alpha": float(xgb_reg_alpha),
    "xgb_reg_lambda": float(xgb_reg_lambda),
}

with st.expander("Notebook export", expanded=False):
    st.caption("Generate a Jupyter notebook with the same run settings and model workflow.")
    notebook_bytes = _build_notebook_bytes(notebook_config)
    notebook_name = (
        f"full_scl_alpha_model_{source_note}_"
        f"{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.ipynb"
    )
    st.download_button(
        label="Download notebook (.ipynb)",
        data=notebook_bytes,
        file_name=notebook_name,
        mime="application/x-ipynb+json",
        key="dl_full_scl_alpha_notebook",
        use_container_width=True,
    )


ridge_tab, rf_tab, xgb_tab = st.tabs(["Ridge Regression", "Random Forest", "XGBoost"])

with ridge_tab:
    pack = report.get("ridge", {})
    _show_with_download(
        "Model Summary",
        pack.get("model_summary", pd.DataFrame()),
        file_name_stem="ridge_model_summary",
        key="dl_ridge_model_summary",
    )
    _show_with_download(
        "Performance Metrics",
        pack.get("performance_metrics", pd.DataFrame()),
        file_name_stem="ridge_performance_metrics",
        key="dl_ridge_performance_metrics",
        formats={
            "mae": "{:.3f}",
            "rmse": "{:.3f}",
            "r2": "{:.3f}",
            "directional_accuracy": "{:.1%}",
            "information_coefficient_pearson": "{:.3f}",
            "information_coefficient_spearman": "{:.3f}",
        },
    )
    _show_with_download(
        "Daily Signal Summary",
        pack.get("daily_signal_summary", pd.DataFrame()),
        file_name_stem="ridge_daily_signal_summary",
        key="dl_ridge_daily_signal_summary",
        formats={
            "avg_predicted_return": "{:.2f}",
            "top_pred_1": "{:.2f}",
            "top_pred_2": "{:.2f}",
            "top_pred_3": "{:.2f}",
        },
    )
    _show_with_download(
        "Model Health",
        pack.get("model_health", pd.DataFrame()),
        file_name_stem="ridge_model_health",
        key="dl_ridge_model_health",
    )

    with st.expander("Predictions (Test)"):
        df = pack.get("predictions_test", pd.DataFrame())
        _show_df(
            df,
            {
                "y_true": "{:.3f}",
                "y_pred": "{:.3f}",
                "residual": "{:.3f}",
                "abs_error": "{:.3f}",
                "confidence_score": "{:.2f}",
            },
            max_rows=300,
        )
        _download_df(df, file_name_stem="ridge_predictions_test", key="dl_ridge_predictions_test")

    with st.expander("Ridge Coefficients"):
        df = pack.get("ridge_coefficients", pd.DataFrame())
        _show_df(df, {"coefficient": "{:.4f}", "abs_coefficient": "{:.4f}"})
        _download_df(df, file_name_stem="ridge_coefficients", key="dl_ridge_coefficients")

    with st.expander("Ridge Hyperparameters"):
        df = pack.get("ridge_hyperparams", pd.DataFrame())
        _show_df(df)
        _download_df(df, file_name_stem="ridge_hyperparams", key="dl_ridge_hyperparams")

    with st.expander("Ridge Diagnostics"):
        df = pack.get("ridge_diagnostics", pd.DataFrame())
        _show_df(df, {"value": "{:.4f}"})
        _download_df(df, file_name_stem="ridge_diagnostics", key="dl_ridge_diagnostics")

with rf_tab:
    pack = report.get("random_forest", {})
    _show_with_download(
        "Model Summary",
        pack.get("model_summary", pd.DataFrame()),
        file_name_stem="random_forest_model_summary",
        key="dl_rf_model_summary",
    )
    _show_with_download(
        "Performance Metrics",
        pack.get("performance_metrics", pd.DataFrame()),
        file_name_stem="random_forest_performance_metrics",
        key="dl_rf_performance_metrics",
        formats={
            "mae": "{:.3f}",
            "rmse": "{:.3f}",
            "r2": "{:.3f}",
            "directional_accuracy": "{:.1%}",
            "information_coefficient_pearson": "{:.3f}",
            "information_coefficient_spearman": "{:.3f}",
        },
    )
    _show_with_download(
        "Daily Signal Summary",
        pack.get("daily_signal_summary", pd.DataFrame()),
        file_name_stem="random_forest_daily_signal_summary",
        key="dl_rf_daily_signal_summary",
        formats={
            "avg_predicted_return": "{:.2f}",
            "top_pred_1": "{:.2f}",
            "top_pred_2": "{:.2f}",
            "top_pred_3": "{:.2f}",
        },
    )
    _show_with_download(
        "Model Health",
        pack.get("model_health", pd.DataFrame()),
        file_name_stem="random_forest_model_health",
        key="dl_rf_model_health",
    )

    with st.expander("Predictions (Test)"):
        df = pack.get("predictions_test", pd.DataFrame())
        _show_df(
            df,
            {
                "y_true": "{:.3f}",
                "y_pred": "{:.3f}",
                "residual": "{:.3f}",
                "abs_error": "{:.3f}",
                "confidence_score": "{:.2f}",
            },
            max_rows=300,
        )
        _download_df(df, file_name_stem="random_forest_predictions_test", key="dl_rf_predictions_test")

    with st.expander("Random Forest Feature Importance"):
        df = pack.get("rf_feature_importance", pd.DataFrame())
        _show_df(df, {"importance_gini_or_mse_decrease": "{:.5f}"})
        _download_df(df, file_name_stem="random_forest_feature_importance", key="dl_rf_feature_importance")

    with st.expander("Random Forest Permutation Importance"):
        df = pack.get("rf_permutation_importance", pd.DataFrame())
        _show_df(df, {"perm_importance_mean": "{:.5f}", "perm_importance_std": "{:.5f}"})
        _download_df(df, file_name_stem="random_forest_permutation_importance", key="dl_rf_permutation_importance")

    with st.expander("Random Forest Hyperparameters"):
        df = pack.get("rf_hyperparams", pd.DataFrame())
        _show_df(df)
        _download_df(df, file_name_stem="random_forest_hyperparams", key="dl_rf_hyperparams")

    with st.expander("Random Forest Diagnostics"):
        df = pack.get("rf_diagnostics", pd.DataFrame())
        _show_df(df, {"value": "{:.4f}"})
        _download_df(df, file_name_stem="random_forest_diagnostics", key="dl_rf_diagnostics")

with xgb_tab:
    pack = report.get("xgboost", {})
    _show_with_download(
        "Model Summary",
        pack.get("model_summary", pd.DataFrame()),
        file_name_stem="xgboost_model_summary",
        key="dl_xgb_model_summary",
    )
    _show_with_download(
        "Performance Metrics",
        pack.get("performance_metrics", pd.DataFrame()),
        file_name_stem="xgboost_performance_metrics",
        key="dl_xgb_performance_metrics",
        formats={
            "mae": "{:.3f}",
            "rmse": "{:.3f}",
            "r2": "{:.3f}",
            "directional_accuracy": "{:.1%}",
            "information_coefficient_pearson": "{:.3f}",
            "information_coefficient_spearman": "{:.3f}",
        },
    )
    _show_with_download(
        "Daily Signal Summary",
        pack.get("daily_signal_summary", pd.DataFrame()),
        file_name_stem="xgboost_daily_signal_summary",
        key="dl_xgb_daily_signal_summary",
        formats={
            "avg_predicted_return": "{:.2f}",
            "top_pred_1": "{:.2f}",
            "top_pred_2": "{:.2f}",
            "top_pred_3": "{:.2f}",
        },
    )
    _show_with_download(
        "Model Health",
        pack.get("model_health", pd.DataFrame()),
        file_name_stem="xgboost_model_health",
        key="dl_xgb_model_health",
    )

    with st.expander("Predictions (Test)"):
        df = pack.get("predictions_test", pd.DataFrame())
        _show_df(
            df,
            {
                "y_true": "{:.3f}",
                "y_pred": "{:.3f}",
                "residual": "{:.3f}",
                "abs_error": "{:.3f}",
                "confidence_score": "{:.2f}",
            },
            max_rows=300,
        )
        _download_df(df, file_name_stem="xgboost_predictions_test", key="dl_xgb_predictions_test")

    with st.expander("XGBoost Feature Importance"):
        df = pack.get("xgb_feature_importance", pd.DataFrame())
        _show_df(df, {"importance_value": "{:.5f}"})
        _download_df(df, file_name_stem="xgboost_feature_importance", key="dl_xgb_feature_importance")

    with st.expander("XGBoost SHAP Summary"):
        df = pack.get("xgb_shap_summary", pd.DataFrame())
        _show_df(df, {"mean_abs_shap": "{:.5f}"})
        _download_df(df, file_name_stem="xgboost_shap_summary", key="dl_xgb_shap_summary")

    with st.expander("XGBoost Hyperparameters"):
        df = pack.get("xgb_hyperparams", pd.DataFrame())
        _show_df(df)
        _download_df(df, file_name_stem="xgboost_hyperparams", key="dl_xgb_hyperparams")

    with st.expander("XGBoost Training History"):
        df = pack.get("xgb_training_history", pd.DataFrame())
        _show_df(df, {"train_metric_value": "{:.5f}", "val_metric_value": "{:.5f}"})
        _download_df(df, file_name_stem="xgboost_training_history", key="dl_xgb_training_history")

    with st.expander("XGBoost Diagnostics"):
        df = pack.get("xgb_diagnostics", pd.DataFrame())
        _show_df(df, {"value": "{:.4f}"})
        _download_df(df, file_name_stem="xgboost_diagnostics", key="dl_xgb_diagnostics")

exports = st.session_state.get("full_scl_alpha_exports")
if isinstance(exports, dict) and exports:
    with st.expander("Exported CSV files"):
        rows = [{"table": k, "path": str(v)} for k, v in exports.items()]
        export_df = pd.DataFrame(rows)
        _show_df(export_df)
        _download_df(export_df, file_name_stem="exported_csv_index", key="dl_exported_csv_index")
