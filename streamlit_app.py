"""Cloud entrypoint with explicit page navigation."""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

navigation = st.navigation(
    [
        st.Page(str(ROOT / "app" / "app.py"), title="Home", icon="🏠"),
        st.Page(str(ROOT / "app" / "pages" / "0_model_lab.py"), title="Model Lab", icon="🧪"),
        st.Page(str(ROOT / "app" / "pages" / "1_signals.py"), title="Signals", icon="📊"),
        st.Page(str(ROOT / "app" / "pages" / "2_backtest.py"), title="Backtest", icon="📈"),
        st.Page(str(ROOT / "app" / "pages" / "3_explain.py"), title="Explain", icon="🔍"),
        st.Page(str(ROOT / "app" / "pages" / "4_data.py"), title="Data", icon="📋"),
    ],
    position="sidebar",
)

navigation.run()
