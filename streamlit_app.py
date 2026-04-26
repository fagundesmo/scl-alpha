"""Cloud entrypoint with explicit page navigation."""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

navigation = st.navigation(
    [
        st.Page(str(ROOT / "app" / "app.py"), title="Home", icon="🏠"),
        st.Page(str(ROOT / "app" / "pages" / "pre_data.py"), title="Pre-data", icon="📥"),
        st.Page(str(ROOT / "app" / "pages" / "sandbox_models.py"), title="Sandbox Models", icon="🧪"),
        st.Page(str(ROOT / "app" / "pages" / "output.py"), title="Output", icon="📊"),
    ],
    position="sidebar",
)

navigation.run()
