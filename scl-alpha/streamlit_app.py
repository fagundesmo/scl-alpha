"""Streamlit Cloud entrypoint — delegates to the repo root app."""

import sys
from pathlib import Path

# ROOT is one level up (the actual repo root where src/, app/ live)
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

navigation = st.navigation(
    [
        st.Page(str(ROOT / "app" / "app.py"), title="Home", icon="🏠"),
        st.Page(str(ROOT / "app" / "pages" / "pre_data.py"), title="Pre-data", icon="📥"),
        st.Page(str(ROOT / "app" / "pages" / "sandbox_models.py"), title="Sandbox Models", icon="🧪"),
        st.Page(str(ROOT / "app" / "pages" / "predictions.py"), title="Predictions", icon="🔮"),
        st.Page(str(ROOT / "app" / "pages" / "output.py"), title="Output", icon="📊"),
    ],
    position="sidebar",
)

navigation.run()
