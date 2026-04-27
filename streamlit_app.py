"""Cloud entrypoint with explicit page navigation."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

# Ensure src/ (which lives next to this file) is importable regardless of
# where Streamlit's working directory is set (e.g. repo root vs subfolder).
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
