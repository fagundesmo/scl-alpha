"""Cloud entrypoint wrapper for Streamlit multipage navigation."""

from pathlib import Path
import runpy

APP_MAIN = Path(__file__).resolve().parent / "app" / "app.py"
runpy.run_path(str(APP_MAIN), run_name="__main__")
