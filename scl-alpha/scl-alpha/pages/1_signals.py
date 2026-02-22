"""Wrapper page for Cloud navigation."""

from pathlib import Path
import runpy

PAGE = Path(__file__).resolve().parent.parent / "app" / "pages" / "1_signals.py"
runpy.run_path(str(PAGE), run_name="__main__")
