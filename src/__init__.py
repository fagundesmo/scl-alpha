"""scl-alpha package."""

import sys

if sys.version_info < (3, 11):
    raise RuntimeError(
        "scl-alpha requires Python 3.11 or newer. "
        f"Detected {sys.version.split()[0]}."
    )

__version__ = "0.1.0"
