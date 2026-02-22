"""
Central configuration for the scl-alpha project.
All magic numbers, ticker lists, date ranges, and hyperparameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def _resolve_project_root() -> Path:
    """
    Resolve a writable project root deterministically across environments.

    Priority:
    1) SCL_ALPHA_HOME env var
    2) Repository root (when running from source checkout)
    3) ~/.scl-alpha fallback (when installed as a package)
    """
    explicit_root = os.getenv("SCL_ALPHA_HOME")
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "src").is_dir() and (repo_root / "app").is_dir():
        return repo_root

    return (Path.home() / ".scl-alpha").resolve()


PROJECT_ROOT = _resolve_project_root()

# Load .env from project root when present, otherwise rely on default lookup.
dotenv_path = PROJECT_ROOT / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
else:
    load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# If app folder exists (source checkout), keep current cache location.
# Otherwise use a package-safe cache directory under PROJECT_ROOT.
if (PROJECT_ROOT / "app").is_dir():
    CACHE_DIR = PROJECT_ROOT / "app" / "cache"
else:
    CACHE_DIR = PROJECT_ROOT / "cache"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, CACHE_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Ticker Universe
# ---------------------------------------------------------------------------
TICKERS = [
    "UPS",   # United Parcel Service — package delivery
    "FDX",   # FedEx — express logistics
    "XPO",   # XPO Inc — freight brokerage
    "CHRW",  # C.H. Robinson — freight brokerage
    "JBHT",  # J.B. Hunt Transport — intermodal / trucking
    "UNP",   # Union Pacific — rail
    "CSX",   # CSX Corp — rail
    "MATX",  # Matson Inc — ocean shipping
    "GXO",   # GXO Logistics — contract logistics (IPO Aug 2021)
    "EXPD",  # Expeditors Int'l — freight forwarding
]

BENCHMARK_ETF = "IYT"   # iShares US Transportation ETF
MARKET_ETF = "SPY"      # S&P 500 ETF

# All symbols we need price data for
ALL_SYMBOLS = TICKERS + [BENCHMARK_ETF, MARKET_ETF]

# ---------------------------------------------------------------------------
# Date Ranges
# ---------------------------------------------------------------------------
DATA_START = "2018-01-01"
DATA_END = "2026-02-14"       # Update as needed

# Walk-forward settings
INITIAL_TRAIN_YEARS = 3       # 2018-2020 for the first training window
RETRAIN_EVERY_WEEKS = 13      # Quarterly retraining
REBALANCE_DAY = "Friday"      # Signal day; trade on next Monday open

# ---------------------------------------------------------------------------
# FRED Series
# ---------------------------------------------------------------------------
FRED_SERIES = {
    "VIXCLS":  "VIX close",
    "GASDESW": "US diesel price ($/gal, weekly)",
    "DGS10":   "10-Year Treasury yield (daily)",
    "ICNSA":   "Initial jobless claims (weekly)",
}

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
LOOKBACK_WINDOWS = {
    "ret_1d": 1,
    "ret_5d": 5,
    "ret_20d": 20,
    "vol_20d": 20,
    "rsi_14": 14,
    "rolling_beta_60d": 60,
    "vol_regime_252d": 252,
}

# Burn-in: the number of trading days we must drop from the start so all
# rolling features are defined. Equal to the longest lookback.
FEATURE_BURN_IN_DAYS = 252

# ---------------------------------------------------------------------------
# Target Variable
# ---------------------------------------------------------------------------
FORWARD_RETURN_DAYS = 5       # Predict the 5-trading-day forward return

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
# Ridge
RIDGE_ALPHA = 1.0

# Random Forest
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "random_state": 42,
    "n_jobs": -1,
}

# XGBoost
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "early_stopping_rounds": 20,
}

# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
TOP_K = 3                     # Number of stocks to go long
TRANSACTION_COST_BPS = 5      # One-way cost in basis points
SLIPPAGE_BPS = 2              # Additional slippage on execution
RISK_FREE_RATE = 0.0          # Simplified; cash earns nothing

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
