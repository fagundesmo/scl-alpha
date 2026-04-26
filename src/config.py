"""
Central configuration for the scl-alpha project.
All magic numbers, ticker lists, date ranges, and hyperparameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv


def _resolve_project_root() -> Path:
    explicit_root = os.getenv("SCL_ALPHA_HOME")
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    repo_root = Path(__file__).resolve().parent.parent
    if (repo_root / "src").is_dir() and (repo_root / "app").is_dir():
        return repo_root

    return (Path.home() / ".scl-alpha").resolve()


PROJECT_ROOT = _resolve_project_root()

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

if (PROJECT_ROOT / "app").is_dir():
    CACHE_DIR = PROJECT_ROOT / "app" / "cache"
else:
    CACHE_DIR = PROJECT_ROOT / "cache"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, CACHE_DIR, EXPERIMENTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ticker Universe  (supply-chain & logistics companies from the notebook)
# ---------------------------------------------------------------------------
TICKERS = [
    "UPS",   # United Parcel Service — package delivery
    "FDX",   # FedEx — express logistics
    "JBHT",  # J.B. Hunt Transport — intermodal / trucking
    "CHRW",  # C.H. Robinson — freight brokerage
    "EXPD",  # Expeditors International — freight forwarding
    "GXO",   # GXO Logistics — contract logistics (IPO Aug 2021)
    "LSTR",  # Landstar System — transportation management
    "HUBG",  # Hub Group — intermodal transportation
    "PBI",   # Pitney Bowes — shipping & mailing
    "FWRD",  # Forward Air — airport-to-airport freight
    "RLGT",  # Radiant Logistics — freight brokerage
]

BENCHMARK_ETF = "IYT"   # iShares US Transportation ETF
MARKET_ETF = "SPY"      # S&P 500 ETF

ALL_SYMBOLS = TICKERS + [BENCHMARK_ETF, MARKET_ETF]

# ---------------------------------------------------------------------------
# Date Range
# ---------------------------------------------------------------------------
DATA_START = "2018-01-01"

# ---------------------------------------------------------------------------
# FRED Series
# ---------------------------------------------------------------------------
FRED_SERIES = ["VIXCLS", "GASDESW", "DGS10", "ICNSA"]

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
BETA_WIN = 63            # rolling beta / correlation window (trading days)
FEATURE_BURN_IN_DAYS = 252

# ---------------------------------------------------------------------------
# Target Variable
# ---------------------------------------------------------------------------
TARGET_COL = "target_fwd_1d"   # next-day log return (from build_features_for_ticker)

# ---------------------------------------------------------------------------
# Modeling defaults
# ---------------------------------------------------------------------------
RIDGE_ALPHA = 1.0

RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "min_samples_leaf": 20,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "early_stopping_rounds": 20,
}

# ---------------------------------------------------------------------------
# Backtesting (kept for compatibility with legacy pages)
# ---------------------------------------------------------------------------
INITIAL_TRAIN_YEARS = 3
RETRAIN_EVERY_WEEKS = 13
TOP_K = 3
TRANSACTION_COST_BPS = 5
SLIPPAGE_BPS = 2
RANDOM_SEED = 42
