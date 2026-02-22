# Data Directory

## Structure

- `raw/` — Cached raw data from API pulls (parquet). Gitignored.
- `processed/` — Feature matrices ready for modeling (parquet). Gitignored.

## How to Populate

Run the data pull from Python:

```python
from src.data_pull import pull_all
panel = pull_all()
```

Or run the notebook `notebooks/01_full_pipeline.ipynb` (Section 1).

## Data Sources

| File | Source | How to refresh |
|------|--------|---------------|
| `raw/prices.parquet` | Yahoo Finance via `yfinance` | Delete file and re-run `fetch_prices()` |
| `raw/fred.parquet` | FRED API via `fredapi` | Delete file and re-run `fetch_fred()` |
| `raw/fama_french.parquet` | Kenneth French Data Library | Delete file and re-run `fetch_fama_french()` |
| `processed/panel.parquet` | Merged from the above | Re-run `pull_all()` |
