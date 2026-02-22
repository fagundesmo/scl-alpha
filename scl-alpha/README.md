# scl-alpha

Portable ML-driven quantitative trading research project for supply-chain and logistics equities.

## Python requirement

- Python `3.11+` is required.
- The project uses modern type syntax (for example `list[str] | None`) and dependency versions targeted at Python 3.11.

## Clean machine quickstart

```bash
cd /path/to/scl-alpha
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install .
```

Install dev/test tooling:

```bash
pip install .[dev]
```

## Environment setup

```bash
cp .env.example .env
```

Set your FRED key in `.env`:

```env
FRED_API_KEY=your_fred_api_key
```

If `FRED_API_KEY` is missing, FRED-backed pulls fail with a clear error. Fama-French download is best-effort and network-dependent.

## Run app

```bash
streamlit run app/app.py
```

## Run tests

```bash
pytest -q
```

## Packaging and install

- First-class install path: `pip install .`
- Dev extras: `pip install .[dev]`
- Package namespace stays `src`, so imports like `from src.backtest import run_backtest` continue to work.

## Dependency sync rule

- Runtime dependencies are declared in both:
  - `pyproject.toml` (`[project.dependencies]`)
  - `requirements.txt` (deployment/runtime)
- Keep them synchronized whenever dependencies are changed.
- Test-only tooling belongs in `pyproject.toml` extras (`[project.optional-dependencies].dev`).
