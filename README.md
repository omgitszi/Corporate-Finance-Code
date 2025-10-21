```markdown
# Corporate-Finance-Code

This repository contains a small pipeline for downloading monthly stock prices, computing return summary statistics and annualised covariance/correlation matrices, and constructing Markowitz-style portfolios (efficient frontier, max Sharpe, min volatility) with saved plots.

## Quick overview
- Source code lives under `src/src/` (yes, there's a nested package named `src`).
- The pipeline entrypoint is the package module `src.main`. From the `src/` folder run:

	```powershell
	# from the repository root
	cd src
	python -m src.main
	```

	This will download price data (via `yfinance`), compute statistics and covariance matrices, run the Markowitz optimization, and save CSVs and PNGs into `src/output`.

## Project structure (important files)
- `src/download_data.py`  — download prices and build monthly returns
- `src/calculations.py`   — compute summary statistics and covariance/correlation matrices; save `summary.csv` (pivoted)
- `src/markowitz.py`     — compute portfolios, efficient frontier and save plots
- `src/main.py`          — orchestrator (editable tickers list at top)
- `output/`              — pipeline outputs (CSV + PNG)

## Outputs you should expect
- `output/monthly_returns.csv` (monthly returns, wide format)
- `output/summary.csv` (pivoted summary statistics: rows = stat, cols = symbols)
- `output/covariance_matrix.csv` (annualised covariance)
- `output/correlation_matrix.csv` (annualised correlation)
- `output/assets_figure.png` (asset scatter / summary)
- `output/efficient_frontier.png` (efficient frontier image)
- `output/weights_comparison.png` (comparison of portfolio weights)

## Reproducible environment
Two options are provided below — Conda (recommended) and pip/venv. Both are kept intentionally small; tweak versions as needed.

### Option A — Conda (recommended)
Create the conda environment from `environment.yml` (this file is included at the repo root):

```powershell
conda env create -f environment.yml
conda activate ec3318
cd src
python -m src.main
```

### Option B — venv + pip
Create a virtual environment and install packages from `requirements.txt` (repo root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r ..\requirements.txt
cd src
python -m src.main
```

Note: On Windows PowerShell use the `Activate.ps1` script shown above. On other shells adapt accordingly.

## Troubleshooting and notes
- If you see an error complaining about `adjustText`, the code includes a fallback that will still run; to enable nicer label layout install it explicitly:

```powershell
pip install adjustText
```

- If you get a `FileNotFoundError` for `output/monthly_returns.csv`, ensure `download_data` was able to download data (internet required) and that you ran the orchestrator from `src/` as shown above.
- If you hit missing-package errors, re-create the environment from `environment.yml` or run `pip install -r requirements.txt`.

## Contributing / extending
- The ticker list is editable at the top of `src/main.py` and is copied/used by `download_data.py` — if you want a single source-of-truth you can move the tickers into a small `config.py` and import it from both modules.
- Outputs are intentionally small CSV/PNG artifacts so you can iterate locally and check the results quickly.

## License
This repo is provided as-is for educational use. Add your own license if you intend to redistribute.

```
# Corporate-Finance-Code
