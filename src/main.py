"""
Main runner for the pipeline (moved inside src/src):
1. Downloads monthly returns for a list of tickers (editable below)
2. Runs calculations to produce summary statistics and assets figure
3. Runs markowitz routines to compute covariance, efficient frontier and weight comparisons

Outputs written to src/output/
"""
from pathlib import Path
import os

# Editable ticker list - change to the tickers you want to analyze
TICKERS = [
    "GOOGL",
    "DIS",
    "MCD",
    "TSLA",
    "KO",
    "WMT",
    "XOM",
    "JPM",
    "JNJ",
    "PFE",
    "BA",
    "AAPL",
    "NVDA",
    "DD",
    "AMT",
    "DUK",
    "EQIX",
    "LULU",
]

# Import local modules (they live under src/src)
import download_data, calculations, markowitz


def main(tickers=None):
    # use shared output folder at src/output (one level up from this package)
    repo_src = Path(__file__).resolve().parents[1]
    output_dir = repo_src / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    if tickers is None:
        tickers = TICKERS

    # 1. Download data
    print("Downloading monthly returns for:", tickers)
    monthly_returns = download_data.get_monthly_returns(tickers, start_date="2015-08-01", end_date="2025-08-29")
    if monthly_returns is None or monthly_returns.empty:
        raise SystemExit("Failed to download monthly returns, aborting.")

    # ensure monthly_returns.csv is written into output (download_data.get_monthly_returns
    # returns a DataFrame but does not save by default)
    monthly_file = output_dir / "monthly_returns.csv"
    monthly_returns.to_csv(str(monthly_file))
    print(f"Saved downloaded monthly returns to: {monthly_file}")

    # 2. Calculations: summary stats, asset summary, figure, covariance
    stats = calculations.calculate_summary_stats(str(output_dir / "monthly_returns.csv"))
    asset_summary = calculations.build_asset_summary(stats)
    # save pivoted summary
    pivoted = asset_summary.set_index("symbol").T
    pivoted.to_csv(str(output_dir / "summary.csv"))
    fig = calculations.build_assets_figure(asset_summary)
    fig.save(str(output_dir / "assets_figure.png"), dpi=150)

    # 3. Markowitz: compute covariance and run portfolio routines
    cov_matrix, mu = markowitz.compute_annualized_covariance(str(output_dir / "monthly_returns.csv"), symbols=list(asset_summary["symbol"]))
    markowitz.max_sharpe_ratio(cov_matrix, mu)
    markowitz.min_vol_port(cov_matrix, mu)
    markowitz.unconstrained_portfolio(cov_matrix, mu)
    # save efficient frontier and weight plots
    markowitz.plot_and_save_efficient_frontier(cov_matrix, mu, str(output_dir / "efficient_frontier.png"))
    key_results = markowitz.compute_key_portfolios(mu, cov_matrix)
    markowitz.plot_and_save_weights_comparison(key_results, str(output_dir / "weights_comparison.png"))

    print("All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
