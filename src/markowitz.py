import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from pypfopt import EfficientFrontier, objective_functions

def compute_annualized_covariance(
    monthly_csv_path="monthly_returns.csv",
    symbols=None,
    annualisation_factor=12,
    output_dir=None,
):
    p = Path(monthly_csv_path)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    returns = df.select_dtypes(include=[np.number]).copy()
    if returns.empty:
        raise ValueError("No numeric return columns found in CSV")
    if symbols is not None:
        symbols_found = [s for s in symbols if s in returns.columns]
        missing = [s for s in symbols if s not in returns.columns]
        if missing:
            import warnings

            warnings.warn(
                f"Some requested symbols not found in returns CSV and will be ignored: {missing}"
            )
        returns = returns.loc[:, symbols_found]
    mu = returns.mean(axis=0) * annualisation_factor
    cov_monthly = returns.cov(ddof=1)
    cov_annual = cov_monthly * annualisation_factor
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cov_file = output_dir / "covariance_matrix.csv"
    cov_annual.to_csv(str(cov_file))
    mu = pd.Series(mu)
    return cov_annual, mu

def max_sharpe_ratio(cov_matrix, mu):
    ef = EfficientFrontier(mu, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.01)
    ef.max_sharpe()
    cleaned_max = ef.clean_weights()
    max_sharpe_perf = ef.portfolio_performance(verbose=True)

    print("\n" + "=" * 50)
    print("MAXIMUM SHARPE RATIO PORTFOLIO")
    print("=" * 50)
    max_sharpe_weights_df = (
        pd.Series(cleaned_max, name="weight").sort_values(ascending=False).to_frame()
    )

    max_sharpe_metrics = pd.Series(
        max_sharpe_perf,
        index=["expected_return", "volatility", "sharpe_ratio"],
        name="max_sharpe_portfolio",
    )
    print("Portfolio weights:")
    print(max_sharpe_weights_df)
    print(f"\nPerformance metrics:")
    print(max_sharpe_metrics)

def min_vol_port(cov_matrix, mu):
    ef = EfficientFrontier(mu, cov_matrix)
    ef.add_objective(objective_functions.L2_reg, gamma=0.01)
    ef.min_volatility()
    cleaned_min = ef.clean_weights()
    min_vol_perf = ef.portfolio_performance(verbose=True)

    print("\n" + "=" * 50)
    print("MINIMUM VOLATILITY PORTFOLIO")
    print("=" * 50)
    min_vol_weights_df = (
        pd.Series(cleaned_min, name="weight").sort_values(ascending=False).to_frame()
    )

    min_vol_metrics = pd.Series(
        min_vol_perf,
        index=["expected_return", "volatility", "sharpe_ratio"],
        name="min_vol_portfolio",
    )
    print("Portfolio weights:")
    print(min_vol_weights_df)
    print(f"\nPerformance metrics:")
    print(min_vol_metrics)

def unconstrained_portfolio(cov_matrix, mu):
    # Allow shorting by setting weight bounds to (-1, 1)
    ef_unconstrained_sharpe = EfficientFrontier(mu, cov_matrix, weight_bounds=(-1, 1))
    ef_unconstrained_sharpe.add_objective(objective_functions.L2_reg, gamma=0.01)
    # Optimize for maximum Sharpe ratio
    ef_unconstrained_sharpe.max_sharpe()
    # Clean weights (removes positions below threshold)
    cleaned_unconstrained_sharpe = ef_unconstrained_sharpe.clean_weights()
    # Calculate performance metrics
    unconstrained_sharpe_perf = ef_unconstrained_sharpe.portfolio_performance(verbose=False)

    print("\nMaximum Sharpe Portfolio (Unconstrained):")
    unconstrained_sharpe_df = (
        pd.Series(cleaned_unconstrained_sharpe, name="weight")
        .sort_values(ascending=False)
        .to_frame()
    )
    print("Weights:")
    print(unconstrained_sharpe_df)
    print(f"\nExpected Return: {unconstrained_sharpe_perf[0]:.4f}")
    print(f"Volatility: {unconstrained_sharpe_perf[1]:.4f}")
    print(f"Sharpe Ratio: {unconstrained_sharpe_perf[2]:.4f}")

    num_shorts = sum(1 for w in cleaned_unconstrained_sharpe.values() if w < 0)
    num_longs = sum(1 for w in cleaned_unconstrained_sharpe.values() if w > 0)
    print(f"\nShort positions: {num_shorts} assets")
    print(f"Long positions: {num_longs} assets")

    # Minimum volatility with short-selling allowed
    ef_unconstrained_minvol = EfficientFrontier(mu, cov_matrix, weight_bounds=(-1, 1))
    # Add L2 regularization
    ef_unconstrained_minvol.add_objective(objective_functions.L2_reg, gamma=0.01)
    # Optimize for minimum volatility
    ef_unconstrained_minvol.min_volatility()
    # Clean weights
    cleaned_unconstrained_minvol = ef_unconstrained_minvol.clean_weights()
    # Calculate performance
    unconstrained_minvol_perf = ef_unconstrained_minvol.portfolio_performance(verbose=False)

    print("\nMinimum Volatility Portfolio (Unconstrained):")
    unconstrained_minvol_df = (
        pd.Series(cleaned_unconstrained_minvol, name="weight")
        .sort_values(ascending=False)
        .to_frame()
    )
    print("Weights:")
    print(unconstrained_minvol_df)
    print(f"\nExpected Return: {unconstrained_minvol_perf[0]:.4f}")
    print(f"Volatility: {unconstrained_minvol_perf[1]:.4f}")
    print(f"Sharpe Ratio: {unconstrained_minvol_perf[2]:.4f}")

    # Count short positions
    num_shorts_mv = sum(1 for w in cleaned_unconstrained_minvol.values() if w < 0)
    num_longs_mv = sum(1 for w in cleaned_unconstrained_minvol.values() if w > 0)
    print(f"\nShort positions: {num_shorts_mv} assets")
    print(f"Long positions: {num_longs_mv} assets")


ridge_gamma = 0.01  # Regularization parameter for all optimizations


def compute_frontier_points(target_returns, mu_vec, cov, gamma, weight_bounds=(-1, 1)):
    """Sample efficient frontier points for plotting.

    Returns a DataFrame with columns: expected_return, volatility, sharpe_ratio
    """
    rows = []
    for target in target_returns:
        ef = EfficientFrontier(mu_vec, cov, weight_bounds=weight_bounds)
        ef.add_objective(objective_functions.L2_reg, gamma=gamma)
        try:
            ef.efficient_return(target_return=float(target))
            perf = ef.portfolio_performance(verbose=False)
        except (ValueError, OverflowError):
            continue
        rows.append({"expected_return": perf[0], "volatility": perf[1], "sharpe_ratio": perf[2]})

    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["volatility"])  # Remove duplicate points
        .sort_values("volatility")
        .reset_index(drop=True)
    )


def compute_key_portfolios(mu, cov, gamma=ridge_gamma):
    """Compute cleaned weights and performance for key portfolios under unconstrained and long-only bounds."""
    results = {}

    # Unconstrained (allows shorting)
    ef_u_sh = EfficientFrontier(mu, cov, weight_bounds=(-1, 1))
    ef_u_sh.add_objective(objective_functions.L2_reg, gamma=gamma)
    ef_u_sh.max_sharpe()
    cleaned_u_sh = ef_u_sh.clean_weights()
    perf_u_sh = ef_u_sh.portfolio_performance(verbose=False)

    ef_u_mv = EfficientFrontier(mu, cov, weight_bounds=(-1, 1))
    ef_u_mv.add_objective(objective_functions.L2_reg, gamma=gamma)
    ef_u_mv.min_volatility()
    cleaned_u_mv = ef_u_mv.clean_weights()
    perf_u_mv = ef_u_mv.portfolio_performance(verbose=False)

    # Long-only
    ef_l_sh = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef_l_sh.add_objective(objective_functions.L2_reg, gamma=gamma)
    ef_l_sh.max_sharpe()
    cleaned_l_sh = ef_l_sh.clean_weights()
    perf_l_sh = ef_l_sh.portfolio_performance(verbose=False)

    ef_l_mv = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef_l_mv.add_objective(objective_functions.L2_reg, gamma=gamma)
    ef_l_mv.min_volatility()
    cleaned_l_mv = ef_l_mv.clean_weights()
    perf_l_mv = ef_l_mv.portfolio_performance(verbose=False)

    results["cleaned_unconstrained_sharpe"] = cleaned_u_sh
    results["perf_unconstrained_sharpe"] = perf_u_sh
    results["cleaned_unconstrained_minvol"] = cleaned_u_mv
    results["perf_unconstrained_minvol"] = perf_u_mv
    results["cleaned_long_only_sharpe"] = cleaned_l_sh
    results["perf_long_only_sharpe"] = perf_l_sh
    results["cleaned_long_only_minvol"] = cleaned_l_mv
    results["perf_long_only_minvol"] = perf_l_mv

    return results


def plot_and_save_weights_comparison(weights_dict, out_path):
    weights_comparison = pd.DataFrame(
        {
            "Max Sharpe (Un.)": pd.Series(weights_dict["cleaned_unconstrained_sharpe"]),
            "Min Vol (Un.)": pd.Series(weights_dict["cleaned_unconstrained_minvol"]),
            "Max Sharpe (Long)": pd.Series(weights_dict["cleaned_long_only_sharpe"]),
            "Min Vol (Long)": pd.Series(weights_dict["cleaned_long_only_minvol"]),
        }
    ).fillna(0)

    print("\n" + "=" * 50)
    print("PORTFOLIO WEIGHTS COMPARISON")
    print("=" * 50)
    print("Portfolio weights summary:")
    print(weights_comparison)

    # Transpose so x-axis is portfolios and stacks are assets
    plot_df = weights_comparison.T

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_df.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Weight")
    ax.set_title("Portfolio Weights Across Different Optimization Strategies")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_and_save_efficient_frontier(cov_matrix, mu, out_path, gamma=ridge_gamma):
    """Create a matplotlib efficient frontier plot comparing unconstrained and long-only frontiers and save to disk."""
    # grid of target returns
    frontier_grid = np.linspace(mu.min(), mu.max(), 50)

    frontier_unconstrained = compute_frontier_points(
        frontier_grid, mu, cov_matrix, gamma, weight_bounds=(-1, 1)
    )
    frontier_long_only = compute_frontier_points(
        frontier_grid, mu, cov_matrix, gamma, weight_bounds=(0, 1)
    )

    # asset positions
    asset_positions = pd.DataFrame(
        {
            "symbol": mu.index,
            "expected_return": mu.values,
            "volatility": np.sqrt(np.diag(cov_matrix)),
        }
    )

    # key portfolios
    def compute_perf(mu_vec, cov, bounds):
        ef = EfficientFrontier(mu_vec, cov, weight_bounds=bounds)
        ef.add_objective(objective_functions.L2_reg, gamma=gamma)
        # compute both min vol and max sharpe
        ef_min = EfficientFrontier(mu_vec, cov, weight_bounds=bounds)
        ef_min.add_objective(objective_functions.L2_reg, gamma=gamma)
        ef_min.min_volatility()
        min_perf = ef_min.portfolio_performance(verbose=False)

        ef_sh = EfficientFrontier(mu_vec, cov, weight_bounds=bounds)
        ef_sh.add_objective(objective_functions.L2_reg, gamma=gamma)
        ef_sh.max_sharpe()
        sh_perf = ef_sh.portfolio_performance(verbose=False)

        return min_perf, sh_perf

    min_unc_perf, sh_unc_perf = compute_perf(mu, cov_matrix, (-1, 1))
    min_long_perf, sh_long_perf = compute_perf(mu, cov_matrix, (0, 1))

    # build plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        frontier_unconstrained["volatility"],
        frontier_unconstrained["expected_return"],
        label="Unconstrained",
        color="#0d3b66",
        linewidth=2,
    )
    ax.plot(
        frontier_long_only["volatility"],
        frontier_long_only["expected_return"],
        label="Long-only",
        color="#fb8500",
        linewidth=2,
    )

    ax.scatter(
        asset_positions["volatility"],
        asset_positions["expected_return"],
        color="#6c757d",
        s=40,
        alpha=0.7,
        label="Assets",
    )

    # annotate assets
    for _, row in asset_positions.iterrows():
        ax.annotate(row["symbol"], (row["volatility"], row["expected_return"]), fontsize=8)

    # add key portfolios
    ax.scatter([min_unc_perf[1], sh_unc_perf[1]], [min_unc_perf[0], sh_unc_perf[0]], color="#0d3b66", s=80, marker="o")
    ax.scatter([min_long_perf[1], sh_long_perf[1]], [min_long_perf[0], sh_long_perf[0]], color="#fb8500", s=80, marker="^")

    # annotate key points
    ax.annotate("Min Vol (Unconstrained)", (min_unc_perf[1], min_unc_perf[0]), xytext=(10, -10), textcoords="offset points")
    ax.annotate("Max Sharpe (Unconstrained)", (sh_unc_perf[1], sh_unc_perf[0]), xytext=(10, -10), textcoords="offset points")
    ax.annotate("Min Vol (Long-only)", (min_long_perf[1], min_long_perf[0]), xytext=(10, 10), textcoords="offset points")
    ax.annotate("Max Sharpe (Long-only)", (sh_long_perf[1], sh_long_perf[0]), xytext=(10, 10), textcoords="offset points")

    # formatting
    ax.set_xlabel("Volatility (annualised)")
    ax.set_ylabel("Expected Return (annualised)")
    ax.set_title("Efficient Frontier: Impact of Short-Selling Constraints")
    ax.legend()
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



if __name__ == "__main__":
    # ensure outputs go to shared src/output (one level up from package)
    out_dir = Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    cov_matrix, mu = compute_annualized_covariance(output_dir=out_dir)
    max_sharpe_ratio(cov_matrix, mu)
    min_vol_port(cov_matrix, mu)
    unconstrained_portfolio(cov_matrix, mu)
    # Save efficient frontier image
    out_img = out_dir / "efficient_frontier.png"
    plot_and_save_efficient_frontier(cov_matrix, mu, str(out_img))
    print(f"Saved efficient frontier plot to: {out_img}")
    # Compute key portfolios and save weights comparison
    key_results = compute_key_portfolios(mu, cov_matrix)
    weights_out = out_dir / "weights_comparison.png"
    plot_and_save_weights_comparison(key_results, str(weights_out))
    print(f"Saved weights comparison plot to: {weights_out}")