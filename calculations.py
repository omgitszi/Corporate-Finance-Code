"""
Calculations for annualizing asset returns and volatilities.
Visualisation of asset summary statistics to see the risk-return tradeoff for each asset.
This class is based of provided lab 2 material.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_segment,
    geom_text,
    scale_x_continuous,
    scale_y_continuous,
    labs,
    theme_minimal,
    theme,
)
from mizani.formatters import percent_format

ANNUALISATION_FACTOR = 12


def calculate_summary_stats(monthly_csv_path="monthly_returns.csv"):
    """
    Calculate summary statistics for asset returns using previously downloaded monthly returns data.
    """
    p = Path(monthly_csv_path)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    returns = df.select_dtypes(include=[np.number]).copy()
    if returns.empty:
        raise ValueError("No numeric return columns found in CSV")
    mean_monthly = returns.mean(axis=0)
    std_monthly = returns.std(axis=0, ddof=1)

    summary_stats = pd.DataFrame(
        {
            "symbol": mean_monthly.index,
            "mean": mean_monthly.values,
            "std": std_monthly.values,
        }
    )

    return summary_stats.reset_index(drop=True)


def build_asset_summary(summary_stats, annualisation_factor=ANNUALISATION_FACTOR):
    """
    Building asset summary dataframe with annualized expected returns and volatilities.
    """
    asset_summary = pd.DataFrame(
        {
            "symbol": summary_stats["symbol"], 
            "mu": summary_stats["mean"] * annualisation_factor,  # Annualized expected return
            "sigma": summary_stats["std"] * np.sqrt(annualisation_factor),  # Annualized volatility
        }
    ).reset_index(drop=True)

    return asset_summary


def build_assets_figure(asset_summary):
    """
    Building a scatter plot with annualized volatility (sigma) on x-axis and annualized expected return (mu) on y-axis.
    Each point represents an asset, labeled with its ticker symbol.
    Visually see the risk-return tradeoff.
    """
    assets_figure = (
        ggplot(asset_summary, aes(x="sigma", y="mu", label="symbol"))
        + geom_point(size=3, alpha=0.7)
        + geom_text(adjust_text={"arrowprops": {"arrowstyle": "-"}})
        + scale_x_continuous(labels=percent_format())
        + scale_y_continuous(labels=percent_format())
        + labs(
            x="Volatility (annualised)",
            y="Expected return (annualised)",
            title="Expected returns and volatilities of portfolio constituents",
            subtitle="Based on historical monthly returns",
        )
        + theme_minimal()
        + theme(figure_size=(10, 7))
    )

    return assets_figure

if __name__ == "__main__":
    stats = calculate_summary_stats("monthly_returns.csv")
    asset_summary = build_asset_summary(stats, annualisation_factor=ANNUALISATION_FACTOR)
    print(asset_summary)
    fig = build_assets_figure(asset_summary)
    out_path = Path(__file__).resolve().parent / "assets_figure.png"
    fig.save(str(out_path), dpi=150)
    # saving graph figure to png file
    print(f"Saved figure to {out_path}")