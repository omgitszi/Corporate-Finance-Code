import pandas as pd 
import numpy as np 
from pathlib import Path
from pypfopt import (
    EfficientFrontier,
    risk_models, 
    expected_returns,
    objective_functions,
)

def cov_calc(summary_csv_path = "summary.csv"):
    p = Path(summary_csv_path)
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    returns = df.select_dtypes(include=[np.number]).copy()
    cov_matrix = risk_models(returns)
    return cov_matrix

if __name__ == "__main__":
    cov_matrix = cov_calc()
    print(cov_matrix)