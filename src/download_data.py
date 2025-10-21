"""
    File to download monthly returns data for specified tickers via connecting to Yahoo Finance using yfinance library

"""

import yfinance as yf
import pandas as pd
import os
from pathlib import Path

def get_monthly_returns(tickers, start_date, end_date):
    """Download data for each ticker individually"""
    all_returns = {}
    
    # downloading data individually through looping through tickers
    for ticker in tickers:
        # catching API errors
        try:
            print(f"Downloading data for {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval='1mo')
            
            if not data.empty:
                # using closing prices
                prices = data['Close']

                # calculating monthly returns
                returns = prices.pct_change().dropna()
                all_returns[ticker] = returns
                
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
    
    monthly_returns = pd.DataFrame(all_returns)
    return monthly_returns



def main():
    # company tickers of chosen assets for portfolio
    company_tickers = {
        'Google': 'GOOGL',
        'Disney': 'DIS',
        'McDonalds': 'MCD',
        'Tesla': 'TSLA',
        'CocaCola': 'KO',
        'Walmart': 'WMT',
        'ExxonMobil': 'XOM',
        'JPMorganChase': 'JPM',
        'Johnson&Johnson': 'JNJ',
        'Pfizer': 'PFE',
        'Boeing': 'BA',
        'Apple': 'AAPL',
        'Nvidia': 'NVDA',
        'DuPont': 'DD',
        'AmericanTower': 'AMT',
        'DukeEnergy': 'DUK',
        'Equinix': 'EQIX',
        'Lululemon': 'LULU'
    }

    # defining time period 
    start_date = '2015-08-01'
    end_date = '2025-08-29'

    monthly_returns = get_monthly_returns(list(company_tickers.values()), start_date, end_date)
    
    # saving data to CSV
    if monthly_returns is not None and not monthly_returns.empty:
        # Save to shared src/output (one level up from package)
        output_dir = Path(__file__).resolve().parents[1] / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "monthly_returns.csv"
        monthly_returns.to_csv(str(output_file))
        print(f"Monthly returns data saved to '{output_file}'.")
        print(f"Data covers period: {monthly_returns.index[0]} to {monthly_returns.index[-1]}")
        print(f"Number of assets: {len(monthly_returns.columns)}")
        print(f"Available tickers: {list(monthly_returns.columns)}")
    else:
        print("Failed to retrieve monthly returns data.")

if __name__ == "__main__":
    main()