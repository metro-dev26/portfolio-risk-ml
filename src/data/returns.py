import yfinance as yf
import numpy as np
import pandas as pd

def download_prices(ticker, start, end):
    """Download stock prices for a given ticker."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True)
    data.columns = data.columns.get_level_values(0)
    return data["Close"]

def compute_returns(prices):
    """Compute simple and log returns from price series."""
    simple = prices.pct_change().dropna()
    log    = np.log(prices / prices.shift(1)).dropna()
    return simple, log

def return_stats(log_returns):
    """Print key statistics about return distribution."""
    print(f"Mean daily return : {log_returns.mean():.4f}")
    print(f"Std (volatility)  : {log_returns.std():.4f}")
    print(f"Min (worst day)   : {log_returns.min():.4f}")
    print(f"Max (best day)    : {log_returns.max():.4f}")
    print(f"Skewness          : {log_returns.skew():.4f}")
    print(f"Kurtosis          : {log_returns.kurt():.4f}")
