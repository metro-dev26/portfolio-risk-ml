import numpy as np
import pandas as pd

def rolling_volatility(returns, window=30, annualize=True):
    """
    Calculate rolling volatility over a given window.
    Annualized by default (multiply by sqrt(252 trading days)).
    """
    rolling_vol = returns.rolling(window=window).std()
    if annualize:
        rolling_vol = rolling_vol * np.sqrt(252)
    return rolling_vol

def volatility_stats(returns):
    """Print volatility statistics for a return series."""
    daily_vol = returns.std()
    annual_vol = daily_vol * np.sqrt(252)
    print(f"Daily  volatility : {daily_vol:.4f}  ({daily_vol*100:.2f}%)")
    print(f"Annual volatility : {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print(f"Volatility clustering check:")
    print(f"  Autocorrelation of |returns|: {abs(returns).autocorr():.4f}")
    print(f"  (above 0.1 = clustering confirmed)")

def compare_volatility(returns_dict, window=30):
    """Compare rolling volatility across multiple assets."""
    vol_df = pd.DataFrame()
    for ticker, returns in returns_dict.items():
        vol_df[ticker] = rolling_volatility(returns, window)
    return vol_df
