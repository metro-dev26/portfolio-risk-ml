import numpy as np
from scipy import stats

def gaussian_vs_reality(returns):
    """
    Compare Gaussian assumptions vs real return properties.
    Returns a dictionary of key statistics.
    """
    mu    = returns.mean()
    sigma = returns.std()
    
    worst_day      = returns.min()
    z_score        = (worst_day - mu) / sigma
    gaussian_prob  = stats.norm.cdf(worst_day, mu, sigma)
    years_expected = int(1 / gaussian_prob / 252)
    
    return {
        "mean"           : mu,
        "std"            : sigma,
        "kurtosis"       : returns.kurt(),
        "skewness"       : returns.skew(),
        "excess_kurtosis": returns.kurt() - 3,
        "worst_day"      : worst_day,
        "z_score"        : z_score,
        "gaussian_prob"  : gaussian_prob,
        "years_expected" : years_expected,
        "fat_tails"      : returns.kurt() > 3
    }

def print_gaussian_report(returns, ticker="Asset"):
    """Print a full Gaussian vs reality report."""
    stats_dict = gaussian_vs_reality(returns)
    print(f"Gaussian vs Reality Report — {ticker}")
    print("=" * 45)
    print(f"Kurtosis        : {stats_dict['kurtosis']:.2f} (Gaussian=3.0)")
    print(f"Excess kurtosis : {stats_dict['excess_kurtosis']:.2f}")
    print(f"Fat tails       : {stats_dict['fat_tails']}")
    print(f"Worst day       : {stats_dict['worst_day']*100:.2f}%")
    print(f"Gaussian says   : once every {stats_dict['years_expected']:,} years")
    print(f"Reality says    : happens every few years")
