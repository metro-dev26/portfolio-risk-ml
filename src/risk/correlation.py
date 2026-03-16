import numpy as np
import pandas as pd

def correlation_matrix(returns_df):
    """Compute correlation matrix for all assets."""
    return returns_df.corr()

def covariance_matrix(returns_df):
    """Compute covariance matrix for all assets."""
    return returns_df.cov()

def crisis_vs_normal_corr(returns_df, 
                           normal_start, normal_end,
                           crisis_start, crisis_end):
    """
    Compare correlations during normal vs crisis periods.
    Shows correlation breakdown during market stress.
    """
    normal = returns_df[normal_start:normal_end].corr()
    crisis = returns_df[crisis_start:crisis_end].corr()
    change = crisis - normal
    return {
        "normal_corr" : normal,
        "crisis_corr" : crisis,
        "change"      : change
    }

def rolling_correlation(series_a, series_b, window=30):
    """
    Compute rolling correlation between two assets.
    Shows how correlation changes over time.
    """
    return series_a.rolling(window).corr(series_b)

def portfolio_volatility(weights, cov_matrix):
    """
    Compute portfolio volatility given weights 
    and covariance matrix.
    Lower = better diversification.
    """
    weights = np.array(weights)
    port_var = weights.T @ cov_matrix @ weights
    return np.sqrt(port_var)
