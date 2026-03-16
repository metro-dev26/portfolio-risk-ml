import numpy as np
import pandas as pd

def monte_carlo_simulation(returns, investment=1_000_000,
                           n_simulations=10_000, n_days=252,
                           seed=42):
    """
    Run Monte Carlo simulation for portfolio.
    
    Parameters:
        returns       : historical portfolio returns
        investment    : starting portfolio value
        n_simulations : number of paths to simulate
        n_days        : number of trading days (252 = 1 year)
        seed          : random seed for reproducibility
    
    Returns:
        final_values  : array of final portfolio values
        cumulative    : full path matrix (n_days x n_simulations)
    """
    mu    = returns.mean()
    sigma = returns.std()
    
    np.random.seed(seed)
    random_shocks = np.random.normal(0, 1, (n_days, n_simulations))
    daily_returns = mu + sigma * random_shocks
    cumulative    = np.cumprod(1 + daily_returns, axis=0)
    final_values  = investment * cumulative[-1]
    
    return final_values, cumulative

def mc_risk_metrics(final_values, investment):
    """
    Compute VaR and CVaR from Monte Carlo final values.
    """
    mc_returns = (final_values - investment) / investment
    mc_returns = pd.Series(mc_returns)
    
    var_95  = -np.percentile(mc_returns, 5)
    cvar_95 = -mc_returns[mc_returns <= -var_95].mean()
    var_99  = -np.percentile(mc_returns, 1)
    cvar_99 = -mc_returns[mc_returns <= -var_99].mean()
    
    return {
        "var_95"  : var_95,
        "cvar_95" : cvar_95,
        "var_99"  : var_99,
        "cvar_99" : cvar_99,
        "worst_case"   : final_values.min(),
        "best_case"    : final_values.max(),
        "median"       : np.median(final_values)
    }
