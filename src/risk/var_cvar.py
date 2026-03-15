import numpy as np
from scipy import stats

def historical_var_cvar(returns, confidence=0.95):
    """
    Calculate VaR and CVaR using actual historical returns.
    No assumptions about distribution.
    """
    var  = -np.percentile(returns, (1 - confidence) * 100)
    cvar = -returns[returns <= -var].mean()
    return var, cvar

def parametric_var_cvar(returns, confidence=0.95):
    """
    Calculate VaR and CVaR assuming normal distribution.
    Underestimates risk due to fat tail ignorance.
    """
    mu    = returns.mean()
    sigma = returns.std()
    var   = -(mu + sigma * stats.norm.ppf(1 - confidence))
    cvar  = -(mu - sigma * stats.norm.pdf(
              stats.norm.ppf(1 - confidence)) / (1 - confidence))
    return var, cvar

def compare_var_methods(returns, confidence=0.95):
    """Compare historical vs parametric VaR and CVaR."""
    hist_var,  hist_cvar  = historical_var_cvar(returns, confidence)
    para_var,  para_cvar  = parametric_var_cvar(returns, confidence)
    underestimate_var  = hist_var  - para_var
    underestimate_cvar = hist_cvar - para_cvar
    return {
        "historical_var"       : hist_var,
        "historical_cvar"      : hist_cvar,
        "parametric_var"       : para_var,
        "parametric_cvar"      : para_cvar,
        "underestimate_var"    : underestimate_var,
        "underestimate_cvar"   : underestimate_cvar
    }
