"""
alpha_decomposition.py
======================
Alpha decomposition: separates true alpha from factor (beta) exposures.
Implements FF3, FF5, and 8-factor OLS regression with full attribution.

Author : Niraj Neupane | github.com/nirajneupane17
Series : Quant Trading Projects — Project 3 of 20
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, List


def factor_regression(portfolio_returns: pd.Series,
                        factor_returns: pd.DataFrame,
                        factors: List[str]) -> Dict:
    """
    OLS factor regression.

    Decomposes portfolio return into:
      Return = alpha + sum(beta_i * Factor_i) + epsilon

    Parameters
    ----------
    portfolio_returns : monthly portfolio return series
    factor_returns    : factor return DataFrame
    factors           : list of factor columns to use

    Returns
    -------
    dict: alpha, betas, R², t-stats, systematic/idiosyncratic split
    """
    common = portfolio_returns.index.intersection(factor_returns.index)
    y = portfolio_returns.loc[common].values
    X = factor_returns.loc[common, factors].values

    lr = LinearRegression()
    lr.fit(X, y)
    residuals = y - lr.predict(X)
    r2 = lr.score(X, y)

    # t-statistics
    n, k = X.shape
    mse = np.sum(residuals**2) / (n - k - 1)
    var_b = mse * np.linalg.inv(X.T @ X).diagonal()
    t_stats = lr.coef_ / np.sqrt(var_b)

    # Annualised attribution
    factor_contribs = {factors[i]: lr.coef_[i] *
                       factor_returns.loc[common, factors[i]].mean() * 12 * 100
                       for i in range(k)}

    return {
        'alpha_monthly':       round(lr.intercept_, 6),
        'alpha_annualised_pct':round(lr.intercept_ * 12 * 100, 4),
        'betas':               {factors[i]: round(lr.coef_[i], 4) for i in range(k)},
        't_stats':             {factors[i]: round(t_stats[i], 3) for i in range(k)},
        'r_squared':           round(r2, 4),
        'factor_contributions_pct': {k: round(v, 4) for k, v in factor_contribs.items()},
        'systematic_pct':      round(r2 * 100, 2),
        'idiosyncratic_pct':   round((1 - r2) * 100, 2),
        'tracking_error_pct':  round(residuals.std() * np.sqrt(12) * 100, 4),
    }


def information_coefficient(signals: pd.Series,
                              forward_returns: pd.Series,
                              lag: int = 1) -> float:
    """
    Information Coefficient (IC) — correlation between signal and forward return.
    IC > 0.05 is considered meaningful in practice.
    IC > 0.10 is strong.
    """
    s = signals.shift(lag).dropna()
    r = forward_returns.reindex(s.index).dropna()
    common = s.index.intersection(r.index)
    if len(common) < 10:
        return np.nan
    return float(np.corrcoef(s.loc[common], r.loc[common])[0, 1])


def rolling_alpha(portfolio_returns: pd.Series,
                   factor_returns: pd.DataFrame,
                   factors: List[str],
                   window: int = 36) -> pd.Series:
    """
    Rolling alpha — detects alpha decay over time.
    If alpha shrinks post-publication, factor is crowded.
    """
    common = portfolio_returns.index.intersection(factor_returns.index)
    y_full = portfolio_returns.loc[common]
    X_full = factor_returns.loc[common, factors]
    alphas = []
    for i in range(window, len(common)):
        y = y_full.iloc[i-window:i].values
        X = X_full.iloc[i-window:i].values
        lr = LinearRegression()
        lr.fit(X, y)
        alphas.append(lr.intercept_ * 12 * 100)
    return pd.Series(alphas, index=common[window:], name='rolling_alpha_pct')


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    f = pd.read_csv('/home/claude/FACTOR/data/monthly_factors.csv',
                     index_col='Date', parse_dates=True)
    r = pd.read_csv('/home/claude/FACTOR/data/stock_returns.csv',
                     index_col='Date', parse_dates=True)
    port = r.mean(axis=1).resample('ME').sum()
    res = factor_regression(port, f, ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA'])
    print(f"Alpha: {res['alpha_annualised_pct']:.2f}%  R²: {res['r_squared']:.3f}")
    print(f"Betas: {res['betas']}")
    print("alpha_decomposition.py OK")
