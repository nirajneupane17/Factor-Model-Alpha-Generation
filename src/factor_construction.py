"""
factor_construction.py
======================
Fama-French factor construction and multi-factor portfolio builder.
Covers FF3, FF5, QMJ (Quality), BAB (Low-Vol), and UMD (Momentum).

Author : Niraj Neupane | github.com/nirajneupane17
Series : Quant Trading Projects — Project 3 of 20
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple


def factor_statistics(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute full statistics for each factor.

    Returns annualised return, vol, Sharpe, max DD, Calmar, win rate,
    and skewness — the standard academic factor tearsheet.
    """
    results = []
    for col in factor_returns.columns:
        r = factor_returns[col].dropna()
        ann_r = r.mean() * 252
        ann_v = r.std() * np.sqrt(252)
        sh = ann_r / ann_v if ann_v > 0 else 0
        cum = (1 + r).cumprod()
        dd = cum / cum.cummax() - 1
        calmar = ann_r / abs(dd.min()) if dd.min() != 0 else 0
        results.append({
            'factor':          col,
            'ann_return_pct':  round(ann_r * 100, 3),
            'ann_vol_pct':     round(ann_v * 100, 3),
            'sharpe_ratio':    round(sh, 4),
            'max_drawdown_pct':round(dd.min() * 100, 3),
            'calmar_ratio':    round(calmar, 4),
            'win_rate_pct':    round((r > 0).mean() * 100, 2),
            'skewness':        round(r.skew(), 4),
            'kurtosis':        round(r.kurtosis(), 4),
        })
    return pd.DataFrame(results).set_index('factor')


def ff3_portfolio(factor_returns: pd.DataFrame,
                   weights: Dict = None) -> pd.Series:
    """
    Fama-French 3-Factor Portfolio: MKT-RF + SMB + HML.
    Default equal-weight across the three factors.
    """
    cols = ['MKT_RF', 'SMB', 'HML']
    if weights is None:
        weights = {c: 1/3 for c in cols}
    r = factor_returns[cols]
    return sum(weights[c] * r[c] for c in cols)


def ff5_portfolio(factor_returns: pd.DataFrame,
                   weights: Dict = None) -> pd.Series:
    """
    Fama-French 5-Factor Portfolio: MKT-RF + SMB + HML + RMW + CMA.
    Default equal-weight across the five factors.
    """
    cols = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
    if weights is None:
        weights = {c: 0.20 for c in cols}
    r = factor_returns[cols]
    return sum(weights[c] * r[c] for c in cols)


def multifactor_portfolio(factor_returns: pd.DataFrame,
                           factors: List[str] = None,
                           weights: Dict = None) -> pd.Series:
    """
    Equal-weight multi-factor portfolio across any set of factors.
    """
    if factors is None:
        factors = list(factor_returns.columns)
    if weights is None:
        weights = {f: 1 / len(factors) for f in factors}
    return sum(weights[f] * factor_returns[f] for f in factors)


def factor_correlation_matrix(factor_returns: pd.DataFrame,
                               window: int = None) -> pd.DataFrame:
    """
    Full or rolling correlation matrix across factors.
    Low factor correlation = high diversification value.
    """
    if window:
        return factor_returns.rolling(window).corr()
    return factor_returns.corr()


def factor_decade_analysis(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Annualised factor returns by decade.
    Reveals factor cyclicality: value worked in 1990s, collapsed in 2010s.
    Growth/momentum dominated 2010s. Quality was consistent.
    """
    decades = [('1990s', '1990-01-01', '1999-12-31'),
               ('2000s', '2000-01-01', '2009-12-31'),
               ('2010s', '2010-01-01', '2019-12-31'),
               ('2020s', '2020-01-01', '2025-06-30')]
    results = {}
    for name, start, end in decades:
        r = factor_returns.loc[start:end]
        results[name] = (r.mean() * 252 * 100).round(2)
    return pd.DataFrame(results)


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    f = pd.read_csv('/home/claude/FACTOR/data/factor_returns.csv',
                     index_col='Date', parse_dates=True)
    stats = factor_statistics(f[['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']])
    print(stats[['ann_return_pct', 'sharpe_ratio', 'max_drawdown_pct']])
    print("\nfactor_construction.py OK")
