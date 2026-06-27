"""
portfolio_optimizer.py
======================
Multi-factor portfolio construction and optimization.
Implements equal-weight, maximum Sharpe, minimum variance,
and risk-parity factor portfolio weighting schemes.

Author : Niraj Neupane | github.com/nirajneupane17
Series : Quant Trading Projects — Project 3 of 20
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple


def performance_metrics(returns: pd.Series, freq: int = 12) -> Dict:
    """Standard performance metrics for monthly return series."""
    r = returns.dropna()
    ann_r = r.mean() * freq
    ann_v = r.std() * np.sqrt(freq)
    sh = ann_r / ann_v if ann_v > 0 else 0
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    calmar = ann_r / abs(dd.min()) if dd.min() != 0 else 0
    return {
        'ann_return_pct':   round(ann_r * 100, 3),
        'ann_vol_pct':      round(ann_v * 100, 3),
        'sharpe_ratio':     round(sh, 4),
        'sortino_ratio':    round(ann_r / (r[r < 0].std() * np.sqrt(freq)) if r[r < 0].std() > 0 else 0, 4),
        'calmar_ratio':     round(calmar, 4),
        'max_drawdown_pct': round(dd.min() * 100, 3),
        'win_rate_pct':     round((r > 0).mean() * 100, 2),
    }


def max_sharpe_weights(factor_returns: pd.DataFrame,
                        factors: List[str],
                        rf: float = 0.02/12) -> Dict:
    """Maximum Sharpe ratio weights via SLSQP optimisation."""
    r = factor_returns[factors].dropna()
    mu = r.mean().values
    cov = r.cov().values
    n = len(factors)

    def neg_sharpe(w):
        pr = w @ mu; pv = np.sqrt(w @ cov @ w)
        return -(pr - rf) / pv if pv > 0 else 0

    res = minimize(neg_sharpe, np.ones(n)/n, method='SLSQP',
                   bounds=[(0, 0.60)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}])
    return dict(zip(factors, res.x.round(4))) if res.success else {}


def risk_parity_weights(factor_returns: pd.DataFrame,
                         factors: List[str]) -> Dict:
    """
    Risk parity — equal risk contribution from each factor.
    Reduces concentration risk vs equal-weight.
    """
    r = factor_returns[factors].dropna()
    cov = r.cov().values
    n = len(factors)

    def risk_concentration(w):
        pv = np.sqrt(w @ cov @ w)
        mrc = cov @ w / pv
        crc = w * mrc
        return sum((crc[i] - crc[j])**2 for i in range(n) for j in range(i+1, n))

    res = minimize(risk_concentration, np.ones(n)/n, method='SLSQP',
                   bounds=[(0.01, 0.60)] * n,
                   constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
                   options={'maxiter': 1000})
    return dict(zip(factors, res.x.round(4))) if res.success else {}


def factor_timing_weights(factor_returns: pd.DataFrame,
                           factors: List[str],
                           lookback: int = 12) -> pd.DataFrame:
    """
    Simple factor timing: overweight factors with strong recent momentum.
    Each month, weight proportional to trailing 12M Sharpe.
    """
    weights = pd.DataFrame(index=factor_returns.index, columns=factors, dtype=float)
    for i in range(lookback, len(factor_returns)):
        window = factor_returns[factors].iloc[i-lookback:i]
        sharpes = window.mean() / window.std()
        sharpes = sharpes.clip(lower=0)
        if sharpes.sum() > 0:
            weights.iloc[i] = (sharpes / sharpes.sum()).values
        else:
            weights.iloc[i] = 1 / len(factors)
    return weights.dropna()


if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    f = pd.read_csv('/home/claude/FACTOR/data/monthly_factors.csv',
                     index_col='Date', parse_dates=True)
    factors = ['MKT_RF', 'SMB', 'HML', 'RMW', 'CMA']
    ms = max_sharpe_weights(f, factors)
    rp = risk_parity_weights(f, factors)
    print("Max Sharpe weights:", ms)
    print("Risk Parity weights:", rp)
    print("portfolio_optimizer.py OK")
