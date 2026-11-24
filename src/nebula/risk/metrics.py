"""Performance and risk metrics. `periods_per_year` is always explicit (252 equities, 365 crypto daily)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nebula.stats.inference import sharpe


def simple_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    eq = equity.dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return float("nan")
    years = (len(eq) - 1) / periods_per_year
    if eq.iloc[-1] <= 0:
        return -1.0
    return float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)


def annual_vol(returns: pd.Series, periods_per_year: int) -> float:
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int, mar: float = 0.0) -> float:
    r = returns.dropna()
    downside = np.minimum(r - mar, 0.0)
    dd = math.sqrt(float((downside**2).mean()))
    if dd == 0:
        return float("nan")
    return float((r.mean() - mar) / dd * math.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def calmar(equity: pd.Series, periods_per_year: int) -> float:
    mdd = max_drawdown(equity)
    return float(cagr(equity, periods_per_year) / abs(mdd)) if mdd < 0 else float("nan")


def var_historical(returns: pd.Series, level: float = 0.95) -> float:
    """One-period historical VaR as a positive loss number."""
    return float(-np.quantile(returns.dropna(), 1 - level))


def expected_shortfall(returns: pd.Series, level: float = 0.95) -> float:
    r = returns.dropna().to_numpy()
    cutoff = np.quantile(r, 1 - level)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else float("nan")


def summarize(equity: pd.Series, periods_per_year: int) -> dict[str, float]:
    r = simple_returns(equity)
    return {
        "cagr": cagr(equity, periods_per_year),
        "ann_vol": annual_vol(r, periods_per_year),
        "sharpe": sharpe(r, periods_per_year),
        "sortino": sortino(r, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar(equity, periods_per_year),
        "var_95": var_historical(r, 0.95),
        "es_95": expected_shortfall(r, 0.95),
        "n_periods": len(r),
    }
