"""Inference helpers for performance statistics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def sharpe(returns: pd.Series | np.ndarray, periods_per_year: int) -> float:
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * math.sqrt(periods_per_year))


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: int, seed: int = 0) -> np.ndarray:
    """(n_boot, n) index matrix for the Politis-Romano stationary bootstrap (vectorized over resamples)."""
    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(n, size=n_boot)
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    starts = rng.integers(n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def sharpe_difference_bootstrap(
    r1: pd.Series,
    r2: pd.Series,
    periods_per_year: int,
    n_boot: int = 2000,
    mean_block: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Paired stationary-bootstrap inference for SR(r1) - SR(r2) on aligned return series."""
    df = pd.concat([r1, r2], axis=1).dropna()
    a, b = df.iloc[:, 0].to_numpy(dtype=float), df.iloc[:, 1].to_numpy(dtype=float)
    k = math.sqrt(periods_per_year)

    def sr(x: np.ndarray) -> np.ndarray:
        sd = x.std(axis=-1, ddof=1)
        return np.where(sd > 0, x.mean(axis=-1) / np.where(sd > 0, sd, 1.0) * k, np.nan)

    point = float(sr(a) - sr(b))
    idx = stationary_bootstrap_indices(len(a), n_boot, mean_block, seed)
    boot = sr(a[idx]) - sr(b[idx])
    boot = boot[np.isfinite(boot)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = 2 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {
        "delta_sr": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": min(1.0, p),
        "n": len(a),
    }


def stationary_bootstrap_sharpe_ci(
    returns: pd.Series,
    periods_per_year: int,
    n_boot: int = 2000,
    mean_block: int = 20,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Politis-Romano stationary bootstrap CI for the annualized Sharpe ratio."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(x)
    if n < 2 * mean_block:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=np.int64)
        idx[0] = rng.integers(n)
        new_block = rng.random(n) < p
        starts = rng.integers(n, size=n)
        for t in range(1, n):
            idx[t] = starts[t] if new_block[t] else (idx[t - 1] + 1) % n
        s = x[idx]
        sd = s.std(ddof=1)
        out[b] = s.mean() / sd * math.sqrt(periods_per_year) if sd > 0 else np.nan
    lo, hi = np.nanquantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & Lopez de Prado 2012) on per-period Sharpe; benchmark also per-period."""
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return float("nan")
    sr = x.mean() / x.std(ddof=1)
    g3 = stats.skew(x)
    g4 = stats.kurtosis(x, fisher=False)
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, trial_sr_variance: float | None = None) -> float:
    """DSR (Bailey & Lopez de Prado 2014): PSR against the expected max Sharpe of `n_trials` null trials.

    trial_sr_variance is the cross-trial variance of per-period Sharpe ratios; if unknown we use the
    iid null approximation 1/(n-1).
    """
    x = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(x)
    if n_trials < 1 or n < 3:
        return float("nan")
    var = trial_sr_variance if trial_sr_variance is not None else 1.0 / (n - 1)
    if n_trials == 1:
        sr0 = 0.0
    else:
        euler = 0.5772156649
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(var) * ((1 - euler) * z1 + euler * z2)
    return probabilistic_sharpe_ratio(pd.Series(x), sr_benchmark=sr0)


def newey_west_tstat(x: pd.Series, lags: int | None = None) -> float:
    """t-statistic of the mean with Newey-West HAC standard error."""
    v = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(v)
    if n < 3:
        return float("nan")
    lags = lags if lags is not None else math.floor(4 * (n / 100) ** (2 / 9))
    e = v - v.mean()
    gamma0 = e @ e / n
    s = gamma0
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (e[k:] @ e[:-k]) / n
    se = math.sqrt(max(s, 1e-300) / n)
    return float(v.mean() / se)
