"""Covariance estimation on trailing windows (causal).

Timing convention: the estimate for decision date t uses returns up to and including t's close and is
used for orders filled at the open of t+1. Nothing after t is read (tested by truncation invariance).

Returns are simple daily returns (portfolio variance is exactly w' S w for simple returns). Units are
per period; annualize with `periods_per_year` (252 equities, 365 crypto) only when reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

CovMethod = Literal["sample", "ledoit_wolf"]


@dataclass
class CovEstimate:
    cov: np.ndarray  # per-period covariance (PSD after repair)
    assets: list[str]
    method: str
    n_obs: int
    end: pd.Timestamp
    condition_number: float
    psd_repaired: bool
    shrinkage: float | None = None


def sample_cov(returns: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance (ddof=1) of a T x N array."""
    x = np.asarray(returns, dtype=float)
    return np.atleast_2d(np.cov(x, rowvar=False, ddof=1))


def ledoit_wolf_cov(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward scaled identity; returns (cov, shrinkage intensity)."""
    lw = LedoitWolf(assume_centered=False).fit(np.asarray(returns, dtype=float))
    return np.asarray(lw.covariance_), float(lw.shrinkage_)


def is_psd(cov: np.ndarray, tol: float = 1e-12) -> bool:
    s = np.asarray(cov, dtype=float)
    if not np.allclose(s, s.T, atol=1e-12):
        return False
    return bool(np.linalg.eigvalsh(s).min() >= -tol * max(1.0, np.abs(s).max()))


def nearest_psd(cov: np.ndarray, min_eig: float = 0.0) -> np.ndarray:
    """Nearest symmetric PSD matrix in Frobenius norm (eigenvalue clipping, Higham 1988 for the
    symmetric case). `min_eig` > 0 forces strict positive definiteness."""
    s = np.asarray(cov, dtype=float)
    s = (s + s.T) / 2
    vals, vecs = np.linalg.eigh(s)
    vals = np.clip(vals, min_eig, None)
    out = (vecs * vals) @ vecs.T
    return (out + out.T) / 2


def condition_number(cov: np.ndarray) -> float:
    vals = np.linalg.eigvalsh((np.asarray(cov) + np.asarray(cov).T) / 2)
    if vals.min() <= 0:
        return float("inf")
    return float(vals.max() / vals.min())


def trailing_window(returns: pd.DataFrame, t: pd.Timestamp, window: int) -> pd.DataFrame:
    """Last `window` complete rows (no NaN) with index <= t. Strictly no rows after t."""
    hist = returns.loc[returns.index <= pd.Timestamp(t)].dropna(how="any")
    return hist.iloc[-window:]


def estimate_covariance(
    returns: pd.DataFrame,
    t: pd.Timestamp,
    window: int = 252,
    method: CovMethod = "ledoit_wolf",
    min_obs: int | None = None,
) -> CovEstimate:
    """Covariance of the columns of `returns` on the trailing window ending at t (inclusive).

    Raises ValueError if fewer than `min_obs` (default = window) complete rows exist or if the matrix
    is still not PSD after nearest-PSD repair (the optimizer must then refuse to run).
    """
    win = trailing_window(returns, t, window)
    need = window if min_obs is None else min_obs
    if len(win) < need:
        raise ValueError(f"only {len(win)} complete return rows up to {t}, need {need}")
    x = win.to_numpy(dtype=float)
    shrink: float | None = None
    if method == "ledoit_wolf":
        cov, shrink = ledoit_wolf_cov(x)
    elif method == "sample":
        cov = sample_cov(x)
    else:
        raise ValueError(f"unknown covariance method {method!r}")
    repaired = False
    if not is_psd(cov):
        cov = nearest_psd(cov)
        repaired = True
        if not is_psd(cov):
            raise ValueError("covariance not PSD after nearest-PSD repair; refusing to optimize")
    return CovEstimate(
        cov=cov,
        assets=list(win.columns),
        method=method,
        n_obs=len(win),
        end=pd.Timestamp(win.index[-1]),
        condition_number=condition_number(cov),
        psd_repaired=repaired,
        shrinkage=shrink,
    )
