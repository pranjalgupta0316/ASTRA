"""Causal target-weight construction at decision dates.

At each decision date t: eligible assets = those with >= `min_history` valid daily returns up to and
including t (no backfilling: an asset enters only once it has the history). Covariance and means use
the trailing `window` complete rows ending at t. Nothing after t is read; targets are filled at t+1 open
by `nebula.portfolio.backtest`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from nebula.portfolio.covariance import CovEstimate, CovMethod, estimate_covariance, trailing_window
from nebula.portfolio.optimizers import (
    equal_weight,
    max_sharpe,
    mean_variance,
    min_variance,
    risk_parity,
    shrunk_mean,
    vol_target,
)

WeightFn = Callable[[pd.DataFrame, CovEstimate, float], tuple[np.ndarray, dict[str, Any]]]
"""(trailing return window, covariance estimate, per-asset cap) -> (weights, extra info)."""


def ew_fn(win: pd.DataFrame, est: CovEstimate, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
    return equal_weight(len(est.assets), cap), {}


def minvar_fn(win: pd.DataFrame, est: CovEstimate, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
    return min_variance(est.cov, cap), {}


def rp_fn(win: pd.DataFrame, est: CovEstimate, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
    return risk_parity(est.cov, caps=cap), {}


def make_mv_fn(risk_aversion: float = 5.0, shrink: float = 0.5) -> WeightFn:
    def fn(win: pd.DataFrame, est: CovEstimate, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
        return mean_variance(shrunk_mean(win.to_numpy(), shrink), est.cov, risk_aversion, cap), {}

    return fn


def make_max_sharpe_fn(shrink: float = 0.5) -> WeightFn:
    def fn(win: pd.DataFrame, est: CovEstimate, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
        w, fb = max_sharpe(shrunk_mean(win.to_numpy(), shrink), est.cov, cap)
        return w, {"fallback_minvar": fb}

    return fn


def make_vol_target_fn(base: WeightFn, target_vol: float, periods_per_year: int) -> WeightFn:
    def fn(win: pd.DataFrame, est: CovEstimate, cap: float) -> tuple[np.ndarray, dict[str, Any]]:
        w, info = base(win, est, cap)
        ws, scale = vol_target(w, est.cov, target_vol, periods_per_year, max_leverage=1.0)
        return ws, {**info, "vt_scale": scale}

    return fn


def eligible_assets(returns: pd.DataFrame, t: pd.Timestamp, min_history: int) -> list[str]:
    hist = returns.loc[returns.index <= pd.Timestamp(t)]
    counts = hist.notna().sum()
    return [c for c in returns.columns if counts[c] >= min_history]


def compute_targets(
    returns: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    weight_fn: WeightFn,
    cap: float,
    periods_per_year: int,
    window: int = 252,
    method: CovMethod = "ledoit_wolf",
    min_history: int = 252,
    min_assets: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Targets (index = decision dates with >= min_assets eligible) and per-decision diagnostics:
    ex-ante forecast vol (annualized, sqrt(w'Sw * P)), condition number, shrinkage, n_assets.

    The per-asset cap is raised to 1/n if n * cap < 1 (cannot be fully invested otherwise); this is
    recorded in the diagnostics as `cap_used`.
    """
    rows, diags = {}, []
    for t in decision_dates:
        elig = eligible_assets(returns, t, min_history)
        if len(elig) < min_assets:
            continue
        r = returns[elig]
        est = estimate_covariance(r, t, window=window, method=method, min_obs=min(window, min_history))
        win = trailing_window(r, t, window)
        cap_used = max(cap, 1.0 / len(elig))
        w, info = weight_fn(win, est, cap_used)
        row = pd.Series(0.0, index=returns.columns)
        row[elig] = w
        rows[pd.Timestamp(t)] = row
        fvar = float(w @ est.cov @ w)
        diags.append(
            {
                "decision_date": pd.Timestamp(t),
                "n_assets": len(elig),
                "assets": ",".join(elig),
                "forecast_vol_ann": float(np.sqrt(max(fvar, 0.0) * periods_per_year)),
                "condition_number": est.condition_number,
                "shrinkage": est.shrinkage,
                "psd_repaired": est.psd_repaired,
                "cap_used": cap_used,
                "invested": float(np.sum(w)),
                **info,
            }
        )
    targets = pd.DataFrame(rows).T if rows else pd.DataFrame(columns=returns.columns)
    diag = pd.DataFrame(diags).set_index("decision_date") if diags else pd.DataFrame()
    return targets, diag
