"""Daily variance proxies, one-step-ahead variance forecasters, forecast losses and DM tests.

Conventions
- `r` is a pd.Series of decimal log returns indexed by session date; position t is "origin t".
- Every forecaster returns a Series indexed like `r` whose value at origin t is the forecast of the
  variance of r_{t+1}, using r_0..r_t only (verified by truncation-invariance tests).
- GARCH-family models are fit by `arch` on 100 * r (percent) and converted back to decimal variance.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from astra.features.technical import garman_klass_var, parkinson_var
from astra.stats.inference import newey_west_tstat

GarchKind = Literal["garch", "gjr"]

__all__ = [
    "GarchForecast",
    "diebold_mariano",
    "ewma_forecast",
    "garch_forecast",
    "garch_variance_path",
    "log_returns",
    "mse_loss",
    "qlike",
    "qlike_rank",
    "realized_variance_proxies",
    "rolling_mean_forecast",
]


# ----------------------------------------------------------------------------------------------
# Realized-variance proxies (per-period variance, decimal units)
# ----------------------------------------------------------------------------------------------
def log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns (first value NaN)."""
    c = pd.Series(close, dtype=float)
    return np.log(c / c.shift(1))


def realized_variance_proxies(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Per-session variance proxies from columns open/high/low/close.

    rv_cc  = r_t^2 with r_t = ln(C_t / C_{t-1})                 (close-to-close)
    rv_pk  = ln(H/L)^2 / (4 ln 2)                                (Parkinson 1980)
    rv_gk  = 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2              (Garman-Klass 1980)
    The range estimators measure open-to-close variance only (no overnight gap).
    """
    o, h, lo, c = (ohlc[k].astype(float) for k in ("open", "high", "low", "close"))
    out = pd.DataFrame(index=ohlc.index)
    out["rv_cc"] = log_returns(c) ** 2
    out["rv_pk"] = parkinson_var(h, lo)
    out["rv_gk"] = garman_klass_var(o, h, lo, c)
    return out


# ----------------------------------------------------------------------------------------------
# Simple forecasters
# ----------------------------------------------------------------------------------------------
def rolling_mean_forecast(r: pd.Series, window: int) -> pd.Series:
    """h_{t+1|t} = mean(r_{t-window+1..t}^2)."""
    return (pd.Series(r, dtype=float) ** 2).rolling(window, min_periods=window).mean()


def ewma_forecast(r: pd.Series, lam: float = 0.94, min_periods: int = 20) -> pd.Series:
    """RiskMetrics: h_{t+1|t} = lam h_{t|t-1} + (1 - lam) r_t^2, seeded with r_0^2 (causal).

    Origins before `min_periods` observations are masked (seed not yet washed out).
    """
    r2 = pd.Series(r, dtype=float) ** 2
    return r2.ewm(alpha=1.0 - lam, adjust=False, min_periods=min_periods).mean()


# ----------------------------------------------------------------------------------------------
# GARCH family with periodic refits
# ----------------------------------------------------------------------------------------------
def _arch_backcast(e: np.ndarray) -> float:
    """Identical to arch's GARCH backcast: 0.94-weighted mean of the first <=75 squared residuals."""
    tau = min(75, len(e))
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(e[:tau] ** 2 * w))


def garch_variance_path(e: np.ndarray, params: dict[str, float], backcast: float | None = None) -> np.ndarray:
    """Conditional variances for a (GJR-)GARCH(1,1) with fixed params, arch-compatible.

    Returns an array of length len(e) + 1: element i (i < n) is sigma2_i (variance of e_i given
    e_0..e_{i-1}); the last element is the one-step forecast for e_n. Units follow `e`.
    """
    e = np.asarray(e, dtype=float)
    n = len(e)
    om = params["omega"]
    a = params["alpha[1]"]
    g = params.get("gamma[1]", 0.0)
    b = params["beta[1]"]
    bc = _arch_backcast(e) if backcast is None else backcast
    s2 = np.empty(n + 1)
    s2[0] = om + (a + 0.5 * g + b) * bc
    e2 = e * e
    neg = e < 0
    for i in range(1, n + 1):
        s2[i] = om + (a + g * neg[i - 1]) * e2[i - 1] + b * s2[i - 1]
    return s2


@dataclass
class GarchForecast:
    forecast: pd.Series  # decimal variance forecast at each origin (NaN before first refit)
    refits: pd.DataFrame  # one row per refit: origin, n_obs, params, loglik, convergence
    meta: dict[str, Any] = field(default_factory=dict)


def _fit(
    y: np.ndarray, kind: GarchKind, dist: str, start: np.ndarray | None
) -> tuple[dict[str, float], float, int]:
    am = arch_model(
        y, mean="Zero", vol="GARCH", p=1, o=1 if kind == "gjr" else 0, q=1, dist=dist, rescale=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", starting_values=start, options={"maxiter": 500})
    return {k: float(v) for k, v in res.params.items()}, float(res.loglikelihood), int(res.convergence_flag)


def garch_forecast(
    r: pd.Series,
    kind: GarchKind = "gjr",
    refit_every: int = 21,
    min_train: int = 504,
    dist: str = "t",
) -> GarchForecast:
    """One-step variance forecasts with expanding-window refits every `refit_every` origins.

    Refit origins are anchored at the START of the sample: o_k = first + k * refit_every, where
    `first` is the position of the `min_train`-th non-NaN return. The model used at origin t is the
    one fit on returns at positions <= o_k for the latest o_k <= t; the variance recursion is then run
    forward with frozen parameters on data <= t. Appending later data therefore never changes an
    earlier forecast.
    """
    rs = pd.Series(r, dtype=float)
    valid = rs.notna().to_numpy()
    pos = np.flatnonzero(valid)
    if len(pos) and not np.all(valid[pos[0] :]):
        raise ValueError("returns must not contain interior NaNs")
    out = pd.Series(np.nan, index=rs.index, name=f"{kind}_fc")
    if len(pos) < min_train:
        return GarchForecast(out, pd.DataFrame(), {"kind": kind, "n_refits": 0})
    offset = int(pos[0])
    y_all = 100.0 * rs.to_numpy()[offset:]  # percent, contiguous
    n = len(y_all)
    first = min_train - 1  # position (in y_all) of the first refit origin
    rows: list[dict[str, Any]] = []
    start: np.ndarray | None = None
    params: dict[str, float] | None = None
    fc = np.full(n, np.nan)
    for o in range(first, n, refit_every):
        y_train = y_all[: o + 1]
        try:
            p, ll, flag = _fit(y_train, kind, dist, start)
            ok = flag == 0 and np.isfinite(ll)
        except (ValueError, np.linalg.LinAlgError, RuntimeError):
            p, ll, flag, ok = {}, float("nan"), -1, False
        if ok or params is None:
            if not p:  # first fit failed outright: cannot forecast this block
                rows.append(
                    {"origin_pos": o + offset, "n_obs": o + 1, "loglik": ll, "flag": flag, "used": False}
                )
                continue
            params = p
            start = np.array(list(p.values()))
        rows.append(
            {
                "origin_pos": o + offset,
                "origin": rs.index[o + offset],
                "n_obs": o + 1,
                "loglik": ll,
                "flag": flag,
                "used": bool(ok),
                **{f"p_{k}": v for k, v in params.items()},
            }
        )
        end = min(o + refit_every, n)  # origins o .. end-1 use these params
        bc = _arch_backcast(y_train)  # depends on the first 75 obs only
        s2 = garch_variance_path(y_all[:end], params, backcast=bc)
        fc[o:end] = s2[o + 1 : end + 1]
    out.iloc[offset:] = fc / 1e4
    refits = pd.DataFrame(rows)
    meta = {
        "kind": kind,
        "dist": dist,
        "refit_every": refit_every,
        "min_train": min_train,
        "n_refits": len(rows),
        "n_failed_or_nonconverged": int((~refits["used"]).sum()) if len(refits) else 0,
    }
    return GarchForecast(out, refits, meta)


# ----------------------------------------------------------------------------------------------
# Losses and Diebold-Mariano
# ----------------------------------------------------------------------------------------------
def qlike(realized: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> np.ndarray:
    """Normalised QLIKE = RV/h - ln(RV/h) - 1 >= 0, zero iff h == RV. Infinite when RV == 0."""
    rv = np.asarray(realized, dtype=float)
    h = np.asarray(forecast, dtype=float)
    ratio = rv / h
    with np.errstate(divide="ignore"):
        return ratio - np.log(ratio) - 1.0


def qlike_rank(realized: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> np.ndarray:
    """Patton (2011) ranking form RV/h + ln h: finite at RV == 0; differences equal QLIKE differences."""
    rv = np.asarray(realized, dtype=float)
    h = np.asarray(forecast, dtype=float)
    return rv / h + np.log(h)


def mse_loss(realized: pd.Series | np.ndarray, forecast: pd.Series | np.ndarray) -> np.ndarray:
    rv = np.asarray(realized, dtype=float)
    h = np.asarray(forecast, dtype=float)
    return (rv - h) ** 2


def diebold_mariano(
    loss_model: np.ndarray | pd.Series, loss_base: np.ndarray | pd.Series, lags: int | None = None
) -> dict[str, float]:
    """DM test of equal predictive accuracy; d = loss_model - loss_base (negative mean: model better).

    HAC (Newey-West) standard error via `newey_west_tstat`; two-sided p from the normal.
    """
    d = pd.Series(np.asarray(loss_model, dtype=float) - np.asarray(loss_base, dtype=float))
    d = d[np.isfinite(d)]
    t = newey_west_tstat(d, lags=lags)
    p = float(2 * stats.norm.sf(abs(t))) if math.isfinite(t) else float("nan")
    return {
        "mean_diff": float(d.mean()) if len(d) else float("nan"),
        "dm_t": t,
        "p_two_sided": p,
        "n": len(d),
    }
