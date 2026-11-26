"""Independent vectorized reference implementations of the bar engines (verification only).

Why: the event-driven engines in `astra.backtest.engine` and `astra.portfolio.backtest` are loops with
order sizing, caps and tolerances. A second implementation written from the *return algebra* instead of
the share ledger cross-checks them: if both agree to floating-point precision on random inputs, the
loop's bookkeeping (cash, positions, marking, fill timing) is right.

Scope: the frictionless case the algebra covers exactly - no costs, no volume cap, no tolerance, every
bar has a target, every bar trades. Target w_t decided at close t is filled at open t+1 (never at t):

    weight drifts overnight:  E_open(t+1)  = E_close(t)    * (1 + h_t     * (o_{t+1}/c_t     - 1))
    rebalance at the open, hold to close:
                              E_close(t+1) = E_open(t+1)   * (1 + w_t     * (c_{t+1}/o_{t+1} - 1))
    weight at that close:     h_{t+1}      = w_t * (c_{t+1}/o_{t+1}) / (1 + w_t * (c_{t+1}/o_{t+1} - 1))

Multi-asset: the same with vectors (h, w) and portfolio returns sum_j w_j * r_j.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reference_equity(
    open_: np.ndarray, close: np.ndarray, targets: np.ndarray, initial_cash: float = 1.0
) -> np.ndarray:
    """Close-to-close equity of a target-weight strategy, frictionless.

    open_, close: shape (T,) or (T, N). targets: same shape; row t = weights decided at close t
    (filled at open t+1). Returns equity at each close, shape (T,)."""
    o = np.asarray(open_, float)
    c = np.asarray(close, float)
    w = np.asarray(targets, float)
    if o.ndim == 1:
        o, c, w = o[:, None], c[:, None], w[:, None]
    t_len = o.shape[0]
    eq = np.empty(t_len)
    eq[0] = initial_cash  # all cash on the first bar: nothing decided before it
    held = np.zeros(o.shape[1])  # weights at the previous close
    for t in range(1, t_len):
        overnight = o[t] / c[t - 1] - 1.0
        e_open = eq[t - 1] * (1.0 + float(held @ overnight))
        intraday = c[t] / o[t] - 1.0
        growth = 1.0 + float(w[t - 1] @ intraday)
        eq[t] = e_open * growth
        held = w[t - 1] * (c[t] / o[t]) / growth if growth != 0 else np.zeros_like(held)
    return eq


def reference_from_frames(bars: pd.DataFrame, weights: pd.Series, initial_cash: float) -> pd.Series:
    """Single-asset convenience wrapper: bars with open/close, weights aligned to the bars' rows."""
    eq = reference_equity(bars["open"].to_numpy(), bars["close"].to_numpy(), weights.to_numpy(), initial_cash)
    return pd.Series(eq, index=pd.to_datetime(bars["session_date"]))
