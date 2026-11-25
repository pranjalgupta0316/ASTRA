"""Long-only, fully-invested portfolio optimizers with per-asset caps (CVXPY, CLARABEL first).

All inputs are per-period (daily) moments; outputs are weights summing to 1 with 0 <= w_i <= cap_i.
Risk aversion in `mean_variance` is unit-free with respect to annualization: maximizing
mu'w - gamma/2 w'Sw in daily units gives the same weights as in annual units (both terms scale by P).

Mean estimates are mostly noise at a 252-day horizon (with one year of data the standard error of
the annual mean is about the annual volatility itself),
so `mean_variance` and `max_sharpe` use `shrunk_mean` and are reported descriptively only.
"""

from __future__ import annotations

from collections.abc import Sequence

import cvxpy as cp
import numpy as np

SOLVERS: tuple[str, ...] = ("CLARABEL", "SCS")
DEFAULT_CAPS = {"US": 0.4, "IN": 0.4, "CRYPTO": 0.6}
_TOL = 1e-7


class OptimizationError(RuntimeError):
    pass


def default_cap(market: str) -> float:
    return DEFAULT_CAPS["CRYPTO" if market == "CRYPTO" else "US"]


def _caps(n: int, caps: float | Sequence[float] | np.ndarray | None) -> np.ndarray:
    c = np.ones(n) if caps is None else np.broadcast_to(np.asarray(caps, dtype=float), (n,)).copy()
    if np.any(c < 0):
        raise ValueError("caps must be non-negative")
    if c.sum() < 1 - 1e-12:
        raise ValueError(f"infeasible: sum of caps {c.sum():.3f} < 1 for a fully-invested portfolio")
    return np.minimum(c, 1.0)


def _solve(prob: cp.Problem) -> None:
    last: Exception | None = None
    installed = set(cp.installed_solvers())
    for s in SOLVERS:
        if s not in installed:
            continue
        try:
            prob.solve(solver=s)
        except cp.error.SolverError as e:  # pragma: no cover - solver-specific failure
            last = e
            continue
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return
    raise OptimizationError(f"optimization failed (status={prob.status}, last error={last})")


def _clean(w: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Remove solver noise: clip to [0, cap], renormalize, and verify constraints."""
    w = np.clip(np.asarray(w, dtype=float).ravel(), 0.0, caps)
    s = w.sum()
    if s <= 0:
        raise OptimizationError("optimizer returned zero weights")
    w = w / s
    # renormalization may push a capped name marginally above its cap; redistribute the excess
    for _ in range(10):
        over = w > caps
        if not over.any():
            break
        excess = float((w[over] - caps[over]).sum())
        w[over] = caps[over]
        room = np.where(w < caps, caps - w, 0.0)
        w = w + excess * room / room.sum()
    if abs(w.sum() - 1) > 1e-8 or np.any(w < -1e-12) or np.any(w > caps + 1e-8):
        raise OptimizationError("weights violate constraints after cleaning")
    return w


def _psd(cov: np.ndarray) -> np.ndarray:
    s = np.asarray(cov, dtype=float)
    return (s + s.T) / 2


def _scale(cov: np.ndarray) -> float:
    """Average variance; dividing the problem by it leaves the argmin unchanged but puts the objective
    near 1, where the solvers' absolute tolerances (~1e-8) translate into accurate weights."""
    k = float(np.mean(np.diag(cov)))
    if not np.isfinite(k) or k <= 0:
        raise OptimizationError("covariance has non-positive average variance")
    return k


def equal_weight(n: int, caps: float | Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    c = _caps(n, caps)
    return _clean(np.full(n, 1.0 / n), c)


def min_variance(cov: np.ndarray, caps: float | Sequence[float] | np.ndarray | None = None) -> np.ndarray:
    s = _psd(cov)
    s = s / _scale(s)
    n = s.shape[0]
    c = _caps(n, caps)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(s))), [cp.sum(w) == 1, w >= 0, w <= c])
    _solve(prob)
    return _clean(w.value, c)


def mean_variance(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 5.0,
    caps: float | Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """argmax mu'w - risk_aversion/2 * w'Sw, long-only, fully invested, capped."""
    s = _psd(cov)
    k = _scale(s)
    s, mu = s / k, np.asarray(mu, float) / k  # both terms divided by k: same argmax
    n = s.shape[0]
    c = _caps(n, caps)
    w = cp.Variable(n)
    obj = cp.Maximize(mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(s)))
    prob = cp.Problem(obj, [cp.sum(w) == 1, w >= 0, w <= c])
    _solve(prob)
    return _clean(w.value, c)


def _max_capped_mean(mu: np.ndarray, caps: np.ndarray) -> float:
    """max mu'w over {w >= 0, sum w = 1, w <= caps} (greedy fill by descending mu)."""
    left, total = 1.0, 0.0
    for i in np.argsort(-mu):
        take = min(caps[i], left)
        total += take * mu[i]
        left -= take
        if left <= 1e-15:
            break
    return total


def max_sharpe(
    mu: np.ndarray,
    cov: np.ndarray,
    caps: float | Sequence[float] | np.ndarray | None = None,
    rf: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """Maximum Sharpe via the convex reformulation (Cornuejols & Tutuncu): with y = k w, k > 0,
    minimize y'Sy s.t. (mu - rf)'y = 1, y >= 0, sum y = k, y <= cap * k; then w = y / sum(y).

    Returns (weights, fallback_used). If no feasible portfolio has positive excess mean the ratio is
    not maximizable in this form; we fall back to minimum variance and flag it.
    """
    s = _psd(cov)
    n = s.shape[0]
    c = _caps(n, caps)
    ex = np.asarray(mu, float) - rf
    if _max_capped_mean(ex, c) <= 0:
        return min_variance(s, c), True
    # rescale so that y'Sy ~ 1 at the optimum: S/k and mu/sqrt(k) leave w = y/sum(y) unchanged
    k = _scale(s)
    s_n, ex_n = s / k, ex / np.sqrt(k)
    y = cp.Variable(n)
    tot = cp.sum(y)
    cons = [ex_n @ y == 1, y >= 0, y <= c * tot]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(s_n))), cons)
    _solve(prob)
    if y.value is None:
        raise RuntimeError("max-Sharpe solve returned no solution")
    return _clean(y.value / float(np.sum(y.value)), c), False


def _spinu_newton(cov: np.ndarray, b: np.ndarray, x0: np.ndarray, iters: int = 50) -> np.ndarray:
    """Damped Newton polish of the unconstrained Spinu problem min 0.5 x'Sx - b'log x (x > 0)."""
    x = x0.copy()
    for _ in range(iters):
        g = cov @ x - b / x
        h = cov + np.diag(b / x**2)
        step = np.linalg.solve(h, g)
        t = 1.0
        while np.any(x - t * step <= 0):
            t *= 0.5
        x = x - t * step
        if np.max(np.abs(g)) < 1e-14:
            break
    return x


def risk_parity(
    cov: np.ndarray,
    budgets: Sequence[float] | np.ndarray | None = None,
    caps: float | Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Risk budgeting via Spinu (2013) convex log-barrier: min 0.5 x'Sx - sum b_i log x_i, x > 0,
    then w = x / sum(x). At the optimum x_i (Sx)_i = b_i, i.e. risk contributions proportional to b.

    Caps are imposed as homogeneous linear constraints x_i <= cap_i * sum(x) (still convex). If a cap
    binds, contributions are no longer exactly equal (reported by `risk_contributions`). When no cap
    binds the solution is polished with Newton steps so that contributions are equal to ~1e-12.
    """
    s = _psd(cov)
    n = s.shape[0]
    c = _caps(n, caps)
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, float) / np.sum(budgets)
    s = s / _scale(s)  # x scales by sqrt(k); w = x / sum(x) unchanged
    x = cp.Variable(n, pos=True)
    cons = [x[i] <= c[i] * cp.sum(x) for i in range(n) if c[i] < 1.0]
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(s)) - b @ cp.log(x)), cons)
    _solve(prob)
    xv = np.maximum(np.asarray(x.value, float), 1e-12)
    w = xv / xv.sum()
    if np.all(w < c - 1e-6):  # no binding cap -> unconstrained KKT holds; polish
        xp = _spinu_newton(s, b, xv)
        wp = xp / xp.sum()
        if np.all(wp <= c):
            w = wp
    return _clean(w, c)


def vol_target(
    weights: np.ndarray,
    cov: np.ndarray,
    target_vol: float,
    periods_per_year: int,
    max_leverage: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Scale weights by min(max_leverage, target_vol / forecast_vol); the remainder is cash.

    forecast_vol = sqrt(w'Sw * periods_per_year) (ex-ante, annualized). Returns (scaled, scale).
    """
    w = np.asarray(weights, float)
    fv = float(np.sqrt(max(w @ _psd(cov) @ w, 0.0) * periods_per_year))
    scale = max_leverage if fv <= 0 else min(max_leverage, target_vol / fv)
    return w * scale, scale


def shrunk_mean(returns: np.ndarray, intensity: float = 0.5) -> np.ndarray:
    """Per-period sample mean shrunk toward its cross-sectional average with fixed intensity.

    A crude, a-priori fixed James-Stein-style shrinkage. Still noisy: with 252 daily observations the
    standard error of each mean is roughly sigma_daily / sqrt(252), comparable to the mean itself.
    """
    m = np.asarray(returns, float).mean(axis=0)
    return (1 - intensity) * m + intensity * m.mean()
