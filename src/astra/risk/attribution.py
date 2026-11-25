"""Risk attribution, concentration, forecast-vs-realized risk, parametric/historical VaR and ES.

Conventions: inputs are per-period (daily) returns/covariances. Annualization is explicit via
`periods_per_year` (252 equities, 365 crypto). VaR/ES are one-period (one-day for daily data) losses
reported as positive numbers; ES >= VaR is checked by tests.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from astra.risk.metrics import expected_shortfall, var_historical

# ---------------------------------------------------------------- risk contributions / concentration


def portfolio_vol(weights: np.ndarray, cov: np.ndarray) -> float:
    w = np.asarray(weights, float)
    return float(math.sqrt(max(float(w @ cov @ w), 0.0)))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> pd.DataFrame:
    """Euler decomposition: RC_i = w_i (S w)_i / sigma_p; sum_i RC_i = sigma_p (per period).
    Columns: weight, rc (vol units), pct (share of total risk, sums to 1)."""
    w = np.asarray(weights, float)
    s = np.asarray(cov, float)
    sig = portfolio_vol(w, s)
    mrc = s @ w
    rc = w * mrc / sig if sig > 0 else np.zeros_like(w)
    pct = rc / sig if sig > 0 else np.zeros_like(w)
    return pd.DataFrame({"weight": w, "rc": rc, "pct": pct})


def hhi(weights: np.ndarray) -> float:
    """Herfindahl index of normalized absolute weights (1/N for equal weight, 1 for a single name)."""
    w = np.abs(np.asarray(weights, float))
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    return float(np.sum(w**2))


def effective_n(weights: np.ndarray) -> float:
    """Effective number of holdings, 1 / HHI(weights)."""
    h = hhi(weights)
    return float(1.0 / h) if h > 0 else float("nan")


def effective_number_of_bets(weights: np.ndarray, cov: np.ndarray) -> float:
    """Risk-based concentration: 1 / sum(pct_rc^2) (N for exact equal risk contributions). Uses Euler
    contributions on the asset basis (not the PCA/minimum-torsion bets of Meucci)."""
    p = risk_contributions(weights, cov)["pct"].to_numpy()
    d = float(np.sum(p**2))
    return float(1.0 / d) if d > 0 else float("nan")


# ---------------------------------------------------------------- forecast vs realized


def forecast_vol(weights: np.ndarray, cov: np.ndarray, periods_per_year: int) -> float:
    """Ex-ante annualized volatility sqrt(w'Sw * P)."""
    return portfolio_vol(weights, cov) * math.sqrt(periods_per_year)


def forecast_vs_realized(
    forecast_ann: pd.Series, returns: pd.Series, periods_per_year: int, min_obs: int = 5
) -> pd.DataFrame:
    """Compare ex-ante forecasts made at decision dates with realized vol over each holding period.

    forecast_ann: annualized forecast indexed by decision date t. Holding period = returns strictly
    after t up to and including the next decision date (orders fill at t+1 open, so the first return
    after t is the first partly exposed one). Realized vol = std(ddof=1) * sqrt(P) of those returns.
    """
    r = returns.dropna()
    dates = list(pd.DatetimeIndex(forecast_ann.index))
    out = []
    for k, t in enumerate(dates):
        end = dates[k + 1] if k + 1 < len(dates) else r.index.max()
        seg = r[(r.index > t) & (r.index <= end)]
        if len(seg) < min_obs:
            continue
        realized = float(seg.std(ddof=1) * math.sqrt(periods_per_year))
        f = float(forecast_ann.iloc[k])
        out.append(
            {
                "decision_date": t,
                "forecast_vol_ann": f,
                "realized_vol_ann": realized,
                "ratio": realized / f if f > 0 else float("nan"),
                "n_obs": len(seg),
            }
        )
    return pd.DataFrame(out).set_index("decision_date") if out else pd.DataFrame()


def forecast_realized_summary(table: pd.DataFrame) -> dict[str, float]:
    """Mean forecast, mean realized (per holding period), median ratio realized/forecast, and the
    fraction of periods where realized exceeded forecast."""
    if table.empty:
        return {}
    return {
        "mean_forecast_vol_ann": float(table["forecast_vol_ann"].mean()),
        "mean_period_realized_vol_ann": float(table["realized_vol_ann"].mean()),
        "median_ratio_realized_to_forecast": float(table["ratio"].median()),
        "frac_realized_above_forecast": float((table["ratio"] > 1).mean()),
        "n_periods": len(table),
    }


# ---------------------------------------------------------------- VaR / ES


def _cf_quantile(z: np.ndarray | float, skew: float, exkurt: float) -> np.ndarray | float:
    """Cornish-Fisher expansion of a standard-normal quantile z (third/fourth moments)."""
    return z + (z**2 - 1) * skew / 6 + (z**3 - 3 * z) * exkurt / 24 - (2 * z**3 - 5 * z) * skew**2 / 36


def var_es(returns: pd.Series, level: float = 0.95, method: str = "historical") -> tuple[float, float]:
    """One-period VaR and ES at `level` (positive loss numbers).

    historical: empirical quantile / tail mean (via astra.risk.metrics).
    gaussian: mu + sigma * z.  cornish_fisher: modified quantile (Favre & Galeano 2002); ES as the
    average of the CF quantile function over the tail (numerical integration on a fine grid). The CF
    expansion is only valid for moderate skew/kurtosis; it may be non-monotone for extreme moments.
    """
    r = pd.Series(returns).dropna()
    alpha = 1 - level
    if method == "historical":
        return var_historical(r, level), expected_shortfall(r, level)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if method == "gaussian":
        z = stats.norm.ppf(alpha)
        return float(-(mu + sd * z)), float(-mu + sd * stats.norm.pdf(z) / alpha)
    if method == "cornish_fisher":
        sk = float(stats.skew(r))
        ek = float(stats.kurtosis(r, fisher=True))
        z = stats.norm.ppf(alpha)
        var = -(mu + sd * float(_cf_quantile(z, sk, ek)))
        u = (np.arange(2000) + 0.5) / 2000 * alpha  # midpoints of the tail (0, alpha)
        es = -(mu + sd * float(np.mean(_cf_quantile(stats.norm.ppf(u), sk, ek))))
        return float(var), float(es)
    raise ValueError(f"unknown VaR method {method!r}")


def var_es_table(returns: pd.Series, levels: tuple[float, ...] = (0.95, 0.99)) -> pd.DataFrame:
    rows = []
    for lvl in levels:
        for m in ("historical", "gaussian", "cornish_fisher"):
            v, e = var_es(returns, lvl, m)
            rows.append({"level": lvl, "method": m, "horizon": "1 period", "var": v, "es": e})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- paired bootstrap for comparisons


def stationary_bootstrap_indices(n: int, n_boot: int, mean_block: int, seed: int) -> np.ndarray:
    """(n_boot, n) index matrix for the Politis-Romano stationary bootstrap (geometric block lengths).
    The same indices applied to two aligned series give a paired (joint) resample."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(n, size=n_boot)
    new_block = rng.random((n_boot, n)) < p
    starts = rng.integers(n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def paired_bootstrap_ci(
    a: pd.Series,
    b: pd.Series,
    periods_per_year: int,
    n_boot: int = 2000,
    mean_block: int = 20,
    alpha: float = 0.05,
    seed: int = 7,
) -> dict[str, float]:
    """Paired stationary-bootstrap CIs for vol(a)/vol(b) and Sharpe(a) - Sharpe(b) (annualized, rf=0)."""
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    x = df.to_numpy(float)
    n = len(x)
    idx = stationary_bootstrap_indices(n, n_boot, mean_block, seed)
    xa, xb = x[:, 0][idx], x[:, 1][idx]
    sa, sb = xa.std(axis=1, ddof=1), xb.std(axis=1, ddof=1)
    ratio = sa / sb
    k = math.sqrt(periods_per_year)
    sdiff = (xa.mean(axis=1) / sa - xb.mean(axis=1) / sb) * k
    q = [alpha / 2, 1 - alpha / 2]
    rlo, rhi = np.nanquantile(ratio, q)
    dlo, dhi = np.nanquantile(sdiff, q)
    return {
        "vol_ratio": float(x[:, 0].std(ddof=1) / x[:, 1].std(ddof=1)),
        "vol_ratio_ci_lo": float(rlo),
        "vol_ratio_ci_hi": float(rhi),
        "sharpe_diff": float(
            (x[:, 0].mean() / x[:, 0].std(ddof=1) - x[:, 1].mean() / x[:, 1].std(ddof=1)) * k
        ),
        "sharpe_diff_ci_lo": float(dlo),
        "sharpe_diff_ci_hi": float(dhi),
        "n_obs": n,
        "n_boot": n_boot,
        "mean_block": mean_block,
    }
