"""Walk-forward Gaussian-mixture volatility regimes with stable (vol-sorted) labels.

Features at origin t use returns r_{t-19..t} only. Models are fit on expanding windows ending at the
refit origin and applied unchanged to the following `refit_every` origins. After every fit the
components are relabelled by ascending training-sample mean of the vol feature, so state 0 = low vol
and state K-1 = high vol in every block (no label switching across refits).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.mixture import GaussianMixture

__all__ = [
    "RegimeResult",
    "fit_mapped_gmm",
    "persistence",
    "regime_features",
    "state_difference_tests",
    "walk_forward_gmm",
]

VOL_COL = "vol20"
FEATURES = ["vol20", "ret20"]


def regime_features(r: pd.Series, window: int = 20) -> pd.DataFrame:
    """vol20_t = std(r_{t-w+1..t}) (ddof=1), ret20_t = sum(r_{t-w+1..t})."""
    rs = pd.Series(r, dtype=float)
    return pd.DataFrame(
        {
            "vol20": rs.rolling(window, min_periods=window).std(),
            "ret20": rs.rolling(window, min_periods=window).sum(),
        }
    )


@dataclass
class MappedGMM:
    gmm: GaussianMixture
    mean: np.ndarray
    scale: np.ndarray
    order: np.ndarray  # order[k] = raw component index of mapped state k
    train_state_vol: np.ndarray  # training-sample mean vol per mapped state

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = self.gmm.predict((x - self.mean) / self.scale)
        inv = np.empty_like(self.order)
        inv[self.order] = np.arange(len(self.order))
        return inv[raw]


def fit_mapped_gmm(
    x: np.ndarray, n_states: int = 2, seed: int = 7, n_init: int = 3, vol_index: int = 0
) -> MappedGMM:
    """Fit a GMM on standardised training features and sort components by training mean vol."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    gmm = GaussianMixture(n_components=n_states, covariance_type="full", n_init=n_init, random_state=seed)
    gmm.fit(z)
    raw = gmm.predict(z)
    vols = np.array([x[raw == k, vol_index].mean() if np.any(raw == k) else np.inf for k in range(n_states)])
    order = np.argsort(vols, kind="stable")
    return MappedGMM(gmm, mean, scale, order, vols[order])


@dataclass
class RegimeResult:
    states: pd.Series  # mapped OOS state at each origin (NaN outside OOS)
    refits: pd.DataFrame
    cond_forecast: pd.Series | None = None  # training mean of target within the predicted state
    uncond_forecast: pd.Series | None = None  # training mean of target (no conditioning)
    meta: dict[str, Any] = field(default_factory=dict)


def walk_forward_gmm(
    features: pd.DataFrame,
    n_states: int = 2,
    min_train: int = 504,
    refit_every: int = 63,
    seed: int = 7,
    n_init: int = 3,
    target_next: pd.Series | None = None,
) -> RegimeResult:
    """Expanding-window GMM; refit origins anchored at row position `min_train - 1` of `features`.

    Training rows for the fit at origin o: positions <= o with complete features.
    If `target_next` is given (value at position p = quantity realised at p + 1, e.g. r_{p+1}^2), the
    result also carries a state-conditional forecast (mean target over training positions <= o - 1,
    i.e. targets realised by o, grouped by in-sample state) and the unconditional training mean.
    """
    f = features[FEATURES].astype(float)
    n = len(f)
    x_all = f.to_numpy()
    complete = np.isfinite(x_all).all(axis=1)
    states = np.full(n, np.nan)
    cond = np.full(n, np.nan)
    uncond = np.full(n, np.nan)
    tgt = None if target_next is None else pd.Series(target_next, dtype=float).to_numpy()
    rows: list[dict[str, Any]] = []
    for o in range(min_train - 1, n, refit_every):
        mask = complete[: o + 1]
        x_train = x_all[: o + 1][mask]
        if len(x_train) < 10 * n_states:
            continue
        m = fit_mapped_gmm(x_train, n_states=n_states, seed=seed, n_init=n_init)
        end = min(o + refit_every, n)
        block = x_all[o:end]
        ok = np.isfinite(block).all(axis=1)
        pred = np.full(end - o, np.nan)
        if ok.any():
            pred[ok] = m.predict(block[ok])
        states[o:end] = pred
        if tgt is not None:
            # purge: the target at position o is realised at o + 1 (future), so use positions <= o - 1
            idx = np.flatnonzero(complete[:o] & np.isfinite(tgt[:o]))
            s_in = m.predict(x_all[idx])
            means = np.array(
                [tgt[idx][s_in == k].mean() if np.any(s_in == k) else np.nan for k in range(n_states)]
            )
            base = float(tgt[idx].mean())
            means = np.where(np.isfinite(means), means, base)
            uncond[o:end] = base
            cond[o:end] = np.where(np.isfinite(pred), means[np.nan_to_num(pred, nan=0).astype(int)], np.nan)
        rows.append(
            {
                "origin": f.index[o],
                "n_train": len(x_train),
                "converged": bool(m.gmm.converged_),
                **{f"train_vol_state{k}": float(v) for k, v in enumerate(m.train_state_vol)},
                **{
                    f"weight_state{k}": float(np.asarray(m.gmm.weights_)[m.order[k]]) for k in range(n_states)
                },
            }
        )
    return RegimeResult(
        pd.Series(states, index=f.index, name="state"),
        pd.DataFrame(rows),
        None if tgt is None else pd.Series(cond, index=f.index, name="cond_fc"),
        None if tgt is None else pd.Series(uncond, index=f.index, name="uncond_fc"),
        {
            "n_states": n_states,
            "min_train": min_train,
            "refit_every": refit_every,
            "seed": seed,
            "n_refits": len(rows),
        },
    )


def persistence(states: pd.Series) -> dict[str, float]:
    """P(s_{t+1} == s_t) on consecutive non-NaN states, with the independence benchmark sum p_k^2."""
    s = pd.Series(states).dropna().to_numpy()
    if len(s) < 2:
        return {"persistence": float("nan"), "independence_benchmark": float("nan"), "n": len(s)}
    same = float(np.mean(s[1:] == s[:-1]))
    _, counts = np.unique(s, return_counts=True)
    p = counts / counts.sum()
    return {"persistence": same, "independence_benchmark": float(np.sum(p**2)), "n": len(s)}


def state_difference_tests(target: pd.Series, high: pd.Series) -> dict[str, float]:
    """Compare target (e.g. next-day RV) between high (1) and low (0) states.

    Mann-Whitney U and Welch t are one-sided (high > low). The HAC check regresses target on the
    high-state dummy (OLS, Newey-West HAC covariance, Bartlett lags floor(4 (n/100)^(2/9)), the same
    rule as `nebula.stats.inference.newey_west_tstat`).
    """
    df = pd.DataFrame({"y": target, "d": high}).dropna()
    y_hi = df.loc[df["d"] == 1, "y"].to_numpy()
    y_lo = df.loc[df["d"] == 0, "y"].to_numpy()
    out: dict[str, float] = {
        "n_high": len(y_hi),
        "n_low": len(y_lo),
        "mean_high": float(y_hi.mean()) if len(y_hi) else float("nan"),
        "mean_low": float(y_lo.mean()) if len(y_lo) else float("nan"),
        "median_high": float(np.median(y_hi)) if len(y_hi) else float("nan"),
        "median_low": float(np.median(y_lo)) if len(y_lo) else float("nan"),
    }
    if len(y_hi) < 2 or len(y_lo) < 2:
        out.update(
            {"mw_p_one_sided": float("nan"), "welch_t": float("nan"), "welch_p_one_sided": float("nan")}
        )
        out.update({"hac_t": float("nan"), "ratio_mean": float("nan")})
        return out
    mw = stats.mannwhitneyu(y_hi, y_lo, alternative="greater")
    w = stats.ttest_ind(y_hi, y_lo, equal_var=False, alternative="greater")
    n = len(df)
    lags = math.floor(4 * (n / 100) ** (2 / 9))
    ols = sm.OLS(df["y"].to_numpy(float), sm.add_constant(df["d"].to_numpy(float))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    b = float(ols.params[1])
    t_hac = float(ols.tvalues[1])
    out.update(
        {
            "mw_u": float(mw.statistic),
            "mw_p_one_sided": float(mw.pvalue),
            "welch_t": float(w.statistic),
            "welch_p_one_sided": float(w.pvalue),
            "hac_slope": b,
            "hac_t": t_hac,
            "hac_lags": lags,
            "ratio_mean": float(y_hi.mean() / y_lo.mean()) if y_lo.mean() > 0 else float("nan"),
        }
    )
    return out
