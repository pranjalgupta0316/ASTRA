from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from nebula.research.volatility_study import MIN_TRAIN, MODELS, build_forecast_frame, evaluate
from nebula.stats.volatility import (
    diebold_mariano,
    ewma_forecast,
    garch_forecast,
    garch_variance_path,
    log_returns,
    mse_loss,
    qlike,
    qlike_rank,
    realized_variance_proxies,
    rolling_mean_forecast,
)

GJR_TRUE = {"omega": 0.03, "alpha[1]": 0.03, "gamma[1]": 0.12, "beta[1]": 0.88}


def simulate_gjr(n: int, seed: int = 0, params: dict[str, float] = GJR_TRUE) -> tuple[pd.Series, np.ndarray]:
    """GJR-GARCH(1,1,1) in percent units with Gaussian innovations; returns decimal r and true variance."""
    rng = np.random.default_rng(seed)
    om, a, g, b = params["omega"], params["alpha[1]"], params["gamma[1]"], params["beta[1]"]
    s2 = np.empty(n + 1)
    e = np.empty(n)
    s2[0] = om / (1 - a - 0.5 * g - b)
    for t in range(n):
        e[t] = math.sqrt(s2[t]) * rng.standard_normal()
        s2[t + 1] = om + (a + g * (e[t] < 0)) * e[t] ** 2 + b * s2[t]
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(e / 100, index=idx), s2 / 1e4  # s2[t+1] = true variance of r_{t+1}


# ---------------------------------------------------------------------------------------------
# estimators on hand examples
# ---------------------------------------------------------------------------------------------
def test_realized_variance_proxies_hand_example() -> None:
    ohlc = pd.DataFrame(
        {"open": [99.0, 100.0], "high": [101.0, 110.0], "low": [98.0, 95.0], "close": [100.0, 105.0]}
    )
    rv = realized_variance_proxies(ohlc)
    assert math.isnan(rv["rv_cc"].iloc[0])
    assert rv["rv_cc"].iloc[1] == pytest.approx(math.log(1.05) ** 2, rel=1e-12)
    assert rv["rv_pk"].iloc[1] == pytest.approx(math.log(110 / 95) ** 2 / (4 * math.log(2)), rel=1e-12)
    gk = 0.5 * math.log(110 / 95) ** 2 - (2 * math.log(2) - 1) * math.log(105 / 100) ** 2
    assert rv["rv_gk"].iloc[1] == pytest.approx(gk, rel=1e-12)


def test_rolling_and_ewma_hand_examples() -> None:
    r = pd.Series([0.01, -0.02, 0.03])
    roll = rolling_mean_forecast(r, 2)
    assert math.isnan(roll.iloc[0])
    assert roll.iloc[1] == pytest.approx((0.0001 + 0.0004) / 2)
    assert roll.iloc[2] == pytest.approx((0.0004 + 0.0009) / 2)
    ew = ewma_forecast(pd.Series([0.1, 0.2, -0.1]), lam=0.5, min_periods=1)
    assert ew.tolist() == pytest.approx([0.01, 0.5 * 0.01 + 0.5 * 0.04, 0.5 * 0.025 + 0.5 * 0.01])
    assert ewma_forecast(pd.Series(np.full(30, 0.01))).iloc[:19].isna().all()


def test_log_returns() -> None:
    lr = log_returns(pd.Series([100.0, 110.0, 99.0]))
    assert lr.iloc[1:].tolist() == pytest.approx([math.log(1.1), math.log(0.9)])


# ---------------------------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------------------------
def test_qlike_zero_at_perfect_forecast_positive_otherwise() -> None:
    rv = np.array([0.5, 1.0, 2.0, 7.3])
    assert np.allclose(qlike(rv, rv), 0.0, atol=1e-15)
    for scale in (0.3, 0.9, 1.1, 4.0):
        assert np.all(qlike(rv, rv * scale) > 0)
    # ranking form: differences equal normalised-QLIKE differences, finite at RV = 0
    h1, h2 = rv * 0.7, rv * 1.6
    assert np.allclose(qlike(rv, h1) - qlike(rv, h2), qlike_rank(rv, h1) - qlike_rank(rv, h2))
    assert np.isfinite(qlike_rank(np.array([0.0]), np.array([1.0]))).all()
    # QLIKE penalises under-prediction more than over-prediction by the same factor
    assert qlike(np.array([1.0]), np.array([0.5]))[0] > qlike(np.array([1.0]), np.array([2.0]))[0]


def test_mse_loss() -> None:
    assert mse_loss(np.array([1.0, 2.0]), np.array([1.0, 4.0])).tolist() == [0.0, 4.0]


def test_diebold_mariano_signs() -> None:
    rng = np.random.default_rng(1)
    base = rng.exponential(size=2000)
    same = diebold_mariano(base, base)
    assert same["dm_t"] == 0.0 and same["p_two_sided"] == pytest.approx(1.0)
    better = diebold_mariano(base - 0.2 + 0.05 * rng.standard_normal(2000), base)
    assert better["dm_t"] < -10 and better["p_two_sided"] < 1e-6
    assert better["mean_diff"] == pytest.approx(-0.2, abs=0.01)


# ---------------------------------------------------------------------------------------------
# causality / truncation invariance
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn", [lambda r: rolling_mean_forecast(r, 20), lambda r: rolling_mean_forecast(r, 60), ewma_forecast]
)
def test_simple_forecasts_truncation_invariant(fn) -> None:  # type: ignore[no-untyped-def]
    r, _ = simulate_gjr(400, seed=3)
    full = fn(r)
    for cut in (100, 257, 399):
        part = fn(r.iloc[:cut])
        pd.testing.assert_series_equal(part, full.iloc[:cut])


def test_garch_recursion_matches_arch() -> None:
    r, _ = simulate_gjr(800, seed=5)
    y = 100 * r.to_numpy()
    res = arch_model(y, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
    s2 = garch_variance_path(y, {k: float(v) for k, v in res.params.items()})
    assert np.allclose(s2[:-1], res.conditional_volatility**2, rtol=1e-10)
    assert s2[-1] == pytest.approx(
        float(res.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]), rel=1e-10
    )


@pytest.mark.parametrize("kind", ["garch", "gjr"])
def test_garch_forecast_causal(kind: str) -> None:
    r, _ = simulate_gjr(420, seed=11)
    min_train, every = 300, 21
    full = garch_forecast(r, kind, every, min_train)  # type: ignore[arg-type]
    # nothing before the first refit origin
    assert full.forecast.iloc[: min_train - 1].isna().all()
    assert full.forecast.iloc[min_train - 1 :].notna().all() and (full.forecast.dropna() > 0).all()
    # truncation: the forecast for origin t only uses data <= t
    for cut in (min_train + 5, min_train + 21, min_train + 50):
        part = garch_forecast(r.iloc[:cut], kind, every, min_train)  # type: ignore[arg-type]
        pd.testing.assert_series_equal(part.forecast, full.forecast.iloc[:cut])
    # perturbing the future (after origin k) leaves forecasts at origins <= k unchanged
    k = min_train + 30
    r2 = r.copy()
    r2.iloc[k + 1 :] = r2.iloc[k + 1 :] * 3.0
    alt = garch_forecast(r2, kind, every, min_train)  # type: ignore[arg-type]
    pd.testing.assert_series_equal(alt.forecast.iloc[: k + 1], full.forecast.iloc[: k + 1])
    assert not np.allclose(alt.forecast.iloc[k + 1 :], full.forecast.iloc[k + 1 :])
    assert full.meta["n_refits"] == len(range(min_train - 1, len(r), every))


def test_gjr_tracks_true_variance_better_than_ewma() -> None:
    r, s2 = simulate_gjr(2500, seed=21)
    g = garch_forecast(r, "gjr", 21, 504).forecast
    e = ewma_forecast(r)
    true_next = pd.Series(s2[1:], index=r.index)  # true variance of r_{t+1} at origin t
    m = g.notna()
    err_g = np.mean((g[m] - true_next[m]) ** 2)
    err_e = np.mean((e[m] - true_next[m]) ** 2)
    assert err_g < 0.5 * err_e


def test_study_pipeline_on_synthetic_bars() -> None:
    r, _ = simulate_gjr(MIN_TRAIN + 120, seed=8)
    close = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r.to_numpy())]))
    idx = pd.bdate_range("2014-12-31", periods=len(close))
    rng = np.random.default_rng(0)
    open_ = close * np.exp(rng.normal(0, 0.002, len(close)))
    adj = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.004,
            "low": np.minimum(open_, close) * 0.996,
            "close": close,
        },
        index=idx,
    )
    frame, fits = build_forecast_frame(adj)
    res = evaluate(frame)
    assert res["n_eval"] == len(r) - MIN_TRAIN  # origins 503 .. n-2
    assert set(res["dm_vs_ewma"]) == set(MODELS) - {"ewma"}
    # target alignment: rv_cc_next at origin t equals r_{t+1}^2
    assert np.allclose(frame["rv_cc_next"].iloc[:-1].to_numpy(), r.to_numpy()[1:] ** 2)
    assert fits["gjr"].meta["n_refits"] == len(range(MIN_TRAIN - 1, len(r), 21))
