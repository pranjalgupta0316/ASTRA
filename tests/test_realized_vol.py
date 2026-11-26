"""Synthetic, offline tests for the realized-volatility study (H-RV)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.research.realized_vol import (
    MIN_1M_BARS,
    fit_har,
    har_design,
    har_forecast,
    realized_variance,
)


def _minute_bars(log_price: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    t = pd.date_range(start, periods=len(log_price), freq="1min", tz="UTC")
    return pd.DataFrame({"open_time_utc": t, "close": np.exp(log_price)})


def test_rv_of_constant_variance_matches_truth() -> None:
    rng = np.random.default_rng(0)
    days, s2 = 60, 1e-7  # per-minute variance
    lp = np.cumsum(rng.normal(0.0, np.sqrt(s2), days * 1440)) + np.log(100.0)
    bars = _minute_bars(lp)
    true_daily = 1440 * s2
    rv5 = realized_variance(bars, 1, 5, MIN_1M_BARS)
    rv1 = realized_variance(bars, 1, 1, MIN_1M_BARS)
    # first day lacks the 00:00 point (no previous-day close): 287 returns; others complete
    assert rv5["n_returns"].iloc[0] == 287 and (rv5["n_returns"].iloc[1:] == 288).all()
    assert (rv1["n_returns"].iloc[1:] == 1440).all()
    assert rv5["valid"].all()
    assert abs(rv5["rv"].iloc[1:].mean() / true_daily - 1) < 0.03
    assert abs(rv1["rv"].iloc[1:].mean() / true_daily - 1) < 0.01


def test_validity_mask_and_gap_returns_never_filled() -> None:
    rng = np.random.default_rng(1)
    lp = np.cumsum(rng.normal(0.0, 1e-3, 4 * 1440))
    bars = _minute_bars(lp)
    drop_small = np.arange(1440 + 100, 1440 + 150)  # day 1: 50 missing -> valid
    drop_big = np.arange(2 * 1440 + 100, 2 * 1440 + 180)  # day 2: 80 missing (> 72) -> invalid
    keep = np.ones(len(bars), dtype=bool)
    keep[np.r_[drop_small, drop_big]] = False
    gapped = bars.loc[keep].reset_index(drop=True)
    # an off-grid bar must be excluded, not counted
    off = pd.DataFrame(
        {"open_time_utc": [bars["open_time_utc"].iloc[3 * 1440 + 10] + pd.Timedelta("17s")], "close": [1e6]}
    )
    gapped = pd.concat([gapped, off], ignore_index=True)
    rv = realized_variance(gapped, 1, 5, MIN_1M_BARS)
    assert list(rv["n_bars"]) == [1440, 1390, 1360, 1440]
    assert list(rv["valid"]) == [True, True, False, True]
    assert rv["n_returns"].iloc[3] == 288  # off-grid bar ignored
    # gap-spanning return: on day 1, grid closes inside the gap are absent -> one longer return
    full = realized_variance(bars, 1, 5, MIN_1M_BARS)
    assert rv["n_returns"].iloc[1] < full["n_returns"].iloc[1]
    assert rv["max_span_min"].iloc[1] > 5
    # exact check: sum of squared returns over consecutive observed 5-min grid closes of day 1
    t = pd.to_datetime(gapped["open_time_utc"], utc=True)
    on = t == t.dt.floor("1min")
    end = (t + pd.Timedelta("1min"))[on]
    c = np.log(gapped["close"][on].to_numpy())
    g = (end.dt.minute % 5 == 0).to_numpy()
    e, lc = end[g].dt.tz_convert(None).to_numpy(), c[g]
    d0, d1 = np.datetime64("2024-01-02T00:00"), np.datetime64("2024-01-03T00:00")
    sel = (e >= d0) & (e <= d1)
    expected = float(np.sum(np.diff(lc[sel]) ** 2))
    assert rv["rv"].iloc[1] == pytest.approx(expected, rel=1e-12)


def _simulate_har(n: int, beta: np.ndarray, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rv = np.full(n, beta[0] / (1 - beta[1:].sum()))
    for t in range(22, n - 1):
        mu = (
            beta[0]
            + beta[1] * rv[t]
            + beta[2] * rv[t - 4 : t + 1].mean()
            + beta[3] * rv[t - 21 : t + 1].mean()
        )
        rv[t + 1] = mu * rng.gamma(4.0, 0.25)  # mean-one multiplicative noise: E[e|X] = 0
    return pd.Series(rv, index=pd.date_range("2000-01-01", periods=n, freq="1D"))


def test_har_recovers_coefficients() -> None:
    beta = np.array([0.1, 0.35, 0.3, 0.2])
    rv = _simulate_har(30000, beta)
    valid = pd.Series(True, index=rv.index)
    X = har_design(rv, valid)
    y = rv.shift(-1)
    ok = X.notna().all(axis=1) & y.notna()
    b = fit_har(X[ok].to_numpy(), y[ok].to_numpy())
    np.testing.assert_allclose(b[1:], beta[1:], atol=0.06)
    assert b[0] == pytest.approx(beta[0], abs=0.05)


def test_har_design_skips_invalid_days_without_filling() -> None:
    rv = pd.Series(np.arange(1.0, 41.0), index=pd.date_range("2020-01-01", periods=40, freq="1D"))
    valid = pd.Series(True, index=rv.index)
    valid.iloc[30] = False
    X = har_design(rv, valid)
    assert X.iloc[:21].isna().all().all()
    assert X.iloc[30].isna().all()  # no forecast at an invalid origin
    assert X["w"].iloc[32] == pytest.approx(np.mean([29, 30, 32, 33]))  # day 31 (value 31) skipped
    assert X["d"].iloc[25] == 26.0


def test_har_forecast_truncation_invariant_and_floored() -> None:
    beta = np.array([0.1, 0.35, 0.3, 0.2])
    rv = _simulate_har(1200, beta, seed=3)
    valid = pd.Series(True, index=rv.index)
    valid.iloc[[300, 301, 900]] = False
    full = har_forecast(rv, valid, refit_every=21, min_train=730)
    assert full.meta["n_refits"] > 0
    first = full.forecast.first_valid_index()
    assert first is not None
    for cut in (800, 901, 1000):
        part = har_forecast(rv.iloc[:cut], valid.iloc[:cut], refit_every=21, min_train=730)
        pd.testing.assert_series_equal(part.forecast, full.forecast.iloc[:cut], check_names=False)
    # forecast at origin t must not change when data after t is perturbed
    t = 950
    rv2 = rv.copy()
    rv2.iloc[t + 1 :] *= 5.0
    alt = har_forecast(rv2, valid, refit_every=21, min_train=730)
    pd.testing.assert_series_equal(alt.forecast.iloc[: t + 1], full.forecast.iloc[: t + 1], check_names=False)
    # flooring: force negative forecasts with a negative-intercept series
    neg = pd.Series(np.r_[np.full(800, 1.0), np.full(400, 1e-6)], index=rv.index[:1200])
    neg.iloc[::7] = 3.0
    h = har_forecast(neg, pd.Series(True, index=neg.index), refit_every=21, min_train=730)
    fc = h.forecast.dropna()
    assert (fc > 0).all()
    assert h.meta["n_floored"] > 0
    assert h.meta["n_floored"] == int((h.raw.dropna() <= 0).sum())
