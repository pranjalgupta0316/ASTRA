"""H-VT-001 / H-XMKT-001 building blocks (offline, synthetic data)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nebula.research.cross_market import (
    fx_asof,
    inverse_vol_weights,
    monthly_weeks,
    simulate_weekly,
    to_usd,
    weekly_last,
    weekly_usd_sleeves,
)
from nebula.research.vol_managed import (
    calibrate,
    ewma_sigma,
    managed_ew_targets,
    managed_weight,
)


def gbm(n: int, seed: int = 0, start: str = "2010-01-01", freq: str = "B", vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, vol, n))), index=idx)


# ------------------------------------------------------------------------------------- H-VT-001
def test_weight_formula_and_cap():
    sig = pd.Series([0.01, 0.02, 0.04, 0.005, np.nan, 0.0])
    w = managed_weight(sig, 0.02)
    assert w.iloc[0] == 1.0  # sigma below target -> capped at 1 (no leverage)
    assert w.iloc[1] == 1.0
    assert w.iloc[2] == pytest.approx(0.5)
    assert w.iloc[3] == 1.0
    assert math.isnan(w.iloc[4])
    assert math.isnan(w.iloc[5])  # zero vol is not a usable estimate
    assert (w.dropna() <= 1.0).all() and (w.dropna() > 0).all()


def test_ewma_sigma_matches_recursion_and_is_causal():
    c = gbm(400, seed=1)
    s = ewma_sigma(c, lam=0.94, min_periods=1)
    r = np.log(c / c.shift(1)).dropna().to_numpy()
    h = r[0] ** 2
    manual = [h]
    for x in r[1:]:
        h = 0.94 * h + 0.06 * x**2
        manual.append(h)
    np.testing.assert_allclose(s.dropna().to_numpy(), np.sqrt(manual), rtol=1e-12)
    # truncation invariance: values known at t never change when later data arrive or change
    full = ewma_sigma(c)
    for cut in (50, 200, 399):
        part = ewma_sigma(c.iloc[:cut])
        pd.testing.assert_series_equal(part, full.iloc[:cut])
    shocked = c.copy()
    shocked.iloc[300:] *= 3.0
    pd.testing.assert_series_equal(ewma_sigma(shocked).iloc[:300], full.iloc[:300])


def test_calibration_window_excluded_and_frozen():
    c = gbm(2000, seed=2)  # business days from 2010-01-01 (~7.7 years)
    sig = ewma_sigma(c)
    cal = calibrate(sig, years=3)
    cutoff = pd.Timestamp("2013-01-01")
    assert cal.calib_start == c.index[0]
    assert cal.calib_end < cutoff <= cal.eval_start
    assert cal.calib_end == c.index[c.index < cutoff][-1]
    assert cal.eval_start == c.index[c.index >= cutoff][0]
    assert cal.sigma_target == pytest.approx(float(sig[sig.index < cutoff].median()))
    # frozen: anything after the calibration window (crash, extension) leaves the target unchanged
    shocked = c.copy()
    shocked.loc[shocked.index >= cutoff] *= np.exp(
        np.random.default_rng(9).normal(0, 0.1, (c.index >= cutoff).sum())
    )
    assert calibrate(ewma_sigma(shocked)).sigma_target == cal.sigma_target
    assert calibrate(ewma_sigma(c.iloc[:900])).sigma_target == cal.sigma_target
    with pytest.raises(ValueError):
        calibrate(sig.iloc[:500])  # shorter than the calibration window


def test_managed_ew_targets_tolerance_and_composition():
    idx = pd.bdate_range("2024-01-29", "2024-02-09")
    scale = pd.Series([1.0, 0.99, 0.95, 0.94, 0.90, 0.905, 0.80, 0.80, 0.80, 0.70], index=idx)
    ew = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=pd.to_datetime(["2024-01-02", "2024-02-01"]))
    drift = pd.DataFrame({"A": 0.6, "B": 0.3}, index=idx)  # unmanaged drifted weights (sum 0.9)
    tg = managed_ew_targets(scale, ew, idx[0], tol=0.02, drift=drift)
    # issued: start 01-29; 01-31 (0.95 vs 1.0 > tol); 02-01 monthly EW date; 02-02 (0.90 vs 0.94);
    # 02-06 (0.80); 02-09 (0.70). Skipped: 01-30 (0.99), 02-05 (0.905 vs 0.90), 02-07/08 (unchanged).
    expected = ["2024-01-29", "2024-01-31", "2024-02-01", "2024-02-02", "2024-02-06", "2024-02-09"]
    assert list(tg.index) == list(pd.to_datetime(expected))
    assert tg.loc["2024-01-29"].tolist() == pytest.approx([0.5, 0.5])  # start uses the EW row
    assert tg.loc["2024-01-31"].tolist() == pytest.approx([0.95 * 2 / 3, 0.95 / 3])  # drifted composition
    assert tg.loc["2024-02-01"].tolist() == pytest.approx([0.47, 0.47])  # monthly -> EW row * scale
    assert (tg.sum(axis=1) <= 1 + 1e-12).all()


# ------------------------------------------------------------------------------------- H-XMKT-001
def test_fx_asof_never_uses_value_before_available():
    fx = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "value": [80.0, 81.0, 82.0, 83.0],
            "available_from": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-05"]),
        }
    )
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"])
    strict = fx_asof(dates, fx, strict=True)
    assert math.isnan(strict["fx"].iloc[0]) and math.isnan(strict["fx"].iloc[1])
    assert strict["fx"].tolist()[2:] == [80.0, 81.0, 83.0]  # ties on availability -> latest obs date
    ok = strict["fx_available_from"].notna()
    assert (strict.loc[ok, "fx_available_from"].to_numpy() < dates[ok.to_numpy()].to_numpy()).all()
    loose = fx_asof(dates, fx, strict=False)
    assert loose["fx"].tolist()[1:] == [80.0, 81.0, 83.0, 83.0]
    assert (loose["fx_available_from"].dropna().to_numpy() <= dates[1:].to_numpy()).all()


def test_weekly_alignment_last_session_per_market():
    # exchange series: Fri 2024-03-29 is a holiday (Good Friday) -> Thursday close is that week's value
    sess = pd.bdate_range("2024-03-18", "2024-04-12").drop(pd.Timestamp("2024-03-29"))
    ex = pd.Series(np.arange(len(sess), dtype=float), index=sess)
    w = weekly_last(ex)
    assert list(w.index) == list(pd.to_datetime(["2024-03-22", "2024-03-29", "2024-04-05", "2024-04-12"]))
    assert w.loc["2024-03-29", "session_date"] == pd.Timestamp("2024-03-28")
    assert w.loc["2024-03-29", "value"] == ex.loc["2024-03-28"]
    # crypto 24/7: Saturday/Sunday bars belong to the NEXT week; the Friday bar closes the week
    days = pd.date_range("2024-03-16", "2024-03-31", freq="D")
    cr = pd.Series(np.arange(len(days), dtype=float), index=days)
    wc = weekly_last(cr)
    assert wc.loc["2024-03-22", "session_date"] == pd.Timestamp("2024-03-22")
    assert wc.loc["2024-03-29", "session_date"] == pd.Timestamp("2024-03-29")
    # a week without any session keeps the last value (return 0), same market only
    gap = ex.drop(ex.loc["2024-04-01":"2024-04-05"].index)
    wg = weekly_last(gap)
    assert wg.loc["2024-04-05", "value"] == wg.loc["2024-03-29", "value"]
    assert wg.loc["2024-04-05", "session_date"] == pd.Timestamp("2024-03-28")


def test_sleeve_conversion_arithmetic():
    assert to_usd(pd.Series([8300.0]), pd.Series([83.0])).iloc[0] == pytest.approx(100.0)
    sess = pd.bdate_range("2024-01-01", "2024-01-31")
    inr = pd.Series(1000.0, index=sess)
    inr.loc["2024-01-15":] = 1100.0  # +10 % in INR in week ending 01-19
    fx_dates = pd.bdate_range("2023-12-20", "2024-01-31")
    fx = pd.DataFrame(
        {
            "date": fx_dates,
            "value": np.where(fx_dates < pd.Timestamp("2024-01-10"), 80.0, 88.0),
            "available_from": fx_dates + pd.offsets.BDay(2),
        }
    )
    usd, audit = weekly_usd_sleeves({"IN": inr, "US": pd.Series(1.0, index=sess)}, fx)
    r = usd["IN"].pct_change()
    # week ending 01-12: FX obs of 01-10 available 01-12 -> not usable at the 01-12 IN close (strict);
    # the 88 rate first applies in the week ending 01-19, together with the +10 % INR move.
    assert audit.loc["2024-01-12", "IN_fx"] == 80.0
    assert r.loc["2024-01-12"] == pytest.approx(0.0)
    assert r.loc["2024-01-19"] == pytest.approx(1.10 * 80.0 / 88.0 - 1.0)
    assert (
        audit["IN_fx_available_from"].dropna()
        < audit.loc[audit["IN_fx_available_from"].notna(), "IN_session_date"]
    ).all()


def test_inverse_vol_weights_analytic():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 60)
    idx = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    rets = pd.DataFrame({"A": 0.01 * z, "B": 0.02 * z, "C": 0.04 * z}, index=idx)
    w = inverse_vol_weights(rets, idx[-1], window=52)
    assert w is not None
    inv = np.array([1 / 0.01, 1 / 0.02, 1 / 0.04])
    np.testing.assert_allclose(w.to_numpy(), inv / inv.sum(), rtol=1e-12)
    assert inverse_vol_weights(rets, idx[50], window=52) is None  # not enough history


def test_monthly_weeks_first_label_of_month():
    idx = pd.date_range("2024-01-05", "2024-03-29", freq="W-FRI")
    assert list(monthly_weeks(idx)) == list(pd.to_datetime(["2024-02-02", "2024-03-01"]))


def test_simulate_weekly_hand_computed_with_lag_and_costs():
    idx = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
    rets = pd.DataFrame({"A": [0.0, 0.0, 0.10, 0.0], "B": [0.0, 0.0, -0.10, 0.0]}, index=idx)
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=idx[:1])
    buy = pd.Series({"A": 10.0, "B": 20.0})
    sell = pd.Series({"A": 10.0, "B": 20.0})
    out = simulate_weekly(rets, targets, buy, sell, exec_lag=1, initial=1.0)
    # decided week 0, executed at the week-1 close: buy 0.5 n of each, n = 1 - 0.5 n (0.001 + 0.002)
    n = 1.0 / (1 + 0.5 * 0.003)
    assert out.index[0] == idx[1]
    assert out["value"].iloc[0] == pytest.approx(n)
    assert out["cost"].iloc[0] == pytest.approx(1.0 - n)
    assert out["value"].iloc[1] == pytest.approx(n)  # week-2 returns 0
    assert out["value"].iloc[2] == pytest.approx(0.5 * n * 1.1 + 0.5 * n * 0.9)
    assert out["w_A"].iloc[2] == pytest.approx(0.55)  # drift, no rebalance
