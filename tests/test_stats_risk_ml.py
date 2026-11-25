from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from astra.ml.classical import classification_metrics, logistic_factory, walk_forward_classifier
from astra.ml.validation import PurgedWalkForward
from astra.risk.metrics import cagr, expected_shortfall, max_drawdown, sortino, var_historical
from astra.stats.inference import (
    deflated_sharpe_ratio,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    sharpe,
)
from astra.stats.tests import ljung_box, stationarity


def test_sharpe_annualization():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.005])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert sharpe(r, 252) == pytest.approx(expected)
    assert sharpe(r, 365) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(365))


def test_cagr_and_mdd_known():
    eq = pd.Series([100.0, 120.0, 90.0, 121.0])
    assert max_drawdown(eq) == pytest.approx(90 / 120 - 1)
    assert cagr(eq, periods_per_year=3) == pytest.approx(0.21)  # 3 periods = 1 year


def test_var_es_ordering():
    r = pd.Series(np.random.default_rng(1).standard_t(4, 5000) * 0.01)
    v, es = var_historical(r, 0.95), expected_shortfall(r, 0.95)
    assert v > 0 and es >= v


def test_sortino_ignores_upside():
    assert np.isnan(sortino(pd.Series([0.01, 0.02, 0.03]), 252))


def test_stationarity_verdicts():
    rng = np.random.default_rng(0)
    rw = pd.Series(np.cumsum(rng.normal(size=2000)))
    wn = pd.Series(rng.normal(size=2000))
    assert stationarity(rw)["verdict"] == "unit_root"
    assert stationarity(wn)["verdict"] == "stationary"


def test_ljung_box_detects_autocorrelation():
    rng = np.random.default_rng(0)
    e = rng.normal(size=3000)
    ar = pd.Series(np.convolve(e, [1, 0.5], mode="same"))
    assert ljung_box(ar)["p_value"] < 0.01
    assert ljung_box(pd.Series(e))["p_value"] > 0.01


def test_newey_west_matches_ols_without_autocorrelation():
    x = pd.Series(np.random.default_rng(3).normal(0.1, 1, 5000))
    t_nw = newey_west_tstat(x, lags=0)
    t_ols = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert t_nw == pytest.approx(t_ols, rel=1e-9)


def test_psr_and_dsr_behaviour():
    r = pd.Series(np.random.default_rng(5).normal(0.001, 0.01, 1000))
    psr = probabilistic_sharpe_ratio(r)
    assert 0.5 < psr <= 1.0
    assert deflated_sharpe_ratio(r, n_trials=100) < deflated_sharpe_ratio(r, n_trials=1)


def test_sharpe_difference_bootstrap():
    from astra.stats.inference import sharpe_difference_bootstrap

    rng = np.random.default_rng(11)
    a = pd.Series(rng.normal(0.0005, 0.01, 1500))
    same = sharpe_difference_bootstrap(a, a, 252, n_boot=300)
    assert same["delta_sr"] == pytest.approx(0.0) and same["ci_low"] <= 0 <= same["ci_high"]
    better = sharpe_difference_bootstrap(a + 0.002, a, 252, n_boot=300)
    assert better["delta_sr"] > 0 and better["ci_low"] > 0 and better["p_two_sided"] < 0.01


def test_purged_walk_forward_no_overlap():
    cv = PurgedWalkForward(min_train=100, test_size=20, label_horizon=2, embargo=3)
    splits = list(cv.split(200))
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.max() + 2 + 3 < te.min() + 1  # gap of label_horizon + embargo
        assert tr.max() < te.min()
    rolling = PurgedWalkForward(min_train=100, test_size=20, expanding=False)
    for tr, _ in rolling.split(200):
        assert len(tr) <= 100


def test_walk_forward_classifier_no_leakage_on_noise():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=800, freq="D")
    X = pd.DataFrame(rng.normal(size=(800, 5)), index=idx)
    y = pd.Series((rng.random(800) > 0.5).astype(float), index=idx)
    res = walk_forward_classifier(X, y, logistic_factory(), PurgedWalkForward(300, 100, 1, 0))
    m = classification_metrics(y, res.proba)
    assert res.proba.iloc[:300].isna().all()
    assert 0.4 < m["auc"] < 0.6  # pure noise -> no skill


def test_walk_forward_classifier_detects_leaky_feature():
    """Canary: a feature equal to the label must give near-perfect AUC (pipeline is wired correctly)."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=600, freq="D")
    y = pd.Series((rng.random(600) > 0.5).astype(float), index=idx)
    X = pd.DataFrame({"leak": y + rng.normal(0, 0.01, 600), "noise": rng.normal(size=600)}, index=idx)
    res = walk_forward_classifier(X, y, logistic_factory(c=10.0), PurgedWalkForward(200, 100))
    assert classification_metrics(y, res.proba)["auc"] > 0.99
