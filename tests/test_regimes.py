from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.stats.regimes import (
    fit_mapped_gmm,
    persistence,
    regime_features,
    state_difference_tests,
    walk_forward_gmm,
)


def markov_switching(
    n: int = 2000, p_stay: float = 0.99, vols: tuple[float, float] = (0.005, 0.03), seed: int = 0
):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    s = np.empty(n, dtype=int)
    s[0] = 0
    for t in range(1, n):
        s[t] = s[t - 1] if rng.random() < p_stay else 1 - s[t - 1]
    r = rng.standard_normal(n) * np.where(s == 1, vols[1], vols[0])
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.Series(r, index=idx), pd.Series(s, index=idx)


def test_regime_features_hand_example() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0])
    f = regime_features(r, window=3)
    assert f["ret20"].iloc[2] == pytest.approx(0.02)
    assert f["vol20"].iloc[2] == pytest.approx(np.std([0.01, -0.02, 0.03], ddof=1))
    assert f.iloc[:2].isna().all().all()


def test_persistence_hand_example() -> None:
    p = persistence(pd.Series([0, 0, 1, 1, 1, 0, np.nan]))
    assert p["persistence"] == pytest.approx(3 / 5)
    assert p["independence_benchmark"] == pytest.approx(0.5)


def test_label_mapping_stable_across_seeds() -> None:
    r, _ = markov_switching(1500, seed=1)
    x = regime_features(r).dropna().to_numpy()
    labels = []
    for seed in range(6):
        m = fit_mapped_gmm(x, n_states=2, seed=seed, n_init=1)
        lab = m.predict(x)
        # mapped state 1 is always the high-vol state
        assert x[lab == 1, 0].mean() > x[lab == 0, 0].mean()
        assert m.train_state_vol[1] > m.train_state_vol[0]
        labels.append(lab)
    agree = [np.mean(lab == labels[0]) for lab in labels[1:]]
    assert min(agree) > 0.99


def test_label_mapping_invariant_to_feature_scale() -> None:
    r, _ = markov_switching(1200, seed=2)
    x = regime_features(r).dropna().to_numpy()
    a = fit_mapped_gmm(x, seed=7).predict(x)
    b = fit_mapped_gmm(x * np.array([100.0, 0.01]), seed=7).predict(x * np.array([100.0, 0.01]))
    assert np.mean(a == b) > 0.99


def test_synthetic_regime_switching_detected_oos() -> None:
    r, s = markov_switching(2000, seed=4)
    feats = regime_features(r)
    rv_next = (r**2).shift(-1)
    wf = walk_forward_gmm(feats, min_train=504, refit_every=63, seed=7, target_next=rv_next)
    st = wf.states.dropna()
    assert st.index[0] == feats.index[503]
    acc = float(np.mean(st.to_numpy() == s.loc[st.index].to_numpy()))
    assert acc > 0.85
    pers = persistence(st)
    assert pers["persistence"] > 0.95 > pers["independence_benchmark"]
    tests = state_difference_tests(rv_next.loc[st.index], (st == 1).astype(float))
    assert tests["mw_p_one_sided"] < 1e-6
    assert tests["hac_t"] > 1.96
    assert tests["ratio_mean"] > 5
    # labels never switch: training-sample vol of state 1 exceeds state 0 in every refit
    assert (wf.refits["train_vol_state1"] > wf.refits["train_vol_state0"]).all()


def test_no_regime_series_gives_no_evidence() -> None:
    rng = np.random.default_rng(9)
    r = pd.Series(rng.standard_normal(1500) * 0.01, index=pd.bdate_range("2010-01-01", periods=1500))
    feats = regime_features(r)
    wf = walk_forward_gmm(feats, min_train=504, refit_every=63, seed=7)
    st = wf.states.dropna()
    tests = state_difference_tests((r**2).shift(-1).loc[st.index], (st == 1).astype(float))
    assert abs(tests["hac_t"]) < 3


def test_walk_forward_gmm_causal() -> None:
    r, _ = markov_switching(900, seed=6)
    feats = regime_features(r)
    rv_next = (r**2).shift(-1)
    full = walk_forward_gmm(feats, min_train=504, refit_every=63, target_next=rv_next)
    assert full.states.iloc[:503].isna().all()
    for cut in (520, 567, 700):
        part = walk_forward_gmm(
            feats.iloc[:cut], min_train=504, refit_every=63, target_next=rv_next.iloc[:cut]
        )
        pd.testing.assert_series_equal(part.states, full.states.iloc[:cut])
        assert part.cond_forecast is not None and full.cond_forecast is not None
        pd.testing.assert_series_equal(part.cond_forecast, full.cond_forecast.iloc[:cut])
    # the conditional forecast never uses the target realised after the refit origin
    r2 = r.copy()
    r2.iloc[600:] *= 5
    alt = walk_forward_gmm(regime_features(r2), min_train=504, refit_every=63, target_next=(r2**2).shift(-1))
    assert alt.cond_forecast is not None and full.cond_forecast is not None
    pd.testing.assert_series_equal(alt.cond_forecast.iloc[:599], full.cond_forecast.iloc[:599])


def test_state_difference_tests_direction() -> None:
    rng = np.random.default_rng(3)
    y = np.concatenate([rng.exponential(1.0, 300), rng.exponential(4.0, 300)])
    d = np.concatenate([np.zeros(300), np.ones(300)])
    t = state_difference_tests(pd.Series(y), pd.Series(d))
    assert t["mean_high"] > t["mean_low"] and t["welch_t"] > 0 and t["mw_p_one_sided"] < 1e-6
    flipped = state_difference_tests(pd.Series(y), pd.Series(1 - d))
    assert flipped["hac_t"] < -1.96 and flipped["mw_p_one_sided"] > 0.99
