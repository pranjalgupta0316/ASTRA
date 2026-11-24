from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebula.common.config import CostConfig
from nebula.portfolio.backtest import (
    MarketPanel,
    PortfolioBacktestConfig,
    monthly_decision_dates,
    run_portfolio_backtest,
    run_portfolio_gross,
)
from nebula.portfolio.construction import compute_targets, minvar_fn
from nebula.portfolio.covariance import (
    condition_number,
    estimate_covariance,
    is_psd,
    ledoit_wolf_cov,
    nearest_psd,
)
from nebula.portfolio.optimizers import (
    equal_weight,
    max_sharpe,
    mean_variance,
    min_variance,
    risk_parity,
    vol_target,
)
from nebula.risk.attribution import (
    effective_n,
    effective_number_of_bets,
    hhi,
    paired_bootstrap_ci,
    risk_contributions,
    var_es,
)


def random_cov(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    vols = rng.uniform(0.01, 0.03, n)
    corr = a @ a.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    corr = 0.5 * corr + 0.5 * np.eye(n)
    return corr * np.outer(vols, vols)


def random_returns(t: int, n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cov = random_cov(n, seed)
    x = rng.multivariate_normal(np.full(n, 3e-4), cov, size=t)
    idx = pd.bdate_range("2020-01-01", periods=t)
    return pd.DataFrame(x, index=idx, columns=[f"A{i}" for i in range(n)])


# ------------------------------------------------------------------------------- optimizers


def test_two_asset_min_variance_analytic():
    s1, s2, rho = 0.02, 0.03, 0.3
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w1 = (s2**2 - rho * s1 * s2) / (s1**2 + s2**2 - 2 * rho * s1 * s2)
    w = min_variance(cov)
    assert w[0] == pytest.approx(w1, abs=1e-6)
    assert w[1] == pytest.approx(1 - w1, abs=1e-6)


def test_two_asset_min_variance_cap_binds():
    cov = np.diag([0.01**2, 0.04**2])  # unconstrained w1 = 16/17 > 0.6
    w = min_variance(cov, caps=0.6)
    np.testing.assert_allclose(w, [0.6, 0.4], atol=1e-7)


def test_risk_parity_equal_risk_contributions():
    cov = random_cov(5, seed=3)
    w = risk_parity(cov)
    rc = risk_contributions(w, cov)
    assert np.max(np.abs(rc["pct"].to_numpy() - 0.2)) < 1e-6
    assert rc["rc"].sum() == pytest.approx(np.sqrt(w @ cov @ w), rel=1e-12)
    assert effective_number_of_bets(w, cov) == pytest.approx(5.0, abs=1e-5)


def test_risk_parity_two_asset_inverse_vol():
    # with two assets ERC weights are inverse-vol, independent of correlation
    s1, s2, rho = 0.01, 0.03, 0.4
    cov = np.array([[s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2]])
    w = risk_parity(cov)
    np.testing.assert_allclose(w, [0.75, 0.25], atol=1e-8)


@pytest.mark.parametrize("cap", [0.4, 0.6])
def test_weights_sum_to_one_and_respect_caps(cap: float):
    cov = random_cov(5, seed=11)
    mu = np.array([5e-4, 1e-4, -2e-4, 8e-4, 3e-4])
    outs = [
        equal_weight(5, cap),
        min_variance(cov, cap),
        risk_parity(cov, caps=cap),
        mean_variance(mu, cov, 5.0, cap),
        max_sharpe(mu, cov, cap)[0],
    ]
    for w in outs:
        assert w.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(w >= -1e-12)
        assert np.all(w <= cap + 1e-8)


def test_max_sharpe_matches_tangency_unconstrained():
    # Uncapped tangency w ∝ S^-1 mu when that is long-only
    cov = np.array([[0.04, 0.006], [0.006, 0.09]]) / 252
    mu = np.array([0.08, 0.10]) / 252
    w_true = np.linalg.solve(cov, mu)
    w_true /= w_true.sum()
    w, fb = max_sharpe(mu, cov)
    assert not fb
    np.testing.assert_allclose(w, w_true, atol=1e-6)


def test_max_sharpe_falls_back_when_no_positive_mean():
    cov = random_cov(3, seed=1)
    w, fb = max_sharpe(np.array([-1e-4, -2e-4, -3e-4]), cov)
    assert fb
    np.testing.assert_allclose(w, min_variance(cov), atol=1e-7)


def test_infeasible_caps_raise():
    with pytest.raises(ValueError):
        min_variance(np.eye(2), caps=0.4)


def test_vol_targeting_scales_correctly():
    cov = np.array([[0.02**2, 0.0], [0.0, 0.02**2]])
    w = np.array([0.5, 0.5])
    fv = np.sqrt(w @ cov @ w * 252)  # = 0.02 / sqrt(2) * sqrt(252)
    ws, scale = vol_target(w, cov, target_vol=0.10, periods_per_year=252)
    assert scale == pytest.approx(0.10 / fv)
    assert np.sqrt(ws @ cov @ ws * 252) == pytest.approx(0.10)
    # target above forecast -> capped at leverage 1 (no borrowing)
    ws2, scale2 = vol_target(w, cov, target_vol=5.0, periods_per_year=252)
    assert scale2 == 1.0
    np.testing.assert_allclose(ws2, w)


# ------------------------------------------------------------------------------- covariance


def test_ledoit_wolf_psd_and_condition():
    r = random_returns(60, 10, seed=2).to_numpy()
    cov, shrink = ledoit_wolf_cov(r)
    assert is_psd(cov)
    assert 0.0 <= shrink <= 1.0
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.isfinite(condition_number(cov))


def test_nearest_psd_repairs_indefinite_matrix():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    assert not is_psd(bad)
    fixed = nearest_psd(bad)
    assert is_psd(fixed)
    np.testing.assert_allclose(fixed, fixed.T)


def test_covariance_truncation_invariance():
    r = random_returns(400, 4, seed=5)
    t = r.index[300]
    full = estimate_covariance(r, t, window=252)
    trunc = estimate_covariance(r.loc[:t], t, window=252)
    np.testing.assert_array_equal(full.cov, trunc.cov)
    assert full.end == t
    # changing any data after t does not change the estimate
    r2 = r.copy()
    r2.loc[r2.index > t] = 0.5
    np.testing.assert_array_equal(estimate_covariance(r2, t, window=252).cov, full.cov)
    # but changing t's own row does (t's close is used)
    r3 = r.copy()
    r3.loc[t] = 0.1
    assert not np.allclose(estimate_covariance(r3, t, window=252).cov, full.cov)


def test_targets_truncation_invariance():
    r = random_returns(400, 3, seed=8)
    dates = monthly_decision_dates(r.index)
    tg_full, _ = compute_targets(r, dates, minvar_fn, cap=0.6, periods_per_year=252)
    cut = dates[-2]
    tg_cut, _ = compute_targets(r.loc[:cut], dates[dates <= cut], minvar_fn, cap=0.6, periods_per_year=252)
    pd.testing.assert_frame_equal(tg_full.loc[:cut], tg_cut)


def test_asset_enters_only_after_min_history():
    r = random_returns(600, 3, seed=9)
    r.iloc[:200, 2] = np.nan  # asset A2 lists at row 200
    dates = monthly_decision_dates(r.index)
    tg, diag = compute_targets(r, dates, minvar_fn, cap=0.6, periods_per_year=252)
    entry = r.index[200 + 251]  # 252 valid returns available at this date
    assert (tg.loc[tg.index < entry, "A2"] == 0).all()
    assert (tg.loc[tg.index >= entry, "A2"] > 0).all()
    assert (diag["n_assets"] >= 2).all()


# ------------------------------------------------------------------------------- backtester


def toy_panel() -> MarketPanel:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    o = pd.DataFrame({"A": [10.0, 10.0, 12.0], "B": [20.0, 20.0, 18.0]}, index=idx)
    c = pd.DataFrame({"A": [10.0, 11.0, 13.0], "B": [20.0, 19.0, 17.0]}, index=idx)
    v = pd.DataFrame({"A": [1e9] * 3, "B": [1e9] * 3}, index=idx)
    return MarketPanel("TEST", o, c, v)


def test_two_asset_three_bar_hand_computed_with_costs():
    panel = toy_panel()
    idx = panel.close.index
    targets = pd.DataFrame({"A": [0.5, 1.0], "B": [0.5, 0.0]}, index=idx[:2])
    costs = CostConfig(commission=10.0, sell_tax=10.0)  # buy 10 bps, sell 20 bps
    cfg = PortfolioBacktestConfig(
        initial_cash=1000.0, costs=costs, max_participation=float("inf"), rebalance_tolerance=0.0
    )
    res = run_portfolio_backtest(panel, targets, cfg)
    b, s = 0.001, 0.002
    # bar 0: decision only, nothing fills
    assert res.equity_net.iloc[0] == pytest.approx(1000.0)
    # bar 1 open: buy 500 of each, sized net of cost -> spend exactly 1000 incl. costs
    qa = 500 / 10 / (1 + b)
    qb = 500 / 20 / (1 + b)
    eq1 = qa * 11 + qb * 19
    assert res.positions.iloc[1].to_numpy() == pytest.approx([qa, qb])
    assert res.ledger["cash"].iloc[1] == pytest.approx(0.0, abs=1e-9)
    assert res.equity_net.iloc[1] == pytest.approx(eq1)
    assert res.ledger["cost"].iloc[1] == pytest.approx((qa * 10 + qb * 20) * b)
    # bar 2 open: target A=1, B=0. Sell all B at 18 (20 bps), then buy A with the proceeds;
    # wanted buy (equity_open - qa*12)/(1+b) is not affordable -> scaled to the cash available.
    proceeds = qb * 18 * (1 - s)
    qa_buy = proceeds / (12 * (1 + b))
    eq2 = (qa + qa_buy) * 13
    assert res.positions.iloc[2].to_numpy() == pytest.approx([qa + qa_buy, 0.0])
    assert res.ledger["cash"].iloc[2] == pytest.approx(0.0, abs=1e-9)
    assert res.ledger["cash"].min() >= 0.0
    assert res.equity_net.iloc[2] == pytest.approx(eq2)
    assert res.ledger["cost"].iloc[2] == pytest.approx(qb * 18 * s + qa_buy * 12 * b)
    # exact gross via cost_multiplier = 0
    g = run_portfolio_gross(panel, targets, cfg)
    ga, gb = 50.0, 25.0
    ga2 = ga + gb * 18 / 12
    assert g.equity_net.iloc[1] == pytest.approx(ga * 11 + gb * 19)
    assert g.equity_net.iloc[2] == pytest.approx(ga2 * 13)
    assert g.ledger["cost"].sum() == 0.0


def test_zero_volume_bar_never_fills_and_leg_stays_pending():
    panel = toy_panel()
    panel.volume.iloc[1, 1] = 0.0  # B untradable at bar 1 open
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=panel.close.index[:1])
    cfg = PortfolioBacktestConfig(
        initial_cash=1000.0, max_participation=float("inf"), rebalance_tolerance=0.0
    )
    res = run_portfolio_backtest(panel, targets, cfg)
    assert res.positions["B"].iloc[1] == 0.0
    assert res.positions["A"].iloc[1] == pytest.approx(50.0)
    # B fills at the next tradable open (bar 2, price 18), sized against equity at that open
    # (0.5 * 1100 = 550) but only 500 cash is available -> scaled down, cash never negative
    eq_open2 = 50 * 12 + 500
    assert 0.5 * eq_open2 > 500
    assert res.positions["B"].iloc[2] == pytest.approx(500 / 18)
    assert res.ledger["cash"].min() >= 0.0
    assert res.fills[-1].cash_scaled


def test_volume_cap_binds():
    panel = toy_panel()
    panel.volume.loc[:, :] = 100.0
    targets = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=panel.close.index[:1])
    cfg = PortfolioBacktestConfig(initial_cash=1000.0, max_participation=0.1, rebalance_tolerance=0.0)
    res = run_portfolio_backtest(panel, targets, cfg)
    assert res.positions.iloc[1].to_numpy() == pytest.approx([10.0, 10.0])
    assert all(f.capped for f in res.fills)


def test_missing_bar_marked_at_last_close():
    panel = toy_panel()
    panel.open.iloc[2, 0] = np.nan
    panel.close.iloc[2, 0] = np.nan
    targets = pd.DataFrame({"A": [1.0]}, index=panel.close.index[:1])
    cfg = PortfolioBacktestConfig(initial_cash=1000.0, max_participation=float("inf"))
    res = run_portfolio_backtest(panel, targets, cfg)
    assert res.equity_net.iloc[2] == pytest.approx(100 * 11)


def test_monthly_decision_dates_first_session():
    idx = pd.bdate_range("2024-01-29", "2024-03-05")
    d = monthly_decision_dates(idx)
    assert list(d) == [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01")]


# ------------------------------------------------------------------------------- risk


def test_concentration_measures():
    assert hhi(np.full(4, 0.25)) == pytest.approx(0.25)
    assert effective_n(np.full(4, 0.25)) == pytest.approx(4.0)
    assert effective_n(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["historical", "gaussian", "cornish_fisher"])
def test_var_es_ordering(method: str):
    rng = np.random.default_rng(0)
    r = pd.Series(rng.standard_t(5, size=5000) * 0.01)
    v95, e95 = var_es(r, 0.95, method)
    v99, e99 = var_es(r, 0.99, method)
    assert 0 < v95 < v99
    assert e95 >= v95
    assert e99 >= v99
    assert e99 > e95


def test_gaussian_var_known_value():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(size=1000))
    r = (r - r.mean()) / r.std(ddof=1) * 0.01  # exact mean 0, sd 0.01
    v, e = var_es(r, 0.95, "gaussian")
    assert v == pytest.approx(0.01 * 1.6448536269514722)
    assert e == pytest.approx(0.01 * 2.0627128075074257)
    # with zero skew / excess kurtosis Cornish-Fisher reduces to Gaussian up to sample moments
    vcf, _ = var_es(r, 0.95, "cornish_fisher")
    assert vcf == pytest.approx(v, rel=0.05)


def test_paired_bootstrap_ratio_brackets_point_estimate():
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(0, 0.005, 1000))
    b = pd.Series(rng.normal(0, 0.01, 1000))
    out = paired_bootstrap_ci(a, b, 252, n_boot=300, seed=1)
    assert out["vol_ratio_ci_lo"] < out["vol_ratio"] < out["vol_ratio_ci_hi"]
    assert out["vol_ratio_ci_hi"] < 1.0
