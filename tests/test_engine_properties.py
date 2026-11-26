"""P2 engine verification: property-based invariants + agreement with an independent reference.

The single-asset (`astra.backtest.engine`) and portfolio (`astra.portfolio.backtest`) engines are checked
on random price paths, targets and cost levels (hypothesis), not on a handful of hand-picked cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from astra.backtest.engine import BacktestConfig, run_backtest, run_gross, signal_strategy
from astra.backtest.reference import reference_equity
from astra.common.config import CostConfig
from astra.portfolio.backtest import MarketPanel, PortfolioBacktestConfig, run_portfolio_backtest

SETTINGS = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def price_paths(draw, n_assets: int = 1, min_len: int = 8, max_len: int = 60):
    n = draw(st.integers(min_len, max_len))
    seed = draw(st.integers(0, 2**31 - 1))
    vol = draw(st.floats(0.002, 0.08))
    rng = np.random.default_rng(seed)
    close = 50 * np.exp(np.cumsum(rng.normal(0, vol, (n, n_assets)), axis=0))
    open_ = close * np.exp(rng.normal(0, vol / 2, (n, n_assets)))
    volume = rng.uniform(1e3, 1e6, (n, n_assets))
    dates = pd.bdate_range("2021-01-04", periods=n)
    return dates, open_, close, volume, rng


def _bars(dates, o, c, v) -> pd.DataFrame:
    return pd.DataFrame({"session_date": dates, "open": o[:, 0], "close": c[:, 0], "volume": v[:, 0]})


def _cfg(**kw) -> BacktestConfig:
    base = {"max_participation": 1e12, "rebalance_tolerance": 0.0, "min_trade_weight": 0.0}
    return BacktestConfig(**{**base, **kw})


def _costs(bps: float) -> CostConfig:
    return CostConfig(commission=bps)


# ---------------------------------------------------------------- single-asset engine


@SETTINGS
@given(price_paths(), st.booleans())
def test_frictionless_engine_matches_reference(data, allow_short):
    dates, o, c, v, rng = data
    lo = -1.0 if allow_short else 0.0
    w = rng.uniform(lo, 1.0, len(dates))
    bars = _bars(dates, o, c, v)
    res = run_gross(bars, signal_strategy(pd.Series(w, index=dates)), _cfg(allow_short=allow_short))
    ref = reference_equity(o[:, 0], c[:, 0], w, 1_000_000.0)
    np.testing.assert_allclose(res.equity_net.to_numpy(), ref, rtol=1e-9)


@SETTINGS
@given(price_paths(), st.floats(0.0, 60.0), st.floats(0.001, 1.0), st.floats(0.0, 0.05))
def test_ledger_invariants(data, bps, participation, tolerance):
    dates, o, c, v, rng = data
    w = rng.uniform(-0.5, 1.0, len(dates))  # negatives must be clipped (long-only)
    bars = _bars(dates, o, c, v)
    cfg = _cfg(costs=_costs(bps), max_participation=participation, rebalance_tolerance=tolerance)
    res = run_backtest(bars, signal_strategy(pd.Series(w, index=dates)), cfg)
    L = res.ledger
    # equity = cash + position x close
    np.testing.assert_allclose(L["equity_net"], L["cash"] + L["position"] * c[:, 0], rtol=1e-10)
    # long-only, cash never borrowed (buys are sized net of their own cost)
    assert (L["position"] >= -1e-12).all()
    assert (L["cash"] >= -1e-6).all()
    # costs: non-negative, exactly bps x notional, and the ledger sums the fills
    assert all(f.cost >= 0 for f in res.fills)
    for f in res.fills:
        assert f.cost == pytest.approx(f.notional * bps / 1e4, rel=1e-12, abs=1e-12)
        vol_today = float(bars.loc[bars["session_date"] == f.session_date, "volume"].iloc[0])
        assert abs(f.quantity) <= participation * vol_today * (1 + 1e-12)
        assert f.price == float(bars.loc[bars["session_date"] == f.session_date, "open"].iloc[0])
    assert L["cost"].sum() == pytest.approx(sum(f.cost for f in res.fills), rel=1e-12)
    # nothing can fill on the first bar: no decision precedes it
    assert all(f.session_date > dates[0] for f in res.fills)
    # cash reconciles to the fills: every traded notional and every cost leaves cash exactly once
    flows = pd.Series(0.0, index=L.index)
    for f in res.fills:
        flows[f.session_date] += f.quantity * f.price + f.cost
    np.testing.assert_allclose(L["cash"], cfg.initial_cash - flows.cumsum(), rtol=1e-10, atol=1e-6)


@SETTINGS
@given(price_paths(min_len=12), st.floats(0.0, 30.0), st.data())
def test_no_look_ahead_truncation_invariance(data, bps, d):
    """The first k ledger rows must not depend on bars after k (prices, volumes, or signals)."""
    dates, o, c, v, rng = data
    k = d.draw(st.integers(2, len(dates) - 1))
    w = pd.Series(rng.uniform(0, 1, len(dates)), index=dates)
    bars = _bars(dates, o, c, v)
    cfg = _cfg(costs=_costs(bps), max_participation=0.3, impact_coef=0.1)
    full = run_backtest(bars, signal_strategy(w), cfg).ledger
    # scramble the future
    fut = bars.copy()
    fut.loc[k:, ["open", "close"]] *= rng.uniform(0.5, 2.0, (len(bars) - k, 1))
    fut.loc[k:, "volume"] = rng.uniform(1.0, 10.0, len(bars) - k)
    w2 = w.copy()
    w2.iloc[k:] = rng.uniform(0, 1, len(w) - k)
    part = run_backtest(fut, signal_strategy(w2), cfg).ledger
    pd.testing.assert_frame_equal(full.iloc[:k], part.iloc[:k])


@SETTINGS
@given(price_paths(), st.floats(0.5, 60.0))
def test_costs_only_reduce_buy_and_hold(data, bps):
    dates, o, c, v, _ = data
    bars = _bars(dates, o, c, v)
    sig = signal_strategy(pd.Series(1.0, index=dates))
    net = run_backtest(bars, sig, _cfg(costs=_costs(bps))).equity_net
    gross = run_gross(bars, sig, _cfg(costs=_costs(bps))).equity_net
    assert (net <= gross * (1 + 1e-12)).all()
    assert run_gross(bars, sig, _cfg(costs=_costs(bps))).ledger["cost"].sum() == 0.0


def test_zero_volume_bar_never_fills():
    dates = pd.bdate_range("2021-01-04", periods=6)
    bars = pd.DataFrame(
        {"session_date": dates, "open": 10.0, "close": 10.0, "volume": [1e6, 0, 0, 1e6, 1e6, 1e6]}
    )
    res = run_backtest(
        bars, signal_strategy(pd.Series([1.0, None, None, None, None, None], index=dates)), _cfg()
    )
    assert [f.session_date for f in res.fills] == [dates[3]]  # pending survives the halt, fills at reopen


# ---------------------------------------------------------------- portfolio engine


def _panel(dates, o, c, v) -> MarketPanel:
    cols = [f"A{j}" for j in range(o.shape[1])]
    mk = lambda x: pd.DataFrame(x, index=dates, columns=cols)  # noqa: E731
    return MarketPanel("TEST", mk(o), mk(c), mk(v))


def _weights(rng, n_rows, n_assets, invest_all: bool) -> np.ndarray:
    raw = rng.dirichlet(np.ones(n_assets), n_rows)
    if invest_all:
        return raw
    return raw * rng.uniform(0, 1, (n_rows, 1))


@SETTINGS
@given(st.integers(2, 6).flatmap(lambda n: price_paths(n_assets=n)))
def test_portfolio_frictionless_matches_reference(data):
    dates, o, c, v, rng = data
    w = _weights(rng, len(dates), o.shape[1], invest_all=False)
    panel = _panel(dates, o, c, v)
    targets = pd.DataFrame(w, index=dates, columns=panel.symbols)
    cfg = PortfolioBacktestConfig(costs=_costs(0.0), max_participation=1e12, min_trade_weight=0.0)
    res = run_portfolio_backtest(panel, targets, cfg)
    ref = reference_equity(o, c, w, cfg.initial_cash)
    np.testing.assert_allclose(res.ledger["equity_net"].to_numpy(), ref, rtol=1e-9)


@SETTINGS
@given(
    st.integers(2, 6).flatmap(lambda n: price_paths(n_assets=n)),
    st.floats(0.0, 60.0),
    st.floats(0.001, 1.0),
)
def test_portfolio_invariants(data, bps, participation):
    dates, o, c, v, rng = data
    w = _weights(rng, len(dates), o.shape[1], invest_all=True)
    panel = _panel(dates, o, c, v)
    targets = pd.DataFrame(w, index=dates, columns=panel.symbols)
    cfg = PortfolioBacktestConfig(costs=_costs(bps), max_participation=participation)
    res = run_portfolio_backtest(panel, targets, cfg)
    L = res.ledger
    marked = (res.positions * panel.close).sum(axis=1)
    np.testing.assert_allclose(L["equity_net"], L["cash"] + marked, rtol=1e-10)
    assert (res.positions.to_numpy() >= -1e-12).all()
    assert (L["cash"] >= -1e-6).all()
    assert (L["cost"] >= 0).all()
    for f in res.fills:
        assert abs(f.quantity) <= participation * panel.volume.loc[f.session_date, f.symbol] * (1 + 1e-12)
        assert f.session_date > dates[0]
    flows = pd.Series(0.0, index=L.index)
    for f in res.fills:
        flows[f.session_date] += f.quantity * f.price + f.cost
    np.testing.assert_allclose(L["cash"], cfg.initial_cash - flows.cumsum(), rtol=1e-10, atol=1e-6)


@SETTINGS
@given(st.integers(2, 6).flatmap(lambda n: price_paths(n_assets=n)), st.floats(0.0, 50.0))
def test_portfolio_stays_fully_invested_at_zero_tolerance(data, bps):
    """Regression guard for the 2026-11-25 cash-drag bug: with tolerance 0 and no volume cap, a
    fully-invested target leaves at most the cost-sizing haircut in cash after each fill."""
    dates, o, c, v, rng = data
    w = _weights(rng, len(dates), o.shape[1], invest_all=True)
    panel = _panel(dates, o, c, v)
    cfg = PortfolioBacktestConfig(costs=_costs(bps), max_participation=1e12)
    res = run_portfolio_backtest(panel, pd.DataFrame(w, index=dates, columns=panel.symbols), cfg)
    # exposure at the close after a fill at the open; cash share <= buy cost haircut + intraday drift
    cash_share = (res.ledger["cash"] / res.ledger["equity_net"]).iloc[1:]
    intraday = np.abs(c / o - 1).max(axis=1)[1:]
    bound = bps / 1e4 + 2 * intraday + 1e-9
    assert (cash_share.to_numpy() <= bound * (1 + bound) + bps / 1e4).all()


@SETTINGS
@given(st.integers(2, 5).flatmap(lambda n: price_paths(n_assets=n, min_len=12)), st.data())
def test_portfolio_truncation_invariance(data, d):
    dates, o, c, v, rng = data
    k = d.draw(st.integers(2, len(dates) - 1))
    w = _weights(rng, len(dates), o.shape[1], invest_all=False)
    cfg = PortfolioBacktestConfig(costs=_costs(10.0), max_participation=0.2)
    cols = [f"A{j}" for j in range(o.shape[1])]
    full = run_portfolio_backtest(
        _panel(dates, o, c, v), pd.DataFrame(w, index=dates, columns=cols), cfg
    ).ledger
    o2, c2, v2, w2 = o.copy(), c.copy(), v.copy(), w.copy()
    o2[k:] *= 1.7
    c2[k:] *= 0.6
    v2[k:] = 5.0
    w2[k:] = _weights(rng, len(dates) - k, o.shape[1], invest_all=True)
    part = run_portfolio_backtest(
        _panel(dates, o2, c2, v2), pd.DataFrame(w2, index=dates, columns=cols), cfg
    ).ledger
    pd.testing.assert_frame_equal(full.iloc[:k], part.iloc[:k])
