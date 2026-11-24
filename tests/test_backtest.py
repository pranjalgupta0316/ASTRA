from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nebula.backtest.audit import cost_sensitivity, leakage_canary
from nebula.backtest.engine import BacktestConfig, buy_and_hold, run_backtest, run_gross, signal_strategy
from nebula.common.config import CostConfig
from nebula.features.technical import build_features


def toy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
            "open": [100.0, 110.0, 121.0, 100.0],
            "close": [105.0, 120.0, 110.0, 90.0],
            "volume": [1e9] * 4,
        }
    )


def test_hand_computed_pnl_no_costs():
    # Decide at close day0 -> buy at open day1 (110). Hold. Equity close day3 = qty * 90.
    res = run_backtest(toy(), buy_and_hold(), BacktestConfig(initial_cash=1100.0))
    qty = 1100.0 / 110.0
    assert res.ledger["position"].iloc[0] == 0.0  # nothing fills on the decision bar
    assert res.ledger["position"].iloc[1] == pytest.approx(qty)
    assert res.equity_net.iloc[-1] == pytest.approx(qty * 90.0)
    assert len(res.fills) == 1 and res.fills[0].price == 110.0


def test_costs_hand_computed():
    costs = CostConfig(commission=10.0, half_spread=5.0, slippage=5.0)  # 20 bps per side
    cfg = BacktestConfig(initial_cash=1100.0, costs=costs)
    res = run_backtest(toy(), buy_and_hold(), cfg)
    # buy sized net of cost: qty * 110 * (1 + 0.002) = 1100  ->  qty = 10 / 1.002, cash ends at 0
    qty = 10 / 1.002
    assert res.fills[0].quantity == pytest.approx(qty)
    assert res.fills[0].cost == pytest.approx(qty * 110 * 0.002)
    assert res.ledger["cash"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert res.equity_net.iloc[-1] == pytest.approx(qty * 90.0)
    assert run_gross(toy(), buy_and_hold(), cfg).equity_net.iloc[-1] == pytest.approx(10 * 90.0)


def test_cash_never_negative_and_no_drift_trades(us_bars):
    """Constant 100% target with costs: one entry fill, cash >= 0 throughout (audit defect 4)."""
    bt = _bt(us_bars)
    sig = pd.Series(1.0, index=pd.to_datetime(bt["session_date"]))
    res = run_backtest(bt, signal_strategy(sig), BacktestConfig(costs=CostConfig(commission=10.0)))
    assert len(res.fills) == 1
    assert (res.ledger["cash"] >= -1e-6).all()


def test_signal_executes_next_open_not_same_bar():
    sig = pd.Series([1.0, 0.0, None, None], index=toy()["session_date"])
    res = run_backtest(toy(), signal_strategy(sig), BacktestConfig(initial_cash=1100.0))
    # buy day1 open 110 (10 sh), sell day2 open 121 -> cash 1210, flat afterwards
    assert res.ledger["position"].iloc[2] == 0.0
    assert res.equity_net.iloc[-1] == pytest.approx(1210.0)


def test_participation_cap():
    bars = toy().assign(volume=[100.0] * 4)
    res = run_backtest(bars, buy_and_hold(), BacktestConfig(initial_cash=1100.0, max_participation=0.05))
    assert res.fills[0].quantity == pytest.approx(5.0) and res.fills[0].capped


def test_zero_volume_bar_does_not_fill():
    """Provider placeholder rows (O=C, V=0) seen for RELIANCE.NS must not produce fills."""
    bars = toy().assign(volume=[1e9, 0.0, 1e9, 1e9])
    res = run_backtest(bars, buy_and_hold(), BacktestConfig(initial_cash=1210.0))
    assert res.ledger["position"].iloc[1] == 0.0
    assert res.fills[0].session_date == pd.Timestamp("2024-01-04") and res.fills[0].price == 121.0


def test_long_only_clips_short_targets():
    sig = pd.Series([-1.0, None, None, None], index=toy()["session_date"])
    res = run_backtest(toy(), signal_strategy(sig), BacktestConfig(initial_cash=1000.0))
    assert (res.ledger["position"] == 0).all()


def _bt(us_bars):
    f = build_features(us_bars)
    return pd.DataFrame(
        {
            "session_date": f["session_date"],
            "open": f["adj_open"],
            "close": f["adj_close"],
            "volume": f["volume"] * 1e3,
        }
    )


def test_leakage_canary_detects_future_information(us_bars):
    out = leakage_canary(_bt(us_bars), 252, BacktestConfig())
    assert out["passed"], out


def test_cost_sensitivity_monotone(us_bars):
    bt = _bt(us_bars)
    sig = pd.Series(
        np.tile([1.0, 0.0], len(bt) // 2 + 1)[: len(bt)], index=pd.to_datetime(bt["session_date"])
    )
    table = cost_sensitivity(bt, sig, 252, BacktestConfig(costs=CostConfig(commission=5.0)))
    assert table["total_cost"].iloc[0] == 0.0
    assert table["cagr"].is_monotonic_decreasing
