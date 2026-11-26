"""P6 execution simulator: fill-model bounds, latency, cancels, taker sizing, fees, mark-outs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.execution import ExecConfig, Order, simulate

S = 1_000_000_000  # 1 s in ns


def _quotes(rows):
    return pd.DataFrame(rows, columns=["ts_ns", "bid", "bid_qty", "ask", "ask_qty"])


def _trades(rows):
    return pd.DataFrame(rows, columns=["ts_ns", "price", "qty", "is_buyer_maker"])


Q = _quotes(
    [(0, 100.0, 5.0, 100.1, 4.0), (10 * S, 100.0, 5.0, 100.1, 4.0), (100 * S, 100.2, 1.0, 100.3, 1.0)]
)
CFG = ExecConfig(latency_ns=S // 10, maker_fee_bps=2.0, taker_fee_bps=5.0)


def _o(models_df, oid, model):
    return models_df[(models_df["order_id"] == oid) & (models_df["model"] == model)].iloc[0]


def test_resting_buy_bounds_through_le_queue_le_touch():
    # seller-aggressor prints at our price 100.0: 3 then 4 units, then one print through at 99.9 (2 units)
    tr = _trades([(1 * S, 100.0, 3.0, True), (2 * S, 100.0, 4.0, True), (3 * S, 99.9, 2.0, True)])
    res = simulate(Q, tr, [Order("b", "buy", 6.0, submit_ns=0, limit_price=100.0)], CFG)
    o = res.orders
    through, queue, touch = (_o(o, "b", m)["filled_qty"] for m in ("through", "queue", "touch"))
    assert through == pytest.approx(2.0)  # only the print through the price
    # queue: 5 ahead at activation; prints at 100.0 eat 5 of 7 -> 2 for us, plus 2 through -> 4
    assert queue == pytest.approx(4.0) and _o(o, "b", "queue")["queue_ahead"] == 5.0
    assert touch == pytest.approx(6.0)
    assert through <= queue <= touch
    assert set(res.fills["liquidity"]) == {"maker"} and (res.fills["price"] == 100.0).all()


def test_buyer_aggressor_trades_never_fill_a_resting_buy():
    tr = _trades([(1 * S, 99.0, 50.0, False)])  # buyer-initiated (lifted an ask), even at a low price
    res = simulate(Q, tr, [Order("b", "buy", 1.0, submit_ns=0, limit_price=100.0)], CFG)
    assert (res.orders["filled_qty"] == 0).all()


def test_latency_and_cancel_window():
    tr = _trades([(S // 20, 99.0, 1.0, True), (5 * S, 99.0, 1.0, True), (20 * S, 99.0, 1.0, True)])
    o = Order("b", "buy", 5.0, submit_ns=0, limit_price=100.0, cancel_ns=10 * S)
    res = simulate(Q, tr, [o], CFG, models=("through",))
    # the 0.05 s print is before activation (0.1 s); the 20 s print is after the cancel took effect
    assert res.fills["ts_ns"].tolist() == [5 * S]


def test_market_order_fills_l1_then_flags_the_rest():
    res = simulate(Q, _trades([(1 * S, 100.1, 1.0, False)]), [Order("m", "buy", 6.0, submit_ns=0)], CFG)
    f = res.fills[res.fills["model"] == "through"]
    assert f["qty"].tolist() == [4.0, 2.0] and f["beyond_l1"].tolist() == [False, True]
    assert f["price"].iloc[0] == 100.1 and f["price"].iloc[1] == pytest.approx(100.1 * 1.0005)
    assert f["fee"].iloc[0] == pytest.approx(100.1 * 4 * 5e-4)
    # taker fills are identical under every model
    assert res.orders.groupby("order_id")["filled_qty"].nunique().iloc[0] == 1


def test_marketable_limit_takes_and_respects_its_limit():
    res = simulate(
        Q, _trades([(1 * S, 100.1, 1.0, False)]), [Order("l", "buy", 6.0, 0, limit_price=100.1)], CFG
    )
    f = res.fills[res.fills["model"] == "queue"]
    assert f["qty"].tolist() == [4.0] and (f["liquidity"] == "taker").all()  # beyond-L1 price > limit


def test_markouts_are_signed_and_never_interpolated():
    tr = _trades([(1 * S, 99.9, 1.0, True)])
    res = simulate(Q, tr, [Order("b", "buy", 1.0, 0, limit_price=100.0)], CFG, models=("through",))
    f = res.fills.iloc[0]
    # mid at 1 s+1 s = 2 s is the quote from t=0: 100.05 -> +5 bps for a buy at 100.0
    assert f["markout_1s_bps"] == pytest.approx((100.05 - 100.0) / 100.0 * 1e4)
    # 60 s later the last quote is still the 10 s one (the 100 s quote is in the future): no interpolation
    assert f["markout_60s_bps"] == pytest.approx(5.0)


def test_future_data_cannot_change_past_fills():
    rng = np.random.default_rng(0)
    n = 400
    ts = np.sort(rng.integers(0, 200 * S, n))
    tr = _trades(
        list(
            zip(
                ts,
                100 + rng.normal(0, 0.1, n).round(1),
                rng.uniform(0.1, 2, n),
                rng.random(n) < 0.5,
                strict=True,
            )
        )
    )
    orders = [
        Order(
            f"o{i}",
            "buy" if i % 2 else "sell",
            3.0,
            int(i * 10 * S),
            limit_price=100.0,
            cancel_ns=int(i * 10 * S + 30 * S),
        )
        for i in range(10)
    ]
    full = simulate(Q, tr, orders, CFG).fills
    cut = 120 * S
    tr2 = tr.copy()
    tr2.loc[tr2["ts_ns"] >= cut, "price"] = 50.0
    part = simulate(Q, tr2, orders, CFG).fills
    a = full[full["ts_ns"] < cut].drop(columns=[c for c in full.columns if c.startswith("markout")])
    b = part[part["ts_ns"] < cut].drop(columns=[c for c in part.columns if c.startswith("markout")])
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_crossed_quotes_rejected():
    with pytest.raises(ValueError, match="crossed"):
        simulate(_quotes([(0, 101.0, 1, 100.0, 1)]), _trades([(1, 100.0, 1.0, True)]), [], CFG)
