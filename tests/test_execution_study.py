"""H-MS-003/004 study core on synthetic quotes/trades (no futures data needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from astra.research.execution_study import NS, evaluate, parent_schedule, run_day


def _market(seed: int, minutes: int = 60, seller_heavy: float = 0.5):
    rng = np.random.default_rng(seed)
    ts = np.arange(0, minutes * 60 * NS, NS // 2)  # quote every 0.5 s
    mid = 100 + np.cumsum(rng.normal(0, 0.01, len(ts)))
    q = pd.DataFrame(
        {
            "ts_ns": ts,
            "bid": (mid - 0.05).round(2),
            "bid_qty": 2.0,
            "ask": (mid + 0.05).round(2),
            "ask_qty": 2.0,
        }
    )
    tt = np.sort(rng.integers(0, ts[-1], len(ts) // 2))
    j = np.searchsorted(ts, tt, side="right") - 1
    ibm = rng.random(len(tt)) < seller_heavy
    px = np.where(ibm, q["bid"].to_numpy()[j] - rng.choice([0, 0.01], len(tt)), q["ask"].to_numpy()[j])
    tr = pd.DataFrame({"ts_ns": tt, "price": px, "qty": rng.uniform(0.1, 3, len(tt)), "is_buyer_maker": ibm})
    return q, tr


def test_schedule_is_on_the_5_minute_grid_and_alternates():
    ps = parent_schedule(7 * NS, 20 * 60 * NS, first_index=3)
    assert [p.t_ns // NS for p in ps] == [300, 600, 900]
    assert [p.side for p in ps] == ["sell", "buy", "sell"]


def test_run_day_rows_and_model_ordering():
    q, tr = _market(0)
    rows = run_day(q, tr, parent_schedule(0, 55 * 60 * NS), qty=1.0).assign(symbol="X")
    ok = rows[~rows["excluded"]]
    assert set(ok["model"]) == {"through", "queue", "touch"} and len(ok) == 3 * 11
    piv = ok.pivot_table(index="parent", columns="model", values="b_passive_fill_ratio")
    assert (piv["through"] <= piv["queue"] + 1e-12).all() and (piv["queue"] <= piv["touch"] + 1e-12).all()
    # taker leg identical across models; passive leg total always completes via fallback
    assert ok.groupby("parent")["is_a_bps"].nunique().max() == 1
    assert ok["b_total_filled"].to_numpy() == pytest.approx(1.0)
    # taker cost ~ half spread + 5 bps fee on average (fills at the quote 50 ms after arrival, so not per row)
    assert ok["is_a_bps"].mean() == pytest.approx((ok["spread_bps"] / 2 + 5.0).mean(), abs=1.0)


def test_stale_quotes_exclude_the_parent():
    q, tr = _market(1)
    q = q[(q["ts_ns"] < 200 * NS) | (q["ts_ns"] > 700 * NS)]  # no quotes between 200 s and 700 s
    rows = run_day(q, tr, parent_schedule(0, 15 * 60 * NS), qty=1.0)
    ex = rows[rows["model"] == "through"].set_index("t_ns")["excluded"]
    assert ex[300 * NS] and ex[600 * NS] and not ex[0]


def test_evaluate_decisions_and_bh():
    rng = np.random.default_rng(2)
    rows = []
    for sym in ("BTCUSDT", "ETHUSDT"):
        for m in ("through", "queue", "touch"):
            for k in range(400):
                rows.append(
                    {
                        "symbol": sym,
                        "model": m,
                        "parent": k,
                        "t_ns": k,
                        "excluded": False,
                        "is_a_bps": 6.0,
                        "is_b_bps": 4.0 + rng.normal(0, 1),
                        "spread_bps": 1.0,
                        "b_passive_fill_ratio": 0.5,
                        "maker_markout_10s": -1.0 + rng.normal(0, 1),
                        "maker_markout_1s": 0.0,
                        "maker_markout_60s": 0.0,
                        "a_beyond_l1": False,
                        "l1_lt_qty": False,
                    }
                )
    df = pd.DataFrame(rows)
    df["d_bps"] = df["is_b_bps"] - df["is_a_bps"]
    out = evaluate(df)
    assert out["H-MS-003"] == "supported" and out["H-MS-004"] == "supported"
    df["d_bps"] = rng.normal(0, 1, len(df))
    assert evaluate(df)["H-MS-003"] == "inconclusive"


def test_windowed_run_equals_full_run():
    q, tr = _market(3, minutes=40)
    ps = parent_schedule(0, 35 * 60 * NS)
    a = run_day(q, tr, ps, qty=1.5, window=True)
    b = run_day(q, tr, ps, qty=1.5, window=False)
    pd.testing.assert_frame_equal(a, b)
