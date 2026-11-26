"""H-MS-003/004: passive-with-fallback vs taker execution, and adverse selection of passive fills, on
Binance USD-M perpetuals (registration: docs/research/H-MS-v2.md incl. Amendment 1).

The per-day core (`run_day`) is pure: quotes + trades in, one row per parent order and fill model out.
Loading, the data lock and records are in `run_execution_study`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from astra.execution import FILL_MODELS, ExecConfig, Order, mid_at, simulate
from astra.stats.inference import newey_west_tstat

NS = 1_000_000_000
PARENT_EVERY_S = 300
PASSIVE_TTL_S = 60
MAX_QUOTE_AGE_S = 5
QTY = {"BTCUSDT": 0.05, "ETHUSDT": 1.0}
CFG = ExecConfig(latency_ns=50_000_000, maker_fee_bps=2.0, taker_fee_bps=5.0, beyond_l1_slippage_bps=5.0)


@dataclass(frozen=True)
class Parent:
    idx: int
    t_ns: int
    side: str


def parent_schedule(day_start_ns: int, day_end_ns: int, first_index: int = 0) -> list[Parent]:
    """Every 5 minutes on the UTC clock; side alternates by the global parent index (buy on even)."""
    start = -(-day_start_ns // (PARENT_EVERY_S * NS)) * PARENT_EVERY_S * NS
    ts = np.arange(start, day_end_ns, PARENT_EVERY_S * NS, dtype=np.int64)
    return [
        Parent(first_index + k, int(t), "buy" if (first_index + k) % 2 == 0 else "sell")
        for k, t in enumerate(ts)
    ]


def _is_bps(fills: pd.DataFrame, side: str, mid: float, qty: float) -> tuple[float, float]:
    """(implementation shortfall incl. fees in bps of arrival notional, filled qty)."""
    if fills.empty:
        return float("nan"), 0.0
    q = float(fills["qty"].sum())
    vwap = float((fills["price"] * fills["qty"]).sum() / q)
    sgn = 1.0 if side == "buy" else -1.0
    return sgn * (vwap - mid) / mid * 1e4 + float(fills["fee"].sum()) / (q * mid) * 1e4, q


WINDOW_S = 130  # 60 s TTL + 2 latencies + 60 s longest mark-out + margin


def run_day(
    quotes: pd.DataFrame, trades: pd.DataFrame, parents: list[Parent], qty: float, window: bool = True
) -> pd.DataFrame:
    """One row per (parent, fill model). quotes: ts_ns, bid, bid_qty, ask, ask_qty; trades: ts_ns, price,
    qty, is_buyer_maker. Both must cover [parent t, t + 60 s + 2 latency] to be meaningful."""
    q = quotes.sort_values("ts_ns", kind="stable").reset_index(drop=True)
    qt = q["ts_ns"].to_numpy()
    arrays = {c: q[c].to_numpy() for c in ("ts_ns", "bid", "ask")}
    bid, ask = q["bid"].to_numpy(float), q["ask"].to_numpy(float)
    bq, aq = q["bid_qty"].to_numpy(float), q["ask_qty"].to_numpy(float)
    tr_sorted = trades.sort_values("ts_ns", kind="stable").reset_index(drop=True)
    tt = tr_sorted["ts_ns"].to_numpy()
    rows: list[dict[str, Any]] = []
    lat = CFG.latency_ns
    for p in parents:
        i = int(np.searchsorted(qt, p.t_ns, side="right")) - 1
        base = {"parent": p.idx, "t_ns": p.t_ns, "side": p.side}
        if i < 0 or (p.t_ns - qt[i]) > MAX_QUOTE_AGE_S * NS or bq[i] <= 0 or aq[i] <= 0:
            rows.extend({**base, "model": m, "excluded": True} for m in FILL_MODELS)
            continue
        mid = float(mid_at(arrays, np.array([p.t_ns]))[0])
        spread_bps = float((ask[i] - bid[i]) / mid * 1e4)
        l1 = float(aq[i] if p.side == "buy" else bq[i])
        limit = float(bid[i] if p.side == "buy" else ask[i])
        # Each parent only needs [last quote <= t, t + WINDOW_S]; slicing keeps a day with ~25 M quotes
        # tractable and is result-identical (tested against window=False).
        if window:
            hi_q = int(np.searchsorted(qt, p.t_ns + WINDOW_S * NS, side="right"))
            lo_t = int(np.searchsorted(tt, p.t_ns, side="left"))
            hi_t = int(np.searchsorted(tt, p.t_ns + WINDOW_S * NS, side="right"))
            qw, tw = q.iloc[i:hi_q], tr_sorted.iloc[lo_t:hi_t]
        else:
            qw, tw = q, tr_sorted
        # A: immediate taker (identical under every model)
        a = simulate(qw, tw, [Order("A", p.side, qty, p.t_ns)], CFG, models=("through",))  # type: ignore[arg-type]
        is_a, _ = _is_bps(a.fills, p.side, mid, qty)
        beyond_a = bool(a.fills["beyond_l1"].any()) if len(a.fills) else False
        # B: passive at the same-side best, cancel requested at t + 60 s
        cancel = p.t_ns + PASSIVE_TTL_S * NS
        b = simulate(qw, tw, [Order("B", p.side, qty, p.t_ns, limit_price=limit, cancel_ns=cancel)], CFG)  # type: ignore[arg-type]
        for m in FILL_MODELS:
            fb = b.fills[b.fills["model"] == m] if len(b.fills) else b.fills
            filled = float(fb["qty"].sum()) if len(fb) else 0.0
            rest = qty - filled
            parts = [fb]
            if rest > 1e-12:  # Amendment 1: fallback submitted when the cancel takes effect
                fbk = simulate(qw, tw, [Order("F", p.side, rest, cancel + lat)], CFG, models=(m,))  # type: ignore[arg-type]
                parts.append(fbk.fills)
            allf = (
                pd.concat([x for x in parts if len(x)], ignore_index=True)
                if any(len(x) for x in parts)
                else fb
            )
            is_b, q_b = _is_bps(allf, p.side, mid, qty)
            maker = fb[fb["liquidity"] == "maker"] if len(fb) else fb
            mk = {}
            for h in ("1", "10", "60"):
                col = f"markout_{h}s_bps"
                mk[f"maker_markout_{h}s"] = (
                    float((maker[col] * maker["qty"]).sum() / maker["qty"].sum()) if len(maker) else np.nan
                )
            first = fb["ts_ns"].min() if len(fb) else np.nan
            rows.append(
                {
                    **base,
                    "model": m,
                    "excluded": False,
                    "mid": mid,
                    "spread_bps": spread_bps,
                    "l1_lt_qty": l1 < qty,
                    "is_a_bps": is_a,
                    "is_b_bps": is_b,
                    "d_bps": is_b - is_a,
                    "b_passive_fill_ratio": filled / qty,
                    "b_total_filled": q_b,
                    "a_beyond_l1": beyond_a,
                    "b_time_to_first_fill_s": (first - p.t_ns) / NS if len(fb) else np.nan,
                    **mk,
                }
            )
    return pd.DataFrame(rows)


def evaluate(rows: pd.DataFrame) -> dict[str, Any]:
    """Registered decisions for H-MS-003 and H-MS-004 from all parent rows (all days, all symbols)."""
    from statsmodels.stats.multitest import multipletests

    ok = rows[~rows["excluded"].astype(bool)]
    cells: list[dict[str, Any]] = []
    for (sym, m), g in ok.groupby(["symbol", "model"]):
        g = g.sort_values("t_ns")
        d = g["d_bps"].dropna()
        mk = g["maker_markout_10s"].dropna()
        cells.append(
            {
                "symbol": sym,
                "model": m,
                "n_parents": len(g),
                "mean_is_a_bps": g["is_a_bps"].mean(),
                "mean_is_b_bps": g["is_b_bps"].mean(),
                "mean_d_bps": d.mean(),
                "nw_t_d": newey_west_tstat(d),
                "passive_fill_ratio": g["b_passive_fill_ratio"].mean(),
                "n_maker_parents": len(mk),
                "mean_markout_10s_bps": mk.mean(),
                "nw_t_markout_10s": newey_west_tstat(mk),
                "mean_markout_1s_bps": g["maker_markout_1s"].mean(),
                "mean_markout_60s_bps": g["maker_markout_60s"].mean(),
                "share_a_beyond_l1": g["a_beyond_l1"].mean(),
                "share_l1_lt_qty": g["l1_lt_qty"].mean(),
                "median_spread_bps": g["spread_bps"].median(),
            }
        )
    t = pd.DataFrame(cells)
    from scipy import stats as sst

    def bh(df: pd.DataFrame, col: str) -> pd.Series:
        p = 2 * sst.norm.sf(df[col].abs().to_numpy())
        return pd.Series(multipletests(p, method="fdr_bh")[1], index=df.index)

    out: dict[str, Any] = {
        "table": t,
        "excluded_parents": int(rows.loc[rows["model"] == "through", "excluded"].sum()),
    }
    thr, tch, que = (t[t["model"] == m] for m in ("through", "touch", "queue"))
    if len(thr):
        thr = thr.assign(p_bh=bh(thr, "nw_t_d"))
        tch = tch.assign(p_bh=bh(tch, "nw_t_d"))
        sup = bool(((thr["nw_t_d"] < -1.96) & (thr["p_bh"] < 0.05)).all() and len(thr) == 2)
        rej = bool(((tch["nw_t_d"] > 1.96) & (tch["p_bh"] < 0.05)).all() and len(tch) == 2)
        out["H-MS-003"] = "supported" if sup else "rejected" if rej else "inconclusive"
    if len(que):
        que = que.assign(p_bh=bh(que, "nw_t_markout_10s"))
        sup = bool(((que["nw_t_markout_10s"] < -1.96) & (que["p_bh"] < 0.05)).all() and len(que) == 2)
        rej = bool(((que["nw_t_markout_10s"] > 1.96) & (que["p_bh"] < 0.05)).all() and len(que) == 2)
        out["H-MS-004"] = "supported" if sup else "rejected" if rej else "inconclusive"
    return out
