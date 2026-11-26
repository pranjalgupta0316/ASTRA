"""Order-level execution simulation on L1 quotes + trades (P6).

Why: ASTRA's bar engines fill every order at the next open with fixed bps costs. That cannot answer
execution questions - does a passive (maker) order get filled, at what adverse selection, versus paying
the spread as a taker? This module simulates individual market and limit orders against best-bid/ask
quotes and the trade tape (e.g. Binance USD-M bookTicker + aggTrades), with latency and cancellation.

What it does NOT claim: aggregated L1/L2 data carries no order-level queue information, so the fill time
of a resting limit order is unknowable. Every resting order is therefore simulated under three explicit
fill models, reported side by side:
  - "through"  (pessimistic): fills only from aggressor trades priced strictly through the limit.
  - "queue"    (estimate):    joins the BACK of the displayed queue at its price; only trades at that price
                              deplete the queue ahead (cancellations ahead are never credited, which biases
                              fills late); trades through the price fill directly. If the limit is below
                              the best quote at activation, the queue ahead is unknown -> "through" rule.
  - "touch"    (optimistic):  fills from any aggressor trade at or through the limit.
The truth lies between "through" and "touch"; "queue" is a labelled assumption, not a measurement.

Conventions
- Timestamps are int64 ns UTC. An order submitted at t becomes active at t + latency; a cancel requested
  at c takes effect at c + latency (fills can still happen until then).
- Aggressor side from Binance `is_buyer_maker`: True -> the seller was the aggressor (hits bids, can fill
  our resting BUY); False -> buyer aggressor (lifts asks, can fill our resting SELL).
- Market orders (and marketable limits) take liquidity at the first quote at or after activation, up to
  the displayed size; the remainder is filled at that price plus `beyond_l1_slippage_bps` and flagged,
  because depth beyond L1 is not observed. No fill is ever interpolated between observations.
- Fees: per side, bps of notional (maker / taker). Funding and margin are out of scope here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["buy", "sell"]
FillModel = Literal["through", "queue", "touch"]
FILL_MODELS: tuple[FillModel, ...] = ("through", "queue", "touch")


@dataclass(frozen=True)
class Order:
    order_id: str
    side: Side
    qty: float
    submit_ns: int
    limit_price: float | None = None  # None = market order
    cancel_ns: int | None = None  # request time; None = good until the data ends


@dataclass(frozen=True)
class ExecConfig:
    latency_ns: int = 50_000_000  # 50 ms each way (order entry and cancel)
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    beyond_l1_slippage_bps: float = 5.0
    markout_horizons_s: tuple[float, ...] = (1.0, 10.0, 60.0)


@dataclass
class SimResult:
    fills: pd.DataFrame
    orders: pd.DataFrame
    notes: list[str] = field(default_factory=list)


def _prep_quotes(q: pd.DataFrame) -> dict[str, np.ndarray]:
    need = {"ts_ns", "bid", "bid_qty", "ask", "ask_qty"}
    if not need <= set(q.columns):
        raise ValueError(f"quotes need columns {sorted(need)}")
    q = q.sort_values("ts_ns", kind="stable")
    if (q["bid"] > q["ask"]).any():
        raise ValueError("crossed quotes (bid > ask) in input; clean them first")
    return {c: q[c].to_numpy() for c in need}


def _prep_trades(t: pd.DataFrame) -> dict[str, np.ndarray]:
    need = {"ts_ns", "price", "qty", "is_buyer_maker"}
    if not need <= set(t.columns):
        raise ValueError(f"trades need columns {sorted(need)}")
    t = t.sort_values("ts_ns", kind="stable")
    return {c: t[c].to_numpy() for c in need}


def mid_at(quotes: dict[str, np.ndarray], ts: np.ndarray) -> np.ndarray:
    """Mid of the last quote at or before each ts (NaN before the first quote). Never interpolates."""
    i = np.searchsorted(quotes["ts_ns"], ts, side="right") - 1
    mid = (quotes["bid"] + quotes["ask"]) / 2
    out = np.where(i >= 0, mid[np.clip(i, 0, None)], np.nan)
    return out.astype(float)


def _take(
    order: Order, q: dict[str, np.ndarray], t_act: int, cfg: ExecConfig, limit: float | None
) -> list[dict]:
    """Taker execution at the first quote at/after activation."""
    i = int(np.searchsorted(q["ts_ns"], t_act, side="left"))
    if i >= len(q["ts_ns"]):
        return []
    px = float(q["ask"][i] if order.side == "buy" else q["bid"][i])
    if limit is not None and (px > limit if order.side == "buy" else px < limit):
        return []  # no longer marketable at the quote we would hit
    size = float(q["ask_qty"][i] if order.side == "buy" else q["bid_qty"][i])
    ts = int(q["ts_ns"][i])
    first = min(order.qty, size)
    fills = [{"ts_ns": ts, "price": px, "qty": first, "liquidity": "taker", "beyond_l1": False}]
    rest = order.qty - first
    if rest > 1e-12:
        sgn = 1 if order.side == "buy" else -1
        px2 = px * (1 + sgn * cfg.beyond_l1_slippage_bps / 1e4)
        if limit is not None and (px2 > limit if order.side == "buy" else px2 < limit):
            return fills  # remainder would violate the limit: it is not filled (no resting remainder)
        fills.append({"ts_ns": ts, "price": px2, "qty": rest, "liquidity": "taker", "beyond_l1": True})
    return fills


def _rest(
    order: Order,
    q: dict[str, np.ndarray],
    tr: dict[str, np.ndarray],
    t_act: int,
    t_end: int,
    model: FillModel,
) -> tuple[list[dict], dict]:
    """Passive fills of a resting limit order over [t_act, t_end)."""
    p = float(order.limit_price)  # type: ignore[arg-type]
    buy = order.side == "buy"
    qi = int(np.searchsorted(q["ts_ns"], t_act, side="right")) - 1
    info = {"queue_ahead": np.nan, "queue_known": False}
    queue_ahead = 0.0
    if model == "queue":
        if qi < 0:
            model_eff: FillModel = "through"
        else:
            best = float(q["bid"][qi] if buy else q["ask"][qi])
            best_qty = float(q["bid_qty"][qi] if buy else q["ask_qty"][qi])
            if (buy and p > best) or (not buy and p < best):
                queue_ahead, info["queue_known"] = 0.0, True  # we improve the quote: first in line
            elif p == best:
                queue_ahead, info["queue_known"] = best_qty, True
            info["queue_ahead"] = queue_ahead if info["queue_known"] else np.nan
            model_eff = "queue" if info["queue_known"] else "through"
    else:
        model_eff = model
    lo = int(np.searchsorted(tr["ts_ns"], t_act, side="left"))
    hi = int(np.searchsorted(tr["ts_ns"], t_end, side="left"))
    ts, px, qty, ibm = tr["ts_ns"][lo:hi], tr["price"][lo:hi], tr["qty"][lo:hi], tr["is_buyer_maker"][lo:hi]
    hits_us = ibm if buy else ~ibm.astype(bool)  # aggressor on the other side
    through = (px < p) if buy else (px > p)
    at = px == p
    remaining, fills = order.qty, []
    for k in np.flatnonzero(hits_us):
        if remaining <= 1e-12:
            break
        avail = 0.0
        if through[k]:
            avail = float(qty[k])
        elif at[k]:
            if model_eff == "touch":
                avail = float(qty[k])
            elif model_eff == "queue":
                eat = min(queue_ahead, float(qty[k]))
                queue_ahead -= eat
                avail = float(qty[k]) - eat
        if avail > 0:
            f = min(avail, remaining)
            remaining -= f
            fills.append(
                {"ts_ns": int(ts[k]), "price": p, "qty": f, "liquidity": "maker", "beyond_l1": False}
            )
    return fills, info


def simulate(
    quotes: pd.DataFrame,
    trades: pd.DataFrame,
    orders: list[Order],
    config: ExecConfig | None = None,
    models: tuple[FillModel, ...] = FILL_MODELS,
) -> SimResult:
    """Simulate every order under every fill model. Taker orders fill identically under all models."""
    cfg = config or ExecConfig()
    q, tr = _prep_quotes(quotes), _prep_trades(trades)
    end = int(max(q["ts_ns"][-1] if len(q["ts_ns"]) else 0, tr["ts_ns"][-1] if len(tr["ts_ns"]) else 0)) + 1
    fill_rows, order_rows = [], []
    for o in orders:
        if o.qty <= 0:
            raise ValueError(f"order {o.order_id}: qty must be > 0")
        t_act = o.submit_ns + cfg.latency_ns
        t_end = min(end, o.cancel_ns + cfg.latency_ns) if o.cancel_ns is not None else end
        qi = int(np.searchsorted(q["ts_ns"], t_act, side="right")) - 1
        marketable = o.limit_price is None
        if o.limit_price is not None and qi >= 0:
            marketable = o.limit_price >= q["ask"][qi] if o.side == "buy" else o.limit_price <= q["bid"][qi]
        for m in models:
            info: dict = {"queue_ahead": np.nan, "queue_known": False}
            if t_act >= t_end:
                fills = []
            elif marketable:
                fills = _take(o, q, t_act, cfg, o.limit_price)
            else:
                fills, info = _rest(o, q, tr, t_act, t_end, m)
            filled = sum(f["qty"] for f in fills)
            for f in fills:
                fee_bps = cfg.taker_fee_bps if f["liquidity"] == "taker" else cfg.maker_fee_bps
                fill_rows.append(
                    {
                        "order_id": o.order_id,
                        "model": m,
                        "side": o.side,
                        **f,
                        "fee": f["price"] * f["qty"] * fee_bps / 1e4,
                    }
                )
            vwap = sum(f["price"] * f["qty"] for f in fills) / filled if filled else np.nan
            order_rows.append(
                {
                    "order_id": o.order_id,
                    "model": m,
                    "side": o.side,
                    "qty": o.qty,
                    "limit_price": o.limit_price,
                    "active_from_ns": t_act,
                    "active_until_ns": t_end,
                    "marketable": bool(marketable),
                    "filled_qty": filled,
                    "fill_ratio": filled / o.qty,
                    "vwap": vwap,
                    "first_fill_ns": min((f["ts_ns"] for f in fills), default=None),
                    **info,
                }
            )
    fills_df = pd.DataFrame(fill_rows)
    if len(fills_df):
        sgn = np.where(fills_df["side"] == "buy", 1.0, -1.0)
        for h in cfg.markout_horizons_s:
            mid_h = mid_at(q, fills_df["ts_ns"].to_numpy() + int(h * 1e9))
            fills_df[f"markout_{h:g}s_bps"] = sgn * (mid_h - fills_df["price"]) / fills_df["price"] * 1e4
    notes = [
        "Resting-order fills are bounded by 'through' (pessimistic) and 'touch' (optimistic); 'queue' "
        "assumes joining the back of the displayed L1 queue with no cancellations ahead credited.",
        "Mark-outs use the last observed mid at fill time + horizon (no interpolation); NaN past the "
        "data end.",
    ]
    return SimResult(fills=fills_df, orders=pd.DataFrame(order_rows), notes=notes)
