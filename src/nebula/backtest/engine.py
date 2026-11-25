"""Event-driven daily backtester (single instrument, long/flat/short target weights).

Event order for each bar t:
  1. MarketOpen(t):  pending orders from the decision at close t-1 are filled at open(t) with costs,
                     participation-capped by volume(t). Nothing decided at t can fill at t.
  2. MarketClose(t): positions marked to close(t).
  3. Decision(t):    strategy sees bars[0..t] only and returns a target weight -> order for t+1.

Costs (per side, bps of traded notional): commission + half-spread + slippage + taxes, from
CostConfig. Optional square-root impact: impact_bps = impact_coef * daily_vol * sqrt(qty/volume) * 1e4.
Buys are sized net of their own cost (no borrowing to pay fees); targets within
`rebalance_tolerance` of the current weight are not traded. Exact gross results come from
`run_gross` (same run, cost_multiplier = 0).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from nebula.common.config import CostConfig

Strategy = Callable[[int, pd.DataFrame], float | None]
"""Called at the close of bar i with bars.iloc[: i + 1]; returns target weight or None (no change)."""


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    costs: CostConfig = field(default_factory=CostConfig)
    cost_multiplier: float = 1.0
    max_participation: float = 0.05  # fraction of bar volume one order may take
    impact_coef: float = 0.0  # square-root impact coefficient (0 = off)
    min_trade_weight: float = 1e-4  # ignore rebalances smaller than this fraction of equity
    rebalance_tolerance: float = 0.02  # skip if |target - current weight| below this (exits always run)
    allow_short: bool = False


@dataclass
class Fill:
    session_date: pd.Timestamp
    quantity: float
    price: float
    notional: float
    cost: float
    capped: bool


@dataclass
class BacktestResult:
    ledger: pd.DataFrame  # per-bar: equity_net, equity_net_plus_costs, cash, position, weight, cost, turnover
    fills: list[Fill]

    @property
    def equity_net(self) -> pd.Series:
        return self.ledger["equity_net"]

    @property
    def equity_net_plus_costs(self) -> pd.Series:
        """Net equity + cumulative costs. NOT exact gross (ignores compounding of costs); use
        `run_gross` for the exact zero-cost path."""
        return self.ledger["equity_net_plus_costs"]


def _order_quantity(
    target_w: float, pos: float, equity_open: float, price: float, volume: float, cfg: BacktestConfig
) -> tuple[float, bool]:
    """Signed quantity to trade at the open, and whether the participation cap bound.

    - within `rebalance_tolerance` of the current weight -> 0 (except full exits);
    - buys sized net of their own cost, so cash never goes negative to pay fees;
    - |quantity| capped at max_participation * bar volume.
    """
    if equity_open <= 0:
        return 0.0, False
    current_w = pos * price / equity_open
    is_exit = target_w == 0.0 and pos != 0.0
    if abs(target_w - current_w) < cfg.rebalance_tolerance and not is_exit:
        return 0.0, False
    target_qty = target_w * equity_open / price
    if target_qty > pos:
        buy_bps = cfg.costs.buy_bps() * cfg.cost_multiplier
        target_qty = pos + (target_qty - pos) / (1 + buy_bps / 1e4)
    delta = target_qty - pos
    cap = cfg.max_participation * volume
    if abs(delta) > cap:
        return math.copysign(cap, delta), True
    return delta, False


def run_backtest(
    bars: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None
) -> BacktestResult:
    """bars: columns session_date, open, close, volume (adjusted prices), time-ordered, unique dates."""
    cfg = config or BacktestConfig()
    bars = bars.reset_index(drop=True)
    if not pd.to_datetime(bars["session_date"]).is_monotonic_increasing:
        raise ValueError("bars must be sorted by session_date")
    cash, pos, cum_cost = cfg.initial_cash, 0.0, 0.0
    pending: float | None = None
    fills: list[Fill] = []
    rows = []
    logret = np.log(bars["close"]).diff()
    vol20 = logret.rolling(20, min_periods=5).std().shift(1)  # known before today's open

    for i in range(len(bars)):
        bar = bars.iloc[i]
        o, c, v = float(bar["open"]), float(bar["close"]), float(bar["volume"])
        day_cost = 0.0
        traded = 0.0
        # 1) open: execute pending order. Zero-volume bars (halts, provider placeholder rows) cannot
        #    fill: the order stays pending and is replaced by the next decision if one is made.
        if pending is not None and v > 0:
            equity_open = cash + pos * o
            delta, capped = _order_quantity(pending, pos, equity_open, o, v, cfg)
            if delta != 0.0 and abs(delta * o) >= cfg.min_trade_weight * max(equity_open, 1e-9):
                bps = cfg.costs.buy_bps() if delta > 0 else cfg.costs.sell_bps()
                if cfg.impact_coef > 0 and v > 0:
                    sig = float(vol20.iloc[i]) if not np.isnan(vol20.iloc[i]) else 0.0
                    bps += cfg.impact_coef * sig * math.sqrt(abs(delta) / v) * 1e4
                notional = abs(delta) * o
                cost = notional * bps * cfg.cost_multiplier / 1e4
                cash -= delta * o + cost
                pos += delta
                cum_cost += cost
                day_cost, traded = cost, notional
                fills.append(Fill(pd.Timestamp(bar["session_date"]), delta, o, notional, cost, capped))
            pending = None
        # 2) close: mark to market
        equity = cash + pos * c
        rows.append(
            {
                "session_date": pd.Timestamp(bar["session_date"]),
                "equity_net": equity,
                "equity_net_plus_costs": equity + cum_cost,
                "cash": cash,
                "position": pos,
                "weight": pos * c / equity if equity != 0 else 0.0,
                "cost": day_cost,
                "turnover": traded / equity if equity != 0 else 0.0,
            }
        )
        # 3) decision at close using history up to and including bar i
        target = strategy(i, bars.iloc[: i + 1])
        if target is not None:
            if not cfg.allow_short:
                target = max(0.0, target)
            pending = float(target)
    ledger = pd.DataFrame(rows).set_index("session_date")
    return BacktestResult(ledger=ledger, fills=fills)


def run_gross(bars: pd.DataFrame, strategy: Strategy, config: BacktestConfig | None = None) -> BacktestResult:
    """Exact gross path: identical run with all costs switched off."""
    cfg = config or BacktestConfig()
    return run_backtest(bars, strategy, replace(cfg, cost_multiplier=0.0))


def signal_strategy(signal: pd.Series) -> Strategy:
    """Strategy from a precomputed target-weight series indexed by session_date.

    The signal value at date t must only use information available at the close of t; this is the
    caller's responsibility and is checked by `nebula.backtest.audit`.
    """
    lookup = {pd.Timestamp(k): float(v) for k, v in signal.dropna().items()}

    def strat(i: int, hist: pd.DataFrame) -> float | None:
        return lookup.get(pd.Timestamp(hist["session_date"].iloc[-1]))

    return strat


def buy_and_hold() -> Strategy:
    """Target 100 % invested on every bar. Re-issuing the target (instead of a single initial order)
    matters when the volume cap cuts the first fill: the remainder is bought on later bars rather than
    left in cash (audit 2026-11-25: single-order benchmarks stayed ~3 % invested for thin early ETFs).
    Once fully invested, the rebalance tolerance suppresses further trades."""

    def strat(i: int, hist: pd.DataFrame) -> float | None:
        return 1.0

    return strat
