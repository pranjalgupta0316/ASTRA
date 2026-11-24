"""Event-driven multi-asset backtester for target weights (one market, its own calendar).

Conventions are those of `nebula.backtest.engine` (single asset), extended to N assets:
  1. Open(t):   pending target weights decided at close t-1 (or earlier, if a leg could not fill) are
                executed at open(t). Sells first, then buys. Buys are sized net of their own cost and
                scaled down pro rata if cash after sells cannot pay for them -> cash is never negative.
                Each leg is capped at max_participation * volume(t); the remainder is cancelled.
                Zero-volume or missing bars never fill; that leg stays pending until it fills or a new
                decision replaces it. Legs within `rebalance_tolerance` of the current weight are skipped
                (full exits always run).
  2. Close(t):  mark to close(t); an asset without a bar today is marked at its last close.
  3. Decision:  if t is a decision date, its target row becomes pending for open(t+1).
Costs: per side, bps of traded notional from CostConfig (commission + half-spread + slippage + taxes),
times `cost_multiplier`. Exact gross = identical run with cost_multiplier = 0 (`run_portfolio_gross`).

Calendar: the panel index is the union of the market's validated session dates. Markets are never mixed
(no forward-filling across exchanges or into 24/7 crypto). Portfolios are in local currency.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from nebula.common.config import CostConfig
from nebula.common.paths import Paths
from nebula.data.storage import read_bars
from nebula.features.technical import adjusted_ohlc


@dataclass
class PortfolioBacktestConfig:
    initial_cash: float = 1_000_000.0
    costs: CostConfig = field(default_factory=CostConfig)
    cost_multiplier: float = 1.0
    max_participation: float = 0.05
    min_trade_weight: float = 1e-4
    rebalance_tolerance: float = 0.02


@dataclass
class MarketPanel:
    """Wide adjusted open/close/volume frames on one market's calendar (NaN = no valid bar)."""

    market: str
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    excluded_rows: dict[str, list[str]] = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def returns(self) -> pd.DataFrame:
        """Simple close-to-close returns; NaN on missing bars and on the bar after a gap."""
        return self.close.pct_change(fill_method=None)

    def slice(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> MarketPanel:
        sl = slice(start, end)
        return MarketPanel(
            self.market, self.open.loc[sl], self.close.loc[sl], self.volume.loc[sl], self.excluded_rows
        )


@dataclass
class PortfolioFill:
    session_date: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    notional: float
    cost: float
    capped: bool
    cash_scaled: bool


@dataclass
class PortfolioBacktestResult:
    ledger: pd.DataFrame  # equity_net, cash, cost, turnover, exposure per bar
    weights: pd.DataFrame  # end-of-bar weights per asset
    positions: pd.DataFrame  # quantities per asset
    fills: list[PortfolioFill]

    @property
    def equity_net(self) -> pd.Series:
        return self.ledger["equity_net"]


def load_market_panel(
    market: str,
    symbols: list[str],
    paths: Paths | None = None,
    window: tuple[str | None, str | None] | None = None,
) -> MarketPanel:
    """Read validated bars, drop provider placeholder rows (volume 0 and H == L), adjust prices."""
    opens, closes, vols, excluded = {}, {}, {}, {}
    for s in symbols:
        bars = read_bars(s, market, paths, window=window)
        ph = (bars["volume"] == 0) & (bars["high"] == bars["low"])
        excluded[s] = [str(d) for d in bars.loc[ph, "session_date"]]
        adj = adjusted_ohlc(bars.loc[~ph].reset_index(drop=True)).set_index("session_date")
        opens[s], closes[s], vols[s] = adj["open"], adj["close"], adj["volume"].astype(float)
    o, c, v = pd.DataFrame(opens), pd.DataFrame(closes), pd.DataFrame(vols)
    idx = c.index.union(o.index).sort_values()
    return MarketPanel(market, o.reindex(idx), c.reindex(idx), v.reindex(idx), excluded)


def monthly_decision_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session of each calendar month (the panel's first row is excluded: we cannot know it is
    the first session of its month without looking before the sample)."""
    idx = pd.DatetimeIndex(index)
    per = idx.to_period("M")
    first = np.r_[False, per[1:] != per[:-1]]
    return idx[first]


def run_portfolio_backtest(
    panel: MarketPanel,
    targets: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
    start: pd.Timestamp | None = None,
) -> PortfolioBacktestResult:
    """Simulate target weights. `targets`: index = decision dates (subset of the panel index), columns =
    panel symbols (missing columns / NaN -> 0). Row sum <= 1 (remainder cash). The simulation starts at
    `start` (default: first decision date) with all cash."""
    cfg = config or PortfolioBacktestConfig()
    syms = panel.symbols
    tg = targets.reindex(columns=syms).fillna(0.0)
    if (tg < -1e-12).to_numpy().any() or (tg.sum(axis=1) > 1 + 1e-9).any():
        raise ValueError("targets must be long-only with row sums <= 1")
    t0 = pd.Timestamp(start) if start is not None else pd.Timestamp(tg.index[0])
    idx = panel.close.index[panel.close.index >= t0]
    missing = tg.index.difference(panel.close.index)
    if len(missing):
        raise ValueError(f"decision dates not in panel calendar: {list(missing[:3])}")
    opens = panel.open.loc[idx].to_numpy(float)
    closes = panel.close.loc[idx].to_numpy(float)
    vols = panel.volume.loc[idx].to_numpy(float)
    decisions = {pd.Timestamp(d): row.to_numpy(float) for d, row in tg.iterrows()}
    n = len(syms)
    buy_bps = cfg.costs.buy_bps() * cfg.cost_multiplier / 1e4
    sell_bps = cfg.costs.sell_bps() * cfg.cost_multiplier / 1e4

    cash = float(cfg.initial_cash)
    pos = np.zeros(n)
    last_close = np.full(n, np.nan)
    pending = np.full(n, np.nan)  # NaN = no order for that asset
    fills: list[PortfolioFill] = []
    rows: list[dict[str, Any]] = []
    W = np.zeros((len(idx), n))
    P = np.zeros((len(idx), n))

    for i, d in enumerate(idx):
        o, c, v = opens[i], closes[i], vols[i]
        day_cost, traded = 0.0, 0.0
        if not np.all(np.isnan(pending)):
            mark_open = np.where(np.isfinite(o), o, last_close)
            held = pos != 0
            if np.any(held & ~np.isfinite(mark_open)):
                raise ValueError(f"held asset without any price on {d}")
            equity_open = cash + float(np.sum(np.where(held, pos * mark_open, 0.0)))
            deltas = np.zeros(n)
            capped = np.zeros(n, dtype=bool)
            for j in range(n):
                tw = pending[j]
                if np.isnan(tw):
                    continue
                if not (np.isfinite(o[j]) and np.isfinite(v[j]) and v[j] > 0):
                    continue  # cannot fill today; leg stays pending
                pending[j] = np.nan  # leg is resolved today (filled, capped remainder cancelled, or skipped)
                if equity_open <= 0:
                    continue
                cur_w = pos[j] * o[j] / equity_open
                is_exit = tw == 0.0 and pos[j] != 0.0
                if abs(tw - cur_w) < cfg.rebalance_tolerance and not is_exit:
                    continue
                dq = tw * equity_open / o[j] - pos[j]
                if dq > 0:
                    dq = dq / (1 + buy_bps)  # buys sized net of their own cost
                cap = cfg.max_participation * v[j]
                if abs(dq) > cap:
                    dq, capped[j] = float(np.copysign(cap, dq)), True
                if abs(dq * o[j]) < cfg.min_trade_weight * equity_open:
                    continue
                deltas[j] = dq
            # sells first
            for j in np.flatnonzero(deltas < 0):
                notional = -deltas[j] * o[j]
                cost = notional * sell_bps
                cash += notional - cost
                pos[j] += deltas[j]
                day_cost += cost
                traded += notional
                fills.append(PortfolioFill(d, syms[j], deltas[j], o[j], notional, cost, capped[j], False))
            # then buys, scaled pro rata if cash cannot pay for all of them (incl. costs)
            buys = np.flatnonzero(deltas > 0)
            need = float(np.sum(deltas[buys] * o[buys])) * (1 + buy_bps)
            scale = 1.0
            if need > cash:
                scale = max(cash, 0.0) / need * (1 - 1e-12)
            for j in buys:
                q = deltas[j] * scale
                notional = q * o[j]
                cost = notional * buy_bps
                cash -= notional + cost
                pos[j] += q
                day_cost += cost
                traded += notional
                fills.append(PortfolioFill(d, syms[j], q, o[j], notional, cost, capped[j], scale < 1.0))
            if cash < 0:  # pragma: no cover - guarded above; floating point only
                if cash < -1e-6 * max(1.0, equity_open):
                    raise AssertionError(f"negative cash {cash} on {d}")
                cash = 0.0
        # close: mark to market
        last_close = np.where(np.isfinite(c), c, last_close)
        mark = np.where(pos != 0, pos * np.nan_to_num(last_close), 0.0)
        equity = cash + float(mark.sum())
        W[i] = mark / equity if equity != 0 else 0.0
        P[i] = pos
        rows.append(
            {
                "session_date": d,
                "equity_net": equity,
                "cash": cash,
                "cost": day_cost,
                "turnover": traded / equity if equity != 0 else 0.0,
                "exposure": float(W[i].sum()),
            }
        )
        # decision at close d
        if d in decisions:
            pending = decisions[d].copy()

    ledger = pd.DataFrame(rows).set_index("session_date")
    weights = pd.DataFrame(W, index=idx, columns=syms)
    positions = pd.DataFrame(P, index=idx, columns=syms)
    return PortfolioBacktestResult(ledger=ledger, weights=weights, positions=positions, fills=fills)


def run_portfolio_gross(
    panel: MarketPanel,
    targets: pd.DataFrame,
    config: PortfolioBacktestConfig | None = None,
    start: pd.Timestamp | None = None,
) -> PortfolioBacktestResult:
    """Exact gross path: identical run with all costs switched off."""
    cfg = config or PortfolioBacktestConfig()
    return run_portfolio_backtest(panel, targets, replace(cfg, cost_multiplier=0.0), start)
