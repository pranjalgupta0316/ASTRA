"""Automated backtest audit checks (part of the backtesting-audit checklist)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from nebula.backtest.engine import BacktestConfig, run_backtest, signal_strategy
from nebula.risk.metrics import summarize


def leakage_canary(bars: pd.DataFrame, periods_per_year: int, config: BacktestConfig) -> dict[str, Any]:
    """A signal that peeks at the next open-to-open return must look spectacular.

    If the engine allowed same-bar fills or mis-aligned dates, a *legitimate* signal could also look
    like this. The canary verifies that the engine rewards (only) truly future information, proving
    the timing path is wired: decision at t -> fill at open t+1 -> P&L from open t+1 onward.
    """
    o = bars["open"].reset_index(drop=True)
    future_oo = np.log(o.shift(-2) / o.shift(-1))
    sig = pd.Series((future_oo > 0).astype(float).to_numpy(), index=pd.to_datetime(bars["session_date"]))
    # Cost-free and uncapped: the canary tests timing, not capacity (a binding volume cap on a
    # compounding perfect-foresight book made the earlier version fragile).
    canary_cfg = replace(config, cost_multiplier=0.0, max_participation=float("inf"))
    s = summarize(run_backtest(bars, signal_strategy(sig), canary_cfg).equity_net, periods_per_year)
    # Same canary shifted two bars later (no longer future-informed) should look ordinary.
    lagged = sig.shift(2)
    s_lag = summarize(run_backtest(bars, signal_strategy(lagged), canary_cfg).equity_net, periods_per_year)
    return {
        "canary_gross_sharpe": s["sharpe"],
        "lagged_canary_gross_sharpe": s_lag["sharpe"],
        "passed": bool(s["sharpe"] > 3.0 and abs(s_lag["sharpe"]) < 3.0),
    }


def cost_sensitivity(
    bars: pd.DataFrame,
    signal: pd.Series,
    periods_per_year: int,
    config: BacktestConfig,
    multipliers: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
) -> pd.DataFrame:
    rows = []
    for m in multipliers:
        res = run_backtest(bars, signal_strategy(signal), replace(config, cost_multiplier=m))
        s = summarize(res.equity_net, periods_per_year)
        rows.append(
            {
                "cost_multiplier": m,
                **s,
                "total_cost": float(res.ledger["cost"].sum()),
                "avg_daily_turnover": float(res.ledger["turnover"].mean()),
                "n_fills": len(res.fills),
            }
        )
    return pd.DataFrame(rows)


def signal_availability_check(signal_dates: pd.Series, feature_close_times: pd.Series) -> dict[str, Any]:
    """Each signal date must correspond to a bar whose close_time_utc is known (no future-dated signals)."""
    sig = pd.to_datetime(pd.Series(signal_dates)).dt.normalize()
    feats = pd.to_datetime(pd.Series(feature_close_times))
    last_close = feats.max()
    future = (
        sig[sig > last_close.tz_localize(None).normalize()] if last_close.tzinfo else sig[sig > last_close]
    )
    return {"future_dated_signals": len(future), "passed": len(future) == 0}
