"""H-XMKT-001: cross-market diversification (registered in docs/research/H-EXT.md before any run).

Sleeves (daily, local currency): the unmanaged monthly-rebalanced equal-weight portfolios of US stocks
(USD), IN stocks (INR) and crypto (USDT) from `astra.research.vol_managed.ew_sleeve` (event-driven
multi-asset engine, net of within-sleeve costs, names enter after 252 valid returns).

Weekly alignment (ADR-0007)
- Week = Saturday..Friday (pandas 'W-FRI', closed and labelled on the right). Each sleeve is sampled at
  its LAST session dated <= that Friday: the US Friday close (20:00/21:00 UTC), the IN Friday close
  (10:00 UTC), the crypto daily bar dated Friday (closes Saturday 00:00 UTC; Binance daily bars are UTC
  days). Saturday/Sunday crypto bars belong to the next week. A holiday Friday -> the last earlier
  session of that week. A week with no session of a market keeps that sleeve's last value (no price
  change can happen on a closed exchange); prices are never forward-filled ACROSS markets or into the
  daily calendar. The three closes of a week are non-synchronous by up to ~14 h (stated limitation).
- FX: USDINR (Yahoo INR=X, INR per USD) from data/validated/macro/market.parquet. The IN sleeve value of
  a week is converted with the latest USDINR observation whose `available_from` is STRICTLY BEFORE the IN
  session date used (the IN close at 15:30 IST precedes the New York session of the same date, so a value
  first usable on date s is not treated as known at the IN close of s). Never an FX value before it was
  available. Deviation from the registration text: the registration named FRED DEXINUS; FRED is
  manual-CSV-only in ASTRA (no FRED data exists), so the Yahoo series with its conservative t+2 session
  availability is used (decided before the run, by the coordinator).
- Crypto USDT is treated as USD (stated assumption; USDT/USD deviations ignored).

Portfolios (weekly USD, annualization 52, rf = 0)
- US-only = 100 % US sleeve.
- Mix = equal-risk (inverse trailing-52-week vol) of the three sleeves. Decision at the first weekly
  observation of each calendar month using weekly USD returns up to and including that week; trades are
  executed at the closes of the FOLLOWING week (one-week implementation lag, strictly after every
  sleeve close used by the decision). Between rebalances sleeve weights drift. Costs: |traded USD
  notional| per sleeve times that market's buy/sell bps (config). INR<->USD conversion costs are not
  modelled. Both portfolios start in cash at the first execution week and pay their initial purchase.
- H1a: vol(mix)/vol(US-only) paired stationary-bootstrap 95% CI upper < 1 (supported; lower > 1
  rejected). H1b: dSharpe(mix - US-only) CI > 0. Block 20 (weeks), 2000 draws, seed 7.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from astra.common.config import get_config
from astra.common.logging import get_logger
from astra.common.paths import Paths, get_paths
from astra.common.seed import set_seed
from astra.data.macro.market import market_path
from astra.data.storage import read_manifest
from astra.research.vol_managed import SURVIVORSHIP_NOTE, EwSleeve, ew_sleeve
from astra.risk.attribution import forecast_realized_summary, forecast_vs_realized, paired_bootstrap_ci
from astra.risk.metrics import max_drawdown, summarize
from astra.stats.inference import sharpe_difference_bootstrap
from astra.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)

HYP = "H-XMKT-001"
SEED = 7
WEEK_RULE = "W-FRI"
PPY = 52
VOL_WINDOW = 52
N_BOOT = 2000
MEAN_BLOCK = 20
SLEEVES = ("US", "IN", "CRYPTO")


# ------------------------------------------------------------------------------------- alignment
def weekly_last(series: pd.Series) -> pd.DataFrame:
    """Value at the last observation dated <= each Friday (week Sat..Fri) and that observation's date.
    Weeks without an observation after the first one carry the last value and session date forward
    (same market only)."""
    s = pd.Series(series, dtype=float).dropna()
    df = pd.DataFrame({"value": s, "session_date": s.index}, index=s.index)
    wk = df.resample(WEEK_RULE, closed="right", label="right").last()
    return wk.ffill()


def load_usdinr(paths: Paths | None = None) -> pd.DataFrame:
    t = pq.read_table(market_path(paths or get_paths())).to_pandas()
    t = t[t["series"] == "USDINR"].copy()
    t["date"] = pd.to_datetime(t["date"])
    t["available_from"] = pd.to_datetime(t["available_from"])
    return t[["date", "value", "available_from"]].reset_index(drop=True)


def fx_asof(dates: pd.Series | pd.DatetimeIndex, fx: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """For each date d: the latest FX observation (by observation date) with available_from < d
    (strict) or <= d. Returns columns fx, fx_date, fx_available_from aligned with `dates`."""
    d = pd.DataFrame({"d": pd.to_datetime(pd.Series(dates)).to_numpy()})
    d["_pos"] = np.arange(len(d))
    f = fx.sort_values(["available_from", "date"]).rename(
        columns={"value": "fx", "date": "fx_date", "available_from": "fx_available_from"}
    )
    # latest observation date among rows available by d (available_from is non-decreasing in date)
    f = f.assign(fx_date_max=f["fx_date"].cummax())
    f = f[f["fx_date"] == f["fx_date_max"]].drop(columns="fx_date_max")
    m = pd.merge_asof(
        d.dropna(subset=["d"]).sort_values("d"),
        f,
        left_on="d",
        right_on="fx_available_from",
        allow_exact_matches=not strict,
    )
    m = m.set_index("_pos").reindex(d["_pos"])
    return m[["fx", "fx_date", "fx_available_from"]].reset_index(drop=True)


def to_usd(local_value: pd.Series, inr_per_usd: pd.Series) -> pd.Series:
    """INR value -> USD value."""
    return local_value / inr_per_usd


def weekly_usd_sleeves(
    daily_equity: dict[str, pd.Series], fx: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly USD sleeve values and an alignment audit frame (session dates, FX used)."""
    vals, audit = {}, {}
    for name, eq in daily_equity.items():
        w = weekly_last(eq)
        audit[f"{name}_session_date"] = w["session_date"]
        if name == "IN":
            f = fx_asof(w["session_date"], fx, strict=True)
            f.index = w.index
            if (f["fx_available_from"].notna() & (f["fx_available_from"] >= w["session_date"])).any():
                raise AssertionError("FX value used before it was available")
            audit["IN_fx"] = f["fx"]
            audit["IN_fx_obs_date"] = f["fx_date"]
            audit["IN_fx_available_from"] = f["fx_available_from"]
            vals[name] = to_usd(w["value"], f["fx"])
        else:  # US in USD; crypto USDT treated as USD
            vals[name] = w["value"]
    return pd.DataFrame(vals), pd.DataFrame(audit)


# ------------------------------------------------------------------------------------- portfolio
def inverse_vol_weights(rets: pd.DataFrame, t: pd.Timestamp, window: int = VOL_WINDOW) -> pd.Series | None:
    """Weights proportional to 1 / std of the last `window` weekly returns up to and including t."""
    hist = rets.loc[rets.index <= t].tail(window)
    if len(hist) < window or hist.isna().any().any():
        return None
    inv = 1.0 / hist.std(ddof=1)
    return inv / inv.sum()


def monthly_weeks(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First weekly label of each calendar month (the first label is excluded)."""
    idx = pd.DatetimeIndex(index)
    per = idx.to_period("M")
    return idx[np.r_[False, per[1:] != per[:-1]]]


def simulate_weekly(
    rets: pd.DataFrame,
    targets: pd.DataFrame,
    buy_bps: pd.Series,
    sell_bps: pd.Series,
    exec_lag: int = 1,
    initial: float = 1.0,
) -> pd.DataFrame:
    """Weekly sleeve-level book. `targets` indexed by decision week; executed at the close of the week
    `exec_lag` labels later (cost = traded notional x bps of that sleeve's market). Before the first
    execution the book is cash. Returns per week: value, cost, turnover, weights."""
    cols = list(rets.columns)
    weeks = list(rets.index)
    pos = {w: i for i, w in enumerate(weeks)}
    execs: dict[int, np.ndarray] = {}
    for t, row in targets.iterrows():
        k = pos[pd.Timestamp(t)] + exec_lag
        if k < len(weeks):
            execs[k] = row.reindex(cols).fillna(0.0).to_numpy(float)
    if not execs:
        raise ValueError("no executable decision")
    first = min(execs)
    b = buy_bps.reindex(cols).to_numpy(float) / 1e4
    s = sell_bps.reindex(cols).to_numpy(float) / 1e4
    R = rets.to_numpy(float)
    V = np.zeros(len(cols))
    cash = initial
    rows = []
    for i in range(first, len(weeks)):
        if i > first:
            if np.any(np.isnan(R[i]) & (V != 0)):
                raise ValueError(f"missing return for a held sleeve in week {weeks[i]}")
            V = V * (1 + np.nan_to_num(R[i]))
        cost, traded = 0.0, 0.0
        if i in execs:
            total = cash + V.sum()
            tgt = execs[i]
            net = total
            for _ in range(50):  # fixed point: cost depends on the post-cost target
                d = tgt * net - V
                c = float(np.sum(np.where(d > 0, d * b, -d * s)))
                new = total - c
                if abs(new - net) < 1e-14 * max(total, 1.0):
                    break
                net = new
            d = tgt * net - V
            cost = float(np.sum(np.where(d > 0, d * b, -d * s)))
            traded = float(np.abs(d).sum())
            V = tgt * net
            cash = total - cost - V.sum()
        value = cash + V.sum()
        rows.append(
            {
                "week": weeks[i],
                "value": value,
                "cost": cost,
                "turnover": traded / value if value else 0.0,
                **{f"w_{c}": v / value for c, v in zip(cols, V, strict=True)},
            }
        )
    return pd.DataFrame(rows).set_index("week")


def _verdict_ratio(lo: float, hi: float) -> Conclusion:
    return "supported" if hi < 1 else ("rejected" if lo > 1 else "inconclusive")


def _verdict_diff(lo: float, hi: float) -> Conclusion:
    return "supported" if lo > 0 else ("rejected" if hi < 0 else "inconclusive")


def run_cross_market(
    paths: Paths | None = None, sleeves: dict[str, EwSleeve] | None = None
) -> dict[str, Any]:
    """Run H-XMKT-001. `sleeves` may be passed to reuse already-built EW sleeves."""
    t0 = time.perf_counter()
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    tracker = Tracker(paths)
    sleeves = sleeves or {m: ew_sleeve(m, paths) for m in SLEEVES}
    daily = {m: sleeves[m].full_net.equity_net for m in SLEEVES}
    fx = load_usdinr(paths)
    usd, audit = weekly_usd_sleeves(daily, fx)
    # drop weeks after the last common data and before every sleeve exists
    rets = usd.pct_change(fill_method=None)
    rets = rets.loc[: min(s.index.max() for s in daily.values())]
    weeks = monthly_weeks(rets.index)
    rows, fc_rows = {}, {}
    for t in weeks:
        w = inverse_vol_weights(rets, t)
        if w is None:
            continue
        rows[t] = w
        hist = rets.loc[rets.index <= t].tail(VOL_WINDOW)
        cov = hist.cov().to_numpy()
        fc_rows[t] = math.sqrt(float(w.to_numpy() @ cov @ w.to_numpy()) * PPY)
    mix_tg = pd.DataFrame(rows).T
    first_dec = mix_tg.index[0]
    us_tg = pd.DataFrame({"US": [1.0], "IN": [0.0], "CRYPTO": [0.0]}, index=[first_dec])
    buy = pd.Series({m: cfg.markets[m].costs_bps.buy_bps() for m in SLEEVES})  # type: ignore[index]
    sell = pd.Series({m: cfg.markets[m].costs_bps.sell_bps() for m in SLEEVES})  # type: ignore[index]
    mix = simulate_weekly(rets, mix_tg, buy, sell)
    mix_gross = simulate_weekly(rets, mix_tg, buy * 0, sell * 0)
    us = simulate_weekly(rets, us_tg, buy, sell)
    r_mix = mix["value"].pct_change().dropna()
    r_us = us["value"].pct_change().dropna()
    common = r_mix.index.intersection(r_us.index)
    r_mix, r_us = r_mix.loc[common], r_us.loc[common]
    boot = paired_bootstrap_ci(r_mix, r_us, PPY, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=SEED)
    sd = sharpe_difference_bootstrap(r_mix, r_us, PPY, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=SEED)
    v1a = _verdict_ratio(boot["vol_ratio_ci_lo"], boot["vol_ratio_ci_hi"])
    v1b = _verdict_diff(sd["ci_low"], sd["ci_high"])

    eq = pd.DataFrame({"mix": mix["value"], "us_only": us["value"], "mix_gross": mix_gross["value"]})
    years = len(r_mix) / PPY

    def perf(v: pd.Series, led: pd.DataFrame) -> dict[str, float]:
        s = summarize(v, PPY)
        return {
            **s,
            "max_drawdown": max_drawdown(v),
            "annual_turnover": float(led["turnover"].iloc[1:].sum() / years),
            "total_cost_pct_initial": float(led["cost"].sum() / 1.0),
        }

    p_mix, p_us = perf(mix["value"], mix), perf(us["value"], us)
    p_mix_g = summarize(mix_gross["value"], PPY)
    sleeve_stats = {
        m: {
            "ann_vol_usd": float(rets[m].loc[common].std(ddof=1) * math.sqrt(PPY)),
            "sharpe_usd": float(
                rets[m].loc[common].mean() / rets[m].loc[common].std(ddof=1) * math.sqrt(PPY)
            ),
        }
        for m in SLEEVES
    }
    corr = rets.loc[common].corr()
    fvr = forecast_vs_realized(pd.Series(fc_rows).loc[mix_tg.index], r_mix, PPY, min_obs=3)
    fvr_sum = forecast_realized_summary(fvr)
    aud = audit.loc[common]
    fx_lag_days = (aud["IN_session_date"] - aud["IN_fx_obs_date"]).dt.days
    alignment = {
        "week_rule": "W-FRI (Sat..Fri), last session <= Friday per market",
        "fx_rule": "USDINR obs with available_from < IN session date (strict)",
        "fx_obs_lag_days_median": float(fx_lag_days.median()),
        "fx_obs_lag_days_max": float(fx_lag_days.max()),
        "fx_obs_lag_days_p99": float(fx_lag_days.quantile(0.99)),
        "weeks_without_new_session": {
            m: int((aud[f"{m}_session_date"].diff().dt.days == 0).sum()) for m in SLEEVES
        },
        "weeks_us_not_friday": int((aud["US_session_date"].dt.dayofweek != 4).sum()),
        "weeks_in_not_friday": int((aud["IN_session_date"].dt.dayofweek != 4).sum()),
        "weeks_crypto_not_friday": int((aud["CRYPTO_session_date"].dt.dayofweek != 4).sum()),
        "exec_lag_weeks": 1,
        "rebalance": "monthly: first weekly label of each month",
        "usdt_as_usd": True,
    }
    period = f"{common[0].date()}..{common[-1].date()}"
    metrics_common = {
        "period": period,
        "n_weeks": len(common),
        "mix": p_mix,
        "us_only": p_us,
        "mix_gross_sharpe": p_mix_g["sharpe"],
        "mix_gross_cagr": p_mix_g["cagr"],
        "mix_cost_drag_cagr": p_mix_g["cagr"] - p_mix["cagr"],
        "sleeves": sleeve_stats,
        "corr": corr.to_dict(),
        "mean_weights_mix": mix_tg.mean().to_dict(),
        "forecast_vs_realized_mix": fvr_sum,
        "alignment": alignment,
        "vol_ratio": boot["vol_ratio"],
        "vol_ratio_ci_lo": boot["vol_ratio_ci_lo"],
        "vol_ratio_ci_hi": boot["vol_ratio_ci_hi"],
        "delta_sr": sd["delta_sr"],
        "delta_sr_ci_low": sd["ci_low"],
        "delta_sr_ci_high": sd["ci_high"],
        "delta_sr_p_two_sided": sd["p_two_sided"],
        "delta_mdd": p_mix["max_drawdown"] - p_us["max_drawdown"],
    }
    validation = {
        "type": "registered rule-based portfolio comparison, weekly USD",
        "bootstrap": f"paired stationary, block {MEAN_BLOCK} weeks, {N_BOOT} draws, seed {SEED}",
        "annualization": PPY,
        "rf": 0.0,
        "adr": "docs/adr/0007-cross-market-weekly-alignment.md",
    }
    datasets = [
        {"symbol": s, "market": m, "sha256": read_manifest(s, m, paths)["sha256"]}
        for m in SLEEVES
        for s in sleeves[m].panel.symbols
    ] + [{"symbol": "USDINR", "source": "data/validated/macro/market.parquet (Yahoo INR=X)"}]
    notes = (
        SURVIVORSHIP_NOTE + " Deviation: registration named FRED DEXINUS; Yahoo INR=X (t+2 availability) "
        "used because FRED is manual-CSV only. USDT treated as USD. INR conversion costs not modelled. "
        "Sleeves are non-synchronous within a week (IN 10:00 UTC, US ~21:00 UTC, crypto 24:00 UTC Fri)."
    )
    model = {
        "sleeves": "EW monthly (local), names enter after 252 valid returns",
        "mix": "inverse trailing-52w vol, monthly, 1-week execution lag",
        "costs_bps": {"buy": buy.to_dict(), "sell": sell.to_dict()},
    }
    reproduce = (
        'uv run --no-sync python -c "from astra.research.cross_market import run_cross_market; '
        'run_cross_market()"'
    )
    rec_a = tracker.new(
        "xmkt_vol_ratio",
        HYP,
        "H1a: realized vol(equal-risk 3-sleeve mix) < vol(US-only) (bootstrap CI of ratio < 1).",
        datasets=datasets,
        model=model,
        validation=validation,
        costs={"buy_bps": buy.to_dict(), "sell_bps": sell.to_dict()},
        seed=SEED,
        metrics=metrics_common,
        conclusion=v1a,
        variants_tried=2,
        notes=notes,
        reproduce=reproduce,
    )
    weights = mix[[c for c in mix.columns if c.startswith("w_")]]
    tracker.save_frame(rec_a, "equity_weekly", eq)
    tracker.save_frame(rec_a, "weekly_usd_returns", rets)
    tracker.save_frame(rec_a, "mix_weights_eow", weights)
    tracker.save_frame(rec_a, "mix_targets", mix_tg)
    tracker.save_frame(rec_a, "alignment_audit", audit)
    tracker.save_frame(rec_a, "forecast_vs_realized", fvr.reset_index())
    tracker.log(rec_a)
    rec_b = tracker.new(
        "xmkt_sharpe",
        HYP,
        "H1b: Sharpe(equal-risk 3-sleeve mix) > Sharpe(US-only) (bootstrap CI of difference > 0).",
        datasets=datasets,
        model=model,
        validation=validation,
        costs={"buy_bps": buy.to_dict(), "sell_bps": sell.to_dict()},
        seed=SEED,
        metrics={**metrics_common, "artifacts_in_experiment": rec_a.experiment_id},
        conclusion=v1b,
        variants_tried=2,
        notes=notes,
        reproduce=reproduce,
        artifacts=list(rec_a.artifacts),
    )
    tracker.log(rec_b)
    return {
        "H1a": {
            "conclusion": v1a,
            "id": rec_a.experiment_id,
            **{k: boot[k] for k in ("vol_ratio", "vol_ratio_ci_lo", "vol_ratio_ci_hi")},
        },
        "H1b": {"conclusion": v1b, "id": rec_b.experiment_id, **sd},
        "metrics": metrics_common,
        "runtime_s": time.perf_counter() - t0,
    }
