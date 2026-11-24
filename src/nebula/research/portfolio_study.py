"""M8 portfolio & risk study: per-market EW / MinVar / RP (tested) and MV / MaxSharpe (descriptive).

Hypotheses registered BEFORE results in docs/research/H-PF.md:
- H-PF-001  MinVar and RP have lower realized volatility than EW (paired stationary-bootstrap 95% CI of
            the vol ratio; upper < 1 supported, lower > 1 rejected, else inconclusive).
- H-PF-002  MinVar and RP have higher net Sharpe than EW. Registered decision test (same as the vertical
            slice): Newey-West t on daily net return differences, |t| > 1.96. Secondary, non-decisive:
            paired bootstrap CI of the Sharpe difference.
Per market the record verdict is: supported iff both strategies supported; rejected if either rejected;
else inconclusive. Fixed a priori: 252-obs window, Ledoit-Wolf, monthly rebalance (decision at the close
of the first session of each month, fill next open), caps 0.4 equities / 0.6 crypto, risk aversion 5,
mean shrinkage 0.5, rf = 0.

Portfolios are built PER MARKET in local currency (USD, INR, USDT) - no FX conversion, no mixing of
calendars. Annualization: 252 (US, IN), 365 (CRYPTO).
"""

from __future__ import annotations

import math
import time
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from nebula.common.config import get_config
from nebula.common.logging import get_logger
from nebula.common.paths import Paths, get_paths
from nebula.common.seed import set_seed
from nebula.data.storage import read_manifest
from nebula.portfolio.backtest import (
    MarketPanel,
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    load_market_panel,
    monthly_decision_dates,
    run_portfolio_backtest,
    run_portfolio_gross,
)
from nebula.portfolio.construction import (
    WeightFn,
    compute_targets,
    ew_fn,
    make_max_sharpe_fn,
    make_mv_fn,
    make_vol_target_fn,
    minvar_fn,
    rp_fn,
)
from nebula.portfolio.covariance import CovMethod, estimate_covariance
from nebula.portfolio.optimizers import default_cap
from nebula.risk.attribution import (
    effective_n,
    effective_number_of_bets,
    forecast_realized_summary,
    forecast_vs_realized,
    paired_bootstrap_ci,
    risk_contributions,
    var_es_table,
)
from nebula.risk.metrics import simple_returns, summarize
from nebula.stats.inference import newey_west_tstat
from nebula.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)

SEED = 7
UNIVERSES: dict[str, list[str]] = {
    "US": ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"],
    "IN": ["RELIANCE", "TCS", "INFY", "HDFCBANK"],  # NIFTY50 is the benchmark, not tradable
    "CRYPTO": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
}
WINDOW = 252
MIN_HISTORY = 252
COV_METHOD: CovMethod = "ledoit_wolf"
RISK_AVERSION = 5.0
MEAN_SHRINK = 0.5
VT_TARGET = {"US": 0.10, "IN": 0.10, "CRYPTO": 0.40}  # descriptive overlay only
N_BOOT = 2000
MEAN_BLOCK = 20
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)
TESTED = ("minvar", "rp")
SURVIVORSHIP_NOTE = (
    "Survivorship: the universe is today's well-known survivors (mega-cap US tech/ETFs, top Indian large "
    "caps, BTC/ETH/SOL), chosen with hindsight; absolute returns are biased upward for every strategy and "
    "relative results are conditional on this universe."
)


def _strategies(market: str, ppy: int) -> dict[str, WeightFn]:
    return {
        "ew": ew_fn,
        "minvar": minvar_fn,
        "rp": rp_fn,
        "max_sharpe": make_max_sharpe_fn(MEAN_SHRINK),
        "mv": make_mv_fn(RISK_AVERSION, MEAN_SHRINK),
        "rp_vt": make_vol_target_fn(rp_fn, VT_TARGET[market], ppy),
    }


def _verdict_t(t: float) -> Conclusion:
    if t > 1.96:
        return "supported"
    if t < -1.96:
        return "rejected"
    return "inconclusive"


def _verdict_ci(lo: float, hi: float) -> Conclusion:
    if hi < 1.0:
        return "supported"
    if lo > 1.0:
        return "rejected"
    return "inconclusive"


def _combine(sub: list[Conclusion]) -> Conclusion:
    if all(s == "supported" for s in sub):
        return "supported"
    if any(s == "rejected" for s in sub):
        return "rejected"
    return "inconclusive"


def _perf(res: PortfolioBacktestResult, gross: PortfolioBacktestResult | None, ppy: int) -> dict[str, Any]:
    eq = res.equity_net
    s = summarize(eq, ppy)
    years = (len(eq) - 1) / ppy
    out = {
        **s,
        "annual_turnover_two_sided": float(res.ledger["turnover"].sum() / years) if years > 0 else math.nan,
        "total_cost": float(res.ledger["cost"].sum()),
        "total_cost_pct_initial": float(res.ledger["cost"].sum() / eq.iloc[0]),
        "avg_exposure": float(res.ledger["exposure"].iloc[1:].mean()),
        "n_fills": len(res.fills),
        "n_capped_fills": int(sum(f.capped for f in res.fills)),
        "n_cash_scaled_fills": int(sum(f.cash_scaled for f in res.fills)),
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
    }
    if gross is not None:
        g = summarize(gross.equity_net, ppy)
        out |= {"gross_cagr": g["cagr"], "gross_sharpe": g["sharpe"], "gross_ann_vol": g["ann_vol"]}
    return out


def _concentration(
    returns: pd.DataFrame, targets: pd.DataFrame, window: int, method: CovMethod
) -> dict[str, float]:
    """Average weight- and risk-based concentration of the targets at decision dates (ex ante)."""
    en, enb, max_rc = [], [], []
    for t, row in targets.iterrows():
        w = row[row > 0]
        est = estimate_covariance(
            returns[list(w.index)], t, window=window, method=method, min_obs=MIN_HISTORY
        )
        wv = w.to_numpy() / w.sum()
        en.append(effective_n(wv))
        enb.append(effective_number_of_bets(wv, est.cov))
        max_rc.append(float(risk_contributions(wv, est.cov)["pct"].max()))
    return {
        "mean_effective_n": float(np.mean(en)),
        "mean_effective_number_of_bets": float(np.mean(enb)),
        "mean_max_risk_share": float(np.mean(max_rc)),
    }


def _last_risk_contributions(
    returns: pd.DataFrame, targets: dict[str, pd.DataFrame], method: CovMethod
) -> pd.DataFrame:
    rows = []
    for name, tg in targets.items():
        t = tg.index[-1]
        w = tg.loc[t]
        w = w[w > 0]
        est = estimate_covariance(returns[list(w.index)], t, window=WINDOW, method=method)
        rc = risk_contributions(w.to_numpy(), est.cov)
        for sym, (_, r) in zip(w.index, rc.iterrows(), strict=True):
            rows.append({"strategy": name, "decision_date": t, "symbol": sym, **r.to_dict()})
    return pd.DataFrame(rows)


def run_market(market: str, paths: Paths | None = None, tracker: Tracker | None = None) -> dict[str, Any]:
    set_seed(SEED)
    t_start = time.perf_counter()
    paths = paths or get_paths()
    tracker = tracker or Tracker(paths)
    cfg = get_config()
    mkt = cfg.markets[market]  # type: ignore[index]
    ppy = mkt.periods_per_year
    syms = UNIVERSES[market]
    panel: MarketPanel = load_market_panel(market, syms, paths, window=get_config().core_window())
    rets = panel.returns()
    dates = monthly_decision_dates(panel.close.index)
    cap = default_cap(market)
    datasets = [read_manifest(s, market, paths) for s in syms]
    bt_cfg = PortfolioBacktestConfig(costs=mkt.costs_bps)
    costs_dict = {
        **mkt.costs_bps.model_dump(),
        "max_participation": bt_cfg.max_participation,
        "rebalance_tolerance": bt_cfg.rebalance_tolerance,
        "units": "bps per side of traded notional",
    }

    strategies = _strategies(market, ppy)
    targets: dict[str, pd.DataFrame] = {}
    diags: dict[str, pd.DataFrame] = {}
    for name, fn in strategies.items():
        targets[name], diags[name] = compute_targets(
            rets, dates, fn, cap, ppy, window=WINDOW, method=COV_METHOD, min_history=MIN_HISTORY
        )
    start = max(tg.index[0] for tg in targets.values())
    targets = {k: v.loc[v.index >= start] for k, v in targets.items()}

    net: dict[str, PortfolioBacktestResult] = {}
    gross: dict[str, PortfolioBacktestResult] = {}
    perf: dict[str, dict[str, Any]] = {}
    for name, tg in targets.items():
        net[name] = run_portfolio_backtest(panel, tg, bt_cfg, start=start)
        gross[name] = run_portfolio_gross(panel, tg, bt_cfg, start=start)
        perf[name] = _perf(net[name], gross[name], ppy)
        perf[name] |= _concentration(rets, tg, WINDOW, COV_METHOD)
        if "fallback_minvar" in diags[name]:
            perf[name]["n_max_sharpe_fallbacks"] = int(diags[name]["fallback_minvar"].sum())
    r = {k: simple_returns(v.equity_net) for k, v in net.items()}

    # forecast (ex ante, LW) vs realized risk, VaR/ES
    fvr: dict[str, dict[str, float]] = {}
    fvr_tables, var_tables = [], []
    for name in targets:
        tab = forecast_vs_realized(diags[name].loc[targets[name].index, "forecast_vol_ann"], r[name], ppy)
        fvr[name] = forecast_realized_summary(tab) | {"full_period_realized_vol_ann": perf[name]["ann_vol"]}
        fvr_tables.append(tab.assign(strategy=name))
        var_tables.append(var_es_table(r[name]).assign(strategy=name))

    # hypothesis statistics (vs EW), on daily net returns over the common period
    tests: dict[str, dict[str, Any]] = {}
    for name in TESTED:
        boot = paired_bootstrap_ci(r[name], r["ew"], ppy, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=SEED)
        diff = (r[name] - r["ew"]).dropna()
        sq_diff = (r["ew"] ** 2 - r[name] ** 2).dropna()
        t_ret = newey_west_tstat(diff)
        tests[name] = {
            **boot,
            "nw_t_return_diff": t_ret,
            "nw_t_sq_return_diff": newey_west_tstat(sq_diff),
            "mean_daily_return_diff": float(diff.mean()),
            "verdict_vol": _verdict_ci(boot["vol_ratio_ci_lo"], boot["vol_ratio_ci_hi"]),
            "verdict_sharpe_registered": _verdict_t(t_ret),
        }

    # descriptive sensitivity: cost multipliers (all strategies), covariance method x window (MinVar, RP)
    sens_rows = []
    for name, tg in targets.items():
        for m in COST_MULTIPLIERS:
            res = run_portfolio_backtest(panel, tg, replace(bt_cfg, cost_multiplier=m), start=start)
            p = _perf(res, None, ppy)
            sens_rows.append({"strategy": name, "cost_multiplier": m, **p})
    cost_sens = pd.DataFrame(sens_rows)
    cov_rows = []
    for method in ("sample", "ledoit_wolf"):
        for win in (126, 252, 504):
            for name, fn in (("minvar", minvar_fn), ("rp", rp_fn)):
                tg, _ = compute_targets(rets, dates, fn, cap, ppy, window=win, method=method, min_history=252)
                tg = tg.loc[tg.index >= start]
                p = _perf(run_portfolio_backtest(panel, tg, bt_cfg, start=start), None, ppy)
                cov_rows.append({"strategy": name, "cov_method": method, "window": win, **p})
    cov_sens = pd.DataFrame(cov_rows)

    # artifact frames
    eq = pd.DataFrame(
        {f"{k}_equity_net": v.equity_net for k, v in net.items()}
        | {f"{k}_equity_gross": v.equity_net for k, v in gross.items()}
    )
    weights = pd.concat({k: v.weights for k, v in net.items()}, axis=1)
    weights.columns = [f"{a}:{b}" for a, b in weights.columns]
    tg_all = pd.concat(targets, axis=0, names=["strategy", "decision_date"]).reset_index()
    diag_all = pd.concat(diags, axis=0, names=["strategy", "decision_date"]).reset_index()
    diag_all = diag_all.astype({c: str for c in diag_all.columns if diag_all[c].dtype == object})
    perf_df = pd.DataFrame(perf).T.reset_index(names="strategy")
    perf_df = perf_df.astype({c: str for c in ("start", "end")})
    rc_last = _last_risk_contributions(rets, targets, COV_METHOD)
    cond = diags["ew"]["condition_number"]

    common_validation = {
        "type": "full-sample rule-based portfolios (no fitted parameters beyond trailing estimates)",
        "period": f"{start.date()}..{panel.close.index[-1].date()}",
        "rebalance": "monthly: decision at close of first session of month, fill at next open",
        "covariance": f"{COV_METHOD}, trailing {WINDOW} complete daily rows ending at t (inclusive)",
        "universe_entry": f">= {MIN_HISTORY} valid daily returns before inclusion; no backfilling",
        "caps": cap,
        "annualization": ppy,
        "currency": mkt.currency,
        "bootstrap": {"type": "paired stationary", "n_boot": N_BOOT, "mean_block": MEAN_BLOCK, "seed": SEED},
        "condition_number_median": float(cond.median()),
        "condition_number_max": float(cond.max()),
    }
    notes = (
        f"market={market}; per-market portfolio in {mkt.currency}, no FX. {SURVIVORSHIP_NOTE} "
        f"Excluded placeholder rows: {panel.excluded_rows}. Max-Sharpe/MV/rp_vt are descriptive only "
        "(mean estimates are noise). Sensitivity tables are descriptive and do not change verdicts."
    )
    model = {
        "strategies": list(strategies),
        "risk_aversion": RISK_AVERSION,
        "mean_shrinkage": MEAN_SHRINK,
        "vol_target": VT_TARGET[market],
        "solver": "CVXPY CLARABEL (SCS fallback)",
        "risk_parity": "Spinu 2013 log-barrier, equal budgets, caps as linear constraints",
    }

    sub1: list[Conclusion] = [tests[n]["verdict_vol"] for n in TESTED]
    rec1 = tracker.new(
        f"portfolio_vol_{market}",
        "H-PF-001",
        "MinVar and RP have lower realized volatility than EW (paired bootstrap CI of vol ratio).",
        datasets=datasets,
        model=model,
        validation=common_validation,
        costs=costs_dict,
        seed=SEED,
        metrics={
            "market": market,
            "tests": {n: {k: v for k, v in tests[n].items() if "sharpe" not in k} for n in TESTED},
            "performance": perf,
            "forecast_vs_realized": fvr,
            "minvar_vol_ratio": tests["minvar"]["vol_ratio"],
            "rp_vol_ratio": tests["rp"]["vol_ratio"],
            "ew_ann_vol": perf["ew"]["ann_vol"],
            "minvar_ann_vol": perf["minvar"]["ann_vol"],
            "rp_ann_vol": perf["rp"]["ann_vol"],
        },
        conclusion=_combine(sub1),
        variants_tried=len(TESTED),
        notes=notes,
        reproduce=f'uv run --no-sync python -c "from nebula.research.portfolio_study import '
        f"run_portfolio_study; run_portfolio_study(['{market}'])\"",
    )
    tracker.save_frame(rec1, "equity", eq)
    tracker.save_frame(rec1, "weights", weights)
    tracker.save_frame(rec1, "targets", tg_all)
    tracker.save_frame(rec1, "decision_diagnostics", diag_all)
    tracker.save_frame(rec1, "performance", perf_df)
    tracker.save_frame(rec1, "cost_sensitivity", cost_sens)
    tracker.save_frame(rec1, "cov_window_sensitivity", cov_sens)
    tracker.save_frame(rec1, "forecast_vs_realized", pd.concat(fvr_tables).reset_index())
    tracker.save_frame(rec1, "var_es", pd.concat(var_tables, ignore_index=True))
    tracker.save_frame(rec1, "risk_contributions_last", rc_last)
    tracker.log(rec1)

    sub2: list[Conclusion] = [tests[n]["verdict_sharpe_registered"] for n in TESTED]
    rec2 = tracker.new(
        f"portfolio_sharpe_{market}",
        "H-PF-002",
        "MinVar and RP have higher net Sharpe than EW (registered test: NW t on daily net return diff).",
        datasets=datasets,
        model=model,
        validation=common_validation,
        costs=costs_dict,
        seed=SEED,
        metrics={
            "market": market,
            "tests": {
                n: {
                    "nw_t_return_diff": tests[n]["nw_t_return_diff"],
                    "mean_daily_return_diff": tests[n]["mean_daily_return_diff"],
                    "sharpe_diff": tests[n]["sharpe_diff"],
                    "sharpe_diff_ci_lo": tests[n]["sharpe_diff_ci_lo"],
                    "sharpe_diff_ci_hi": tests[n]["sharpe_diff_ci_hi"],
                    "verdict": tests[n]["verdict_sharpe_registered"],
                }
                for n in TESTED
            },
            "performance": perf,
            "artifacts_in_experiment": rec1.experiment_id,
            "ew_sharpe": perf["ew"]["sharpe"],
            "minvar_sharpe": perf["minvar"]["sharpe"],
            "rp_sharpe": perf["rp"]["sharpe"],
            "max_sharpe_sharpe": perf["max_sharpe"]["sharpe"],
            "mv_sharpe": perf["mv"]["sharpe"],
        },
        conclusion=_combine(sub2),
        variants_tried=len(TESTED),
        notes=notes + " The registered test is on mean return differences, not Sharpe (disclosed at "
        "registration); the bootstrap Sharpe-difference CI is secondary and non-decisive.",
        reproduce=rec1.reproduce,
        artifacts=list(rec1.artifacts),
    )
    tracker.log(rec2)

    out = {
        "market": market,
        "currency": mkt.currency,
        "periods_per_year": ppy,
        "period": common_validation["period"],
        "performance": perf,
        "tests": tests,
        "forecast_vs_realized": fvr,
        "H-PF-001": {
            "conclusion": rec1.conclusion,
            "id": rec1.experiment_id,
            "sub": dict(zip(TESTED, sub1, strict=True)),
        },
        "H-PF-002": {
            "conclusion": rec2.conclusion,
            "id": rec2.experiment_id,
            "sub": dict(zip(TESTED, sub2, strict=True)),
        },
        "runtime_s": time.perf_counter() - t_start,
    }
    log.info("portfolio_study.market_done", market=market, runtime_s=round(out["runtime_s"], 1))
    return out


def run_portfolio_study(markets: list[str] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    """Run the M8 study per market (default: US, IN, CRYPTO). Returns {market: summary, 'runtime_s': s}."""
    t0 = time.perf_counter()
    paths = paths or get_paths()
    tracker = Tracker(paths)
    out: dict[str, Any] = {}
    for m in markets or list(UNIVERSES):
        out[m] = run_market(m, paths, tracker)
    out["runtime_s"] = time.perf_counter() - t0
    return out
