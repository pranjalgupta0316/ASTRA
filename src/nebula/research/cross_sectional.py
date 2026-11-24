"""H-CS-001 cross-sectional momentum and H-CS-002 short-term reversal (registered in docs/research/H-EXT.md).

Universes (today's names -> survivorship-biased): US stocks (kind=stock, 30), IN stocks (kind=stock, 49;
NIFTY50 index excluded), crypto (16 pairs). Full history (window=None). Per market, per hypothesis:
  strategy   = equal-weight long the top tercile of the signal among eligible names,
  benchmark  = equal-weight all eligible names (same dates, same engine, same costs),
decided at the close of each rebalance date, filled at the next open by `nebula.portfolio.backtest`.
Signals: equities 12-1 momentum log(C[t-21]/C[t-252]); crypto 4-week momentum log(C[t-1]/C[t-28]);
reversal -(C[t]/C[t-5]-1). Rebalance: equity momentum monthly (first session of month); reversal and
all crypto signals weekly (first session of ISO week = Monday for crypto).
Calendar: union of validated session dates, minus rows where < 50% of already-listed names have a bar
(IN has 16 such single-name rows, mostly 2003-2005 exchange holidays; US and crypto have none).
Eligibility: valid bar at t, >= 252 valid bars up to t, finite signal (no backfilling).
Decision: Newey-West t on daily net return difference (strategy - benchmark); > 1.96 supported,
< -1.96 rejected, else inconclusive; Benjamini-Hochberg (q = 0.05) across the 3 markets per hypothesis.
Descriptive (non-decisive): gross vs net, turnover, 0/1/2/4x costs, halves, bootstrap Sharpe difference,
top-minus-bottom gross return series (no short execution - the engine is long-only).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

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
    run_portfolio_backtest,
)
from nebula.research.signals import (
    eligibility,
    equal_weight_targets,
    momentum_signal,
    monthly_first_sessions,
    n_rankable,
    regular_sessions,
    reversal_signal,
    weekly_first_sessions,
)
from nebula.risk.attribution import paired_bootstrap_ci
from nebula.risk.metrics import simple_returns, summarize
from nebula.stats.inference import newey_west_tstat
from nebula.tracking.tracker import Conclusion, ExperimentRecord, Tracker

log = get_logger(__name__)

SEED = 7
MARKETS = ("US", "IN", "CRYPTO")
KIND = {"US": "stock", "IN": "stock", "CRYPTO": "crypto"}
MIN_HISTORY = 252
MIN_COVERAGE = 0.5  # calendar rows where < 50% of listed names have a bar are not sessions (IN: 16 rows)
SUPERSEDED_RUN = "20261124T200122 (same code without the session filter; IN cells affected)"
MIN_RANKABLE = 6  # study starts at the first rebalance with >= 6 rankable names (>= 2 per tercile)
COST_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)
N_BOOT = 2000
MEAN_BLOCK = 20
BH_Q = 0.05
T_CRIT = 1.96
HYPOTHESES = {
    "H-CS-001": "Cross-sectional momentum: EW top-tercile net daily returns exceed the EW eligible universe.",
    "H-CS-002": "Short-term reversal (past 5-session losers): EW top-tercile net daily returns exceed the "
    "EW eligible universe.",
}
SURVIVORSHIP_NOTE = (
    "SURVIVORSHIP BIAS: universes are TODAY's large caps / NIFTY constituents / crypto majors applied to "
    "full history; names that failed, were delisted or dropped out are absent. Absolute returns of both "
    "strategy and benchmark are biased upward, and momentum/reversal relative results are conditional on "
    "this hindsight universe (losers that later recovered are over-represented)."
)


@dataclass(frozen=True)
class SignalSpec:
    name: str
    rebalance: str  # "monthly" | "weekly"
    description: str


def signal_spec(hypothesis: str, market: str) -> SignalSpec:
    crypto = market == "CRYPTO"
    if hypothesis == "H-CS-001":
        if crypto:
            return SignalSpec("mom_4w", "weekly", "log(C[t-1]/C[t-28]) (4-week momentum, daily rows)")
        return SignalSpec("mom_12_1", "monthly", "log(C[t-21]/C[t-252]) (12-1 momentum, sessions)")
    if hypothesis == "H-CS-002":
        return SignalSpec("rev_5", "weekly", "-(C[t]/C[t-5]-1) (5-session reversal)")
    raise KeyError(hypothesis)


def compute_signal(hypothesis: str, market: str, close: pd.DataFrame) -> pd.DataFrame:
    if hypothesis == "H-CS-001":
        return momentum_signal(close, 28, 1) if market == "CRYPTO" else momentum_signal(close, 252, 21)
    return reversal_signal(close, 5)


def rebalance_dates(spec: SignalSpec, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return monthly_first_sessions(index) if spec.rebalance == "monthly" else weekly_first_sessions(index)


def bh_verdicts(tstats: dict[str, float], q: float = BH_Q) -> dict[str, dict[str, Any]]:
    """Two-sided normal p-values of NW t, BH-adjusted across the family. BH verdict: supported if
    rejected by BH with t > 0, rejected if rejected by BH with t < 0, else inconclusive."""
    keys = list(tstats)
    t = np.array([tstats[k] for k in keys], dtype=float)
    p = 2 * stats.norm.sf(np.abs(np.nan_to_num(t, nan=0.0)))
    rej, p_adj, *_ = multipletests(p, alpha=q, method="fdr_bh")
    out = {}
    for k, ti, pi, pa, r in zip(keys, t, p, p_adj, rej, strict=True):
        v: Conclusion = "inconclusive"
        if r and ti > 0:
            v = "supported"
        elif r and ti < 0:
            v = "rejected"
        out[k] = {"p_two_sided": float(pi), "p_bh": float(pa), "bh_reject": bool(r), "verdict_bh": v}
    return out


def verdict_t(t: float) -> Conclusion:
    if t > T_CRIT:
        return "supported"
    if t < -T_CRIT:
        return "rejected"
    return "inconclusive"


def _perf(res: PortfolioBacktestResult, ppy: int) -> dict[str, Any]:
    eq = res.equity_net
    years = (len(eq) - 1) / ppy
    s = summarize(eq, ppy)
    return {
        **s,
        "annual_turnover_two_sided": float(res.ledger["turnover"].sum() / years) if years > 0 else math.nan,
        "total_cost_pct_initial": float(res.ledger["cost"].sum() / eq.iloc[0]),
        "avg_exposure": float(res.ledger["exposure"].iloc[1:].mean()),
        "n_fills": len(res.fills),
        "n_capped_fills": int(sum(f.capped for f in res.fills)),
        "n_cash_scaled_fills": int(sum(f.cash_scaled for f in res.fills)),
    }


def _diff_stats(a: pd.Series, b: pd.Series, ppy: int) -> dict[str, float]:
    d = (a - b).dropna()
    return {
        "nw_t": newey_west_tstat(d),
        "mean_daily_diff": float(d.mean()),
        "ann_mean_diff": float(d.mean() * ppy),
        "n_days": len(d),
    }


def _seg_stats(r_s: pd.Series, r_b: pd.Series, ppy: int) -> dict[str, Any]:
    eq_s, eq_b = (1 + r_s).cumprod(), (1 + r_b).cumprod()
    eq_s = pd.concat([pd.Series([1.0]), eq_s.reset_index(drop=True)])
    eq_b = pd.concat([pd.Series([1.0]), eq_b.reset_index(drop=True)])
    ss, sb = summarize(eq_s, ppy), summarize(eq_b, ppy)
    return {
        "start": str(r_s.index[0].date()),
        "end": str(r_s.index[-1].date()),
        "strategy_cagr": ss["cagr"],
        "strategy_sharpe": ss["sharpe"],
        "benchmark_cagr": sb["cagr"],
        "benchmark_sharpe": sb["sharpe"],
        **_diff_stats(r_s, r_b, ppy),
    }


def run_cell(
    hypothesis: str, market: str, panel: MarketPanel, bt_cfg: PortfolioBacktestConfig, ppy: int
) -> dict:
    """All computations for one (hypothesis, market) cell; no tracker side effects."""
    spec = signal_spec(hypothesis, market)
    close = panel.close
    sig = compute_signal(hypothesis, market, close)
    elig = eligibility(close, MIN_HISTORY)
    dates_all = rebalance_dates(spec, close.index)
    counts = n_rankable(sig.loc[dates_all], elig.loc[dates_all])
    ok = counts[counts >= MIN_RANKABLE]
    if ok.empty:
        raise RuntimeError(f"{hypothesis}/{market}: never >= {MIN_RANKABLE} rankable names")
    start = pd.Timestamp(ok.index[0])
    dates = dates_all[dates_all >= start]
    tg = {leg: equal_weight_targets(sig, elig, dates, leg) for leg in ("top", "all", "bottom")}

    runs: dict[tuple[str, float], PortfolioBacktestResult] = {}
    for m in COST_MULTIPLIERS:
        for leg in ("top", "all"):
            runs[(leg, m)] = run_portfolio_backtest(panel, tg[leg], replace(bt_cfg, cost_multiplier=m), start)
    runs[("bottom", 0.0)] = run_portfolio_backtest(
        panel, tg["bottom"], replace(bt_cfg, cost_multiplier=0.0), start
    )
    runs[("bottom", 1.0)] = run_portfolio_backtest(panel, tg["bottom"], bt_cfg, start)

    r = {k: simple_returns(v.equity_net) for k, v in runs.items()}
    r_s, r_b = r[("top", 1.0)], r[("all", 1.0)]
    main = _diff_stats(r_s, r_b, ppy)
    boot = paired_bootstrap_ci(r_s, r_b, ppy, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=SEED)

    perf = {
        "strategy_net": _perf(runs[("top", 1.0)], ppy),
        "strategy_gross": _perf(runs[("top", 0.0)], ppy),
        "benchmark_net": _perf(runs[("all", 1.0)], ppy),
        "benchmark_gross": _perf(runs[("all", 0.0)], ppy),
        "bottom_net": _perf(runs[("bottom", 1.0)], ppy),
        "bottom_gross": _perf(runs[("bottom", 0.0)], ppy),
    }
    gross_diff = _diff_stats(r[("top", 0.0)], r[("all", 0.0)], ppy)

    cost_rows = []
    for m in COST_MULTIPLIERS:
        ps, pb = _perf(runs[("top", m)], ppy), _perf(runs[("all", m)], ppy)
        d = _diff_stats(r[("top", m)], r[("all", m)], ppy)
        cost_rows.append(
            {
                "cost_multiplier": m,
                "strategy_cagr": ps["cagr"],
                "strategy_sharpe": ps["sharpe"],
                "benchmark_cagr": pb["cagr"],
                "benchmark_sharpe": pb["sharpe"],
                "nw_t": d["nw_t"],
                "ann_mean_diff": d["ann_mean_diff"],
                "verdict_rule": verdict_t(d["nw_t"]),
            }
        )
    cost_sens = pd.DataFrame(cost_rows)

    n = len(r_s)
    half = n // 2
    sub_rows = [
        {"half": "first", **_seg_stats(r_s.iloc[:half], r_b.iloc[:half], ppy)},
        {"half": "second", **_seg_stats(r_s.iloc[half:], r_b.iloc[half:], ppy)},
    ]
    subperiods = pd.DataFrame(sub_rows)

    # descriptive long-short: daily top-gross minus bottom-gross return (legs rebalanced separately by the
    # engine; equivalent to a $1 long / $1 short book re-levered daily; no borrow cost, not executable)
    ls = (r[("top", 0.0)] - r[("bottom", 0.0)]).dropna()
    ls_eq = pd.concat([pd.Series([1.0], index=[start]), (1 + ls).cumprod()])
    ls_sum = summarize(ls_eq, ppy)
    long_short = {
        "ann_mean": float(ls.mean() * ppy),
        "ann_vol": ls_sum["ann_vol"],
        "sharpe": ls_sum["sharpe"],
        "cagr_compounded": ls_sum["cagr"],
        "max_drawdown": ls_sum["max_drawdown"],
        "nw_t": newey_west_tstat(ls),
    }

    equity = pd.DataFrame(
        {
            "strategy_net": runs[("top", 1.0)].equity_net,
            "strategy_gross": runs[("top", 0.0)].equity_net,
            "benchmark_net": runs[("all", 1.0)].equity_net,
            "benchmark_gross": runs[("all", 0.0)].equity_net,
            "bottom_gross": runs[("bottom", 0.0)].equity_net,
        }
    )
    equity["long_short_gross_ret"] = ls.reindex(equity.index)
    targets = pd.concat(tg, axis=0, names=["leg", "decision_date"]).reset_index()
    return {
        "hypothesis": hypothesis,
        "market": market,
        "spec": spec,
        "start": start,
        "end": close.index[-1],
        "n_rebalances": len(dates),
        "rankable_min": int(counts.loc[dates].min()),
        "rankable_max": int(counts.loc[dates].max()),
        "rankable_counts": counts.loc[dates].rename("n_rankable").to_frame(),
        "main": main,
        "verdict_rule": verdict_t(main["nw_t"]),
        "gross_diff": gross_diff,
        "bootstrap_sharpe": boot,
        "perf": perf,
        "cost_sensitivity": cost_sens,
        "subperiods": subperiods,
        "long_short": long_short,
        "equity": equity,
        "targets": targets,
    }


def _jsonable(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in row.items()}
        for row in df.to_dict("records")
    ]


def run_cross_sectional(markets: list[str] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    """Run H-CS-001 and H-CS-002 on each market (default US, IN, CRYPTO); log one Tracker record per
    hypothesis per market plus one family record per hypothesis. Returns a summary dict."""
    t0 = time.perf_counter()
    set_seed(SEED)
    paths = paths or get_paths()
    tracker = Tracker(paths)
    cfg = get_config()
    markets = list(markets or MARKETS)

    cells: dict[tuple[str, str], dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for market in markets:
        mkt = cfg.markets[market]  # type: ignore[index]
        syms = cfg.symbols(market=market, kind=KIND[market])
        raw = load_market_panel(market, syms, paths, window=None)
        keep = regular_sessions(raw.close, MIN_COVERAGE)
        panel = MarketPanel(
            market, raw.open.loc[keep], raw.close.loc[keep], raw.volume.loc[keep], raw.excluded_rows
        )
        # Full rebalance on every decision date. An absolute 2pp tolerance (the engine default) never
        # topped up 2-4% underweights, leaving the EW benchmark 11-17% in cash (independent audit,
        # 2026-11-25) and inflating H-CS excess returns. Tolerance 0 keeps both legs fully invested.
        bt_cfg = PortfolioBacktestConfig(costs=mkt.costs_bps, rebalance_tolerance=0.0)
        meta[market] = {
            "symbols": syms,
            "currency": mkt.currency,
            "ppy": mkt.periods_per_year,
            "datasets": [read_manifest(s, market, paths) for s in syms],
            "excluded_rows": {k: len(v) for k, v in panel.excluded_rows.items() if v},
            "costs": {
                **mkt.costs_bps.model_dump(),
                "round_trip_bps": mkt.costs_bps.buy_bps() + mkt.costs_bps.sell_bps(),
                "max_participation": bt_cfg.max_participation,
                "rebalance_tolerance": bt_cfg.rebalance_tolerance,
                "units": "bps per side of traded notional",
            },
            "holdout_overlap": sorted(set(syms) & set(cfg.holdout)),
            "dropped_sessions": [str(d.date()) for d in raw.close.index[~keep.to_numpy()]],
        }
        for hyp in HYPOTHESES:
            tc = time.perf_counter()
            cells[(hyp, market)] = run_cell(hyp, market, panel, bt_cfg, mkt.periods_per_year)
            log.info(
                "cross_sectional.cell_done",
                hypothesis=hyp,
                market=market,
                s=round(time.perf_counter() - tc, 1),
            )

    out: dict[str, Any] = {}
    for hyp, text in HYPOTHESES.items():
        hm = [m for m in markets if (hyp, m) in cells]
        bh = bh_verdicts({m: cells[(hyp, m)]["main"]["nw_t"] for m in hm})
        recs: dict[str, ExperimentRecord] = {}
        for m in hm:
            c, md = cells[(hyp, m)], meta[m]
            spec: SignalSpec = c["spec"]
            notes = (
                f"market={m}, {len(md['symbols'])} names, {md['currency']}, no FX. {SURVIVORSHIP_NOTE} "
                f"Placeholder rows excluded (count per symbol): {md['excluded_rows']}. "
                f"Symbols also in the core holdout list (touched here as registered in H-EXT): "
                f"{md['holdout_overlap']}. Irregular calendar rows dropped (< {MIN_COVERAGE:.0%} of listed "
                f"names with a bar; data hygiene, decided after run {SUPERSEDED_RUN} exposed them): "
                f"{md['dropped_sessions']}. Long-short is descriptive gross only (no short execution). "
                "Record conclusion = BH-adjusted verdict across the 3 markets; raw-rule verdict in metrics. "
                "Daily-bar fills; spread is an assumption per market."
            )
            rec = tracker.new(
                f"cs_{spec.name}_{m}",
                hyp,
                text,
                datasets=md["datasets"],
                features=[spec.name],
                model={
                    "signal": spec.description,
                    "portfolio": "long-only EW top tercile (k = max(1, n//3), ties at boundary included)",
                    "benchmark": "EW all eligible names with a finite signal, same dates/engine/costs",
                    "rebalance": spec.rebalance,
                },
                validation={
                    "type": "full-history rule-based backtest (no fitted parameters, no tuning)",
                    "period": f"{c['start'].date()}..{c['end'].date()}",
                    "decision": "close of rebalance date; fill next open (nebula.portfolio.backtest)",
                    "eligibility": f"valid bar at t and >= {MIN_HISTORY} valid bars up to t; finite signal",
                    "study_start": f"first rebalance with >= {MIN_RANKABLE} rankable names",
                    "n_rebalances": c["n_rebalances"],
                    "rankable_names_min_max": [c["rankable_min"], c["rankable_max"]],
                    "test": "Newey-West t (default lags) on daily net return diff; |t| > 1.96; BH q=0.05",
                    "annualization": md["ppy"],
                },
                costs=md["costs"],
                seed=SEED,
                metrics={
                    "market": m,
                    "nw_t": c["main"]["nw_t"],
                    "ann_mean_diff": c["main"]["ann_mean_diff"],
                    "verdict_rule": c["verdict_rule"],
                    **bh[m],
                    "strategy_cagr": c["perf"]["strategy_net"]["cagr"],
                    "strategy_sharpe": c["perf"]["strategy_net"]["sharpe"],
                    "benchmark_cagr": c["perf"]["benchmark_net"]["cagr"],
                    "benchmark_sharpe": c["perf"]["benchmark_net"]["sharpe"],
                    "strategy_turnover": c["perf"]["strategy_net"]["annual_turnover_two_sided"],
                    "benchmark_turnover": c["perf"]["benchmark_net"]["annual_turnover_two_sided"],
                    "gross_nw_t": c["gross_diff"]["nw_t"],
                    "performance": c["perf"],
                    "bootstrap_sharpe_diff": c["bootstrap_sharpe"],
                    "cost_sensitivity": _jsonable(c["cost_sensitivity"]),
                    "subperiods": _jsonable(c["subperiods"]),
                    "long_short_descriptive": c["long_short"],
                },
                conclusion=bh[m]["verdict_bh"],
                variants_tried=3,
                notes=notes,
                reproduce='uv run --no-sync python -c "from nebula.research.cross_sectional import '
                f"run_cross_sectional; run_cross_sectional(['{m}'])\"",
            )
            tracker.save_frame(rec, "equity", c["equity"])
            tracker.save_frame(rec, "targets", c["targets"])
            tracker.save_frame(rec, "cost_sensitivity", c["cost_sensitivity"])
            tracker.save_frame(rec, "subperiods", c["subperiods"])
            tracker.save_frame(rec, "rankable_counts", c["rankable_counts"])
            tracker.log(rec)
            recs[m] = rec

        verdicts = [bh[m]["verdict_bh"] for m in hm]
        fam: Conclusion = "inconclusive"
        if "supported" in verdicts and "rejected" not in verdicts:
            fam = "supported"
        elif "rejected" in verdicts and "supported" not in verdicts:
            fam = "rejected"
        table = {
            m: {
                "experiment_id": recs[m].experiment_id,
                "nw_t": cells[(hyp, m)]["main"]["nw_t"],
                "verdict_rule": cells[(hyp, m)]["verdict_rule"],
                **bh[m],
            }
            for m in hm
        }
        frec = tracker.new(
            f"cs_family_{hyp}",
            hyp,
            text + " Family across markets (BH q=0.05).",
            datasets=[d for m in hm for d in meta[m]["datasets"]],
            validation={"family": hm, "correction": "Benjamini-Hochberg q=0.05 on two-sided NW p-values"},
            seed=SEED,
            metrics={
                "markets": table,
                "n_supported_bh": verdicts.count("supported"),
                "n_rejected_bh": verdicts.count("rejected"),
            },
            conclusion=fam,
            variants_tried=len(hm),
            notes=f"{SURVIVORSHIP_NOTE} Family verdict: supported if >= 1 market BH-supported and none "
            "BH-rejected; rejected if >= 1 BH-rejected and none supported; else inconclusive.",
            reproduce='uv run --no-sync python -c "from nebula.research.cross_sectional import '
            'run_cross_sectional; run_cross_sectional()"',
        )
        tracker.log(frec)
        out[hyp] = {
            "family_id": frec.experiment_id,
            "family_conclusion": fam,
            "markets": {
                m: {
                    **table[m],
                    "period": f"{cells[(hyp, m)]['start'].date()}..{cells[(hyp, m)]['end'].date()}",
                    "n_rebalances": cells[(hyp, m)]["n_rebalances"],
                    "rankable_min_max": [cells[(hyp, m)]["rankable_min"], cells[(hyp, m)]["rankable_max"]],
                    "perf": cells[(hyp, m)]["perf"],
                    "gross_nw_t": cells[(hyp, m)]["gross_diff"]["nw_t"],
                    "ann_mean_diff": cells[(hyp, m)]["main"]["ann_mean_diff"],
                    "bootstrap_sharpe": cells[(hyp, m)]["bootstrap_sharpe"],
                    "cost_sensitivity": _jsonable(cells[(hyp, m)]["cost_sensitivity"]),
                    "subperiods": _jsonable(cells[(hyp, m)]["subperiods"]),
                    "long_short": cells[(hyp, m)]["long_short"],
                }
                for m in hm
            },
        }
    out["runtime_s"] = time.perf_counter() - t0
    return out
