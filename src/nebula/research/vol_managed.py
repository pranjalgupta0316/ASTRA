"""H-VT-001: volatility-managed exposure (Moreira & Muir 2017, concept only), registered in
docs/research/H-EXT.md before any run.

Rule (fixed, no tuning)
- sigma_hat_t = sqrt(EWMA(lambda = 0.94) variance) of daily log returns, known at the close of t
  (`nebula.stats.volatility.ewma_forecast`, causal; first 20 origins masked).
- Calibration window = the first 3 calendar years of each series [t0, t0 + 3y). sigma_target = median
  of sigma_hat over that window, then FROZEN. The calibration window is excluded from evaluation.
- weight_t = min(1, sigma_target / sigma_hat_t) (leverage cap 1.0, de-risking only).
- Decision at close t, fill at open t+1 (event-driven engines), market costs from config.

Series
- Every configured instrument (111), single-asset engine `nebula.backtest.engine`. Unmanaged =
  buy-and-hold over the same evaluation window. Indices (kind=index, i.e. NIFTY50) have unreliable
  volume: no participation cap and volume is not used for fills (as in holdout_study).
- Each market's equal-weight stock portfolio (US stocks 30, IN stocks 49, crypto 16), multi-asset engine
  `nebula.portfolio.backtest`, local currency. Unmanaged EW = monthly rebalance (first session of the
  month) to 1/n over names with >= 252 valid returns. Managed EW = scale_t * composition (EW row on
  monthly dates, else the unmanaged portfolio's drifted weights at t);
  sigma_hat comes from the full-history UNMANAGED EW gross daily returns.

Rebalance tolerance (TOL = 0.02 = the single-asset engine's default `rebalance_tolerance`)
- Single asset: the engine itself skips trades with |target - current weight| < TOL.
- EW portfolio: the per-leg engine tolerance would block de-risking for 30-49 legs of 2-3 % each, so the
  tolerance is applied to the portfolio exposure: a managed decision is issued when |scale_t - last
  issued scale| > TOL, or on a monthly EW date; legs are then reset to scale * EW. The engine's per-leg
  tolerance is set to 0 for BOTH managed and unmanaged EW runs (identical engine settings).

Inference: paired stationary bootstrap (block 20, 2000 draws, seed 7) of Sharpe(managed) -
Sharpe(unmanaged) on daily net returns over the evaluation window; per series CI > 0 supported, CI < 0
rejected, else inconclusive; Benjamini-Hochberg across the family (111 instruments + 3 portfolios).
Annualization: 252 (US, IN), 365 (CRYPTO); rf = 0. Universe = today's names -> survivorship-biased.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from nebula.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    buy_and_hold,
    run_backtest,
    signal_strategy,
)
from nebula.common.config import get_config
from nebula.common.logging import get_logger
from nebula.common.paths import Paths, get_paths
from nebula.common.seed import set_seed
from nebula.data.storage import read_bars, read_manifest
from nebula.features.technical import adjusted_ohlc
from nebula.portfolio.backtest import (
    MarketPanel,
    PortfolioBacktestConfig,
    PortfolioBacktestResult,
    load_market_panel,
    monthly_decision_dates,
    run_portfolio_backtest,
)
from nebula.risk.metrics import simple_returns, summarize
from nebula.stats.inference import sharpe_difference_bootstrap
from nebula.stats.volatility import ewma_forecast, log_returns
from nebula.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)

HYP = "H-VT-001"
SEED = 7
LAM = 0.94
EWMA_MIN_PERIODS = 20
CALIB_YEARS = 3
TOL = 0.02  # == BacktestConfig().rebalance_tolerance (single-asset engine default)
N_BOOT = 2000
MEAN_BLOCK = 20
MIN_HISTORY = 252
EW_COST_MULTIPLIERS = (0.0, 1.0, 2.0, 4.0)
EW_UNIVERSES: dict[str, tuple[str, str]] = {
    "US": ("US", "stock"),
    "IN": ("IN", "stock"),
    "CRYPTO": ("CRYPTO", "crypto"),
}
SURVIVORSHIP_NOTE = (
    "Survivorship: the universe is TODAY's large caps / NIFTY names / crypto majors, chosen with hindsight; "
    "names that were delisted, dropped from indices or died are absent. Absolute returns of both managed "
    "and unmanaged series are biased upward; the managed-vs-unmanaged comparison is conditional on survival "
    "(survivors may have had fewer prolonged high-volatility collapses than the full population)."
)


# ------------------------------------------------------------------------------------- core rule
@dataclass(frozen=True)
class Calibration:
    sigma_target: float  # daily, decimal
    calib_start: pd.Timestamp
    calib_end: pd.Timestamp  # last session of the calibration window
    eval_start: pd.Timestamp  # first session of the evaluation window
    n_calib_obs: int


def ewma_sigma(close: pd.Series, lam: float = LAM, min_periods: int = EWMA_MIN_PERIODS) -> pd.Series:
    """Daily EWMA volatility known at the close of each date (uses closes <= t only)."""
    r = log_returns(pd.Series(close, dtype=float)).dropna()
    h = ewma_forecast(r, lam=lam, min_periods=min_periods)
    return np.sqrt(h).reindex(pd.Series(close).index)


def calibrate(sigma_hat: pd.Series, years: int = CALIB_YEARS) -> Calibration:
    """sigma_target = median sigma_hat over the first `years` calendar years of the series (frozen);
    evaluation starts at the first date >= t0 + years."""
    idx = pd.DatetimeIndex(sigma_hat.index)
    t0 = idx[0]
    cutoff = t0 + pd.DateOffset(years=years)
    in_cal = idx < cutoff
    if in_cal.all():
        raise ValueError("series shorter than the calibration window")
    cal = sigma_hat[in_cal].dropna()
    if cal.empty:
        raise ValueError("no sigma estimates in the calibration window")
    return Calibration(
        sigma_target=float(cal.median()),
        calib_start=t0,
        calib_end=idx[in_cal][-1],
        eval_start=idx[~in_cal][0],
        n_calib_obs=len(cal),
    )


def managed_weight(sigma_hat: pd.Series, sigma_target: float) -> pd.Series:
    """weight_t = min(1, sigma_target / sigma_hat_t); NaN where sigma_hat is unknown."""
    s = pd.Series(sigma_hat, dtype=float)
    w = (sigma_target / s.where(s > 0)).clip(upper=1.0)
    return w.where(s.notna())


def managed_ew_targets(
    scale: pd.Series,
    ew_rows: pd.DataFrame,
    start: pd.Timestamp,
    tol: float = TOL,
    drift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Decisions for the managed EW portfolio from `start` on: issued at `start`, on every monthly EW
    date, and whenever |scale_t - last issued scale| > tol. Target = scale_t * composition_t, where the
    composition is the EW row on `start` / monthly dates and otherwise the UNMANAGED portfolio's drifted
    end-of-day weights at t normalised to 1 (`drift`, known at the close of t), so the managed portfolio
    differs from the unmanaged one only by its exposure scale. Without `drift` the latest EW row is used."""
    s = scale.loc[scale.index >= start].dropna()
    ew = ew_rows.sort_index()
    out, last = {}, None
    for t, k in s.items():
        is_month = t in ew.index and t > start
        if last is None or is_month or abs(k - last) > tol:
            comp = None
            if not is_month and last is not None and drift is not None and t in drift.index:
                d = drift.loc[t]
                if d.sum() > 0:
                    comp = d / d.sum()
            if comp is None:
                row = ew.loc[ew.index <= t]
                if row.empty:
                    continue
                comp = row.iloc[-1]
            out[t] = comp * k
            last = k
    return pd.DataFrame(out).T.reindex(columns=ew.columns).fillna(0.0)


def ew_rows(returns: pd.DataFrame, dates: pd.DatetimeIndex, min_history: int = MIN_HISTORY) -> pd.DataFrame:
    """Equal weight over names with >= min_history valid returns up to and including t (no backfill)."""
    counts = returns.notna().cumsum()
    rows = {}
    for t in dates:
        elig = counts.loc[t] >= min_history
        n = int(elig.sum())
        if n >= 2:
            rows[pd.Timestamp(t)] = elig.astype(float) / n
    return pd.DataFrame(rows).T


# ------------------------------------------------------------------------------------- helpers
def _verdict(lo: float, hi: float) -> Conclusion:
    return "supported" if lo > 0 else ("rejected" if hi < 0 else "inconclusive")


def _perf(net: pd.Series, gross: pd.Series, ledger: pd.DataFrame, ppy: int) -> dict[str, float]:
    s, g = summarize(net, ppy), summarize(gross, ppy)
    years = (len(net) - 1) / ppy
    return {
        "sharpe": s["sharpe"],
        "cagr": s["cagr"],
        "ann_vol": s["ann_vol"],
        "max_drawdown": s["max_drawdown"],
        "gross_sharpe": g["sharpe"],
        "gross_cagr": g["cagr"],
        "cost_drag_cagr": g["cagr"] - s["cagr"],
        "annual_turnover": float(ledger["turnover"].sum() / years) if years > 0 else math.nan,
        "total_cost_pct_initial": float(ledger["cost"].sum() / net.iloc[0]),
    }


def _forecast_realized(weight: pd.Series, sigma_hat: pd.Series, r: pd.Series, ppy: int) -> dict[str, float]:
    """Ex-ante forecast vol of the position (w_t * sigma_hat_t, annualized) for the NEXT day vs realized."""
    f = (weight * sigma_hat).shift(1).reindex(r.index)  # forecast made at t-1 for the return of t
    ok = f.notna() & r.notna()
    k = math.sqrt(ppy)
    return {
        "forecast_vol_ann_mean": float(f[ok].mean() * k),
        "forecast_vol_ann_rms": float(np.sqrt((f[ok] ** 2).mean()) * k),
        "realized_vol_ann": float(r[ok].std(ddof=1) * k),
    }


def load_single(symbol: str, market: str, kind: str, paths: Paths) -> pd.DataFrame:
    """Validated bars, provider placeholder rows (volume 0 and H == L) dropped, adjusted prices."""
    bars = read_bars(symbol, market, paths)
    ph = (bars["volume"] == 0) & (bars["high"] == bars["low"])
    adj = adjusted_ohlc(bars.loc[~ph].reset_index(drop=True))
    out = adj[["session_date", "open", "close", "volume"]].copy()
    if kind == "index":
        out["volume"] = np.inf  # unreliable index volume: not used, no participation cap
    return out.reset_index(drop=True)


def _single_cfg(costs: Any, kind: str, mult: float = 1.0) -> BacktestConfig:
    c = BacktestConfig(costs=costs, cost_multiplier=mult)
    if kind == "index":
        c.max_participation = float("inf")
    return c


# ------------------------------------------------------------------------------------- per series
def run_instrument(symbol: str, paths: Paths) -> dict[str, Any]:
    cfg = get_config()
    inst = cfg.instrument(symbol)
    mkt = cfg.markets[inst.market]
    ppy = mkt.periods_per_year
    bars = load_single(symbol, inst.market, inst.kind, paths)
    close = pd.Series(bars["close"].to_numpy(), index=pd.DatetimeIndex(bars["session_date"]))
    sig = ewma_sigma(close)
    cal = calibrate(sig)
    w = managed_weight(sig, cal.sigma_target)
    bt = bars.loc[bars["session_date"] >= cal.calib_end].reset_index(drop=True)
    w_eval = w.loc[w.index >= cal.calib_end]
    bcfg = _single_cfg(mkt.costs_bps, inst.kind)
    gcfg = replace(bcfg, cost_multiplier=0.0)
    m_net = run_backtest(bt, signal_strategy(w_eval), bcfg)
    m_gr = run_backtest(bt, signal_strategy(w_eval), gcfg)
    u_net = run_backtest(bt, buy_and_hold(), bcfg)
    u_gr = run_backtest(bt, buy_and_hold(), gcfg)
    return _evaluate(
        name=symbol,
        market=inst.market,
        kind=inst.kind,
        ppy=ppy,
        cal=cal,
        sigma=sig,
        weight=w,
        m=(m_net, m_gr),
        u=(u_net, u_gr),
        datasets=[read_manifest(symbol, inst.market, paths)],
        costs={**mkt.costs_bps.model_dump(), "max_participation": bcfg.max_participation, "tolerance": TOL},
    )


def _ledger(res: BacktestResult | PortfolioBacktestResult) -> pd.DataFrame:
    return res.ledger


def _evaluate(
    name: str,
    market: str,
    kind: str,
    ppy: int,
    cal: Calibration,
    sigma: pd.Series,
    weight: pd.Series,
    m: tuple[Any, Any],
    u: tuple[Any, Any],
    datasets: list[dict[str, Any]],
    costs: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    (m_net, m_gr), (u_net, u_gr) = m, u
    rm, ru = simple_returns(m_net.equity_net), simple_returns(u_net.equity_net)
    if rm.index[0] < cal.eval_start:  # pragma: no cover - guarded by construction
        raise AssertionError("evaluation returns overlap the calibration window")
    sd = sharpe_difference_bootstrap(rm, ru, ppy, n_boot=N_BOOT, mean_block=MEAN_BLOCK, seed=SEED)
    pm = _perf(m_net.equity_net, m_gr.equity_net, _ledger(m_net), ppy)
    pu = _perf(u_net.equity_net, u_gr.equity_net, _ledger(u_net), ppy)
    w_eval = weight.loc[weight.index >= cal.calib_end]
    fr_m = _forecast_realized(weight, sigma, rm, ppy)
    fr_u = _forecast_realized(pd.Series(1.0, index=sigma.index), sigma, ru, ppy)
    row = {
        "series": name,
        "market": market,
        "kind": kind,
        "ppy": ppy,
        "calib_start": str(cal.calib_start.date()),
        "calib_end": str(cal.calib_end.date()),
        "eval_start": str(cal.eval_start.date()),
        "eval_end": str(rm.index[-1].date()),
        "sigma_target_daily": cal.sigma_target,
        "sigma_target_ann": cal.sigma_target * math.sqrt(ppy),
        "mean_weight_target": float(w_eval.mean()),
        "frac_days_weight_lt_1": float((w_eval < 1).mean()),
        **{f"managed_{k}": v for k, v in pm.items()},
        **{f"unmanaged_{k}": v for k, v in pu.items()},
        "delta_mdd": pm["max_drawdown"] - pu["max_drawdown"],
        "delta_turnover": pm["annual_turnover"] - pu["annual_turnover"],
        "delta_cost_drag_cagr": pm["cost_drag_cagr"] - pu["cost_drag_cagr"],
        "managed_forecast_vol_ann_mean": fr_m["forecast_vol_ann_mean"],
        "managed_realized_vol_ann": fr_m["realized_vol_ann"],
        "unmanaged_forecast_vol_ann_mean": fr_u["forecast_vol_ann_mean"],
        "unmanaged_realized_vol_ann": fr_u["realized_vol_ann"],
        **sd,
        "verdict_ci": _verdict(sd["ci_low"], sd["ci_high"]),
        **(extra or {}),
    }
    frames = pd.DataFrame(
        {
            "managed_equity_net": m_net.equity_net,
            "managed_equity_gross": m_gr.equity_net,
            "unmanaged_equity_net": u_net.equity_net,
            "unmanaged_equity_gross": u_gr.equity_net,
            "sigma_hat_daily": sigma.reindex(m_net.equity_net.index),
            "target_weight": w_eval.reindex(m_net.equity_net.index),
        }
    )
    if "weight" in m_net.ledger:
        frames["managed_weight_eod"] = m_net.ledger["weight"]
    else:
        frames["managed_exposure_eod"] = m_net.ledger["exposure"]
    return {"row": row, "equity": frames, "datasets": datasets, "costs": costs}


# ------------------------------------------------------------------------------------- EW portfolios
@dataclass
class EwSleeve:
    market: str
    panel: MarketPanel
    rows: pd.DataFrame  # monthly EW targets
    full_gross: PortfolioBacktestResult
    full_net: PortfolioBacktestResult
    config: PortfolioBacktestConfig


def ew_sleeve(market: str, paths: Paths) -> EwSleeve:
    """Unmanaged monthly-rebalanced EW portfolio of a market's stocks/crypto over its full history."""
    cfg = get_config()
    mk, kind = EW_UNIVERSES[market]
    syms = cfg.symbols(market=mk, kind=kind)
    panel = load_market_panel(mk, syms, paths, window=None)
    rows = ew_rows(panel.returns(), monthly_decision_dates(panel.close.index))
    pcfg = PortfolioBacktestConfig(costs=cfg.markets[mk].costs_bps, rebalance_tolerance=0.0)  # type: ignore[index]
    net = run_portfolio_backtest(panel, rows, pcfg)
    gross = run_portfolio_backtest(panel, rows, replace(pcfg, cost_multiplier=0.0))
    return EwSleeve(market, panel, rows, gross, net, pcfg)


def run_ew_portfolio(sl: EwSleeve, paths: Paths) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = get_config()
    mkt = cfg.markets[sl.market]  # type: ignore[index]
    ppy = mkt.periods_per_year
    eq = sl.full_gross.equity_net
    sig = ewma_sigma(eq.iloc[1:])  # first ledger row is all cash (no return)
    cal = calibrate(sig)
    scale = managed_weight(sig, cal.sigma_target)
    start = cal.calib_end
    base = sl.rows.loc[sl.rows.index <= start].iloc[[-1]].set_axis([start])
    un_tg = pd.concat([base, sl.rows.loc[sl.rows.index > start]])
    gcfg = replace(sl.config, cost_multiplier=0.0)
    u_net = run_portfolio_backtest(sl.panel, un_tg, sl.config, start=start)
    u_gr = run_portfolio_backtest(sl.panel, un_tg, gcfg, start=start)
    m_tg = managed_ew_targets(scale, sl.rows, start, drift=u_net.weights)
    m_net = run_portfolio_backtest(sl.panel, m_tg, sl.config, start=start)
    m_gr = run_portfolio_backtest(sl.panel, m_tg, gcfg, start=start)
    sens = []
    for mult in EW_COST_MULTIPLIERS:
        c = replace(sl.config, cost_multiplier=mult)
        for lab, tg in (("managed", m_tg), ("unmanaged", un_tg)):
            r = run_portfolio_backtest(sl.panel, tg, c, start=start)
            s = summarize(r.equity_net, ppy)
            sens.append({"series": f"EW_{sl.market}", "portfolio": lab, "cost_multiplier": mult, **s})
    out = _evaluate(
        name=f"EW_{sl.market}",
        market=sl.market,
        kind="ew_portfolio",
        ppy=ppy,
        cal=cal,
        sigma=sig,
        weight=scale,
        m=(m_net, m_gr),
        u=(u_net, u_gr),
        datasets=[
            {"symbol": s, "market": sl.market, "sha256": read_manifest(s, sl.market, paths)["sha256"]}
            for s in sl.panel.symbols
        ],
        costs={
            **mkt.costs_bps.model_dump(),
            "max_participation": sl.config.max_participation,
            "engine_leg_tolerance": 0.0,
            "exposure_tolerance": TOL,
        },
        extra={
            "n_names": len(sl.panel.symbols),
            "n_managed_decisions": len(m_tg),
            "n_unmanaged_decisions": len(un_tg),
            "ew_start": str(eq.index[0].date()),
        },
    )
    w = pd.concat({"managed": m_net.weights, "unmanaged": u_net.weights}, axis=1)
    w.columns = [f"{a}:{b}" for a, b in w.columns]
    out["weights"] = w
    return out, pd.DataFrame(sens)


# ------------------------------------------------------------------------------------- driver
def _record(tracker: Tracker, res: dict[str, Any], bh: dict[str, Any]) -> str:
    row = res["row"]
    metrics = {
        k: (
            float(v)
            if isinstance(v, int | float | np.floating | np.integer) and not isinstance(v, bool)
            else v
        )
        for k, v in {**row, **bh}.items()
    }
    rec = tracker.new(
        f"vol_managed_{row['series']}",
        HYP,
        "Volatility-managed exposure min(1, sigma_target/sigma_hat) has a higher net Sharpe than the "
        "unmanaged series (paired bootstrap CI of dSharpe > 0).",
        datasets=[
            {**{k: v for k, v in d.items() if k in ("symbol", "market", "sha256")}} for d in res["datasets"]
        ],
        model={
            "type": "rule",
            "ewma_lambda": LAM,
            "calibration_years": CALIB_YEARS,
            "sigma_target": "median EWMA vol over calibration window (frozen)",
            "leverage_cap": 1.0,
        },
        validation={
            "type": "registered rule, post-calibration evaluation window only",
            "bootstrap": f"paired stationary, block {MEAN_BLOCK}, {N_BOOT} draws, seed {SEED}",
            "multiple_testing": "Benjamini-Hochberg across 114 series (111 instruments + 3 EW portfolios)",
            "annualization": row["ppy"],
            "rf": 0.0,
        },
        costs=res["costs"],
        seed=SEED,
        metrics=metrics,
        conclusion=row["verdict_ci"],
        variants_tried=1,
        notes=SURVIVORSHIP_NOTE + " Verdict from the raw 95% CI (registered); BH-adjusted significance "
        "is in metrics (p_bh, bh_significant).",
        reproduce='uv run --no-sync python -c "from nebula.research.vol_managed import run_vol_managed; '
        'run_vol_managed()"',
    )
    tracker.save_frame(rec, "equity", res["equity"])
    if "weights" in res:
        tracker.save_frame(rec, "weights", res["weights"])
    tracker.log(rec)
    return rec.experiment_id


def run_vol_managed(markets: list[str] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    """Run H-VT-001 for every instrument of `markets` (default all three) and each market's EW
    portfolio. BH is applied across the series actually run. Returns {'table', 'family', ...}."""
    t0 = time.perf_counter()
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    markets = markets or ["US", "IN", "CRYPTO"]
    tracker = Tracker(paths)
    results: list[dict[str, Any]] = []
    for m in markets:
        for s in cfg.symbols(market=m):
            ts = time.perf_counter()
            results.append(run_instrument(s, paths))
            log.info("vol_managed.instrument", symbol=s, runtime_s=round(time.perf_counter() - ts, 1))
    sens_frames = []
    for m in markets:
        sl = ew_sleeve(m, paths)
        res, sens = run_ew_portfolio(sl, paths)
        results.append(res)
        sens_frames.append(sens)
        log.info("vol_managed.ew", market=m)
    table = pd.DataFrame([r["row"] for r in results])
    rej, p_adj, *_ = multipletests(table["p_two_sided"], alpha=0.05, method="fdr_bh")
    table["p_bh"], table["bh_significant"] = p_adj, rej
    table["verdict_bh"] = np.where(
        rej & (table["delta_sr"] > 0),
        "supported",
        np.where(rej & (table["delta_sr"] < 0), "rejected", "inconclusive"),
    )
    ids = []
    for r, (_, t) in zip(results, table.iterrows(), strict=True):
        ids.append(
            _record(
                tracker,
                r,
                {
                    "p_bh": float(t["p_bh"]),
                    "bh_significant": bool(t["bh_significant"]),
                    "verdict_bh": t["verdict_bh"],
                },
            )
        )
    table["experiment_id"] = ids
    counts = table.assign(group=np.where(table["kind"] == "ew_portfolio", "EW", table["market"])).pivot_table(
        index="group", columns="verdict_ci", values="series", aggfunc="count", fill_value=0
    )
    counts_bh = table.assign(
        group=np.where(table["kind"] == "ew_portfolio", "EW", table["market"])
    ).pivot_table(index="group", columns="verdict_bh", values="series", aggfunc="count", fill_value=0)
    pos = int((table["verdict_bh"] == "supported").sum())
    neg = int((table["verdict_bh"] == "rejected").sum())
    family: Conclusion = (
        "supported" if pos > 0 and neg == 0 else ("rejected" if neg > 0 and pos == 0 else "inconclusive")
    )
    ew = table[table["kind"] == "ew_portfolio"]
    fam = tracker.new(
        "vol_managed_family",
        HYP,
        "Family decision across all instruments + 3 EW portfolios (BH on bootstrap p-values).",
        datasets=[{"markets": markets, "n_series": len(table)}],
        model={"ewma_lambda": LAM, "calibration_years": CALIB_YEARS, "leverage_cap": 1.0},
        validation={
            "bootstrap": f"paired stationary, block {MEAN_BLOCK}, {N_BOOT} draws, seed {SEED}",
            "multiple_testing": "Benjamini-Hochberg, alpha 0.05",
            "family_rule": "supported if >= 1 BH-significant positive and none negative; rejected if >= 1 "
            "BH-significant negative and none positive; else inconclusive (aggregation rule fixed before "
            "the run, not in the registration text)",
            "annualization": "252 equities, 365 crypto",
        },
        metrics={
            "n_series": len(table),
            "bh_significant_positive": pos,
            "bh_significant_negative": neg,
            "ci_supported": int((table["verdict_ci"] == "supported").sum()),
            "ci_rejected": int((table["verdict_ci"] == "rejected").sum()),
            "ci_inconclusive": int((table["verdict_ci"] == "inconclusive").sum()),
            "median_delta_sr": float(table["delta_sr"].median()),
            "median_delta_mdd": float(table["delta_mdd"].median()),
            "counts_ci_by_group": counts.to_dict(),
            "counts_bh_by_group": counts_bh.to_dict(),
            "ew_portfolios": ew[["series", "delta_sr", "ci_low", "ci_high", "p_bh", "delta_mdd"]].to_dict(
                "records"
            ),
        },
        variants_tried=len(table),
        conclusion=family,
        notes=SURVIVORSHIP_NOTE,
        reproduce='uv run --no-sync python -c "from nebula.research.vol_managed import run_vol_managed; '
        'run_vol_managed()"',
    )
    tracker.save_frame(fam, "table", table)
    tracker.save_frame(fam, "ew_cost_sensitivity", pd.concat(sens_frames, ignore_index=True))
    tracker.log(fam)
    runtime = time.perf_counter() - t0
    return {
        "family": family,
        "family_id": fam.experiment_id,
        "table": table,
        "counts_ci": counts,
        "counts_bh": counts_bh,
        "ew": ew,
        "runtime_s": runtime,
    }
