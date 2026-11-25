"""M4/M5/M6 vertical slice: statistics, a quant baseline, and an ML baseline per instrument.

Hypotheses (registered in docs/RESEARCH_HYPOTHESES.md BEFORE results were produced):
- H-XM-000  Stylized facts: log prices have a unit root; log returns are stationary; squared returns are
            autocorrelated (volatility clustering).
- H-XM-001  60-day time-series momentum (long if 60d log return > 0, else flat) earns higher net daily
            returns than buy-and-hold. H0: mean(r_tsmom - r_bh) <= 0. Newey-West t-test.
- H-XM-002  A regularized logistic regression on 13 causal technical features predicts the sign of the
            next open-to-open return better than the base rate (lower OOS log loss), and its long/flat
            strategy beats buy-and-hold net of costs over the OOS period.
All parameters are fixed a priori (no tuning) -> variants_tried = 1 per hypothesis per instrument.
Decision rule for return comparisons: NW t > 1.96 supported, t < -1.96 rejected, else inconclusive.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from astra.backtest.audit import cost_sensitivity, leakage_canary
from astra.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    buy_and_hold,
    run_backtest,
    run_gross,
    signal_strategy,
)
from astra.common.config import get_config
from astra.common.logging import get_logger
from astra.common.paths import Paths, get_paths
from astra.common.seed import set_seed
from astra.data.storage import read_bars, read_manifest
from astra.features.technical import FEATURE_COLUMNS, build_features
from astra.ml.classical import classification_metrics, logistic_factory, walk_forward_classifier
from astra.ml.validation import PurgedWalkForward
from astra.risk.metrics import simple_returns, summarize
from astra.stats.inference import newey_west_tstat, stationary_bootstrap_sharpe_ci
from astra.stats.tests import describe_returns, ljung_box, stationarity
from astra.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)
SEED = 7


def _verdict_from_t(t: float) -> Conclusion:
    if t > 1.96:
        return "supported"
    if t < -1.96:
        return "rejected"
    return "inconclusive"


def _bt_frame(feats: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_date": feats["session_date"],
            "open": feats["adj_open"],
            "close": feats["adj_close"],
            "volume": feats["volume"],
        }
    )


def _equity_frame(net: BacktestResult, gross: BacktestResult, bh: BacktestResult) -> pd.DataFrame:
    df = net.ledger.copy()
    df["equity_gross"] = gross.ledger["equity_net"]
    df["bh_equity_net"] = bh.ledger["equity_net"]
    return df


def _compare(
    strat_eq: pd.Series, bh_eq: pd.Series, ppy: int
) -> tuple[dict[str, Any], dict[str, Any], float, tuple[float, float]]:
    s, b = summarize(strat_eq, ppy), summarize(bh_eq, ppy)
    diff = (simple_returns(strat_eq) - simple_returns(bh_eq)).dropna()
    t = newey_west_tstat(diff)
    ci = stationary_bootstrap_sharpe_ci(simple_returns(strat_eq), ppy, n_boot=500, seed=SEED)
    return s, b, t, ci


def run_symbol(symbol: str, paths: Paths | None = None) -> dict[str, Any]:
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    inst = cfg.instrument(symbol)
    mkt = cfg.markets[inst.market]
    ppy = mkt.periods_per_year
    bars = read_bars(symbol, inst.market, paths, window=cfg.core_window())
    manifest = read_manifest(symbol, inst.market, paths)
    datasets = [manifest]
    # Provider placeholder rows (no trading: volume 0 and H == L) are excluded from research, not
    # repaired; the validated store keeps them and the count is recorded in every experiment.
    placeholder = (bars["volume"] == 0) & (bars["high"] == bars["low"])
    excluded = [str(d) for d in bars.loc[placeholder, "session_date"]]
    bars = bars.loc[~placeholder].reset_index(drop=True)
    feats = build_features(bars)
    tracker = Tracker(paths)
    bt_cfg = BacktestConfig(costs=mkt.costs_bps)
    costs_dict = {**mkt.costs_bps.model_dump(), "max_participation": bt_cfg.max_participation}
    out: dict[str, Any] = {"symbol": symbol, "market": inst.market}

    # ---- H-XM-000 stylized facts ------------------------------------------------------------
    logp = pd.Series(np.log(feats["adj_close"].astype(float)))
    st_price, st_ret = stationarity(logp), stationarity(feats["ret_1"])
    lb = ljung_box(feats["ret_1"], lags=10)
    desc = describe_returns(feats["ret_1"], ppy)
    holds = (
        st_price["verdict"] == "unit_root"
        and st_ret["verdict"] == "stationary"
        and lb["p_value_squared"] < 0.05
    )
    rec0 = tracker.new(
        "stylized_facts",
        "H-XM-000",
        "Log prices unit root; log returns stationary; squared returns autocorrelated.",
        datasets=datasets,
        validation={
            "sample": f"{feats['session_date'].min():%Y-%m-%d}..{feats['session_date'].max():%Y-%m-%d}"
        },
        seed=SEED,
        metrics={"stationarity_log_price": st_price, "stationarity_returns": st_ret, "ljung_box": lb, **desc},
        conclusion="supported" if holds else "rejected",
        notes=f"symbol={symbol}; ADF+KPSS jointly; LB on r and r^2 at lag 10; "
        f"excluded placeholder rows: {excluded}",
        reproduce=f"uv run astra research {symbol}",
    )
    tracker.log(rec0)
    out["H-XM-000"] = {"conclusion": rec0.conclusion, "id": rec0.experiment_id}

    bt = _bt_frame(feats)
    warm = 60
    bt_eval = bt.iloc[warm:].reset_index(drop=True)

    # ---- H-XM-001 time-series momentum ------------------------------------------------------
    sig = pd.Series(
        (feats["mom_60"] > 0).astype(float).to_numpy(), index=pd.to_datetime(feats["session_date"])
    )
    sig = sig.where(feats["mom_60"].notna().to_numpy())
    res = run_backtest(bt_eval, signal_strategy(sig), bt_cfg)
    res_gross = run_gross(bt_eval, signal_strategy(sig), bt_cfg)
    res_bh = run_backtest(bt_eval, buy_and_hold(), bt_cfg)
    s, b, t, ci = _compare(res.equity_net, res_bh.equity_net, ppy)
    sens = cost_sensitivity(bt_eval, sig, ppy, bt_cfg)
    canary = leakage_canary(bt_eval, ppy, bt_cfg)
    gross = summarize(res_gross.equity_net, ppy)
    rec1 = tracker.new(
        "tsmom_60_long_flat",
        "H-XM-001",
        "60d TS momentum long/flat beats buy-and-hold net of costs (NW t on daily return difference).",
        datasets=datasets,
        features=["mom_60"],
        model={"type": "rule", "lookback": 60, "positions": "long/flat"},
        validation={
            "type": "full-sample rule (no fitting)",
            "start": str(bt_eval["session_date"].iloc[0].date()),
        },
        costs=costs_dict,
        seed=SEED,
        metrics={
            "net": s,
            "gross": gross,
            "buy_hold_net": b,
            "nw_t_diff": t,
            "net_sharpe_ci95": ci,
            "total_cost": float(res.ledger["cost"].sum()),
            "avg_turnover": float(res.ledger["turnover"].mean()),
            "exposure": float((res.ledger["weight"] > 0.01).mean()),
            "n_fills": len(res.fills),
            "leakage_canary": canary,
            "net_sharpe": s["sharpe"],
            "bh_sharpe": b["sharpe"],
        },
        conclusion=_verdict_from_t(t),
        notes=f"excluded placeholder rows: {excluded}",
        reproduce=f"uv run astra research {symbol}",
    )
    tracker.save_frame(rec1, "equity", _equity_frame(res, res_gross, res_bh))
    tracker.save_frame(rec1, "cost_sensitivity", sens)
    tracker.log(rec1)
    out["H-XM-001"] = {
        "conclusion": rec1.conclusion,
        "id": rec1.experiment_id,
        "net_sharpe": s["sharpe"],
        "bh_sharpe": b["sharpe"],
        "nw_t": t,
    }

    # ---- H-XM-002 logistic regression -------------------------------------------------------
    data = feats.dropna(subset=[*FEATURE_COLUMNS, "target_up"]).reset_index(drop=True)
    X = data[FEATURE_COLUMNS].set_axis(pd.to_datetime(data["session_date"]))
    y = data["target_up"].set_axis(X.index)
    min_train = 2 * ppy
    cv = PurgedWalkForward(min_train=min_train, test_size=ppy // 4, label_horizon=2, embargo=5)
    wf = walk_forward_classifier(X, y, logistic_factory(c=0.1, seed=SEED), cv)
    cm = classification_metrics(y, wf.proba, wf.baseline_proba)
    sig_ml = (wf.proba > 0.5).astype(float).where(wf.proba.notna())
    oos_start = wf.proba.first_valid_index()
    bt_oos = bt[pd.to_datetime(bt["session_date"]) >= oos_start].reset_index(drop=True)
    res_ml = run_backtest(bt_oos, signal_strategy(sig_ml), bt_cfg)
    res_ml_gross = run_gross(bt_oos, signal_strategy(sig_ml), bt_cfg)
    res_bh2 = run_backtest(bt_oos, buy_and_hold(), bt_cfg)
    s2, b2, t2, ci2 = _compare(res_ml.equity_net, res_bh2.equity_net, ppy)
    sens2 = cost_sensitivity(bt_oos, sig_ml, ppy, bt_cfg)
    # Original registered rule (kept for transparency): gate on point log loss vs the OOS sample's own
    # base rate. Independent audit (2026-11-24) found that baseline uses hindsight and the gate had no
    # significance test. Amended rule (see RESEARCH_HYPOTHESES.md, amendment A1):
    #   model significantly worse than train-fold base rate (DM t < -1.96) -> rejected
    #   strategy significantly worse than buy-and-hold (NW t < -1.96)    -> rejected
    #   both significantly better (t > 1.96)                             -> supported
    #   otherwise                                                         -> inconclusive
    original_rule = "rejected" if cm["log_loss"] >= cm["hindsight_baseline_log_loss"] else _verdict_from_t(t2)
    ll_t = cm["ll_improvement_nw_t"]
    concl: Conclusion
    if ll_t < -1.96 or t2 < -1.96:
        concl = "rejected"
    elif ll_t > 1.96 and t2 > 1.96:
        concl = "supported"
    else:
        concl = "inconclusive"
    rec2 = tracker.new(
        "logit_13feat_long_flat",
        "H-XM-002",
        "Logistic regression on causal features beats base rate (log loss) "
        "and buy-and-hold net of costs OOS.",
        datasets=datasets,
        features=FEATURE_COLUMNS,
        model={"type": "LogisticRegression", "C": 0.1, "scaler": "StandardScaler (fit per train fold)"},
        validation={
            "type": "PurgedWalkForward expanding",
            "min_train": min_train,
            "test_size": cv.test_size,
            "label_horizon": 2,
            "embargo": 5,
            "n_folds": len(wf.folds),
            "oos_start": str(oos_start),
            "folds": wf.folds,
        },
        costs=costs_dict,
        seed=SEED,
        metrics={
            **{f"clf_{k}": v for k, v in cm.items()},
            "net": s2,
            "gross": summarize(res_ml_gross.equity_net, ppy),
            "buy_hold_net": b2,
            "nw_t_diff": t2,
            "net_sharpe_ci95": ci2,
            "total_cost": float(res_ml.ledger["cost"].sum()),
            "avg_turnover": float(res_ml.ledger["turnover"].mean()),
            "exposure": float((res_ml.ledger["weight"] > 0.01).mean()),
            "n_fills": len(res_ml.fills),
            "net_sharpe": s2["sharpe"],
            "bh_sharpe": b2["sharpe"],
            "verdict_original_rule": original_rule,
        },
        conclusion=concl,
        notes="Fixed hyperparameters, no tuning. Decision rule amendment A1 (post-audit) applied; "
        f"verdict under the original registered rule: {original_rule}. "
        f"Excluded placeholder rows: {excluded}",
        reproduce=f"uv run astra research {symbol}",
    )
    preds = pd.DataFrame({"proba": wf.proba, "baseline_proba": wf.baseline_proba, "y": y})
    tracker.save_frame(rec2, "oos_predictions", preds)
    tracker.save_frame(rec2, "equity", _equity_frame(res_ml, res_ml_gross, res_bh2))
    tracker.save_frame(rec2, "cost_sensitivity", sens2)
    tracker.log(rec2)
    out["H-XM-002"] = {
        "conclusion": rec2.conclusion,
        "original_rule": original_rule,
        "id": rec2.experiment_id,
        "auc": cm["auc"],
        "log_loss": cm["log_loss"],
        "baseline_log_loss": cm["baseline_log_loss"],
        "ll_improvement_nw_t": ll_t,
        "net_sharpe": s2["sharpe"],
        "bh_sharpe": b2["sharpe"],
        "nw_t": t2,
    }
    log.info("research.done", **{k: str(v) for k, v in out.items()})
    return out
