"""H-XM-003: confirmatory Sharpe-difference test of 60d TSMOM vs buy-and-hold on held-out instruments."""

from __future__ import annotations

from typing import Any

import pandas as pd
from statsmodels.stats.multitest import multipletests

from astra.backtest.engine import BacktestConfig, buy_and_hold, run_backtest, signal_strategy
from astra.common.config import get_config
from astra.common.paths import Paths, get_paths
from astra.common.seed import set_seed
from astra.data.storage import read_manifest
from astra.ml.datasets import load_features
from astra.risk.metrics import simple_returns, summarize
from astra.stats.inference import sharpe_difference_bootstrap
from astra.tracking.tracker import Conclusion, Tracker

SEED = 7


def run_holdout_tsmom(symbols: list[str] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or cfg.holdout
    tracker = Tracker(paths)
    rows = []
    for sym in symbols:
        inst = cfg.instrument(sym)
        mkt = cfg.markets[inst.market]
        ppy = mkt.periods_per_year
        f = load_features(sym, paths, window=cfg.core_window())
        bt = (
            pd.DataFrame(
                {
                    "session_date": f["session_date"],
                    "open": f["adj_open"],
                    "close": f["adj_close"],
                    "volume": f["volume"],
                }
            )
            .iloc[60:]
            .reset_index(drop=True)
        )
        sig = pd.Series((f["mom_60"] > 0).astype(float).to_numpy(), index=pd.to_datetime(f["session_date"]))
        sig = sig.where(f["mom_60"].notna().to_numpy())
        bcfg = BacktestConfig(costs=mkt.costs_bps)
        if sym == "NIFTY50":
            # index has unreliable volume: no participation cap (it is a non-tradable benchmark proxy)
            bcfg.max_participation = float("inf")
        res = run_backtest(bt, signal_strategy(sig), bcfg)
        bh = run_backtest(bt, buy_and_hold(), bcfg)
        sd = sharpe_difference_bootstrap(
            simple_returns(res.equity_net), simple_returns(bh.equity_net), ppy, seed=SEED
        )
        s, b = summarize(res.equity_net, ppy), summarize(bh.equity_net, ppy)
        rows.append(
            {
                "symbol": sym,
                "market": inst.market,
                "tsmom_sharpe": s["sharpe"],
                "bh_sharpe": b["sharpe"],
                "tsmom_mdd": s["max_drawdown"],
                "bh_mdd": b["max_drawdown"],
                **sd,
                "manifest_sha256": read_manifest(sym, inst.market, paths)["sha256"],
                "equity": res.ledger,
                "bh": bh.ledger,
            }
        )
    table = pd.DataFrame([{k: v for k, v in r.items() if k not in ("equity", "bh")} for r in rows])
    rej, p_adj, *_ = multipletests(table["p_two_sided"], alpha=0.05, method="fdr_bh")
    table["p_bh"], table["bh_significant"] = p_adj, rej
    pos = int(((table["delta_sr"] > 0) & table["bh_significant"]).sum())
    neg = int(((table["delta_sr"] < 0) & table["bh_significant"]).sum())
    family: Conclusion = (
        "supported" if pos >= 3 and neg == 0 else ("rejected" if neg > pos else "inconclusive")
    )
    for r, (_, t) in zip(rows, table.iterrows(), strict=True):
        concl: Conclusion = (
            "supported" if t["ci_low"] > 0 else ("rejected" if t["ci_high"] < 0 else "inconclusive")
        )
        rec = tracker.new(
            "tsmom_sharpe_vs_bh_holdout",
            "H-XM-003",
            "60d TSMOM long/flat has a higher net Sharpe than buy-and-hold (held-out instrument).",
            datasets=[{"symbol": t["symbol"], "sha256": t["manifest_sha256"]}],
            features=["mom_60"],
            model={"type": "rule", "lookback": 60},
            costs=cfg.markets[t["market"]].costs_bps.model_dump(),
            seed=SEED,
            validation={
                "type": "confirmatory, held-out instrument",
                "bootstrap": "paired stationary, block 20, 2000",
            },
            metrics={
                k: (float(v) if isinstance(v, int | float) else v)
                for k, v in t.items()
                if k not in ("manifest_sha256",)
            },
            conclusion=concl,
            reproduce="uv run astra holdout",
        )
        eq = r["equity"].copy()
        eq["bh_equity_net"] = r["bh"]["equity_net"]
        tracker.save_frame(rec, "equity", eq)
        tracker.log(rec)
    summary = tracker.new(
        "tsmom_holdout_family",
        "H-XM-003",
        "Family decision across 10 held-out instruments (BH).",
        datasets=[{"symbols": symbols}],
        metrics={
            "bh_significant_positive": pos,
            "bh_significant_negative": neg,
            "median_delta_sr": float(table["delta_sr"].median()),
        },
        variants_tried=len(table),
        conclusion=family,
        reproduce="uv run astra holdout",
    )
    tracker.save_frame(summary, "table", table)
    tracker.log(summary)
    return {"conclusion": family, "positive": pos, "negative": neg, "table": table}
