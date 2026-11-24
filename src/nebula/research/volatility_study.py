"""M5 volatility-forecasting study (H-VOL-001, registered in docs/research/H-VOL.md before results).

For each instrument: one-step-ahead daily variance forecasts from rolling-mean r^2 (20, 60), EWMA(0.94),
GARCH(1,1)-t and GJR-GARCH(1,1,1)-t (expanding refits every 21 sessions, min train 504 returns),
evaluated out of sample with QLIKE and MSE against the r^2 proxy; Diebold-Mariano (Newey-West) of each
model vs EWMA. Decision (GJR vs EWMA, QLIKE) uses Benjamini-Hochberg across the instruments run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from nebula.common.config import get_config
from nebula.common.logging import get_logger
from nebula.common.paths import Paths, get_paths
from nebula.common.seed import set_seed
from nebula.data.storage import read_bars, read_manifest
from nebula.features.technical import adjusted_ohlc
from nebula.stats.volatility import (
    diebold_mariano,
    ewma_forecast,
    garch_forecast,
    log_returns,
    mse_loss,
    qlike,
    qlike_rank,
    realized_variance_proxies,
    rolling_mean_forecast,
)
from nebula.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)
SEED = 7
MIN_TRAIN = 504
REFIT_EVERY = 21
EWMA_LAMBDA = 0.94
GARCH_DIST = "t"
BH_Q = 0.05
VOLUME_UNRELIABLE = {"NIFTY50"}  # Yahoo index volume is not a trading-activity signal
MODELS = ["roll20", "roll60", "ewma", "garch", "gjr"]
BASELINE = "ewma"


def load_study_frame(
    symbol: str, paths: Paths | None = None
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Adjusted OHLC indexed by session date, placeholder rows removed -> (frame, manifest, excluded)."""
    paths = paths or get_paths()
    inst = get_config().instrument(symbol)
    bars = read_bars(symbol, inst.market, paths, window=get_config().core_window())
    manifest = read_manifest(symbol, inst.market, paths)
    if symbol in VOLUME_UNRELIABLE:
        placeholder = bars["high"] == bars["low"]
    else:
        placeholder = (bars["volume"] == 0) & (bars["high"] == bars["low"])
    excluded = [str(d) for d in bars.loc[placeholder, "session_date"]]
    bars = bars.loc[~placeholder].reset_index(drop=True)
    adj = adjusted_ohlc(bars).set_index("session_date")
    return adj, manifest, excluded


def build_forecast_frame(adj: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Origin-indexed frame: realised proxies for t+1 and every model's forecast made at t (decimal)."""
    rv = realized_variance_proxies(adj)
    r = log_returns(adj["close"]).iloc[1:]
    rv = rv.iloc[1:]
    fc = pd.DataFrame(index=r.index)
    fc["roll20"] = rolling_mean_forecast(r, 20)
    fc["roll60"] = rolling_mean_forecast(r, 60)
    fc["ewma"] = ewma_forecast(r, EWMA_LAMBDA)
    g = garch_forecast(r, "garch", REFIT_EVERY, MIN_TRAIN, GARCH_DIST)
    j = garch_forecast(r, "gjr", REFIT_EVERY, MIN_TRAIN, GARCH_DIST)
    fc["garch"], fc["gjr"] = g.forecast, j.forecast
    frame = pd.DataFrame(
        {
            "target_date": pd.Series(r.index, index=r.index).shift(-1),
            "r_next": r.shift(-1),
            "rv_cc_next": rv["rv_cc"].shift(-1),
            "rv_pk_next": rv["rv_pk"].shift(-1),
            "rv_gk_next": rv["rv_gk"].shift(-1),
        },
        index=r.index,
    )
    frame = frame.join(fc)
    frame["oos"] = (np.arange(len(frame)) >= MIN_TRAIN - 1) & frame["r_next"].notna().to_numpy()
    frame.index.name = "origin_date"
    return frame, {"garch": g, "gjr": j}


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    """Losses in percent^2 units (variance x 1e4) on OOS rows; DM of each model vs EWMA."""
    ev = frame[frame["oos"]]
    if ev[MODELS].isna().any().any() or (ev[MODELS] <= 0).any().any():
        raise ValueError("missing or non-positive forecasts in the OOS window")
    rv = ev["rv_cc_next"].to_numpy() * 1e4
    gk = ev["rv_gk_next"].clip(lower=1e-12).to_numpy() * 1e4
    pos = rv > 0
    losses: dict[str, dict[str, np.ndarray]] = {}
    table: dict[str, dict[str, float]] = {}
    for m in MODELS:
        h = ev[m].to_numpy() * 1e4
        losses[m] = {"qlike": qlike_rank(rv, h), "mse": mse_loss(rv, h), "qlike_gk": qlike_rank(gk, h)}
        table[m] = {
            "qlike_mean": float(np.mean(qlike(rv[pos], h[pos]))),
            "qlike_rank_mean": float(np.mean(losses[m]["qlike"])),
            "mse_mean": float(np.mean(losses[m]["mse"])),
            "qlike_gk_rank_mean": float(np.mean(losses[m]["qlike_gk"])),
            "mean_forecast_pct2": float(np.mean(h)),
        }
    dm: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        if m == BASELINE:
            continue
        dm[m] = {
            loss: diebold_mariano(losses[m][loss], losses[BASELINE][loss])
            for loss in ("qlike", "mse", "qlike_gk")
        }
    half = len(ev) // 2
    stab = {
        "first_half": diebold_mariano(losses["gjr"]["qlike"][:half], losses[BASELINE]["qlike"][:half]),
        "second_half": diebold_mariano(losses["gjr"]["qlike"][half:], losses[BASELINE]["qlike"][half:]),
        "split_date": str(ev.index[half].date()),
    }
    return {
        "n_eval": len(ev),
        "n_zero_rv": int((~pos).sum()),
        "mean_rv_pct2": float(rv.mean()),
        "oos_start": str(ev.index[0].date()),
        "oos_end": str(ev.index[-1].date()),
        "models": table,
        "dm_vs_ewma": dm,
        "gjr_vs_ewma_subperiods": stab,
    }


def _verdict(p_bh: float, reject: bool, t: float) -> Conclusion:
    if reject and t < 0:
        return "supported"
    if reject and t > 0:
        return "rejected"
    return "inconclusive"


def run_volatility_study(symbols: list[str] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    """Run H-VOL-001 on `symbols` (default: every configured instrument). Logs one record per instrument."""
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or cfg.symbols(tier="core")
    tracker = Tracker(paths)
    per: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        adj, manifest, excluded = load_study_frame(sym, paths)
        frame, fits = build_forecast_frame(adj)
        res = evaluate(frame)
        per[sym] = {"frame": frame, "fits": fits, "res": res, "manifest": manifest, "excluded": excluded}
        log.info("vol.evaluated", symbol=sym, dm_t=res["dm_vs_ewma"]["gjr"]["qlike"]["dm_t"])

    pvals = np.array([per[s]["res"]["dm_vs_ewma"]["gjr"]["qlike"]["p_two_sided"] for s in symbols])
    reject, p_bh, *_ = multipletests(pvals, alpha=BH_Q, method="fdr_bh")
    summary_rows = []
    out: dict[str, Any] = {
        "hypothesis_id": "H-VOL-001",
        "bh_q": BH_Q,
        "n_instruments": len(symbols),
        "per_symbol": {},
    }
    for k, sym in enumerate(symbols):
        d = per[sym]
        res = d["res"]
        dmq = res["dm_vs_ewma"]["gjr"]["qlike"]
        concl = _verdict(float(p_bh[k]), bool(reject[k]), dmq["dm_t"])
        inst = cfg.instrument(sym)
        rec = tracker.new(
            "vol_gjr_vs_ewma_qlike",
            "H-VOL-001",
            "GJR-GARCH(1,1,1)-t one-step variance forecasts have lower QLIKE than EWMA(0.94) (DM, BH across "
            "instruments).",
            datasets=[d["manifest"]],
            features=["log_return_cc"],
            model={
                "forecasters": MODELS,
                "baseline": BASELINE,
                "ewma_lambda": EWMA_LAMBDA,
                "garch": {"mean": "Zero", "dist": GARCH_DIST, "scale": "percent", "refit_every": REFIT_EVERY},
                "rolling_windows": [20, 60],
            },
            validation={
                "type": "expanding walk-forward, one-step-ahead",
                "min_train": MIN_TRAIN,
                "refit_every": REFIT_EVERY,
                "oos_start": res["oos_start"],
                "oos_end": res["oos_end"],
                "test_touches": 1,
                "proxy": "r_{t+1}^2 close-to-close (primary); GK descriptive",
                "decision": "BH(q=0.05) over instruments on two-sided DM p of QLIKE(GJR)-QLIKE(EWMA)",
                "family": symbols,
                "garch_refits": {
                    "garch": d["fits"]["garch"].meta,
                    "gjr": d["fits"]["gjr"].meta,
                },
            },
            seed=SEED,
            metrics={
                **res,
                "dm_t": dmq["dm_t"],
                "dm_p": dmq["p_two_sided"],
                "dm_p_bh": float(p_bh[k]),
                "qlike_gjr": res["models"]["gjr"]["qlike_mean"],
                "qlike_ewma": res["models"]["ewma"]["qlike_mean"],
            },
            conclusion=concl,
            variants_tried=1,
            notes=f"symbol={sym}; market={inst.market}; Student-t GARCH fixed a priori; secondary model and "
            f"loss comparisons are descriptive only. Excluded placeholder rows: {d['excluded']}",
            reproduce='uv run --no-sync python -c "from nebula.research.volatility_study import '
            'run_volatility_study as f; f()"',
        )
        tracker.save_frame(rec, "forecasts", d["frame"].reset_index())
        for kind in ("garch", "gjr"):
            tracker.save_frame(rec, f"refits_{kind}", d["fits"][kind].refits)
        tracker.log(rec)
        row = {
            "symbol": sym,
            "market": inst.market,
            "n_eval": res["n_eval"],
            **{f"qlike_{m}": res["models"][m]["qlike_mean"] for m in MODELS},
            **{f"dm_t_{m}": res["dm_vs_ewma"][m]["qlike"]["dm_t"] for m in MODELS if m != BASELINE},
            "dm_t_gjr_mse": res["dm_vs_ewma"]["gjr"]["mse"]["dm_t"],
            "dm_t_gjr_qlike_gk": res["dm_vs_ewma"]["gjr"]["qlike_gk"]["dm_t"],
            "dm_p": dmq["p_two_sided"],
            "dm_p_bh": float(p_bh[k]),
            "conclusion": concl,
            "experiment_id": rec.experiment_id,
        }
        summary_rows.append(row)
        out["per_symbol"][sym] = row
    tab = pd.DataFrame(summary_rows)
    out["table"] = tab
    out["counts"] = tab["conclusion"].value_counts().to_dict()
    out["raw_sig_gjr_better"] = int((tab["dm_t_gjr"] < -1.96).sum())
    out["raw_sig_ewma_better"] = int((tab["dm_t_gjr"] > 1.96).sum())
    return out
