"""M5 regime study (H-REG-001, registered in docs/research/H-VOL.md before results).

2-state Gaussian mixture on (20d realised vol, 20d return), expanding walk-forward (min train 504
returns, refit every 63 sessions), vol-sorted labels. Tests whether the OOS high-vol state predicts
higher next-day realised variance r_{t+1}^2 (Mann-Whitney, BH across instruments, plus HAC check).
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
from nebula.research.volatility_study import load_study_frame
from nebula.stats.regimes import (
    FEATURES,
    persistence,
    regime_features,
    state_difference_tests,
    walk_forward_gmm,
)
from nebula.stats.volatility import diebold_mariano, log_returns, qlike_rank
from nebula.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)
SEED = 7
MIN_TRAIN = 504
REFIT_EVERY = 63
N_STATES = 2
N_INIT = 3
BH_Q = 0.05


def build_state_frame(adj: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Origin-indexed frame with features, OOS state, next-day r and r^2, and the conditional forecasts."""
    r = log_returns(adj["close"]).iloc[1:]
    feats = regime_features(r)
    rv_next = (r**2).shift(-1)
    wf = walk_forward_gmm(feats, N_STATES, MIN_TRAIN, REFIT_EVERY, SEED, N_INIT, target_next=rv_next)
    frame = feats.copy()
    frame["state"] = wf.states
    frame["r_next"] = r.shift(-1)
    frame["rv_next"] = rv_next
    frame["cond_fc"] = wf.cond_forecast
    frame["uncond_fc"] = wf.uncond_forecast
    frame.index.name = "origin_date"
    return frame, {"refits": wf.refits, "meta": wf.meta}


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    st = frame["state"]
    oos = frame[st.notna()]
    ev = oos[oos["rv_next"].notna()]
    high = (ev["state"] == N_STATES - 1).astype(float)
    tests_rv = state_difference_tests(ev["rv_next"], high)
    tests_abs = state_difference_tests(ev["r_next"].abs(), high)
    pers = persistence(oos["state"])
    rv = ev["rv_next"].to_numpy() * 1e4
    lc = qlike_rank(rv, ev["cond_fc"].to_numpy() * 1e4)
    lu = qlike_rank(rv, ev["uncond_fc"].to_numpy() * 1e4)
    return {
        "n_eval": len(ev),
        "oos_start": str(ev.index[0].date()),
        "oos_end": str(ev.index[-1].date()),
        "share_high": float(high.mean()),
        "persistence": pers,
        "rv_next": tests_rv,
        "abs_r_next": tests_abs,
        "cond_vs_uncond_qlike": {
            "qlike_rank_cond": float(np.mean(lc)),
            "qlike_rank_uncond": float(np.mean(lu)),
            **diebold_mariano(lc, lu),
        },
    }


def _verdict(reject_bh: bool, hac_t: float) -> Conclusion:
    if np.isfinite(hac_t) and hac_t < -1.96:
        return "rejected"
    if reject_bh and np.isfinite(hac_t) and hac_t > 1.96:
        return "supported"
    return "inconclusive"


def run_regime_study(symbols: list[str] | None = None, paths: Paths | None = None) -> dict[str, Any]:
    """Run H-REG-001 on `symbols` (default: every configured instrument). Logs one record per instrument."""
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or [i.symbol for i in cfg.instruments]
    tracker = Tracker(paths)
    per: dict[str, dict[str, Any]] = {}
    for sym in symbols:
        adj, manifest, excluded = load_study_frame(sym, paths)
        frame, fit = build_state_frame(adj)
        per[sym] = {
            "frame": frame,
            "fit": fit,
            "res": evaluate(frame),
            "manifest": manifest,
            "excluded": excluded,
        }
        log.info("regime.evaluated", symbol=sym)

    p = np.array([per[s]["res"]["rv_next"]["mw_p_one_sided"] for s in symbols], dtype=float)
    p_filled = np.where(np.isfinite(p), p, 1.0)  # a state never visited OOS counts as no evidence
    reject, p_bh, *_ = multipletests(p_filled, alpha=BH_Q, method="fdr_bh")
    rows = []
    out: dict[str, Any] = {
        "hypothesis_id": "H-REG-001",
        "bh_q": BH_Q,
        "n_instruments": len(symbols),
        "per_symbol": {},
    }
    for k, sym in enumerate(symbols):
        d = per[sym]
        res = d["res"]
        t_rv = res["rv_next"]
        concl = _verdict(bool(reject[k]), t_rv["hac_t"])
        inst = cfg.instrument(sym)
        rec = tracker.new(
            "gmm2_state_next_day_rv",
            "H-REG-001",
            "OOS high-vol GMM state predicts higher next-day realised variance than the low-vol state.",
            datasets=[d["manifest"]],
            features=FEATURES,
            model={
                "type": "GaussianMixture",
                "n_components": N_STATES,
                "covariance_type": "full",
                "n_init": N_INIT,
                "standardise": "training-window mean/std",
                "label_mapping": "sort by training mean vol20 (state 1 = high vol)",
            },
            validation={
                "type": "expanding walk-forward",
                "min_train": MIN_TRAIN,
                "refit_every": REFIT_EVERY,
                "oos_start": res["oos_start"],
                "oos_end": res["oos_end"],
                "test_touches": 1,
                "decision": "BH(q=0.05) on one-sided Mann-Whitney p of r_{t+1}^2 AND HAC t > 1.96 "
                "-> supported; "
                "HAC t < -1.96 -> rejected; else inconclusive",
                "family": symbols,
                "gmm_meta": d["fit"]["meta"],
            },
            seed=SEED,
            metrics={
                **res,
                "mw_p": t_rv["mw_p_one_sided"],
                "mw_p_bh": float(p_bh[k]),
                "hac_t": t_rv["hac_t"],
                "ratio_mean_rv": t_rv["ratio_mean"],
                "oos_persistence": res["persistence"]["persistence"],
            },
            conclusion=concl,
            variants_tried=1,
            notes=f"symbol={sym}; market={inst.market}; persistence/conditional-forecast metrics "
            "descriptive. "
            f"Excluded placeholder rows: {d['excluded']}",
            reproduce="uv run --no-sync python -c "
            '"from nebula.research.regime_study import run_regime_study as f; f()"',
        )
        tracker.save_frame(rec, "states", d["frame"].reset_index())
        tracker.save_frame(rec, "refits", d["fit"]["refits"])
        tracker.log(rec)
        row = {
            "symbol": sym,
            "market": inst.market,
            "n_eval": res["n_eval"],
            "share_high": res["share_high"],
            "persistence": res["persistence"]["persistence"],
            "persistence_indep": res["persistence"]["independence_benchmark"],
            "mean_rv_high_pct2": t_rv["mean_high"] * 1e4,
            "mean_rv_low_pct2": t_rv["mean_low"] * 1e4,
            "ratio_mean_rv": t_rv["ratio_mean"],
            "welch_t": t_rv["welch_t"],
            "mw_p": t_rv["mw_p_one_sided"],
            "mw_p_bh": float(p_bh[k]),
            "hac_t": t_rv["hac_t"],
            "dm_t_cond_vs_uncond": res["cond_vs_uncond_qlike"]["dm_t"],
            "conclusion": concl,
            "experiment_id": rec.experiment_id,
        }
        rows.append(row)
        out["per_symbol"][sym] = row
    tab = pd.DataFrame(rows)
    out["table"] = tab
    out["counts"] = tab["conclusion"].value_counts().to_dict()
    return out
