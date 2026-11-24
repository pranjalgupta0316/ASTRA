"""H-ML-001 (per-instrument model x instrument grid) and H-ML-002 (pooled panel, deep learning).

Registered in docs/research/H-ML.md before execution. Fixed hyperparameters; no tuning.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

from nebula.common.config import get_config
from nebula.common.logging import get_logger
from nebula.common.paths import Paths, get_paths
from nebula.common.seed import set_seed
from nebula.dl.device import gpu_memory, reset_peak
from nebula.dl.models import TorchClassifier
from nebula.features.technical import FEATURE_COLUMNS
from nebula.ml.classical import (
    _pointwise_log_loss,
    classification_metrics,
    logistic_factory,
    walk_forward_from_splits,
)
from nebula.ml.datasets import date_folds, lag_columns, load_features, model_frame, panel
from nebula.ml.validation import PurgedWalkForward
from nebula.stats.inference import newey_west_tstat
from nebula.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)
SEED = 7
SEQ_LEN = 10
MODELS = ["logit", "rf", "xgb", "mlp", "gru"]


def model_factory(name: str, device: str | None = None) -> Callable[[], Any]:
    if name == "logit":
        return logistic_factory(c=0.1, seed=SEED)
    if name == "rf":
        return lambda: RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=50, n_jobs=-1, random_state=SEED
        )
    if name == "xgb":
        return lambda: XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
            verbosity=0,
        )
    if name == "mlp":
        return lambda: TorchClassifier(kind="mlp", seed=SEED, device=device)
    if name == "gru":
        return lambda: TorchClassifier(
            kind="gru", n_features=len(FEATURE_COLUMNS), seq_len=SEQ_LEN, seed=SEED, device=device
        )
    raise ValueError(name)


def _inputs(df: pd.DataFrame, name: str) -> np.ndarray:
    cols = lag_columns(SEQ_LEN) if name == "gru" else FEATURE_COLUMNS
    return df[cols].to_numpy(dtype=np.float64)


def _date_dm(y: np.ndarray, p_ref: np.ndarray, p_model: np.ndarray, dates: np.ndarray) -> float:
    """NW t-stat of the per-date mean of (log loss of reference - log loss of model); > 0 favours model."""
    """Row-weighted mean loss difference with date-clustered (Newey-West over dates) inference:
    per-date SUMS scaled by the mean rows per date, so every row has equal weight (crypto-only weekend
    dates are not over-weighted) while within-date correlation is absorbed by clustering."""
    d = _pointwise_log_loss(y, p_ref) - _pointwise_log_loss(y, p_model)
    g = pd.Series(d).groupby(dates)
    daily = g.sum() / g.size().mean()
    return newey_west_tstat(daily)


def _one_sided_p(t: float) -> float:
    return float(1 - stats.norm.cdf(t)) if np.isfinite(t) else 1.0


def run_instrument_grid(
    symbols: list[str] | None = None, device: str | None = None, paths: Paths | None = None
) -> dict[str, Any]:
    """H-ML-001: every model on every instrument, identical rows and folds per instrument."""
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or cfg.symbols(tier="core")
    rows = []
    t0 = time.perf_counter()
    tracemalloc.start()
    reset_peak()
    for sym in symbols:
        ppy = cfg.market_of(sym).periods_per_year
        df = model_frame(load_features(sym, paths, window=cfg.core_window()), SEQ_LEN)
        y = df["target_up"].to_numpy()
        cv = PurgedWalkForward(min_train=2 * ppy, test_size=ppy // 4, label_horizon=2, embargo=5)
        for name in MODELS:
            ts = time.perf_counter()
            proba, base, folds = walk_forward_from_splits(
                _inputs(df, name), y, model_factory(name, device), cv.split(len(df))
            )
            m = classification_metrics(pd.Series(y), pd.Series(proba), pd.Series(base))
            rows.append(
                {
                    "symbol": sym,
                    "model": name,
                    "n_oos": m["n"],
                    "n_folds": len(folds),
                    "auc": m["auc"],
                    "log_loss": m["log_loss"],
                    "baseline_log_loss": m["baseline_log_loss"],
                    "dm_t": m["ll_improvement_nw_t"],
                    "p_one_sided": _one_sided_p(m["ll_improvement_nw_t"]),
                    "brier": m["brier"],
                    "seconds": round(time.perf_counter() - ts, 2),
                }
            )
            log.info("ml.cell", symbol=sym, model=name, dm_t=m["ll_improvement_nw_t"])
    grid = pd.DataFrame(rows)
    reject, p_adj, *_ = multipletests(grid["p_one_sided"], alpha=0.05, method="fdr_bh")
    grid["p_bh"] = p_adj
    grid["bh_significant"] = reject
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_sig = int(grid["bh_significant"].sum())
    n_nominal = int((grid["dm_t"] > 1.96).sum())
    n_worse = int((grid["dm_t"] < -1.96).sum())
    conclusion: Conclusion = "supported" if n_sig > 0 else "rejected"
    tracker = Tracker(paths)
    rec = tracker.new(
        "ml_grid_per_instrument",
        "H-ML-001",
        "At least one of 5 models x 13 instruments beats its ex-ante base rate on OOS log loss after BH.",
        datasets=[{"symbols": symbols}],
        features=FEATURE_COLUMNS,
        model={"models": MODELS, "seq_len_gru": SEQ_LEN, "device": str(device or "auto")},
        validation={
            "type": "PurgedWalkForward expanding",
            "min_train": "2y",
            "test": "1 quarter",
            "label_horizon": 2,
            "embargo": 5,
        },
        seed=SEED,
        metrics={
            "cells": len(grid),
            "bh_significant": n_sig,
            "nominal_t_gt_1.96": n_nominal,
            "significantly_worse_t_lt_-1.96": n_worse,
            "median_auc": float(grid["auc"].median()),
            "runtime_s": round(time.perf_counter() - t0, 1),
            "peak_python_heap_mb": round(peak / 1e6, 1),
            "gpu": gpu_memory(),
        },
        variants_tried=len(grid),
        conclusion=conclusion,
        notes="Trading evaluation skipped unless a cell passes the BH gate (registered rule).",
        reproduce="uv run nebula ml-study",
    )
    tracker.save_frame(rec, "grid", grid)
    tracker.log(rec)
    return {
        "experiment_id": rec.experiment_id,
        "conclusion": conclusion,
        "bh_significant": n_sig,
        "nominal": n_nominal,
        "worse": n_worse,
        "grid": grid,
    }


def run_panel(
    symbols: list[str] | None = None, device: str | None = None, paths: Paths | None = None
) -> dict[str, Any]:
    """H-ML-002: pooled panel, date-based purged walk-forward, MLP/GRU vs logistic and baseline."""
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or cfg.symbols(tier="core")
    df = panel(symbols, SEQ_LEN, paths, window=cfg.core_window())
    y = df["target_up"].to_numpy()
    t0 = time.perf_counter()
    reset_peak()
    results: dict[str, dict[str, Any]] = {}
    probas: dict[str, np.ndarray] = {}
    base = None
    for name in ["logit", "mlp", "gru"]:
        splits = date_folds(df["session_date"], min_train_dates=730, test_dates=91, gap_dates=7)
        proba, base, folds = walk_forward_from_splits(
            _inputs(df, name), y, model_factory(name, device), splits
        )
        probas[name] = proba
        m = classification_metrics(pd.Series(y), pd.Series(proba), pd.Series(base))
        results[name] = {
            "auc": m["auc"],
            "log_loss": m["log_loss"],
            "baseline_log_loss": m["baseline_log_loss"],
            "dm_t_vs_baseline": m["ll_improvement_nw_t"],
            "n_oos": m["n"],
            "n_folds": len(folds),
        }
    # Inference on the DATE-aggregated loss differential (rows sharing a date are correlated).
    mask = ~np.isnan(probas["logit"])
    dates = df.loc[mask, "session_date"].to_numpy()
    yb, bb = y[mask], np.asarray(base)[mask]
    for name in ["logit", "mlp", "gru"]:
        results[name]["dm_t_vs_baseline_rowwise"] = results[name].pop("dm_t_vs_baseline")
        results[name]["dm_t_vs_baseline"] = _date_dm(yb, bb, probas[name][mask], dates)
    tracker = Tracker(paths)
    out: dict[str, Any] = {"rows": len(df), "symbols": symbols, "results": results}
    for name in ["mlp", "gru"]:
        t_vs_logit = _date_dm(yb, probas["logit"][mask], probas[name][mask], dates)
        results[name]["dm_t_vs_logit"] = t_vs_logit
        t_base = results[name]["dm_t_vs_baseline"]
        concl: Conclusion
        if t_base < -1.96:
            concl = "rejected"
        elif t_base > 1.96 and t_vs_logit > 1.96:
            concl = "supported"
        else:
            concl = "inconclusive"
        rec = tracker.new(
            f"panel_{name}",
            "H-ML-002",
            f"Pooled {name.upper()} beats ex-ante baseline and pooled logistic on OOS log loss.",
            datasets=[{"symbols": symbols, "rows": len(df)}],
            features=lag_columns(SEQ_LEN) if name == "gru" else FEATURE_COLUMNS,
            model={"type": name, "seq_len": SEQ_LEN if name == "gru" else 1},
            validation={
                "type": "date-based purged expanding walk-forward",
                "min_train_dates": 730,
                "test_dates": 91,
                "gap_dates": 7,
            },
            seed=SEED,
            metrics={
                **results[name],
                "logit": results["logit"],
                "runtime_s": round(time.perf_counter() - t0, 1),
                "gpu": gpu_memory(),
            },
            conclusion=concl,
            notes="DM t-stats computed on per-date mean log-loss differentials (Newey-West), because rows "
            "sharing a date are cross-sectionally correlated; row-wise t kept for reference.",
            reproduce="uv run nebula ml-study --panel",
        )
        tracker.log(rec)
        out[name] = {"conclusion": concl, "id": rec.experiment_id, **results[name]}
    out["logit"] = results["logit"]
    out["runtime_s"] = round(time.perf_counter() - t0, 1)
    out["gpu"] = gpu_memory()
    return out
