"""Walk-forward training of scikit-learn style classifiers with out-of-sample predictions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nebula.ml.validation import PurgedWalkForward
from nebula.stats.inference import newey_west_tstat


def logistic_factory(c: float = 0.1, seed: int = 0) -> Callable[[], Pipeline]:
    def make() -> Pipeline:
        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(C=c, max_iter=1000, random_state=seed))]
        )

    return make


@dataclass
class WalkForwardResult:
    proba: pd.Series  # OOS P(y=1), indexed like input
    baseline_proba: pd.Series  # each test row's TRAINING-fold base rate (known at prediction time)
    folds: list[dict[str, Any]] = field(default_factory=list)


def walk_forward_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    make_model: Callable[[], ClassifierMixin | Pipeline],
    cv: PurgedWalkForward,
) -> WalkForwardResult:
    """Fit on each training window (scaler inside the pipeline -> fit on train only); predict test."""
    if not X.index.equals(y.index):
        raise ValueError("X and y must share an index")
    proba = pd.Series(np.nan, index=X.index, dtype=float)
    baseline = pd.Series(np.nan, index=X.index, dtype=float)
    folds = []
    Xv, yv = X.to_numpy(), y.to_numpy()
    for k, (tr, te) in enumerate(cv.split(len(X))):
        model = make_model()
        model.fit(Xv[tr], yv[tr])
        p = model.predict_proba(Xv[te])[:, 1]  # type: ignore[union-attr]
        proba.iloc[te] = p
        baseline.iloc[te] = float(yv[tr].mean())
        folds.append(
            {
                "fold": k,
                "train_start": str(X.index[tr[0]]),
                "train_end": str(X.index[tr[-1]]),
                "test_start": str(X.index[te[0]]),
                "test_end": str(X.index[te[-1]]),
                "n_train": len(tr),
                "n_test": len(te),
                "train_base_rate": float(yv[tr].mean()),
            }
        )
    return WalkForwardResult(proba=proba, baseline_proba=baseline, folds=folds)


def walk_forward_from_splits(
    X: np.ndarray,
    y: np.ndarray,
    make_model: Callable[[], Any],
    splits: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Generic walk-forward over explicit (train_idx, test_idx) splits (e.g. date-based panel folds).
    Returns OOS probabilities, ex-ante train-fold base-rate baseline, and fold info."""
    proba = np.full(len(X), np.nan)
    base = np.full(len(X), np.nan)
    folds = []
    for k, (tr, te) in enumerate(splits):
        model = make_model()
        model.fit(X[tr], y[tr])
        proba[te] = model.predict_proba(X[te])[:, 1]
        base[te] = float(y[tr].mean())
        folds.append({"fold": k, "n_train": len(tr), "n_test": len(te)})
    return proba, base, folds


def _pointwise_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def classification_metrics(
    y: pd.Series, proba: pd.Series, baseline_proba: pd.Series | None = None
) -> dict[str, Any]:
    """OOS classification metrics against a no-skill baseline.

    `baseline_proba` should be the training-fold base rate per row (an honest, ex-ante baseline).
    If omitted, the OOS sample's own base rate is used - that baseline uses hindsight and is only
    reported for reference (`hindsight_baseline_log_loss`).
    `ll_improvement_nw_t` is a Diebold-Mariano style Newey-West t-stat on per-row log-loss
    differences (baseline - model); > 1.96 = model significantly better, < -1.96 = significantly worse.
    """
    m = proba.notna() & y.notna()
    if baseline_proba is not None:
        m &= baseline_proba.notna()
    yt, pt = y[m].astype(int).to_numpy(), proba[m].clip(1e-6, 1 - 1e-6).to_numpy()
    base = float(yt.mean())
    bt = baseline_proba[m].to_numpy() if baseline_proba is not None else np.full_like(pt, base)
    ll_model, ll_base = _pointwise_log_loss(yt, pt), _pointwise_log_loss(yt, bt)
    return {
        "n": int(m.sum()),
        "base_rate": base,
        "auc": float(roc_auc_score(yt, pt)) if len(set(yt)) > 1 else float("nan"),
        "log_loss": float(ll_model.mean()),
        "brier": float(brier_score_loss(yt, pt)),
        "accuracy": float(((pt > 0.5).astype(int) == yt).mean()),
        "baseline_log_loss": float(ll_base.mean()),
        "baseline_kind": "train-fold base rate"
        if baseline_proba is not None
        else "OOS base rate (hindsight)",
        "hindsight_baseline_log_loss": float(log_loss(yt, np.full_like(pt, base), labels=[0, 1])),
        "ll_improvement_nw_t": newey_west_tstat(pd.Series(ll_base - ll_model)),
        "baseline_brier": float(brier_score_loss(yt, bt)),
        "baseline_accuracy": float(max(base, 1 - base)),
    }
