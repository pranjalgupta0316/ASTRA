"""H-MM-001: multimodal ablation on US stocks (price + SEC fundamentals + macro), registered in
docs/research/H-EXT.md before execution.

Design (as registered):
- Rows: month-end decision dates (last XNYS session of each calendar month in the data; the month of the
  final available session is excluded as possibly incomplete). Instruments enter after 252 sessions.
- Target: 1[R_i - R_EW > 0], R = simple return from the open of t+1 to the open of t+22 (21 sessions held;
  decision at close t, fill at open t+1), R_EW = equal-weight mean of R over the eligible US stocks that
  have the same forward window on that date.
- Feature sets P / P+F / P+M / P+F+M. P = 13 price features at t. F = point-in-time SEC fundamentals
  (scale-free subset of `fundamental_features`, chosen before running: revenue_growth_yoy, net_margin,
  roe, leverage, earnings_yield, earnings_yield_ni, book_to_market, log_market_cap; raw USD/EPS levels are
  not comparable across firms and are excluded). M = every column of `macro_features` (as-of).
- Models: fixed logit (C=0.1, standardized) and XGBoost (H-ML-001/002 settings). Missing F/M values:
  median imputation fitted on the training fold only + missing-indicator columns (sklearn Pipeline).
- Monthly expanding walk-forward: 60 decision months initial training, 12-month test blocks, purge 1 month
  + embargo 1 month (the 2 decision months before each test block are dropped from training).
- Inference: row-weighted DM on per-row log-loss differences clustered by decision month (Newey-West over
  months); each set vs P; BH over 3 comparisons x 2 models; one-sided.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

from nebula.common.config import get_config
from nebula.common.logging import get_logger
from nebula.common.paths import Paths, get_paths
from nebula.common.seed import set_seed
from nebula.data.storage import read_bars
from nebula.features.technical import FEATURE_COLUMNS, build_features
from nebula.ml.classical import _pointwise_log_loss
from nebula.research.ml_extended import (
    MIN_HISTORY,
    SURVIVORSHIP,
    ResourceProbe,
    _safe_auc,
    data_versions,
    fold_dispersion,
    importance_summary,
)
from nebula.research.ml_study import SEED, _date_dm, _one_sided_p
from nebula.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)

HORIZON = 21
PRICE = list(FEATURE_COLUMNS)
FUND = [
    "revenue_growth_yoy",
    "net_margin",
    "roe",
    "leverage",
    "earnings_yield",
    "earnings_yield_ni",
    "book_to_market",
    "log_market_cap",
]
SETS = ["P", "P+F", "P+M", "P+F+M"]
COMPARISONS = ["P+F", "P+M", "P+F+M"]
MODELS = ["logit", "xgb"]
INITIAL_MONTHS, TEST_MONTHS, PURGE_MONTHS, EMBARGO_MONTHS = 60, 12, 1, 1


# ------------------------------------------------------------------------------ rows and target


def clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Placeholder exclusion identical to `nebula.ml.datasets.load_features` for stocks."""
    flat = bars["high"] == bars["low"]
    return bars.loc[~(flat & (bars["volume"] == 0))].reset_index(drop=True)


def month_end_dates(sessions: pd.Series) -> pd.DatetimeIndex:
    """Last session of each calendar month; the month of the final session is dropped (incomplete)."""
    s = pd.Series(pd.to_datetime(pd.unique(pd.to_datetime(sessions)))).sort_values()
    per = s.dt.to_period("M")
    last = s.groupby(per).max()
    last = last[last.index != per.iloc[-1]]
    return pd.DatetimeIndex(last.to_numpy())


def symbol_daily(symbol: str, bars: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Daily price features + forward return R(t) = O_{t+1+h} / O_{t+1} - 1 (adjusted opens) and the
    bar position (for the 252-session entry rule)."""
    b = clean_bars(bars)
    f = build_features(b)
    out = f[["session_date", *PRICE]].copy()
    out["symbol"] = symbol
    out["pos"] = np.arange(len(f))
    o = f["adj_open"]
    out["fwd_ret"] = o.shift(-(1 + horizon)) / o.shift(-1) - 1.0
    return out


def add_excess_target(rows: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional target on eligible rows (fwd_ret defined): excess = R_i - mean_j R_j per date."""
    r = rows.dropna(subset=["fwd_ret"]).copy()
    r["ew_ret"] = r.groupby("session_date")["fwd_ret"].transform("mean")
    r["n_universe"] = r.groupby("session_date")["fwd_ret"].transform("size")
    r["excess_ret"] = r["fwd_ret"] - r["ew_ret"]
    r["target"] = (r["excess_ret"] > 0).astype(float)
    return r


def fundamentals_block(
    symbol: str, bars: pd.DataFrame, facts: pd.DataFrame | None, paths: Paths | None
) -> pd.DataFrame:
    from nebula.data.fundamentals import fundamental_features

    ff = fundamental_features(symbol, bars, facts=facts, paths=paths)
    ff["session_date"] = pd.to_datetime(ff["session_date"])
    with np.errstate(divide="ignore", invalid="ignore"):
        mc = pd.to_numeric(ff["market_cap"], errors="coerce")
        ff["log_market_cap"] = np.log(mc.where(mc > 0))
    # Non-finite ratios (e.g. book_to_market when a filed share count of 0 gives market_cap = 0) are
    # treated as missing -> imputed on the training fold with a missing indicator.
    ff[FUND] = ff[FUND].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return ff[["session_date", *FUND, "fund_filed"]]


def build_monthly_frame(
    bars_by_symbol: dict[str, pd.DataFrame],
    facts: pd.DataFrame | None = None,
    macro_table: pd.DataFrame | None = None,
    paths: Paths | None = None,
    min_history: int = MIN_HISTORY,
    with_fundamentals: bool = True,
    with_macro: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Month-end decision rows for all symbols. Returns (frame, macro column names)."""
    daily, funds = [], []
    for sym, bars in bars_by_symbol.items():
        d = symbol_daily(sym, bars)
        daily.append(d)
        if with_fundamentals:
            funds.append(fundamentals_block(sym, bars, facts, paths).assign(symbol=sym))
    allday = pd.concat(daily, ignore_index=True)
    allday["session_date"] = pd.to_datetime(allday["session_date"])
    me = month_end_dates(allday["session_date"])
    rows = allday[allday["session_date"].isin(me) & (allday["pos"] >= min_history)]
    rows = add_excess_target(rows)
    rows = rows.dropna(subset=PRICE)
    if with_fundamentals:
        rows = rows.merge(pd.concat(funds, ignore_index=True), on=["session_date", "symbol"], how="left")
    else:
        for c in FUND:
            rows[c] = np.nan
        rows["fund_filed"] = pd.NaT
    macro_cols: list[str] = []
    if with_macro:
        from nebula.data.macro import macro_features

        dates = pd.Series(sorted(rows["session_date"].unique()))
        m = macro_features(dates, table=macro_table, paths=paths)
        m.index = pd.to_datetime(m.index)
        macro_cols = [f"m_{c}" for c in m.columns]
        m.columns = macro_cols
        rows = rows.merge(m, left_on="session_date", right_index=True, how="left")
    rows["month"] = rows["session_date"].dt.to_period("M")
    rows = rows.sort_values(["session_date", "symbol"], kind="mergesort").reset_index(drop=True)
    return rows, macro_cols


def feature_columns(feature_set: str, macro_cols: list[str]) -> list[str]:
    parts = feature_set.split("+")
    unknown = set(parts) - {"P", "F", "M"}
    if unknown or parts[0] != "P":
        raise ValueError(f"unknown feature set {feature_set!r}")
    cols = list(PRICE)
    if "F" in parts:
        cols += FUND
    if "M" in parts:
        cols += list(macro_cols)
    return cols


# ------------------------------------------------------------------------------ folds and models


def month_folds(
    months: pd.Series,
    initial: int = INITIAL_MONTHS,
    test: int = TEST_MONTHS,
    purge: int = PURGE_MONTHS,
    embargo: int = EMBARGO_MONTHS,
) -> Iterator[tuple[np.ndarray, np.ndarray, dict[str, Any]]]:
    """Expanding walk-forward over decision months: tests = blocks of `test` months after the first
    `initial` months; training = months strictly before (test start - purge - embargo)."""
    um = np.array(sorted(pd.unique(months)))
    m = months.to_numpy()
    start = initial
    k = 0
    while start < len(um):
        stop = min(start + test, len(um))
        cut = start - purge - embargo
        if cut > 0:
            tr = np.flatnonzero(m < um[cut])
            te = np.flatnonzero((m >= um[start]) & (m <= um[stop - 1]))
            if len(tr) and len(te):
                info = {
                    "fold": k,
                    "train_last_month": str(um[cut - 1]),
                    "test_first_month": str(um[start]),
                    "test_last_month": str(um[stop - 1]),
                    "n_train": len(tr),
                    "n_test": len(te),
                }
                yield tr, te, info
                k += 1
        start = stop


def _imputer() -> SimpleImputer:
    return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)


def make_model(name: str) -> Pipeline:
    """Fixed H-ML-001/002 hyperparameters, preceded by a train-fold-only median imputer + indicators."""
    if name == "logit":
        return Pipeline(
            [
                ("impute", _imputer()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)),
            ]
        )
    if name == "xgb":
        return Pipeline(
            [
                ("impute", _imputer()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.03,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=SEED,
                        n_jobs=4,
                        tree_method="hist",
                        verbosity=0,
                    ),
                ),
            ]
        )
    raise ValueError(name)


def walk_forward_set(
    df: pd.DataFrame,
    cols: list[str],
    model: str,
    folds: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    factory: Callable[[str], Any] = make_model,
    importance: bool = False,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    X = df[cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)
    proba, base = np.full(len(y), np.nan), np.full(len(y), np.nan)
    per_fold, imps = [], []
    for tr, te, info in folds:
        m = factory(model)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        proba[te], base[te] = p, float(y[tr].mean())
        per_fold.append({**info, "ll": float(_pointwise_log_loss(y[te], p).mean())})
        if importance:
            pi = permutation_importance(
                m, X[te], y[te], scoring="neg_log_loss", n_repeats=5, random_state=SEED
            )
            imps.append(
                pd.DataFrame(
                    {
                        "fold": info["fold"],
                        "feature": cols,
                        "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std,
                        "n_test": len(te),
                    }
                )
            )
    return (
        proba,
        base,
        pd.DataFrame(per_fold),
        (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame()),
    )


def _fold_deltas(fold_id: np.ndarray, ll_ref: np.ndarray, ll_new: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"fold": fold_id, "d": ll_ref - ll_new})
    return d.groupby("fold")["d"].mean().rename("ll_improvement").reset_index()


# ------------------------------------------------------------------------------ study


def run_multimodal(
    symbols: list[str] | None = None,
    paths: Paths | None = None,
    log_tracker: bool = True,
) -> dict[str, Any]:
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or cfg.symbols(market="US", kind="stock")
    t0 = time.perf_counter()
    from nebula.data.fundamentals import load_facts

    with ResourceProbe() as probe:
        facts = load_facts(None, paths)
        bars = {s: read_bars(s, "US", paths) for s in symbols}
        df, macro_cols = build_monthly_frame(bars, facts=facts, paths=paths)
        del bars
        folds = list(month_folds(df["month"]))
        fold_id = np.full(len(df), -1)
        for tr_te in folds:
            fold_id[tr_te[1]] = tr_te[2]["fold"]
        y = df["target"].to_numpy(dtype=np.float64)
        probas: dict[tuple[str, str], np.ndarray] = {}
        per_fold, imps = [], {}
        base = np.full(len(y), np.nan)
        for model in MODELS:
            for fs in SETS:
                set_seed(SEED)
                p, base, pf, imp = walk_forward_set(
                    df, feature_columns(fs, macro_cols), model, folds, importance=model == "xgb"
                )
                probas[(model, fs)] = p
                per_fold.append(pf.assign(model=model, feature_set=fs))
                if not imp.empty:
                    imps[fs] = imp.assign(feature_set=fs)
                log.info("mm.cell", model=model, feature_set=fs)
    mask = ~np.isnan(base)
    months = df.loc[mask, "month"].astype(str).to_numpy()
    yb, bb, fid = y[mask], base[mask], fold_id[mask]
    llb = _pointwise_log_loss(yb, bb)
    rows = [
        {
            "model": "baseline (ex-ante train-fold base rate)",
            "feature_set": "-",
            "n_oos": int(mask.sum()),
            "log_loss": float(llb.mean()),
            "auc": 0.5,
        }
    ]
    comps, disp = [], []
    for model in MODELS:
        llp = _pointwise_log_loss(yb, probas[(model, "P")][mask])
        for fs in SETS:
            pm = probas[(model, fs)][mask]
            ll = _pointwise_log_loss(yb, pm)
            rows.append(
                {
                    "model": model,
                    "feature_set": fs,
                    "n_oos": int(mask.sum()),
                    "log_loss": float(ll.mean()),
                    "auc": _safe_auc(yb, pm),
                    "dm_t_vs_baseline": _date_dm(yb, bb, pm, months),
                    "dm_t_vs_P": _date_dm(yb, probas[(model, "P")][mask], pm, months)
                    if fs != "P"
                    else np.nan,
                    "delta_ll_vs_P": float((llp - ll).mean()) if fs != "P" else np.nan,
                }
            )
        # marginal contributions (descriptive; only the 3 'vs P' comparisons enter the BH family)
        for ref, new, label in [
            ("P", "P+F", "F | P"),
            ("P", "P+M", "M | P"),
            ("P", "P+F+M", "F+M | P"),
            ("P+M", "P+F+M", "F | P+M"),
            ("P+F", "P+F+M", "M | P+F"),
        ]:
            pr, pn = probas[(model, ref)][mask], probas[(model, new)][mask]
            lr, ln = _pointwise_log_loss(yb, pr), _pointwise_log_loss(yb, pn)
            fd = _fold_deltas(fid, lr, ln)
            t = _date_dm(yb, pr, pn, months)
            post = pd.PeriodIndex(df.loc[mask, "month"]) >= pd.Period("2011-01", "M")
            comps.append(
                {
                    "model": model,
                    "contribution": label,
                    "ref": ref,
                    "new": new,
                    "delta_ll": float((lr - ln).mean()),
                    "dm_t": t,
                    "in_bh_family": ref == "P",
                    "p_one_sided": _one_sided_p(t) if ref == "P" else np.nan,
                    "delta_ll_test_2011plus": float((lr - ln)[post].mean()),
                    "dm_t_test_2011plus": _date_dm(yb[post], pr[post], pn[post], months[post]),
                    **{f"fold_{k}": v for k, v in fold_dispersion(fd).items()},
                }
            )
            disp.append(fd.assign(model=model, contribution=label))
    table = pd.DataFrame(rows)
    comp = pd.DataFrame(comps)
    fam = comp["in_bh_family"].to_numpy()
    reject, p_adj, *_ = multipletests(comp.loc[fam, "p_one_sided"], alpha=0.05, method="fdr_bh")
    comp["p_bh"], comp["bh_significant"] = np.nan, False
    comp.loc[fam, "p_bh"] = p_adj
    comp.loc[fam, "bh_significant"] = reject
    comp["verdict"] = np.where(
        comp["bh_significant"],
        "BH-significant improvement",
        np.where(comp["dm_t"] < -1.96, "significantly worse (nominal)", "n.s."),
    )
    full = comp[(comp["new"] == "P+F+M") & (comp["ref"] == "P")]
    conclusion: Conclusion = "supported" if bool(full["bh_significant"].any()) else "rejected"
    # fundamentals coverage (share of OOS rows with any non-missing F), by year
    has_f = df.loc[mask, FUND].notna().any(axis=1)
    coverage = (
        has_f.groupby(df.loc[mask, "session_date"].dt.year).mean().rename("share_rows_with_fundamentals")
    )
    preds = df.loc[mask, ["session_date", "symbol", "target", "excess_ret", "fwd_ret", "ew_ret"]].copy()
    preds["fold"], preds["baseline"] = fid, bb
    for (model, fs), p in probas.items():
        preds[f"p_{model}_{fs}"] = p[mask]
    imp_all = pd.concat(imps.values(), ignore_index=True) if imps else pd.DataFrame()
    imp_sum = (
        pd.concat([importance_summary(v).assign(feature_set=k) for k, v in imps.items()], ignore_index=True)
        if imps
        else pd.DataFrame()
    )
    res: dict[str, Any] = {
        "rows": len(df),
        "n_oos": int(mask.sum()),
        "n_folds": len(folds),
        "first_test": folds[0][2]["test_first_month"] if folds else None,
        "last_test": folds[-1][2]["test_last_month"] if folds else None,
        "macro_cols": macro_cols,
        "table": table,
        "comparisons": comp,
        "coverage": coverage,
        "conclusion": conclusion,
        "resources": probe.result,
        "base_rate_oos": float(yb.mean()),
        "importance": imp_sum,
        "runtime_s": round(time.perf_counter() - t0, 1),
    }
    if not log_tracker:
        return res
    tracker = Tracker(paths)
    rec = tracker.new(
        "h_mm_001_multimodal_ablation",
        "H-MM-001",
        "P+F+M has lower OOS log loss than P for the sign of 21-session excess return of US stocks at "
        "month ends (month-clustered DM, BH over 3 comparisons x 2 models).",
        datasets=[
            {
                "universe": "US stocks (30, config kind=stock)",
                "rows": len(df),
                "survivorship": SURVIVORSHIP,
                "manifests": data_versions(symbols, paths),
                "fundamentals": "data/validated/fundamentals/us_facts.parquet "
                "(first-filed, usable after filed)",
                "macro": "data/validated/macro/{market,bls}.parquet (as-of available_from)",
            }
        ],
        features=feature_columns("P+F+M", macro_cols),
        model={
            "models": MODELS,
            "logit": "L2 C=0.1, standardized",
            "xgb": "300 trees, depth 3, lr 0.03, subsample 0.8, colsample 0.8",
            "imputation": "median fitted on training fold + missing indicators (F and M)",
            "feature_sets": {fs: feature_columns(fs, macro_cols) for fs in SETS},
        },
        validation={
            "type": "monthly expanding walk-forward",
            "initial_months": INITIAL_MONTHS,
            "test_months": TEST_MONTHS,
            "purge_months": PURGE_MONTHS,
            "embargo_months": EMBARGO_MONTHS,
            "n_folds": len(folds),
            "first_test": res["first_test"],
            "last_test": res["last_test"],
            "inference": "row-weighted DM clustered by decision month (Newey-West over months)",
        },
        seed=SEED,
        metrics={
            "per_cell": table.to_dict(orient="records"),
            "comparisons": comp.drop(columns=[c for c in comp.columns if c.startswith("fold_")]).to_dict(
                orient="records"
            ),
            "base_rate_oos": res["base_rate_oos"],
            "resources": probe.result,
        },
        variants_tried=len(COMPARISONS) * len(MODELS),
        conclusion=conclusion,
        notes=SURVIVORSHIP
        + " Full history (2000+): SEC XBRL fundamentals start 2009-2011, so F is imputed (missing) in early "
        "folds; a descriptive 2011+ test subsample is reported (not part of the verdict). Permutation "
        "importance computed on test folds, reported only. BLS data: Source: U.S. Bureau of Labor "
        "Statistics, Public Data API v1, retrieved per bls.manifest.json. BLS.gov cannot vouch for the data "
        "or analyses derived from these data.",
        reproduce='uv run --no-sync python -c "from nebula.research.multimodal import run_multimodal; '
        'run_multimodal()"',
    )
    tracker.save_frame(rec, "cells", table)
    tracker.save_frame(rec, "comparisons", comp)
    tracker.save_frame(rec, "fold_deltas", pd.concat(disp, ignore_index=True))
    tracker.save_frame(rec, "folds", pd.concat(per_fold, ignore_index=True))
    tracker.save_frame(rec, "oos_predictions", preds.reset_index(drop=True))
    tracker.save_frame(rec, "fundamentals_coverage", coverage.reset_index())
    if not imp_all.empty:
        tracker.save_frame(rec, "xgb_permutation_importance_by_fold", imp_all)
        tracker.save_frame(rec, "xgb_permutation_importance", imp_sum)
    tracker.log(rec)
    res["experiment_id"] = rec.experiment_id
    return res
