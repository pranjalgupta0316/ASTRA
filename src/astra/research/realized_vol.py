"""Realized-volatility study: H-RV-001/002/003 (registered in docs/research/H-RV.md, incl. Amendment 1).

Pipeline per asset
1. Realized variance per UTC day from on-grid intraday closes (`realized_variance`): squared log returns
   between consecutive *observed* grid closes whose both endpoints lie in [day 00:00, day+1 00:00].
   Missing bars are never filled; a gap becomes one longer return. A day is valid only if it has at
   least `min_bars` on-grid bars.
2. HAR-RV (levels, OLS) with an expanding window, first fit after 730 usable training rows, re-fit every
   21 calendar days, parameters frozen in between (`har_forecast`). Weekly/monthly regressors are means
   over the valid days of the window. Negative forecasts are floored at the smallest positive valid
   training RV (counted).
3. Benchmarks on daily close-to-close returns (1d klines): EWMA(0.94), GARCH(1,1)-t, GJR-GARCH(1,1,1)-t
   (`astra.stats.volatility`); RV random walk and 22-day valid-mean of RV.
4. Evaluation on the common sample (all forecasts positive, target day valid): QLIKE (primary), MSE,
   Diebold-Mariano with Newey-West SE, Benjamini-Hochberg within each hypothesis family.

Convention: the forecast stored at origin t (row index = UTC day t) targets day t+1 and uses only data
up to the end of day t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from astra.common.config import get_config
from astra.common.logging import get_logger
from astra.common.paths import Paths, get_paths
from astra.common.seed import set_seed
from astra.data.storage import read_bars, read_manifest
from astra.stats.volatility import (
    diebold_mariano,
    ewma_forecast,
    garch_forecast,
    log_returns,
    mse_loss,
    qlike,
    qlike_rank,
)
from astra.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)

SEED = 7
LOCK_HINT = "v0.3.0-baseline"
MIN_TRAIN = 730
REFIT_EVERY = 21
EWMA_LAMBDA = 0.94
GARCH_DIST = "t"
BH_Q = 0.05
MIN_1M_BARS = 1368  # 95 % of 1440
MIN_1H_BARS = 22  # registered: >= 22 of 24 hourly bars
SCALE = 1e4  # decimal variance -> percent^2
NS_MIN = 60 * 1_000_000_000
NS_DAY = 1440 * NS_MIN
RV_ASSETS = ["BTCUSDT", "ETHUSDT"]
MODELS_1M = ["har", "ewma", "garch", "gjr", "rw", "mean22"]
REPRODUCE = "uv run astra --lock v0.3.0-baseline rv-study"


# ----------------------------------------------------------------------------------------------
# Realized variance
# ----------------------------------------------------------------------------------------------
def _ns(ts: pd.Series) -> np.ndarray:
    t = pd.to_datetime(ts, utc=True).dt.tz_convert(None)
    return t.to_numpy().astype("datetime64[ns]").astype(np.int64)


def realized_variance(
    bars: pd.DataFrame, bar_minutes: int, sample_minutes: int, min_bars: int
) -> pd.DataFrame:
    """Daily realized variance from intraday bars (columns `open_time_utc`, `close`).

    Returns a frame indexed by UTC day (tz-naive midnight) over the full calendar range with columns
    rv (sum of squared log returns, NaN if none), n_bars (on-grid bars opened that day), n_returns,
    max_span_min (longest return interval) and valid (n_bars >= min_bars and n_returns > 0).
    """
    t_open = _ns(bars["open_time_utc"])
    step = bar_minutes * NS_MIN
    on = t_open % step == 0  # off-grid bars excluded
    t_open = t_open[on]
    close = bars["close"].to_numpy(dtype=float)[on]
    order = np.argsort(t_open, kind="stable")
    t_open, close = t_open[order], close[order]
    if len(t_open) and np.any(np.diff(t_open) == 0):
        raise ValueError("duplicate bar timestamps")
    t_end = t_open + step
    day_bar = t_open // NS_DAY * NS_DAY
    first_day, last_day = int(day_bar.min()), int(day_bar.max())
    n_days = (last_day - first_day) // NS_DAY + 1  # integer count (np.arange on ns ints loses precision)
    days = pd.DatetimeIndex(pd.to_datetime(first_day + NS_DAY * np.arange(n_days, dtype=np.int64), unit="ns"))
    n_bars = np.bincount((day_bar - first_day) // NS_DAY, minlength=n_days)

    grid = t_end % (sample_minutes * NS_MIN) == 0
    tg, lp = t_end[grid], np.log(close[grid])
    ta, tb = tg[:-1], tg[1:]
    r = np.diff(lp)
    day_b = (tb - 1) // NS_DAY * NS_DAY  # the day whose (00:00, 24:00] contains the end point
    keep = (ta >= day_b) & (day_b >= first_day) & (day_b <= last_day)
    pos = ((day_b[keep] - first_day) // NS_DAY).astype(np.int64)
    rv = np.bincount(pos, weights=r[keep] ** 2, minlength=n_days)
    n_ret = np.bincount(pos, minlength=n_days)
    span = np.zeros(n_days)
    np.maximum.at(span, pos, (tb[keep] - ta[keep]) / NS_MIN)
    out = pd.DataFrame(
        {
            "rv": np.where(n_ret > 0, rv, np.nan),
            "n_bars": n_bars,
            "n_returns": n_ret,
            "max_span_min": span,
        },
        index=days,
    )
    out["valid"] = (out["n_bars"] >= min_bars) & (out["n_returns"] > 0)
    out.index.name = "day"
    return out


# ----------------------------------------------------------------------------------------------
# HAR-RV
# ----------------------------------------------------------------------------------------------
def har_design(rv: pd.Series, valid: pd.Series) -> pd.DataFrame:
    """HAR regressors at each origin t: d = RV_t, w / m = mean of valid RV over t-4..t / t-21..t.

    NaN where RV_t is invalid or fewer than 22 calendar days of history exist.
    """
    x = pd.Series(rv, dtype=float).where(pd.Series(valid, dtype=bool))
    d = x
    w = x.rolling(5, min_periods=1).mean()
    m = x.rolling(22, min_periods=1).mean()
    X = pd.DataFrame({"d": d, "w": w, "m": m}, index=x.index)
    X.loc[d.isna()] = np.nan
    X.iloc[:21] = np.nan
    return X


def fit_har(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS of y on [1, X]; returns (b0, b_d, b_w, b_m)."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta


@dataclass
class HarForecast:
    forecast: pd.Series  # floored forecast at each origin (NaN where unavailable)
    raw: pd.Series  # unfloored forecast
    refits: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def har_forecast(
    rv: pd.Series, valid: pd.Series, refit_every: int = REFIT_EVERY, min_train: int = MIN_TRAIN
) -> HarForecast:
    """Expanding-window HAR one-step forecasts (index = origin day t, value = forecast of RV_{t+1}).

    Training row s (regressors at s, target RV_{s+1}) is usable at refit origin o iff s + 1 <= o and
    both are defined/valid. First refit at the first o with >= min_train usable rows; then every
    `refit_every` positions; params frozen in between.
    """
    rv = pd.Series(rv, dtype=float)
    valid = pd.Series(valid, dtype=bool).reindex(rv.index).fillna(False).astype(bool)
    X = har_design(rv, valid)
    Xa = X.to_numpy()
    y_next = rv.where(valid).shift(-1).to_numpy()
    row_ok = np.isfinite(Xa).all(axis=1) & np.isfinite(y_next)
    n = len(rv)
    # usable rows at origin o: s <= o - 1
    cum = np.concatenate([[0], np.cumsum(row_ok)])  # cum[o] = #usable rows with s < o
    raw = np.full(n, np.nan)
    fc = np.full(n, np.nan)
    rows: list[dict[str, Any]] = []
    starts = np.flatnonzero(cum[:n] >= min_train)
    n_floored = 0
    if len(starts):
        rv_valid = rv.where(valid).to_numpy()
        for o in range(int(starts[0]), n, refit_every):
            tr = np.flatnonzero(row_ok[:o])
            beta = fit_har(Xa[tr], y_next[tr])
            hist = rv_valid[: o + 1]
            floor = float(np.nanmin(hist[hist > 0]))
            end = min(o + refit_every, n)
            f = beta[0] + Xa[o:end] @ beta[1:]
            raw[o:end] = f
            neg = np.isfinite(f) & (f <= 0)
            n_floored += int(neg.sum())
            fc[o:end] = np.where(neg, floor, f)
            rows.append(
                {
                    "origin": rv.index[o],
                    "n_train": len(tr),
                    "b0": beta[0],
                    "b_d": beta[1],
                    "b_w": beta[2],
                    "b_m": beta[3],
                    "floor": floor,
                    "n_floored_in_block": int(neg.sum()),
                }
            )
    refits = pd.DataFrame(rows)
    meta = {
        "min_train": min_train,
        "refit_every": refit_every,
        "n_refits": len(rows),
        "first_origin": str(rows[0]["origin"].date()) if rows else None,
        "n_floored": n_floored,
    }
    return HarForecast(
        pd.Series(fc, index=rv.index, name="har"), pd.Series(raw, index=rv.index), refits, meta
    )


# ----------------------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------------------
def daily_returns(symbol: str, paths: Paths) -> pd.Series:
    """Close-to-close log returns from validated 1d klines, indexed by UTC day (tz-naive)."""
    d = read_bars(symbol, "CRYPTO", paths, freq="1d")
    t = pd.to_datetime(d["open_time_utc"], utc=True).dt.tz_convert(None)
    if ((d["high"] == d["low"]) & (d["volume"] == 0)).any():
        raise ValueError(f"{symbol}: placeholder daily bars present")
    c = pd.Series(d["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(t)).sort_index()
    if (c.index.to_series().diff().dropna() != pd.Timedelta("1D")).any():
        raise ValueError(f"{symbol}: gaps in daily bars")
    return log_returns(c)


def benchmark_forecasts(r: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    fc = pd.DataFrame(index=r.index)
    fc["ewma"] = ewma_forecast(r, EWMA_LAMBDA)
    g = garch_forecast(r, "garch", REFIT_EVERY, MIN_TRAIN, GARCH_DIST)
    j = garch_forecast(r, "gjr", REFIT_EVERY, MIN_TRAIN, GARCH_DIST)
    fc["garch"], fc["gjr"] = g.forecast, j.forecast
    return fc, {"garch": g, "gjr": j}


def _rv_block(rvf: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, HarForecast]:
    h = har_forecast(rvf["rv"], rvf["valid"])
    X = har_design(rvf["rv"], rvf["valid"])
    out = pd.DataFrame(index=rvf.index)
    out[f"{prefix}"] = rvf["rv"]
    out[f"{prefix}_valid"] = rvf["valid"]
    out[f"{prefix}_next"] = rvf["rv"].shift(-1)
    out[f"{prefix}_valid_next"] = rvf["valid"].shift(-1, fill_value=False).astype(bool)
    out[f"har_{prefix}"] = h.forecast
    out[f"har_raw_{prefix}"] = h.raw
    out[f"rw_{prefix}"] = X["d"]
    out[f"mean22_{prefix}"] = X["m"]
    return out, h


def build_asset_frame(symbol: str, paths: Paths) -> dict[str, Any]:
    """1-minute assets: RV5 + RV1, HAR on each, daily benchmarks. Origin-indexed frame."""
    bars = read_bars(symbol, "CRYPTO", paths, freq="1m")[["open_time_utc", "close", "high", "low", "volume"]]
    n_rows = len(bars)
    t = pd.to_datetime(bars["open_time_utc"], utc=True)
    n_offgrid = int((t != t.dt.floor("1min")).sum())
    n_flat0 = int(((bars["high"] == bars["low"]) & (bars["volume"] == 0) & (t == t.dt.floor("1min"))).sum())
    rv5 = realized_variance(bars, 1, 5, MIN_1M_BARS)
    rv1 = realized_variance(bars, 1, 1, MIN_1M_BARS)
    del bars
    r = daily_returns(symbol, paths)
    idx = r.index.union(rv5.index)
    if not (idx.to_series().diff().dropna() == pd.Timedelta("1D")).all():
        raise ValueError("calendar index not contiguous")
    rv5 = rv5.reindex(idx)
    rv5["valid"] = rv5["valid"].fillna(False).astype(bool)
    rv1 = rv1.reindex(idx)
    rv1["valid"] = rv1["valid"].fillna(False).astype(bool)
    b5, h5 = _rv_block(rv5, "rv5")
    b1, h1 = _rv_block(rv1, "rv1")
    bench, fits = benchmark_forecasts(r.reindex(idx))
    frame = pd.concat([b5, b1, bench], axis=1)
    frame["r"] = r.reindex(idx)
    frame["n_bars_1m"] = rv5["n_bars"].fillna(0).astype(int)
    frame["n_returns_5m"] = rv5["n_returns"].fillna(0).astype(int)
    frame["max_span_5m"] = rv5["max_span_min"]
    frame.index.name = "origin_day"
    v = rv5["valid"]
    validity = {
        "n_1m_rows": n_rows,
        "n_offgrid_excluded": n_offgrid,
        "n_flat_zero_volume_minutes_kept": n_flat0,
        "calendar_days": len(idx),
        "valid_days": int(v.sum()),
        "invalid_days": int((~v).sum()),
        "days_without_1m_bars": int((rv5["n_bars"].fillna(0) == 0).sum()),
        "invalid_day_list": [str(d.date()) for d in idx[~v.to_numpy()]],
        "valid_days_complete_288": int((v & (rv5["n_returns"] == 288)).sum()),
        "valid_days_with_gap_returns": int((v & (rv5["max_span_min"] > 5)).sum()),
    }
    return {
        "frame": frame,
        "har5": h5,
        "har1": h1,
        "garch_fits": fits,
        "validity": validity,
        "manifests": [read_manifest(symbol, "CRYPTO", paths, f) for f in ("1m", "1d")],
    }


def build_hourly_frame(symbol: str, paths: Paths) -> dict[str, Any]:
    bars = read_bars(symbol, "CRYPTO", paths, freq="1h")[["open_time_utc", "close"]]
    t = pd.to_datetime(bars["open_time_utc"], utc=True)
    n_offgrid = int((t != t.dt.floor("1h")).sum())
    rvh = realized_variance(bars, 60, 60, MIN_1H_BARS)
    r = daily_returns(symbol, paths)
    idx = r.index.union(rvh.index)
    rvh = rvh.reindex(idx)
    rvh["valid"] = rvh["valid"].fillna(False).astype(bool)
    b, h = _rv_block(rvh, "rvh")
    frame = b.copy()
    frame["ewma"] = ewma_forecast(r.reindex(idx), EWMA_LAMBDA)
    frame["r"] = r.reindex(idx)
    frame.index.name = "origin_day"
    v = rvh["valid"]
    validity = {
        "n_1h_rows": len(bars),
        "n_offgrid_excluded": n_offgrid,
        "calendar_days": len(idx),
        "valid_days": int(v.sum()),
        "invalid_days": int((~v).sum()),
    }
    return {
        "frame": frame,
        "har": h,
        "validity": validity,
        "manifests": [read_manifest(symbol, "CRYPTO", paths, f) for f in ("1h", "1d")],
    }


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def eval_mask(frame: pd.DataFrame, models: list[str], target: str) -> np.ndarray:
    f = frame[models].to_numpy(dtype=float)
    ok = np.isfinite(f).all(axis=1) & (f > 0).all(axis=1)
    ok &= frame[f"{target}_valid_next"].to_numpy(dtype=bool)
    ok &= np.isfinite(frame[f"{target}_next"].to_numpy(dtype=float))
    return ok


def model_losses(y: np.ndarray, h: np.ndarray) -> dict[str, np.ndarray]:
    """Losses in percent^2 units. QLIKE in ranking form for DM when some y == 0 (same differences)."""
    y, h = y * SCALE, h * SCALE
    return {"qlike": qlike(y, h) if (y > 0).all() else qlike_rank(y, h), "mse": mse_loss(y, h)}


def dm_better(loss_model: np.ndarray, loss_bench: np.ndarray) -> dict[str, float]:
    """DM on mean(loss_bench - loss_model): positive t = model better."""
    return diebold_mariano(loss_bench, loss_model)


def mincer_zarnowitz(y: np.ndarray, h: np.ndarray) -> dict[str, float]:
    y, h = y * SCALE, h * SCALE
    A = np.column_stack([np.ones(len(h)), h])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1.0 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return {"intercept_pct2": float(beta[0]), "slope": float(beta[1]), "r2": r2}


def evaluate_models(
    frame: pd.DataFrame, models: list[str], target: str, regime_col: str | None = None
) -> dict[str, Any]:
    """Common-sample losses, MZ regressions and (optional) tercile regime table for `models`."""
    ok = eval_mask(frame, models, target)
    ev = frame.loc[ok]
    y = ev[f"{target}_next"].to_numpy(dtype=float)
    losses = {m: model_losses(y, ev[m].to_numpy(dtype=float)) for m in models}
    table = {
        m: {
            "qlike_mean": float(np.mean(losses[m]["qlike"])),
            "mse_mean_pct4": float(np.mean(losses[m]["mse"])),
            "mean_forecast_pct2": float(ev[m].mean() * SCALE),
            **{f"mz_{k}": v for k, v in mincer_zarnowitz(y, ev[m].to_numpy(dtype=float)).items()},
        }
        for m in models
    }
    out: dict[str, Any] = {
        "n_eval": int(ok.sum()),
        "n_zero_target": int((y <= 0).sum()),
        "eval_start": str(ev.index[0].date()) if len(ev) else None,
        "eval_end": str(ev.index[-1].date()) if len(ev) else None,
        "mean_target_pct2": float(y.mean() * SCALE) if len(y) else float("nan"),
        "models": table,
        "losses": losses,
        "mask": ok,
    }
    if regime_col is not None and len(ev):
        z = ev[regime_col].to_numpy(dtype=float)
        q = np.quantile(z, [1 / 3, 2 / 3])
        lab = np.where(z <= q[0], "low", np.where(z <= q[1], "mid", "high"))
        reg = {}
        for g in ("low", "mid", "high"):
            sel = lab == g
            reg[g] = {"n": int(sel.sum()), **{m: float(np.mean(losses[m]["qlike"][sel])) for m in models}}
        out["regimes_qlike"] = reg
        out["regime_cuts_pct2"] = [float(c * SCALE) for c in q]
    return out


def cell_verdict(reject: bool, t: float) -> Conclusion:
    if reject and t > 0:
        return "supported"
    if reject and t < 0:
        return "rejected"
    return "inconclusive"


def _bh(pvals: list[float]) -> tuple[np.ndarray, np.ndarray]:
    reject, p_bh, *_ = multipletests(np.asarray(pvals, dtype=float), alpha=BH_Q, method="fdr_bh")
    return np.asarray(reject, dtype=bool), np.asarray(p_bh, dtype=float)


# ----------------------------------------------------------------------------------------------
# Study runner
# ----------------------------------------------------------------------------------------------
def _cell_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reject, p_bh = _bh([c["dm_qlike"]["p_two_sided"] for c in cells])
    for k, c in enumerate(cells):
        c["p_bh"] = float(p_bh[k])
        c["bh_reject"] = bool(reject[k])
        c["conclusion"] = cell_verdict(bool(reject[k]), c["dm_qlike"]["dm_t"])
    return cells


def _log_cell(tracker: Tracker, hid: str, statement: str, c: dict[str, Any], extra: dict[str, Any]) -> str:
    rec = tracker.new(
        f"rv_{c['model']}_vs_{c['bench']}_qlike",
        hid,
        statement,
        datasets=extra["manifests"],
        features=extra["features"],
        model=extra["model"],
        validation={
            "type": "expanding walk-forward, one-step-ahead (day t+1 from data <= end of day t)",
            "min_train_valid_days": MIN_TRAIN,
            "refit_every_days": REFIT_EVERY,
            "eval_start": c["eval_start"],
            "eval_end": c["eval_end"],
            "target": c["target"],
            "test_touches": 1,
            "decision": f"BH(q={BH_Q}) within {hid} on two-sided DM p of QLIKE; DM t>0 = {c['model']} better",
            "family_size": extra["family_size"],
            "registration": "docs/research/H-RV.md (incl. Amendment 1)",
        },
        seed=SEED,
        metrics={
            "symbol": c["symbol"],
            "n_eval": c["n_eval"],
            f"qlike_{c['model']}": c["qlike_model"],
            f"qlike_{c['bench']}": c["qlike_bench"],
            f"mse_{c['model']}_pct4": c["mse_model"],
            f"mse_{c['bench']}_pct4": c["mse_bench"],
            "dm_t_qlike": c["dm_qlike"]["dm_t"],
            "dm_p_qlike": c["dm_qlike"]["p_two_sided"],
            "dm_p_bh": c["p_bh"],
            "dm_t_mse": c["dm_mse"]["dm_t"],
            "dm_p_mse": c["dm_mse"]["p_two_sided"],
            **extra.get("metrics", {}),
        },
        conclusion=c["conclusion"],
        variants_tried=1,
        notes=extra.get("notes", ""),
        reproduce=REPRODUCE,
    )
    if "frame" in extra:
        tracker.save_frame(rec, "forecasts", extra["frame"].reset_index())
    for name, df in extra.get("frames", {}).items():
        if len(df):
            tracker.save_frame(rec, name, df)
    tracker.log(rec)
    return rec.experiment_id


def _make_cell(symbol: str, model: str, bench: str, target: str, ev: dict[str, Any]) -> dict[str, Any]:
    L = ev["losses"]
    return {
        "symbol": symbol,
        "model": model,
        "bench": bench,
        "target": target,
        "n_eval": ev["n_eval"],
        "eval_start": ev["eval_start"],
        "eval_end": ev["eval_end"],
        "qlike_model": ev["models"][model]["qlike_mean"],
        "qlike_bench": ev["models"][bench]["qlike_mean"],
        "mse_model": ev["models"][model]["mse_mean_pct4"],
        "mse_bench": ev["models"][bench]["mse_mean_pct4"],
        "dm_qlike": dm_better(L[model]["qlike"], L[bench]["qlike"]),
        "dm_mse": dm_better(L[model]["mse"], L[bench]["mse"]),
    }


def _table(cells: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hypothesis": c.get("hid"),
                "symbol": c["symbol"],
                "cell": f"{c['model']} vs {c['bench']}",
                "n": c["n_eval"],
                "qlike_model": c["qlike_model"],
                "qlike_bench": c["qlike_bench"],
                "mse_model": c["mse_model"],
                "mse_bench": c["mse_bench"],
                "dm_t_qlike": c["dm_qlike"]["dm_t"],
                "p": c["dm_qlike"]["p_two_sided"],
                "p_bh": c["p_bh"],
                "dm_t_mse": c["dm_mse"]["dm_t"],
                "conclusion": c["conclusion"],
                "experiment_id": c.get("experiment_id"),
            }
            for c in cells
        ]
    )


def _family_record(
    tracker: Tracker,
    hid: str,
    statement: str,
    verdict: Conclusion,
    table: pd.DataFrame,
    metrics: dict[str, Any],
    manifests: list[dict[str, Any]],
    frames: dict[str, pd.DataFrame] | None = None,
) -> str:
    rec = tracker.new(
        "rv_family_summary",
        hid,
        statement,
        datasets=manifests,
        validation={"family_cells": len(table), "bh_q": BH_Q, "registration": "docs/research/H-RV.md"},
        seed=SEED,
        metrics={
            "n_cells": len(table),
            "counts": table["conclusion"].value_counts().to_dict(),
            "cell_experiment_ids": table["experiment_id"].tolist(),
            **metrics,
        },
        conclusion=verdict,
        variants_tried=len(table),
        notes="One pre-registered specification per cell; variants_tried = number of cells in the family.",
        reproduce=REPRODUCE,
    )
    tracker.save_frame(rec, "cells", table)
    for name, df in (frames or {}).items():
        tracker.save_frame(rec, name, df)
    tracker.log(rec)
    return rec.experiment_id


def run_rv_study(paths: Paths | None = None, hourly_symbols: list[str] | None = None) -> dict[str, Any]:
    """Run H-RV-001, H-RV-002, H-RV-003 and the descriptive tables; log Tracker records."""
    set_seed(SEED)
    paths = paths or get_paths()
    tracker = Tracker(paths)
    out: dict[str, Any] = {}

    # ---------------- 1-minute assets (H-RV-001, H-RV-003, descriptive) ----------------
    assets: dict[str, dict[str, Any]] = {}
    for sym in RV_ASSETS:
        a = build_asset_frame(sym, paths)
        fr = a["frame"]
        fr["har"], fr["rw"], fr["mean22"] = fr["har_rv5"], fr["rw_rv5"], fr["mean22_rv5"]
        a["ev5"] = evaluate_models(fr, MODELS_1M, "rv5", regime_col="mean22_rv5")
        fr["har1"], fr["rw1"], fr["mean22_1"] = fr["har_rv1"], fr["rw_rv1"], fr["mean22_rv1"]
        a["ev1"] = evaluate_models(fr, ["har1", "ewma", "garch", "gjr", "rw1", "mean22_1"], "rv1")
        assets[sym] = a
        log.info(
            "rv.asset", symbol=sym, n_eval=a["ev5"]["n_eval"], **{"floored": a["har5"].meta["n_floored"]}
        )

    man_all = [m for a in assets.values() for m in a["manifests"]]
    model_desc = {
        "har": "RV_{t+1} = b0 + b_d RV_t + b_w mean5(valid RV) + b_m mean22(valid RV), OLS levels",
        "ewma_lambda": EWMA_LAMBDA,
        "garch": {"mean": "Zero", "dist": GARCH_DIST, "scale": "percent", "refit_every": REFIT_EVERY},
        "rv": "RV5 = sum sq. 5-min log returns of 1m closes between consecutive observed grid points",
        "valid_day": f">= {MIN_1M_BARS} of 1440 on-grid 1m bars",
    }

    # H-RV-001
    c001: list[dict[str, Any]] = []
    for sym in RV_ASSETS:
        for bench in ("ewma", "garch", "gjr"):
            c = _make_cell(sym, "har", bench, "rv5", assets[sym]["ev5"])
            c["hid"] = "H-RV-001"
            c001.append(c)
    _cell_rows(c001)
    st001 = "HAR-RV one-step forecasts of RV5_{t+1} have lower QLIKE than EWMA / GARCH(1,1)-t / GJR-t."
    for c in c001:
        a = assets[c["symbol"]]
        c["experiment_id"] = _log_cell(
            tracker,
            "H-RV-001",
            st001,
            c,
            {
                "manifests": a["manifests"],
                "features": ["rv5", "log_return_cc_1d"],
                "model": model_desc,
                "family_size": len(c001),
                "metrics": {
                    "har_n_floored": a["har5"].meta["n_floored"],
                    "har_n_refits": a["har5"].meta["n_refits"],
                    "garch_failed_refits": a["garch_fits"]["garch"].meta["n_failed_or_nonconverged"],
                    "gjr_failed_refits": a["garch_fits"]["gjr"].meta["n_failed_or_nonconverged"],
                    **{f"validity_{k}": v for k, v in a["validity"].items() if not isinstance(v, list)},
                },
                "notes": f"symbol={c['symbol']}; invalid days: {a['validity']['invalid_day_list']}",
                "frame": a["frame"].loc[a["ev5"]["mask"]],
                "frames": {"har_refits": a["har5"].refits},
            },
        )
    t001 = _table(c001)
    key = t001[t001["cell"].isin(["har vs ewma", "har vs garch"])]
    if (key["conclusion"] == "supported").all():
        v001: Conclusion = "supported"
    elif (key["conclusion"] == "rejected").all():
        v001 = "rejected"
    else:
        v001 = "inconclusive"

    # Descriptive tables
    desc_rows, mz_rows, reg_rows, rv1_rows = [], [], [], []
    for sym, a in assets.items():
        for m, d in a["ev5"]["models"].items():
            mz_rows.append({"symbol": sym, "model": m, **d})
        for g, d in a["ev5"]["regimes_qlike"].items():
            reg_rows.append({"symbol": sym, "regime": g, **d})
        L1 = a["ev1"]["losses"]
        for m, d in a["ev1"]["models"].items():
            row = {"symbol": sym, "model": m, "n_eval": a["ev1"]["n_eval"], **d}
            if m != "har1":
                row["dm_t_har1_better_qlike"] = dm_better(L1["har1"]["qlike"], L1[m]["qlike"])["dm_t"]
            rv1_rows.append(row)
        desc_rows.append(
            {
                "symbol": sym,
                **{k: v for k, v in a["validity"].items() if not isinstance(v, list)},
                "n_eval_rv5": a["ev5"]["n_eval"],
                "eval_start": a["ev5"]["eval_start"],
                "eval_end": a["ev5"]["eval_end"],
                "har5_n_floored": a["har5"].meta["n_floored"],
                "har5_first_origin": a["har5"].meta["first_origin"],
                "har5_n_refits": a["har5"].meta["n_refits"],
                "har1_n_floored": a["har1"].meta["n_floored"],
                "garch_failed_refits": a["garch_fits"]["garch"].meta["n_failed_or_nonconverged"],
                "gjr_failed_refits": a["garch_fits"]["gjr"].meta["n_failed_or_nonconverged"],
                "regime_cuts_pct2": a["ev5"]["regime_cuts_pct2"],
            }
        )
    mz, reg, rv1t, desc = (pd.DataFrame(x) for x in (mz_rows, reg_rows, rv1_rows, desc_rows))
    fam001 = _family_record(
        tracker,
        "H-RV-001",
        st001 + " Family: HAR significantly better than EWMA and GARCH(1,1) on both assets.",
        v001,
        t001,
        {"family_verdict": v001},
        man_all,
        {"mincer_zarnowitz": mz, "regimes_qlike": reg, "rv1_robustness": rv1t, "data_validity": desc},
    )

    # H-RV-003
    c003 = []
    for sym in RV_ASSETS:
        c = _make_cell(sym, "gjr", "ewma", "rv5", assets[sym]["ev5"])
        c["hid"] = "H-RV-003"
        c003.append(c)
    _cell_rows(c003)
    st003 = "GJR-GARCH(1,1,1)-t has lower QLIKE than EWMA(0.94) when evaluated against RV5_{t+1}."
    for c in c003:
        a = assets[c["symbol"]]
        c["experiment_id"] = _log_cell(
            tracker,
            "H-RV-003",
            st003,
            c,
            {
                "manifests": a["manifests"],
                "features": ["log_return_cc_1d"],
                "model": model_desc,
                "family_size": len(c003),
                "notes": f"symbol={c['symbol']}; H-VOL-001 re-test (RV5 proxy); originals untouched",
                "frames": {"gjr_refits": a["garch_fits"]["gjr"].refits},
            },
        )
    t003 = _table(c003)
    concl003 = set(t003["conclusion"])
    v003: Conclusion = (
        "supported"
        if concl003 == {"supported"}
        else "rejected"
        if concl003 == {"rejected"}
        else "inconclusive"
    )
    fam003 = _family_record(tracker, "H-RV-003", st003, v003, t003, {"family_verdict": v003}, man_all)

    # ---------------- H-RV-002 hourly ----------------
    syms = hourly_symbols or get_config().symbols(market="CRYPTO")
    hourly: dict[str, dict[str, Any]] = {}
    c002 = []
    for sym in syms:
        h = build_hourly_frame(sym, paths)
        fr = h["frame"]
        fr["har"] = fr["har_rvh"]
        h["ev"] = evaluate_models(fr, ["har", "ewma"], "rvh")
        hourly[sym] = h
        c = _make_cell(sym, "har", "ewma", "rvh", h["ev"])
        c["hid"] = "H-RV-002"
        c002.append(c)
    _cell_rows(c002)
    st002 = (
        "HAR on daily RV from hourly returns has lower QLIKE than EWMA(0.94) of daily returns (target RV_h)."
    )
    for c in c002:
        h = hourly[c["symbol"]]
        c["experiment_id"] = _log_cell(
            tracker,
            "H-RV-002",
            st002,
            c,
            {
                "manifests": h["manifests"],
                "features": ["rv_hourly", "log_return_cc_1d"],
                "model": {**model_desc, "valid_day": f">= {MIN_1H_BARS} of 24 on-grid 1h bars"},
                "family_size": len(c002),
                "metrics": {
                    "har_n_floored": h["har"].meta["n_floored"],
                    **{f"validity_{k}": v for k, v in h["validity"].items()},
                    **{
                        f"mz_{m}_{k}": v
                        for m in ("har", "ewma")
                        for k, v in h["ev"]["models"][m].items()
                        if k.startswith("mz_")
                    },
                },
                "notes": f"symbol={c['symbol']}",
                "frames": {"har_refits": h["har"].refits},
            },
        )
    t002 = _table(c002)
    n_better = int((t002["conclusion"] == "supported").sum())
    n_worse = int((t002["conclusion"] == "rejected").sum())
    v002: Conclusion = "supported" if n_better >= 12 else "rejected" if n_worse >= 12 else "inconclusive"
    hdesc = pd.DataFrame(
        [
            {
                "symbol": s,
                **h["validity"],
                "har_n_floored": h["har"].meta["n_floored"],
                "har_first_origin": h["har"].meta["first_origin"],
                **{f"{m}_{k}": v for m in ("har", "ewma") for k, v in h["ev"]["models"][m].items()},
            }
            for s, h in hourly.items()
        ]
    )
    fam002 = _family_record(
        tracker,
        "H-RV-002",
        st002 + " Family: >= 12 of 16 cells BH-significant improvements.",
        v002,
        t002,
        {"family_verdict": v002, "n_supported": n_better, "n_rejected": n_worse},
        [m for h in hourly.values() for m in h["manifests"]],
        {"hourly_descriptive": hdesc},
    )

    out["H-RV-001"] = {"table": t001, "verdict": v001, "family_id": fam001}
    out["H-RV-002"] = {"table": t002, "verdict": v002, "family_id": fam002}
    out["H-RV-003"] = {"table": t003, "verdict": v003, "family_id": fam003}
    out["descriptive"] = {
        "mincer_zarnowitz": mz,
        "regimes_qlike": reg,
        "rv1_robustness": rv1t,
        "data_validity": desc,
        "hourly": hdesc,
    }
    return out
