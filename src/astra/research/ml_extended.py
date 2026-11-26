"""H-ML-003 (pooled daily panel over the full extended universe) and H-ML-004 (hourly crypto panel).

Registered in docs/research/H-EXT.md (2026-11-25) before execution. Models and hyperparameters are the
fixed H-ML-001/002 ones (`astra.research.ml_study.model_factory`); nothing is tuned.

Implementation choices fixed before the first run (documented in docs/research/results/H-ML-EXT.md):
- Entry rule: an instrument's rows are eligible only from its 253rd post-placeholder bar on (252 bars of
  history), per the H-EXT common rules. For 1h bars the same rule is applied as 252 x 24 bars (252 days).
- Folds are CALENDAR-time folds (`calendar_folds`): initial training span, test span, and a purge gap
  measured in wall-clock time. With full history (2000+) the union of dates is not every calendar day
  before crypto starts (2017), so union-date counting (`date_folds`) would not give 2 y / 1 quarter.
- DL early-stopping purge: the TorchClassifier validation tail is separated from the fit rows by
  3 x (max rows per date) rows, which guarantees >= 2 full dates between them in the pooled panel
  (H-ML-002 used a 2-ROW gap, which in a pooled panel is less than one date). Only early stopping is
  affected; test rows are untouched.
- Inference: row-weighted, date-clustered DM (`ml_study._date_dm`, amendment A2) vs the ex-ante
  training-fold base rate, and DL vs logit; one-sided p-values, Benjamini-Hochberg over the models.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
import tracemalloc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from astra.common.config import get_config
from astra.common.logging import get_logger
from astra.common.paths import Paths, get_paths
from astra.common.seed import set_seed
from astra.data.lock import resolved_exists
from astra.data.storage import read_manifest, validated_path
from astra.features.technical import FEATURE_COLUMNS
from astra.ml.classical import _pointwise_log_loss
from astra.ml.datasets import lag_columns, lagged_matrix, load_features
from astra.research.ml_study import SEED, SEQ_LEN, _date_dm, _one_sided_p, model_factory
from astra.stats.inference import newey_west_tstat
from astra.tracking.tracker import Conclusion, Tracker

log = get_logger(__name__)

MIN_HISTORY = 252
PANEL_MODELS = ["logit", "xgb", "mlp", "gru"]
HOURLY_MODELS = ["logit", "xgb", "mlp"]
SURVIVORSHIP = (
    "Extended universe = TODAY's large caps / NIFTY names / crypto majors -> survivorship-biased "
    "(H-EXT common rules)."
)
CRYPTO_ROUND_TRIP_BPS = 26.0


# ------------------------------------------------------------------------------ resource probing


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def working_set_bytes() -> tuple[int, int]:
    """(current, lifetime-peak) process working set / RSS in bytes; (0, 0) if unavailable."""
    if sys.platform == "win32":
        k32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_PMC), ctypes.c_ulong]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return int(pmc.WorkingSetSize), int(pmc.PeakWorkingSetSize)
        return 0, 0
    try:
        status = open("/proc/self/status", encoding="ascii").read()  # noqa: SIM115
        kv = {ln.split(":")[0]: ln.split()[1] for ln in status.splitlines() if ln.startswith("Vm")}
        return int(kv.get("VmRSS", 0)) * 1024, int(kv.get("VmHWM", 0)) * 1024
    except OSError:
        return 0, 0


class ResourceProbe:
    """Wall time, tracemalloc peak (Python-heap allocations incl. numpy), sampled peak process working
    set (0.2 s polling) and peak CUDA memory for one block of work."""

    def __init__(self, interval: float = 0.2) -> None:
        self.interval = interval
        self.result: dict[str, Any] = {}

    def __enter__(self) -> ResourceProbe:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        tracemalloc.reset_peak()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except ImportError:  # pragma: no cover - torch is installed in this project
            pass
        self._peak_ws = working_set_bytes()[0]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def _poll(self) -> None:
        while not self._stop.wait(self.interval):
            self._peak_ws = max(self._peak_ws, working_set_bytes()[0])

    def snapshot(self) -> dict[str, Any]:
        """Peaks so far (used to checkpoint resource usage of an interrupted run)."""
        _, heap_peak = tracemalloc.get_traced_memory()
        snap: dict[str, Any] = {
            "peak_tracemalloc_mb": round(heap_peak / 1e6, 1),
            "peak_working_set_mb_sampled": round(max(self._peak_ws, working_set_bytes()[0]) / 1e6, 1),
            "peak_vram_allocated_mb": 0.0,
            "peak_vram_reserved_mb": 0.0,
        }
        try:
            import torch

            if torch.cuda.is_available():
                snap["peak_vram_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
                snap["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)
        except ImportError:  # pragma: no cover
            pass
        return snap

    def __exit__(self, *exc: object) -> None:
        secs = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        cur, lifetime_peak = working_set_bytes()
        _, heap_peak = tracemalloc.get_traced_memory()
        gpu: dict[str, Any] = {"cuda": False}
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu = {
                    "cuda": True,
                    "device": torch.cuda.get_device_name(0),
                    "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1),
                    "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 1),
                }
        except ImportError:  # pragma: no cover
            pass
        self.result = {
            "runtime_s": round(secs, 1),
            "peak_tracemalloc_mb": round(heap_peak / 1e6, 1),
            "peak_working_set_mb_sampled": round(max(self._peak_ws, cur) / 1e6, 1),
            "process_lifetime_peak_working_set_mb": round(lifetime_peak / 1e6, 1),
            "gpu": gpu,
        }


# ------------------------------------------------------------------------------ data


def data_versions(symbols: list[str], paths: Paths, freq: str = "1d") -> list[dict[str, Any]]:
    cfg = get_config()
    out = []
    for s in symbols:
        mkt = cfg.instrument(s).market
        try:
            man = read_manifest(s, mkt, paths, freq)
            out.append(
                {
                    "symbol": s,
                    "market": mkt,
                    "freq": freq,
                    "sha256": man.get("sha256"),
                    "rows": man.get("rows"),
                    "last_session": man.get("last_session"),
                }
            )
        except FileNotFoundError:
            out.append({"symbol": s, "market": mkt, "freq": freq, "sha256": None})
    return out


def eligible_frame(
    feats: pd.DataFrame,
    seq_len: int,
    min_history: int = MIN_HISTORY,
    keys: tuple[str, ...] = ("session_date", "symbol", "target_up", "target_ret_oo"),
) -> pd.DataFrame:
    """Like `model_frame`, plus the H-EXT entry rule: rows at bar positions < `min_history` (fewer than
    `min_history` bars of history) are dropped. Lag columns (only if seq_len > 1) are float32."""
    parts = [feats[list(keys)], feats[FEATURE_COLUMNS].astype(np.float64)]
    if seq_len > 1:
        parts.append(lagged_matrix(feats, FEATURE_COLUMNS, seq_len).astype(np.float32))
    df = pd.concat(parts, axis=1)
    df = df.iloc[min_history:]
    return df.dropna().reset_index(drop=True)


def build_panel(
    symbols: list[str],
    paths: Paths,
    seq_len: int = SEQ_LEN,
    freq: str = "1d",
    min_history: int = MIN_HISTORY,
    time_col: str = "session_date",
) -> pd.DataFrame:
    keys: tuple[str, ...] = ("session_date", "symbol", "target_up", "target_ret_oo")
    if time_col not in keys:
        keys = (time_col, *keys)
    frames = []
    for s in symbols:
        f = load_features(s, paths, window=None, freq=freq)
        frames.append(eligible_frame(f, seq_len, min_history, keys))
    df = pd.concat(frames, ignore_index=True)
    df[time_col] = pd.to_datetime(df[time_col])
    if time_col != "session_date":
        df["session_date"] = pd.to_datetime(df["session_date"])
    return df.sort_values([time_col, "symbol"], kind="mergesort").reset_index(drop=True)


@dataclass(frozen=True)
class Fold:
    k: int
    train: np.ndarray | slice
    test: np.ndarray | slice
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int


def _as_slice(idx: np.ndarray) -> np.ndarray | slice:
    if len(idx) and idx[-1] - idx[0] + 1 == len(idx):
        return slice(int(idx[0]), int(idx[-1]) + 1)  # contiguous -> view, no copy
    return idx


def calendar_folds(
    times: pd.Series, initial: pd.Timedelta, test: pd.Timedelta, gap: pd.Timedelta
) -> Iterator[Fold]:
    """Expanding walk-forward in wall-clock time: the first test block starts `initial` after the
    first timestamp; each test block covers [start, start + test); training rows have time < start - gap.
    Every row sharing a timestamp falls in the same block."""
    ts = pd.Series(pd.to_datetime(times))
    if ts.dt.tz is not None:  # intraday bars: compare in naive UTC
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    t = ts.to_numpy(dtype="datetime64[ns]")
    first, last = t.min(), t.max()
    start = first + initial.to_timedelta64()
    k = 0
    while start <= last:
        stop = start + test.to_timedelta64()
        tr = np.flatnonzero(t < start - gap.to_timedelta64())
        te = np.flatnonzero((t >= start) & (t < stop))
        if len(tr) and len(te):
            yield Fold(
                k,
                _as_slice(tr),
                _as_slice(te),
                pd.Timestamp(t[tr[-1]]),
                pd.Timestamp(t[te[0]]),
                pd.Timestamp(t[te[-1]]),
                len(tr),
                len(te),
            )
            k += 1
        start = stop


# ------------------------------------------------------------------------------ models


def make_factory(
    name: str, device: str | None, val_gap: int, dl_overrides: dict[str, dict[str, Any]] | None = None
) -> Callable[[], Any]:
    """Fixed H-ML-001/002 models. DL models get the panel-aware early-stopping purge `val_gap` and any
    documented implementation overrides (batch size / max epochs) from `dl_overrides`."""
    if name in ("logit", "xgb", "rf"):
        return model_factory(name, device)
    from astra.dl.models import TorchClassifier

    kw: dict[str, Any] = {"kind": name, "seed": SEED, "device": device, "val_gap": val_gap}
    if name == "gru":
        kw.update(n_features=len(FEATURE_COLUMNS), seq_len=SEQ_LEN)
    kw.update((dl_overrides or {}).get(name, {}))
    return lambda: TorchClassifier(**kw)


def _inputs(df: pd.DataFrame, name: str) -> np.ndarray:
    if name == "gru":
        return np.ascontiguousarray(df[lag_columns(SEQ_LEN)].to_numpy(dtype=np.float32))
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def walk_forward(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    factory: Callable[[], Any],
    feature_names: list[str] | None = None,
    importance: bool = False,
    checkpoint: Path | None = None,
    on_fold: Callable[[], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """OOS probabilities, ex-ante base rate, per-fold table, and (optionally) permutation importance
    computed on each TEST fold (reported only; never used for selection).

    `checkpoint`: directory for per-fold results. A fold already stored there is loaded instead of
    refit, so a run interrupted by e.g. a GPU driver fault resumes where it stopped. Every model
    re-seeds inside `fit`, so a fold's result does not depend on earlier folds."""
    proba = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    rows, imps = [], []
    for f in folds:
        ck = checkpoint / f"{name}_fold{f.k:03d}" if checkpoint is not None else None
        if ck is not None and ck.with_suffix(".json").exists():
            saved = json.loads(ck.with_suffix(".json").read_text(encoding="utf-8"))
            proba[f.test] = np.load(ck.with_suffix(".npy"))
            base[f.test] = saved["row"]["train_base_rate"]
            rows.append({**saved["row"], "resumed": True})
            if saved.get("importance"):
                imps.append(pd.DataFrame(saved["importance"]))
            continue
        ts = time.perf_counter()
        model = factory()
        model.fit(X[f.train], y[f.train])
        p = model.predict_proba(X[f.test])[:, 1]
        b = float(y[f.train].mean())
        proba[f.test], base[f.test] = p, b
        yt = y[f.test]
        ll_m = _pointwise_log_loss(yt, p).mean()
        ll_b = _pointwise_log_loss(yt, np.full_like(p, b)).mean()
        row = {
            "model": name,
            "fold": f.k,
            "train_end": f.train_end,
            "test_start": f.test_start,
            "test_end": f.test_end,
            "n_train": f.n_train,
            "n_test": f.n_test,
            "train_base_rate": b,
            "log_loss": float(ll_m),
            "baseline_log_loss": float(ll_b),
            "ll_improvement": float(ll_b - ll_m),
            "auc": _safe_auc(yt, p),
            "seconds": round(time.perf_counter() - ts, 2),
        }
        hist = getattr(model, "history", None)
        if isinstance(hist, dict) and hist:
            row["epochs"] = hist.get("epochs")
        rows.append(row)
        if importance and feature_names is not None:
            pi = permutation_importance(
                model, X[f.test], yt, scoring="neg_log_loss", n_repeats=3, random_state=SEED, n_jobs=1
            )
            imps.append(
                pd.DataFrame(
                    {
                        "fold": f.k,
                        "feature": feature_names,
                        "importance_mean": pi.importances_mean,
                        "importance_std": pi.importances_std,
                        "n_test": f.n_test,
                    }
                )
            )
        if ck is not None:
            np.save(ck.with_suffix(".npy"), p)
            payload = {
                "row": {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in row.items()},
                "importance": imps[-1].to_dict(orient="list") if importance and feature_names else None,
            }
            ck.with_suffix(".json").write_text(json.dumps(payload, default=float), encoding="utf-8")
        if on_fold is not None:
            on_fold()
        if hist:  # torch model: release cached CUDA blocks so expanding folds do not grow the reserve
            del model
            _empty_cuda_cache()
        log.info("ml_ext.fold", model=name, fold=f.k, n_train=f.n_train, dll=row["ll_improvement"])
    return proba, base, pd.DataFrame(rows), (pd.concat(imps, ignore_index=True) if imps else pd.DataFrame())


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def importance_summary(imp: pd.DataFrame) -> pd.DataFrame:
    """Test-row-weighted mean permutation importance across folds (increase in log loss when permuted)."""
    if imp.empty:
        return imp
    w = imp.assign(wi=imp["importance_mean"] * imp["n_test"])
    g = w.groupby("feature")
    out = pd.DataFrame(
        {
            "importance_wmean": g["wi"].sum() / g["n_test"].sum(),
            "fold_std": g["importance_mean"].std(),
            "share_folds_positive": g["importance_mean"].apply(lambda s: float((s > 0).mean())),
        }
    )
    return out.sort_values("importance_wmean", ascending=False).reset_index()


def fold_dispersion(per_fold: pd.DataFrame, col: str = "ll_improvement") -> dict[str, float]:
    s = per_fold[col].astype(float)
    return {
        "n_folds": len(s),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "share_positive": float((s > 0).mean()),
    }


def _verdict(t_base: float, bh_sig: bool) -> Conclusion:
    if bh_sig:
        return "supported"
    if t_base < -1.96:
        return "rejected"
    return "inconclusive"


def _merge_resources(
    current: dict[str, Any], per_fold: pd.DataFrame, checkpoint_dir: Path | None, name: str
) -> dict[str, Any]:
    """If folds were resumed from a checkpoint, runtime = sum of per-fold fit seconds over all segments
    and memory peaks = max over the segments (each segment = one process)."""
    resumed = "resumed" in per_fold and bool(per_fold["resumed"].fillna(False).astype(bool).any())
    if not resumed or checkpoint_dir is None:
        return {**current, "segments": 1}
    segs = [
        json.loads(q.read_text(encoding="utf-8")) for q in checkpoint_dir.glob(f"{name}_resources_pid*.json")
    ]
    out = dict(current)
    out["runtime_s"] = round(float(per_fold["seconds"].sum()), 1)
    out["runtime_note"] = "sum of per-fold fit seconds across resumed segments"
    for k in ("peak_tracemalloc_mb", "peak_working_set_mb_sampled"):
        out[k] = max([current[k], *[sg[k] for sg in segs]])
    gpu = dict(current["gpu"])
    for k, gk in (
        ("peak_vram_allocated_mb", "peak_allocated_mb"),
        ("peak_vram_reserved_mb", "peak_reserved_mb"),
    ):
        gpu[gk] = max([gpu.get(gk, 0.0), *[sg[k] for sg in segs]])
    out["gpu"], out["segments"] = gpu, len(segs)
    return out


# ------------------------------------------------------------------------------ shared study driver


def _run_pooled_study(
    *,
    hypothesis_id: str,
    df: pd.DataFrame,
    time_col: str,
    folds: list[Fold],
    models: list[str],
    device: str | None,
    dl_overrides: dict[str, dict[str, Any]],
    val_gap: int,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    y = df["target_up"].to_numpy(dtype=np.float64)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "hypothesis": hypothesis_id,
            "rows": len(df),
            "first": str(df[time_col].iloc[0]),
            "last": str(df[time_col].iloc[-1]),
            "n_folds": len(folds),
            "val_gap": val_gap,
            "dl_overrides": dl_overrides,
            "target_sum": float(y.sum()),
        }
        mp = checkpoint_dir / "meta.json"
        if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != json.loads(json.dumps(meta)):
            raise ValueError(f"checkpoint {checkpoint_dir} was written for a different panel/config")
        mp.write_text(json.dumps(meta), encoding="utf-8")
    probas: dict[str, np.ndarray] = {}
    per_fold, imps, resources = [], {}, {}
    base = np.full(len(y), np.nan)
    for name in models:
        set_seed(SEED)
        X = _inputs(df, name)
        seg = checkpoint_dir / f"{name}_resources_pid{os.getpid()}.json" if checkpoint_dir else None
        with ResourceProbe() as probe:

            def _save_snapshot(probe: ResourceProbe = probe, seg: Path | None = seg) -> None:
                if seg is not None:
                    seg.write_text(json.dumps(probe.snapshot()), encoding="utf-8")

            proba, base, pf, imp = walk_forward(
                name,
                X,
                y,
                folds,
                make_factory(name, device, val_gap, dl_overrides),
                feature_names=FEATURE_COLUMNS if name == "xgb" else None,
                importance=name == "xgb",
                checkpoint=checkpoint_dir,
                on_fold=_save_snapshot,
            )
        del X
        resources[name] = _merge_resources(probe.result, pf, checkpoint_dir, name)
        probas[name], imps[name] = proba, imp
        per_fold.append(pf)
        log.info("ml_ext.model_done", hypothesis=hypothesis_id, model=name, **probe.result)
    mask = ~np.isnan(base)
    for name in models:  # identical rows/folds for every model
        mask &= ~np.isnan(probas[name])
    times = df.loc[mask, time_col].to_numpy()
    yb, bb = y[mask], base[mask]
    ll_base = _pointwise_log_loss(yb, bb)
    folds_df = pd.concat(per_fold, ignore_index=True)
    rows = []
    for name in models:
        pm = probas[name][mask]
        t = _date_dm(yb, bb, pm, times)
        rows.append(
            {
                "model": name,
                "n_oos": int(mask.sum()),
                "n_folds": len(folds),
                "auc": _safe_auc(yb, pm),
                "log_loss": float(_pointwise_log_loss(yb, pm).mean()),
                "baseline_log_loss": float(ll_base.mean()),
                "brier": float(np.mean((pm - yb) ** 2)),
                "dm_t_vs_baseline": t,
                "dm_t_vs_baseline_rowwise": newey_west_tstat(
                    pd.Series(ll_base - _pointwise_log_loss(yb, pm))
                ),
                "p_one_sided": _one_sided_p(t),
                "dm_t_vs_logit": _date_dm(yb, probas["logit"][mask], pm, times)
                if name in ("mlp", "gru") and "logit" in probas
                else float("nan"),
                **{
                    f"fold_dll_{k}": v
                    for k, v in fold_dispersion(folds_df[folds_df["model"] == name]).items()
                },
                "runtime_s": resources[name]["runtime_s"],
                "peak_tracemalloc_mb": resources[name]["peak_tracemalloc_mb"],
                "peak_working_set_mb": resources[name]["peak_working_set_mb_sampled"],
                "peak_vram_allocated_mb": resources[name]["gpu"].get("peak_allocated_mb", 0.0),
                "peak_vram_reserved_mb": resources[name]["gpu"].get("peak_reserved_mb", 0.0),
            }
        )
    table = pd.DataFrame(rows)
    reject, p_adj, *_ = multipletests(table["p_one_sided"], alpha=0.05, method="fdr_bh")
    table["p_bh"], table["bh_significant"] = p_adj, reject
    table["verdict"] = [_verdict(t, bool(r)) for t, r in zip(table["dm_t_vs_baseline"], reject, strict=True)]
    baseline_row = {
        "model": "baseline (ex-ante train-fold base rate)",
        "n_oos": int(mask.sum()),
        "n_folds": len(folds),
        "auc": 0.5,
        "log_loss": float(ll_base.mean()),
        "baseline_log_loss": float(ll_base.mean()),
        "brier": float(np.mean((bb - yb) ** 2)),
    }
    table = pd.concat([pd.DataFrame([baseline_row]), table], ignore_index=True)
    preds = df.loc[mask, [time_col, "symbol", "target_up"]].copy()
    if time_col != "session_date":
        preds["session_date"] = df.loc[mask, "session_date"].to_numpy()
    preds["baseline"] = bb
    for name in models:
        preds[f"p_{name}"] = probas[name][mask].astype(np.float32)
    return {
        "table": table,
        "folds": folds_df,
        "predictions": preds.reset_index(drop=True),
        "importance": imps.get("xgb", pd.DataFrame()),
        "resources": resources,
    }


def _log_records(
    *,
    hypothesis_id: str,
    statement: str,
    res: dict[str, Any],
    models: list[str],
    paths: Paths,
    datasets: list[dict[str, Any]],
    validation: dict[str, Any],
    model_cfg: dict[str, Any],
    features: list[str],
    notes: str,
    reproduce: str,
    extra_metrics: dict[str, Any],
) -> dict[str, Any]:
    tracker = Tracker(paths)
    table = res["table"]
    mt = table[table["model"].isin(models)].set_index("model")
    n_sig = int(mt["bh_significant"].sum())
    family_concl: Conclusion = "supported" if n_sig > 0 else "rejected"
    fam = tracker.new(
        f"{hypothesis_id.lower()}_family",
        hypothesis_id,
        statement,
        datasets=datasets,
        features=features,
        model=model_cfg,
        validation=validation,
        seed=SEED,
        metrics={
            "bh_significant_models": n_sig,
            "nominal_t_gt_1.96": int((mt["dm_t_vs_baseline"] > 1.96).sum()),
            "significantly_worse": int((mt["dm_t_vs_baseline"] < -1.96).sum()),
            "per_model": mt.drop(columns=[c for c in mt.columns if c.startswith("fold_dll_")]).to_dict(
                orient="index"
            ),
            **extra_metrics,
        },
        variants_tried=len(models),
        conclusion=family_concl,
        notes=notes,
        reproduce=reproduce,
    )
    tracker.save_frame(fam, "metrics", table)
    tracker.save_frame(fam, "folds", res["folds"])
    tracker.save_frame(fam, "oos_predictions", res["predictions"])
    if not res["importance"].empty:
        tracker.save_frame(fam, "xgb_permutation_importance_by_fold", res["importance"])
        tracker.save_frame(fam, "xgb_permutation_importance", importance_summary(res["importance"]))
    tracker.log(fam)
    out: dict[str, Any] = {"family_id": fam.experiment_id, "conclusion": family_concl, "models": {}}
    for name in models:
        r = mt.loc[name].to_dict()
        rec = tracker.new(
            f"{hypothesis_id.lower()}_{name}",
            hypothesis_id,
            f"{name}: OOS log loss below the ex-ante base rate (date-clustered DM, BH over {len(models)}).",
            datasets=[{"family_record": fam.experiment_id, "rows": int(r["n_oos"])}],
            features=lag_columns(SEQ_LEN) if name == "gru" else FEATURE_COLUMNS,
            model={"type": name, **model_cfg.get("per_model", {}).get(name, {})},
            validation=validation,
            seed=SEED,
            metrics={**r, "resources": res["resources"][name]},
            variants_tried=len(models),
            conclusion=r["verdict"],
            notes=f"Family record {fam.experiment_id} holds predictions/folds/importance. " + notes,
            reproduce=reproduce,
        )
        tracker.log(rec)
        out["models"][name] = {"id": rec.experiment_id, **r}
    return out


# ------------------------------------------------------------------------------ H-ML-003


def run_ml_panel_extended(
    symbols: list[str] | None = None,
    device: str | None = None,
    paths: Paths | None = None,
    models: list[str] | None = None,
    dl_overrides: dict[str, dict[str, Any]] | None = None,
    log_tracker: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """H-ML-003: pooled daily panel over the whole configured universe, full history."""
    set_seed(SEED)
    paths = paths or get_paths()
    cfg = get_config()
    symbols = symbols or cfg.symbols()
    models = models or PANEL_MODELS
    dl_overrides = dl_overrides or {}
    with ResourceProbe() as load_probe:
        df = build_panel(symbols, paths, seq_len=SEQ_LEN)
    max_rows_per_date = int(df.groupby("session_date").size().max())
    val_gap = 3 * max_rows_per_date
    folds = list(
        calendar_folds(
            df["session_date"], pd.Timedelta(days=730), pd.Timedelta(days=91), pd.Timedelta(days=7)
        )
    )
    log.info("ml_ext.panel", rows=len(df), symbols=len(symbols), folds=len(folds), **load_probe.result)
    res = _run_pooled_study(
        hypothesis_id="H-ML-003",
        df=df,
        time_col="session_date",
        folds=folds,
        models=models,
        device=device,
        dl_overrides=dl_overrides,
        val_gap=val_gap,
        checkpoint_dir=checkpoint_dir,
    )
    out: dict[str, Any] = {
        "rows": len(df),
        "n_symbols": len(symbols),
        "n_folds": len(folds),
        "first_test": str(folds[0].test_start.date()) if folds else None,
        "last_test": str(folds[-1].test_end.date()) if folds else None,
        "load": load_probe.result,
        "val_gap_rows": val_gap,
        **res,
    }
    if not log_tracker:
        return out
    validation = {
        "type": "calendar-time purged expanding walk-forward (all rows of a date share a fold)",
        "initial_train_days": 730,
        "test_days": 91,
        "gap_days": 7,
        "n_folds": len(folds),
        "first_test_start": out["first_test"],
        "last_test_end": out["last_test"],
        "dl_early_stopping": f"last 15% of training rows, purge {val_gap} rows (= 3 x max rows/date)",
    }
    model_cfg = {
        "models": models,
        "hyperparameters": "fixed H-ML-001/002 (ml_study.model_factory / TorchClassifier defaults)",
        "dl_overrides": dl_overrides,
        "device": str(device or "auto"),
        "per_model": {k: {"overrides": v} for k, v in dl_overrides.items()},
    }
    out["records"] = _log_records(
        hypothesis_id="H-ML-003",
        statement="At least one of logit/XGBoost/MLP/GRU pooled over all instruments (full history) has OOS "
        "log loss significantly below the ex-ante base rate (date-clustered DM, BH across 4 models).",
        res=res,
        models=models,
        paths=paths,
        datasets=[
            {
                "universe": "config/markets.yaml all instruments",
                "n_symbols": len(symbols),
                "rows": len(df),
                "window": "full history",
                "entry_rule": f"{MIN_HISTORY} bars of history",
                "survivorship": SURVIVORSHIP,
                "manifests": data_versions(symbols, paths),
            }
        ],
        validation=validation,
        model_cfg=model_cfg,
        features=FEATURE_COLUMNS,
        notes=SURVIVORSHIP
        + " Trading evaluation only for models passing the gate (hand to backtest-auditor).",
        reproduce='uv run --no-sync python -c "from astra.research.ml_extended import '
        f'run_ml_panel_extended as r; r(dl_overrides={dl_overrides!r})"',
        extra_metrics={"rows": len(df), "n_symbols": len(symbols), "load_resources": load_probe.result},
    )
    return out


# ------------------------------------------------------------------------------ H-ML-004


def hourly_data_ready(paths: Paths | None = None) -> tuple[bool, list[str]]:
    """True only if every configured crypto pair has validated 1h bars (BTCUSDT included)."""
    paths = paths or get_paths()
    syms = get_config().symbols(market="CRYPTO")
    missing = [s for s in syms if not resolved_exists(validated_path(paths, "CRYPTO", s, "1h"), paths)]
    return (not missing and "BTCUSDT" in syms), missing


def hourly_backtest(preds: pd.DataFrame, model: str, round_trip_bps: float = CRYPTO_ROUND_TRIP_BPS) -> dict:
    """Descriptive long-only/flat hourly backtest of a model that PASSED the gate: per pair, hold from
    open t+1 to open t+2 when p > 0.5; half the round-trip cost per position change; equal-weight across
    pairs with a bar. Hand any trading claim to the backtest-auditor."""
    d = preds.sort_values(["symbol", "close_time_utc"]).copy()
    d["pos"] = (d[f"p_{model}"] > 0.5).astype(float)
    d["turn"] = d.groupby("symbol")["pos"].diff().abs().fillna(d["pos"])
    d["net"] = d["pos"] * (np.exp(d["target_ret_oo"]) - 1) - d["turn"] * round_trip_bps / 2 / 1e4
    port = d.groupby("close_time_utc")["net"].mean()
    ann = np.sqrt(24 * 365)
    return {
        "bars": len(port),
        "mean_bar_return": float(port.mean()),
        "sharpe_annualized": float(port.mean() / port.std() * ann) if port.std() > 0 else float("nan"),
        "nw_t": newey_west_tstat(port),
        "avg_exposure": float(d["pos"].mean()),
        "round_trip_bps": round_trip_bps,
    }


def run_hourly_crypto(
    device: str | None = None,
    paths: Paths | None = None,
    models: list[str] | None = None,
    log_tracker: bool = True,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """H-ML-004: pooled hourly crypto panel (16 pairs, 1h bars, full history)."""
    paths = paths or get_paths()
    ready, missing = hourly_data_ready(paths)
    if not ready:
        raise FileNotFoundError(f"H-ML-004 needs validated 1h bars for every crypto pair; missing: {missing}")
    set_seed(SEED)
    symbols = get_config().symbols(market="CRYPTO")
    models = models or HOURLY_MODELS
    with ResourceProbe() as load_probe:
        df = build_panel(
            symbols, paths, seq_len=1, freq="1h", min_history=MIN_HISTORY * 24, time_col="close_time_utc"
        )
    max_rows = int(df.groupby("close_time_utc").size().max())
    folds = list(
        calendar_folds(
            df["close_time_utc"], pd.Timedelta(days=730), pd.Timedelta(days=91), pd.Timedelta(hours=48)
        )
    )
    res = _run_pooled_study(
        hypothesis_id="H-ML-004",
        df=df,
        time_col="close_time_utc",
        folds=folds,
        models=models,
        device=device,
        dl_overrides={},
        val_gap=3 * max_rows,
        checkpoint_dir=checkpoint_dir,
    )
    out: dict[str, Any] = {"rows": len(df), "n_folds": len(folds), "load": load_probe.result, **res}
    tab = res["table"]
    passed = tab.loc[tab["bh_significant"].fillna(False).astype(bool), "model"].tolist()
    out["backtests"] = {m: hourly_backtest(_with_returns(res["predictions"], df), m) for m in passed}
    if not log_tracker:
        return out
    out["records"] = _log_records(
        hypothesis_id="H-ML-004",
        statement="At least one of logit/XGBoost/MLP pooled over 16 crypto pairs on 1h bars has OOS log loss "
        "significantly below the ex-ante base rate (time-clustered DM, BH across 3 models).",
        res=res,
        models=models,
        paths=paths,
        datasets=[
            {
                "universe": "16 crypto pairs (config)",
                "freq": "1h",
                "rows": len(df),
                "entry_rule": f"{MIN_HISTORY * 24} bars (252 days)",
                "survivorship": SURVIVORSHIP,
                "manifests": data_versions(symbols, paths, "1h"),
            }
        ],
        validation={
            "type": "calendar-time purged expanding walk-forward",
            "initial_train_days": 730,
            "test_days": 91,
            "gap_hours": 48,
            "n_folds": len(folds),
        },
        model_cfg={"models": models, "hyperparameters": "fixed H-ML-001/002"},
        features=FEATURE_COLUMNS,
        notes=SURVIVORSHIP
        + " Cost-aware hourly backtest (26 bps round trip) only for models passing the gate.",
        reproduce='uv run --no-sync python -c "from astra.research.ml_extended import run_hourly_crypto; '
        'run_hourly_crypto()"',
        extra_metrics={"rows": len(df), "backtests": out["backtests"], "load_resources": load_probe.result},
    )
    return out


def _with_returns(preds: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return preds.merge(
        df[["close_time_utc", "symbol", "target_ret_oo"]], on=["close_time_utc", "symbol"], how="left"
    )
