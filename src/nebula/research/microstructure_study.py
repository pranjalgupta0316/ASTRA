"""M7 microstructure studies (registered in docs/research/H-MS.md before analysis).

H-MS-000: Kyle-lambda-style impact on one day of Binance spot aggTrades (BTCUSDT).
H-MS-001: top-5 L2 book imbalance -> sign of the next 1 s / 10 s mid change, on a short live capture.

Only small summary artifacts are saved to experiments/ - never raw ticks.
L2 only (Binance public depth); no L3, no historical depth.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nebula.common.paths import get_paths
from nebula.microstructure.capture import load_capture
from nebula.microstructure.orderbook import ofi_series, replay
from nebula.microstructure.trades import aggtrades_summary, aggtrades_url, download_latest_complete
from nebula.tracking.tracker import ExperimentRecord, Tracker

H_MS_000 = (
    "1-minute VWAP change (bps) is positively related to aggressor-signed volume "
    "(OLS slope lambda > 0, Newey-West HAC t > 1.96) on one day of BTCUSDT spot aggTrades."
)
H_MS_001 = (
    "Top-5 L2 book imbalance at t predicts the sign of the mid change over the next 1 s and 10 s "
    "(ROC AUC > 0.5; 95% moving-block bootstrap CI lower bound > 0.5 at both horizons)."
)
HORIZONS_S = (1, 10)
BLOCK_S = 30
N_BOOT = 2000
SEED = 0
MIN_NONZERO = 50
MIN_BLOCKS = 5


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------ H-MS-000
def run_aggtrades_study(symbol: str = "BTCUSDT", today: date | None = None) -> ExperimentRecord:
    day, zip_path = download_latest_complete(symbol, today=today)
    res = aggtrades_summary(zip_path)
    k = res["kyle"]
    t = k.get("t_hac", float("nan"))
    lam = k.get("lambda_bps_per_unit", float("nan"))
    if not math.isfinite(t):
        conclusion = "inconclusive"
    elif lam > 0 and t > 1.96:
        conclusion = "supported"
    elif t < -1.96:
        conclusion = "rejected"
    else:
        conclusion = "inconclusive"
    tr = Tracker()
    meta_path = zip_path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rec = tr.new(
        name=f"aggTrades impact {symbol} {day.isoformat()}",
        hypothesis_id="H-MS-000",
        hypothesis=H_MS_000,
        datasets=[
            {
                "provider": "binance_vision",
                "url": aggtrades_url(symbol, day),
                "path": _rel(zip_path, get_paths().root),
                "sha256": meta.get("sha256") or _sha256(zip_path),
                "checksum_verified": bool(meta.get("checksum_verified", False)),
                "day_utc": day.isoformat(),
            }
        ],
        features=["signed_volume_1m"],
        model={"type": "OLS", "y": "1e4*dlog(VWAP_1m)", "x": "signed base volume (buy-sell, aggressor)"},
        validation={"cov": "HAC Newey-West", "maxlags_rule": "floor(4*(n/100)^(2/9))", "sample": "1 day"},
        seed=None,
    )
    rec.metrics = {**res["stats"], **{f"kyle_{a}": b for a, b in k.items()}}
    rec.conclusion = conclusion  # type: ignore[assignment]
    rec.notes = (
        "Contemporaneous reduced-form regression on one UTC day; association, not causal and not tradable. "
        "VWAP change is a trade-price proxy (no quotes in aggTrades). Aggressor from is_buyer_maker "
        "(True -> sell). Realized vol per bar uses trade-to-trade returns (includes bid-ask bounce)."
    )
    rec.reproduce = "uv run --no-sync python -m nebula.research.microstructure_study aggtrades"
    b1m = res["bars_1m"].to_pandas()
    tr.save_frame(rec, "bars_1m", b1m)
    tr.save_frame(rec, "kyle_lambda", pd.DataFrame([k]))
    hourly = (
        b1m.assign(hour=b1m["ts"].dt.floor("h"))
        .groupby("hour")
        .agg(
            n_trades=("n_trades", "sum"),
            volume=("volume", "sum"),
            signed_volume=("signed_volume", "sum"),
            imbalance_mean=("trade_imbalance", "mean"),
        )
        .reset_index()
    )
    tr.save_frame(rec, "hourly_summary", hourly)
    tr.log(rec)
    return rec


# ------------------------------------------------------------------ H-MS-001 helpers
def auc(score: np.ndarray, label: np.ndarray) -> float:
    """ROC AUC via Mann-Whitney U with average ranks for ties. label: 1 = up, 0 = down."""
    pos = label == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def moving_block_bootstrap_auc(
    t_s: np.ndarray,
    score: np.ndarray,
    label: np.ndarray,
    block_s: float = BLOCK_S,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Moving-block bootstrap over time-contiguous windows of `block_s` seconds -> (lo, hi, n_blocks)."""
    order = np.argsort(t_s, kind="stable")
    t_s, score, label = t_s[order], score[order], label[order]
    span = float(t_s[-1] - t_s[0]) if len(t_s) else 0.0
    n_blocks = math.floor(span / block_s)
    if n_blocks < 1:
        return float("nan"), float("nan"), n_blocks
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    hi_start = t_s[-1] - block_s
    for b in range(n_boot):
        starts = (
            rng.uniform(t_s[0], hi_start, size=n_blocks) if hi_start > t_s[0] else np.full(n_blocks, t_s[0])
        )
        lo_i = np.searchsorted(t_s, starts, side="left")
        hi_i = np.searchsorted(t_s, starts + block_s, side="left")
        idx = np.concatenate([np.arange(a, z) for a, z in zip(lo_i, hi_i, strict=True)])
        out[b] = auc(score[idx], label[idx])
    lo, hi = np.nanquantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), n_blocks


def future_mid_change(states: pd.DataFrame, horizon_s: float) -> pd.Series:
    """mid(t+h) - mid(t) with mid(t+h) = last valid mid with E <= t+h in the same valid segment.

    NaN where t+h is beyond the end of the segment.
    """
    out = pd.Series(np.nan, index=states.index)
    h_ms = round(horizon_s * 1000)
    for _, g in states.groupby("segment", sort=False):
        e = g["E_ms"].to_numpy(dtype=np.int64)
        m = g["mid"].to_numpy(dtype=float)
        j = np.searchsorted(e, e + h_ms, side="right") - 1
        ok = e + h_ms <= e[-1]
        dm = np.where(ok, m[j] - m, np.nan)
        out.loc[g.index] = dm
    return out


def book_states(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    snaps, verify, events = load_capture(run_dir)
    rows, st = replay(snaps, events, k=5, verify=verify)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, st.as_dict()
    df = df[df["valid"]].sort_values(["segment", "E_ms"], kind="stable").reset_index(drop=True)
    ofi = np.full(len(df), np.nan)
    for _, g in df.groupby("segment", sort=False):
        s = ofi_series(
            g["best_bid"].tolist(), g["bid_qty"].tolist(), g["best_ask"].tolist(), g["ask_qty"].tolist()
        )
        ofi[g.index.to_numpy()] = s
    df["ofi"] = ofi
    return df, st.as_dict()


def h_ms_001(states: pd.DataFrame, score_col: str = "imbalance_5") -> dict[str, Any]:
    res: dict[str, Any] = {}
    for h in HORIZONS_S:
        dm = future_mid_change(states, h)
        ok = dm.notna() & states[score_col].notna()
        d = dm[ok].to_numpy()
        n_all = int(ok.sum())
        nz = d != 0
        t_s = states.loc[ok, "E_ms"].to_numpy(dtype=float)[nz] / 1000.0
        score = states.loc[ok, score_col].to_numpy(dtype=float)[nz]
        label = (d[nz] > 0).astype(int)
        a = auc(score, label)
        lo, hi, nb = (
            moving_block_bootstrap_auc(t_s, score, label) if len(label) else (float("nan"), float("nan"), 0)
        )
        res[f"h{h}s"] = {
            "n_obs": n_all,
            "n_nonzero": int(nz.sum()),
            "zero_change_rate": float(1 - nz.mean()) if n_all else float("nan"),
            "up_rate_given_change": float(label.mean()) if len(label) else float("nan"),
            "auc": a,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_blocks": nb,
        }
    return res


def h_ms_001_verdict(res: dict[str, Any]) -> tuple[str, str]:
    hs = [res[f"h{h}s"] for h in HORIZONS_S]
    if any(r["n_nonzero"] < MIN_NONZERO or r["n_blocks"] < MIN_BLOCKS for r in hs):
        return "inconclusive", "insufficient data (registered minimums not met)"
    if any(not (math.isfinite(r["ci_lo"]) and math.isfinite(r["ci_hi"])) for r in hs):
        return "inconclusive", "bootstrap CI undefined"
    if all(r["ci_lo"] > 0.5 for r in hs):
        return "supported", "CI lower bound > 0.5 at both horizons"
    if all(r["ci_hi"] < 0.5 for r in hs):
        return "rejected", "CI upper bound < 0.5 at both horizons"
    return "inconclusive", "CI includes 0.5 at one or both horizons, or horizons disagree"


def capture_descriptives(states: pd.DataFrame, run_dir: Path) -> dict[str, Any]:
    d: dict[str, Any] = {}
    if states.empty:
        return d
    d["valid_states"] = len(states)
    d["segments"] = int(states["segment"].nunique())
    d["span_s"] = float((states["E_ms"].max() - states["E_ms"].min()) / 1000)
    d["spread_bps_median"] = float(states["spread_bps"].median())
    d["spread_bps_p95"] = float(states["spread_bps"].quantile(0.95))
    tick = float(np.round((states["best_ask"] - states["best_bid"]).min(), 8))
    d["min_spread_abs"] = tick
    d["share_min_spread"] = float(np.isclose(states["best_ask"] - states["best_bid"], tick).mean())
    d["depth5_bid_median"] = float(states["depth_bid_5"].median())
    d["depth5_ask_median"] = float(states["depth_ask_5"].median())
    d["imbalance5_mean"] = float(states["imbalance_5"].mean())
    d["imbalance5_std"] = float(states["imbalance_5"].std())
    d["mid_first"] = float(states["mid"].iloc[0])
    d["mid_last"] = float(states["mid"].iloc[-1])
    lat_ms = states["recv_ns"] / 1e6 - states["E_ms"]
    d["recv_minus_E_ms_median"] = float(lat_ms.median())  # latency + local clock offset (not separable)
    # descriptive (not registered): contemporaneous 1 s OFI vs 1 s mid change
    s = states.assign(sec=(states["E_ms"] // 1000).astype("int64"))
    per_s = s.groupby(["segment", "sec"]).agg(ofi=("ofi", "sum"), mid_last=("mid", "last")).reset_index()
    per_s["dmid"] = per_s.groupby("segment")["mid_last"].diff()
    ok = per_s["dmid"].notna()
    d["ofi_1s_corr_dmid"] = (
        float(np.corrcoef(per_s.loc[ok, "ofi"], per_s.loc[ok, "dmid"])[0, 1])
        if ok.sum() > 2
        else float("nan")
    )
    tp = run_dir / "agg_trades.parquet"
    if tp.exists():
        tdf = pd.read_parquet(tp)
        sign = np.where(tdf["is_buyer_maker"], -1.0, 1.0)
        d["ws_agg_trades"] = len(tdf)
        d["ws_trade_imbalance"] = (
            float((sign * tdf["qty"]).sum() / tdf["qty"].sum()) if len(tdf) else float("nan")
        )
    return d


def run_capture_study(run_dir: Path | None = None, symbol: str = "BTCUSDT") -> ExperimentRecord:
    paths = get_paths()
    if run_dir is None:
        cands = sorted(p for p in (paths.data / "microstructure" / symbol).iterdir() if p.is_dir())
        run_dir = cands[-1]
    run_dir = run_dir.resolve()
    cap_rep = json.loads((run_dir / "capture_report.json").read_text(encoding="utf-8"))
    states, rstats = book_states(run_dir)
    desc = capture_descriptives(states, run_dir)
    res = h_ms_001(states) if not states.empty else {}
    res_desc_l1 = h_ms_001(states, "imbalance_1") if not states.empty else {}  # descriptive only
    verdict, reason = h_ms_001_verdict(res) if res else ("inconclusive", "no valid book states")
    tr = Tracker()
    files = sorted(run_dir.glob("*.parquet"))
    rec = tr.new(
        name=f"L2 imbalance -> mid sign, live capture {symbol} {run_dir.name}",
        hypothesis_id="H-MS-001",
        hypothesis=H_MS_001,
        datasets=[
            {
                "provider": "binance_ws_l2_live_capture",
                "data_level": "L2 market-by-price (no L3)",
                "run_dir": _rel(run_dir, paths.root),
                "files": {f.name: {"bytes": f.stat().st_size, "sha256": _sha256(f)} for f in files},
                "capture_report": {k: v for k, v in cap_rep.items() if k != "errors"},
            }
        ],
        features=["imbalance_5"],
        model={"type": "rank score (no fit)", "score": "top-5 book imbalance", "target": "sign(dmid)"},
        validation={
            "metric": "ROC AUC on non-zero mid changes",
            "ci": f"moving-block bootstrap, block={BLOCK_S}s, n_boot={N_BOOT}, seed={SEED}",
            "horizons_s": list(HORIZONS_S),
            "decision_rule": "docs/research/H-MS.md H-MS-001 item 6",
        },
        seed=SEED,
    )
    flat: dict[str, Any] = {}
    for h, r in res.items():
        for k2, v in r.items():
            flat[f"{h}_{k2}"] = v
    rec.metrics = {
        **flat,
        **{f"desc_imb1_{h}_auc": r["auc"] for h, r in res_desc_l1.items()},
        **{f"replay_{k}": v for k, v in rstats.items()},
        **{f"desc_{k}": v for k, v in desc.items()},
        "capture_parquet_mb_per_hour": cap_rep.get("parquet_mb_per_hour"),
        "capture_ws_json_mb_per_hour": cap_rep.get("ws_json_mb_per_hour"),
        "capture_gaps_detected": cap_rep.get("gaps_detected"),
        "capture_resyncs": cap_rep.get("resyncs"),
        "verdict_reason": reason,
    }
    rec.conclusion = verdict  # type: ignore[assignment]
    rec.notes = (
        "Registered expectation: weak test - one ~5 minute capture, overlapping horizons, one regime. "
        "AUC > 0.5 would not imply a tradable edge after taker fees and spread. L2 only, single venue. "
        "imbalance_1 AUC and OFI correlation are descriptive extras, not part of the registered decision."
    )
    rec.reproduce = f"uv run --no-sync python -m nebula.research.microstructure_study capture {run_dir}"
    if not states.empty:
        per_s = (
            states.assign(sec=pd.to_datetime(states["E_ms"] // 1000 * 1000, unit="ms", utc=True))
            .groupby("sec")
            .agg(
                mid=("mid", "last"),
                spread_bps=("spread_bps", "mean"),
                imbalance_5=("imbalance_5", "mean"),
                microprice=("microprice", "last"),
                ofi=("ofi", "sum"),
                updates=("mid", "size"),
            )
            .reset_index()
        )
        tr.save_frame(rec, "book_1s_summary", per_s)
    tr.save_frame(rec, "h_ms_001_results", pd.DataFrame([{"horizon": h, **r} for h, r in res.items()]))
    tr.log(rec)
    return rec


# ================================================================== H-MS-002
# Registered in docs/research/H-EXT.md. Operational details below were fixed in code BEFORE any
# H-MS-002 number was computed (the registry entry is short; these follow H-MS-001 definitions):
#  - primary sample: the six 1-hour main runs; the ~60 s disk-budget probes are a labelled sensitivity
#    pool only; the 5-minute H-MS-001 capture is excluded (it already served as evidence for H-MS-001).
#  - capture valid iff replay integrity_failures == 0, verify mismatches == 0, unsynced events < 5%.
#  - future mid as in H-MS-001, never across a (re)sync segment boundary.
#  - moving-block bootstrap, 30 s blocks that never cross a segment boundary, 2000 resamples, seed 0;
#    >= 30 blocks required. Verdict rule = H-MS-001 (supported iff CI lo > 0.5 at both horizons).
#  - descriptive taker test: non-overlapping 10 s grid per segment; at each grid point trade
#    sign(imbalance_5) (skip if 0), enter at the touch, exit 10 s later at the touch; cost per leg =
#    10 bps taker fee + half-spread (bps) at that time; gross = signed mid-to-mid return (bps).
H_MS_002 = (
    "Pooled over ~6 h of BTCUSDT L2 captures, top-5 book imbalance at t predicts the sign of the mid "
    "change over the next 1 s and 10 s (AUC 95% moving-block bootstrap CI lower bound > 0.5 at both)."
)
H_MS_002_MAIN_RUNS = (
    "20261124T190304Z",
    "20261124T200410Z",
    "20261124T210516Z",
    "20261124T220622Z",
    "20261124T230728Z",
    "20261125T000833Z",
)
H_MS_002_PROBES = (
    "20261124T190202Z",
    "20261124T200307Z",
    "20261124T210413Z",
    "20261124T220520Z",
    "20261124T230625Z",
    "20261125T000731Z",
)
MIN_BLOCKS_H2 = 30
TAKER_FEE_BPS = 10.0
TRADE_H_S = 10
PROBE_N_BOOT = 200


def segment_block_bootstrap_auc(
    seg: np.ndarray,
    t_s: np.ndarray,
    score: np.ndarray,
    label: np.ndarray,
    block_s: float = BLOCK_S,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    alpha: float = 0.05,
) -> tuple[float, float, int]:
    """Moving-block bootstrap over several disjoint time segments; blocks never cross a segment.

    Each resample draws n_blocks = sum_s floor(span_s / block_s) block starts: a segment chosen with
    probability proportional to its admissible start range, then a uniform start inside it.
    Returns (lo, hi, n_blocks).
    """
    order = np.lexsort((t_s, seg))
    seg, t_s, score, label = seg[order], t_s[order], score[order], label[order]
    if len(seg) == 0:
        return float("nan"), float("nan"), 0
    segs, first = np.unique(seg, return_index=True)
    bounds = list(zip(first, [*first[1:], len(seg)], strict=True))
    t0 = np.array([t_s[a] for a, _ in bounds])
    span = np.array([t_s[b - 1] - t_s[a] for a, b in bounds])
    n_blocks = int(np.sum(np.floor(span / block_s)))
    room = np.clip(span - block_s, 0.0, None)
    if n_blocks < 1 or room.sum() <= 0:
        return float("nan"), float("nan"), n_blocks
    p = room / room.sum()
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        which = rng.choice(len(segs), size=n_blocks, p=p)
        starts = t0[which] + rng.uniform(0.0, 1.0, size=n_blocks) * room[which]
        parts: list[np.ndarray] = []
        for j in np.unique(which):
            a, z = bounds[j]
            ts = t_s[a:z]
            st = starts[which == j]
            lo_i = a + np.searchsorted(ts, st, side="left")
            hi_i = a + np.searchsorted(ts, st + block_s, side="left")
            parts.extend(np.arange(x, y) for x, y in zip(lo_i, hi_i, strict=True))
        idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
        out[b] = auc(score[idx], label[idx]) if len(idx) else np.nan
    lo, hi = np.nanquantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), n_blocks


def horizon_samples(states: pd.DataFrame, horizon_s: float, score_col: str = "imbalance_5") -> pd.DataFrame:
    """Per-state (segment, t_s, score, dmid) for one horizon; states without a full horizon are dropped."""
    dm = future_mid_change(states, horizon_s)
    ok = dm.notna() & states[score_col].notna()
    return pd.DataFrame(
        {
            "segment": states.loc[ok, "segment"].to_numpy(),
            "t_s": states.loc[ok, "E_ms"].to_numpy(dtype=float) / 1000.0,
            "score": states.loc[ok, score_col].to_numpy(dtype=float),
            "dmid": dm[ok].to_numpy(),
        }
    )


def auc_test(samples: pd.DataFrame, n_boot: int = N_BOOT) -> dict[str, Any]:
    """AUC on non-zero changes + segment-respecting block-bootstrap CI + zero-change base rate."""
    d = samples["dmid"].to_numpy()
    nz = d != 0
    s = samples[nz]
    label = (s["dmid"].to_numpy() > 0).astype(int)
    a = auc(s["score"].to_numpy(), label)
    lo, hi, nb = segment_block_bootstrap_auc(
        s["segment"].to_numpy(), s["t_s"].to_numpy(), s["score"].to_numpy(), label, n_boot=n_boot
    )
    return {
        "n_obs": len(samples),
        "n_nonzero": int(nz.sum()),
        "zero_change_rate": float(1 - nz.mean()) if len(samples) else float("nan"),
        "up_rate_given_change": float(label.mean()) if len(label) else float("nan"),
        "auc": a,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_blocks": nb,
    }


def _component_verdict(r: dict[str, Any]) -> str:
    if not r or r["n_blocks"] < MIN_BLOCKS_H2 or r["n_nonzero"] < MIN_NONZERO:
        return "inconclusive"
    if not (math.isfinite(r["ci_lo"]) and math.isfinite(r["ci_hi"])):
        return "inconclusive"
    if r["ci_lo"] > 0.5:
        return "supported"
    if r["ci_hi"] < 0.5:
        return "rejected"
    return "inconclusive"


def h_ms_002_verdict(res: dict[str, dict[str, Any]]) -> tuple[str, str]:
    hs = [res.get(f"h{h}s", {}) for h in HORIZONS_S]
    if any(not r or r["n_blocks"] < MIN_BLOCKS_H2 or r["n_nonzero"] < MIN_NONZERO for r in hs):
        return "inconclusive", f"insufficient data (< {MIN_BLOCKS_H2} blocks or < {MIN_NONZERO} changes)"
    comp = [_component_verdict(r) for r in hs]
    if all(c == "supported" for c in comp):
        return "supported", "CI lower bound > 0.5 at both horizons"
    if all(c == "rejected" for c in comp):
        return "rejected", "CI upper bound < 0.5 at both horizons"
    return "inconclusive", "CI includes 0.5 at one or both horizons, or horizons disagree"


def taker_strategy(
    states: pd.DataFrame, horizon_s: float = TRADE_H_S, fee_bps: float = TAKER_FEE_BPS
) -> pd.DataFrame:
    """Descriptive taker test on a non-overlapping grid per segment. One row per trade.

    gross = side * (mid_exit / mid_entry - 1) in bps; cost = 2 * fee + half-spread at entry + at exit.
    """
    rows: list[dict[str, Any]] = []
    h_ms = round(horizon_s * 1000)
    for seg, g in states.groupby("segment", sort=False):
        e = g["E_ms"].to_numpy(dtype=np.int64)
        mid = g["mid"].to_numpy(dtype=float)
        sp = g["spread_bps"].to_numpy(dtype=float)
        imb = g["imbalance_5"].to_numpy(dtype=float)
        if len(e) < 2 or e[-1] - e[0] < h_ms:
            continue
        grid = np.arange(e[0], e[-1] - h_ms + 1, h_ms)
        i0 = np.searchsorted(e, grid, side="right") - 1
        i1 = np.searchsorted(e, grid + h_ms, side="right") - 1
        for t, a, z in zip(grid, i0, i1, strict=True):
            sgn = float(np.sign(imb[a]))
            if sgn == 0 or not np.isfinite(sgn):
                continue
            move = (mid[z] / mid[a] - 1) * 1e4
            cost = 2 * fee_bps + sp[a] / 2 + sp[z] / 2
            rows.append(
                {
                    "segment": seg,
                    "t_ms": int(t),
                    "side": sgn,
                    "gross_bps": sgn * move,
                    "cost_bps": cost,
                    "net_bps": sgn * move - cost,
                    "abs_move_bps": abs(move),
                }
            )
    return pd.DataFrame(rows)


def ofi_association(states: pd.DataFrame) -> dict[str, float]:
    """Contemporaneous 1 s OFI (sum of e_n) vs 1 s mid change (USD), consecutive seconds in a segment."""
    import statsmodels.api as sm

    s = states.assign(sec=(states["E_ms"] // 1000).astype("int64"))
    per = s.groupby(["segment", "sec"]).agg(ofi=("ofi", "sum"), mid=("mid", "last")).reset_index()
    per["dmid"] = per.groupby("segment")["mid"].diff()
    per["adj"] = per.groupby("segment")["sec"].diff() == 1
    d = per[per["adj"] & per["dmid"].notna()]
    if len(d) < 30:
        return {
            "n_seconds": float(len(d)),
            "corr": float("nan"),
            "slope": float("nan"),
            "t_hac": float("nan"),
        }
    y, x = d["dmid"].to_numpy(), d["ofi"].to_numpy()
    lags = math.floor(4 * (len(d) / 100) ** (2 / 9))
    res = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {
        "n_seconds": float(len(d)),
        "corr": float(np.corrcoef(x, y)[0, 1]),
        "slope": float(np.asarray(res.params)[1]),  # USD mid change per BTC of OFI
        "t_hac": float(np.asarray(res.tvalues)[1]),
        "r2": float(res.rsquared),
    }


def _capture_row(run_dir: Path, pool: str, seg_offset: int) -> tuple[dict[str, Any], pd.DataFrame]:
    cap = json.loads((run_dir / "capture_report.json").read_text(encoding="utf-8"))
    states, rstats = book_states(run_dir)
    unsynced_frac = rstats["unsynced_events"] / max(rstats["events"], 1)
    valid = (
        not states.empty
        and rstats["integrity_failures"] == 0
        and rstats["verify_mismatches"] == 0
        and unsynced_frac < 0.05
    )
    row: dict[str, Any] = {
        "run": run_dir.name,
        "pool": pool,
        "valid": bool(valid),
        "hours": cap.get("actual_duration_s", 0) / 3600,
        "reconnects": cap.get("reconnects"),
        "online_gaps": cap.get("gaps_detected"),
        "online_resyncs": cap.get("resyncs"),
        "online_sync_snapshots": cap.get("snapshots_sync"),
        "unsynced_frac": unsynced_frac,
        **{f"replay_{k}": v for k, v in rstats.items()},
    }
    if not states.empty:
        states = states.assign(segment=states["segment"] + seg_offset)
        row["segments"] = int(states["segment"].nunique())
        row["valid_span_h"] = float(
            states.groupby("segment")["E_ms"].agg(lambda e: (e.max() - e.min()) / 3.6e6).sum()
        )
        row["max_intra_segment_gap_s"] = float(states.groupby("segment")["E_ms"].diff().max() / 1000)
        row["spread_bps_median"] = float(states["spread_bps"].median())
        row["share_one_tick"] = float(np.isclose(states["best_ask"] - states["best_bid"], 0.01).mean())
    return row, states


def run_h_ms_002(symbol: str = "BTCUSDT", n_boot: int = N_BOOT) -> dict[str, Any]:
    """Run H-MS-002 (docs/research/H-EXT.md); log Tracker records (one per horizon + family)."""
    import time as _time

    from nebula.stats.inference import newey_west_tstat

    t_start = _time.time()
    paths = get_paths()
    base = paths.data / "microstructure" / symbol
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[pd.DataFrame]] = {f"h{h}s": [] for h in HORIZONS_S}
    probe_pool: dict[str, list[pd.DataFrame]] = {f"h{h}s": [] for h in HORIZONS_S}
    trades: list[pd.DataFrame] = []
    seg_offset = 0
    for pool, runs in (("main", H_MS_002_MAIN_RUNS), ("probe", H_MS_002_PROBES)):
        for run in runs:
            rd = base / run
            if not (rd / "capture_report.json").exists():
                rows.append({"run": run, "pool": pool, "valid": False, "note": "missing"})
                continue
            row, st = _capture_row(rd, pool, seg_offset)
            if not st.empty:
                seg_offset = int(st["segment"].max()) + 1
                for h in HORIZONS_S:
                    smp = horizon_samples(st, h)
                    r = auc_test(smp, n_boot=n_boot if pool == "main" else PROBE_N_BOOT)
                    row.update({f"h{h}s_{k}": v for k, v in r.items()})
                    if row["valid"]:
                        (pooled if pool == "main" else probe_pool)[f"h{h}s"].append(smp)
                row.update({f"ofi_{k}": v for k, v in ofi_association(st).items()})
                if row["valid"] and pool == "main":
                    tdf_run = taker_strategy(st)
                    if not tdf_run.empty:
                        trades.append(tdf_run.assign(run=run))
            rows.append(row)
            log_row = {k: v for k, v in row.items() if k in ("run", "valid", "h1s_auc", "h10s_auc")}
            print("h_ms_002.capture", log_row, flush=True)
    res = {k: auc_test(pd.concat(v, ignore_index=True), n_boot=n_boot) for k, v in pooled.items() if v}
    sens = {
        k: auc_test(pd.concat(pooled[k] + probe_pool[k], ignore_index=True), n_boot=n_boot)
        for k in pooled
        if pooled[k] and probe_pool[k]
    }
    verdict, reason = h_ms_002_verdict(res)
    tdf = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    strat: dict[str, Any] = {}
    if not tdf.empty:
        strat = {
            "n_trades": len(tdf),
            "gross_mean_bps": float(tdf["gross_bps"].mean()),
            "gross_nw_t": newey_west_tstat(tdf["gross_bps"]),
            "hit_rate_gross": float((tdf["gross_bps"] > 0).mean()),
            "cost_mean_bps": float(tdf["cost_bps"].mean()),
            "net_mean_bps": float(tdf["net_bps"].mean()),
            "net_nw_t": newey_west_tstat(tdf["net_bps"]),
            "share_trades_net_positive": float((tdf["net_bps"] > 0).mean()),
            "mean_abs_move_10s_bps": float(tdf["abs_move_bps"].mean()),
            "p99_abs_move_10s_bps": float(tdf["abs_move_bps"].quantile(0.99)),
            "survives_costs": bool(tdf["net_bps"].mean() > 0),
        }
    elapsed = _time.time() - t_start
    pc = pd.DataFrame(rows)
    tr = Tracker()
    datasets = [
        {
            "provider": "binance_ws_l2_live_capture",
            "data_level": "L2 market-by-price (no L3)",
            "root": _rel(base, paths.root),
            "runs_main": list(H_MS_002_MAIN_RUNS),
            "runs_probe_sensitivity_only": list(H_MS_002_PROBES),
        }
    ]
    validation = {
        "metric": "ROC AUC on non-zero mid changes",
        "ci": f"segment-respecting moving-block bootstrap, block={BLOCK_S}s, n_boot={n_boot}, seed={SEED}",
        "min_blocks": MIN_BLOCKS_H2,
        "registry": "docs/research/H-EXT.md H-MS-002",
    }
    ids: dict[str, str] = {}
    for h in HORIZONS_S:
        key = f"h{h}s"
        rec = tr.new(
            name=f"H-MS-002 component {h}s: imbalance_5 -> sign dmid, pooled {symbol}",
            hypothesis_id=f"H-MS-002-{h}s",
            hypothesis=f"{H_MS_002} Component: horizon {h} s.",
            datasets=datasets,
            features=["imbalance_5"],
            validation=validation,
            seed=SEED,
        )
        rec.metrics = {**res.get(key, {}), "sensitivity_with_probes": sens.get(key, {})}
        rec.conclusion = _component_verdict(res.get(key, {}))  # type: ignore[assignment]
        rec.notes = "Component record; the family record carries the registered both-horizon decision."
        keep = [c for c in pc.columns if not c.startswith("h") or c.startswith(f"{key}_") or c == "hours"]
        tr.save_frame(rec, "per_capture", pc[keep])
        tr.log(rec)
        ids[key] = rec.experiment_id
    fam = tr.new(
        name=f"H-MS-002 family: L2 imbalance -> mid sign, pooled {symbol}",
        hypothesis_id="H-MS-002",
        hypothesis=H_MS_002,
        datasets=datasets,
        features=["imbalance_5"],
        model={"type": "rank score (no fit)"},
        validation=validation,
        costs={"taker_fee_bps_per_leg": TAKER_FEE_BPS, "half_spread": "per leg, at entry and exit"},
        seed=SEED,
    )
    fam.metrics = {
        **{f"{k}_{k2}": v for k, r in res.items() for k2, v in r.items()},
        **{f"probes_sens_{k}_{k2}": v for k, r in sens.items() for k2, v in r.items()},
        **{f"strategy10s_{k}": v for k, v in strat.items()},
        "component_ids": ids,
        "verdict_reason": reason,
        "runtime_s": elapsed,
    }
    fam.conclusion = verdict  # type: ignore[assignment]
    fam.notes = (
        "L2 only, single venue, one night (main runs 19:03-01:08 UTC). Every resync followed a WebSocket "
        "reconnect; post-resync data come from a freshly synced book and are kept, but no horizon window, "
        "bootstrap block or strategy trade crosses a segment boundary. Taker test is descriptive."
    )
    fam.reproduce = (
        'uv run --no-sync python -c "from nebula.research.microstructure_study import run_h_ms_002; '
        'run_h_ms_002()"'
    )
    tr.save_frame(fam, "per_capture", pc)
    pooled_df = pd.DataFrame(
        [{"horizon": k, "pool": "main", **r} for k, r in res.items()]
        + [{"horizon": k, "pool": "main+probes", **r} for k, r in sens.items()]
    )
    tr.save_frame(fam, "pooled_results", pooled_df)
    if not tdf.empty:
        by_run = (
            tdf.groupby("run")
            .agg(
                n=("net_bps", "size"),
                gross_mean_bps=("gross_bps", "mean"),
                net_mean_bps=("net_bps", "mean"),
                mean_abs_move_bps=("abs_move_bps", "mean"),
            )
            .reset_index()
        )
        tr.save_frame(fam, "strategy10s_by_run", by_run)
    tr.log(fam)
    ids["family"] = fam.experiment_id
    return {
        "ids": ids,
        "pooled": res,
        "sensitivity": sens,
        "strategy": strat,
        "per_capture": pc,
        "verdict": verdict,
        "reason": reason,
        "runtime_s": elapsed,
    }


def main(argv: list[str] | None = None) -> int:
    import sys

    args = sys.argv[1:] if argv is None else argv
    what = args[0] if args else "all"
    if what in ("aggtrades", "all"):
        rec = run_aggtrades_study()
        print(rec.experiment_id, rec.conclusion, json.dumps(rec.metrics, default=str, indent=1))
    if what == "h-ms-002":
        out = run_h_ms_002()
        print(out["ids"], out["verdict"], out["reason"], json.dumps(out["pooled"], default=str))
        print(json.dumps(out["strategy"], default=str), "runtime_s", out["runtime_s"])
    if what in ("capture", "all"):
        rd = Path(args[1]) if what == "capture" and len(args) > 1 else None
        rec = run_capture_study(rd)
        print(rec.experiment_id, rec.conclusion, json.dumps(rec.metrics, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
