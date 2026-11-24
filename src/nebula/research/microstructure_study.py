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


def main(argv: list[str] | None = None) -> int:
    import sys

    args = sys.argv[1:] if argv is None else argv
    what = args[0] if args else "all"
    if what in ("aggtrades", "all"):
        rec = run_aggtrades_study()
        print(rec.experiment_id, rec.conclusion, json.dumps(rec.metrics, default=str, indent=1))
    if what in ("capture", "all"):
        rd = Path(args[1]) if what == "capture" and len(args) > 1 else None
        rec = run_capture_study(rd)
        print(rec.experiment_id, rec.conclusion, json.dumps(rec.metrics, default=str, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
