"""Golden regression of headline research numbers under a data lock.

Why: an engine or feature refactor can shift a published number silently. With the data pinned by a lock
(astra.data.lock), the vertical-slice numbers for a fixed set of instruments are fully deterministic, so a
stored copy ("golden file") can be compared exactly after every change. A deliberate change must update the
golden file in the same commit (scripts/update_golden.py), which makes the shift visible in review.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from astra.common.paths import get_paths

GOLDEN_SYMBOLS = ("AAPL", "RELIANCE", "BTCUSDT")  # one per market
GOLDEN_LOCK = "v0.3.0-baseline"


def golden_path(lock: str = GOLDEN_LOCK) -> Path:
    return get_paths().root / "tests" / "golden" / f"vertical_slice_{lock}.json"


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(d, bool | str) or d is None:
        out[prefix[:-1]] = d
    elif isinstance(d, int | float):
        out[prefix[:-1]] = None if isinstance(d, float) and math.isnan(d) else d
    return out  # lists and other containers are not headline numbers


def collect_headlines(symbols: tuple[str, ...] = GOLDEN_SYMBOLS, lock: str = GOLDEN_LOCK) -> dict[str, Any]:
    """Run the vertical slice for `symbols` under `lock`; records go to a throw-away experiments dir."""
    from astra.data.lock import use_lock
    from astra.research.vertical_slice import run_symbol
    from astra.tracking.tracker import Tracker

    prev = os.environ.get("ASTRA_EXPERIMENTS_DIR")
    with tempfile.TemporaryDirectory() as tmp, use_lock(lock):
        os.environ["ASTRA_EXPERIMENTS_DIR"] = tmp
        try:
            for s in symbols:
                run_symbol(s)
            tr = Tracker()
            out: dict[str, Any] = {}
            for eid in tr.list()["experiment_id"]:
                rec = tr.load(eid)
                key = (
                    f"{rec.datasets[0].get('symbol', '?')}/{rec.hypothesis_id}"
                    if rec.datasets
                    else rec.hypothesis_id
                )
                out[key] = {"conclusion": rec.conclusion, "metrics": _flatten(rec.metrics)}
        finally:
            if prev is None:
                os.environ.pop("ASTRA_EXPERIMENTS_DIR", None)
            else:
                os.environ["ASTRA_EXPERIMENTS_DIR"] = prev
    return dict(sorted(out.items()))


def compare(expected: dict[str, Any], actual: dict[str, Any], rtol: float = 1e-9) -> list[str]:
    """Human-readable list of differences (empty = identical within rtol)."""
    diffs: list[str] = []
    for key in sorted(set(expected) | set(actual)):
        if key not in actual or key not in expected:
            diffs.append(f"{key}: {'missing' if key not in actual else 'unexpected'} record")
            continue
        e, a = expected[key], actual[key]
        if e["conclusion"] != a["conclusion"]:
            diffs.append(f"{key}: conclusion {e['conclusion']} -> {a['conclusion']}")
        for m in sorted(set(e["metrics"]) | set(a["metrics"])):
            ev, av = e["metrics"].get(m, "<absent>"), a["metrics"].get(m, "<absent>")
            if isinstance(ev, int | float) and isinstance(av, int | float) and not isinstance(ev, bool):
                if not math.isclose(ev, av, rel_tol=rtol, abs_tol=1e-12):
                    diffs.append(f"{key}: {m} {ev!r} -> {av!r}")
            elif ev != av:
                diffs.append(f"{key}: {m} {ev!r} -> {av!r}")
    return diffs


def write_golden(data: dict[str, Any], lock: str = GOLDEN_LOCK) -> Path:
    p = golden_path(lock)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
        fh.write("\n")
    return p
