"""Bounded live capture of Binance spot L2 diff depth + aggTrades.

Usage:
  uv run --no-sync python scripts/capture_depth.py --symbol BTCUSDT --duration 300
  uv run --no-sync python scripts/capture_depth.py --symbol BTCUSDT --duration 3600 --max-disk-mb 500

Captures longer than 10 minutes first run a 60 s probe, print the measured MB/hour and projected disk
use, and abort unless --max-disk-mb is given and the projection fits under it.
L2 only - Binance public streams carry no order IDs (no L3).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nebula.microstructure.capture import run_capture

MAX_DURATION_S = 24 * 3600
PROBE_THRESHOLD_S = 600
PROBE_S = 60


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--duration", type=float, default=300.0, help="capture seconds (hard stop)")
    ap.add_argument("--out-dir", type=Path, default=None, help="default: data/microstructure")
    ap.add_argument(
        "--max-disk-mb",
        type=float,
        default=500.0,
        help="hard cap on raw bytes received (disk use is always below it); default 500",
    )
    ap.add_argument("--verify-every", type=float, default=120.0, help="verification snapshot interval, s")
    a = ap.parse_args(argv)
    if not 0 < a.duration <= MAX_DURATION_S:
        ap.error(f"--duration must be in (0, {MAX_DURATION_S}] seconds")
    if a.max_disk_mb <= 0:
        ap.error("--max-disk-mb must be positive")

    if a.duration > PROBE_THRESHOLD_S:
        probe = run_capture(a.symbol, PROBE_S, a.out_dir, verify_every_s=0)
        projected_mb = probe.parquet_mb_per_hour * a.duration / 3600
        print(
            f"probe: {probe.parquet_mb_per_hour:.1f} MB/hour parquet "
            f"({probe.ws_json_mb_per_hour:.1f} MB/hour raw JSON); projected {projected_mb:.0f} MB "
            f"for {a.duration:.0f} s"
        )
        if projected_mb > a.max_disk_mb:
            print("aborting: pass --max-disk-mb above the projection to proceed", file=sys.stderr)
            return 2

    rep = run_capture(
        a.symbol, a.duration, a.out_dir, verify_every_s=a.verify_every, max_raw_mb=a.max_disk_mb
    )
    d = asdict(rep)
    print(json.dumps({k: v for k, v in d.items() if k != "errors"}, indent=2))
    if rep.errors:
        print(f"errors ({len(rep.errors)}): {rep.errors[:5]}", file=sys.stderr)
    print(
        f"MB/hour (parquet): {rep.parquet_mb_per_hour:.2f}; "
        f"projected GB/day: {rep.projected_gb_per_day:.2f}; "
        f"gaps: {rep.gaps_detected}; resyncs: {rep.resyncs}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
