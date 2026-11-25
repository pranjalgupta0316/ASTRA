"""Layered Parquet storage (raw -> validated) with provenance manifests and DuckDB access."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astra.common.paths import Paths, get_paths
from astra.data.adapters.base import RawBatch
from astra.data.contracts import BAR_SCHEMA
from astra.data.validation import QualityReport


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_dir(paths: Paths, provider: str, symbol: str, freq: str = "1d") -> Path:
    return paths.raw / provider / symbol / freq


def write_raw(batch: RawBatch, paths: Paths | None = None) -> Path:
    """Persist the provider frame verbatim (immutable, timestamped) plus a metadata sidecar."""
    paths = paths or get_paths()
    d = raw_dir(paths, batch.provider, batch.instrument.provider_symbol, batch.request.get("interval", "1d"))
    d.mkdir(parents=True, exist_ok=True)
    stamp = batch.retrieved_at_utc.strftime("%Y%m%dT%H%M%SZ")
    out = d / f"raw_{stamp}.parquet"
    frame = batch.frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), out, compression="zstd")
    meta = {
        "provider": batch.provider,
        "instrument": batch.instrument.model_dump(),
        "retrieved_at_utc": batch.retrieved_at_utc.isoformat(),
        "request": batch.request,
        "rows": len(frame),
        "sha256": sha256_file(out),
        "source_files": [
            {"path": str(p.relative_to(paths.root)), "sha256": sha256_file(p)} for p in batch.files
        ],
        "notes": batch.notes,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def validated_path(paths: Paths, market: str, symbol: str, freq: str = "1d") -> Path:
    return paths.validated / f"market={market}" / f"symbol={symbol}" / f"freq={freq}" / "bars.parquet"


def write_validated(
    df: pd.DataFrame,
    report: QualityReport,
    raw_file: Path,
    paths: Paths | None = None,
    freq: str = "1d",
    notes: list[str] | None = None,
) -> Path:
    paths = paths or get_paths()
    sym, mkt = str(df["symbol"].iloc[0]), str(df["market"].iloc[0])
    out = validated_path(paths, mkt, sym, freq)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=BAR_SCHEMA, preserve_index=False)
    pq.write_table(table, out, compression="zstd")
    report_path = write_report(report, paths)
    manifest: dict[str, Any] = {
        "symbol": sym,
        "market": mkt,
        "freq": freq,
        "rows": len(df),
        "first_session": report.first_session,
        "last_session": report.last_session,
        "quality_status": report.status,
        "quality_report": str(report_path.relative_to(paths.root)),
        "raw_file": str(raw_file.relative_to(paths.root)),
        "raw_sha256": sha256_file(raw_file),
        "sha256": sha256_file(out),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": 1,
        "transform_notes": notes or [],
    }
    out.with_name("manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def write_report(report: QualityReport, paths: Paths | None = None) -> Path:
    paths = paths or get_paths()
    d = paths.reports / "quality"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(report.created_at_utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "" if report.freq == "1d" else f"_{report.freq}"
    p = d / f"{report.market}_{report.symbol}{suffix}_{stamp}.json"
    p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return p


def read_bars(
    symbol: str,
    market: str,
    paths: Paths | None = None,
    freq: str = "1d",
    window: tuple[str | None, str | None] | None = None,
) -> pd.DataFrame:
    """Validated bars; `window=(start, end)` (inclusive session dates) restricts the sample, e.g. the
    frozen sample of a registered study (see AstraConfig.core_window)."""
    paths = paths or get_paths()
    p = validated_path(paths, market, symbol, freq)
    if not p.exists():
        raise FileNotFoundError(f"No validated data for {market}/{symbol}; run `astra ingest` first ({p})")
    df = pq.read_table(p).to_pandas()
    if window is not None:
        d = pd.to_datetime(df["session_date"])
        start, end = window
        keep = pd.Series(True, index=df.index)
        if start:
            keep &= d >= pd.Timestamp(start)
        if end:
            keep &= d <= pd.Timestamp(end)
        df = df.loc[keep].reset_index(drop=True)
    return df


def read_manifest(symbol: str, market: str, paths: Paths | None = None, freq: str = "1d") -> dict[str, Any]:
    paths = paths or get_paths()
    p = validated_path(paths, market, symbol, freq).with_name("manifest.json")
    return json.loads(p.read_text(encoding="utf-8"))


def query(sql: str, paths: Paths | None = None) -> pd.DataFrame:
    """Run DuckDB SQL; `bars` is a view over all validated daily bars (hive-partitioned)."""
    paths = paths or get_paths()
    con = duckdb.connect()
    glob = (paths.validated / "**" / "bars.parquet").as_posix()
    con.execute(f"CREATE VIEW bars AS SELECT * FROM read_parquet('{glob}', hive_partitioning=false)")
    return con.execute(sql).df()
