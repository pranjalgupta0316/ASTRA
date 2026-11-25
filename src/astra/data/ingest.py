"""Ingestion pipeline: fetch -> raw store -> normalize -> validate -> validated store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from astra.common.config import AstraConfig, get_config
from astra.common.logging import get_logger
from astra.common.paths import Paths, get_paths
from astra.data.adapters import get_adapter
from astra.data.storage import raw_dir, write_raw, write_validated
from astra.data.validation import QualityReport, validate_bars

log = get_logger(__name__)


@dataclass
class IngestResult:
    symbol: str
    raw_file: Path
    validated_file: Path | None
    report: QualityReport


def ingest_symbol(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    config: AstraConfig | None = None,
    paths: Paths | None = None,
    allow_warn: bool = True,
    freq: str = "1d",
) -> IngestResult:
    config = config or get_config()
    paths = (paths or get_paths()).ensure()
    inst = config.instrument(symbol)
    market = config.markets[inst.market]
    adapter = get_adapter(inst.provider)
    start = start or config.start_for(symbol)
    files_dir = raw_dir(paths, inst.provider, inst.provider_symbol, freq) / "files"
    if freq == "1d":
        batch = adapter.fetch_daily(inst, start, end, files_dir)
        raw_file = write_raw(batch, paths)
        bars = adapter.normalize_daily(batch, market)
    else:
        fetch, normalize = getattr(adapter, "fetch", None), getattr(adapter, "normalize", None)
        if fetch is None or normalize is None:
            raise ValueError(f"provider {inst.provider} has no intraday history (only Binance does)")
        batch = fetch(inst, start, end, files_dir, interval=freq)
        raw_file = write_raw(batch, paths)
        bars = normalize(batch, market)
    report = validate_bars(bars, market, freq=freq)
    log.info("ingest.validated", symbol=symbol, freq=freq, status=report.status, rows=report.rows)
    ok = report.status == "pass" or (allow_warn and report.status == "warn")
    validated = write_validated(bars, report, raw_file, paths, freq=freq, notes=batch.notes) if ok else None
    if not ok:
        from astra.data.storage import write_report

        write_report(report, paths)
    return IngestResult(symbol=symbol, raw_file=raw_file, validated_file=validated, report=report)
