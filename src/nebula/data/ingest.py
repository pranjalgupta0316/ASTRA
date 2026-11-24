"""Ingestion pipeline: fetch -> raw store -> normalize -> validate -> validated store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nebula.common.config import NebulaConfig, get_config
from nebula.common.logging import get_logger
from nebula.common.paths import Paths, get_paths
from nebula.data.adapters import get_adapter
from nebula.data.storage import raw_dir, write_raw, write_validated
from nebula.data.validation import QualityReport, validate_bars

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
    config: NebulaConfig | None = None,
    paths: Paths | None = None,
    allow_warn: bool = True,
) -> IngestResult:
    config = config or get_config()
    paths = (paths or get_paths()).ensure()
    inst = config.instrument(symbol)
    market = config.markets[inst.market]
    adapter = get_adapter(inst.provider)
    start = start or config.defaults.get("start", "2018-01-01")
    batch = adapter.fetch_daily(
        inst, start, end, raw_dir(paths, inst.provider, inst.provider_symbol) / "files"
    )
    raw_file = write_raw(batch, paths)
    bars = adapter.normalize_daily(batch, market)
    report = validate_bars(bars, market)
    log.info("ingest.validated", symbol=symbol, status=report.status, rows=report.rows)
    ok = report.status == "pass" or (allow_warn and report.status == "warn")
    validated = write_validated(bars, report, raw_file, paths, notes=batch.notes) if ok else None
    if not ok:
        from nebula.data.storage import write_report

        write_report(report, paths)
    return IngestResult(symbol=symbol, raw_file=raw_file, validated_file=validated, report=report)
