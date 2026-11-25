"""Adapter protocol: fetch raw data (stored verbatim) and normalize it to the canonical bar contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

import pandas as pd

from astra.common.config import InstrumentConfig, MarketConfig


@dataclass
class RawBatch:
    """What a provider returned, before normalization."""

    provider: str
    instrument: InstrumentConfig
    retrieved_at_utc: datetime
    frame: pd.DataFrame
    files: list[Path] = field(default_factory=list)  # raw files on disk backing this batch
    request: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


class Adapter(Protocol):
    name: str

    def fetch_daily(
        self, instrument: InstrumentConfig, start: str, end: str | None, raw_dir: Path
    ) -> RawBatch: ...

    def normalize_daily(self, batch: RawBatch, market: MarketConfig) -> pd.DataFrame: ...


def get_adapter(provider: str) -> Adapter:
    if provider == "yahoo":
        from astra.data.adapters.yahoo import YahooAdapter

        return YahooAdapter()
    if provider == "binance_vision":
        from astra.data.adapters.binance import BinanceVisionAdapter

        return BinanceVisionAdapter()
    raise ValueError(f"Unknown provider {provider!r}")
