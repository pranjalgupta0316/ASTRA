"""Typed loading of config/markets.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from nebula.common.paths import get_paths

MarketCode = Literal["US", "IN", "CRYPTO"]


class CostConfig(BaseModel):
    """Per-side costs in basis points."""

    commission: float = 0.0
    half_spread: float = 0.0
    slippage: float = 0.0
    buy_tax: float = 0.0
    sell_tax: float = 0.0

    def buy_bps(self) -> float:
        return self.commission + self.half_spread + self.slippage + self.buy_tax

    def sell_bps(self) -> float:
        return self.commission + self.half_spread + self.slippage + self.sell_tax


class MarketConfig(BaseModel):
    description: str
    calendar: str
    exchange_tz: str
    currency: str
    periods_per_year: int
    costs_bps: CostConfig
    start: str | None = None  # earliest history to ingest for this market

    @property
    def is_24_7(self) -> bool:
        return self.calendar == "24/7"


class InstrumentConfig(BaseModel):
    symbol: str
    market: MarketCode
    provider: Literal["yahoo", "binance_vision"]
    provider_symbol: str
    exchange: str
    # core = the 13 instruments behind every hypothesis registered before 2026-11-25 (their scope is
    # frozen); extended = the broader universe added for cross-sectional and large-panel studies.
    tier: Literal["core", "extended"] = "core"
    kind: Literal["stock", "etf", "index", "crypto"] = "stock"
    intraday: list[str] = Field(default_factory=list)  # extra bar frequencies to ingest, e.g. ["1h"]


class NebulaConfig(BaseModel):
    markets: dict[MarketCode, MarketConfig]
    instruments: list[InstrumentConfig]
    holdout: list[str] = Field(default_factory=list)
    defaults: dict[str, str] = Field(default_factory=dict)

    def symbols(
        self, tier: str | None = None, market: str | None = None, kind: str | None = None
    ) -> list[str]:
        return [
            i.symbol
            for i in self.instruments
            if (tier is None or i.tier == tier)
            and (market is None or i.market == market)
            and (kind is None or i.kind == kind)
        ]

    def core_window(self) -> tuple[str, str]:
        """Frozen sample of every hypothesis registered on the core universe (2026-11-24)."""
        return self.defaults.get("core_start", "2018-01-01"), self.defaults.get("core_end", "2026-11-23")

    def start_for(self, symbol: str) -> str:
        m = self.market_of(symbol)
        return m.start or self.defaults.get("start", "2018-01-01")

    def instrument(self, symbol: str) -> InstrumentConfig:
        for inst in self.instruments:
            if inst.symbol == symbol:
                return inst
        raise KeyError(f"Unknown instrument {symbol!r}; configured: {[i.symbol for i in self.instruments]}")

    def market_of(self, symbol: str) -> MarketConfig:
        return self.markets[self.instrument(symbol).market]


def load_config(path: Path | None = None) -> NebulaConfig:
    path = path or get_paths().config / "markets.yaml"
    with path.open(encoding="utf-8") as fh:
        return NebulaConfig.model_validate(yaml.safe_load(fh))


@lru_cache(maxsize=1)
def get_config() -> NebulaConfig:
    return load_config()
