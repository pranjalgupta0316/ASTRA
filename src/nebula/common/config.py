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

    @property
    def is_24_7(self) -> bool:
        return self.calendar == "24/7"


class InstrumentConfig(BaseModel):
    symbol: str
    market: MarketCode
    provider: Literal["yahoo", "binance_vision"]
    provider_symbol: str
    exchange: str


class NebulaConfig(BaseModel):
    markets: dict[MarketCode, MarketConfig]
    instruments: list[InstrumentConfig]
    holdout: list[str] = Field(default_factory=list)
    defaults: dict[str, str] = Field(default_factory=dict)

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
