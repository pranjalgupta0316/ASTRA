"""Binance Public Data (https://data.binance.vision) spot klines.

Files are static zips with a published SHA-256 CHECKSUM. Monthly files cover complete months; daily
files cover the current month. Spot kline timestamps are milliseconds up to 2024-12-31 and
microseconds from 2025-01-01 - detected per row by magnitude.
"""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from nebula.common.config import InstrumentConfig, MarketConfig
from nebula.common.logging import get_logger
from nebula.data.adapters.base import RawBatch
from nebula.data.contracts import BAR_COLUMNS
from nebula.data.download import checked_download

log = get_logger(__name__)

BASE = "https://data.binance.vision/data/spot"
KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]
USER_AGENT = "nebula-research/0.1 (+local research; respects rate limits)"


def _to_utc(ts: pd.Series) -> pd.Series:
    """Convert Binance epoch integers (ms or us) to UTC timestamps."""
    v = ts.astype("int64")
    us = v.where(v > 10**14, v * 1000)  # values below 1e14 are milliseconds
    return pd.to_datetime(us, unit="us", utc=True)


def _months(start: date, end_exclusive: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while date(y, m, 1) < end_exclusive:
        out.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


INTERVALS: dict[str, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
}


class BinanceVisionAdapter:
    name = "binance_vision"

    def __init__(self, session: requests.Session | None = None) -> None:
        self.http = session or requests.Session()
        self.http.headers["User-Agent"] = USER_AGENT

    def _download(self, url: str, dest: Path) -> Path | None:
        """Checksummed streamed download (cache re-verified). Returns None on 404 (not published)."""
        return checked_download(self.http, url, dest)

    def fetch_daily(
        self, instrument: InstrumentConfig, start: str, end: str | None, raw_dir: Path
    ) -> RawBatch:
        return self.fetch(instrument, start, end, raw_dir, interval="1d")

    def normalize_daily(self, batch: RawBatch, market: MarketConfig) -> pd.DataFrame:
        return self.normalize(batch, market)

    def fetch(
        self, instrument: InstrumentConfig, start: str, end: str | None, raw_dir: Path, interval: str = "1d"
    ) -> RawBatch:
        """Klines at `interval` (e.g. 1d, 1h, 1m): complete monthly files, then daily files for the
        current month up to yesterday (today's bars may be incomplete and are never fetched)."""
        if interval not in INTERVALS:
            raise ValueError(f"unsupported interval {interval!r}; use one of {sorted(INTERVALS)}")
        sym = instrument.provider_symbol
        retrieved = datetime.now(UTC)
        start_d = pd.Timestamp(start).date()
        today = retrieved.date()
        end_d = min(pd.Timestamp(end).date(), today) if end else today
        month_start = end_d.replace(day=1)
        files: list[Path] = []
        notes: list[str] = []
        for ym in _months(start_d, month_start):
            name = f"{sym}-{interval}-{ym}.zip"
            p = self._download(f"{BASE}/monthly/klines/{sym}/{interval}/{name}", raw_dir / "monthly" / name)
            if p is None:
                notes.append(f"monthly file not published: {name}")
            else:
                files.append(p)
        d = max(month_start, start_d)
        while d < end_d:  # daily files for the current month; today's bar is incomplete -> excluded
            name = f"{sym}-{interval}-{d.isoformat()}.zip"
            p = self._download(f"{BASE}/daily/klines/{sym}/{interval}/{name}", raw_dir / "daily" / name)
            if p is None:
                notes.append(f"daily file not yet published: {name}")
            else:
                files.append(p)
            d += timedelta(days=1)
        frames = []
        for f in files:
            with zipfile.ZipFile(f) as zf:
                inner = zf.namelist()[0]
                text = zf.read(inner).decode()
            first = text.splitlines()[0] if text else ""
            header = 0 if first.startswith("open_time") else None
            part = pd.read_csv(io.StringIO(text), header=header, names=None if header == 0 else KLINE_COLS)
            part.columns = KLINE_COLS
            frames.append(part)
        if not frames:
            raise RuntimeError(f"No Binance files downloaded for {sym}")
        df = pd.concat(frames, ignore_index=True)
        log.info("binance.fetch", symbol=sym, files=len(files), rows=len(df))
        return RawBatch(
            provider=self.name,
            instrument=instrument,
            retrieved_at_utc=retrieved,
            frame=df,
            files=files,
            request={"symbol": sym, "start": start, "end": str(end_d), "interval": interval, "base": BASE},
            notes=notes,
        )

    def normalize(self, batch: RawBatch, market: MarketConfig) -> pd.DataFrame:
        raw = batch.frame
        inst = batch.instrument
        step = INTERVALS[batch.request.get("interval", "1d")]
        open_utc = _to_utc(raw["open_time"])
        close_utc = open_utc + step  # Binance close_time is open+step-1unit; use the boundary
        out = pd.DataFrame(
            {
                "session_date": open_utc.dt.date,
                "open_time_utc": open_utc,
                "close_time_utc": close_utc,
                "symbol": inst.symbol,
                "provider_symbol": inst.provider_symbol,
                "market": inst.market,
                "exchange": inst.exchange,
                "exchange_tz": market.exchange_tz,
                "currency": market.currency,
                "open": raw["open"].astype(float),
                "high": raw["high"].astype(float),
                "low": raw["low"].astype(float),
                "close": raw["close"].astype(float),
                "adj_close": raw["close"].astype(float),
                "volume": raw["volume"].astype(float),
                "quote_volume": raw["quote_volume"].astype(float),
                "trade_count": raw["trade_count"].astype(float),
                "dividend": 0.0,
                "split_ratio": 1.0,
                "provider": self.name,
                "ingested_at_utc": pd.Timestamp(batch.retrieved_at_utc),
            }
        )
        return out[BAR_COLUMNS].sort_values("close_time_utc").reset_index(drop=True)
