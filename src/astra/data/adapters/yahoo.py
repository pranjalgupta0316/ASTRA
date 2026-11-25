"""Yahoo Finance daily bars via yfinance (unofficial; personal research use; no SLA).

Price semantics (verified against yfinance behaviour, see docs/DATA_SOURCES.md):
- with auto_adjust=False, Open/High/Low/Close are split-adjusted but not dividend-adjusted;
- 'Adj Close' is split- and dividend-adjusted;
- index is the session date at local midnight in the exchange timezone.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from astra.common.config import InstrumentConfig, MarketConfig
from astra.common.logging import get_logger
from astra.data.adapters.base import RawBatch
from astra.data.calendars import session_schedule
from astra.data.contracts import BAR_COLUMNS

log = get_logger(__name__)
MAX_ATTEMPTS = 4
SETTLE_BUFFER = pd.Timedelta(hours=2)
BACKOFF_S = 5.0


def _col(raw: pd.DataFrame, name: str) -> pd.Series:
    """Float column, or zeros when the provider omitted it (Yahoo drops action columns with no events)."""
    if name in raw.columns:
        return raw[name].astype(float)
    return pd.Series(0.0, index=raw.index)


class YahooAdapter:
    name = "yahoo"

    def fetch_daily(
        self, instrument: InstrumentConfig, start: str, end: str | None, raw_dir: Path
    ) -> RawBatch:
        import yfinance as yf

        retrieved = datetime.now(UTC)
        ticker = yf.Ticker(instrument.provider_symbol)
        df, last_err = None, None
        for attempt in range(MAX_ATTEMPTS):  # Yahoo throttles bursts; back off politely, never hammer
            try:
                df = ticker.history(start=start, end=end, interval="1d", auto_adjust=False, actions=True)
                if df is not None and not df.empty:
                    break
                last_err = "empty response"
            except Exception as exc:  # yfinance raises assorted network/rate-limit errors
                last_err = repr(exc)
            log.warning("yahoo.retry", symbol=instrument.provider_symbol, attempt=attempt + 1, error=last_err)
            time.sleep(BACKOFF_S * 2**attempt)
        if df is None or df.empty:
            raise RuntimeError(
                f"Yahoo returned no data for {instrument.provider_symbol} "
                f"after {MAX_ATTEMPTS} attempts: {last_err}"
            )
        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        log.info("yahoo.fetch", symbol=instrument.provider_symbol, rows=len(df))
        return RawBatch(
            provider=self.name,
            instrument=instrument,
            retrieved_at_utc=retrieved,
            frame=df,
            request={
                "symbol": instrument.provider_symbol,
                "start": start,
                "end": str(end),
                "interval": "1d",
                "auto_adjust": "False",
                "yfinance_version": getattr(yf, "__version__", "unknown"),
            },
        )

    def normalize_daily(self, batch: RawBatch, market: MarketConfig) -> pd.DataFrame:
        raw = batch.frame.copy()
        inst = batch.instrument
        idx = pd.DatetimeIndex(pd.to_datetime(raw["Date"]))
        if idx.tz is None:
            idx = idx.tz_localize(market.exchange_tz)
        local = idx.tz_convert(market.exchange_tz)
        session = pd.DatetimeIndex(local.normalize().tz_localize(None))

        sched = session_schedule(market, session.min(), session.max())
        open_utc = pd.Series(sched["open"].reindex(session).to_numpy(), dtype="datetime64[ns, UTC]")
        close_utc = pd.Series(sched["close"].reindex(session).to_numpy(), dtype="datetime64[ns, UTC]")
        # Off-calendar rows: conservative availability = end of local day (never earlier than reality).
        fallback_open = pd.Series(session.tz_localize(market.exchange_tz).tz_convert("UTC"))
        fallback_close = fallback_open + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        open_utc = open_utc.fillna(fallback_open)
        close_utc = close_utc.fillna(fallback_close)

        out = pd.DataFrame(
            {
                "session_date": session.date,
                "open_time_utc": open_utc.to_numpy(),
                "close_time_utc": close_utc.to_numpy(),
                "symbol": inst.symbol,
                "provider_symbol": inst.provider_symbol,
                "market": inst.market,
                "exchange": inst.exchange,
                "exchange_tz": market.exchange_tz,
                "currency": market.currency,
                "open": raw["Open"].astype(float).to_numpy(),
                "high": raw["High"].astype(float).to_numpy(),
                "low": raw["Low"].astype(float).to_numpy(),
                "close": raw["Close"].astype(float).to_numpy(),
                "adj_close": raw["Adj Close"].astype(float).to_numpy()
                if "Adj Close" in raw
                else float("nan"),
                "volume": raw["Volume"].astype(float).to_numpy(),
                "quote_volume": float("nan"),
                "trade_count": float("nan"),
                "dividend": _col(raw, "Dividends").to_numpy(),
                "split_ratio": _col(raw, "Stock Splits").replace(0.0, 1.0).to_numpy(),
                "provider": self.name,
                "ingested_at_utc": pd.Timestamp(batch.retrieved_at_utc),
            }
        )
        out["open_time_utc"] = pd.to_datetime(out["open_time_utc"], utc=True)
        out["close_time_utc"] = pd.to_datetime(out["close_time_utc"], utc=True)
        # Yahoo returns the in-progress session as a daily bar. A bar whose session had not closed when
        # we retrieved it is incomplete (and would leak a partial, later-revised close) -> drop it.
        # Settlement buffer: NSE's official close (30-min VWAP) is published after 15:30 IST, so a bar
        # fetched shortly after the close may still be provisional.
        incomplete = out["close_time_utc"] + SETTLE_BUFFER > pd.Timestamp(batch.retrieved_at_utc)
        if incomplete.any():
            dropped = [str(d) for d in out.loc[incomplete, "session_date"]]
            batch.notes.append(f"dropped incomplete in-progress session(s): {dropped}")
            log.info("yahoo.drop_incomplete", symbol=inst.provider_symbol, sessions=dropped)
            out = out.loc[~incomplete]
        return out[BAR_COLUMNS].sort_values("close_time_utc").reset_index(drop=True)
