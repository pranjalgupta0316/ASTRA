"""Data-quality validation for canonical daily bars.

Checks never modify data. They return a QualityReport; callers decide whether to proceed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from astra.common.config import MarketConfig
from astra.data.calendars import session_schedule
from astra.data.contracts import BAR_COLUMNS, PRICE_COLUMNS

Status = Literal["pass", "warn", "fail"]


class CheckResult(BaseModel):
    name: str
    status: Status
    count: int = 0
    details: str = ""
    examples: list[str] = Field(default_factory=list)


class QualityReport(BaseModel):
    symbol: str
    market: str
    provider: str
    freq: str = "1d"
    rows: int
    first_session: str | None
    last_session: str | None
    created_at_utc: str
    checks: list[CheckResult]

    @property
    def status(self) -> Status:
        statuses = {c.status for c in self.checks}
        if "fail" in statuses:
            return "fail"
        if "warn" in statuses:
            return "warn"
        return "pass"

    def summary(self) -> str:
        lines = [f"{self.symbol} [{self.market}/{self.provider}] rows={self.rows} status={self.status}"]
        lines += [f"  {c.status.upper():4s} {c.name}: {c.details}" for c in self.checks]
        return "\n".join(lines)


def _examples(values: pd.Index | pd.Series, n: int = 5) -> list[str]:
    return [str(v) for v in list(values)[:n]]


def _ok(name: str, details: str = "ok") -> CheckResult:
    return CheckResult(name=name, status="pass", details=details)


def check_schema(df: pd.DataFrame) -> CheckResult:
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in BAR_COLUMNS]
    if missing:
        return CheckResult(name="schema", status="fail", count=len(missing), details=f"missing {missing}")
    if extra:
        return CheckResult(name="schema", status="warn", count=len(extra), details=f"unexpected {extra}")
    return _ok("schema")


def check_nulls(df: pd.DataFrame) -> CheckResult:
    required = [*PRICE_COLUMNS, "volume", "session_date", "close_time_utc"]
    nulls = df[required].isna().sum()
    bad = nulls[nulls > 0]
    if len(bad):
        return CheckResult(
            name="nulls", status="fail", count=int(bad.sum()), details=bad.to_dict().__repr__()
        )
    return _ok("nulls")


def check_duplicates(df: pd.DataFrame, key: str = "session_date") -> CheckResult:
    dup = df[key].duplicated(keep=False)
    if dup.any():
        return CheckResult(
            name="duplicates",
            status="fail",
            count=int(dup.sum()),
            details=f"duplicate {key} rows",
            examples=_examples(df.loc[dup, key]),
        )
    return _ok("duplicates")


def check_intraday_continuity(df: pd.DataFrame, freq: str) -> CheckResult:
    """24/7 intraday bars: every expected bar between first and last should exist. Exchange outages
    (e.g. Binance maintenance) produce real gaps - warn below 2% missing, fail above."""
    ts = pd.DatetimeIndex(pd.to_datetime(df["open_time_utc"], utc=True))
    if ts.empty:
        return CheckResult(name="sessions", status="fail", details="no rows")
    step = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h"}[freq]
    expected = pd.date_range(ts.min(), ts.max(), freq=step)
    missing = expected.difference(ts)
    off_grid = ts.difference(expected)
    frac = len(missing) / max(len(expected), 1)
    off_frac = len(off_grid) / max(len(ts), 1)
    # Off-grid bars are a known Binance artifact (after the 2018-02-09 outage klines were stamped at
    # :28:14.789 for a while). Kept in storage, excluded by research loaders; fail only if widespread.
    status: Status
    if not len(missing) and not len(off_grid):
        status = "pass"
    elif frac <= 0.02 and off_frac <= 0.01:
        status = "warn"
    else:
        status = "fail"
    # summarize gaps as contiguous runs (an outage is one run, not hundreds of bars)
    runs = 0
    if len(missing):
        runs = int((pd.Series(missing).diff() != pd.Timedelta(step)).sum())
    return CheckResult(
        name="sessions",
        status=status,
        count=len(missing) + len(off_grid),
        details=f"expected={len(expected)} present={len(ts)} missing={len(missing)} ({frac:.3%}) "
        f"in {runs} gap(s); off-grid={len(off_grid)}",
        examples=_examples(pd.Index(missing).strftime("%Y-%m-%d %H:%M")),
    )


def check_monotonic(df: pd.DataFrame) -> CheckResult:
    ts = pd.to_datetime(df["close_time_utc"])
    if not ts.is_monotonic_increasing:
        n = int((ts.diff().dt.total_seconds() < 0).sum())
        return CheckResult(
            name="monotonic_time", status="fail", count=n, details="close_time_utc not increasing"
        )
    bad_tz = getattr(ts.dt, "tz", None) is None
    if bad_tz:
        return CheckResult(name="monotonic_time", status="fail", details="timestamps are timezone-naive")
    return _ok("monotonic_time")


def check_prices(df: pd.DataFrame) -> CheckResult:
    p = df[PRICE_COLUMNS]
    nonpos = (p <= 0).any(axis=1)
    incons = (df["low"] > df[["open", "close"]].min(axis=1) * (1 + 1e-9)) | (
        df["high"] < df[["open", "close"]].max(axis=1) * (1 - 1e-9)
    )
    bad = nonpos | incons
    if bad.any():
        return CheckResult(
            name="price_consistency",
            status="fail",
            count=int(bad.sum()),
            details=f"non-positive={int(nonpos.sum())} ohlc_inconsistent={int(incons.sum())}",
            examples=_examples(df.loc[bad, "session_date"]),
        )
    return _ok("price_consistency")


def check_volume(df: pd.DataFrame) -> CheckResult:
    neg = df["volume"] < 0
    zero = df["volume"] == 0
    if neg.any():
        return CheckResult(name="volume", status="fail", count=int(neg.sum()), details="negative volume")
    if zero.any():
        return CheckResult(
            name="volume",
            status="warn",
            count=int(zero.sum()),
            details="zero-volume sessions (possible halts or provider placeholders)",
            examples=_examples(df.loc[zero, "session_date"]),
        )
    return _ok("volume")


def check_sessions(df: pd.DataFrame, market: MarketConfig) -> CheckResult:
    """Compare present sessions with the exchange calendar (or continuous days for crypto)."""
    if df.empty:
        return CheckResult(name="sessions", status="fail", details="no rows")
    dates = pd.DatetimeIndex(pd.to_datetime(df["session_date"]))
    sched = session_schedule(market, dates.min(), dates.max())
    expected = pd.DatetimeIndex(sched.index)
    missing = expected.difference(dates)
    extra = dates.difference(expected)
    status: Status = "pass"
    parts = [f"expected={len(expected)} present={len(dates)}"]
    if len(missing):
        frac = len(missing) / max(len(expected), 1)
        status = "fail" if frac > 0.02 else "warn"
        parts.append(f"missing={len(missing)} ({frac:.2%})")
    if len(extra):
        status = "fail" if status == "fail" else "warn"
        parts.append(f"off-calendar={len(extra)}")
    examples = _examples(missing.strftime("%Y-%m-%d")) + _examples(extra.strftime("%Y-%m-%d"))
    return CheckResult(
        name="sessions",
        status=status,
        count=len(missing) + len(extra),
        details=" ".join(parts),
        examples=examples,
    )


def check_outlier_returns(df: pd.DataFrame, z: float = 12.0) -> CheckResult:
    """Flag |log return| beyond z robust std devs (MAD) - warn only, extreme moves can be real."""
    px = df["adj_close"].fillna(df["close"]).astype(float)
    r = np.log(px).diff().dropna()
    if len(r) < 30:
        return _ok("outlier_returns", "too few rows to assess")
    mad = float(np.median(np.abs(r - r.median()))) * 1.4826
    if mad == 0:
        return CheckResult(name="outlier_returns", status="warn", details="zero MAD (flat series)")
    flagged = r[np.abs(r - r.median()) > z * mad]
    if len(flagged):
        idx = df.loc[flagged.index, "session_date"]
        return CheckResult(
            name="outlier_returns",
            status="warn",
            count=len(flagged),
            details=f"|r| > {z} MAD-sigma; verify against corporate actions",
            examples=_examples(idx),
        )
    return _ok("outlier_returns")


def check_corporate_actions(df: pd.DataFrame) -> CheckResult:
    """Splits must coincide with a price gap roughly equal to the split ratio in UNadjusted data.

    Yahoo's `close` is already split-adjusted, so we only check that split rows exist with ratio > 0 and
    that adj_close/close factor is monotone non-decreasing in time (dividend adjustment accumulates
    backwards). Large non-monotonic jumps indicate a provider adjustment error.
    """
    if (df["split_ratio"] <= 0).any():
        return CheckResult(name="corporate_actions", status="fail", details="non-positive split ratio")
    if df["adj_close"].isna().all():
        return _ok("corporate_actions", "no adj_close provided")
    nonpos = df["adj_close"] <= 0
    if nonpos.any():
        return CheckResult(
            name="corporate_actions",
            status="warn",
            count=int(nonpos.sum()),
            details="non-positive adj_close (provider back-adjustment error); rows treated as missing",
            examples=_examples(df.loc[nonpos, "session_date"]),
        )
    factor = (df["adj_close"] / df["close"]).astype(float)
    decreasing = factor.diff() < -1e-6
    n_div = int((df["dividend"] > 0).sum())
    n_split = int((df["split_ratio"] != 1).sum())
    if decreasing.any():
        return CheckResult(
            name="corporate_actions",
            status="warn",
            count=int(decreasing.sum()),
            details=f"adjustment factor decreases on {int(decreasing.sum())} rows; "
            f"dividends={n_div} splits={n_split}",
            examples=_examples(df.loc[decreasing, "session_date"]),
        )
    return _ok("corporate_actions", f"dividends={n_div} splits={n_split}; factor monotone")


def validate_bars(df: pd.DataFrame, market: MarketConfig, freq: str = "1d") -> QualityReport:
    checks = [check_schema(df)]
    if checks[0].status != "fail":
        daily = freq == "1d"
        checks += [
            check_nulls(df),
            check_duplicates(df, "session_date" if daily else "open_time_utc"),
            check_monotonic(df),
            check_prices(df),
            check_volume(df),
            check_sessions(df, market) if daily else check_intraday_continuity(df, freq),
            check_outlier_returns(df),
            check_corporate_actions(df),
        ]
    first = str(df["session_date"].min()) if len(df) else None
    last = str(df["session_date"].max()) if len(df) else None
    return QualityReport(
        symbol=str(df["symbol"].iloc[0]) if len(df) and "symbol" in df else "?",
        market=str(df["market"].iloc[0]) if len(df) and "market" in df else "?",
        provider=str(df["provider"].iloc[0]) if len(df) and "provider" in df else "?",
        freq=freq,
        rows=len(df),
        first_session=first,
        last_session=last,
        created_at_utc=datetime.now(UTC).isoformat(),
        checks=checks,
    )
