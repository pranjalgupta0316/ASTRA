"""Market-based daily macro series from Yahoo Finance (same access path and terms as ASTRA price data:
unofficial endpoints, personal research use).

Availability: a daily close stamped on local date t is usable from the NEXT XNYS session (t+1). INR=X is
stamped by Yahoo at London midnight with an end-of-day snapshot whose exact time is undocumented, so it
is treated conservatively: usable from the SECOND XNYS session after its date (t+2).
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from astra.common.config import InstrumentConfig
from astra.common.paths import Paths, get_paths
from astra.data.macro.fred import MACRO_SCHEMA
from astra.data.validation import CheckResult, QualityReport

PROVIDER = "yahoo"
# series name -> (Yahoo symbol, sessions of lag, description)
MARKET_SERIES: dict[str, tuple[str, int, str]] = {
    "TNX": ("^TNX", 1, "Cboe 10-year Treasury yield index (percent)"),
    "FVX": ("^FVX", 1, "Cboe 5-year Treasury yield index (percent)"),
    "IRX": ("^IRX", 1, "Cboe 13-week T-bill discount rate (percent)"),
    "VIX": ("^VIX", 1, "Cboe Volatility Index"),
    "DXY": ("DX-Y.NYB", 1, "ICE US Dollar Index"),
    "USDINR": ("INR=X", 2, "INR per USD (Yahoo FX, London-midnight stamp -> t+2)"),
}
DEFAULT_START = "2000-01-01"
CAL_START = "1990-01-02"
JUMP_LOG = 0.5  # |log change| above this between consecutive observations is flagged
JUMP_RATE_PP = 1.0  # yields: |change| above 1 percentage point is flagged (log changes explode near 0)
RATE_SERIES = {"TNX", "FVX", "IRX"}
SETTLE_HOURS = 26  # bar for local date d is final once UTC time >= d + 26h (NY close + >= 5h; London +2h)


def _instrument(name: str) -> InstrumentConfig:
    return InstrumentConfig(
        symbol=name,
        market="US",
        provider="yahoo",
        provider_symbol=MARKET_SERIES[name][0],
        exchange="INDEX",
        kind="index",
    )


@lru_cache(maxsize=1)
def _xnys_sessions() -> np.ndarray:
    """XNYS sessions from CAL_START (exchange_calendars' default window starts ~20 years back, too late
    for 2000+ macro history)."""
    cal = xcals.get_calendar("XNYS", start=CAL_START)
    return pd.DatetimeIndex(cal.sessions).tz_localize(None).astype("datetime64[ns]").to_numpy()


def next_sessions(dates: pd.Series, k: int) -> pd.Series:
    """The k-th XNYS session STRICTLY after each date (NaT outside the calendar range)."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True).dt.normalize()
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    if d.empty:
        return out
    sess = _xnys_sessions()
    x = d.astype("datetime64[ns]").to_numpy()
    pos = np.searchsorted(sess, x, side="right") + (k - 1)
    ok = (pos < len(sess)) & (x >= sess[0])
    out[ok] = sess[pos[ok]]
    return out


def normalize_yahoo(frame: pd.DataFrame, name: str, retrieved_at_utc: datetime | None = None) -> pd.DataFrame:
    """Yahoo history frame (Date index reset to a column, local tz) -> long macro rows. Bars that may
    still be in progress at retrieval (retrieved < bar date + SETTLE_HOURS, in UTC) are dropped:
    Yahoo returns the live quote as the latest daily "Close"."""
    date_col = "Date" if "Date" in frame.columns else frame.columns[0]
    stamps = pd.to_datetime(frame[date_col])
    local_date = (
        stamps.dt.tz_localize(None).dt.normalize() if stamps.dt.tz is not None else stamps.dt.normalize()
    )
    df = pd.DataFrame({"date": local_date, "value": pd.to_numeric(frame["Close"], errors="coerce")})
    df = df.dropna().drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    if retrieved_at_utc is not None:
        cutoff = pd.Timestamp(retrieved_at_utc).tz_convert("UTC").tz_localize(None) - pd.Timedelta(
            hours=SETTLE_HOURS
        )
        df = df[df["date"] <= cutoff].reset_index(drop=True)
    avail = next_sessions(df["date"], MARKET_SERIES[name][1])
    return pd.DataFrame(
        {
            "series": name,
            "date": df["date"].dt.date,
            "period_end": df["date"].dt.date,
            "value": df["value"].astype("float64"),
            "available_from": avail.dt.date,
        }
    ).dropna(subset=["available_from"])


def quality_report_market(
    df: pd.DataFrame, names: list[str], errors: dict[str, str]
) -> tuple[QualityReport, list]:
    checks: list[CheckResult] = []
    nulls = int(df[["series", "date", "value", "available_from"]].isna().sum().sum())
    checks.append(
        CheckResult(name="nulls", status="fail" if nulls else "pass", count=nulls, details="required columns")
    )
    dups = int(df.duplicated(["series", "date"]).sum())
    checks.append(
        CheckResult(
            name="duplicates", status="fail" if dups else "pass", count=dups, details="(series, date)"
        )
    )
    nonmono = [s for s, g in df.groupby("series") if not pd.to_datetime(g["date"]).is_monotonic_increasing]
    checks.append(
        CheckResult(
            name="monotonic_dates",
            status="fail" if nonmono else "pass",
            count=len(nonmono),
            details="per series",
            examples=nonmono,
        )
    )
    early = int((pd.to_datetime(df["available_from"]) <= pd.to_datetime(df["date"])).sum())
    checks.append(
        CheckResult(
            name="available_after_date",
            status="fail" if early else "pass",
            count=early,
            details="available_from > observation date",
        )
    )
    nonpos = df[(df["value"] <= 0) & (df["series"] != "IRX")]  # T-bill rate can legitimately be ~0
    checks.append(
        CheckResult(
            name="non_positive",
            status="warn" if len(nonpos) else "pass",
            count=len(nonpos),
            details="value <= 0 (IRX exempt)",
            examples=[f"{r.series} {r.date} {r.value}" for r in nonpos.head(5).itertuples()],
        )
    )
    jumps = []
    for s, g in df.groupby("series"):
        if s in RATE_SERIES:
            big = g["value"].diff().abs() > JUMP_RATE_PP
        else:
            big = np.log(g["value"].where(g["value"] > 0)).diff().abs() > JUMP_LOG
        for i in big[big].index:
            jumps.append(f"{s} {g.loc[i, 'date']} {g['value'].shift(1)[i]:.4g}->{g.loc[i, 'value']:.4g}")
    checks.append(
        CheckResult(
            name="jumps",
            status="warn" if jumps else "pass",
            count=len(jumps),
            details=f"|log change| > {JUMP_LOG} (yields: |change| > {JUMP_RATE_PP} pp) between consecutive "
            "observations (kept, not repaired)",
            examples=jumps[:10],
        )
    )
    coverage = []
    for s in names:
        g = df[df["series"] == s]
        coverage.append(
            {
                "series": s,
                "yahoo": MARKET_SERIES[s][0],
                "rows": len(g),
                "first_date": str(g["date"].min()) if len(g) else None,
                "last_date": str(g["date"].max()) if len(g) else None,
                "lag_sessions": MARKET_SERIES[s][1],
                "error": errors.get(s),
            }
        )
    absent = [c["series"] for c in coverage if not c["rows"]]
    checks.append(
        CheckResult(
            name="coverage",
            status="warn" if absent else "pass",
            count=len(absent),
            details=f"{len(names) - len(absent)}/{len(names)} series with data",
            examples=absent,
        )
    )
    rep = QualityReport(
        symbol="MARKET_MACRO",
        market="US",
        provider=PROVIDER,
        freq="macro",
        rows=len(df),
        first_session=str(df["date"].min()) if len(df) else None,
        last_session=str(df["date"].max()) if len(df) else None,
        created_at_utc=datetime.now(UTC).isoformat(),
        checks=checks,
    )
    return rep, coverage


def market_path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).validated / "macro" / "market.parquet"


def write_macro_table(
    df: pd.DataFrame,
    out: Path,
    rep: QualityReport,
    coverage: list,
    report_name: str,
    paths: Paths,
    extra: dict[str, Any],
) -> tuple[str, Path]:
    """Shared writer: zstd Parquet (MACRO_SCHEMA) + manifest + quality report JSON."""
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df[MACRO_SCHEMA.names], schema=MACRO_SCHEMA, preserve_index=False),
        out,
        compression="zstd",
    )
    qdir = paths.reports / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    rpath = (
        qdir / f"{report_name}_{datetime.fromisoformat(rep.created_at_utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    rpath.write_text(
        json.dumps(
            {**json.loads(rep.model_dump_json()), "status": rep.status, "coverage": coverage},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest = {
        "dataset": out.stem,
        "rows": len(df),
        "series": sorted(df["series"].unique().tolist()),
        "sha256": sha,
        "quality_status": rep.status,
        "quality_report": rpath.name,
        "written_at_utc": datetime.now(UTC).isoformat(),
        **extra,
    }
    out.with_name(f"{out.stem}.manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    return sha, rpath


def ingest_market_macro(
    series: list[str] | None = None,
    paths: Paths | None = None,
    start: str = DEFAULT_START,
    adapter: Any = None,
) -> dict[str, Any]:
    """Fetch daily closes via the Yahoo adapter, store raw (adapter raw layer) + validated
    data/validated/macro/market.parquet + quality report."""
    from astra.data.storage import sha256_file, write_raw

    t0 = time.monotonic()
    paths = paths or get_paths()
    names = list(series) if series else list(MARKET_SERIES)
    unknown = [s for s in names if s not in MARKET_SERIES]
    if unknown:
        raise KeyError(f"Unknown market macro series {unknown}")
    if adapter is None:
        from astra.data.adapters.yahoo import YahooAdapter

        adapter = YahooAdapter()
    frames, raw, errors = [], {}, {}
    for name in names:
        try:
            batch = adapter.fetch_daily(_instrument(name), start, None, paths.raw / "yahoo")
        except Exception as e:  # adapter raises RuntimeError after its own retries
            errors[name] = repr(e)
            continue
        raw_file = write_raw(batch, paths)
        raw[name] = {"file": raw_file.name, "sha256": sha256_file(raw_file), "rows": len(batch.frame)}
        frames.append(normalize_yahoo(batch.frame, name, batch.retrieved_at_utc))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MACRO_SCHEMA.names)
    out = market_path(paths)
    if out.exists():
        old = pq.read_table(out).to_pandas()
        old = old[~old["series"].isin(df["series"].unique())]
        df = pd.concat([old, df], ignore_index=True) if len(old) else df
    df = df.sort_values(["series", "date"]).reset_index(drop=True)
    rep, coverage = quality_report_market(df, names, errors)
    sha, rpath = write_macro_table(
        df, out, rep, coverage, "MARKET_macro", paths, {"provider": PROVIDER, "raw": raw, "start": start}
    )
    return {
        "provider": PROVIDER,
        "series": names,
        "rows": len(df),
        "validated": str(out),
        "sha256": sha,
        "raw": raw,
        "errors": errors,
        "quality_report": str(rpath),
        "quality_status": rep.status,
        "coverage": coverage,
        "runtime_s": round(time.monotonic() - t0, 2),
    }
