"""FRED macro series with conservative publication lags (no vintages) -> as-of macro features.

Endpoint: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` (keyless CSV).

TERMS (fred.stlouisfed.org/legal, checked 2026-11-25): the FRED Terms of Use prohibit, for all FRED
services, using "scripts, spiders, scrapers, bots, or any ... automatic technology" to access content,
"store, cache, or archive any portion" of FRED content, and use "in connection with the development or
training of any software program or system or machine learning". The general license only allows
downloading a copy for personal, non-commercial use. Automated download is therefore OFF by default:
`ingest_fred` reads CSVs the user downloaded manually (`csv_dir`) unless `allow_download=True` is
passed explicitly by a human who has reviewed the terms. Some series are third-party copyrighted
(VIXCLS: Cboe; UMCSENT: University of Michigan).

No vintages are available without an API key, so the stored values are the CURRENT (revised) values.
Point-in-time safety is approximated with fixed publication lags (`LAGS`); see
docs/data/FUNDAMENTALS_MACRO.md.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay, MonthEnd

from astra.common.paths import Paths, get_paths
from astra.data.validation import CheckResult, QualityReport

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
PROVIDER = "fred"
TIMEOUT_S = 30
MIN_INTERVAL_S = 1.0
MAX_ATTEMPTS = 4
BACKOFF_S = 2.0
RETRY_STATUS = {429, 500, 502, 503, 504}
FED_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


@dataclass(frozen=True)
class Lag:
    """available_from = (period end of the observation) + lag.

    freq "D": period end = observation date; lag in Federal-Reserve business days (`bdays`) and/or
    calendar days (`days`, then rolled forward to a Fed business day).
    freq "M": FRED dates monthly series at the 1st of the month; period end = month end; `days` calendar.
    """

    freq: str
    days: int = 0
    bdays: int = 0
    source: str = ""
    rationale: str = ""


LAGS: dict[str, Lag] = {
    "DGS10": Lag("D", bdays=1, source="Federal Reserve H.15 (public)", rationale="H.15 posts day d on d+1"),
    "DGS2": Lag("D", bdays=1, source="Federal Reserve H.15 (public)", rationale="H.15 posts day d on d+1"),
    "T10Y2Y": Lag("D", bdays=1, source="FRB St. Louis from H.15 (public)", rationale="derived from H.15"),
    "DFF": Lag(
        "D", bdays=1, source="FRBNY EFFR via H.15 (public)", rationale="EFFR published next business day"
    ),
    "VIXCLS": Lag(
        "D", bdays=1, source="Cboe (copyrighted, third party)", rationale="close known same day; +1 bday"
    ),
    "DEXINUS": Lag(
        "D",
        days=8,
        source="Federal Reserve H.10 (public)",
        rationale="H.10 is released WEEKLY (Mondays) for the prior week -> obs + 8 calendar days",
    ),
    "DTWEXBGS": Lag(
        "D",
        days=8,
        source="Federal Reserve H.10 (public)",
        rationale="H.10 weekly release -> obs + 8 calendar days",
    ),
    "CPIAUCSL": Lag(
        "M", days=45, source="BLS CPI (public)", rationale="CPI ~10-15 days after month; +45 conservative"
    ),
    "UNRATE": Lag(
        "M", days=10, source="BLS Employment Situation (public)", rationale="first Friday of next month"
    ),
    "INDPRO": Lag(
        "M", days=20, source="Federal Reserve G.17 (public)", rationale="G.17 mid-month (~day 15-17)"
    ),
    "UMCSENT": Lag(
        "M",
        days=30,
        source="University of Michigan (copyrighted, third party)",
        rationale="prelim mid-month, final end of month; +30 after month end",
    ),
}
DEFAULT_SERIES = list(LAGS)

MACRO_SCHEMA = pa.schema(
    [
        ("series", pa.string()),
        ("date", pa.date32()),
        ("period_end", pa.date32()),
        ("value", pa.float64()),
        ("available_from", pa.date32()),
    ]
)


class FredTermsNotAcceptedError(RuntimeError):
    pass


def available_from(series: str, dates: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """Conservative first date on which an observation dated `dates` may be used."""
    lag = LAGS[series]
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    if lag.freq == "M":
        end = d + MonthEnd(0)
        return (end + pd.Timedelta(days=lag.days)).dt.normalize()
    out = d + pd.Timedelta(days=lag.days)
    if lag.days:
        out = out.map(lambda x: FED_BDAY.rollforward(x))
    if lag.bdays:
        out = out.map(lambda x: x + lag.bdays * FED_BDAY)
    return pd.to_datetime(out).dt.normalize()


def period_end(series: str, dates: pd.Series) -> pd.Series:
    d = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
    return d + MonthEnd(0) if LAGS[series].freq == "M" else d


def parse_fred_csv(text: str, series: str) -> tuple[pd.DataFrame, int]:
    """fredgraph.csv -> (date, value); '.' / blank = missing (dropped, count returned)."""
    raw = pd.read_csv(io.StringIO(text), dtype=str)
    date_col = next(c for c in raw.columns if c.lower() in ("observation_date", "date"))
    val_col = series if series in raw.columns else next(c for c in raw.columns if c != date_col)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "value": pd.to_numeric(raw[val_col], errors="coerce"),
        }
    )
    n_missing = int(df["value"].isna().sum())
    return df.dropna(subset=["date", "value"]).reset_index(drop=True), n_missing


def normalize_series(text: str, series: str) -> tuple[pd.DataFrame, int]:
    df, n_missing = parse_fred_csv(text, series)
    out = pd.DataFrame(
        {
            "series": series,
            "date": df["date"].dt.date,
            "period_end": period_end(series, df["date"]).dt.date,
            "value": df["value"].astype("float64"),
            "available_from": available_from(series, df["date"]).dt.date,
        }
    )
    return out, n_missing


# ----------------------------------------------------------------------------------- raw layer
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _store_raw(content: bytes, series: str, paths: Paths, origin: str) -> dict[str, Any]:
    d = paths.raw / "fred" / series
    d.mkdir(parents=True, exist_ok=True)
    sha = _sha(content)
    metas = sorted(d.glob(f"{series}_*.meta.json"))
    if metas:
        last = json.loads(metas[-1].read_text(encoding="utf-8"))
        if last.get("sha256") == sha and (d / last["file"]).exists():
            return {**last, "reused": True}
    now = datetime.now(UTC)
    fname = f"{series}_{now.strftime('%Y%m%dT%H%M%SZ')}.csv"
    (d / fname).write_bytes(content)
    meta = {
        "provider": PROVIDER,
        "series": series,
        "url": FRED_CSV_URL.format(series=series),
        "origin": origin,  # "download" or "manual:<filename>"
        "retrieved_at_utc": now.isoformat(),
        "file": fname,
        "sha256": sha,
        "lag": LAGS[series].__dict__,
        "note": "current (revised) values; no vintages",
    }
    (d / fname.replace(".csv", ".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {**meta, "reused": False}


def download_series(http: requests.Session, series: str) -> bytes:
    url = FRED_CSV_URL.format(series=series)
    r: requests.Response | None = None
    for attempt in range(MAX_ATTEMPTS):
        r = http.get(url, timeout=TIMEOUT_S)
        if r.status_code not in RETRY_STATUS:
            break
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(BACKOFF_S * 2**attempt)
    assert r is not None
    r.raise_for_status()
    time.sleep(MIN_INTERVAL_S)
    return r.content


# ----------------------------------------------------------------------------------- quality
def quality_report_macro(
    df: pd.DataFrame, missing: dict[str, int], series: list[str]
) -> tuple[QualityReport, list[dict]]:
    checks: list[CheckResult] = []
    nulls = int(df[["series", "date", "value", "available_from"]].isna().sum().sum())
    checks.append(
        CheckResult(
            name="nulls",
            status="fail" if nulls else ("warn" if sum(missing.values()) else "pass"),
            count=nulls,
            details=f"nulls after parse: {nulls}; provider missing markers ('.') dropped: {missing}",
        )
    )
    dups = int(df.duplicated(["series", "date"]).sum())
    checks.append(
        CheckResult(
            name="duplicates", status="fail" if dups else "pass", count=dups, details="(series, date)"
        )
    )
    nonmono = [
        s for s, g in df.groupby("series") if not pd.Series(pd.to_datetime(g["date"])).is_monotonic_increasing
    ]
    checks.append(
        CheckResult(
            name="monotonic_dates",
            status="fail" if nonmono else "pass",
            count=len(nonmono),
            details="dates strictly increasing per series",
            examples=nonmono,
        )
    )
    early = df[pd.to_datetime(df["available_from"]) <= pd.to_datetime(df["period_end"])]
    checks.append(
        CheckResult(
            name="available_after_period_end",
            status="fail" if len(early) else "pass",
            count=len(early),
            details="available_from > period_end for every row",
        )
    )
    coverage = []
    for s in series:
        g = df[df["series"] == s]
        coverage.append(
            {
                "series": s,
                "rows": len(g),
                "first_date": str(g["date"].min()) if len(g) else None,
                "last_date": str(g["date"].max()) if len(g) else None,
                "last_available_from": str(g["available_from"].max()) if len(g) else None,
                "missing_markers": missing.get(s, 0),
                "lag": LAGS[s].__dict__,
            }
        )
    absent = [c["series"] for c in coverage if not c["rows"]]
    checks.append(
        CheckResult(
            name="coverage",
            status="warn" if absent else "pass",
            count=len(absent),
            details=f"{len(series) - len(absent)}/{len(series)} series with data",
            examples=absent,
        )
    )
    rep = QualityReport(
        symbol="FRED_MACRO",
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


# ----------------------------------------------------------------------------------- ingestion
def fred_path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).validated / "macro" / "fred.parquet"


def ingest_fred(
    series: list[str] | None = None,
    paths: Paths | None = None,
    *,
    csv_dir: Path | None = None,
    allow_download: bool = False,
    http: requests.Session | None = None,
) -> dict[str, Any]:
    """Build data/validated/macro/fred.parquet from manually downloaded CSVs (`csv_dir/<SERIES>.csv`)
    or, only with `allow_download=True` (explicit human decision after reading the FRED terms), from
    the fredgraph.csv endpoint."""
    t0 = time.monotonic()
    paths = paths or get_paths()
    series = list(series) if series else DEFAULT_SERIES
    unknown = [s for s in series if s not in LAGS]
    if unknown:
        raise KeyError(f"No publication lag defined for {unknown}; add one to LAGS first")
    if csv_dir is None and not allow_download:
        raise FredTermsNotAcceptedError(
            "FRED Terms of Use prohibit automated access, local storage/caching and ML use of FRED content. "
            "Download the CSVs manually for personal use and pass csv_dir=..., or pass allow_download=True "
            "only after a human has reviewed https://fred.stlouisfed.org/legal/."
        )
    http = http or (requests.Session() if allow_download else None)
    frames, missing, raw_meta, errors = [], {}, {}, {}
    for s in series:
        try:
            if csv_dir is not None and (Path(csv_dir) / f"{s}.csv").exists():
                content, origin = (Path(csv_dir) / f"{s}.csv").read_bytes(), f"manual:{s}.csv"
            elif allow_download:
                content, origin = download_series(http, s), "download"  # type: ignore[arg-type]
            else:
                errors[s] = "no CSV in csv_dir"
                continue
            meta = _store_raw(content, s, paths, origin)
            df, n_missing = normalize_series(content.decode("utf-8"), s)
        except (requests.RequestException, ValueError, StopIteration) as e:
            errors[s] = str(e)
            continue
        frames.append(df)
        missing[s] = n_missing
        raw_meta[s] = {
            "file": meta["file"],
            "sha256": meta["sha256"],
            "reused": meta["reused"],
            "origin": origin,
        }
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MACRO_SCHEMA.names)
    out = fred_path(paths)
    if out.exists():
        old = pq.read_table(out).to_pandas()
        old = old[~old["series"].isin(df["series"].unique())]
        df = pd.concat([old, df], ignore_index=True) if len(old) else df
    df = df.sort_values(["series", "date"]).reset_index(drop=True)
    rep, coverage = quality_report_macro(df, missing, series)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df[MACRO_SCHEMA.names], schema=MACRO_SCHEMA, preserve_index=False),
        out,
        compression="zstd",
    )
    qdir = paths.reports / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    rpath = qdir / f"FRED_macro_{datetime.fromisoformat(rep.created_at_utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    rpath.write_text(
        json.dumps(
            {**json.loads(rep.model_dump_json()), "status": rep.status, "coverage": coverage},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    sha = _sha(out.read_bytes())
    manifest = {
        "dataset": "fred",
        "rows": len(df),
        "series": sorted(df["series"].unique().tolist()),
        "sha256": sha,
        "raw": raw_meta,
        "quality_status": rep.status,
        "quality_report": rpath.name,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "notes": ["current revised values (no vintages); availability from fixed publication lags"],
    }
    out.with_name("fred.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "provider": PROVIDER,
        "series": series,
        "rows": len(df),
        "validated": str(out),
        "sha256": sha,
        "raw": raw_meta,
        "errors": errors,
        "quality_report": str(rpath),
        "quality_status": rep.status,
        "coverage": coverage,
        "runtime_s": round(time.monotonic() - t0, 2),
    }


def load_fred(paths: Paths | None = None) -> pd.DataFrame:
    return pq.read_table(fred_path(paths)).to_pandas()


def macro_features(
    dates: pd.Series | pd.DatetimeIndex | list,
    series: list[str] | None = None,
    table: pd.DataFrame | None = None,
    paths: Paths | None = None,
) -> pd.DataFrame:
    """As-of macro values: for each date t and series, the value of the LATEST observation whose
    `available_from <= t` (forward-filled only from availability). Index = the given dates."""
    table = load_fred(paths) if table is None else table
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    series = series or sorted(table["series"].unique())
    out = pd.DataFrame(index=idx)
    left = pd.DataFrame({"key": idx.astype("datetime64[ns]"), "_i": np.arange(len(idx))}).sort_values("key")
    for s in series:
        g = table[table["series"] == s].copy()
        if g.empty:
            out[s] = np.nan
            continue
        g["key"] = pd.to_datetime(g["available_from"]).astype("datetime64[ns]")
        g["_d"] = pd.to_datetime(g["date"])
        g = g.sort_values(["key", "_d"])
        # state at each availability date = latest observation date known so far
        g["_best"] = g["_d"].cummax()
        g = g[g["_d"] == g["_best"]].drop_duplicates("key", keep="last")[["key", "value"]]
        m = pd.merge_asof(left, g, on="key", direction="backward").sort_values("_i")
        out[s] = m["value"].to_numpy()
    out.index.name = "date"
    return out
