"""BLS Public Data API v1 (no key): CPI-U all items SA and the unemployment rate.

Terms (checked 2026-11-25): BLS material is public domain ("free to use our public domain material
without specific permission"); API ToS asks users to cite the retrieval date and to state that BLS
"cannot vouch for the data or analyses derived from these data". v1 limits for unregistered users:
25 queries/day, 25 series/query, 10 years/query. ASTRA caps itself at MAX_QUERIES_PER_DAY, logs every
query in data/raw/bls/query_log.json, and reuses cached raw responses for year-chunks old enough that
seasonal revisions (last 5 years) no longer change them.

Availability: available_from = max(period end + fixed lag, actual first-release date). The release
dates are parsed from BLS's own news-release archive pages (empsit / cpi), so delayed or cancelled
releases (2013 and 2025 shutdowns) are honoured. Releases are at 08:30 ET, before the US open.
Values are the CURRENT published values (SA CPI is revised every February for 5 years) - not vintages.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from pandas.tseries.offsets import MonthEnd

from astra.common.paths import Paths, get_paths
from astra.data.macro.fred import MACRO_SCHEMA
from astra.data.validation import CheckResult, QualityReport

API_URL = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
RELEASE_PAGES = {
    "empsit": "https://www.bls.gov/bls/news-release/empsit.htm",
    "cpi": "https://www.bls.gov/bls/news-release/cpi.htm",
}
USER_AGENT = "ASTRA research astra@example.com"
TIMEOUT_S = 30
PAUSE_S = 1.0
MAX_QUERIES_PER_DAY = 20  # BLS v1 allows 25
YEARS_PER_QUERY = 10
REVISION_YEARS = 5  # SA factors revised for the last 5 years each February
PROVIDER = "bls"

# BLS series id -> (ASTRA name, fixed lag in days after month end, release page)
BLS_SERIES: dict[str, tuple[str, int, str]] = {
    "CUSR0000SA0": ("CPI", 45, "cpi"),
    "LNS14000000": ("UNRATE", 10, "empsit"),
}
MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ],
        start=1,
    )
}


class BlsQuotaError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------- raw / quota
def _raw_dir(paths: Paths) -> Path:
    return paths.raw / "bls"


def _store(content: bytes, sub: str, stem: str, paths: Paths, meta: dict) -> dict:
    d = _raw_dir(paths) / sub
    d.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    fname = f"{stem}_{now.strftime('%Y%m%dT%H%M%SZ')}.gz"
    gz = gzip.compress(content, mtime=0)
    (d / fname).write_bytes(gz)
    m = {
        "provider": PROVIDER,
        "retrieved_at_utc": now.isoformat(),
        "file": fname,
        "sha256": hashlib.sha256(gz).hexdigest(),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "user_agent": USER_AGENT,
        **meta,
    }
    (d / f"{fname[:-3]}.meta.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


def _latest(sub: str, stem: str, paths: Paths) -> tuple[dict, bytes] | None:
    d = _raw_dir(paths) / sub
    metas = sorted(d.glob(f"{stem}_*.meta.json")) if d.exists() else []
    if not metas:
        return None
    m = json.loads(metas[-1].read_text(encoding="utf-8"))
    return m, gzip.decompress((d / m["file"]).read_bytes())


def _consume_query(paths: Paths) -> int:
    p = _raw_dir(paths) / "query_log.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    log = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    today = datetime.now(UTC).date().isoformat()
    used = int(log.get(today, 0))
    if used >= MAX_QUERIES_PER_DAY:
        raise BlsQuotaError(f"BLS v1 daily budget exhausted ({used}/{MAX_QUERIES_PER_DAY} on {today} UTC)")
    log[today] = used + 1
    p.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return used + 1


# ----------------------------------------------------------------------------------- parsing
def parse_release_dates(html: str, kind: str) -> dict[pd.Timestamp, pd.Timestamp]:
    """BLS archive page -> {reference month start: release date}. Each <li> holds "<Month> <YYYY>
    <title>" and links named <kind>_<MMDDYYYY>.(htm|txt|pdf) (both the 2008+ and the older layout).
    Commented-out (scheduled) entries are ignored."""
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    title = "Employment Situation" if kind == "empsit" else "Consumer Price Index"
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    for li in re.findall(r"<li>(.*?)</li>", html, flags=re.S):
        ref = re.search(rf"(\w+) (\d{{4}}) {title}", re.sub(r"<[^>]+>", "", li))
        rel = re.search(rf"/{kind}_(\d{{8}})\.(?:htm|txt|pdf)", li)
        if not ref or not rel or ref.group(1) not in MONTHS:
            continue
        month = pd.Timestamp(int(ref.group(2)), MONTHS[ref.group(1)], 1)
        d = rel.group(1)
        date = pd.Timestamp(f"{d[4:]}-{d[:2]}-{d[2:4]}")
        out[month] = min(date, out.get(month, date))
    return out


def parse_api_response(payload: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    """API JSON -> (series_id, date, value); monthly periods only; '-' (not collected) dropped+counted."""
    if payload.get("status") != "REQUEST_SUCCEEDED":
        raise RuntimeError(f"BLS API: {payload.get('status')} {payload.get('message')}")
    rows, missing = [], {}
    for s in payload["Results"]["series"]:
        sid = s["seriesID"]
        for x in s["data"]:
            if not re.fullmatch(r"M(0[1-9]|1[0-2])", x["period"]):
                continue  # M13 = annual average
            v = pd.to_numeric(x["value"], errors="coerce")
            if pd.isna(v):
                missing[sid] = missing.get(sid, 0) + 1
                continue
            rows.append(
                {
                    "series_id": sid,
                    "date": pd.Timestamp(int(x["year"]), int(x["period"][1:]), 1),
                    "value": float(v),
                }
            )
    return pd.DataFrame(rows, columns=["series_id", "date", "value"]), missing


def availability(series_id: str, dates: pd.Series, releases: dict[pd.Timestamp, pd.Timestamp]) -> pd.Series:
    """max(month end + fixed lag, first release date) for monthly observations dated the 1st."""
    _, lag, _ = BLS_SERIES[series_id]
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    fixed = d + MonthEnd(0) + pd.Timedelta(days=lag)
    rel = d.map(lambda x: releases.get(x, pd.NaT))
    return pd.concat([fixed, rel], axis=1).max(axis=1)


def normalize_bls(obs: pd.DataFrame, releases: dict[str, dict]) -> pd.DataFrame:
    frames = []
    for sid, g in obs.groupby("series_id"):
        name, _, kind = BLS_SERIES[sid]
        g = g.drop_duplicates("date").sort_values("date")
        frames.append(
            pd.DataFrame(
                {
                    "series": name,
                    "date": g["date"].dt.date.to_numpy(),
                    "period_end": (g["date"] + MonthEnd(0)).dt.date.to_numpy(),
                    "value": g["value"].to_numpy(),
                    "available_from": availability(sid, g["date"], releases.get(kind, {})).dt.date.to_numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=MACRO_SCHEMA.names)


# ----------------------------------------------------------------------------------- ingestion
def _chunks(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [
        (y, min(y + YEARS_PER_QUERY - 1, end_year)) for y in range(start_year, end_year + 1, YEARS_PER_QUERY)
    ]


def bls_path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).validated / "macro" / "bls.parquet"


def ingest_bls(
    series: list[str] | None = None,
    paths: Paths | None = None,
    start_year: int = 2000,
    http: requests.Session | None = None,
) -> dict[str, Any]:
    from astra.data.macro.market import write_macro_table

    t0 = time.monotonic()
    paths = paths or get_paths()
    ids = list(series) if series else list(BLS_SERIES)
    http = http or requests.Session()
    http.headers.update({"User-Agent": USER_AGENT})
    now = datetime.now(UTC)
    queries, reused, errors = 0, 0, {}
    # release calendars (HTML pages, not API queries)
    releases: dict[str, dict] = {}
    for kind in sorted({BLS_SERIES[s][2] for s in ids}):
        try:
            r = http.get(RELEASE_PAGES[kind], timeout=TIMEOUT_S)
            r.raise_for_status()
            _store(r.content, "release_calendar", kind, paths, {"url": RELEASE_PAGES[kind]})
            html = r.text
        except requests.RequestException as e:
            cached = _latest("release_calendar", kind, paths)
            if cached is None:
                errors[f"release_calendar:{kind}"] = repr(e)
                continue
            html = cached[1].decode("utf-8")
            errors[f"release_calendar:{kind}"] = f"used cached copy ({e!r})"
        releases[kind] = parse_release_dates(html, kind)
        time.sleep(PAUSE_S)
    frames, missing = [], {}
    for y0, y1 in _chunks(start_year, now.year):
        stem = f"{'-'.join(ids)}_{y0}-{y1}"
        cached = _latest("api", stem, paths)
        fresh_enough = cached is not None and (
            y1 < now.year - REVISION_YEARS or cached[0]["retrieved_at_utc"][:10] == now.date().isoformat()
        )
        if fresh_enough and cached is not None:
            payload = json.loads(cached[1])
            reused += 1
        else:
            try:
                _consume_query(paths)
                queries += 1
                body = {"seriesid": ids, "startyear": str(y0), "endyear": str(y1)}
                r = http.post(API_URL, json=body, timeout=TIMEOUT_S)
                r.raise_for_status()
                payload = r.json()
                parse_api_response(payload)  # validate before caching
                _store(r.content, "api", stem, paths, {"url": API_URL, "request": body})
            except (BlsQuotaError, requests.RequestException, RuntimeError) as e:
                errors[f"{y0}-{y1}"] = repr(e)
                if cached is None:
                    continue
                payload = json.loads(cached[1])
                reused += 1
            time.sleep(PAUSE_S)
        obs, miss = parse_api_response(payload)
        frames.append(obs)
        for k, v in miss.items():
            missing[k] = missing.get(k, 0) + v
    obs = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["series_id", "date", "value"])
    )
    df = normalize_bls(obs, releases).sort_values(["series", "date"]).reset_index(drop=True)
    rep, coverage = quality_report_bls(df, missing, ids, releases)
    out = bls_path(paths)
    sha, rpath = write_macro_table(
        df,
        out,
        rep,
        coverage,
        "BLS_macro",
        paths,
        {"provider": PROVIDER, "api": API_URL, "citation": f"BLS Public Data API v1, retrieved {now.date()}"},
    )
    return {
        "provider": PROVIDER,
        "series": ids,
        "rows": len(df),
        "validated": str(out),
        "sha256": sha,
        "api_queries_used": queries,
        "chunks_reused": reused,
        "errors": errors,
        "missing_values": missing,
        "quality_report": str(rpath),
        "quality_status": rep.status,
        "coverage": coverage,
        "runtime_s": round(time.monotonic() - t0, 2),
    }


def quality_report_bls(
    df: pd.DataFrame, missing: dict[str, int], ids: list[str], releases: dict[str, dict]
) -> tuple[QualityReport, list]:
    checks: list[CheckResult] = []
    nulls = int(df[["series", "date", "value", "available_from"]].isna().sum().sum())
    checks.append(
        CheckResult(
            name="nulls",
            status="fail" if nulls else ("warn" if missing else "pass"),
            count=nulls,
            details=f"nulls: {nulls}; BLS '-' (not collected/published) dropped: {missing}",
        )
    )
    dups = int(df.duplicated(["series", "date"]).sum())
    checks.append(
        CheckResult(
            name="duplicates", status="fail" if dups else "pass", count=dups, details="(series, date)"
        )
    )
    gaps = []
    for s, g in df.groupby("series"):
        m = pd.to_datetime(g["date"]).dt.to_period("M")
        full = pd.period_range(m.min(), m.max(), freq="M")
        gaps += [f"{s} {p}" for p in full.difference(pd.PeriodIndex(m))]
    checks.append(
        CheckResult(
            name="monotonic_dates",
            status="warn" if gaps else "pass",
            count=len(gaps),
            details="monthly sequence; missing months listed",
            examples=gaps[:10],
        )
    )
    early = int((pd.to_datetime(df["available_from"]) <= pd.to_datetime(df["period_end"])).sum())
    checks.append(
        CheckResult(
            name="available_after_period_end",
            status="fail" if early else "pass",
            count=early,
            details="available_from > period_end",
        )
    )
    coverage = []
    for sid in ids:
        name, lag, kind = BLS_SERIES[sid]
        g = df[df["series"] == name]
        d = pd.to_datetime(g["date"])
        fixed = d + MonthEnd(0) + pd.Timedelta(days=lag)
        later = int((pd.to_datetime(g["available_from"]) > fixed).sum())
        rel = releases.get(kind, {})
        coverage.append(
            {
                "series": name,
                "bls_id": sid,
                "rows": len(g),
                "first_date": str(g["date"].min()) if len(g) else None,
                "last_date": str(g["date"].max()) if len(g) else None,
                "fixed_lag_days": lag,
                "months_with_release_date": int(d.isin(list(rel)).sum()),
                "months_where_release_later_than_fixed_lag": later,
            }
        )
    no_rel = [c["series"] for c in coverage if c["rows"] and not c["months_with_release_date"]]
    checks.append(
        CheckResult(
            name="release_calendar",
            status="warn" if no_rel else "pass",
            count=len(no_rel),
            details="series without parsed release dates fall back to fixed lags",
            examples=no_rel,
        )
    )
    rep = QualityReport(
        symbol="BLS_MACRO",
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
