"""SEC EDGAR XBRL companyfacts -> point-in-time US fundamentals.

Source: `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` (US government work, public
domain) and `https://www.sec.gov/files/company_tickers.json` for ticker -> CIK.

Fair-access rules (sec.gov "Accessing EDGAR Data", checked 2026-11-25): declare a descriptive
User-Agent, max 10 requests/second. ASTRA stays at <= 4 requests/second, uses a timeout on every
request and backs off on 429/5xx. HTTP 403 means the client was refused -> `SecAccessBlockedError`.

User-Agent: taken from env var `SEC_USER_AGENT`, else `DEFAULT_USER_AGENT`. The env value is never
written to disk; metadata records only the string "env:SEC_USER_AGENT".

Point-in-time rules (see docs/data/FUNDAMENTALS_MACRO.md):
- A fact is usable only from the first session STRICTLY AFTER its `filed` date (companyfacts carries
  the filing date, not the acceptance time, so an after-close filing is assumed).
- Restatements: for every (concept, period) the value FIRST filed is used forever; later amendments
  and restated comparatives never leak back into history.
- Derived values (Q4 = FY - 9M YTD, TTM sums, ratios) become available at the latest `filed` date of
  their components.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from astra.common.config import get_config
from astra.common.paths import Paths, get_paths
from astra.data.validation import CheckResult, QualityReport

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
DEFAULT_USER_AGENT = "ASTRA research astra@example.com"
UA_ENV = "SEC_USER_AGENT"
TIMEOUT_S = 30
MIN_INTERVAL_S = 0.25  # <= 4 requests/second (SEC allows 10; task cap 5)
MAX_ATTEMPTS = 5
BACKOFF_S = 2.0
RETRY_STATUS = {429, 500, 502, 503, 504}
PROVIDER = "sec_edgar"

# canonical concept -> candidate (taxonomy, tag) in priority order; unit expected for the concept
CONCEPTS: dict[str, tuple[list[tuple[str, str]], str]] = {
    "Revenues": (
        [
            ("us-gaap", "Revenues"),
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
            ("us-gaap", "SalesRevenueNet"),
        ],
        "USD",
    ),
    "NetIncomeLoss": ([("us-gaap", "NetIncomeLoss")], "USD"),
    "Assets": ([("us-gaap", "Assets")], "USD"),
    "Liabilities": ([("us-gaap", "Liabilities")], "USD"),
    "StockholdersEquity": (
        [
            ("us-gaap", "StockholdersEquity"),
            ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
        ],
        "USD",
    ),
    "EarningsPerShareDiluted": ([("us-gaap", "EarningsPerShareDiluted")], "USD/shares"),
    "CommonStockSharesOutstanding": (
        [("dei", "EntityCommonStockSharesOutstanding"), ("us-gaap", "CommonStockSharesOutstanding")],
        "shares",
    ),
    "OperatingIncomeLoss": ([("us-gaap", "OperatingIncomeLoss")], "USD"),
    "CashAndCashEquivalentsAtCarryingValue": (
        [("us-gaap", "CashAndCashEquivalentsAtCarryingValue")],
        "USD",
    ),
    "LongTermDebt": ([("us-gaap", "LongTermDebt")], "USD"),
}
# Registrants whose CURRENT ticker CIK is a successor holding company; the predecessor's facts are used
# only for filings made before the successor's first filing (verified via data.sec.gov/submissions,
# 2026-11-25): XOM -> ExxonMobil Holdings Corp (CIK 2115436, filing since 2026-08) succeeds Exxon Mobil
# Corp (CIK 34088); DIS -> The Walt Disney Company (CIK 1744489, since 2019) succeeds TWDC Enterprises
# 18 Corp. (CIK 1001039, named "WALT DISNEY CO/" until 2019-03-18); GOOGL -> Alphabet Inc. (CIK 1652044,
# since 2015-10) succeeds Google Inc. (CIK 1288776).
PREDECESSOR_CIKS: dict[str, list[str]] = {
    "XOM": ["0000034088"],
    "DIS": ["0001001039"],
    "GOOGL": ["0001288776"],
}
# Latest period older than this (days before the session) -> feature is NaN, not carried forward.
# 10-K deadline <= 90 days after a <= 371-day fiscal year end, so a live filer is never this stale.
MAX_AGE_DAYS = 460
# Price-based ratios are NOT computed: SEC shares/EPS are Class-A equivalents (1 A = 1500 B since 2010)
# while the configured price is the B share.
NO_PRICE_RATIOS = {"BRK_B": "shares/EPS reported per Class A share; price is Class B"}
DURATION_CONCEPTS = ("Revenues", "NetIncomeLoss", "EarningsPerShareDiluted", "OperatingIncomeLoss")

FACT_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("cik", pa.string()),
        ("concept", pa.string()),
        ("taxonomy", pa.string()),
        ("tag", pa.string()),
        ("unit", pa.string()),
        ("value", pa.float64()),
        ("period_start", pa.date32()),
        ("period_end", pa.date32()),
        ("fiscal_year", pa.int32()),
        ("fiscal_period", pa.string()),
        ("form", pa.string()),
        ("filed", pa.date32()),
        ("accession", pa.string()),
        ("frame", pa.string()),
    ]
)
FACT_COLUMNS = [f.name for f in FACT_SCHEMA]


class SecAccessBlockedError(RuntimeError):
    """SEC answered 403 (User-Agent refused or client blocked)."""


# ----------------------------------------------------------------------------------- HTTP layer
def user_agent() -> str:
    return os.environ.get(UA_ENV) or DEFAULT_USER_AGENT


def user_agent_label() -> str:
    """What may be persisted about the User-Agent: never the env value itself."""
    return f"env:{UA_ENV}" if os.environ.get(UA_ENV) else DEFAULT_USER_AGENT


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent(), "Accept-Encoding": "gzip, deflate"})
    return s


class _Throttle:
    def __init__(self, min_interval: float = MIN_INTERVAL_S) -> None:
        self.min_interval = min_interval
        self._last = 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.min_interval:
            time.sleep(self.min_interval - dt)
        self._last = time.monotonic()


_THROTTLE = _Throttle()


def _get(http: requests.Session, url: str, throttle: _Throttle | None = None) -> requests.Response:
    throttle = throttle or _THROTTLE
    r: requests.Response | None = None
    for attempt in range(MAX_ATTEMPTS):
        throttle.wait()
        r = http.get(url, timeout=TIMEOUT_S)
        if r.status_code == 403:
            raise SecAccessBlockedError(f"SEC returned 403 for {url} (User-Agent: {user_agent_label()})")
        if r.status_code not in RETRY_STATUS:
            return r
        retry_after = r.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else BACKOFF_S * 2**attempt
        if attempt < MAX_ATTEMPTS - 1:
            time.sleep(min(wait, 120.0))
    assert r is not None
    r.raise_for_status()
    return r


# ----------------------------------------------------------------------------------- raw layer
def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_raw_json(content: bytes, out_dir: Path, stem: str, url: str, extra: dict[str, Any]) -> dict:
    """Store gzip'd provider bytes + sidecar; reuse the latest file when the content is unchanged."""
    out_dir.mkdir(parents=True, exist_ok=True)
    content_sha = _sha256(content)
    metas = sorted(out_dir.glob(f"{stem}_*.meta.json"))
    if metas:
        last = json.loads(metas[-1].read_text(encoding="utf-8"))
        if last.get("content_sha256") == content_sha and (out_dir / last["file"]).exists():
            return {**last, "reused": True}
    now = datetime.now(UTC)
    fname = f"{stem}_{now.strftime('%Y%m%dT%H%M%SZ')}.json.gz"
    gz = gzip.compress(content, compresslevel=9, mtime=0)
    (out_dir / fname).write_bytes(gz)
    meta = {
        "provider": PROVIDER,
        "url": url,
        "retrieved_at_utc": now.isoformat(),
        "user_agent": user_agent_label(),
        "file": fname,
        "compression": "gzip",
        "bytes_uncompressed": len(content),
        "content_sha256": content_sha,
        "sha256": _sha256(gz),
        **extra,
    }
    (out_dir / fname.replace(".json.gz", ".meta.json")).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return {**meta, "reused": False}


def read_raw_json(path: Path) -> Any:
    return json.loads(gzip.decompress(path.read_bytes()))


def fetch_ticker_map(http: requests.Session, paths: Paths) -> tuple[dict[str, str], dict]:
    r = _get(http, TICKERS_URL)
    r.raise_for_status()
    meta = _write_raw_json(r.content, paths.raw / "sec" / "_reference", "company_tickers", TICKERS_URL, {})
    data = r.json()
    mapping = {str(v["ticker"]).upper(): f"{int(v['cik_str']):010d}" for v in data.values()}
    return mapping, meta


def sec_ticker(symbol: str) -> str:
    """ASTRA symbol -> SEC ticker (config provider_symbol, e.g. BRK_B -> BRK-B)."""
    try:
        return get_config().instrument(symbol).provider_symbol.upper()
    except KeyError:
        return symbol.replace("_", "-").upper()


def fetch_companyfacts(http: requests.Session, cik: str, paths: Paths) -> tuple[dict, dict]:
    url = FACTS_URL.format(cik=cik)
    r = _get(http, url)
    r.raise_for_status()
    meta = _write_raw_json(r.content, paths.raw / "sec" / cik, "companyfacts", url, {"cik": cik})
    return r.json(), meta


# ----------------------------------------------------------------------------------- normalize
def normalize_companyfacts(doc: dict, symbol: str) -> pd.DataFrame:
    """Every filing's value for the selected concepts (long, point-in-time; nothing collapsed)."""
    cik = f"{int(doc['cik']):010d}"
    facts = doc.get("facts", {})
    rows: list[dict[str, Any]] = []
    for concept, (tags, unit) in CONCEPTS.items():
        for taxonomy, tag in tags:
            node = facts.get(taxonomy, {}).get(tag)
            if not node:
                continue
            for item in node.get("units", {}).get(unit, []):
                rows.append(
                    {
                        "symbol": symbol,
                        "cik": cik,
                        "concept": concept,
                        "taxonomy": taxonomy,
                        "tag": tag,
                        "unit": unit,
                        "value": item.get("val"),
                        "period_start": item.get("start"),
                        "period_end": item.get("end"),
                        "fiscal_year": item.get("fy"),
                        "fiscal_period": item.get("fp"),
                        "form": item.get("form"),
                        "filed": item.get("filed"),
                        "accession": item.get("accn"),
                        "frame": item.get("frame"),
                    }
                )
    df = pd.DataFrame(rows, columns=FACT_COLUMNS)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
    for c in ("period_start", "period_end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.date
    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce").astype("Int32")
    return df


def _drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = ["symbol", "tag", "unit", "period_start", "period_end", "accession", "value", "filed"]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first")
    return df, before - len(df)


def _sort_facts(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["symbol", "concept", "period_end", "period_start", "filed", "accession"], na_position="first"
    ).reset_index(drop=True)


# ----------------------------------------------------------------------------------- quality
def quality_report_facts(
    df: pd.DataFrame, exact_dups_removed: int = 0, symbols: Iterable[str] = ()
) -> tuple[QualityReport, list[dict[str, Any]]]:
    checks: list[CheckResult] = []
    required = ["symbol", "cik", "concept", "unit", "value", "period_end", "filed", "accession"]
    nulls = df[required].isna().sum()
    bad = nulls[nulls > 0]
    checks.append(
        CheckResult(
            name="nulls",
            status="fail" if len(bad) else "pass",
            count=int(bad.sum()),
            details=repr(bad.to_dict()) if len(bad) else "no nulls in required columns",
        )
    )
    dup_key = ["symbol", "tag", "unit", "period_start", "period_end", "accession"]
    conflicting = df[df.duplicated(dup_key, keep=False)]
    conflicting = conflicting.groupby(dup_key, dropna=False)["value"].nunique()
    conflicting = conflicting[conflicting > 1]
    checks.append(
        CheckResult(
            name="duplicates",
            status="warn" if len(conflicting) else "pass",
            count=len(conflicting),
            details=f"{exact_dups_removed} exact duplicate rows removed; {len(conflicting)} (tag, period, "
            "accession) keys carry >1 distinct value",
            examples=[str(k) for k in list(conflicting.index)[:5]],
        )
    )
    dur = df.dropna(subset=["period_start"])
    inverted = dur[pd.to_datetime(dur["period_start"]) > pd.to_datetime(dur["period_end"])]
    checks.append(
        CheckResult(
            name="period_order",
            status="fail" if len(inverted) else "pass",
            count=len(inverted),
            details="period_start <= period_end",
            examples=[str(r) for r in inverted.head(3).to_dict("records")],
        )
    )
    early = df[pd.to_datetime(df["filed"]) < pd.to_datetime(df["period_end"])]
    checks.append(
        CheckResult(
            name="filed_after_period_end",
            status="warn" if len(early) else "pass",
            count=len(early),
            details="rows with filed < period_end (kept; availability is keyed on filed)",
            examples=[
                f"{r.symbol} {r.concept} end={r.period_end} filed={r.filed} {r.form}"
                for r in early.head(5).itertuples()
            ],
        )
    )
    filed = pd.to_datetime(df["filed"])
    out_of_range = df[(filed < pd.Timestamp("2009-01-01")) | (filed > pd.Timestamp(datetime.now(UTC).date()))]
    sorted_ok = df.equals(_sort_facts(df)) if len(df) else True
    checks.append(
        CheckResult(
            name="monotonic_dates",
            status="pass" if sorted_ok and not len(out_of_range) else "warn",
            count=len(out_of_range),
            details=f"table sorted by symbol/concept/period_end/filed: {sorted_ok}; filed outside "
            f"[2009-01-01, today]: {len(out_of_range)}",
        )
    )
    distinct = df.drop_duplicates(["symbol", "concept", "period_start", "period_end", "value"])
    pkey = ["symbol", "concept", "period_start", "period_end"]
    restated = len(distinct.loc[distinct.duplicated(pkey), pkey].drop_duplicates())
    checks.append(
        CheckResult(
            name="restatements",
            status="pass",
            count=int(restated),
            details="(symbol, concept, period) with >1 distinct filed value; first-filed is used",
        )
    )
    coverage: list[dict[str, Any]] = []
    for sym in sorted(set(symbols) | set(df["symbol"].unique())):
        g = df[df["symbol"] == sym]
        present = sorted(g["concept"].unique())
        coverage.append(
            {
                "symbol": sym,
                "rows": len(g),
                "first_filed": str(min(g["filed"])) if len(g) else None,
                "last_filed": str(max(g["filed"])) if len(g) else None,
                "concepts_missing": [c for c in CONCEPTS if c not in present],
                "revenue_tags": sorted(g.loc[g["concept"] == "Revenues", "tag"].unique()),
            }
        )
    missing_symbols = [c["symbol"] for c in coverage if c["rows"] == 0]
    checks.append(
        CheckResult(
            name="coverage",
            status="warn" if missing_symbols or any(c["concepts_missing"] for c in coverage) else "pass",
            count=len(missing_symbols),
            details=f"{len(coverage) - len(missing_symbols)}/{len(coverage)} symbols with facts; "
            f"{sum(bool(c['concepts_missing']) for c in coverage)} symbols missing >=1 concept",
            examples=missing_symbols[:5],
        )
    )
    rep = QualityReport(
        symbol="US_STOCKS",
        market="US",
        provider=PROVIDER,
        freq="fundamentals",
        rows=len(df),
        first_session=str(min(df["filed"])) if len(df) else None,
        last_session=str(max(df["filed"])) if len(df) else None,
        created_at_utc=datetime.now(UTC).isoformat(),
        checks=checks,
    )
    return rep, coverage


def write_quality_report(rep: QualityReport, coverage: list[dict], paths: Paths, name: str) -> Path:
    d = paths.reports / "quality"
    d.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromisoformat(rep.created_at_utc).strftime("%Y%m%dT%H%M%SZ")
    p = d / f"{name}_{stamp}.json"
    payload = {**json.loads(rep.model_dump_json()), "status": rep.status, "coverage": coverage}
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


# ----------------------------------------------------------------------------------- ingestion
def facts_path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).validated / "fundamentals" / "us_facts.parquet"


def default_symbols() -> list[str]:
    return get_config().symbols(market="US", kind="stock")


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_sec(
    symbols: list[str] | None = None,
    paths: Paths | None = None,
    http: requests.Session | None = None,
) -> dict[str, Any]:
    """Download companyfacts for US stocks, store raw + validated long table + quality report."""
    t0 = time.monotonic()
    paths = paths or get_paths()
    symbols = list(symbols) if symbols else default_symbols()
    http = http or make_session()
    result: dict[str, Any] = {"provider": PROVIDER, "user_agent": user_agent_label(), "symbols": symbols}
    try:
        mapping, tick_meta = fetch_ticker_map(http, paths)
    except SecAccessBlockedError as e:
        return {**result, "blocked": True, "error": str(e), "runtime_s": round(time.monotonic() - t0, 2)}
    result["ticker_map_sha256"] = tick_meta["sha256"]
    frames: list[pd.DataFrame] = []
    per_symbol: dict[str, Any] = {}
    for sym in symbols:
        cik = mapping.get(sec_ticker(sym))
        if cik is None:
            per_symbol[sym] = {"status": "no_cik", "sec_ticker": sec_ticker(sym)}
            continue
        try:
            doc, meta = fetch_companyfacts(http, cik, paths)
            preds = [(p, *fetch_companyfacts(http, p, paths)) for p in PREDECESSOR_CIKS.get(sym, [])]
        except SecAccessBlockedError as e:
            result.update(blocked=True, error=str(e))
            break
        except requests.HTTPError as e:
            per_symbol[sym] = {"status": "http_error", "cik": cik, "error": str(e)}
            continue
        df = normalize_companyfacts(doc, sym)
        info: dict[str, Any] = {
            "status": "ok",
            "cik": cik,
            "entity": doc.get("entityName"),
            "raw_file": f"raw/sec/{cik}/{meta['file']}",
            "raw_sha256": meta["sha256"],
            "raw_reused": meta["reused"],
        }
        if preds and len(df):
            cutoff = min(df["filed"])
            info["predecessors"] = []
            for p, pdoc, pmeta in preds:
                pdf = normalize_companyfacts(pdoc, sym)
                pdf = pdf[pdf["filed"] < cutoff]  # predecessor only until the successor's first filing
                df = pd.concat([pdf, df], ignore_index=True)
                info["predecessors"].append(
                    {
                        "cik": p,
                        "entity": pdoc.get("entityName"),
                        "rows_used": len(pdf),
                        "filed_before": str(cutoff),
                        "raw_file": f"raw/sec/{p}/{pmeta['file']}",
                        "raw_sha256": pmeta["sha256"],
                    }
                )
        info["rows"] = len(df)
        frames.append(df)
        per_symbol[sym] = info
    result.setdefault("blocked", False)
    new = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FACT_COLUMNS)
    out = facts_path(paths)
    if out.exists():  # keep other symbols from earlier runs; replace the ones fetched now
        old = pq.read_table(out).to_pandas()
        old = old[~old["symbol"].isin(new["symbol"].unique())]
        for c in ("period_start", "period_end", "filed"):
            old[c] = pd.to_datetime(old[c]).dt.date
        new = pd.concat([old, new], ignore_index=True) if len(old) else new
    new, n_dups = _drop_exact_duplicates(new)
    new = _sort_facts(new)
    new["fiscal_year"] = new["fiscal_year"].astype("Int32")
    rep, coverage = quality_report_facts(new, n_dups, symbols)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(new[FACT_COLUMNS], schema=FACT_SCHEMA, preserve_index=False),
        out,
        compression="zstd",
    )
    rpath = write_quality_report(rep, coverage, paths, "US_sec_companyfacts")
    manifest = {
        "dataset": "us_facts",
        "provider": PROVIDER,
        "rows": len(new),
        "symbols": sorted(new["symbol"].unique().tolist()),
        "first_filed": rep.first_session,
        "last_filed": rep.last_session,
        "sha256": _file_sha(out),
        "quality_status": rep.status,
        "quality_report": rpath.name,
        "raw": {s: v.get("raw_sha256") for s, v in per_symbol.items()},
        "user_agent": user_agent_label(),
        "written_at_utc": datetime.now(UTC).isoformat(),
        "schema_version": 1,
        "notes": ["long table of every filing's value; select first-filed per period for PIT use"],
    }
    out.with_name("us_facts.manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result.update(
        per_symbol=per_symbol,
        rows=len(new),
        validated=str(out),
        sha256=manifest["sha256"],
        quality_report=str(rpath),
        quality_status=rep.status,
        coverage=coverage,
        runtime_s=round(time.monotonic() - t0, 2),
    )
    return result


def load_facts(symbol: str | None = None, paths: Paths | None = None) -> pd.DataFrame:
    from astra.data.lock import resolve

    df = pq.read_table(resolve(facts_path(paths), paths)).to_pandas()
    if symbol is not None:
        df = df[df["symbol"] == symbol].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------------- PIT logic
def first_filed(facts: pd.DataFrame) -> pd.DataFrame:
    """One row per (concept, period): the value from the EARLIEST filing (ties: tag priority, then
    original over amendment, then accession). Later restatements are ignored by construction."""
    if facts.empty:
        return facts.copy()
    df = facts.copy()
    prio = {(c, tag): i for c, (tags, _) in CONCEPTS.items() for i, (_, tag) in enumerate(tags)}
    df["_prio"] = [prio.get((c, t), 99) for c, t in zip(df["concept"], df["tag"], strict=True)]
    df["_amend"] = df["form"].fillna("").str.endswith("/A").astype(int)
    df["p_start"] = pd.to_datetime(df["period_start"])
    df["p_end"] = pd.to_datetime(df["period_end"])
    df["p_filed"] = pd.to_datetime(df["filed"])
    df = df.dropna(subset=["value", "p_end", "p_filed"])
    df = df.sort_values(
        ["concept", "p_start", "p_end", "p_filed", "_prio", "_amend", "accession"], na_position="first"
    )
    df = df.drop_duplicates(subset=["concept", "p_start", "p_end"], keep="first")
    return df.reset_index(drop=True)


def _days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return int((end - start).days)


def _n_quarters(days: int) -> int | None:
    if 80 <= days <= 100:
        return 1
    if 170 <= days <= 195:
        return 2
    if 260 <= days <= 285:
        return 3
    if 350 <= days <= 380:
        return 4
    return None


def _near(a: pd.Timestamp, b: pd.Timestamp, tol: int) -> bool:
    return abs((a - b).days) <= tol


def discrete_quarters(ff: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Quarterly values from first-filed duration facts of ONE concept: direct 3-month facts, plus
    quarters derived as YTD(k) - YTD(k-1) with the same start (e.g. Q4 = FY - 9M). Columns:
    start, end, value, avail (latest component filing date), derived (bool)."""
    d = ff.dropna(subset=["p_start"]).copy()
    d["nq"] = [_n_quarters(_days(s, e)) for s, e in zip(d["p_start"], d["p_end"], strict=True)]
    d = d.dropna(subset=["nq"])
    q = [
        {
            "start": r.p_start,
            "end": r.p_end,
            "value": getattr(r, value_col),
            "avail": r.p_filed,
            "derived": False,
        }
        for r in d[d["nq"] == 1].itertuples()
    ]
    ytd = d[d["nq"] > 1]
    for r in ytd.itertuples():
        if any(_near(x["end"], r.p_end, 3) for x in q):
            continue
        prev = d[(d["nq"] == r.nq - 1)]
        prev = prev[[_near(s, r.p_start, 3) for s in prev["p_start"]]]
        if prev.empty:
            continue
        p = prev.iloc[0]
        q.append(
            {
                "start": p["p_end"] + pd.Timedelta(days=1),
                "end": r.p_end,
                "value": getattr(r, value_col) - p[value_col],
                "avail": max(r.p_filed, p["p_filed"]),
                "derived": True,
            }
        )
    out = pd.DataFrame(q, columns=["start", "end", "value", "avail", "derived"])
    return out.sort_values(["end", "avail"]).drop_duplicates("end", keep="first").reset_index(drop=True)


def ttm(ff: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Trailing-twelve-month values per quarter end. A first-filed annual (FY) fact is used when its
    end matches; otherwise the sum of 4 contiguous discrete quarters. avail = latest component filing."""
    qs = discrete_quarters(ff, value_col)
    d = ff.dropna(subset=["p_start"])
    fy = d[[(_n_quarters(_days(s, e)) == 4) for s, e in zip(d["p_start"], d["p_end"], strict=True)]]
    rows: dict[pd.Timestamp, dict] = {}
    for r in fy.itertuples():
        rows[r.p_end] = {"end": r.p_end, "ttm": getattr(r, value_col), "avail": r.p_filed, "source": "FY"}
    for i in range(len(qs)):
        end = qs.loc[i, "end"]
        if any(_near(end, k, 3) for k in rows):
            continue
        chain = [qs.loc[i]]
        while len(chain) < 4:
            want = chain[-1]["start"] - pd.Timedelta(days=1)
            cand = qs[[_near(e, want, 5) for e in qs["end"]]]
            if cand.empty:
                break
            chain.append(cand.iloc[0])
        if len(chain) == 4:
            rows[end] = {
                "end": end,
                "ttm": float(sum(c["value"] for c in chain)),
                "avail": max(c["avail"] for c in chain),
                "source": "4Q",
            }
    out = pd.DataFrame(list(rows.values()), columns=["end", "ttm", "avail", "source"])
    return out.sort_values("end").reset_index(drop=True)


def _yoy(t: pd.DataFrame) -> pd.DataFrame:
    """TTM growth vs the TTM ending ~1 year earlier (365 +- 10 days)."""
    rows = []
    for r in t.itertuples():
        prev = t[[_near(e, r.end - pd.Timedelta(days=365), 10) for e in t["end"]]]
        if prev.empty or prev.iloc[0]["ttm"] == 0 or not np.isfinite(prev.iloc[0]["ttm"]):
            continue
        p = prev.iloc[0]
        rows.append(
            {
                "end": r.end,
                "value": r.ttm / abs(p["ttm"]) - 1.0 if p["ttm"] > 0 else np.nan,
                "avail": max(r.avail, p["avail"]),
            }
        )
    return pd.DataFrame(rows, columns=["end", "value", "avail"])


def _asof(
    sessions: pd.Series, events: pd.DataFrame, col: str = "value", max_age_days: int | None = MAX_AGE_DAYS
) -> pd.Series:
    """For each session date, the value of the LATEST period (max `end`) whose `avail` (filing date)
    is STRICTLY before the session. NaN when that period ended more than `max_age_days` before the
    session (a tag the filer stopped using must not be carried forward for years)."""
    sess = pd.to_datetime(pd.Series(sessions)).reset_index(drop=True).astype("datetime64[ns]")
    if events is None or events.empty:
        return pd.Series(np.nan, index=sess.index)
    ev = events.dropna(subset=[col]).sort_values(["avail", "end"]).reset_index(drop=True)
    if ev.empty:
        return pd.Series(np.nan, index=sess.index)
    state, best_end, best_val = [], None, np.nan
    for r in ev.itertuples():
        if best_end is None or r.end >= best_end:
            best_end, best_val = r.end, getattr(r, col)
        state.append({"key": r.avail + pd.Timedelta(days=1), "v": best_val, "end": best_end})
    st = pd.DataFrame(state).drop_duplicates("key", keep="last")
    st["key"] = st["key"].astype("datetime64[ns]")
    st["end"] = pd.to_datetime(st["end"]).astype("datetime64[ns]")
    left = pd.DataFrame({"key": sess, "_i": np.arange(len(sess))}).sort_values("key")
    m = pd.merge_asof(left, st, on="key", direction="backward").sort_values("_i").reset_index(drop=True)
    v = m["v"].astype("float64")
    if max_age_days is not None:
        v = v.where((m["key"] - m["end"]).dt.days <= max_age_days)
    return v


def split_factor_after(prices: pd.DataFrame) -> Callable[[pd.Timestamp], float]:
    """F(d) = product of split ratios effective on sessions STRICTLY AFTER d (converts a per-share
    figure reported on date d to the current, fully split-adjusted share basis of Yahoo `close`).
    NaN when d precedes the first price session (splits before that are unknown)."""
    p = prices[["session_date"]].copy()
    p["d"] = pd.to_datetime(p["session_date"])
    sr = prices["split_ratio"] if "split_ratio" in prices else pd.Series(1.0, index=prices.index)
    p["sr"] = pd.to_numeric(sr, errors="coerce").fillna(1.0).replace(0.0, 1.0).to_numpy()
    p = p.sort_values("d").reset_index(drop=True)
    suffix = np.cumprod(p["sr"].to_numpy()[::-1])[::-1]  # product over sessions >= i
    dates = p["d"].to_numpy()

    def f(d: pd.Timestamp) -> float:
        if len(dates) == 0 or d < dates[0]:
            return np.nan
        i = int(np.searchsorted(dates, np.datetime64(d), side="right"))
        return 1.0 if i >= len(dates) else float(suffix[i])

    return f


def fundamental_features(
    symbol: str, prices_df: pd.DataFrame, facts: pd.DataFrame | None = None, paths: Paths | None = None
) -> pd.DataFrame:
    """Daily as-of fundamental features for `symbol` on the sessions of `prices_df` (needs
    `session_date`, `close`; `split_ratio` used for per-share adjustments).

    A value is visible on session s only if every component was filed STRICTLY before s.
    Columns: ttm_revenue, ttm_net_income, revenue_growth_yoy, net_margin, roe, leverage,
    ttm_eps_diluted (current share basis), earnings_yield (EPS-based E/P, preferred),
    market_cap, earnings_yield_ni (TTM NI / market cap), book_to_market, fund_period_end, fund_filed.
    """
    if facts is None:
        facts = load_facts(symbol, paths)
    else:
        facts = pd.DataFrame(facts[facts["symbol"] == symbol]) if "symbol" in facts else facts
    prices = prices_df.sort_values("session_date").reset_index(drop=True)
    sessions = pd.to_datetime(prices["session_date"])
    out = pd.DataFrame({"session_date": prices["session_date"].to_numpy()})
    ff = first_filed(facts)
    F = split_factor_after(prices)

    def concept(name: str) -> pd.DataFrame:
        return ff[ff["concept"] == name].copy()

    def inst(name: str) -> pd.DataFrame:
        c = concept(name)
        return pd.DataFrame({"end": c["p_end"], "value": c["value"], "avail": c["p_filed"]})

    rev_t, ni_t = ttm(concept("Revenues")), ttm(concept("NetIncomeLoss"))
    eps = concept("EarningsPerShareDiluted")
    eps["adj"] = [v / F(d) for v, d in zip(eps["value"], eps["p_filed"], strict=True)]
    eps_t = ttm(eps, "adj")
    shares = concept("CommonStockSharesOutstanding")
    shares_ev = pd.DataFrame(
        {
            "end": shares["p_end"],
            "value": [v * F(d) for v, d in zip(shares["value"], shares["p_filed"], strict=True)],
            "avail": shares["p_filed"],
        }
    )
    eq, assets, liab = inst("StockholdersEquity"), inst("Assets"), inst("Liabilities")

    def ev(t: pd.DataFrame) -> pd.DataFrame:
        return t.rename(columns={"ttm": "value"})[["end", "value", "avail"]]

    def combine(a: pd.DataFrame, b: pd.DataFrame, fn, tol: int = 3) -> pd.DataFrame:
        rows = []
        for r in a.itertuples():
            m = b[[_near(e, r.end, tol) for e in b["end"]]]
            if m.empty:
                continue
            x = m.iloc[0]
            rows.append({"end": r.end, "value": fn(r.value, x["value"]), "avail": max(r.avail, x["avail"])})
        return pd.DataFrame(rows, columns=["end", "value", "avail"])

    def safe_div(a: float, b: float) -> float:
        return a / b if b and np.isfinite(b) and b != 0 else np.nan

    out["ttm_revenue"] = _asof(sessions, ev(rev_t)).to_numpy()
    out["ttm_net_income"] = _asof(sessions, ev(ni_t)).to_numpy()
    out["revenue_growth_yoy"] = _asof(sessions, _yoy(rev_t)).to_numpy()
    out["net_margin"] = _asof(sessions, combine(ev(ni_t), ev(rev_t), safe_div)).to_numpy()
    out["roe"] = _asof(
        sessions, combine(ev(ni_t), eq, lambda n, e: safe_div(n, e) if e > 0 else np.nan)
    ).to_numpy()
    out["leverage"] = _asof(sessions, combine(liab, assets, safe_div, tol=0)).to_numpy()
    out["ttm_eps_diluted"] = _asof(sessions, ev(eps_t)).to_numpy()
    close = pd.to_numeric(prices["close"], errors="coerce").to_numpy()
    out["earnings_yield"] = out["ttm_eps_diluted"].to_numpy() / close
    sh = _asof(sessions, shares_ev).to_numpy()
    if symbol in NO_PRICE_RATIOS:
        sh = np.full(len(sessions), np.nan)
        out["earnings_yield"] = np.nan
    out["market_cap"] = sh * close
    out["earnings_yield_ni"] = out["ttm_net_income"].to_numpy() / out["market_cap"].to_numpy()
    out["book_to_market"] = _asof(sessions, eq).to_numpy() / out["market_cap"].to_numpy()
    nev = ev(ni_t)
    out["fund_period_end"] = _asof(
        sessions, nev.assign(value=nev["end"].map(pd.Timestamp.toordinal).astype(float))
    )
    out["fund_period_end"] = [
        date.fromordinal(int(v)) if np.isfinite(v) else None for v in out["fund_period_end"]
    ]
    out["fund_filed"] = _asof(
        sessions, nev.assign(value=nev["avail"].map(pd.Timestamp.toordinal).astype(float))
    )
    out["fund_filed"] = [date.fromordinal(int(v)) if np.isfinite(v) else None for v in out["fund_filed"]]
    out.insert(1, "symbol", symbol)
    return out
