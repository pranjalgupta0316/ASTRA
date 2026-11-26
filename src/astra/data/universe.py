"""Point-in-time S&P 500 membership (P8) reconstructed from Wikipedia, and a survivorship check.

Why: ASTRA's US universe is today's well-known large caps, so every cross-sectional or portfolio result
on it carries survivorship bias of unknown size (KNOWN_ISSUES). Historical index membership is the
standard point-in-time universe; free sources are scarce. Wikipedia keeps the current constituents and a
table of index changes (added / removed with effective dates). Walking the changes backwards from today
reconstructs membership on any past date.

Source and licence: en.wikipedia.org pages "List of S&P 500 companies" and "Historical components of
the S&P 500", text under CC BY-SA 4.0 (attribution: Wikipedia contributors). Fetched through the public
MediaWiki API, two requests, generic User-Agent. The page revision ids are stored so the exact input can
be re-fetched (`oldid`). Raw JSON stays local (git-ignored); only derived counts are published.

Limits (stated, not hidden): the changes table is incomplete in early years (2-13 events a year before
2011 vs ~20-30 actual), so reconstruction is trusted only from the first year after which every year has
>= 15 recorded changes (`reliable_from`). The member count cannot serve as that check: paired add/remove
events keep it near 500 even when events are missing. Ticker renames the table does not record (e.g.
FB -> META) are mapped through RENAMES; unmatched events are counted as `orphans` in the manifest.
Membership is not tradability: prices of removed / delisted names are mostly unavailable from free
sources, so this module measures the bias; it cannot remove it.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from astra.common.paths import Paths, get_paths

API = "https://en.wikipedia.org/w/api.php"
PAGES = {"constituents": "List_of_S&P_500_companies", "changes": "Historical_components_of_the_S&P_500"}
USER_AGENT = "ASTRA-research/0.3 (personal non-commercial research)"
LICENCE = "CC BY-SA 4.0 - Wikipedia contributors"
# ASTRA symbol -> tickers it has had in the index (old first)
ALIASES: dict[str, list[str]] = {"BRK_B": ["BRK.B"], "META": ["FB", "META"], "GOOGL": ["GOOGL", "GOOG"]}
MIN_CHANGES_PER_YEAR = 15
# Ticker changes the changes table does not record as index events: old -> current ticker. Unmatched
# adds/removes (`orphans` in the manifest) show where more entries are needed.
RENAMES: dict[str, str] = {
    "FB": "META",
    "PCLN": "BKNG",
    "CDAY": "DAY",
    "FLT": "CPAY",
    "RE": "EG",
    "KORS": "CPRI",
    "DLPH": "APTV",
}


def raw_dir(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).raw / "wikipedia" / "sp500"


def out_path(paths: Paths | None = None) -> Path:
    return (paths or get_paths()).validated / "universe" / "sp500_membership.parquet"


def fetch_pages(paths: Paths | None = None) -> dict[str, Path]:
    """Download both pages (parsed HTML + revision id) and store them immutably by revision id."""
    import requests

    paths = paths or get_paths()
    out: dict[str, Path] = {}
    for key, page in PAGES.items():
        r = requests.get(
            API,
            params={
                "action": "parse",
                "page": page,
                "prop": "text|revid",
                "format": "json",
                "formatversion": 2,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        r.raise_for_status()
        body = r.content
        parsed = json.loads(body)["parse"]
        dest = raw_dir(paths) / f"{key}_{parsed['revid']}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(body)
        meta = {
            "page": page,
            "revid": parsed["revid"],
            "fetched_at_utc": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(body).hexdigest(),
            "licence": LICENCE,
            "url": f"https://en.wikipedia.org/w/index.php?title={page}&oldid={parsed['revid']}",
        }
        dest.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        out[key] = dest
    return out


def latest_raw(key: str, paths: Paths | None = None) -> Path:
    """Newest stored revision of a page (excludes the .meta.json sidecars)."""
    cands = [p for p in raw_dir(paths).glob(f"{key}_*.json") if not p.name.endswith(".meta.json")]
    if not cands:
        raise FileNotFoundError(f"no stored {key} page under {raw_dir(paths)}; run fetch_pages()")
    return max(cands, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def _html(path: Path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["parse"]["text"]


def _norm(t: Any) -> str | None:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    s = re.sub(r"[^A-Z0-9.]", "", str(t).strip().upper())
    return RENAMES.get(s, s) or None


def parse_constituents(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})
    df = tables[0]
    return pd.DataFrame({"ticker": df["Symbol"].map(_norm), "security": df["Security"]}).dropna(
        subset=["ticker"]
    )


def parse_changes(html: str) -> pd.DataFrame:
    """One row per change: date, added ticker (or None), removed ticker (or None)."""
    df = max(pd.read_html(io.StringIO(html)), key=len)
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("unexpected changes table layout (expected two header rows)")
    cols = {c: c for c in df.columns}
    date_col = next(c for c in cols if "date" in str(c[0]).lower())
    add_col = next(c for c in cols if str(c[0]).lower() == "added" and str(c[1]).lower() == "ticker")
    rem_col = next(c for c in cols if str(c[0]).lower() == "removed" and str(c[1]).lower() == "ticker")
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df[date_col], errors="coerce", format="mixed"),
            "added": df[add_col].map(_norm),
            "removed": df[rem_col].map(_norm),
        }
    )
    bad = out["date"].isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} change rows with unparseable dates")
    return out.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)


def reconstruct(
    current: set[str], changes: pd.DataFrame, today: pd.Timestamp
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Membership intervals [start, end) per ticker by walking changes backwards from `today`.

    Effective date d: a ticker added on d is a member from d; a ticker removed on d is a member until d
    (exclusive). Intervals whose start is unknown (member before the earliest change) get start = NaT."""
    members = set(current)
    open_end: dict[str, pd.Timestamp | None] = dict.fromkeys(members)  # currently members: end = None
    rows: list[dict[str, Any]] = []
    orphans: list[dict[str, Any]] = []
    for d, grp in changes.groupby("date", sort=False):
        d = pd.Timestamp(d)
        for a_raw, r_raw in zip(grp["added"], grp["removed"], strict=True):
            a = a_raw if isinstance(a_raw, str) else None  # empty cells arrive as float NaN
            r = r_raw if isinstance(r_raw, str) else None
            if a and a in members:  # before d it was not a member: close its interval at start = d
                rows.append({"ticker": a, "start": d, "end": open_end.pop(a, None)})
                members.discard(a)
            elif a:  # added on d but not a member afterwards: unrecorded rename/removal later
                orphans.append({"date": d, "ticker": a, "event": "added_not_member_after"})
            if r and r not in members:  # before d it was a member: open an interval ending at d
                members.add(r)
                open_end[r] = d
            elif r:
                orphans.append({"date": d, "ticker": r, "event": "removed_but_member_after"})
    for t in members:
        rows.append({"ticker": t, "start": pd.NaT, "end": open_end.get(t)})
    iv = pd.DataFrame(rows)
    iv["end"] = pd.to_datetime(iv["end"])
    iv["start"] = pd.to_datetime(iv["start"])
    iv = iv.sort_values(["ticker", "start"], na_position="first").reset_index(drop=True)
    return iv, orphans


def members_on(intervals: pd.DataFrame, date: str | pd.Timestamp) -> set[str]:
    d = pd.Timestamp(date)
    m = (intervals["start"].isna() | (intervals["start"] <= d)) & (
        intervals["end"].isna() | (intervals["end"] > d)
    )
    return set(intervals.loc[m, "ticker"])


def member_counts(intervals: pd.DataFrame, changes: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    dates = sorted({*changes["date"], today})
    return pd.Series({d: len(members_on(intervals, d)) for d in dates}).sort_index()


def reliable_from(changes: pd.DataFrame, min_per_year: int = MIN_CHANGES_PER_YEAR) -> pd.Timestamp | None:
    """Start of the first year from which every calendar year records >= `min_per_year` index changes.

    The member count is NOT a completeness check: paired add/remove events conserve it by construction,
    so a reconstruction with missing events still shows ~500 members. Change density is: the S&P 500
    has had roughly 20-30 changes a year, so sparse years mean unrecorded events."""
    per_year = changes.groupby(changes["date"].dt.year).size()
    years = range(int(per_year.index.min()), int(per_year.index.max()) + 1)
    per_year = per_year.reindex(years, fill_value=0)
    sparse = per_year[per_year < min_per_year]
    first = int(sparse.index.max()) + 1 if len(sparse) else int(per_year.index.min())
    return pd.Timestamp(f"{first}-01-01") if first <= per_year.index.max() else None


def build(paths: Paths | None = None, files: dict[str, Path] | None = None) -> dict[str, Any]:
    """Parse the newest stored revisions, reconstruct, validate, write validated parquet + manifest."""
    paths = paths or get_paths()
    if files is None:
        files = {k: latest_raw(k, paths) for k in PAGES}
    cur = parse_constituents(_html(files["constituents"]))
    ch = parse_changes(_html(files["changes"]))
    meta = json.loads(files["constituents"].with_suffix(".meta.json").read_text(encoding="utf-8"))
    today = pd.Timestamp(meta["fetched_at_utc"][:10])
    iv, orphans = reconstruct(set(cur["ticker"]), ch, today)
    counts = member_counts(iv, ch, today)
    rel = reliable_from(ch)
    out = out_path(paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    iv.to_parquet(out)
    manifest = {
        "dataset": "sp500_membership",
        "rows": len(iv),
        "n_current": len(cur),
        "n_changes": len(ch),
        "changes_span": [str(ch["date"].min().date()), str(ch["date"].max().date())],
        "reliable_from": str(rel.date()) if rel is not None else None,
        "count_min_max": [int(counts.min()), int(counts.max())],
        "changes_per_year": {int(k): int(v) for k, v in ch.groupby(ch["date"].dt.year).size().items()},
        "orphans_total": len(orphans),
        "orphans_after_reliable": [
            {**o, "date": str(o["date"].date())} for o in orphans if rel is not None and o["date"] >= rel
        ],
        "sources": {
            k: json.loads(p.with_suffix(".meta.json").read_text(encoding="utf-8")) for k, p in files.items()
        },
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    (out.parent / "sp500_membership.manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )
    return manifest


def survivorship_report(dates: list[str], paths: Paths | None = None) -> pd.DataFrame:
    """For each date D: index members on D, how many are still members today, and how many of ASTRA's US
    single stocks were members on D (the rest entered the index later = picked with hindsight)."""
    from astra.common.config import get_config
    from astra.data.lock import resolve

    paths = paths or get_paths()
    iv = pd.read_parquet(resolve(out_path(paths), paths))
    man = json.loads(
        resolve(out_path(paths).parent / "sp500_membership.manifest.json", paths).read_text("utf-8")
    )
    rel = pd.Timestamp(man["reliable_from"]) if man.get("reliable_from") else None
    cfg = get_config()
    stocks = [s for s in cfg.symbols() if cfg.instrument(s).market == "US" and s in _single_stocks(cfg)]
    now = members_on(
        iv, iv["end"].max() + pd.Timedelta(days=1) if iv["end"].notna().any() else pd.Timestamp.now()
    )
    rows = []
    for d in dates:
        m = members_on(iv, d)
        ours = [s for s in stocks if any(t in m for t in ALIASES.get(s, [s]))]
        rows.append(
            {
                "date": d,
                "reliable": bool(rel is not None and pd.Timestamp(d) >= rel),
                "members": len(m),
                "still_members_today": len(m & now),
                "share_removed_since": 1 - len(m & now) / len(m) if m else float("nan"),
                "astra_us_stocks": len(stocks),
                "astra_stocks_member_on_date": len(ours),
                "astra_stocks_not_yet_members": sorted(set(stocks) - set(ours)),
            }
        )
    return pd.DataFrame(rows)


def _single_stocks(cfg: Any) -> set[str]:
    return {
        s for s in cfg.symbols() if cfg.instrument(s).market == "US" and cfg.instrument(s).kind == "stock"
    }
