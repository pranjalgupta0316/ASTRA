"""Offline tests for SEC fundamentals and macro (FRED manual CSV, Yahoo market macro, BLS);
synthetic fixtures, no network."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from astra.common.config import get_config
from astra.common.paths import get_paths
from astra.data.adapters.base import RawBatch
from astra.data.calendars import session_schedule
from astra.data.fundamentals import sec
from astra.data.macro import bls, fred, market
from astra.data.macro.features import derived_series, macro_features

# ------------------------------------------------------------------------------------ fixtures


def _f(start, end, val, filed, form="10-Q", accn=None, fy=2019, fp="Q1"):
    d = {
        "end": end,
        "val": val,
        "accn": accn or f"acc-{form}-{filed}",
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
    }
    if start:
        d["start"] = start
    return d


def companyfacts_doc() -> dict:
    """2019 quarters via 10-Q (3M + YTD), FY2019 via 10-K filed Wed 2020-02-05, Q1 2020 via 10-Q,
    then a 10-K/A (2020-06-01) and the FY2020 10-K (2021-02-03) RESTATING FY2019 net income."""
    ni = [
        _f("2019-01-01", "2019-03-31", 100, "2019-05-01"),
        _f("2019-04-01", "2019-06-30", 110, "2019-08-01", fp="Q2"),
        _f("2019-01-01", "2019-06-30", 210, "2019-08-01", fp="Q2"),
        _f("2019-07-01", "2019-09-30", 120, "2019-11-01", fp="Q3"),
        _f("2019-01-01", "2019-09-30", 330, "2019-11-01", fp="Q3"),
        _f("2019-01-01", "2019-12-31", 460, "2020-02-05", form="10-K", fp="FY"),
        _f("2020-01-01", "2020-03-31", 140, "2020-05-01", fy=2020),
        _f("2019-01-01", "2019-12-31", 999, "2020-06-01", form="10-K/A", fp="FY"),
        _f("2019-01-01", "2019-12-31", 888, "2021-02-03", form="10-K", fy=2020, fp="FY"),
    ]
    rev = [{**x, "val": x["val"] * 10} for x in ni[:7]]
    eps = [
        _f("2019-01-01", "2019-12-31", 4.0, "2020-02-05", form="10-K", fp="FY"),
    ]
    equity = [
        _f(None, "2019-12-31", 2000, "2020-02-05", form="10-K", fp="FY"),
    ]
    shares = [_f(None, "2020-01-31", 100, "2020-02-05", form="10-K", fp="FY")]
    return {
        "cik": 1,
        "entityName": "Synthetic Corp",
        "facts": {
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}},
            "us-gaap": {
                "NetIncomeLoss": {"units": {"USD": ni}},
                "Revenues": {"units": {"USD": rev}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": eps}},
                "StockholdersEquity": {"units": {"USD": equity}},
            },
        },
    }


def prices(split_on: str | None = None, ratio: float = 2.0) -> pd.DataFrame:
    sched = session_schedule(get_config().markets["US"], "2019-01-02", "2021-06-30")
    df = pd.DataFrame({"session_date": sched.index.date, "close": 50.0, "split_ratio": 1.0})
    if split_on:
        df.loc[df["session_date"] == pd.Timestamp(split_on).date(), "split_ratio"] = ratio
    return df


@pytest.fixture
def facts() -> pd.DataFrame:
    return sec.normalize_companyfacts(companyfacts_doc(), "SYN")


# ------------------------------------------------------------------------------------ SEC


def test_asof_never_before_filed_plus_one_session(facts):
    feat = sec.fundamental_features("SYN", prices(), facts=facts)
    has = feat.dropna(subset=["ttm_net_income"])
    assert len(has) > 0
    assert (pd.to_datetime(has["session_date"]) > pd.to_datetime(has["fund_filed"])).all()
    by_day = feat.set_index("session_date")
    d = pd.Timestamp
    # FY2019 10-K filed Wed 2020-02-05: invisible on the filing day, visible the next session
    assert by_day.loc[d("2020-02-05").date(), "ttm_net_income"] != 460
    assert by_day.loc[d("2020-02-06").date(), "ttm_net_income"] == 460
    # nothing at all before the first TTM could be built from filings
    assert feat.loc[pd.to_datetime(feat["session_date"]) <= d("2020-02-05"), "ttm_net_income"].isna().all()


def test_first_filed_value_used_when_later_10k_restates(facts):
    ff = sec.first_filed(facts)
    fy = ff[
        (ff["concept"] == "NetIncomeLoss") & (ff["p_start"] == "2019-01-01") & (ff["p_end"] == "2019-12-31")
    ]
    assert fy["value"].tolist() == [460.0]
    feat = sec.fundamental_features("SYN", prices(), facts=facts)
    assert not feat["ttm_net_income"].isin([999.0, 888.0]).any()
    by_day = feat.set_index("session_date")
    # after the 10-K/A (2020-06-01) and the restating 10-K (2021-02-03) the TTM still uses 460-based Q4
    assert by_day.loc[pd.Timestamp("2020-06-02").date(), "ttm_net_income"] == 500
    assert by_day.loc[pd.Timestamp("2021-02-04").date(), "ttm_net_income"] == 500


def test_ttm_from_quarterly_and_annual(facts):
    ff = sec.first_filed(facts)
    t = sec.ttm(ff[ff["concept"] == "NetIncomeLoss"]).set_index("end")
    fy = t.loc[pd.Timestamp("2019-12-31")]
    assert fy["ttm"] == 460 and fy["source"] == "FY" and fy["avail"] == pd.Timestamp("2020-02-05")
    # Q2'19 + Q3'19 + Q4'19 (= FY 460 - 9M 330) + Q1'20
    q1 = t.loc[pd.Timestamp("2020-03-31")]
    assert q1["ttm"] == 110 + 120 + 130 + 140 and q1["source"] == "4Q"
    assert q1["avail"] == pd.Timestamp("2020-05-01")
    q = sec.discrete_quarters(ff[ff["concept"] == "NetIncomeLoss"]).set_index("end")
    q4 = q.loc[pd.Timestamp("2019-12-31")]
    assert q4["value"] == 130 and bool(q4["derived"]) and q4["start"] == pd.Timestamp("2019-10-01")


def test_ratios_and_split_adjusted_eps(facts):
    feat = sec.fundamental_features("SYN", prices(split_on="2020-03-02", ratio=2.0), facts=facts)
    by_day = feat.set_index("session_date")
    row = by_day.loc[pd.Timestamp("2020-02-10").date()]
    assert row["net_margin"] == pytest.approx(0.1)
    assert row["roe"] == pytest.approx(460 / 2000)
    # EPS 4.0 filed before a 2:1 split -> 2.0 on the current (split-adjusted) share basis
    assert row["ttm_eps_diluted"] == pytest.approx(2.0)
    assert row["earnings_yield"] == pytest.approx(2.0 / 50.0)
    assert row["market_cap"] == pytest.approx(100 * 2 * 50.0)
    assert np.isnan(row["leverage"])  # Liabilities not provided -> no fabricated value


def test_sec_ticker_mapping():
    assert sec.sec_ticker("BRK_B") == "BRK-B"
    assert sec.sec_ticker("AAPL") == "AAPL"
    assert "SPY" not in sec.default_symbols() and "BRK_B" in sec.default_symbols()


class _Resp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self.content = json.dumps(payload).encode()
        self.status_code = status
        self.headers: dict[str, str] = {}

    def json(self) -> dict:
        return json.loads(self.content)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(self.status_code)


class _FakeHttp:
    def __init__(self, status: int = 200) -> None:
        self.urls: list[str] = []
        self.status = status
        self.headers = {"User-Agent": sec.user_agent()}

    def get(self, url: str, timeout: float) -> _Resp:
        assert timeout and timeout > 0
        self.urls.append(url)
        if url == sec.TICKERS_URL:
            return _Resp({"0": {"cik_str": 1, "ticker": "AAPL", "title": "x"}}, self.status)
        return _Resp(companyfacts_doc(), self.status)


def test_user_agent_default_has_no_personal_contact(monkeypatch):
    monkeypatch.delenv(sec.UA_ENV, raising=False)
    ua = sec.user_agent()
    assert ua == sec.DEFAULT_USER_AGENT
    assert "@" not in ua.replace("@example.com", "")  # only the reserved placeholder domain
    assert sec.make_session().headers["User-Agent"] == ua


def test_env_user_agent_used_but_never_persisted(monkeypatch, tmp_path):
    sentinel = "Tester sentinel-contact@test.invalid"
    monkeypatch.setenv(sec.UA_ENV, sentinel)
    monkeypatch.delenv("ASTRA_DATA_DIR", raising=False)
    assert sec.user_agent() == sentinel
    assert sec.make_session().headers["User-Agent"] == sentinel
    paths = get_paths(tmp_path)
    res = sec.ingest_sec(["AAPL"], paths=paths, http=_FakeHttp())  # type: ignore[arg-type]
    assert res["user_agent"] == "env:SEC_USER_AGENT" and not res["blocked"]
    assert res["per_symbol"]["AAPL"]["cik"] == "0000000001"
    assert sentinel not in json.dumps(res, default=str)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert any(p.suffix == ".gz" for p in files) and any("quality" in str(p) for p in files)
    for p in files:
        blob = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        assert b"sentinel-contact" not in blob, p
    meta = json.loads(next((tmp_path / "data/raw/sec/0000000001").glob("*.meta.json")).read_text())
    assert meta["user_agent"] == "env:SEC_USER_AGENT"
    # idempotent: same provider bytes -> raw file reused, no new version
    res2 = sec.ingest_sec(["AAPL"], paths=paths, http=_FakeHttp())  # type: ignore[arg-type]
    assert res2["per_symbol"]["AAPL"]["raw_reused"] is True
    assert len(list((tmp_path / "data/raw/sec/0000000001").glob("*.json.gz"))) == 1
    assert res2["rows"] == res["rows"]


def test_sec_403_reported_as_blocked(monkeypatch, tmp_path):
    monkeypatch.delenv(sec.UA_ENV, raising=False)
    monkeypatch.delenv("ASTRA_DATA_DIR", raising=False)
    res = sec.ingest_sec(["AAPL"], paths=get_paths(tmp_path), http=_FakeHttp(status=403))  # type: ignore[arg-type]
    assert res["blocked"] is True and "403" in res["error"]


def test_facts_quality_report_flags_problems(facts):
    bad = pd.concat([facts, facts.iloc[[0]].assign(value=-1.0)], ignore_index=True)
    bad.loc[1, "filed"] = pd.Timestamp("2019-01-15").date()  # filed before period end
    rep, cov = sec.quality_report_facts(sec._sort_facts(bad), 0, ["SYN", "NOPE"])
    st = {c.name: c for c in rep.checks}
    assert st["duplicates"].status == "warn" and st["duplicates"].count == 1
    assert st["filed_after_period_end"].status == "warn" and st["filed_after_period_end"].count == 1
    assert st["coverage"].status == "warn" and "NOPE" in st["coverage"].examples
    assert st["restatements"].count >= 1
    assert {c["symbol"] for c in cov} == {"SYN", "NOPE"}


# ------------------------------------------------------------------------------------ FRED


CPI_CSV = "observation_date,CPIAUCSL\n2023-11-01,307.0\n2023-12-01,308.0\n2024-01-01,309.0\n2024-02-01,.\n"
DGS10_CSV = "observation_date,DGS10\n2024-01-11,4.0\n2024-01-12,4.1\n2024-01-15,.\n2024-01-16,4.2\n"


def test_fred_publication_lags():
    d = pd.Series(pd.to_datetime(["2024-01-01"]))
    assert fred.available_from("CPIAUCSL", d)[0] == pd.Timestamp("2024-03-16")  # Jan 31 + 45
    assert fred.available_from("UNRATE", d)[0] == pd.Timestamp("2024-02-10")
    assert fred.available_from("INDPRO", d)[0] == pd.Timestamp("2024-02-20")
    assert fred.available_from("UMCSENT", d)[0] == pd.Timestamp("2024-03-01")
    # Fri 2024-01-12 + 1 Fed business day skips the weekend and MLK day (Mon 2024-01-15)
    assert fred.available_from("DGS10", pd.Series([pd.Timestamp("2024-01-12")]))[0] == pd.Timestamp(
        "2024-01-16"
    )
    # H.10 weekly release: + 8 calendar days
    assert fred.available_from("DEXINUS", pd.Series([pd.Timestamp("2024-01-08")]))[0] == pd.Timestamp(
        "2024-01-16"
    )
    for s in fred.LAGS:
        obs = pd.Series(
            pd.date_range("2020-01-01", "2024-12-31", freq="MS" if fred.LAGS[s].freq == "M" else "B")
        )
        assert (fred.available_from(s, obs) > fred.period_end(s, obs)).all(), s


def test_parse_and_macro_features_never_before_available():
    cpi, miss = fred.normalize_series(CPI_CSV, "CPIAUCSL")
    assert miss == 1 and len(cpi) == 3
    dgs, _ = fred.normalize_series(DGS10_CSV, "DGS10")
    table = pd.concat([cpi, dgs], ignore_index=True)
    dates = pd.date_range("2023-12-01", "2024-04-30", freq="D")
    feat = fred.macro_features(dates, table=table)
    assert feat.loc["2024-03-15", "CPIAUCSL"] == 308.0  # Jan value not yet available
    assert feat.loc["2024-03-16", "CPIAUCSL"] == 309.0
    assert np.isnan(feat.loc["2024-01-11", "DGS10"])  # Thu Jan 11 observation usable from Fri Jan 12
    assert feat.loc["2024-01-12", "DGS10"] == 4.0
    assert feat.loc["2024-01-15", "DGS10"] == 4.0  # Fri Jan 12 obs: next Fed business day is Tue Jan 16
    assert feat.loc["2024-01-16", "DGS10"] == 4.1
    assert feat.loc["2024-01-17", "DGS10"] == 4.2
    # generic invariant: each value used equals the latest observation with available_from <= t
    for s in ("CPIAUCSL", "DGS10"):
        g = table[table["series"] == s]
        for t in dates:
            ok = g[pd.to_datetime(g["available_from"]) <= t]
            exp = ok.sort_values("date")["value"].iloc[-1] if len(ok) else np.nan
            got = feat.loc[t, s]
            assert (np.isnan(exp) and np.isnan(got)) or exp == got, (s, t)


def test_ingest_fred_requires_manual_csv_or_explicit_download(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTRA_DATA_DIR", raising=False)
    paths = get_paths(tmp_path)
    with pytest.raises(fred.FredTermsNotAcceptedError):
        fred.ingest_fred(["CPIAUCSL"], paths=paths)
    src = tmp_path / "manual"
    src.mkdir()
    (src / "CPIAUCSL.csv").write_text(CPI_CSV, encoding="utf-8")
    (src / "DGS10.csv").write_text(DGS10_CSV, encoding="utf-8")
    res = fred.ingest_fred(["CPIAUCSL", "DGS10", "UNRATE"], paths=paths, csv_dir=src)
    assert res["rows"] == 6 and res["errors"] == {"UNRATE": "no CSV in csv_dir"}
    assert Path(res["quality_report"]).exists() and Path(res["validated"]).exists()
    rep = json.loads(Path(res["quality_report"]).read_text())
    assert {c["name"]: c["status"] for c in rep["checks"]}["available_after_period_end"] == "pass"
    stored = fred.load_fred(paths)
    assert (pd.to_datetime(stored["available_from"]) > pd.to_datetime(stored["period_end"])).all()


def test_stale_facts_not_carried_forward_and_brk_excluded(facts):
    sched = session_schedule(get_config().markets["US"], "2019-01-02", "2021-12-31")
    px = pd.DataFrame({"session_date": sched.index.date, "close": 50.0, "split_ratio": 1.0})
    feat = sec.fundamental_features("SYN", px, facts=facts).set_index("session_date")
    # last TTM period ends 2020-03-31: usable up to MAX_AGE_DAYS later, NaN afterwards
    assert feat.loc[pd.Timestamp("2021-06-30").date(), "ttm_net_income"] == 500
    assert np.isnan(feat.loc[pd.Timestamp("2021-07-06").date(), "ttm_net_income"])
    brk = sec.fundamental_features("BRK_B", px, facts=facts.assign(symbol="BRK_B"))
    assert brk[["market_cap", "earnings_yield", "book_to_market"]].isna().all().all()
    assert brk["ttm_net_income"].notna().any()


def test_predecessor_cik_used_only_before_successor_first_filing(monkeypatch, tmp_path):
    monkeypatch.delenv(sec.UA_ENV, raising=False)
    monkeypatch.delenv("ASTRA_DATA_DIR", raising=False)
    monkeypatch.setattr(sec, "PREDECESSOR_CIKS", {"AAPL": ["0000000002"]})
    pred = {
        "cik": 2,
        "entityName": "Old Corp",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            _f("2018-01-01", "2018-03-31", 90, "2018-05-01", fy=2018),
                            _f("2019-01-01", "2019-03-31", 7777, "2019-06-03"),  # after successor start
                        ]
                    }
                }
            }
        },
    }

    class Http(_FakeHttp):
        def get(self, url: str, timeout: float) -> _Resp:
            if "CIK0000000002" in url:
                return _Resp(pred)
            return super().get(url, timeout)

    res = sec.ingest_sec(["AAPL"], paths=get_paths(tmp_path), http=Http())  # type: ignore[arg-type]
    p = res["per_symbol"]["AAPL"]["predecessors"][0]
    assert p["rows_used"] == 1 and p["filed_before"] == "2019-05-01"
    stored = sec.load_facts("AAPL", get_paths(tmp_path))
    assert 90.0 in stored["value"].tolist() and 7777.0 not in stored["value"].tolist()


# ------------------------------------------------------------------------------------ market macro + BLS


def _yahoo_frame(dates: list[str], closes: list[float], tz: str) -> pd.DataFrame:
    return pd.DataFrame({"Date": pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(tz), "Close": closes})


def test_market_macro_availability_next_session_and_inr_t_plus_2():
    tnx = market.normalize_yahoo(
        _yahoo_frame(["2024-01-11", "2024-01-12"], [4.0, 4.1], "America/New_York"), "TNX"
    )
    # Thu -> Fri; Fri 2024-01-12 -> Tue 2024-01-16 (NYSE closed for MLK day)
    assert [str(x) for x in tnx["available_from"]] == ["2024-01-12", "2024-01-16"]
    inr = market.normalize_yahoo(_yahoo_frame(["2024-01-12"], [83.0], "Europe/London"), "USDINR")
    assert str(inr["available_from"].iloc[0]) == "2024-01-17"  # second XNYS session after the date
    assert (pd.to_datetime(tnx["available_from"]) > pd.to_datetime(tnx["date"])).all()


class _FakeAdapter:
    def fetch_daily(self, instrument, start, end, raw_dir):
        f = _yahoo_frame(["2024-01-10", "2024-01-11", "2024-01-12"], [4.0, 4.1, 4.2], "America/New_York")
        return RawBatch(
            provider="yahoo",
            instrument=instrument,
            retrieved_at_utc=datetime.now(UTC),
            frame=f,
            request={"interval": "1d"},
        )


def test_ingest_market_macro_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTRA_DATA_DIR", raising=False)
    res = market.ingest_market_macro(["TNX", "VIX"], paths=get_paths(tmp_path), adapter=_FakeAdapter())
    assert res["rows"] == 6 and not res["errors"]
    assert Path(res["validated"]).exists() and Path(res["quality_report"]).exists()
    assert len(list((tmp_path / "data/raw/yahoo").rglob("*.meta.json"))) == 2
    rep = json.loads(Path(res["quality_report"]).read_text())
    assert {c["name"]: c["status"] for c in rep["checks"]}["available_after_date"] == "pass"


RELEASE_HTML = """
<ul><!-- <li><a href="/news.release/archives/empsit_12042026.htm">November 2026 Employment Situation</a>
</li> -->
<li><a href="/news.release/archives/empsit_11202025.htm">September 2025 Employment Situation</a></li>
<li><a href="/news.release/archives/empsit_09052025.htm">August 2025 Employment Situation</a></li></ul>
"""


def test_bls_release_calendar_overrides_fixed_lag():
    rel = bls.parse_release_dates(RELEASE_HTML, "empsit")
    assert rel == {  # the commented-out (scheduled) entry is ignored
        pd.Timestamp("2025-09-01"): pd.Timestamp("2025-11-20"),
        pd.Timestamp("2025-08-01"): pd.Timestamp("2025-09-05"),
    }
    a = bls.availability(
        "LNS14000000", pd.Series(pd.to_datetime(["2025-08-01", "2025-09-01", "2024-01-01"])), rel
    )
    # Aug: fixed lag (Sep 10) later than the release (Sep 5); Sep: shutdown-delayed release wins
    assert list(a) == [pd.Timestamp("2025-09-10"), pd.Timestamp("2025-11-20"), pd.Timestamp("2024-02-10")]
    c = bls.availability("CUSR0000SA0", pd.Series([pd.Timestamp("2024-01-01")]), {})
    assert c[0] == pd.Timestamp("2024-03-16")


def _bls_payload(y0: int, y1: int) -> dict:
    data: dict[str, list] = {"CUSR0000SA0": [], "LNS14000000": []}
    for y in range(y0, y1 + 1):
        for m in range(1, 13):
            data["CUSR0000SA0"].append(
                {"year": str(y), "period": f"M{m:02d}", "value": str(100 + y - 2000 + m / 100)}
            )
            v = "-" if (y, m) == (2025, 10) else "4.0"
            data["LNS14000000"].append({"year": str(y), "period": f"M{m:02d}", "value": v})
        data["LNS14000000"].append({"year": str(y), "period": "M13", "value": "4.0"})
    return {
        "status": "REQUEST_SUCCEEDED",
        "message": [],
        "Results": {"series": [{"seriesID": k, "data": v} for k, v in data.items()]},
    }


class _BlsHttp:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.posts: list[dict] = []

    def get(self, url, timeout):
        r = _Resp({})
        r.content = RELEASE_HTML.encode()
        r.text = RELEASE_HTML  # type: ignore[attr-defined]
        return r

    def post(self, url, json, timeout):
        assert timeout > 0
        self.posts.append(json)
        assert int(json["endyear"]) - int(json["startyear"]) + 1 <= bls.YEARS_PER_QUERY
        return _Resp(_bls_payload(int(json["startyear"]), int(json["endyear"])))


def test_parse_bls_drops_missing_and_annual():
    obs, miss = bls.parse_api_response(_bls_payload(2025, 2025))
    assert miss == {"LNS14000000": 1} and len(obs) == 12 + 11


def test_ingest_bls_quota_and_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("ASTRA_DATA_DIR", raising=False)
    monkeypatch.setattr(bls, "PAUSE_S", 0.0)
    monkeypatch.setattr(bls, "MAX_QUERIES_PER_DAY", 2)
    paths = get_paths(tmp_path)
    http = _BlsHttp()
    res = bls.ingest_bls(paths=paths, start_year=2000, http=http)  # type: ignore[arg-type]
    n_chunks = len(bls._chunks(2000, datetime.now(UTC).year))
    assert n_chunks >= 3
    assert res["api_queries_used"] == 2 and len(http.posts) == 2
    assert sum("BlsQuotaError" in v for v in res["errors"].values()) == n_chunks - 2
    stored = pq.read_table(res["validated"]).to_pandas()
    assert set(stored["series"]) == {"CPI", "UNRATE"} and len(stored) == res["rows"] > 0
    http2 = _BlsHttp()
    res2 = bls.ingest_bls(paths=paths, start_year=2000, http=http2)  # type: ignore[arg-type]
    assert len(http2.posts) == 0 and res2["chunks_reused"] == 2  # same-day cache; budget exhausted
    assert res2["rows"] == res["rows"]


def _macro_table() -> pd.DataFrame:
    days = pd.bdate_range("2024-01-02", periods=60)
    rows = []
    for i, d in enumerate(days):
        av = d + pd.Timedelta(days=1)
        rows += [("TNX", d, 4.0 + 0.01 * i, av), ("IRX", d, 5.0, av), ("VIX", d, 15.0 + (i % 5), av)]
    for k, m in enumerate(pd.date_range("2023-01-01", periods=12, freq="MS")):
        end = m + pd.offsets.MonthEnd(0)
        rows += [
            ("CPI", m, 300.0 * 1.01**k, end + pd.Timedelta(days=45)),
            ("UNRATE", m, 3.5 + 0.1 * k, end + pd.Timedelta(days=10)),
        ]
    t = pd.DataFrame(rows, columns=["series", "date", "value", "available_from"])
    t["date"] = t["date"].dt.date
    t["available_from"] = t["available_from"].dt.date
    return t


def test_derived_macro_features_values_and_availability():
    t = _macro_table()
    der = derived_series(t)
    feat = macro_features(pd.date_range("2023-01-01", "2024-04-30"), table=t)
    d = pd.Timestamp
    # slope uses same-day TNX and IRX, visible from the next day only
    assert np.isnan(feat.loc[d("2024-01-02"), "slope_10y_3m"])
    assert feat.loc[d("2024-01-03"), "slope_10y_3m"] == pytest.approx(-1.0)
    chg = der[der["series"] == "tnx_chg_20d"]["value"]
    assert chg.iloc[0] == pytest.approx(0.2)  # 20 observations x 0.01
    vix = t[t["series"] == "VIX"]["value"].to_numpy()
    z = der[der["series"] == "vix_z20"]["value"].iloc[0]
    assert z == pytest.approx((vix[19] - vix[:20].mean()) / vix[:20].std(ddof=1))
    # CPI 3m annualized for Apr 2023 = (1.01**3)**4 - 1, usable only from Apr 30 + 45 days
    cpi = der[der["series"] == "cpi_3m_ann"].set_index("date")
    apr = pd.Timestamp("2023-04-01")
    assert cpi.loc[apr, "value"] == pytest.approx(1.01**12 - 1)
    assert cpi.loc[apr, "available_from"] == pd.Timestamp("2023-06-14")
    un = der[der["series"] == "unrate_chg_3m"].set_index("date")
    assert un.loc[apr, "value"] == pytest.approx(0.3)
    # as-of invariant: value at t equals the latest observation with available_from <= t
    long = pd.concat([t, der], ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    for s in ("cpi_3m_ann", "unrate_chg_3m", "CPI", "vix_z20", "slope_10y_3m"):
        g = long[long["series"] == s]
        for day in feat.index[::5]:
            ok = g[pd.to_datetime(g["available_from"]) <= day]
            exp = ok.sort_values("date")["value"].iloc[-1] if len(ok) else np.nan
            got = feat.loc[day, s]
            assert (np.isnan(exp) and np.isnan(got)) or exp == pytest.approx(got), (s, day)


def test_market_macro_drops_in_progress_bar():
    f = _yahoo_frame(["2024-01-11", "2024-01-12"], [4.0, 4.1], "America/New_York")
    during = datetime(2024, 1, 12, 19, 30, tzinfo=UTC)  # 14:30 New York, session still open
    out = market.normalize_yahoo(f, "TNX", during)
    assert [str(x) for x in out["date"]] == ["2024-01-11"]
    after = datetime(2024, 1, 13, 3, 0, tzinfo=UTC)
    assert len(market.normalize_yahoo(f, "TNX", after)) == 2


def test_next_sessions_cover_early_history():
    out = market.next_sessions(pd.Series(pd.to_datetime(["2000-01-03", "2001-09-10", "1980-01-02"])), 1)
    # 2001-09-11..14 NYSE closed after the attacks; dates before the calendar range -> NaT
    assert list(out[:2]) == [pd.Timestamp("2000-01-04"), pd.Timestamp("2001-09-17")]
    assert pd.isna(out[2])
