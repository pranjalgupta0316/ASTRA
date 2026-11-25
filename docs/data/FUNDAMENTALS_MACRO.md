# Fundamentals (SEC EDGAR) and macro (Yahoo market macro, BLS, manual FRED)

Code: `src/astra/data/fundamentals/sec.py`; `src/astra/data/macro/` (`market.py`, `bls.py`,
`features.py`, `fred.py`).
Tests: `tests/test_fundamentals_macro.py` (offline, synthetic fixtures).
Terms and endpoints checked 2026-11-25.

## 1. Sources and licensing

| Dataset | Endpoint | Licence / terms | Access rules |
|---|---|---|---|
| Ticker -> CIK | `https://www.sec.gov/files/company_tickers.json` | US government work, public domain | descriptive User-Agent; <= 10 req/s (SEC fair access) |
| XBRL company facts | `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` | public domain | same; ASTRA uses <= 4 req/s, 30 s timeout, backoff on 429/5xx, 403 -> `SecAccessBlockedError` |
| FRED series | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES>` | FRED Terms of Use + per-series third-party copyright | **manual CSV only**, see section 1.1 |
| Market macro | Yahoo Finance via `astra.data.adapters.yahoo.YahooAdapter.fetch_daily` | same as ASTRA price data: unofficial endpoints, personal research use; index data owned by Cboe (^TNX/^FVX/^IRX/^VIX) and ICE (DX-Y.NYB) | adapter retries with backoff; 6 requests per run |
| BLS CPI-U SA, unemployment rate | `https://api.bls.gov/publicAPI/v1/timeseries/data/` (POST); release dates from `https://www.bls.gov/bls/news-release/{empsit,cpi}.htm` | public domain (section 1.2) | v1: 25 queries/day, 25 series/query, 10 years/query; ASTRA caps at 20/day |

**SEC User-Agent.** Read from env var `SEC_USER_AGENT`, else `ASTRA research astra@example.com`
(a reserved placeholder domain). The env value is sent only as the HTTP header. It is never written to
disk. Raw metadata, manifests and results record only `env:SEC_USER_AGENT`, or the default string when
the env var is unset. The 2026-11-25 ingestion used the default; SEC did not block it.

### 1.1 FRED terms: automated download is disabled by default
The FRED Terms of Use (https://fred.stlouisfed.org/legal/) prohibit, for all FRED services:
- using scripts, bots or any "automatic technology" to access content;
- data mining, mirroring or scraping;
- storing, caching or archiving any portion of FRED content;
- (Summarized Terms) using FRED content to develop or train software or machine-learning systems.

The general licence allows downloading a copy for personal, non-commercial use only, with copyright
notices kept intact. Some series are owned by third parties:
- VIXCLS: Cboe
- UMCSENT: University of Michigan

FRED cannot grant rights to those series.

As a result, `ingest_fred` reads CSVs that a person downloaded by hand (`csv_dir/<SERIES>.csv`). It
calls the endpoint only when `allow_download=True` is passed by someone who has reviewed the terms.
ASTRA's ML use of FRED data needs a human decision. Alternatives are the primary public-domain
publishers:
- Federal Reserve Data Download Program: H.15, H.10, G.17
- US Treasury yield-curve CSVs
- BLS public API: CPI, UNRATE

No real FRED ingestion has been run. Coordinator decision (2026-11-25): FRED stays manual-CSV only.
`macro_features` includes FRED series (prefixed `FRED_`) only if `data/validated/macro/fred.parquet`
was built from manually downloaded CSVs.

### 1.2 BLS terms (checked 2026-11-25)
- Copyright page (bls.gov/bls/linksite.htm): BLS material is public domain. "You are free to use our
  public domain material without specific permission." The BLS emblem must not be used.
- API FAQ (bls.gov/developers/api_faqs.htm): "Unregistered users may request up to 25 queries daily."
  Also 25 series per query and 10 years per query. Registration is required only for v2. Exceeding the
  limits returns HTTP 429.
- API Terms of Service (bls.gov/developers/termsOfService.htm):
  - Users must "cite the date that data were accessed or retrieved using the API".
  - Users must state that BLS.gov "cannot vouch for the data or analyses derived from these data".
  - Content must not be modified or falsely represented while citing BLS.
  - There is no clause against automated access within the limits, local storage, or research use.

**Required notice for any ASTRA output that uses BLS data:** "Source: U.S. Bureau of Labor
Statistics, Public Data API v1, retrieved <date in bls.manifest.json>. BLS.gov cannot vouch for the
data or analyses derived from these data."

The `query_log.json` budget counts only queries made through `ingest_bls`. One manual probe query on
2026-11-25 is not in the log.

## 2. Storage layout

| Layer | Path | Content |
|---|---|---|
| raw | `data/raw/sec/_reference/company_tickers_<UTC>.json.gz` (+ `.meta.json`) | gzip of the exact bytes SEC returned |
| raw | `data/raw/sec/<CIK10>/companyfacts_<UTC>.json.gz` (+ `.meta.json`) | metadata includes `url`, `retrieved_at_utc`, `user_agent` label, `sha256` (gz), `content_sha256` (uncompressed) and byte counts |
| raw | `data/raw/fred/<SERIES>/<SERIES>_<UTC>.csv` (+ `.meta.json`) | CSV bytes, `origin` (`manual:<file>` or `download`), sha256, lag rule |
| validated | `data/validated/fundamentals/us_facts.parquet` (+ `us_facts.manifest.json`) | long point-in-time fact table |
| raw | `data/raw/yahoo/<YahooSymbol>/1d/raw_<UTC>.parquet` (+ `.meta.json`) | adapter raw layer (`storage.write_raw`), e.g. `^TNX`, `INR=X` |
| raw | `data/raw/bls/api/<ids>_<y0>-<y1>_<UTC>.gz`, `data/raw/bls/release_calendar/{empsit,cpi}_<UTC>.gz` (+ `.meta.json`) | gzip of exact API JSON / archive HTML; `query_log.json` = queries per UTC day |
| validated | `data/validated/macro/fred.parquet` (+ `fred.manifest.json`) | long series table (manual CSVs only) |
| validated | `data/validated/macro/market.parquet`, `bls.parquet` (+ manifests) | long series tables, same schema as `fred.parquet` |
| quality | `data/reports/quality/US_sec_companyfacts_<UTC>.json`, `FRED_macro_<UTC>.json`, `MARKET_macro_<UTC>.json`, `BLS_macro_<UTC>.json` | QualityReport plus a per-symbol or per-series coverage table |

Raw files are immutable. If a re-download is byte-identical to the latest stored file (same
`content_sha256`), that file is reused and no new version is written. gzip uses `mtime=0`, so the
compressed sha256 is deterministic. zstd for raw JSON would need the `zstandard` package, which is not
a dependency.

## 3. Schemas

### `us_facts.parquet`
This table has one row per (filing, fact). Every filing's value is kept, including amendments and
restated comparatives.

| column | type | meaning |
|---|---|---|
| symbol | str | ASTRA symbol (BRK_B; SEC ticker taken from config `provider_symbol`, e.g. BRK-B) |
| cik | str | 10-digit CIK of the **filing** registrant (can be a predecessor, see 4.4) |
| concept | str | canonical concept (below) |
| taxonomy, tag | str | XBRL source, e.g. `us-gaap` / `RevenueFromContractWithCustomerExcludingAssessedTax` |
| unit | str | USD, USD/shares, shares |
| value | float64 | value as filed |
| period_start | date (null for instants) | duration start |
| period_end | date | duration end / instant date (for dei shares: the cover-page date) |
| fiscal_year, fiscal_period | int32 / str | `fy` / `fp` of the **filing** (not of the fact's period: comparatives carry the filing's fy/fp) |
| form | str | 10-K, 10-Q, 10-K/A, 8-K, ... |
| filed | date | filing date (availability key) |
| accession | str | accession number |
| frame | str | SEC calendar frame when SEC assigned one |

Canonical concepts and tag candidates, in priority order:

| concept | tags |
|---|---|
| Revenues | Revenues, RevenueFromContractWithCustomerExcludingAssessedTax, SalesRevenueNet |
| NetIncomeLoss | NetIncomeLoss |
| Assets | Assets |
| Liabilities | Liabilities |
| StockholdersEquity | StockholdersEquity, StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest |
| EarningsPerShareDiluted | EarningsPerShareDiluted |
| CommonStockSharesOutstanding | dei:EntityCommonStockSharesOutstanding, us-gaap:CommonStockSharesOutstanding |
| OperatingIncomeLoss | OperatingIncomeLoss |
| CashAndCashEquivalentsAtCarryingValue | CashAndCashEquivalentsAtCarryingValue |
| LongTermDebt | LongTermDebt |

The StockholdersEquity fallback includes non-controlling interests. It is only selected for periods
where plain StockholdersEquity was not filed first.

### `fred.parquet`, `market.parquet`, `bls.parquet` (shared `MACRO_SCHEMA`)
`series` str, `date` date (observation date; for monthly series the 1st of the month),
`period_end` date, `value` float64, `available_from` date.

- Market series names: TNX, FVX, IRX, VIX, DXY, USDINR. Yields are in percent; the Yahoo ^TNX quote is
  already in percent, e.g. 4.9, not 49.
- BLS series names: CPI (CUSR0000SA0, index 1982-84 = 100) and UNRATE (LNS14000000, percent).

## 4. Point-in-time rules (fundamentals)

1. **Availability.** A fact is usable only from the first session strictly after its `filed` date.
   Companyfacts provides the filing date but not `acceptedDateTime`, so every filing is treated as made
   after the close. In code, a session `s` uses a fact only if `s > filed`.
2. **Restatements: first-filed wins.** For each (concept, period_start, period_end), only the value
   from the earliest filing is used. Ties are broken by tag priority, then original before amendment,
   then accession. Later 10-K/A filings and restated comparatives in later 10-Ks never replace history.
   This is the choice a trader would have faced at the time. It also means a known error stays in the
   series once corrected. The long table keeps every filed value, so revision studies remain possible.
   The quality report counts (symbol, concept, period) keys with more than one distinct filed value: 2,240.
3. **Quarters and TTM.**
   - Discrete quarters come from direct 3-month facts.
   - A missing quarter is derived as YTD(k) - YTD(k-1) with the same start, e.g. Q4 = FY - 9M.
   - TTM at a quarter end is the first-filed FY fact when one ends there. Otherwise it is the sum of 4
     contiguous quarters (end/start tolerance 5 days for 52/53-week years).
   - Derived values become available at the latest `filed` date of their components.
   - Derived Q4 EPS (FY - 9M) is an approximation, because share counts differ between periods.
4. **As-of selection.** On each session, a feature takes the latest period whose components are all
   available. If that period ended more than 460 days before the session, the feature is NaN, so a tag
   the filer stopped using is not carried forward (`MAX_AGE_DAYS`).
5. **Successor registrants (`PREDECESSOR_CIKS`).** Each entry was verified against
   `data.sec.gov/submissions`:

   | symbol | current CIK | predecessor CIK |
   |---|---|---|
   | XOM | 2115436, ExxonMobil Holdings Corp, filing since 2026-08 | 34088 |
   | DIS | 1744489 | 1001039, TWDC Enterprises 18 Corp. |
   | GOOGL | 1652044, Alphabet | 1288776, Google Inc. |

   Predecessor facts are kept only if they were filed before the successor's first filing.

### Features: `fundamental_features(symbol, prices_df, facts=None, paths=None)`
These are daily values on the sessions of `prices_df`. It needs `session_date` and `close`, and uses
`split_ratio` if present.

| feature | definition |
|---|---|
| ttm_revenue, ttm_net_income | TTM sums (USD) |
| revenue_growth_yoy | TTM revenue / TTM revenue ending 365 +- 10 days earlier - 1 (NaN if prior <= 0) |
| net_margin | TTM NI / TTM revenue (same quarter end) |
| roe | TTM NI / StockholdersEquity at that quarter end (NaN if equity <= 0) |
| leverage | Liabilities / Assets (same balance-sheet date) |
| ttm_eps_diluted | TTM diluted EPS on the **current** share basis |
| earnings_yield | ttm_eps_diluted / close. This is the preferred E/P |
| market_cap | shares outstanding (current basis) x close |
| earnings_yield_ni | TTM NI / market_cap |
| book_to_market | StockholdersEquity / market_cap |
| fund_period_end, fund_filed | period end and availability date of the TTM NI in use (audit) |

**Split adjustment.** Yahoo `close` is split-adjusted to today's share basis, but a per-share number
was reported on the share basis in force when it was filed. Under GAAP, splits before issuance are
applied retroactively. EPS is therefore divided by `F(filed)`, and share counts multiplied by it, where
`F(d)` is the product of `split_ratio` over sessions strictly after `d`. If `d` is before the first
price session, `F` is unknown and the feature is NaN. EPS-based E/P avoids most share-count problems,
so it is preferred over NI / market cap.

**Multi-class share issues.**
- **BRK_B:** SEC shares are Class-A equivalents, and the BRK-B tags stop in 2011. No diluted-EPS tag
  is present. All price ratios are therefore forced to NaN (`NO_PRICE_RATIOS`).
- **META:** has no share-count facts, so `market_cap` and `book_to_market` are NaN.
- **GOOGL:** shares are the all-class total, priced at the Class A price. This is approximate, as the
  classes trade near parity.
- **V:** has no EarningsPerShareDiluted tag (multi-class EPS).

## 5. Macro availability

### 5.1 FRED (manual CSVs): publication lags (`LAGS`)
`available_from = period end + lag`. Daily series: period end = observation date, and lags count
Federal Reserve business days (`USFederalHolidayCalendar`). Monthly series: period end = month end.
A value is usable on dates `t >= available_from`. `macro_features(dates, series=None, table=None)`
forward-fills only from availability, taking the latest observation date known on that day.

| series | meaning | lag | rationale |
|---|---|---|---|
| DGS10, DGS2 | 10y / 2y Treasury CMT | +1 Fed business day | H.15 posts day d on d+1 |
| T10Y2Y | 10y - 2y spread | +1 Fed business day | derived from H.15 |
| DFF | effective fed funds | +1 Fed business day | EFFR published next business day |
| VIXCLS | VIX close (Cboe, copyrighted) | +1 Fed business day | close known same day |
| DEXINUS | INR per USD | +8 calendar days, rolled to a business day | H.10 is released **weekly** (Monday, prior week); a +1 day lag would leak |
| DTWEXBGS | broad USD index | +8 calendar days | H.10 weekly |
| CPIAUCSL | CPI-U SA | month end + 45 days | released ~10-15 days after month end; conservative |
| UNRATE | unemployment rate | month end + 10 days | first Friday of the next month |
| INDPRO | industrial production | month end + 20 days | G.17 around the 15th-17th |
| UMCSENT | Michigan sentiment (copyrighted) | month end + 30 days | preliminary mid-month, final end of month |

### 5.2 Market macro (Yahoo)
The value is the daily `Close`, not adjusted.

| series | Yahoo | available_from |
|---|---|---|
| TNX, FVX, IRX, VIX, DXY | ^TNX, ^FVX, ^IRX, ^VIX, DX-Y.NYB | first XNYS session strictly after the bar date (t+1) |
| USDINR | INR=X | **second** XNYS session after the bar date (t+2). Yahoo stamps FX bars at London midnight with an undocumented snapshot time, so this is conservative |

- Sessions come from an XNYS calendar built from 1990. The default exchange_calendars window starts in
  2006, which earlier caused late (not early) availability dates for 2000-2006; this is fixed and tested.
- Bars that may still be in progress are dropped. A bar is kept only if retrieval time (UTC) is at
  least 26 h after its date, because Yahoo returns the live quote as the latest daily "Close".

### 5.3 BLS: actual release dates
`available_from = max(month end + fixed lag, first-release date)`:
- fixed lag: CPI +45 days, UNRATE +10 days;
- first-release date: parsed from the BLS news-release archive pages (2008+ and older layouts;
  scheduled entries in HTML comments are ignored).

Releases are at 08:30 ET, before the US open, so the release day is usable. On 2026-11-25 release
dates were found for all 321 months of each series from 2000. The release date was later than the
fixed lag for 4 UNRATE months:

| reference month | released | fixed-lag date |
|---|---|---|
| 2013-09 | 2013-10-22 | 2013-10-10 |
| 2025-09 | 2025-11-20 | 2025-10-10 |
| 2025-11 | 2025-12-16 | 2025-12-10 |
| 2026-01 | 2026-02-11 | 2026-02-10 |

A plain +10 day lag would have leaked in each of these months. October 2025 was never published
(2025 appropriations lapse; BLS value "-"). It is dropped and reported as a missing month.

Caching: an API chunk (10 years) is reused if it was retrieved the same UTC day. It is also reused if
it ends more than 5 years before the current year, since seasonal factors are revised only for the
last 5 years. Otherwise it is re-queried, subject to the daily budget. If the budget is exhausted, the
cached copy is used and the error is reported.

### 5.4 Combined features: `astra.data.macro.macro_features(dates, table=None, paths=None)`
- Long tables (market + BLS + FRED if present) are joined as-of on `available_from`. A value is used on
  date t only if `available_from <= t`, and the latest observation date known on t wins.
- Derived series are computed on each source's observation sequence and take the latest
  `available_from` of their components before the join.

| column | definition |
|---|---|
| TNX, FVX, IRX, VIX, DXY, USDINR, CPI, UNRATE | levels |
| slope_10y_3m | TNX - IRX, same observation date. IRX is a T-bill discount rate, so this approximates the 10y-3m slope |
| tnx_chg_20d | TNX(obs i) - TNX(obs i-20), in percentage points |
| vix_z20 | (VIX - mean of last 20 obs) / sample std of last 20 obs (window includes the current obs) |
| cpi_3m_ann | (CPI(m) / CPI(m-3))^4 - 1. Requires month m-3 to exist, so values that need Oct 2025 are NaN |
| unrate_chg_3m | UNRATE(m) - UNRATE(m-3), in percentage points |

The FRED-only as-of join remains available as `astra.data.macro.fred_macro_features`.

## 6. Quality checks
- **SEC:**
  - nulls in required columns
  - duplicates (exact duplicates dropped and counted; conflicting values per tag/period/accession)
  - period_start <= period_end
  - `filed >= period_end` (violations kept, with a warning)
  - table sort order, and `filed` within [2009-01-01, today]
  - restatement count
  - coverage per symbol (rows, first and last filed, missing concepts, revenue tags)
- **Market macro:**
  - nulls, duplicates on (series, date), monotonic dates
  - `available_from > date`
  - non-positive values (IRX exempt)
  - jumps: |log change| > 0.5, or > 1 pp for yields. Flagged, not repaired
  - coverage per series

  Run of 2026-11-25: 4 VIX jumps (2018-02-05, 2024-08-05, 2024-12-18, 2026-11-05). These are
  real volatility spikes.
- **BLS:**
  - nulls, and "-" values dropped and counted
  - duplicates, missing months listed
  - `available_from > period_end`
  - release-calendar coverage
  - months where the release date was later than the fixed lag
- **FRED:**
  - nulls, and provider missing markers (`.`) dropped and counted
  - duplicates on (series, date)
  - monotonic dates
  - `available_from > period_end`
  - coverage per series

## 7. Limitations
- **No vintages for FRED.** Without an ALFRED key, values are the current revised ones, not first
  releases. CPI, UNRATE, INDPRO and UMCSENT are revised. Fixed lags address timing but not the size of
  revisions. Fixed lags are also wrong when releases are disrupted, as in the 2013 and 2025 US
  government shutdowns, when BLS releases were delayed or cancelled.
- **FRED terms** (section 1.1) restrict automated access, local storage and ML use. FRED is therefore
  manual-CSV only.
- **BLS values are not vintages either.** The API serves current values. Seasonally adjusted CPI is
  revised every February for 5 years, and UNRATE SA is revised in January. Release dates are first
  release; values are latest.
- **Yahoo market macro** is unofficial and can break or be revised silently. ^IRX is a discount rate,
  not a bond-equivalent yield. DX-Y.NYB and INR=X timestamps are not exchange closes. INR=X starts
  2003-12-01 on Yahoo.
- **XBRL tagging is inconsistent.** Missing concepts per symbol as of 2026-11-25:
  - Liabilities: ABBV, AMD, AMZN, DIS, INTC, KO, MRK, ORCL, WMT (these firms do not tag total
    liabilities, so `leverage` is NaN)
  - OperatingIncomeLoss: BAC, CVX, JPM, MRK, XOM
  - LongTermDebt: BRK_B, UNH, XOM
  - EarningsPerShareDiluted: BRK_B, V
  - CommonStockSharesOutstanding: META

  Revenue tags change over time, e.g. SalesRevenueNet -> RevenueFromContract... after ASC 606.
  Companyfacts excludes dimensional facts, so per-class shares are missing. Two facts are tagged with
  a period end after their filing date (ADBE shares, WMT cash). These are filer errors and are kept.
- **Filing date, not acceptance time.** Same-day pre-open filings are made available one session late
  (conservative).
- **XBRL history** starts in 2009-2011, depending on filer size.
- **Survivorship.** The extended US list is today's large caps (see `config/markets.yaml`).
- **No Indian fundamentals.** No free official machine-readable source exists. NSE/BSE terms restrict
  automated access, and XBRL filings on MCA require registration. Indian fundamentals are documented
  as unavailable.
