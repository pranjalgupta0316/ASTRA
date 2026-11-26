# Survivorship check of the US universe (P8) - 2026-11-26

Descriptive, no hypothesis. `uv run astra universe` (`src/astra/data/universe.py`).

## Data
- **Source:** point-in-time S&P 500 membership reconstructed backwards from two Wikipedia pages (CC BY-SA 4.0,
  Wikipedia contributors). Revisions: constituents 1384572480, changes 1387311792.
- **Size:** 503 current constituents and 414 recorded index changes (1976-2026), giving 886 membership
  intervals.
- **Validated output:** `data/validated/universe/sp500_membership.parquet`, plus a manifest with the
  sources, revision ids, sha256 and the orphan list.
- **Coverage gate:**
  - The table records only 2-13 changes a year before 2011, against roughly 20-30 actual.
  - So membership is trusted only from **2011-01-01**: the first year after which every year has at least 15
    recorded changes.
  - The member count (502-509 on every date) is no evidence of completeness: paired add/remove events keep it
    near 500 whatever is missing.
- **Ticker renames:**
  - Renames the table does not record are mapped explicitly: FB→META, PCLN→BKNG, CDAY→DAY, FLT→CPAY,
    RE→EG, KORS→CPRI, DLPH→APTV.
  - 21 events remain unmatched ("orphans"), 5 of them after 2011: SATS, UA.C, FOX, AGN, JOYG. They are
    listed in the manifest. Individual tickers involved in them may have wrong intervals.
- **Spot check:** TSLA is a member from 2020-12-21, not on 2020-12-18. Correct.

## Result
| Date | Index members | Still members today | Removed since | ASTRA US stocks that were members then |
|---|---|---|---|---|
| 2011-01-03 | 503 | 292 | 41.9 % | 28 / 30 (not yet: ABBV, META) |
| 2015-01-02 | 504 | 325 | 35.5 % | 29 / 30 (not yet: AMD) |
| 2018-01-02 | 506 | 375 | 25.9 % | 30 / 30 |
| 2022-01-03 | 506 | 426 | 15.8 % | 30 / 30 |

- **ASTRA's universe is survivors only.** All 30 US single stocks were index members at the start of the
  frozen core window (2018-01-02), and all are members today. Of the 506 stocks in the index on that date,
  26 % have since left it, through acquisitions, demotions or bankruptcies. None of those are in ASTRA.
  For studies from 2011, 42 % of the starting index is missing.
- **Look-ahead in the earlier picks:** ABBV, META and AMD joined the index after some study start dates.
  Including them from those dates is selection with hindsight.
- **Size of the bias is not measured.** Free daily prices for removed and delisted names are mostly unavailable
  (Yahoo drops many delisted tickers). The bias on returns is therefore *not* estimated here; this check
  sizes the universe gap only.
- **Practical rule for new US cross-sectional studies:** restrict each date's universe to index members on
  that date (`members_on`), from 2011 onward, and report coverage, i.e. the share of members that have
  price data.

## What this does not fix
Membership is not tradability. Without delisted price histories, a point-in-time universe still drops
the names that failed. Indian (NIFTY) membership has no comparable free source yet.
