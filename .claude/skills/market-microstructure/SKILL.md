---
name: market-microstructure
description: Use when working on ASTRA limit-order-book or trade-level analytics - L1/L2/L3 data semantics, Binance depth-stream capture and order-book reconstruction, sequence-gap detection, spread, microprice, order-flow and trade imbalance, liquidity, price impact, queue position or adverse selection.
---

# Market Microstructure (ASTRA)

## Purpose
Analytics that are honest about the data level actually available. Code: `src/astra/microstructure/`,
independent of the daily OHLCV pipeline.

## Activate when
Capturing or parsing trades/depth, reconstructing a book, computing spread/microprice/OFI/imbalance or
impact, or making any claim about what order-book data ASTRA has.

## Data levels (never blur these)
- **L1**: best bid/ask (+ sizes). **L2 / market-by-price**: aggregated size per price level.
- **L3 / market-by-order**: individual orders with IDs. Binance public streams are **L2 only**; ASTRA
  has **no L3** source and must not claim queue-position accuracy beyond L2 approximations.
- Historical free data: Binance aggTrades/trades yes; historical spot depth **no**. Order-book research
  is therefore **prospective**: capture now, analyse later.

## Book reconstruction workflow (Binance diff depth)
1. Open WS `<symbol>@depth@100ms`, buffer events.
2. GET REST snapshot `/api/v3/depth?limit=1000` -> `lastUpdateId`.
3. Drop buffered events with `u <= lastUpdateId`. First applied event must satisfy
   `U <= lastUpdateId+1 <= u`.
4. Each subsequent event must have `U == prev_u + 1`; otherwise **gap -> mark book invalid, resync**.
5. Quantity 0 removes level. Store both exchange event time `E` and local receipt time.
6. Integrity checks: best bid < best ask, no negative sizes, periodic comparison with fresh snapshot.

## Analytics definitions
- mid = (b + a)/2; spread_bps = (a - b)/mid * 1e4
- microprice = (a * Qb + b * Qa)/(Qb + Qa)
- book imbalance (top-k) = (sum Qb - sum Qa)/(sum Qb + sum Qa)
- OFI (Cont, Kukanov & Stoikov, 2014): per update e_n from best-level price/size changes, summed per window.
- trade imbalance = (buy_vol - sell_vol)/(buy_vol + sell_vol); aggressor side from Binance `m` flag
  (`m = true` -> buyer is maker -> aggressive **sell**).
- Kyle-lambda style impact: regress mid change on signed volume per interval.

## Failure modes
Misreading `isBuyerMaker`; applying diffs without sync; ignoring gaps; using local time as event time;
unbounded capture filling the disk (measure MB/hour first).

## Verification / acceptance
Unit tests: synthetic snapshot + diffs incl. a gap -> detected; microprice/imbalance on hand examples;
aggressor sign test. Capture script reports MB/hour and gap count.

## Limitations
No L3, no historical depth, single venue (Binance) - fragmentation across venues ignored.

## Interacts with
`financial-data-engineering` (storage, timestamps), `financial-ml-research` (short-horizon models),
`backtesting-audit` (any microstructure signal must survive realistic taker costs, usually it will not).
