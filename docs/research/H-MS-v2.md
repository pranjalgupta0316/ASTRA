# Microstructure v2 hypotheses on Binance USD-M perpetuals (registered 2026-11-26, before any futures data was analysed)

Data: Binance Vision USD-M futures `bookTicker` (best bid/ask, ~2023-12 to 2024-03 is all that exists)
and `aggTrades` over the days on which BOTH exist for BTCUSDT and ETHUSDT perpetuals (market CRYPTO_UM,
never mixed with spot). The window is fixed by availability, not by results. A new data lock is created
before the run, and the run uses it. Licence policy: docs/data/BINANCE_FUTURES_DATA_TERMS.md (local
non-commercial research; only aggregate statistics are published).

Simulator: `astra.execution.simulate` (P6). Latency 50 ms each way. Fees: VIP0 USD-M maker 2.0 bps,
taker 5.0 bps. Market-order quantity beyond L1 is filled at +5 bps and flagged; the number of such fills
is reported. No exact queue priority is claimed. Every passive result is reported under the three fill
models: through (pessimistic), queue (estimate) and touch (optimistic).

Parent orders (fixed a priori): one every 5 minutes on the UTC clock. The side alternates buy/sell by
index. Quantity is 0.05 BTC or 1.0 ETH. A parent time is excluded when the last quote is more than 5 s
old, or when either side's L1 size is zero. The excluded count is reported.

## H-MS-003 Passive-with-fallback is cheaper than immediate taker execution
- A (taker): market order at t.
- B (passive): limit order at the same-side best quote at t (bid for a buy), cancel requested at t+60 s.
  Any unfilled remainder is sent as a market order at the cancel time.
- Cost per parent: implementation shortfall vs the arrival mid m_t, in bps, including fees.
  Buy: (vwap - m_t)/m_t*1e4 + fee bps. Sell: (m_t - vwap)/m_t*1e4 + fee bps.
- Test per symbol and fill model: d = IS_B - IS_A, Newey-West t (lag floor(4(n/100)^(2/9))).
  BH within the 2-symbol family on the primary model.
- Decision:
  - **supported** if d < 0 with BH-significant t < -1.96 under **through** for both symbols. Passive is
    cheaper even when fills are pessimistic.
  - **rejected** if d > 0 with t > 1.96 under **touch** for both symbols. Passive is worse even when
    fills are optimistic.
  - Otherwise **inconclusive**. The answer then depends on the unobservable queue position, and the report
    says so.

## H-MS-004 Passive fills are adversely selected
- Over all maker fills of B under the **queue** model, the signed 10 s mark-out
  (mid_{t+10s} - fill price, sign +1 for a buy) is negative on average.
- Test: one-sample t on per-parent average mark-outs, Newey-West. Supported if t < -1.96 for both symbols
  (BH); rejected if t > 1.96 for both; else inconclusive.
- 1 s and 60 s mark-outs and the through/touch variants are descriptive only.

## Descriptive (no decision)
Fill ratio of B by model; time to first fill; spread distribution; share of parents where L1 size was
smaller than the order.

## Amendment 1 (2026-11-26, before any futures data was loaded or analysed)
- "Sent as a market order at the cancel time" means: the fallback market order is submitted when the cancel
  *takes effect* (t + 60 s + latency). It then becomes active one more latency later. This prevents the
  limit order and the fallback from both filling the same remainder during the cancel latency.
- The limit price of B is the same-side best quote observed at t (the decision time). The order becomes
  active at t + latency. If the market has moved through that price by activation, B is marketable and
  executes as a taker, under the simulator's rule.
