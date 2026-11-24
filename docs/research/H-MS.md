# Microstructure hypotheses register (H-MS)

Registered on 2026-11-24 **before** any capture was analysed or any aggTrades file was downloaded.
Parameters are fixed a priori; no tuning. Results are filled in from experiment records
(`experiments/`), never edited by hand to look better.

Data-level honesty: Binance public streams are **L2 (market-by-price) only**. NEBULA has no L3
(market-by-order) source and no historical spot depth; every order-book result below comes from a
short prospective live capture made by `scripts/capture_depth.py`.

## H-MS-000 Kyle-lambda-style price impact from aggTrades (descriptive / sanity)
1. Intuition: net aggressive buying moves price up (Kyle 1985 lambda; here only a reduced-form
   regression, not a structural estimate).
2. Formulation: one complete UTC day of Binance Public Data spot aggTrades for BTCUSDT (most recent
   complete day published; try yesterday, then the day before). 1-minute bars. Aggressor side from
   `is_buyer_maker` (True -> aggressive SELL). Signed volume = buy base volume - sell base volume.
   Regress dP_t = 1e4 * (log VWAP_t - log VWAP_{t-1}) (bps) on signed volume_t (BTC), OLS with
   intercept, Newey-West HAC SE with maxlags = floor(4 (n/100)^(2/9)).
3. H0: lambda <= 0. Supported iff lambda > 0 and HAC t > 1.96; rejected iff HAC t < -1.96; else
   inconclusive. Also reported (descriptive, no decision): trade count, volume, 1s/1m trade imbalance
   distribution, realized volatility.
4. Failure modes: one day only (regime-specific); VWAP change is not a mid change (bid-ask bounce
   is partly averaged out but not removed); contemporaneous regression is not causal and says nothing
   about tradable predictability.

## H-MS-001 Top-5 book imbalance predicts the sign of the next mid-price change
1. Intuition: queue imbalance at the touch/near-touch predicts the next mid move (e.g. Cartea,
   Jaimungal & Penalva 2015; Gould & Bonart 2016 - cited for the concept only).
2. Signal: I_t = (sum_{k<=5} Qb_k - sum_{k<=5} Qa_k) / (sum Qb_k + sum Qa_k) from the reconstructed
   L2 book, evaluated at every applied depth event (exchange event time E, ~100 ms cadence) while the
   book is valid (synchronized, no gap since last resync, not crossed).
3. Target: sign(mid(t + h) - mid(t)), h in {1 s, 10 s}, where mid(t + h) is the last valid mid with
   E <= t + h (as-of join on exchange time). Observations with t + h beyond the end of the valid
   segment are dropped.
4. Data: one ~5-minute live capture of BTCUSDT `@depth@100ms` + REST snapshot (limit=1000).
5. Metric: ROC AUC of I_t for (up vs down) on observations with a non-zero mid change. Base rate of
   zero change is reported per horizon. 95% CI by moving-block bootstrap over time-contiguous blocks
   of 30 s (2000 resamples, seed 0) to respect overlap/autocorrelation.
6. H0: AUC <= 0.5. Decision: **supported** iff the CI lower bound > 0.5 at both horizons;
   **rejected** iff the CI upper bound < 0.5 at both horizons; otherwise **inconclusive**. If either
   horizon has fewer than 50 non-zero-change observations or fewer than 5 bootstrap blocks, the verdict
   is **inconclusive (insufficient data)** regardless of AUC.
7. Expectation stated up front: this test is **weak**. Five minutes is one regime, observations are
   heavily overlapping, BTCUSDT spot often has a one-tick spread with long zero-change stretches, and a
   positive AUC is not a tradable edge (taker fees and the spread would dominate). Any result is
   descriptive of this capture only.
8. Failure modes: short sample; mid changes driven by trades not visible in L2 diffs; 100 ms batching
   hides intra-batch dynamics; single venue (cross-venue fragmentation ignored).
