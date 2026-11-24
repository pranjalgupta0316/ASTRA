# Extended-universe hypotheses (registered 2026-11-25, committed before any of them was run)

Common rules
- Data: validated bars from `config/markets.yaml`. Extended universe = **today's** large caps / NIFTY
  names / crypto majors → **survivorship-biased**; every record states it. Instruments enter a study only
  after 252 sessions of history (no backfilling); placeholder rows excluded as before.
- Execution: decisions at close t, fills at open t+1, per-market costs from config, volume caps,
  long-only unless stated. Gross results from zero-cost reruns.
- Inference: Newey-West t on daily return differences (|t| > 1.96) unless stated; Sharpe differences
  via paired stationary bootstrap (block 20, 2000 draws); Benjamini-Hochberg across the family.
- Fixed parameters below; no tuning. `variants_tried` = number of registered cells.

## H-CS-001 Cross-sectional momentum (Jegadeesh & Titman 1993, J. Finance - concept only)
- Universes: US stocks (kind=stock, 30), IN stocks (49, NIFTY50 index excluded), crypto (16).
- Signal: equities 12-1 momentum = log return t-252 → t-21; crypto 4-week momentum = t-28 → t-1
  (crypto horizon motivated by Liu, Tsyvinski & Wu 2022, J. Finance).
- Portfolio: monthly rebalance (equities, first session of month) / weekly (crypto, Monday), equal-weight
  long the top tercile. Benchmark: equal-weight all eligible names, same rebalance.
- H1: top-tercile net daily returns exceed the benchmark (NW t > 1.96). Family = 3 markets, BH.
- Descriptive only: long top / short bottom tercile gross return (shorting not executable in the engine).

## H-CS-002 Short-term reversal (Jegadeesh 1990; Lehmann 1990 - concept only)
- Same universes; signal = −(5-session return); weekly rebalance; long top tercile (past losers).
- H1 as H-CS-001 (net of costs). Expected to fail net of costs (high turnover) - registered anyway.

## H-VT-001 Volatility-managed exposure (Moreira & Muir 2017, J. Finance - concept only)
- For every instrument and for each market's equal-weight portfolio: weight_t = min(1, σ_target / σ̂_t)
  with σ̂ = EWMA(λ=0.94) daily vol known at close t, σ_target = the instrument's own full-sample median
  EWMA vol **computed on data before the evaluation period** (first 3 years used only for calibration,
  then frozen). Leverage cap 1.0 (de-risking only).
- H1: ΔSharpe(managed − unmanaged) > 0 (bootstrap CI > 0). Family = all instruments + 3 portfolios, BH.

## H-XMKT-001 Cross-market diversification
- Weekly (Friday-close) USD returns of three equal-weight sleeves: US stocks, IN stocks (converted with
  FRED DEXINUS, as-of availability), crypto (USDT ≈ USD, stated assumption). Portfolios: US-only vs
  equal-risk (inverse trailing-52w vol) mix of the three, rebalanced monthly, costs as per market.
- H1a: realized vol(mix) < vol(US-only) (bootstrap CI of ratio < 1). H1b: ΔSharpe > 0.

## H-ML-003 Large pooled daily panel (all 112 instruments, full history)
- Same 13 features, target and models as H-ML-002 (logit, XGBoost, MLP, GRU; fixed hyperparameters);
  date-based purged walk-forward, 2 calendar years initial training, 1 quarter tests, gap 7 days;
  row-weighted, date-clustered DM vs ex-ante base rate (and vs logit for DL). BH across 4 models.
- H1: at least one model has OOS log loss significantly below the ex-ante base rate.

## H-ML-004 Hourly crypto panel (16 pairs, 1h bars)
- Same 13 features computed on hourly bars (windows in bars), target = sign of next open-to-open hourly
  return (decision at bar close, execution next bar open). Folds: 2 years train, 1 quarter test, gap 48
  bars. Models: logit, XGBoost, MLP. H1 as H-ML-003. If any model passes, a cost-aware hourly backtest
  (26 bps round trip) is run; otherwise not.

## H-MM-001 Multimodal ablation (US stocks, price + fundamentals + macro)
- Target: sign of next-21-session return in excess of the equal-weight US-stock universe (cross-
  sectional). Rows: month-end decision dates. Feature sets: P (13 price features), P+F (SEC point-in-time
  fundamentals), P+M (FRED macro, as-of publication lag), P+F+M. Model: logit and XGBoost (fixed).
- Purged walk-forward by month: 5 years initial training, 1 year tests, purge 1 month, embargo 1 month.
- H1: P+F+M has lower OOS log loss than P (date-clustered DM t > 1.96); ablation reports each modality's
  marginal contribution. Fundamentals only from the first session after the SEC `filed` date.

## H-MS-002 Order-book imbalance across multi-hour captures (BTCUSDT)
- Replaces the 5-minute evidence of H-MS-001: same definitions (top-5 imbalance → sign of next 1 s / 10 s
  mid change), pooled over all valid captures (~6 h), moving-block bootstrap with ≥ 30 blocks.
- H1: AUC CI lower bound > 0.5 at both horizons. Additionally (descriptive): does a taker strategy
  trading the signal at 10 s survive 10 bps taker fee + half-spread? Expected: no.
