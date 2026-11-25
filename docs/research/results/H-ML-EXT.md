# Results: H-ML-003, H-ML-004, H-MM-001

## Independent audit (2026-11-25) - no leakage; verdicts stand with this wording
- **H-ML-003 - supported, economically negligible.** XGBoost (DM t 3.97) and MLP (3.23) beat the
  ex-ante base rate by ~0.0004 nats/row (AUC 0.517), robust to HAC lag choices (xgb 3.7-4.0) and to a
  trailing-252-session base rate (xgb 3.26, MLP 2.82). The **logistic model is borderline** (t 2.06
  default, 1.79 with 24 lags, 1.61 weekly clusters, 1.87 vs trailing base rate). The signal sits in the
  most recent bar (features lagged one extra day: logit t −0.25). Concentrated in crypto/US rows since
  2015; survivorship-biased universe. **No economic claim.**
- **H-ML-004 - statistically robust, economically not significant.** Leaky-target canary scores AUC
  0.998 (pipeline would catch leakage); features lagged an extra bar still give AUC 0.523, DM 6.9.
  Simpler trailing up-rate baselines do not explain the gain (xgb still t ≈ 10.5). Most of the AUC is a
  market-wide timing signal (within-timestamp cross-sectional AUC 0.515). **Breakeven ≈ 5 bps round
  trip** vs 26 bps registered (net Sharpe −4.4 logit / −6.4 xgb). Gross Sharpe 1.0-1.4 is partly crypto
  beta (β 0.59; alpha t 1.6 logit, 3.6 xgb). The backtest ignores bid-ask bounce and drift
  rebalancing, which would make it worse, not better.
- **H-MM-001 - rejected stands.** Fundamentals strictly after `filed`, macro after availability,
  imputer/scaler inside the training pipeline. The 2010 logit P+F fold blow-up (log loss 4.90) is
  registered behaviour.
- Pooled hourly open-vs-previous-close gap: median 0.25 bp, 59 % within 1 bp (the "88 %" below is
  BTC-only). The daily panel has grown to 602,729 rows since the run (602,489 logged); manifests pin
  the exact data version used.

---

Registration: `docs/research/H-EXT.md` (2026-11-25, committed before any run). Earlier ML design and
amendment A2: `docs/research/H-ML.md`. Code: `src/astra/research/ml_extended.py`,
`src/astra/research/multimodal.py`. Tests: `tests/test_ml_extended.py`.
Seed 7 everywhere. Models and hyperparameters are the fixed H-ML-001/002 ones; nothing was tuned.
**Survivorship:** the extended universe is today's large caps, NIFTY names and crypto majors, so it is
survivorship-biased. Every record says so.

Data versions: per-symbol validated-bar sha256 values are stored in each record's `datasets.manifests`.
Fundamentals come from `us_facts.parquet` (sha256 `bddc8507...`, filed 2009-05-07 to 2026-11-20). Macro
comes from `market.parquet` and `bls.parquet`.
BLS data: Source: U.S. Bureau of Labor Statistics, Public Data API v1, retrieved on the date in
bls.manifest.json. BLS.gov cannot vouch for the data or analyses derived from these data.

## Implementation choices (fixed before the first run, not tuning)

| item | choice | why |
|---|---|---|
| Entry rule | Rows are used only from an instrument's 253rd post-placeholder bar onwards | H-EXT common rule (252 sessions of history) |
| H-ML-003 folds | Calendar time: first test block starts 730 days after the first date; 91-day test blocks; training rows have dates < test start - 7 days | Before 2017 the union of dates is not every calendar day, so counting union dates (`date_folds`) would not give 2 y / 1 quarter (amendment A2, point 2) |
| DL early stopping | Validation tail = last 15% of training rows, as in H-ML-002. The purge between fit rows and tail is **3 x max rows per date (333 rows)**, not 2 rows | In a 111-instrument panel, 2 rows is less than one date. 333 rows guarantees >= 2 whole dates. This affects early stopping only |
| DL batch / epochs | **Unchanged** (batch 256, <= 200 epochs, patience 15) | Benchmark of the largest fold (595,547 training rows): MLP 80 s, GRU 96 s. No override was needed |
| H-MM-001 F columns | revenue_growth_yoy, net_margin, roe, leverage, earnings_yield, earnings_yield_ni, book_to_market, log(market_cap) | Scale-free columns. Raw USD TTM levels and the EPS level are not comparable across firms. Non-finite ratios (a filed share count of 0 gives market_cap = 0, e.g. MRK/KO in 2009, book_to_market = inf) are set to missing |
| H-MM-001 M columns | Every `macro_features` column: TNX, FVX, IRX, VIX, DXY, USDINR, CPI, UNRATE, slope_10y_3m, tnx_chg_20d, vix_z20, cpi_3m_ann, unrate_chg_3m | as registered ("macro, as-of"). FRED is absent, because it is manual-CSV only |
| H-MM-001 target | R = adjusted open(t+22) / open(t+1) - 1 (decision at close t, fill at open t+1, 21 sessions held); target = 1[R_i > equal-weight mean of R over eligible US stocks on that date] | Same execution convention as all ASTRA targets |
| H-MM-001 month ends | Last session of each calendar month; the month of the final data session is excluded | That month may be incomplete |
| H-MM-001 purge/embargo | The 2 decision months before each test year are dropped from training (purge 1 + embargo 1) | Same convention as `PurgedWalkForward` (gap = horizon + embargo) |
| Imputation | `SimpleImputer(median, add_indicator=True)` inside the Pipeline, fitted on training rows only. It is applied to all sets and both models; P has no missing values, so it is a no-op there | as instructed |
| Inference | Row-weighted mean loss difference, Newey-West over clusters (dates for H-ML-003, decision months for H-MM-001), one-sided p, BH at 5% | amendment A2 |
| Importance | Permutation importance (neg log loss) on each **test** fold, XGBoost only; weighted by test rows; reported, not used for any selection | as instructed |
| Resource probes | Wall time; `tracemalloc` peak; process working set sampled every 0.2 s (Win32 `GetProcessMemoryInfo`, no psutil); `torch.cuda.max_memory_allocated/reserved` | per model |

## H-ML-003 Large pooled daily panel

Experiments: family `20261124T231623-H-ML-003-5c7b09`, which holds the metrics, folds, OOS predictions
and XGBoost test-fold permutation importance. Per-model records: logit `20261124T231624-H-ML-003-a714d1`,
xgb `...-688a09`, mlp `...-4e29c4`, gru `...-b640bd`.
The panel has 111 instruments (config lists 111; the registration says 112) and 602,489 rows. There are
96 calendar folds, with tests from 2003-01-02 to 2026-11-21, and 574,013 OOS rows. Every model is scored
on identical rows. Building the panel took 11 s, with a peak working set of 1.96 GB.

| model | AUC | OOS log loss | baseline LL | DM t vs base (date-clustered) | row-wise t | DM t vs logit | p_BH | verdict | folds better | fold dLL mean +- sd (1e-4) |
|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** (train-fold base rate) | 0.500 | 0.69270 | - | - | - | - | - | - | - | - |
| logit | 0.5134 | 0.69243 | 0.69270 | 2.06 | 3.40 | - | 0.026 | **BH-significant** | 68% | 2 +- 13 |
| xgb | 0.5169 | 0.69224 | 0.69270 | 3.97 | 5.62 | - | 0.0001 | **BH-significant** | 67% | 4 +- 13 |
| mlp | 0.5145 | 0.69229 | 0.69270 | 3.23 | 4.85 | 1.38 | 0.0012 | **BH-significant** | 59% | 3 +- 12 |
| gru | 0.5133 | 0.69249 | 0.69270 | 1.23 | 1.89 | -0.73 | 0.109 | n.s. (inconclusive) | 58% | 1 +- 14 |

Resources per model (full walk-forward). Runtime is wall time. RAM is the sampled process working set,
with the tracemalloc peak in parentheses.

| model | runtime | peak RAM (tracemalloc) | peak VRAM allocated / reserved |
|---|---|---|---|
| logit | 25 s | 1.71 GB (0.61 GB) | 0 / 0 |
| xgb (incl. permutation importance) | 269 s | 1.65 GB (0.48 GB) | 0 / 0 (CPU, 4 threads) |
| mlp | 4,898 s | 1.99 GB (0.62 GB) | 81 MB / 166 MB |
| gru | 4,162 s | **7.35 GB** (1.32 GB) | 1,444 MB / **15,594 MB** |

**H1 is supported** (family conclusion `supported`): 3 of 4 models (logit, XGBoost, MLP) have OOS log
loss significantly below the ex-ante base rate after BH. The DL models do not beat logit (MLP t = 1.38,
GRU t = -0.73), so the H-ML-002 "DL beats logit" criterion would not be met.

How large the effect is, and where it comes from. These breakdowns are descriptive, post hoc, and not
tests:
- The effect is tiny: 0.0002-0.0005 nats per row, AUC 0.513-0.517.
- By market (date-clustered t for xgb / mlp): **crypto** dLL +26e-4 / +18e-4 (t 4.09 / 3.11);
  **US** +4.7e-4 / +5.0e-4 (t 2.57 / 2.96, logit t 3.58); **IN** about 0 (t 0.78 / 0.05).
- By era: 2003-2014, no model has t > 1.4; 2015-2026, xgb t 5.03, mlp 3.73, gru 2.17. The signal is
  recent and concentrated in crypto and US rows, and the universe is survivorship-biased.
- XGBoost test-fold permutation importance: parkinson_20 (0.00052, positive in 82% of folds), ret_1
  (0.00043, 81%), mom_5, rsi_14, vol_20.
- **No trading claim is made.** The registered gate is passed, so a cost-aware backtest of these
  predictions belongs to the backtest-auditor. Statistical skill of this size may not survive costs.

Resource notes:
- The first attempt died in MLP fold 49 with a CUDA `CUBLAS_STATUS_EXECUTION_FAILED` / `unknown error`.
  This was a transient WDDM driver fault while other GPU applications were running; the GPU tested
  healthy right afterwards. Fold-level checkpointing was added (`walk_forward(checkpoint=...)`) and the
  study was rerun from scratch. That rerun completed in one segment, with no resumed folds, and it is the
  run that was logged.
- **The GRU exceeded the ~6 GB RAM budget** (peak working set 7.35 GB). The CUDA caching allocator's
  reserve grew to 15.6 GB across the expanding folds. That is more than the 8 GB card, so WDDM spilled it
  into shared system memory. Allocated memory never exceeded 1.44 GB. `torch.cuda.empty_cache()` is now
  called after every DL fold. This does not affect results; it only affects future runs, and H-ML-004
  already uses it.

## H-MM-001 Multimodal ablation (US stocks; experiment `20261124T201859-H-MM-001-d7acf0`)

The sample has 8,709 month-end rows for 30 stocks from 2001-01 to 2026-09. There are 21 test years
(2006-01 to 2026-09) and 7,232 OOS rows. The OOS base rate of "beats the equal-weight universe" is 0.474.
Runtime for all 8 fits plus test-fold permutation importance: 198 s. Peak working set 1.08 GB, tracemalloc
peak 114 MB, no GPU.

| model | set | OOS log loss | AUC | DM t vs baseline | DM t vs P | delta LL vs P (+ = better) |
|---|---|---|---|---|---|---|
| **baseline** (ex-ante train base rate) | - | 0.6919 | 0.500 | - | - | - |
| logit | P | 0.6935 | 0.505 | -1.83 | - | - |
| logit | P+F | 0.8976 | 0.511 | -1.74 | -1.73 | -0.2041 |
| logit | P+M | 0.7013 | 0.501 | -3.09 | -2.64 | -0.0078 |
| logit | P+F+M | 0.9016 | 0.506 | -1.77 | -1.76 | -0.2082 |
| xgb | P | 0.7010 | 0.501 | -4.61 | - | - |
| xgb | P+F | 0.7032 | 0.518 | -4.03 | -1.09 | -0.0022 |
| xgb | P+M | 0.7044 | 0.499 | -5.71 | -1.79 | -0.0034 |
| xgb | P+F+M | 0.7052 | 0.512 | -5.40 | -2.07 | -0.0041 |

**BH family (3 comparisons x 2 models; all p_BH = 0.996): no comparison is significant. H1 is
rejected (conclusion `rejected`).** P+F+M is not better than P for either model. For XGBoost it is
nominally worse (t = -2.07).

Marginal contributions. Delta is the mean per-row log-loss improvement. "Folds +" is the share of the
21 test years with an improvement. F|P+M and M|P+F are descriptive only and outside the BH family.

| model | contribution | delta LL | DM t | fold std | folds + | 2011+ tests: delta LL (t) |
|---|---|---|---|---|---|---|
| logit | F given P | -0.2041 | -1.73 | 0.917 | 24% | -0.0110 (-2.90) |
| logit | M given P | -0.0078 | -2.64 | 0.016 | 19% | -0.0080 (-2.16) |
| logit | F+M given P | -0.2082 | -1.76 | 0.919 | 14% | -0.0137 (-3.36) |
| logit | F given P+M | -0.2004 | -1.69 | 0.919 | 38% | -0.0057 (-1.36) |
| logit | M given P+F | -0.0041 | -2.54 | 0.009 | 24% | -0.0028 (-1.59) |
| xgb | F given P | -0.0022 | -1.09 | 0.011 | 52% | -0.0008 (-0.33) |
| xgb | M given P | -0.0034 | -1.79 | 0.010 | 38% | -0.0050 (-2.31) |
| xgb | F+M given P | -0.0041 | -2.07 | 0.012 | 33% | -0.0037 (-1.57) |
| xgb | F given P+M | -0.0007 | -0.42 | 0.011 | 43% | +0.0013 (+0.67) |
| xgb | M given P+F | -0.0020 | -1.04 | 0.012 | 43% | -0.0029 (-1.38) |

Notes and diagnostics:
- **The logit P+F result is dominated by one fold, and this is a property of the registered design, not a
  code error.** The 2010 test year trains on data through 2009-10. Only 2.6% of those training rows have
  any fundamental value, because XBRL starts in 2009. After median imputation the F columns are almost
  constant, so the StandardScaler's std is tiny and test-year z-scores reach about 8,800 (roe). That fold
  has log loss 4.90 (fold minimum -4.21). This is not the only problem: on the 2011+ test years, where
  every row has fundamentals, logit F is still worse than P (-0.011, t = -2.90). No winsorization was registered, and none was added after
  seeing this.
- Share of OOS rows with any fundamental: 0% in 2006-2008, 36% in 2009, 97% in 2010, 100% from 2011.
  Fundamentals can only matter from 2010. The 2011+ subsample is reported descriptively and does not
  change the conclusion.
- Macro variables are the same for every stock on a date. They can only help a cross-sectional target
  through interactions (XGBoost) or by shifting the predicted base rate. In practice they hurt both models.
- XGBoost test-fold permutation importance (mean increase in log loss when permuted): log_market_cap
  leads in P+F (0.0042, positive in 71% of folds) and in P+F+M (0.0041). No macro column exceeds 0.0006.
  The size effect is consistent with the AUC gain of P+F (0.518 vs 0.501), but it does not reach a better
  log loss.
- Even the P-only models are worse than the base rate (xgb t = -4.61). The 13 technical features carry
  no usable cross-sectional 21-day signal in this 30-stock universe.

## H-ML-004 Hourly crypto panel (family `20261125T002159-H-ML-004-b672d7`)

Per-model records: logit `20261125T002159-H-ML-004-c2e0a2`, xgb `20261125T002200-H-ML-004-7c7dc1`, mlp
`20261125T002200-H-ML-004-6c941b`.

Setup:
- 16 pairs, 1h bars, 1,020,409 rows. Off-grid Binance bars are filtered in `load_features`.
- Entry after 252 x 24 bars. The 13 features use windows in bars, with seq_len 1.
- Target = sign of log(O_{t+2} / O_{t+1}): decision at the close of bar t, execution at the next open.
- Calendar folds: 730 d initial, 91 d tests, 48 h gap. That gives 27 folds, with tests from 2020-04-28
  to 2026-11-23 and 892,047 OOS rows. The DM test is clustered by bar timestamp.
- Alignment check: open(t+1) equals close(t) to within 1 bp in 88% of BTC bars, so the label window lies
  entirely after the last input to the features.

| model | AUC | OOS log loss | baseline LL | DM t vs base | DM t vs logit | p_BH | verdict | folds better | fold dLL mean +- sd (1e-4) | runtime | peak RAM (tracemalloc) | peak VRAM alloc / reserved |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.500 | 0.69310 | - | - | - | - | - | - | - | - | - | - |
| logit | 0.5370 | 0.69120 | 0.69310 | 10.48 | - | <1e-6 | **BH-significant** | 93% | 19 +- 15 | 12 s | 1.59 GB (0.51 GB) | 0 / 0 |
| xgb | 0.5406 | 0.69068 | 0.69310 | 11.78 | - | <1e-6 | **BH-significant** | 89% | 24 +- 19 | 133 s | 1.46 GB (0.29 GB) | 0 / 0 |
| mlp | 0.5394 | 0.69082 | 0.69310 | 11.43 | 3.45 | <1e-6 | **BH-significant** | 96% | 23 +- 18 | 2,128 s | 1.92 GB (0.49 GB) | 120 MB / 155 MB |

**H1 is supported**: all 3 models beat the ex-ante base rate. The MLP also beats logit (t = 3.45).
- XGBoost test-fold permutation importance: ret_1 (0.0022, positive in 96% of folds), mom_5 (0.0011),
  rsi_14, bb_z_20. This is a short-horizon reversal/continuation pattern in the last 1-5 bars.
- Panel load took 2 s. The run completed in one segment.

**Registered cost-aware backtest.** Long when p > 0.5, flat otherwise, per pair; equal weight across
pairs; 26 bps round trip, charged as half per position change. Descriptive, from `hourly_backtest`:

| model | gross mean/bar | gross Sharpe (ann.) | gross NW t | net Sharpe @ 26 bps | net NW t | turnover per bar | avg exposure |
|---|---|---|---|---|---|---|---|
| logit | 6.4e-5 | 1.04 | 2.94 | **-4.43** | -12.8 | 0.26 | 0.46 |
| xgb | 8.7e-5 | 1.43 | 4.09 | **-6.44** | -18.9 | 0.37 | 0.47 |
| mlp | 8.4e-5 | 1.40 | 4.00 | **-6.16** | -18.0 | 0.35 | 0.45 |

**The statistical skill is real but not tradable at 26 bps.** A position flip costs about 13 bps on each
side. The models change position on 26-37% of bars, while the gross edge is under 1 bp per bar. The gross
figures also include long-only crypto beta. This is a negative trading result. Any further trading
evaluation (e.g. a hysteresis band or lower-turnover rules) would be a new, separately registered
hypothesis, and it belongs to the backtest-auditor.

## Reproduce

```
uv run --no-sync python -c "from astra.research.ml_extended import run_ml_panel_extended; run_ml_panel_extended()"
uv run --no-sync python -c "from astra.research.multimodal import run_multimodal; run_multimodal()"
uv run --no-sync python -c "from astra.research.ml_extended import run_hourly_crypto; run_hourly_crypto()"
uv run --no-sync pytest tests/test_ml_extended.py
```
