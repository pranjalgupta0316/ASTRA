# H-RV results: realized volatility (H-RV-001 / 002 / 003)

Registration: `docs/research/H-RV.md` (committed 0327d69 before implementation; Amendment 1 committed
4a00a75 after inspecting data coverage only, before any RV, forecast or loss was computed).
Code: `src/astra/research/realized_vol.py` at commit 2838bee (records: `code_version.dirty = false`).
Data: lock `v0.3.0-baseline` (fingerprint `4fd83821...b7b729`, stored in every record's `data_lock`).
Reproduce: `uv run astra --lock v0.3.0-baseline rv-study` (about 20 s). Each cell is one
pre-registered specification (`variants_tried = 1`). Family records count their cells as variants.

**Disclosure (added 2026-11-26 after the independent audit):** the study touched the real data **twice**,
not once. A first full run (10:49:48 UTC) used uncommitted code on top of 4c449a4, into a scratch
experiments directory; the code was then committed (2838bee, no change to the method) and the recorded
run followed at 10:51:06 UTC. The first run's H-RV-001 and H-RV-003 DM statistics and verdicts are
identical to the recorded run, so nothing was changed after seeing results. The stored records say
`test_touches: 1`; the correct value is 2. Records are left unaltered (provenance) and corrected here.

**Verdict: all three families are inconclusive.**

**Caveat on the two supported ETH cells:** the ETH GARCH(1,1)-t fit sits at the stationarity boundary
(alpha + beta = 0.99999999, nu = 3.24); its mean forecast is 24.8 %² against a mean target of 18.3 %².
The two significant ETH cells therefore mostly reflect GARCH over-forecasting, not a general HAR
advantage: HAR vs EWMA on ETH is not significant (t = 0.93).

- H-RV-001: 2 of 6 cells supported (ETHUSDT HAR vs GARCH, HAR vs GJR). The family criterion (HAR
  significantly better than EWMA and GARCH(1,1) on both assets) is not met.
- H-RV-002: 0 of 16 cells are BH-significant (needed: 12).
- H-RV-003: both cells inconclusive. The precise proxy does not resolve H-VOL-001.

No cell was rejected: no benchmark is significantly better than HAR anywhere. These are claims about
forecasting accuracy, not tradeable claims. Nothing here has been audited yet.

## Conventions
- DM t = Newey-West t of mean(loss_benchmark - loss_model), lag 8 (default floor(4(n/100)^(2/9))).
  Positive means HAR is better (GJR in H-RV-003). p is two-sided normal. p_BH is Benjamini-Hochberg
  within the family, q = 0.05.
- QLIKE is normalised (RV/h - ln(RV/h) - 1). MSE is in percent^4 (variance x 1e4, squared). Losses are
  computed on the common evaluation sample of each asset. No target RV was zero.

## H-RV-001: HAR-RV vs EWMA / GARCH(1,1)-t / GJR-t, target RV5_{t+1}
Evaluation: 2019-10-14 to 2026-11-22, n = 2569 days per asset.

| Asset | Cell | QLIKE HAR | QLIKE bench | MSE HAR | MSE bench | DM t (QLIKE) | p | p_BH | DM t (MSE) | Conclusion | Experiment ID |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | HAR vs EWMA | 0.3868 | 0.4256 | 843.0 | 864.9 | 1.33 | 0.184 | 0.220 | 0.55 | inconclusive | 20261126T105106-H-RV-001-234a6f |
| BTCUSDT | HAR vs GARCH | 0.3868 | 0.4043 | 843.0 | 858.0 | 1.50 | 0.133 | 0.199 | 0.37 | inconclusive | 20261126T105106-H-RV-001-314065 |
| BTCUSDT | HAR vs GJR | 0.3868 | 0.4086 | 843.0 | 868.3 | 1.64 | 0.100 | 0.199 | 0.65 | inconclusive | 20261126T105106-H-RV-001-9b53ed |
| ETHUSDT | HAR vs EWMA | 0.3496 | 0.3752 | 1999 | 2097 | 0.93 | 0.354 | 0.354 | 1.27 | inconclusive | 20261126T105106-H-RV-001-894e63 |
| ETHUSDT | HAR vs GARCH | 0.3496 | 0.4003 | 1999 | 2051 | 6.37 | 1.9e-10 | 5.7e-10 | 0.30 | **supported** | 20261126T105106-H-RV-001-99a4ce |
| ETHUSDT | HAR vs GJR | 0.3496 | 0.4013 | 1999 | 2056 | 6.40 | 1.5e-10 | 5.7e-10 | 0.33 | **supported** | 20261126T105106-H-RV-001-159172 |

Family record `20261126T105106-H-RV-001-255492`: **inconclusive** (supported cells: 2, rejected: 0,
inconclusive: 4).

All models on the same sample (baselines included; RW and 22-day mean are descriptive):

| Asset | Model | QLIKE | MSE | Mean forecast (%^2) | MZ intercept (%^2) | MZ slope | MZ R^2 |
|---|---|---|---|---|---|---|---|
| BTCUSDT | HAR | 0.3868 | 843.0 | 13.24 | -0.34 | 0.877 | 0.193 |
| BTCUSDT | EWMA | 0.4256 | 864.9 | 10.13 | 0.50 | 1.063 | 0.166 |
| BTCUSDT | GARCH | 0.4043 | 858.0 | 11.94 | -1.11 | 1.036 | 0.172 |
| BTCUSDT | GJR | 0.4086 | 868.3 | 11.86 | -0.69 | 1.008 | 0.161 |
| BTCUSDT | RV random walk | 0.4903 | 1204 | 11.24 | 6.56 | 0.418 | 0.175 |
| BTCUSDT | 22-day mean RV | 0.4800 | 1002 | 11.24 | 4.75 | 0.579 | 0.067 |
| ETHUSDT | HAR | 0.3496 | 1999 | 21.20 | 0.26 | 0.849 | 0.169 |
| ETHUSDT | EWMA | 0.3752 | 2097 | 17.96 | 3.18 | 0.840 | 0.123 |
| ETHUSDT | GARCH | 0.4003 | 2051 | 24.79 | 0.46 | 0.718 | 0.184 |
| ETHUSDT | GJR | 0.4013 | 2056 | 24.91 | 0.52 | 0.712 | 0.184 |
| ETHUSDT | RV random walk | 0.4091 | 3007 | 18.22 | 11.56 | 0.368 | 0.135 |
| ETHUSDT | 22-day mean RV | 0.4190 | 2282 | 18.23 | 7.38 | 0.597 | 0.074 |

Mean target RV5 over the sample: BTC 11.27, ETH 18.27 (%^2).

HAR coefficients: 124 refits per asset. First refit 2019-10-14 (730 training rows), last 2026-11-09
(3285 rows). BTC b_d/b_w/b_m: 0.32/0.26/0.22 at the first refit, 0.32/0.22/0.24 at the last. ETH:
0.31/0.26/0.18 at the first, 0.25/0.29/0.19 at the last. **Floored HAR forecasts: 0** (RV5 and RV1, both
assets). GARCH/GJR failed or non-converged refits: 0.

## H-RV-002: HAR on hourly RV vs EWMA, target RV_h_{t+1}, 16 pairs
Family record `20261126T105108-H-RV-002-2c59fc`: **inconclusive**. 0 of 16 cells are BH-significant
(criterion: >= 12). Before BH, 7 cells have p < 0.05 in favour of HAR and none against.

| Pair | n | QLIKE HAR | QLIKE EWMA | DM t (QLIKE) | p | p_BH | DM t (MSE) | Conclusion | Experiment ID |
|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2605 | 0.4771 | 0.5025 | 1.86 | 0.063 | 0.126 | 1.50 | inconclusive | 20261126T105107-H-RV-002-84a326 |
| ETHUSDT | 2605 | 0.4118 | 0.4535 | 2.43 | 0.015 | 0.061 | 2.74 | inconclusive | 20261126T105107-H-RV-002-b25ec2 |
| SOLUSDT | 1538 | 0.3743 | 0.3960 | 0.88 | 0.381 | 0.554 | 1.68 | inconclusive | 20261126T105107-H-RV-002-772743 |
| BNBUSDT | 2524 | 0.4359 | 0.5190 | 2.86 | 0.004 | 0.061 | 2.45 | inconclusive | 20261126T105107-H-RV-002-b62b5b |
| XRPUSDT | 2350 | 0.5396 | 0.6757 | 2.06 | 0.040 | 0.098 | 1.17 | inconclusive | 20261126T105107-H-RV-002-1faec3 |
| ADAUSDT | 2367 | 0.3988 | 0.4448 | 2.51 | 0.012 | 0.061 | 1.70 | inconclusive | 20261126T105107-H-RV-002-ee907f |
| DOGEUSDT | 1935 | 0.5424 | 0.5395 | -0.08 | 0.939 | 0.968 | -0.20 | inconclusive | 20261126T105107-H-RV-002-52d240 |
| LTCUSDT | 2487 | 0.4288 | 0.4829 | 2.02 | 0.043 | 0.098 | 0.25 | inconclusive | 20261126T105107-H-RV-002-4000e0 |
| LINKUSDT | 2101 | 0.3624 | 0.3677 | 0.32 | 0.746 | 0.918 | 2.11 | inconclusive | 20261126T105107-H-RV-002-4aa347 |
| DOTUSDT | 1531 | 0.4561 | 0.4548 | -0.04 | 0.968 | 0.968 | -0.86 | inconclusive | 20261126T105108-H-RV-002-499ead |
| TRXUSDT | 2314 | 0.8140 | 0.7320 | -0.65 | 0.518 | 0.691 | 1.52 | inconclusive | 20261126T105108-H-RV-002-93ee89 |
| AVAXUSDT | 1496 | 0.3964 | 0.4008 | 0.17 | 0.861 | 0.968 | 0.82 | inconclusive | 20261126T105108-H-RV-002-e8f286 |
| BCHUSDT | 1791 | 0.4513 | 0.5162 | 2.29 | 0.022 | 0.071 | 2.31 | inconclusive | 20261126T105108-H-RV-002-2c5e2a |
| ATOMUSDT | 2002 | 0.3899 | 0.4168 | 0.88 | 0.377 | 0.554 | 2.25 | inconclusive | 20261126T105108-H-RV-002-1b71a4 |
| XLMUSDT | 2323 | 0.4535 | 0.4853 | 1.16 | 0.246 | 0.436 | 0.16 | inconclusive | 20261126T105108-H-RV-002-6aa08b |
| ETCUSDT | 2313 | 0.4663 | 0.5397 | 2.47 | 0.014 | 0.061 | -0.50 | inconclusive | 20261126T105108-H-RV-002-4c9c80 |

Hourly data validity and Mincer-Zarnowitz (descriptive):

| Pair | Valid days | Invalid days | Off-grid 1h bars excluded | MZ slope HAR | MZ R^2 HAR | MZ slope EWMA | MZ R^2 EWMA |
|---|---|---|---|---|---|---|---|
| BTCUSDT | 3368 | 18 | 43 | 0.945 | 0.252 | 0.881 | 0.163 |
| ETHUSDT | 3368 | 18 | 43 | 1.013 | 0.292 | 0.682 | 0.153 |
| SOLUSDT | 2292 | 4 | 0 | 1.004 | 0.351 | 0.593 | 0.104 |
| BNBUSDT | 3288 | 17 | 43 | 0.955 | 0.327 | 0.718 | 0.227 |
| XRPUSDT | 3113 | 13 | 0 | 0.947 | 0.228 | 0.896 | 0.199 |
| ADAUSDT | 3130 | 13 | 0 | 0.866 | 0.156 | 0.797 | 0.109 |
| DOGEUSDT | 2692 | 7 | 0 | 0.717 | 0.180 | 0.940 | 0.114 |
| LTCUSDT | 3251 | 17 | 43 | 0.744 | 0.127 | 0.783 | 0.109 |
| LINKUSDT | 2860 | 9 | 0 | 1.063 | 0.228 | 0.756 | 0.158 |
| DOTUSDT | 2285 | 4 | 0 | 0.794 | 0.088 | 0.750 | 0.076 |
| TRXUSDT | 3075 | 13 | 0 | 0.922 | 0.198 | 0.540 | 0.131 |
| AVAXUSDT | 2250 | 4 | 0 | 0.936 | 0.201 | 0.820 | 0.120 |
| BCHUSDT | 2547 | 6 | 0 | 1.042 | 0.178 | 0.623 | 0.098 |
| ATOMUSDT | 2758 | 8 | 0 | 1.050 | 0.250 | 0.731 | 0.144 |
| XLMUSDT | 3086 | 13 | 0 | 0.870 | 0.164 | 0.910 | 0.156 |
| ETCUSDT | 3076 | 11 | 0 | 0.771 | 0.148 | 0.927 | 0.170 |

Floored HAR forecasts: 0 for every pair.

## H-RV-003: GJR-GARCH-t vs EWMA, target RV5_{t+1} (re-test of H-VOL-001)
Family record `20261126T105106-H-RV-003-ccc323`: **inconclusive**. The original H-VOL-001 records are
untouched.

| Asset | n | QLIKE GJR | QLIKE EWMA | MSE GJR | MSE EWMA | DM t (QLIKE) | p | p_BH | DM t (MSE) | Conclusion | Experiment ID |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 2569 | 0.4086 | 0.4256 | 868.3 | 864.9 | 0.85 | 0.395 | 0.395 | -0.66 | inconclusive | 20261126T105106-H-RV-003-800585 |
| ETHUSDT | 2569 | 0.4013 | 0.3752 | 2056 | 2097 | -0.95 | 0.343 | 0.395 | 0.28 | inconclusive | 20261126T105106-H-RV-003-656f41 |

Answer to the H-RV-003 question: a more precise proxy does not separate GJR from EWMA on BTC or ETH.
The point estimates point in opposite directions on the two assets.

## Descriptive: QLIKE by volatility regime (terciles of the origin's 22-day mean RV5)
Cut points (%^2): BTC 5.43 / 10.21, ETH 8.96 / 16.81. n = 857/856/856.

| Asset | Regime | HAR | EWMA | GARCH | GJR | RW | 22-day mean |
|---|---|---|---|---|---|---|---|
| BTCUSDT | low | 0.5019 | 0.5132 | 0.4724 | 0.4732 | 0.6623 | 0.5527 |
| BTCUSDT | mid | 0.3783 | 0.4022 | 0.3899 | 0.3910 | 0.4932 | 0.4666 |
| BTCUSDT | high | 0.2801 | 0.3614 | 0.3504 | 0.3617 | 0.3151 | 0.4207 |
| ETHUSDT | low | 0.4756 | 0.4362 | 0.4933 | 0.4970 | 0.5563 | 0.4740 |
| ETHUSDT | mid | 0.3055 | 0.3159 | 0.3626 | 0.3627 | 0.3996 | 0.3683 |
| ETHUSDT | high | 0.2677 | 0.3735 | 0.3450 | 0.3441 | 0.2712 | 0.4147 |

HAR's QLIKE advantage is concentrated in the high-volatility tercile. In the low tercile, GARCH and GJR (BTC)
and EWMA and the 22-day mean of RV (ETH) have lower QLIKE than HAR. No tests were run on these subsets.

## Descriptive robustness: RV1 (HAR re-fit on RV1, target RV1_{t+1}; not in any family, no verdict)
| Asset | Model | QLIKE | MSE | MZ slope | MZ R^2 | DM t (HAR1 better, QLIKE) |
|---|---|---|---|---|---|---|
| BTCUSDT | HAR (RV1) | 0.3219 | 605.8 | 0.945 | 0.295 | - |
| BTCUSDT | EWMA | 0.3737 | 693.2 | 1.035 | 0.191 | 2.75 |
| BTCUSDT | GARCH | 0.3639 | 687.0 | 1.008 | 0.197 | 4.13 |
| BTCUSDT | GJR | 0.3672 | 695.4 | 0.985 | 0.187 | 4.01 |
| BTCUSDT | RV1 random walk | 0.4071 | 793.3 | 0.536 | 0.287 | 4.30 |
| BTCUSDT | 22-day mean RV1 | 0.4132 | 809.4 | 0.611 | 0.089 | 3.77 |
| ETHUSDT | HAR (RV1) | 0.3092 | 1474 | 0.954 | 0.253 | - |
| ETHUSDT | EWMA | 0.3442 | 1704 | 0.813 | 0.140 | 2.09 |
| ETHUSDT | GARCH | 0.3756 | 1670 | 0.698 | 0.210 | 8.24 |
| ETHUSDT | GJR | 0.3766 | 1676 | 0.692 | 0.210 | 8.25 |
| ETHUSDT | RV1 random walk | 0.3391 | 2008 | 0.489 | 0.239 | 1.79 |
| ETHUSDT | 22-day mean RV1 | 0.3827 | 1864 | 0.610 | 0.086 | 3.49 |

With RV1 (a less noisy but microstructure-noise-exposed proxy), the HAR-vs-benchmark DM t-statistics
are larger than with RV5, and all six H-RV-001-type comparisons have t > 1.96 unadjusted. This is a
descriptive robustness check outside the registered decision. It does not change any verdict, and it
should not be read as "HAR wins once you pick the right proxy".

## Data validity (1-minute assets)
| | BTCUSDT | ETHUSDT |
|---|---|---|
| 1m rows in lock | 4,867,039 | 4,867,038 |
| Off-grid 1m bars excluded | 21,602 | 21,602 |
| Zero-volume flat minutes kept (genuine no-trade, return 0) | 23,922 | 32,024 |
| Calendar days (1d klines span) | 3386 | 3386 |
| Valid days (>= 1368 of 1440 bars) | 3343 | 3343 |
| Invalid days | 43 | 43 |
| ... of which with no 1m bars at all | 14 | 14 |
| Valid days with all 288 5-min returns | 3338 | 3338 |
| Valid days with a gap-spanning return (> 5 min) | 5 | 5 |

The invalid-day list is stored in each H-RV-001 cell record's notes. It includes 2017-12-04..18, when
the 1m archive has no bars while the 1d klines exist. Over the evaluation period, mean r^2 / mean RV5 is
0.90 (BTC) and 0.98 (ETH): daily-return benchmarks and RV5 targets are on comparable levels.

## Limitations
- **Power.** QLIKE differentials of crypto variance forecasts are heavy-tailed. BTC HAR has about 9 %
  lower QLIKE than EWMA, yet DM t is only 1.33. Absence of significance is not evidence of equal
  accuracy.
- **Different information sets by design.** HAR uses intraday RV; the benchmarks use daily returns only
  (as registered). A significant HAR win would measure the value of intraday data plus the HAR
  structure, not the HAR structure alone.
- **Two assets for H-RV-001/003.** BTC and ETH are highly correlated, so the cells are not independent.
  BH remains valid under positive dependence.
- **Hourly RV (H-RV-002)** is built from 24 returns, a much noisier target. Pairs listed later have
  shorter samples (about 1500 days for SOL, DOT, AVAX).
- **Common sample.** Origins with an invalid RV_t (no HAR forecast) or an invalid target day are
  excluded for every model. There are about 28 such days per asset in the evaluation window.
- **Not tested.** Log-HAR, HAR-J/CJ and other HAR variants were not tried (not registered). No economic
  (trading or risk) metric was computed.
- **Regime and RV1 tables** are descriptive and unadjusted for multiplicity.
