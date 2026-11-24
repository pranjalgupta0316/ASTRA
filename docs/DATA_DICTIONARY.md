# Data dictionary

## Validated daily bars (`data/validated/market=*/symbol=*/freq=1d/bars.parquet`)
Authoritative definition with types: `src/nebula/data/contracts.py` (`BAR_SCHEMA`, schema_version 1).
Each partition has `manifest.json`: rows, date range, quality status, quality report path, raw file
path and sha256, own sha256, write time.

## Raw layer (`data/raw/<provider>/<provider_symbol>/1d/`)
`raw_<UTC stamp>.parquet` = provider frame verbatim; `.meta.json` = provider, request params, library
version, retrieval time, row count, sha256, source files (Binance zips with their sha256).
`files/` (Binance) = original checksummed zips.

## Features (`nebula.features.technical.build_features`)
| column | definition (P = total-return adjusted close, all windows trailing) |
|---|---|
| ret_1 | log(P_t / P_{t-1}) |
| mom_k (5,20,60) | log(P_t / P_{t-k}) |
| sma_ratio_k (20,50) | P_t / SMA_k(P)_t − 1 |
| vol_20 | std of ret_1 over 20 sessions (per-period, not annualized) |
| parkinson_20 | sqrt(mean_20(ln(H/L)² / (4 ln 2))) |
| rsi_14 | Wilder RSI, α = 1/14 |
| macd_hist_norm | (MACD(12,26) − signal(9)) / P_t |
| bb_z_20 | (P_t − SMA_20) / std_20(P) |
| volume_z_20 | z-score of log volume vs trailing 20 |
| autocorr_20 | rolling 20-session corr(ret_1, ret_1 lagged 1) |
| target_ret_oo | log(O_{t+2} / O_{t+1}) - return of a position decided at close t (**forward-looking**) |
| target_ret_cc | ret_1 at t+1 (**forward-looking**) |
| target_up | 1[target_ret_oo > 0] (**forward-looking**) |

## Quality report (`data/reports/quality/<market>_<symbol>_<stamp>.json`)
Checks: schema, nulls, duplicates, monotonic_time (+tz-aware), price_consistency, volume, sessions
(vs calendar), outlier_returns (MAD z > 12, warn), corporate_actions (adjustment factor monotone).

## Experiment record (`experiments/<id>/record.json`)
See `nebula.tracking.tracker.ExperimentRecord`.
