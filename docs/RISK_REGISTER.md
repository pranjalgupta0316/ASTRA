# Technical risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Yahoo endpoint changes / blocks | medium | high | raw snapshots archived; adapter interface; alternative adapters (Stooq, manual bhavcopy) |
| R2 | Binance geo-restriction of api.binance.com | medium | medium | use data.binance.vision + data-api.binance.vision mirrors |
| R3 | Look-ahead via adjusted prices / features | medium | high | returns-only features, truncation test, leakage canary |
| R4 | Overfitting via repeated OOS inspection | high | high | hypotheses registered first, fixed params, variants counted, DSR |
| R5 | 16 GB RAM exhausted by tick data | medium | medium | hourly partitioned Parquet, DuckDB/Polars lazy scans, capture-rate measurement first |
| R6 | 8 GB VRAM limits DL | low | low | small models; daily data is small; CPU fallback |
| R7 | NSE calendar proxy (XBOM) differs on special sessions (Muhurat) | medium | low | sessions check flags off-calendar rows; documented |
| R8 | Survivorship bias from hand-picked universe | high | medium | documented; no cross-sectional generalization claims |
| R9 | Free-tier dependencies vanish (FRED key etc.) | low | medium | local caches of all downloads |
| R10 | Windows path / timezone quirks | medium | low | pathlib everywhere; tz-aware UTC; tests on Windows |
