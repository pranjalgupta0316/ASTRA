# Commit map (history squashed on 2026-11-27)

Experiment records and docs cite commit ids from the original history. The full original history is kept
locally (tag `archive/full-history-2026-11-27`, branch `backup/pre-squash-2026-11-27`). Each old commit's
code is contained in the new commit listed next to it (same files as the last old commit of its group).

| old commit | new commit | subject |
|---|---|---|
| 44f357b | 0e2c4fc | Add NEBULA vertical slice: data, research, backtester, dashboard |
| e0915d9 | 0e2c4fc | Record verified GPU/PyTorch environment |
| d379710 | ac2380c | Expand universe to 13 instruments; drop in-progress Yahoo sessions |
| 1304fea | ac2380c | Add ML/DL study (H-ML-001/002) and held-out momentum test (H-XM-003) |
| ea3a3ec | ac2380c | Add deterministic runtime research agents and report generator |
| 44a00a8 | ac2380c | Add market microstructure module (M7) |
| b3fab7a | ac2380c | Add portfolio construction and risk module (M8); lint fixes |
| 79de175 | ac2380c | Add volatility forecasting and regime detection (M5) |
| 367952d | cb269a8 | Expand dashboard to all research modules; shared record subject helper |
| 5c74de4 | cb269a8 | Add study CLI commands and run-all pipeline |
| 06b9574 | cb269a8 | Fix .gitignore: track the src/nebula/data package |
| 2ec4dbe | cb269a8 | Apply final audit and security review fixes |
| 642025c | d595823 | Docs: README commands/universe, endpoints, ADR-0005/0006, known issues |
| 50b3508 | d595823 | Final validation, dashboard test, and project reports |
| 3a7f4af | e8b60d0 | Extended universe and intraday ingestion infrastructure |
| 41e7bc1 | e8b60d0 | Pre-register extended-universe hypotheses (H-CS, H-VT, H-XMKT, H-ML-003/004, H-MM-001, H-MS-002) |
| e50e633 | e8b60d0 | Local LLM: 8k context window and benchmark script |
| 02c489c | e8b60d0 | Add SEC EDGAR point-in-time fundamentals; FRED reader limited to manual CSVs |
| bae60ff | e8b60d0 | Add terms-compliant macro data: Yahoo market series + BLS CPI/unemployment |
| cf22d31 | e8b60d0 | CLI: ingest-fundamentals and ingest-macro commands |
| c1d282a | 6796969 | Add cross-sectional momentum and reversal studies (H-CS-001/002) |
| 0f5db1b | 6796969 | Fix H-CS benchmark cash drag; momentum now inconclusive, reversal rejected |
| 3dbbc45 | 6796969 | Add vol-managed (H-VT-001) and cross-market (H-XMKT-001) studies; fix engine tolerance default; handle negative adjusted prices |
| cb95b0d | 6796969 | Intraday: off-grid Binance bars warn (not fail) and are excluded by research loaders |
| 4138728 | 8887f50 | Dashboard: extended-research page (verdicts, equity curves, results write-ups) |
| 2f8e927 | 8887f50 | Fix buy-and-hold benchmark under-investment when the volume cap cuts the first fill |
| 7450b96 | 8887f50 | H-VT-001 results: corrected single-instrument counts after benchmark fix |
| 0b3c64b | 8887f50 | Add large-panel ML (H-ML-003), hourly crypto ML (H-ML-004), multimodal ablation (H-MM-001) |
| 12292ee | 8887f50 | ML extended: audit findings and wording (statistical vs economic significance) |
| 5b3bd96 | 8887f50 | Fix docstring line length |
| c9e49db | a77a83c | CLI: ext-study, ml-ext and run-all covering every study |
| 1acf2b9 | a77a83c | Docs: ADR-0008 (macro sources, FRED terms), extended data sources, known issues |
| d3653bb | a77a83c | Add H-MS-002 (6 h of BTCUSDT L2): imbalance predicts mid moves, not tradeable |
| 5a3e35f | a77a83c | v0.3 final report, status, validation, next steps; README for the extended platform |
| 5ed006b | 016e121 | Bump version to 0.3.0 |
| 5b281b9 | 016e121 | Rename project NEBULA -> ASTRA |
| 9306eb7 | f2021af | P1: dataset versioning with immutable snapshots and named data locks |
| 0327d69 | f2021af | Pre-register realized-volatility hypotheses H-RV-001/002/003 |
| 0bcf39f | f81e94e | P2: engine verification - property tests, independent reference, golden regression |
| dac05b9 | f81e94e | ADR-0009: data locks |
| 0303cf0 | f81e94e | Docs: sync architecture, methodology and roadmap with the implementation |
| c85e4fb | 4d985d3 | Merge P3: realized-volatility study (H-RV-001/002/003) |
| 5094cf6 | 4d985d3 | H-RV results: disclose the second data touch and add audit caveats |
| b2bfd90 | 2e83560 | P7: global FDR, deflated Sharpe and calibration; P9: computed superseded flag |
| 36fc8b6 | 2e83560 | P9: dashboard shows superseded/commit/data-lock per record and a data-versions & multiplicity page |
| 4c8ab8a | 2e83560 | P8: point-in-time S&P 500 membership and a survivorship check |
| 8bb7f29 | e8dd0bd | P6: order-level execution simulator on L1 quotes + trades |
| 49c5859 | e8dd0bd | Pre-register H-MS-003/004 (passive vs taker execution, adverse selection) on USD-M perps |
| 66eeb83 | e8dd0bd | Docs: README commands and changelog for the 0.4 improvement programme (so far) |
| e81458d | e8dd0bd | H-MS-v2 Amendment 1: fallback timing and limit-price reference (before any data) |
| a6b8a0c | e8dd0bd | H-MS-003/004 study core (per-day, pure) with synthetic tests |
| 077c1e4 | e8dd0bd | Execution study: per-parent time windows (result-identical, tested); ~25 s per 20M-quote day |
| 0f081af | ff51c4a | Merge GitHub's initial commit (keep the project README) |
| 38430c3 | ff51c4a | README: full project name and tagline |
| e01e7e8 | b20cc46 | CI: run without the optional torch extra |
| 4a00a75 | 4d985d3 | H-RV: Amendment 1 (implementation details fixed before any run) (merged P3 branch) |
| 4c449a4 | 4d985d3 | Tracker.save_frame: allow ASTRA_EXPERIMENTS_DIR outside the repo root (merged P3 branch) |
| 2838bee | 4d985d3 | H-RV: realized-volatility study (RV5/RV1/RV_h, HAR-RV, benchmarks, DM/BH) + astra rv-study (merged P3 branch) |
| c6b31e5 | 4d985d3 | H-RV results: all three families inconclusive (H-RV-001 2/6 cells supported, H-RV-002 0/16, H-RV-003 0/2) (merged P3 branch) |
| 0f79fda | ff51c4a | GitHub's initial README commit (merged) |
