# Known issues and limitations

| ID | Issue | Impact | Status / workaround |
|---|---|---|---|
| KI-1 | Yahoo intermittently fails (one transient AAPL network-test failure) | ingestion flakiness | 4-attempt exponential backoff; not observed failing since |
| KI-2 | PyTorch cu128 download took ~1 h | one-time install delay | resolved; CPU-only alternative: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| KI-3 | pandas stubs produce ~280 pyright warnings (Series\|bool, Timestamp\|NaT unions) | type-check signal-to-noise | six rules downgraded to warnings; all other rules are errors (0 errors) |
| KI-4 | NSE sessions validated against XBOM (BSE) calendar; Diwali Muhurat sessions flagged off-calendar | Indian quality = warn | expected; rows kept |
| KI-5 | Yahoo placeholder rows (V=0, O=H=L=C) for Indian names (2025-03-18, 2026-01-15); NIFTY50 volume unreliable | fake returns / fills | excluded from research (recorded per experiment); engines never fill on zero volume |
| KI-6 | Yahoo missing 2-3 NSE sessions per symbol | small gaps | flagged by the sessions check |
| KI-7 | Pandas4Warning (Series.values on tz-aware) from a dependency path | none today | revisit when pandas 4 lands |
| KI-8 | Serena MCP server times out on connect in Claude Code | dev tooling only | pyright + ripgrep used instead |
| KI-9 | Equity spreads/slippage are assumptions (no free historical quotes) | net results depend on them | 0×/1×/2×/4× cost sensitivity stored per backtest |
| KI-10 | Universe = today's well-known large caps | survivorship bias | stated in every record; no cross-sectional generalization claimed |
| KI-11 | DL predictions differ CPU vs GPU (≤ 0.10 probability) | exact cross-device repro impossible | device recorded per experiment (ADR-0005) |
| KI-12 | H-MS-001 rests on one 5-minute capture, 9 bootstrap blocks; 1 s result near-mechanical with a one-tick spread | weak microstructure evidence | longer captures (disk-capped script) before any claim |
| KI-13 | H-REG-001 largely restates volatility clustering (state built from 20d vol) | limited novelty | incremental value over EWMA not tested |
| KI-14 | H-PF-002 tests mean return, not Sharpe; two strategies per market without multiplicity correction | test/claim mismatch | Sharpe-difference CI reported as secondary; no result is marginal |
| KI-15 | Several decision rules were amended after an audit (A1 for H-XM-002, A2 for H-ML-002) | post-hoc risk | amendments disclosed; original-rule verdicts kept in records; verdict direction unchanged |
| KI-16 | Local LLM (qwen2.5:7b) benchmarked: ~37 tok/s, ~5.1 GB VRAM, 0 invented numbers in 3 runs, but wording not deterministic across runs | narration only, never numbers | numbers in reports always come from records |
| KI-17 | No FX conversion / cross-market portfolio | no cross-market diversification study | ADR-0006 |
| KI-18 | News / alternative data not built; no free official Indian fundamentals | multimodal limited to US price + SEC fundamentals + macro | SEC and macro built 2026-11-25 (H-MM-001) |
| KI-20 | Extended universe (111 names) = today's large caps / NIFTY names / crypto majors | strong survivorship bias for cross-sectional results (momentum excess concentrated in later mega caps) | stated in every record; point-in-time constituents not freely available |
| KI-21 | FRED automation not permitted by its terms | registered FRED series replaced by Yahoo market series + BLS | ADR-0008; deviation disclosed in H-MM / H-XMKT records |
| KI-22 | BLS values are revised, not first-release vintages | small look-ahead in macro levels | availability uses actual release dates; documented |
| KI-23 | Yahoo back-adjustment produced negative adj_close (ADANIENT 2002-07..09) | invalid returns | validation warns; research treats rows as missing |
| KI-24 | Binance hourly/minute klines off the bar grid after the 2018-02-09 outage | misaligned bars | validation warns (<1%); research filters to the grid |
| KI-25 | GRU on the 600k-row panel peaked at 7.35 GB RAM and spilled VRAM into shared memory (8 GB card) | memory pressure on 16 GB laptop | CUDA cache cleared per fold; don't run concurrently with other heavy jobs |
| KI-26 | Two engine defects found by audits and fixed: absolute 2 pp rebalance tolerance (benchmark cash drag) and single-order buy-and-hold under-invested after volume caps | earlier H-CS and H-VT single-instrument results were wrong | fixed (commits 3dbbc45, 2f8e927); superseded records kept, corrections documented |
| KI-27 | Some extended-study records were produced while other agents' files were uncommitted (dirty tree) | provenance flag | final `run-all` from a clean commit regenerates them |
| KI-19 | Modules without direct unit tests: env_report, logging, dashboard app (manually exercised), some study runners (exercised end-to-end by run-all) | regressions could slip | CLI smoke test + run-all reproduction |
