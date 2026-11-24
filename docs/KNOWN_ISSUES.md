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
| KI-16 | Optional local-LLM narration tested only against a fake server; no model installed/benchmarked | feature unvalidated | platform does not depend on it |
| KI-17 | No FX conversion / cross-market portfolio | no cross-market diversification study | ADR-0006 |
| KI-18 | Fundamentals (SEC EDGAR), macro (FRED/ALFRED) and news adapters are researched only | multimodal research not possible yet | see NEXT_STEPS |
| KI-19 | Modules without direct unit tests: env_report, logging, dashboard app (manually exercised), some study runners (exercised end-to-end by run-all) | regressions could slip | CLI smoke test + run-all reproduction |
