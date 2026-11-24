# Known issues

| ID | Issue | Impact | Status / workaround |
|---|---|---|---|
| KI-1 | Yahoo intermittently fails (one AAPL network-test failure, not reproduced in 4 reruns; likely throttling) | ingestion flakiness | 4-attempt exponential backoff added in YahooAdapter; not yet observed failing since |
| KI-2 | PyTorch cu128 wheel download from download.pytorch.org was very slow (~1 h) | one-time install delay | **resolved** 2026-11-24: torch 2.11.0+cu128 installed; CPU-only fallback: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| KI-3 | pandas stubs produce ~100 pyright warnings (Series|bool, Timestamp|NaT unions) | type-check signal-to-noise | six rules downgraded to warnings in pyproject; other rules remain errors |
| KI-4 | NSE sessions validated against XBOM (BSE) calendar; Diwali Muhurat sessions show as off-calendar | RELIANCE quality = warn | expected; rows kept |
| KI-5 | Yahoo placeholder rows (2025-03-18, 2026-01-15 RELIANCE: V=0, O=H=L=C) | could create fake fills/returns | excluded in research (recorded), engine skips zero-volume fills; 2025-03-18 may be a real session with missing data |
| KI-6 | Yahoo missing 2 NSE sessions (2019-02-13, 2019-03-29) | small gaps | flagged by sessions check |
| KI-7 | Pandas4Warning (Series.values on tz-aware) from a dependency path | none today | revisit when pandas 4 lands |
| KI-8 | Serena MCP server times out on connect in Claude Code | dev tooling only | pyright + ripgrep used instead |
| KI-9 | Equity spreads are assumptions (no free historical quotes) | net results depend on assumption | cost sensitivity 0×/1×/2×/4× stored per experiment |
| KI-10 | Hand-picked current large-caps → survivorship bias | no cross-sectional generalization | documented in methodology |
