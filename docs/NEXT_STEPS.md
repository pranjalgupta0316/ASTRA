# Next steps (ranked)

1. **Push to a private GitHub repo** (needs the developer's approval) so the CI workflow actually runs;
   CI installs without the torch extra (DL tests skip).
2. **Fundamentals & macro (multimodal)**: SEC EDGAR companyfacts keyed on filing date; FRED/ALFRED
   vintages; then the registered ablation design (price-only vs + fundamentals vs + macro).
3. **Longer microstructure captures** with the disk-capped script (e.g. 6 × 1 h across sessions) to give
   H-MS-001 real power; add trade-sign/OFI horizon tests net of taker fees.
4. **Cross-market portfolio** with an FX adapter and a calendar-alignment ADR (ADR-0006 follow-up).
5. **Volatility targeting study**: register a hypothesis that uses GJR/EWMA forecasts to size positions,
   audited by backtest-auditor (H-VOL/H-REG results are forecasting claims, not trading claims).
6. Point-in-time index constituents (if a free legal source exists) to address survivorship.
7. Optional: install a small local LLM (e.g. via Ollama), benchmark RAM/VRAM/latency/quality, and enable
   report narration (`NEBULA_LLM_URL`, `NEBULA_LLM_MODEL`).

**Precise next task**: create the private remote and confirm CI passes (developer decision).
