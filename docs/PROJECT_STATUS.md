# Project status

Last update: 2026-11-25. Version 0.3 - extended universe, fundamentals/macro, large-scale ML, 6 h of
order-book data. Final report: [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

| Area | State | Notes |
|---|---|---|
| Data: 111 instruments daily (2000→), 16 crypto hourly, BTC/ETH 1-minute | tested on real data | 637k daily bars, ~1M hourly, ~9.7M minute bars; 0 ingestion failures |
| SEC fundamentals (30 US stocks), Yahoo/BLS macro | tested on real data | point-in-time / release-date-aware; FRED manual-only (terms) |
| Core studies (13 instruments, frozen window) | validated | reproduced from clean commit d3653bb |
| Extended studies (H-CS, H-VT, H-XMKT, H-MM, H-ML-003/004) | validated after audits | three audits forced fixes; corrections documented |
| Order-book research (H-MS-001/002) | validated (single night, single venue) | 6 × 1 h captures, 0 gaps, 163 verify snapshots matched |
| Local LLM (Ollama, qwen2.5:7b) | tested | narration only; ~37 tok/s, ~5.1 GB VRAM, 0 invented numbers |
| Dashboard | tested | 13 pages render (headless AppTest) |
| Repository | clean, all work committed locally | never pushed (by request) |

## Reproduction provenance
`astra run-all --skip-ingest` from clean commit d3653bb (32.7 min, 218 records, 0 dirty) plus
H-MS-001/002 re-run from the same commit. Not re-run in the final pass: H-ML-003/004 MLP and GRU
(~2.5 h; records from the earlier audited run) and the live captures themselves (not replayable).
Versus the previous clean run (642025c): 216/218 verdicts identical; SOLUSDT H-XM-001 changed because
the buy-and-hold benchmark fix corrected an under-invested benchmark, and the pooled GRU (H-ML-002)
crossed −1.96 after Yahoo restated adjusted prices (≈1e-6 relative changes elsewhere).

## Environment (measured)
i7-14700HX 20C/28T · 15.8 GB RAM · RTX 4060 Laptop 8.59 GB (driver 616.92) · Python 3.12.16 (uv) ·
torch 2.11.0+cu128 · Ollama 0.40.18. Disk: data ≈ 1.5 GB incl. minute bars and captures.
Heaviest job: pooled GRU on 600k rows - 69 min, peak 7.35 GB RAM (KI-25).
