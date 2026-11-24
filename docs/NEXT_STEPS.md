# Next steps (in order)

1. **Make the first commit** (nothing committed yet) and, if desired, push to a private GitHub repo so
   the CI workflow runs (needs the developer's approval - publishing).
2. Finish PyTorch install (KI-2); run `uv run nebula env` to verify `torch_cuda_available` and VRAM.
3. Expand universe per config `expansion` (MSFT, NVDA, SPY, QQQ; TCS, INFY, HDFCBANK, ^NSEI; ETH, SOL)
   - re-run ingestion + quality reports; these instruments also serve as the held-out set for H-XM-003.
4. Run H-XM-003 (Sharpe-difference test) on held-out instruments only.
5. Volatility modelling (M5 cont.): EWMA / GARCH / GJR-GARCH (arch) vs realized, Parkinson and
   Garman-Klass proxies; QLIKE evaluation.
6. Classical ML breadth (M6): ridge/lasso, RF, XGBoost on identical folds with the A1 decision rule;
   then a small PyTorch MLP/GRU baseline on GPU with memory profiling.
7. Microstructure (M7): Binance aggTrades analytics; bounded depth-stream capture (measure MB/hour
   first) with U/u gap detection tests.
8. Portfolio & risk (M8): CVXPY min-var / risk parity with Ledoit-Wolf; analytic tests.

**Precise next task**: `git add -A; git commit` (after reviewing `git status`), then step 2.
