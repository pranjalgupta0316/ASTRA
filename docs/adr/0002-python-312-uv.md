# ADR-0002: Python 3.12 managed by uv (system Python 3.14 not used)

- **Decision**: pin `requires-python = ">=3.12,<3.13"`; uv installs CPython 3.12 and manages `.venv` +
  `uv.lock`. PyTorch comes from the official CUDA 12.8 wheel index (`[tool.uv.sources]`).
- **Problem**: the machine's default Python is 3.14.4. Scientific wheels (PyTorch CUDA, cvxpy solvers,
  arch, xgboost) have the broadest, most battle-tested Windows support on 3.12.
- **Alternatives**: 3.14 (newest; higher risk of missing/new wheels and fewer known-good combos);
  conda (heavier; slower; second package manager); pip+venv (no lockfile).
- **Benefits**: deterministic lockfile, fast installs, one tool.
- **Trade-offs**: an extra interpreter on disk (~100 MB).
- **Risks**: cu128 wheels require a driver supporting CUDA 12.8 - driver 616.92 reports CUDA 13.4 UMD,
  which is backward compatible. No system CUDA toolkit is needed for PyTorch wheels.
- **Reversibility**: high - bump the pin and re-lock.
