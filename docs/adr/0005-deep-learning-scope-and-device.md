# ADR-0005: Small PyTorch models, GPU by default, results tagged with device

- **Decision**: deep learning limited to small MLP (13→32→16) and GRU (hidden 16, 10-day window) behind a
  scikit-learn-style `TorchClassifier`; CUDA when available, CPU fallback (`NEBULA_DEVICE=cpu`); every
  record stores the device; pooled panels are used to give DL a fair sample (~29.5k rows).
- **Problem**: per-instrument daily samples (~1.5-2k training rows) are too small for larger networks;
  the prompt forbids large architectures before baselines work.
- **Evidence**: benchmark (`scripts/bench_dl_device.py`, AAPL, 27 folds): MLP 10.2 s CPU vs 8.1 s GPU,
  GRU 21.4 s vs 8.8 s; peak VRAM 18-96 MB of 8 GB. CPU and GPU predictions differ by up to 0.10 in
  probability (different RNG streams / float order), so results are device-dependent.
- **Alternatives**: Transformers / TCNs (not justified: no model beat the base rate), LightGBM (XGBoost
  already covers boosting).
- **Trade-offs**: GPU benefit is modest at this size; exact cross-device reproducibility not possible.
- **Reversibility**: high.
