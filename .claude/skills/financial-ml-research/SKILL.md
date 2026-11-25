---
name: financial-ml-research
description: Use when building, training or evaluating ASTRA ML/DL models on financial data - classical models (linear, ridge/lasso, logistic, RF, gradient boosting, XGBoost, SVM), PyTorch MLP/LSTM/GRU/TCN/Transformer forecasters, multimodal fusion, calibration, uncertainty, feature importance, drift, or walk-forward evaluation on GPU/CPU.
---

# Financial ML Research (ASTRA)

## Purpose
Models that beat honest baselines out of sample, or a documented record that they did not.
Code: `src/astra/ml/` (classical, validation), `src/astra/dl/` (PyTorch).

## Activate when
Defining targets/labels, building feature matrices, training any model, comparing models, ablations,
choosing a validation scheme, GPU/batch-size decisions.

## Workflow
1. **Target** defined with horizon h and availability: label at t uses prices in (t, t+h].
2. **Baselines first**: zero / historical-mean / sign-of-last-return / logistic on 1-2 features.
3. **Validation**: `PurgedWalkForward` (expanding or rolling) with purge = h and embargo >= h.
   Scalers/imputers fit inside each training fold (sklearn `Pipeline`).
4. **Classical models** next; DL only when classical models show signal or the data is large enough
   (rule of thumb: >= 50k samples for sequence models; daily bars per instrument are ~5k - say so).
5. **Metrics**: task metrics (AUC, log-loss, Brier for classification; MAE/RMSE, QLIKE for vol) **and**
   a cost-aware backtest of the predictions via `backtesting-audit`.
6. **Ablation** for multimodal: drop each modality; report delta with fold-level dispersion.
7. **Track** with seed, data version, config hash, folds, metrics (experiment tracker).

## Standards
- Deterministic seeds (`astra.common.seed.set_seed`), `torch.backends.cudnn.benchmark=False` for repro runs.
- Device: `astra.dl.device.get_device()` - CUDA if available else CPU; RTX 4060 Laptop has 8 GB VRAM:
  start with hidden <= 128, batch <= 512 for daily data; log `torch.cuda.max_memory_allocated`.
- Class imbalance: report base rate; AUC alone hides it.
- Calibration: reliability curve + Brier; isotonic only fit on validation fold.
- Feature importance: permutation importance on validation data, not impurity importance.

## Failure modes
Random K-fold on time series; scaler fit on all data; tuning on the test fold; directional accuracy
near the base rate called "edge"; DL trained on 3k samples and compared to an untuned linear model.

## Verification / acceptance
Leakage canary (future-return feature must give near-perfect score and be flagged); fold dates printed;
baseline row present in every comparison table.

## Limitations
Small free datasets -> high variance estimates; report fold dispersion, not just means.

## Interacts with
`quantitative-research` (hypotheses), `backtesting-audit` (independent validation),
`financial-data-engineering` (feature availability), `research-paper-analysis` (architectures from papers).
