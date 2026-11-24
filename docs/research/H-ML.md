# ML / DL hypotheses (registered 2026-11-24, before any of these models were run)

Common design (fixed a priori, no tuning):
- Target `target_up` = 1[log(O_{t+2}/O_{t+1}) > 0]; features = the 13 causal features of H-XM-002.
- Purged expanding walk-forward: min_train = 2 years of sessions, test = 1 quarter, purge = 2, embargo = 5.
- Baseline = training-fold base rate (ex-ante). Skill test = Diebold-Mariano-style Newey-West t on
  per-row log-loss differences (baseline − model). Multiple testing: Benjamini-Hochberg (FDR 5%) on
  one-sided p-values across every model × instrument cell; the cell count is the variant count.
- Placeholder rows excluded (as in the vertical slice).

Models (fixed hyperparameters):
| id | model | key settings |
|---|---|---|
| logit | L2 logistic regression | C = 0.1, standardized per fold |
| rf | random forest | 300 trees, max_depth 4, min_samples_leaf 50 |
| xgb | XGBoost | 300 trees, depth 3, lr 0.03, subsample 0.8, colsample 0.8 |
| mlp | PyTorch MLP | 13→32→16→1, dropout 0.2, AdamW 1e-3, wd 1e-3, ≤ 200 epochs, early stop (patience 15) on last 15% of the training window |
| gru | PyTorch GRU | sequences of the last 10 days of the 13 features, hidden 16, 1 layer, same optimizer/early stopping |

## H-ML-001 Per-instrument skill
H1: at least one model × instrument cell (5 × 13 = 65) has OOS log loss significantly below its
ex-ante baseline after BH correction. H0: none does.
Rationale: liquid instruments, daily horizon, technical features → strong prior for the null.
Decision: supported if ≥ 1 BH-significant improvement; otherwise rejected (with the count of nominally
significant cells reported).

## H-ML-002 Pooled panel with deep learning
Intuition: per-instrument samples (~1.5-2k training rows) are too small for DL; pooling the 13
instruments (~28k rows) may let MLP/GRU learn shared structure.
Design: one pooled model per model type, date-based purged walk-forward (all rows of a date share a
fold). H1: pooled GRU and pooled MLP each beat (a) the ex-ante baseline and (b) pooled logistic on OOS
log loss (DM t > 1.96). Decision per model: supported if both (a) and (b) hold; rejected if model is
significantly worse than baseline; else inconclusive.

Trading evaluation is performed only for models that pass the statistical gate (prevents choosing the
best of 65 backtests).

## Amendment A2 (2026-11-24, after the first run - disclosed as post hoc; verdicts unchanged)
Independent audit of the first run found:
1. H-ML-002 per-date DM statistic weighted each date equally, over-weighting crypto-only weekend dates
   (MLP t flipped from -0.61 row-wise to +0.20). Now: row-weighted mean with date-clustered Newey-West
   inference (per-date sums / mean rows per date).
2. Panel folds were 504/63 *union dates* (≈1.4 y / 2 months, because crypto makes every calendar day a
   date), not the registered 2 y / 1 quarter. Now 730/91/7 calendar days = the registered design.
3. MLP/GRU early stopping had no purge between the fit rows and the validation tail; now a 2-row gap
   (= label horizon).
First-run records are kept in experiments/ (superseded, not deleted). Results below are from the rerun.
