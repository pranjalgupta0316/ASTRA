# Research methodology

1. **Register** a hypothesis (docs/RESEARCH_HYPOTHESES.md) with decision rule before computing results.
2. **Data**: only validated bars; features causal (truncation-invariance test); targets separate.
3. **Validation**: purged expanding walk-forward (purge = label horizon, embargo ≥ horizon). Scalers fit
   per training fold inside sklearn Pipelines. No hyperparameter tuning in the vertical slice → no
   selection on the OOS period; `variants_tried` recorded for multiple-testing adjustments.
4. **Execution model**: decision at close t, fill at open t+1; costs per side = commission + half-spread
   + slippage + taxes (config/markets.yaml); fills capped at 5% of bar volume; long/flat by default.
5. **Inference**: Newey-West t on daily return differences vs buy-and-hold; stationary-bootstrap 95% CI
   for Sharpe; PSR/DSR available for multi-trial settings.
6. **Audit**: leakage canary (future-informed signal must look spectacular; lagged canary must not),
   transaction-cost sensitivity (0×, 1×, 2×, 4×), gross vs net reported.
7. **Record** every run, including rejected and inconclusive ones.

## Known methodological limitations (vertical slice)
- Single-instrument time-series tests have low power; |t| < 1.96 is reported as inconclusive, not "no effect".
- Spreads for equities are assumed, not measured (no free historical quotes).
- Yahoo adjusted closes are restated retroactively; raw snapshots preserve what was used.
- Universe is hand-picked from today's well-known names → survivorship bias for any generalization.
- Annualization: 252 (equities), 365 (crypto daily).
