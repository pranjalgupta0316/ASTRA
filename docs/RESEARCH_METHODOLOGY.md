# Research methodology

1. **Register** a hypothesis (docs/RESEARCH_HYPOTHESES.md, docs/research/H-*.md) with its decision rule and
   commit it before computing results; later changes are dated amendments, never silent edits.
2. **Data**: only validated bars, pinned by a data lock (ADR-0009) for new studies; the lock is stored in
   every record. Features causal (truncation-invariance test); targets separate.
3. **Validation**: purged expanding walk-forward (purge = label horizon, embargo ≥ horizon). Scalers fit
   per training fold inside sklearn Pipelines. No hyperparameter tuning on the OOS period; `variants_tried` recorded for multiple-testing adjustments.
4. **Execution model**: decision at close t, fill at open t+1; costs per side = commission + half-spread
   + slippage + taxes (config/markets.yaml); fills capped at 5% of bar volume; long/flat by default.
5. **Inference**: Newey-West t on daily return differences vs buy-and-hold; stationary-bootstrap 95% CI
   for Sharpe; PSR/DSR available for multi-trial settings.
6. **Audit**: leakage canary (future-informed signal must look spectacular; lagged canary must not),
   transaction-cost sensitivity (0×, 1×, 2×, 4×), gross vs net reported; independent auditor agent
   re-runs checks before a result is reported. Engines are verified by property-based tests against an
   independent reference and a golden regression of headline numbers under the baseline lock.
7. **Record** every run, including rejected and inconclusive ones.

## Known methodological limitations
- Single-instrument time-series tests have low power; |t| < 1.96 is reported as inconclusive, not "no effect".
- Spreads for equities are assumed, not measured (no free historical quotes).
- Yahoo adjusted closes are restated retroactively; raw snapshots preserve what was used.
- Universe is hand-picked from today's well-known names → survivorship bias for any generalization
  (no point-in-time index membership; see KNOWN_ISSUES).
- Annualization: 252 (equities), 365 (crypto daily).
