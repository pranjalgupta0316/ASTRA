# Next steps (ranked)

1. **Point-in-time universes** - the main limitation of every cross-sectional result. Candidate free
   sources: historical index-change announcements (S&P, NSE press releases) reconstructed by hand.
2. **Maker-side microstructure study** - taker strategies lose ~20 bps/trade; a passive (limit-order)
   strategy with queue-position modelling is the only way the 0.91 AUC signal could matter. Needs
   longer captures across several days and a fill model (L2 only, so queue position is approximate).
3. **Lower-cost venues / fee tiers** - H-ML-004 breaks even at ≈ 5 bps round trip; re-evaluate with a
   documented maker-fee assumption (paper trading only).
4. **Push to a private GitHub repository** (developer decision) so CI runs; consider replacing the
   personal commit email first.
5. **News / text features** from permitted RSS headlines with a small local model (Ollama is installed)
   - only after the price/fundamental results are re-checked on point-in-time universes.
6. **First-release macro vintages** (e.g., BLS archived releases) to remove the revised-data limitation.
7. Re-run the heavy DL (`run-all --heavy-dl`, ~3 h) from the final commit for complete provenance.
