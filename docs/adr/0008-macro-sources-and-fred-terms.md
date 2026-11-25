# ADR-0008: Macro data from Yahoo market series and BLS; FRED manual-only

- **Decision**: automated macro features come from (a) Yahoo market series (^TNX, ^FVX, ^IRX, ^VIX,
  DX-Y.NYB, INR=X; same personal-research terms as our Yahoo prices) and (b) the BLS public API v1
  (CPI-U SA, unemployment rate; public-domain US government data). FRED is read only from CSVs a person
  downloaded by hand (`ingest_fred(csv_dir=...)`); automated download requires an explicit opt-in flag.
- **Problem**: the registered design (H-EXT) named FRED series. FRED's legal terms (checked 2026-11-25)
  prohibit automated access, storing/caching content, and using it to train ML systems - all three of
  which the pipeline would do.
- **Alternatives**: FRED with an API key (terms still apply); Fed Data Download Program / Treasury CSVs
  (not yet verified); dropping macro entirely.
- **Consequences**: no ALFRED vintages - BLS values are current revised figures (stated limitation);
  availability = max(fixed publication lag, actual BLS release date) to avoid leaks during release
  delays (2013 and 2025 shutdowns). H-MM-001 and H-XMKT-001 disclose the FRED → Yahoo/BLS deviation.
- **Reversibility**: high - `macro_features` merges FRED automatically if manual CSVs are provided.
