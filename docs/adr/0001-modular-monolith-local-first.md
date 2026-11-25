# ADR-0001: Local-first modular monolith on Parquet + DuckDB

- **Decision**: one Python package with clear sub-packages; storage as zstd Parquet files in a layered
  directory tree; DuckDB for SQL access; no database server, message broker or cloud.
- **Problem**: need reproducible analytics for three markets on one 16 GB laptop with ₹0 budget.
- **Alternatives**: PostgreSQL/TimescaleDB (server to run, little benefit at this scale); ArcticDB
  (good, but adds a dependency for data volumes we do not have); microservices (complexity without need).
- **Reasoning**: daily data for hundreds of instruments is megabytes; tick data later is still
  single-machine scale with partitioned Parquet + DuckDB out-of-core queries.
- **Benefits**: zero ops, files are inspectable, reproducible by sha256, Windows-friendly.
- **Trade-offs**: no concurrent writers; no server-side access control.
- **Risks**: file sprawl → mitigated by manifests and fixed layout.
- **Reversibility**: high - Parquet can be bulk-loaded into any database later.
- **Consequences**: all writers go through `astra.data.storage`.
