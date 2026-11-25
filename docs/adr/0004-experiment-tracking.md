# ADR-0004: File-based experiment tracker first; MLflow optional

- **Decision**: `astra.tracking.Tracker` writes `experiments/<id>/record.json` + Parquet artifacts and
  an append-only `index.jsonl`. `mlflow-skinny` is an optional extra, not required.
- **Problem**: every experiment (including failures) must be recorded with hypothesis, data hashes,
  code version, config, seed, metrics and conclusion.
- **Alternatives**: full MLflow (server + UI, heavier deps, its schema has no first-class hypothesis /
  conclusion / variants-tried fields); DVC (data versioning focus); W&B (cloud, account).
- **Reasoning**: our record schema is research-specific; files are diffable and readable by the dashboard.
- **Trade-offs**: no built-in comparison UI (dashboard page covers it); no concurrent-writer locking.
- **Reversibility**: high - records can be exported to MLflow runs later.
- **Update 2026-11-24**: the optional `mlflow-skinny` extra was removed because nothing imported it.
