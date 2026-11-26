# ADR-0009: Data locks - immutable content-addressed snapshots of validated data

- **Decision**: every file under `data/validated` can be pinned by a named, immutable *data lock*.
  `astra data-lock create NAME` copies each file once into `data/snapshots/objects/<sha[:2]>/<sha256><ext>`
  (deduplicated by content, git-ignored) and writes `config/data_locks/NAME.json` (relative path ->
  sha256 + size; tracked in git). A lock is activated by `astra --lock NAME ...`, the `ASTRA_DATA_LOCK`
  environment variable, or `astra.data.lock.use_lock(NAME)`. While active, every research read of
  validated data resolves to the snapshot; a path the lock does not contain raises `LockError` (strict,
  no silent fallback to live data). Every experiment record stores `data_lock = {name, fingerprint}`
  (or `{name: null}` for live data). Ingestion always reads and writes live data.
- **Problem**: studies read the latest validated files. A re-download (Yahoo restating adjusted prices,
  a new Binance day) silently changed inputs of already-recorded results; records stored file hashes
  but nothing could re-create those exact files.
- **Alternatives**: DVC / git-annex (extra tooling and a remote, over-engineered for one machine);
  date-filtering live data (does not survive restatements); committing data to git (size, licences).
- **Consequences**: ~0.5 GB of snapshots for `v0.3.0-baseline` (264 files); later locks only add changed
  files. Verified: `astra --lock v0.3.0-baseline research AAPL` reproduces the published d3653bb records
  with 0 metric deviations; the golden regression test (`tests/test_golden_regression.py`) re-checks
  AAPL/RELIANCE/BTCUSDT under this lock on every run. `astra data-lock verify NAME` re-hashes snapshots.
  Records written before 2026-11-26 have an empty `data_lock` (pre-dates the feature).
- **Reversibility**: high - with no lock active, behaviour is exactly the pre-lock behaviour.
