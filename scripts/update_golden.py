"""Regenerate tests/golden/vertical_slice_<lock>.json. Run ONLY for a deliberate, explained change and commit
the new golden file together with that change: `uv run python scripts/update_golden.py`."""

from astra.tracking.golden import collect_headlines, golden_path, write_golden

if __name__ == "__main__":
    data = collect_headlines()
    print(write_golden(data), f"{len(data)} records")
    print("compare against git diff of", golden_path())
