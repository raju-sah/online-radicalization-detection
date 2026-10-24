#!/usr/bin/env python
"""Discard free text from local interim Parquet files, batch by batch.

Raw dumps stay in data/raw. Non-text fields, row counts, and order are retained.
Replacement is atomic per file; rerunning after interruption is safe. This is
local data minimization, not an export or a data acquisition command.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from radpath.config import load_config, resolve_paths

TEXT_COLUMNS = {"title", "selftext", "body", "text"}


def quarantine_vocabulary(cfg: dict) -> None:
    """Keep a legacy word list, which may contain identifiers, in raw storage."""
    paths = resolve_paths(cfg)
    source = paths["processed"] / "vocab_ranking.csv"
    if source.exists():
        destination = paths["raw"] / "_legacy_text_artifacts" / "vocab_ranking.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError("Legacy vocabulary destination already exists; refusing to overwrite.")
        source.replace(destination)
        print("moved one legacy vocabulary artifact to excluded raw storage", flush=True)


def minimize(path: Path, batch_size: int) -> int:
    """Atomically replace free-text fields with empty strings, preserving metadata."""
    source = pq.ParquetFile(path)
    pending = path.with_suffix(".minimized.tmp")
    expected = source.metadata.num_rows
    rows = 0
    try:
        with pq.ParquetWriter(pending, source.schema_arrow, compression="zstd") as writer:
            for batch in source.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                for name in TEXT_COLUMNS.intersection(table.column_names):
                    index = table.schema.get_field_index(name)
                    field = table.schema.field(index)
                    table = table.set_column(index, field, pa.array([""] * len(table), type=field.type))
                writer.write_table(table)
                rows += len(table)
        if rows != expected or pq.ParquetFile(pending).metadata.num_rows != expected:
            raise RuntimeError("Row count changed during text minimization.")
        # Check written content without printing any source strings.
        check = pq.ParquetFile(pending)
        for batch in check.iter_batches(batch_size=batch_size, columns=sorted(TEXT_COLUMNS.intersection(check.schema_arrow.names))):
            if any(pc.any(pc.greater(pc.utf8_length(col), 0)).as_py() for col in batch.columns):
                raise RuntimeError("Free text remains after minimization.")
        pending.replace(path)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return rows


def main() -> None:
    """Minimize only configured interim files and log aggregate counts."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/release.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = resolve_paths(cfg)["interim"]
    files = sorted(p for kind in ("submissions", "comments") for p in (root / kind).glob("*.parquet"))
    total = 0
    for i, path in enumerate(files):
        total += minimize(path, cfg["ingest"]["batch_size"])
        print(f"minimized {i + 1}/{len(files)} files; rows={total}", flush=True)
    quarantine_vocabulary(cfg)


if __name__ == "__main__":
    main()
