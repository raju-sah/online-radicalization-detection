#!/usr/bin/env python
"""Gate 1: positives per split for Task A under every label variant (aggregate)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import polars as pl  # noqa: E402

from radpath.cohort import CohortParams  # noqa: E402
from radpath.config import ensure_data_dirs, load_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Write per-variant, per-split class balance; exit 1 if the primary fails the gate."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--min-train", type=int, default=300)
    ap.add_argument("--min-test", type=int, default=100)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    cohort = pl.read_parquet(paths["processed"] / "cohort.parquet")
    rows = []
    for v in params.all_variants():
        col = f"label_deep_a_{v.name}"
        for split in ("train", "val", "test"):
            d = cohort.filter((pl.col("split") == split) & pl.col(col).is_not_null())
            rows.append({"variant": v.name, "split": split, "n": d.height,
                         "positives": int(d[col].sum()),
                         "pos_rate": float(d[col].mean()) if d.height else float("nan")})
    out = pl.DataFrame(rows)
    res = paths["reports"] / "results"
    res.mkdir(parents=True, exist_ok=True)
    out.write_csv(res / "gate1_class_balance.csv")
    prim = out.filter(pl.col("variant") == params.primary_variant.name)
    print(prim)
    tr = prim.filter(pl.col("split") == "train")["positives"][0]
    te = prim.filter(pl.col("split") == "test")["positives"][0]
    ok = tr >= args.min_train and te >= args.min_test
    print(f"GATE 1 {'PASS' if ok else 'FAIL'}: train positives {tr} (>= {args.min_train}), test positives {te} (>= {args.min_test})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
