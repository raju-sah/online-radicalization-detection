#!/usr/bin/env python
"""Run aggregate-only EDA on the cohort.

Example::

    python scripts/run_eda.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radpath.config import ensure_data_dirs, load_config  # noqa: E402
from radpath.coverage import apply_coverage_overrides, load_coverage  # noqa: E402
from radpath.eda import run_eda  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
        help="Cohort Parquet (default: <processed>/cohort.parquet).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate EDA tables and figures."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    cohort_path = args.cohort or (paths["processed"] / "cohort.parquet")
    if not cohort_path.exists():
        raise SystemExit(f"Cohort not found: {cohort_path}. Run scripts/build_cohort.py first.")

    cohort = pl.read_parquet(cohort_path)
    coverage = apply_coverage_overrides(
        load_coverage(paths["interim"]),
        cfg.get("coverage_overrides"),
    )
    meta_path = cohort_path.with_name("cohort_meta.json")
    primary_variant = None
    if meta_path.exists():
        import json

        primary_variant = json.loads(
            meta_path.read_text(encoding="utf-8")
        ).get("primary_variant")

    result = run_eda(cohort, cfg, paths["reports"], coverage=coverage)
    print(f"EDA written to {paths['reports']} for {result['n_users']:,} users")
    for row in result["class_balance"]:
        if primary_variant and row["variant"] != primary_variant:
            continue
        rate = row["positive_rate"]
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        print(
            f"  Task {row['task'].upper()} [{row['variant']}]: "
            f"{row['eligible_users']:,} eligible, {row['positive']:,} deep ({rate_text})"
        )


if __name__ == "__main__":
    main()
