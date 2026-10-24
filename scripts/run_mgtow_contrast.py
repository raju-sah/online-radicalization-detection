#!/usr/bin/env python
"""Phase 8: repeat the Task A pipeline with MGTOW as the target (comments only).

Uses ``configs/mgtow.yaml``. Outputs go to ``reports/mgtow`` and
``data/processed_mgtow``; results are reported separately from the primary
analysis and never feed back into it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import polars as pl  # noqa: E402

from radpath.cohort import CohortParams, build_cohort  # noqa: E402
from radpath.config import ensure_data_dirs, load_config  # noqa: E402
from radpath.features import build_task_features  # noqa: E402
from radpath.ingest import load_contributions  # noqa: E402
from radpath.models import evaluate_task  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Build the MGTOW cohort, check class balance and evaluate the models."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mgtow.yaml")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    contributions = load_contributions(paths["interim"]).filter(
        (pl.col("subreddit").str.to_lowercase() == "mgtow") & (pl.col("kind") == "comment")
    )
    cohort = build_cohort(
        contributions, params, cfg["target_subreddits"], cfg["gateway_subreddits"],
        coverage_overrides=cfg["coverage_overrides"],
    )
    # D2: MGTOW data end in 2017-12, so quantile splits over all entrants would
    # leave val/test with no observable labels. Split among label-eligible users.
    from radpath.split import temporal_split

    cohort = temporal_split(
        cohort.filter(pl.col("eligible_a")), cfg["split"], label_end_col="label_window_end"
    )
    cohort.write_parquet(paths["processed"] / "cohort.parquet")
    label = f"label_deep_a_{params.primary_variant.name}"
    bal = (
        cohort.filter(pl.col(label).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
        .group_by("split").agg(pl.len().alias("n"), pl.col(label).sum().alias("positives")).sort("split")
    )
    print(bal)
    out_dir = paths["reports"] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    bal.write_csv(out_dir / "class_balance_mgtow.csv")
    users = cohort.filter(pl.col(label).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
    feats, texts = build_task_features(
        contributions, users, "a", cfg["target_subreddits"], cfg["gateway_subreddits"],
        first_n=params.first_n, feature_days=params.feature_days,
    )
    out = evaluate_task(feats, texts, cohort, "a", label, cfg, out_dir, paths["reports"] / "test_log.csv", "real_mgtow")
    print(out.select("model", "n_test", "pos_test", "auroc", "auprc", "f1", "ece", "leakage_flag"))


if __name__ == "__main__":
    main()
