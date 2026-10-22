#!/usr/bin/env python
"""Train and evaluate baseline models on a built cohort (aggregate outputs only).

Example (synthetic data)::

    python scripts/run_models.py --config /path/synthetic.yaml --dataset synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import polars as pl  # noqa: E402

from radpath.cohort import CohortParams  # noqa: E402
from radpath.config import ensure_data_dirs, load_config  # noqa: E402
from radpath.features import build_task_features  # noqa: E402
from radpath.ingest import load_contributions  # noqa: E402
from radpath.models import evaluate_task  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Run all models for the requested tasks and label."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", required=True, help="Tag, e.g. synthetic or real.")
    ap.add_argument("--tasks", nargs="+", default=["a", "b"])
    ap.add_argument("--label-variant", default=None, help="Defaults to the primary variant.")
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    variant = args.label_variant or params.primary_variant.name
    cohort = pl.read_parquet(paths["processed"] / "cohort.parquet")
    contributions = load_contributions(paths["interim"])
    reports = paths["reports"]
    results = args.results_dir or (reports / "results" / args.dataset)
    synthetic = args.dataset.startswith("synthetic")

    for task in args.tasks:
        label = f"label_deep_{task}_{variant}"
        users = cohort.filter(pl.col(label).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
        if task == "b":
            users = users.filter(pl.col(f"task_b_included_f{cfg['followup_days']}"))
        feats, texts = build_task_features(
            contributions, users, task, cfg["target_subreddits"], cfg["gateway_subreddits"],
            first_n=params.first_n, feature_days=params.feature_days,
            pre_entry_days=params.pre_entry_days,
        )
        out = evaluate_task(
            feats, texts, cohort, task, label, cfg, results,
            reports / "test_log.csv", args.dataset, allow_repeat=synthetic,
        )
        print(out.select("task", "model", "n_test", "pos_rate_test", "auroc", "auprc", "f1", "ece", "leakage_flag"))
        if not synthetic and out["leakage_flag"].any():
            raise SystemExit(f"STOP: AUROC > threshold on task {task}; investigate leakage.")


if __name__ == "__main__":
    main()
