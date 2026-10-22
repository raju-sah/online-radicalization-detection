#!/usr/bin/env python
"""Build the user-level cohort, labels and temporal split.

Reads cleaned interim Parquet, writes ``data/processed/cohort.parquet`` and a
small aggregate metadata JSON. Example::

    python scripts/build_cohort.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radpath import __version__  # noqa: E402
from radpath.cohort import (  # noqa: E402
    CohortParams,
    build_cohort,
)
from radpath.config import ensure_data_dirs, load_config  # noqa: E402
from radpath.coverage import (  # noqa: E402
    apply_coverage_overrides,
    coverage_from_contributions,
    load_coverage,
)
from radpath.ingest import load_contributions  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Cohort Parquet path (default: <processed>/cohort.parquet).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Build and write the cohort."""
    args = parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    contributions = load_contributions(paths["interim"])
    if contributions.height == 0:
        raise SystemExit(
            f"No interim records found under {paths['interim']}. Run scripts/ingest.py first."
        )

    params = CohortParams.from_config(cfg)
    coverage = load_coverage(paths["interim"])
    if coverage.height == 0:
        coverage = coverage_from_contributions(
            contributions, cfg["target_subreddits"]
        )
    coverage = apply_coverage_overrides(coverage, cfg.get("coverage_overrides"))
    cohort = build_cohort(
        contributions=contributions,
        params=params,
        target_subreddits=cfg["target_subreddits"],
        gateway_subreddits=cfg["gateway_subreddits"],
        coverage=coverage,
        split_fractions=cfg["split"],
    )

    out_path = args.out or (paths["processed"] / "cohort.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(out_path)

    eligible_a = int(cohort["eligible_a"].sum())
    eligible_b = int(cohort["eligible_b"].sum())
    included_b = int(cohort["task_b_included"].sum())
    positive_a = int(cohort["label_deep_a"].fill_null(False).sum())
    positive_b = int(cohort["label_deep_b"].fill_null(False).sum())
    embargoed = int(cohort["split"].is_null().sum()) if "split" in cohort.columns else 0
    variant_meta = [
        {
            "name": variant.name,
            "min_contributions": variant.min_contributions,
            "min_active_weeks": variant.min_active_weeks,
            "followup_days": variant.followup_days,
            "kinds": list(variant.kinds),
            "primary": variant.name == params.primary_variant.name,
            "positive_a": int(
                cohort[f"label_deep_a_{variant.name}"].fill_null(False).sum()
            ),
            "positive_b": int(
                cohort[f"label_deep_b_{variant.name}"].fill_null(False).sum()
            ),
        }
        for variant in params.all_variants()
    ]
    meta = {
        "radpath_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_contributions": contributions.height,
        "n_users": cohort.height,
        "n_embargoed": embargoed,
        "eligible_a": eligible_a,
        "eligible_b": eligible_b,
        "task_b_included": included_b,
        "positive_a": positive_a,
        "positive_b": positive_b,
        "primary_variant": params.primary_variant.name,
        "label_variants": variant_meta,
        "params": {
            "first_n": params.first_n,
            "feature_days": params.feature_days,
            "pre_entry_days": params.pre_entry_days,
            "followup_days": params.followup_days,
            "deep_min_contributions": params.deep_min_contributions,
            "deep_min_active_weeks": params.deep_min_active_weeks,
            "near_coverage_start_days": params.near_coverage_start_days,
        },
        "target_subreddits": [str(value).lower() for value in cfg["target_subreddits"]],
        "gateway_subreddits": [str(value).lower() for value in cfg["gateway_subreddits"]],
        "coverage": [
            {
                "subreddit": row["subreddit"],
                "start_utc": row["start_utc"],
                "end_utc": row["end_utc"],
                "n_records": row["n_records"],
                "overridden": row["overridden"],
            }
            for row in coverage.iter_rows(named=True)
        ],
    }
    meta_path = out_path.with_name("cohort_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Cohort: {cohort.height:,} users -> {out_path}")
    print(
        f"Task A eligible: {eligible_a:,} (deep: {positive_a:,}); "
        f"Task B eligible: {eligible_b:,} (with gateway: {included_b:,}, "
        f"deep: {positive_b:,})"
    )
    print(f"Embargoed (null split): {embargoed:,}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
