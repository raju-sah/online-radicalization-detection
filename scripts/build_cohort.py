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
    resolve_data_end,
)
from radpath.config import ensure_data_dirs, load_config  # noqa: E402
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
    data_end = resolve_data_end(contributions, cfg.get("data_end_utc"))
    cohort = build_cohort(
        contributions=contributions,
        params=params,
        target_subreddits=cfg["target_subreddits"],
        gateway_subreddits=cfg["gateway_subreddits"],
        data_end_utc=data_end,
        split_fractions=cfg["split"],
    )

    out_path = args.out or (paths["processed"] / "cohort.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(out_path)

    eligible_a = int(cohort["eligible_a"].sum())
    eligible_b = int(cohort["eligible_b"].sum())
    positive_a = int(cohort["label_deep_a"].fill_null(False).sum())
    positive_b = int(cohort["label_deep_b"].fill_null(False).sum())
    meta = {
        "radpath_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_end_utc": data_end,
        "n_contributions": contributions.height,
        "n_users": cohort.height,
        "eligible_a": eligible_a,
        "eligible_b": eligible_b,
        "positive_a": positive_a,
        "positive_b": positive_b,
        "params": {
            "first_n": params.first_n,
            "feature_days": params.feature_days,
            "pre_entry_days": params.pre_entry_days,
            "followup_days": params.followup_days,
            "deep_min_contributions": params.deep_min_contributions,
            "deep_min_active_weeks": params.deep_min_active_weeks,
        },
        "target_subreddits": [str(value).lower() for value in cfg["target_subreddits"]],
        "gateway_subreddits": [str(value).lower() for value in cfg["gateway_subreddits"]],
    }
    meta_path = out_path.with_name("cohort_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"Cohort: {cohort.height:,} users -> {out_path}")
    print(
        f"Task A eligible: {eligible_a:,} (deep: {positive_a:,}); "
        f"Task B eligible: {eligible_b:,} (deep: {positive_b:,})"
    )
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
