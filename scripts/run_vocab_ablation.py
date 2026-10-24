#!/usr/bin/env python
"""Phase 7: remove the most community-specific vocabulary and retrain text models.

Terms are ranked by weighted log-odds with an informative Dirichlet prior
(target feature-window text vs IncelTears text), computed on the TRAIN split
only. The ranking stays under ``<processed>`` (gitignored); reports contain
only aggregate performance numbers.
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
from radpath.vocab import count_terms, log_odds_informative_dirichlet, strip_terms, top_terms  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Rank terms on train, ablate top-k, retrain text models."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="real")
    ap.add_argument("--ks", type=int, nargs="+", default=[50, 200, 1000])
    ap.add_argument("--contrast", default="inceltears")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    label = f"label_deep_a_{params.primary_variant.name}"
    cohort = pl.read_parquet(paths["processed"] / "cohort.parquet")
    users = cohort.filter(pl.col(label).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
    contributions = load_contributions(paths["interim"])
    feats, texts = build_task_features(
        contributions, users, "a", cfg["target_subreddits"], cfg["gateway_subreddits"],
        first_n=params.first_n, feature_days=params.feature_days,
    )

    train_users = users.filter(pl.col("split") == "train")
    train_end = int(train_users["t0"].max())
    train_docs = texts.join(train_users.select("author_hash"), on="author_hash")["doc"].to_list()
    contrast = contributions.filter(
        (pl.col("subreddit").str.to_lowercase() == args.contrast)
        & pl.col("has_text")
        & (pl.col("created_utc") <= train_end)
    )["text"].to_list()
    print(f"train docs {len(train_docs)}, contrast texts {len(contrast)}", flush=True)
    ranked = log_odds_informative_dirichlet(count_terms(train_docs), count_terms(contrast))
    pl.DataFrame(ranked, schema=["term", "z"], orient="row").write_csv(
        paths["processed"] / "vocab_ranking.csv"  # local only; never committed
    )

    results = paths["reports"] / "results" / args.dataset
    outs = []
    for k in args.ks:
        removed = top_terms(ranked, k)
        ablated = texts.with_columns(
            pl.Series("doc", strip_terms(texts["doc"].to_list(), removed))
        )
        out = evaluate_task(
            feats, ablated, cohort, "a", label, cfg, results, paths["reports"] / "test_log.csv",
            args.dataset, only=("tfidf_lr", "combined_lr"), tag=f"vocab_top{k}",
        )
        outs.append(out.with_columns(pl.lit(k).alias("k_removed"), pl.lit(len(removed)).alias("n_terms_removed")))
    summary = pl.concat(outs, how="diagonal_relaxed").filter(pl.col("model") != "majority")
    summary.write_csv(results / "vocab_ablation_summary.csv")
    print(summary.select("k_removed", "model", "auroc", "auprc", "f1"))


if __name__ == "__main__":
    main()
