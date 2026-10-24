#!/usr/bin/env python
"""Phase 8: all label variants, near-coverage-start exclusion, summary + forest plot."""

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

MODELS = ("activity_lr", "behaviour_lgbm", "combined_lr")


def forest_plot(summary: pl.DataFrame, path: Path) -> None:
    """Forest plot of AUROC and AUPRC (95% CI) for the behavioural LightGBM."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = summary.filter(pl.col("model") == "behaviour_lgbm").sort("analysis", descending=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 0.28 * d.height + 1.5), sharey=True)
    y = list(range(d.height))
    for ax, m in zip(axes, ("auroc", "auprc")):
        v, lo, hi = d[m].to_numpy(), d[f"{m}_lo"].to_numpy(), d[f"{m}_hi"].to_numpy()
        ax.errorbar(v, y, xerr=[v - lo, hi - v], fmt="o", ms=3, lw=1)
        ax.set_xlabel(m.upper())
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(d["analysis"].to_list(), fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the sensitivity grid on Task A and write one summary table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="real")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    cohort = pl.read_parquet(paths["processed"] / "cohort.parquet")
    contributions = load_contributions(paths["interim"])
    labels = [f"label_deep_a_{v.name}" for v in params.all_variants()]
    in_split = pl.col("split").is_in(["train", "val", "test"])
    any_label = pl.any_horizontal([pl.col(c).is_not_null() for c in labels])
    users = cohort.filter(in_split & any_label)
    feats, texts = build_task_features(
        contributions, users, "a", cfg["target_subreddits"], cfg["gateway_subreddits"],
        first_n=params.first_n, feature_days=params.feature_days,
    )
    results = paths["reports"] / "results" / args.dataset
    primary = f"label_deep_a_{params.primary_variant.name}"
    outs = []
    for col in labels:
        if col == primary:
            continue  # primary already evaluated in Phase 5
        o = evaluate_task(feats, texts, cohort, "a", col, cfg, results, paths["reports"] / "test_log.csv",
                          args.dataset, only=MODELS, tag="robust")
        outs.append(o.with_columns(pl.lit(col.replace("label_deep_a_", "")).alias("analysis")))
    near = cohort.filter(~pl.col("near_coverage_start"))
    o = evaluate_task(feats, texts, near, "a", primary, cfg, results, paths["reports"] / "test_log.csv",
                      args.dataset, only=MODELS, tag="excl_near_start")
    outs.append(o.with_columns(pl.lit("primary_excl_near_coverage_start").alias("analysis")))
    base = pl.read_csv(results / f"metrics_{args.dataset}_a_{primary}.csv").filter(
        pl.col("model").is_in(("majority", *MODELS))
    ).with_columns(pl.lit("primary").alias("analysis"))
    summary = pl.concat([base, *outs], how="diagonal_relaxed").filter(pl.col("model") != "majority")
    keep = ["analysis", "model", "n_test", "pos_test", "pos_rate_train", "pos_rate_test",
            "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo", "auprc_hi", "f1", "f1_lo", "f1_hi", "ece"]
    summary = summary.select(keep)
    summary.write_csv(results / "robustness_summary.csv")
    forest_plot(summary, results / "robustness_forest.png")
    print(summary.filter(pl.col("model") == "behaviour_lgbm").select("analysis", "n_test", "pos_test", "auroc", "auprc"))


if __name__ == "__main__":
    main()
