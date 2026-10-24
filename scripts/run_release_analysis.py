#!/usr/bin/env python
"""Run the frozen, metadata-only exploratory release analysis (aggregate outputs).

The input projection never reads free text, scores, or removal flags. Existing
results are reused only when their input/config/code provenance is unchanged.
No per-user features, predictions, or explanations are written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import polars as pl
import numpy as np
from sklearn.inspection import permutation_importance
from radpath.config import load_config, ensure_data_dirs
from radpath.features import build_task_features
from radpath.models import evaluate_task, _lgbm

INPUT_COLUMNS = ["author_hash", "id", "kind", "subreddit", "created_utc", "link_id", "parent_id"]


def digest(path: Path) -> str:
    """Hash a local file without loading the entire file into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    """Evaluate the predeclared variants once and preserve the evaluation log."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/release.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    plan = cfg["release_analysis"]
    dataset = plan["dataset"]
    results = paths["reports"] / "results" / dataset
    results.mkdir(parents=True, exist_ok=True)
    cohort_path = paths["processed"] / "cohort.parquet"
    files = sorted(p for kind in ("submissions", "comments")
                   for p in (paths["interim"] / kind).glob("*.parquet"))
    provenance = {"config": cfg, "cohort_sha256": digest(cohort_path),
                  "input_sha256": {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in files},
                  "code_sha256": {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in
                                  [Path(__file__), PROJECT_ROOT / "src/radpath/features.py",
                                   PROJECT_ROOT / "src/radpath/models.py"]}}
    manifest = results / "provenance.json"
    if manifest.exists() and json.loads(manifest.read_text()) != provenance:
        raise SystemExit("STOP: release inputs changed; preserve existing results and declare a new analysis.")
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    cohort = pl.read_parquet(cohort_path)
    users = cohort.filter(pl.col("split").is_in(["train", "val", "test"]))
    contributions = pl.concat([pl.read_parquet(p, columns=INPUT_COLUMNS) for p in files])
    contributions = contributions.with_columns(pl.lit(0).alias("score"),
                                               pl.lit(False).alias("has_text"))
    feats, texts = build_task_features(
        contributions, users, "a", cfg["target_subreddits"], cfg["gateway_subreddits"],
        first_n=cfg["feature_window"]["first_n"], feature_days=cfg["feature_window"]["days"],
        with_text_stats=False, creation_time_only=True,
    )
    del contributions
    summary = []
    for analysis in plan["analyses"]:
        label = "label_deep_a_" + analysis["variant"]
        sample = cohort.filter(~pl.col("near_coverage_start")) if analysis.get("exclude_near_start") else cohort
        tag = analysis["name"]
        done = results / f"metrics_{dataset}_a_{label}_{tag}.csv"
        if done.exists():
            out = pl.read_csv(done)
        else:
            out = evaluate_task(feats, texts, sample, "a", label, cfg, results,
                                paths["reports"] / "test_log.csv", dataset,
                                only=tuple(plan["models"]), tag=tag)
        if out["leakage_flag"].any():
            raise SystemExit("STOP: test AUROC exceeds the configured leakage trigger.")
        summary.append(out.with_columns(pl.lit(tag).alias("analysis")))
        print(f"completed {tag}: {out.height} models, test n={out['n_test'][0]}", flush=True)
    pl.concat(summary, how="diagonal_relaxed").write_csv(results / "summary.csv")
    # Dates and class counts only; never export membership or identifiers.
    primary = "label_deep_a_" + plan["analyses"][0]["variant"]
    balances = cohort.filter(pl.col(primary).is_not_null() & pl.col("split").is_not_null()).group_by("split").agg(
        pl.len().alias("n"), pl.col(primary).sum().alias("positives"),
        pl.col("t0").min().alias("entry_start_utc"), pl.col("t0").max().alias("entry_end_utc"),
    )
    balances.write_csv(results / "class_balance.csv")
    # Interpret the selected primary model on validation, with no new test scoring.
    primary_result = summary[0].filter(pl.col("model") == "behaviour_lgbm").row(0, named=True)
    frame = cohort.filter(pl.col(primary).is_not_null()).join(feats, on="author_hash")
    columns = [c for c in feats.columns if c != "author_hash"]
    train = frame.filter(pl.col("split") == "train")
    val = frame.filter(pl.col("split") == "val")
    prior = train[primary].mean()
    mc = cfg["models"]
    model = _lgbm(json.loads(primary_result["config"]), mc["lgbm_common"],
                  int(mc["seed"]), (1 - prior) / prior)
    model.fit(train.select(columns).to_numpy(), train[primary].to_numpy().astype(int))
    importance = permutation_importance(model, val.select(columns).to_numpy(),
        val[primary].to_numpy().astype(int), scoring="average_precision",
        n_repeats=plan["permutation_repeats"], random_state=int(mc["seed"]))
    pl.DataFrame({"feature": columns, "auprc_drop_mean": importance.importances_mean,
                  "auprc_drop_sd": importance.importances_std}).sort(
        "auprc_drop_mean", descending=True).write_csv(results / "permutation_importance_val.csv")
    print("release analysis complete", flush=True)


if __name__ == "__main__":
    main()
