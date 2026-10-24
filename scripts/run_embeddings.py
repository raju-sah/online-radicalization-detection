#!/usr/bin/env python
"""Phase 6: encode Task A feature-window texts with MiniLM (CPU), then evaluate.

Texts are encoded in resumable chunks cached under ``<processed>/embeddings``
(gitignored). Only per-user mean-pooled vectors are used downstream; nothing
text-derived is written to ``reports``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from radpath.cohort import CohortParams  # noqa: E402
from radpath.config import ensure_data_dirs, load_config  # noqa: E402
from radpath.features import window_records  # noqa: E402
from radpath.ingest import load_contributions  # noqa: E402
from radpath.models import evaluate_dense  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main(argv: list[str] | None = None) -> None:
    """Encode (resumable), pool per user and evaluate."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="real")
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--max-seq-length", type=int, default=128)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    label = f"label_deep_a_{params.primary_variant.name}"
    cohort = pl.read_parquet(paths["processed"] / "cohort.parquet")
    users = cohort.filter(pl.col(label).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
    recs = (
        window_records(load_contributions(paths["interim"]), users, cfg["target_subreddits"], "a")
        .filter(pl.col("has_text"))
        .sort(["author_hash", "created_utc", "id"])
        .select("author_hash", "text")
    )
    cache = paths["processed"] / "embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    texts = recs["text"].to_list()
    n_chunks = (len(texts) + args.chunk - 1) // args.chunk
    model = None
    for i in range(n_chunks):
        path = cache / f"chunk_{i:05d}.npy"
        if path.exists():
            continue
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL, device="cpu")
            model.max_seq_length = args.max_seq_length
        vec = model.encode(texts[i * args.chunk:(i + 1) * args.chunk], batch_size=64, show_progress_bar=False)
        np.save(path.with_suffix(".tmp.npy"), vec.astype(np.float32))
        path.with_suffix(".tmp.npy").rename(path)
        print(f"chunk {i + 1}/{n_chunks}", flush=True)
    mat = np.concatenate([np.load(cache / f"chunk_{i:05d}.npy") for i in range(n_chunks)])
    dim = mat.shape[1]
    df = pl.DataFrame(mat, schema=[f"e{j}" for j in range(dim)]).with_columns(recs["author_hash"])
    pooled = df.group_by("author_hash").agg([pl.col(f"e{j}").mean() for j in range(dim)])
    pooled = users.select("author_hash").join(pooled, on="author_hash", how="left").fill_null(0.0)
    out = evaluate_dense(
        pooled, cohort, label, cfg, paths["reports"] / "results" / args.dataset,
        paths["reports"] / "test_log.csv", args.dataset, "minilm",
    )
    print(out.select("model", "n_test", "auroc", "auprc", "f1", "ece", "leakage_flag"))
    if out["leakage_flag"].any():
        raise SystemExit("STOP: AUROC above leakage threshold; investigate.")


if __name__ == "__main__":
    main()
