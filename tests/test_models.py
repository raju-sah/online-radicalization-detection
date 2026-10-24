"""Metric helpers and the test-evaluation log guard (no real data)."""

from __future__ import annotations

import numpy as np
import pytest

from radpath.models import (
    best_f1_threshold,
    bootstrap_ci,
    ece_score,
    log_test_evaluation,
    metrics,
)


def test_ece_perfect_and_bad():
    y = np.array([0, 0, 1, 1])
    assert ece_score(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert ece_score(y, np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(1.0)


def test_threshold_and_bootstrap_are_reproducible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.4 + rng.random(300) * 0.6, 0, 1)
    thr = best_f1_threshold(y, p)
    assert 0 < thr < 1
    assert metrics(y, p, thr)["auroc"] > 0.7
    a = bootstrap_ci(y, p, thr, 50, seed=1)
    b = bootstrap_ci(y, p, thr, 50, seed=1)
    assert a == b
    assert a["auroc"][0] <= a["auroc"][1]


def test_test_log_refuses_repeat(tmp_path):
    row = {"dataset": "real", "task": "a", "label": "l", "model": "m", "config_hash": "h"}
    path = tmp_path / "test_log.csv"
    log_test_evaluation(path, row, allow_repeat=False)
    with pytest.raises(RuntimeError):
        log_test_evaluation(path, row, allow_repeat=False)
    log_test_evaluation(path, row, allow_repeat=True)


def test_majority_baseline_predicts_the_training_majority(tmp_path):
    """Validation F1 tuning must not turn the majority baseline all-positive."""
    import polars as pl
    from radpath.config import load_config
    from radpath.models import evaluate_task

    keys = [f"synthetic_{i}" for i in range(30)]
    cohort = pl.DataFrame({
        "author_hash": keys, "split": [s for s in ("train", "val", "test") for _ in range(10)],
        "label": [i % 10 == 0 for i in range(30)],
    })
    feats = pl.DataFrame({"author_hash": keys, "a_n": [float(i % 5 + 1) for i in range(30)]})
    texts = pl.DataFrame({"author_hash": keys, "doc": [""] * 30})
    cfg = load_config()
    cfg["models"]["bootstrap_n"] = 10
    result = evaluate_task(feats, texts, cohort, "a", "label", cfg, tmp_path,
                           tmp_path / "test_log.csv", "synthetic_majority", only=("majority",))
    assert result["model"].to_list() == ["majority"]
    assert result["f1"][0] == 0.0
