"""Tests for user-level temporal splitting."""

from __future__ import annotations

import polars as pl
import pytest

from radpath.split import SPLIT_ORDER, split_summary, temporal_split

DAY = 86_400
T0 = 1_600_000_000


def _frame(n_users: int, spacing_days: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "author_hash": [f"user_{index:04d}" for index in range(n_users)],
            "t0": [T0 + index * spacing_days * DAY for index in range(n_users)],
            "label_deep_a": [index % 5 == 0 for index in range(n_users)],
        }
    )


def test_split_sizes_and_temporal_ordering() -> None:
    frame = _frame(100)
    result = temporal_split(frame, {"train": 0.70, "val": 0.15, "test": 0.15})

    counts = result.group_by("split").len()
    sizes = dict(zip(counts["split"], counts["len"]))
    assert sizes == {"train": 70, "val": 15, "test": 15}

    bounds = {
        split: result.filter(pl.col("split") == split)["t0"]
        for split in SPLIT_ORDER
    }
    assert bounds["train"].max() <= bounds["val"].min()
    assert bounds["val"].max() <= bounds["test"].min()

    assert result["author_hash"].n_unique() == result.height
    assert set(result["split"].unique().to_list()) == set(SPLIT_ORDER)


def test_split_is_deterministic_and_not_row_random() -> None:
    frame = _frame(60)
    first = temporal_split(frame, {"train": 0.5, "val": 0.25, "test": 0.25})
    second = temporal_split(frame, {"train": 0.5, "val": 0.25, "test": 0.25})
    assert first.equals(second)

    # Shuffling input rows must not change the assignment.
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=7)
    third = temporal_split(shuffled, {"train": 0.5, "val": 0.25, "test": 0.25})
    assert (
        first.sort("author_hash")
        .select("author_hash", "split")
        .equals(third.sort("author_hash").select("author_hash", "split"))
    )


def test_split_handles_tied_entry_dates() -> None:
    frame = pl.DataFrame(
        {
            "author_hash": [f"user_{index}" for index in range(12)],
            "t0": [T0] * 12,
        }
    )
    result = temporal_split(frame, {"train": 0.5, "val": 0.25, "test": 0.25})
    assert result.height == 12
    assert set(result["split"].unique().to_list()) == set(SPLIT_ORDER)
    assert result["author_hash"].n_unique() == 12


def test_split_rejects_invalid_inputs() -> None:
    frame = _frame(10)
    with pytest.raises(ValueError):
        temporal_split(frame, {"train": 0.8, "val": 0.2, "test": 0.2})
    with pytest.raises(ValueError):
        temporal_split(frame, {"train": 0.0, "val": 0.5, "test": 0.5})
    with pytest.raises(ValueError):
        temporal_split(frame.head(2), {"train": 0.7, "val": 0.15, "test": 0.15})
    with pytest.raises(ValueError):
        temporal_split(frame.drop("t0"), {"train": 0.7, "val": 0.15, "test": 0.15})


def test_split_summary_reports_class_balance() -> None:
    result = temporal_split(_frame(100), {"train": 0.70, "val": 0.15, "test": 0.15})
    summary = split_summary(result, label_columns=("label_deep_a",))
    assert summary.height == 3
    assert summary["n_users"].sum() == 100
    total_positive = summary["label_deep_a_n_positive"].sum()
    assert total_positive == int(result["label_deep_a"].sum())
