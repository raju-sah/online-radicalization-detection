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


def test_users_sharing_an_entry_date_share_a_split() -> None:
    frame = pl.DataFrame(
        {
            "author_hash": [f"user_{index}" for index in range(30)],
            "t0": [T0] * 10 + [T0 + 10 * DAY] * 10 + [T0 + 20 * DAY] * 10,
        }
    )
    result = temporal_split(frame, {"train": 0.5, "val": 0.25, "test": 0.25})
    assert result.height == 30
    assert set(result["split"].unique().to_list()) == set(SPLIT_ORDER)
    assert result["author_hash"].n_unique() == 30

    date_split = (
        result.with_columns(
            pl.from_epoch(pl.col("t0"), time_unit="s").dt.date().alias("entry_date")
        )
        .group_by("entry_date")
        .agg(pl.col("split").n_unique().alias("n_splits"))
    )
    assert date_split["n_splits"].to_list() == [1, 1, 1]


def test_same_date_cohort_raises_instead_of_splitting_ties() -> None:
    frame = pl.DataFrame(
        {
            "author_hash": [f"user_{index}" for index in range(12)],
            "t0": [T0] * 12,
        }
    )
    with pytest.raises(ValueError, match="distinct entry dates"):
        temporal_split(frame, {"train": 0.5, "val": 0.25, "test": 0.25})


def test_embargo_drops_windows_crossing_split_boundaries() -> None:
    frame = pl.DataFrame(
        {
            "author_hash": [f"user_{index:03d}" for index in range(120)],
            "t0": [T0 + index * DAY for index in range(120)],
            "label_window_end": [T0 + index * DAY + 5 * DAY for index in range(120)],
        }
    )
    result = temporal_split(
        frame,
        {"train": 0.70, "val": 0.15, "test": 0.15},
        label_end_col="label_window_end",
    )
    assigned = result.filter(pl.col("split").is_not_null())
    assert set(assigned["split"].unique().to_list()) == set(SPLIT_ORDER)

    val_start = int(
        assigned.filter(pl.col("split") == "val")["t0"].min()
    )
    val_start_midnight = val_start - (val_start % DAY)
    test_start = int(
        assigned.filter(pl.col("split") == "test")["t0"].min()
    )
    test_start_midnight = test_start - (test_start % DAY)

    train = result.filter(pl.col("split") == "train")
    val = result.filter(pl.col("split") == "val")
    assert train.height > 0 and val.height > 0
    assert train["label_window_end"].max() < val_start_midnight
    assert val["label_window_end"].max() < test_start_midnight
    assert result["split"].is_null().sum() > 0
    dropped = result.filter(pl.col("split").is_null())
    assert dropped.height > 0
    assert set(dropped["author_hash"]).isdisjoint(
        set(assigned["author_hash"])
    )


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


def test_eligible_only_split_with_short_embargo(params, make_contributions):
    from dataclasses import replace

    from radpath.cohort import build_cohort

    day = 86_400
    rows = []
    for i in range(60):
        t0 = 1_600_000_000 + i * 20 * day
        rows.append({"author_hash": f"u{i}", "subreddit": "incels", "created_utc": t0})
        rows.append({"author_hash": f"u{i}", "subreddit": "incels", "created_utc": t0 + 2 * day})
    p = replace(params, split_eligible_only=True, embargo_followup_days=90)
    cohort = build_cohort(
        make_contributions(rows), p, ["incels"], ["lonely"],
        split_fractions={"train": 0.5, "val": 0.25, "test": 0.25},
        data_end_utc=1_600_000_000 + 40 * 20 * day,
    )
    labelled = cohort.filter(pl.col("split").is_not_null())
    assert set(labelled["split"].unique()) == {"train", "val", "test"}
    assert labelled["eligible_a"].all()
    # temporal ordering of splits
    hi = labelled.group_by("split").agg(pl.col("t0").max().alias("hi"), pl.col("t0").min().alias("lo"))
    d = {r["split"]: r for r in hi.iter_rows(named=True)}
    assert d["train"]["hi"] < d["val"]["lo"] <= d["val"]["hi"] < d["test"]["lo"]
