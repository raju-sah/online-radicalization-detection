"""Leakage tests: features stop at the cutoff, labels start after it.

These tests reconstruct the feature and label sets independently from the
raw contribution table and compare them with the cohort output.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from radpath.cohort import CohortParams, build_cohort

DAY = 86_400
T0 = 1_600_000_000
TARGETS = ["incels", "braincels"]
GATEWAYS = ["lonely", "foreveralone"]


def _build(frame: pl.DataFrame, params: CohortParams, data_end: int) -> pl.DataFrame:
    return build_cohort(
        frame,
        params,
        target_subreddits=TARGETS,
        gateway_subreddits=GATEWAYS,
        data_end_utc=data_end,
    )


def _leaky_rows() -> tuple[list[dict[str, Any]], list[int]]:
    """One user with clear pre-window, post-window and far-future records."""
    rows: list[dict[str, Any]] = []
    for index in range(5):
        rows.append({"author_hash": "leaky", "subreddit": "incels", "created_utc": T0 + index})
    label_offsets = [T0 + 5 * DAY + index * 3 * DAY for index in range(25)]
    for timestamp in label_offsets:
        rows.append({"author_hash": "leaky", "subreddit": "incels", "created_utc": timestamp})
    # Far future, well after the label window.
    for index in range(100):
        rows.append(
            {
                "author_hash": "leaky",
                "subreddit": "incels",
                "created_utc": T0 + 4 + 90 * DAY + 1 + index,
            }
        )
    # Gateway records: 10 pre-entry, one at t0-1, one at t0, five after cutoff.
    for index in range(10):
        rows.append(
            {
                "author_hash": "leaky",
                "subreddit": "lonely",
                "created_utc": T0 - (10 + index) * DAY,
            }
        )
    rows.append({"author_hash": "leaky", "subreddit": "ForeverAlone", "created_utc": T0 - 1})
    rows.append({"author_hash": "leaky", "subreddit": "ForeverAlone", "created_utc": T0})
    for index in range(5):
        rows.append(
            {
                "author_hash": "leaky",
                "subreddit": "lonely",
                "created_utc": T0 + 4 + (10 + index) * DAY,
            }
        )
    return rows, label_offsets


def test_features_only_use_records_at_or_before_cutoff(
    make_contributions, params: CohortParams
) -> None:
    rows, _ = _leaky_rows()
    frame = make_contributions(rows)
    cohort = _build(frame, params, data_end=T0 + 1_000 * DAY)
    row = cohort.filter(pl.col("author_hash") == "leaky").row(0, named=True)

    cutoff_a = row["cutoff_a"]
    cutoff_b = row["cutoff_b"]
    user = frame.filter(pl.col("author_hash") == "leaky")

    feature_a = user.filter(
        pl.col("subreddit").is_in(TARGETS) & (pl.col("created_utc") <= cutoff_a)
    )
    assert feature_a.height == row["n_target_window_a"] == 5
    assert feature_a["created_utc"].max() <= cutoff_a

    feature_b = user.filter(
        pl.col("subreddit").is_in(GATEWAYS)
        & (pl.col("created_utc") >= row["t0"] - 180 * DAY)
        & (pl.col("created_utc") < row["t0"])
    )
    assert feature_b.height == row["n_gateway_pre_b"] == 11
    assert feature_b["created_utc"].max() <= cutoff_b
    # The record at exactly t0 is a label-window record, not a feature.
    assert not feature_b.filter(pl.col("created_utc") == row["t0"]).height


def test_labels_only_use_post_window_target_records(
    make_contributions, params: CohortParams
) -> None:
    rows, label_offsets = _leaky_rows()
    frame = make_contributions(rows)
    cohort = _build(frame, params, data_end=T0 + 1_000 * DAY)
    row = cohort.filter(pl.col("author_hash") == "leaky").row(0, named=True)

    cutoff_a = row["cutoff_a"]
    end_a = row["followup_end_a"]
    user = frame.filter(pl.col("author_hash") == "leaky")
    label_a = user.filter(
        pl.col("subreddit").is_in(TARGETS)
        & (pl.col("created_utc") > cutoff_a)
        & (pl.col("created_utc") <= end_a)
    )

    assert label_a.height == row["label_n_a"] == 25
    assert label_a["created_utc"].min() > cutoff_a
    assert label_a["created_utc"].max() <= end_a
    expected_weeks = {
        (timestamp - cutoff_a - 1) // (7 * DAY)
        for timestamp in label_offsets
    }
    assert row["label_weeks_a"] == len(expected_weeks)
    assert row["label_deep_a"] is True

    # Gateway activity is never part of the label, even inside the window.
    cutoff_b = row["cutoff_b"]
    end_b = row["followup_end_b"]
    label_b = user.filter(
        pl.col("subreddit").is_in(TARGETS)
        & (pl.col("created_utc") > cutoff_b)
        & (pl.col("created_utc") <= end_b)
    )
    assert label_b.height == row["label_n_b"] == 30
    assert row["label_n_b"] > row["n_target_window_a"]


def test_far_future_records_do_not_change_features_or_labels(
    make_contributions, params: CohortParams
) -> None:
    rows, _ = _leaky_rows()
    future_start = T0 + 4 + 90 * DAY + 1
    without_future = [row for row in rows if row["created_utc"] < future_start]
    data_end = T0 + 1_000 * DAY

    full = _build(make_contributions(rows), params, data_end=data_end)
    trimmed = _build(make_contributions(without_future), params, data_end=data_end)

    columns = [
        "cutoff_a",
        "cutoff_b",
        "n_target_window_a",
        "n_gateway_pre_b",
        "label_n_a",
        "label_weeks_a",
        "label_deep_a",
        "label_n_b",
        "label_weeks_b",
        "label_deep_b",
    ]
    full_row = full.filter(pl.col("author_hash") == "leaky").select(columns)
    trimmed_row = trimmed.filter(pl.col("author_hash") == "leaky").select(columns)
    assert full_row.equals(trimmed_row)


def test_records_inside_label_window_change_labels_not_features(
    make_contributions, params: CohortParams
) -> None:
    rows, _ = _leaky_rows()
    frame = make_contributions(rows)
    data_end = T0 + 1_000 * DAY
    before = _build(frame, params, data_end=data_end)
    before_row = before.filter(pl.col("author_hash") == "leaky").row(0, named=True)

    extra_inside = make_contributions(
        rows
        + [
            {
                "author_hash": "leaky",
                "subreddit": "incels",
                "created_utc": before_row["cutoff_a"] + 10 * DAY,
            }
        ]
    )
    after = _build(extra_inside, params, data_end=data_end)
    after_row = after.filter(pl.col("author_hash") == "leaky").row(0, named=True)
    assert after_row["n_target_window_a"] == before_row["n_target_window_a"]
    assert after_row["label_n_a"] == before_row["label_n_a"] + 1

    # A record exactly at the cutoff belongs to features, not labels.
    boundary = make_contributions(
        rows
        + [
            {
                "author_hash": "leaky",
                "subreddit": "incels",
                "created_utc": before_row["cutoff_a"],
            }
        ]
    )
    boundary_row = (
        _build(boundary, params, data_end=data_end)
        .filter(pl.col("author_hash") == "leaky")
        .row(0, named=True)
    )
    assert boundary_row["n_target_window_a"] == before_row["n_target_window_a"] + 1
    assert boundary_row["label_n_a"] == before_row["label_n_a"]


def test_insufficient_followup_is_null_not_negative(
    make_contributions, params: CohortParams
) -> None:
    t0 = T0
    rows: list[dict[str, Any]] = []
    for index in range(5):
        rows.append({"author_hash": "short", "subreddit": "incels", "created_utc": t0 + index})
    # Twenty post-window contributions, but observation ends one day too early.
    for index in range(20):
        rows.append(
            {
                "author_hash": "short",
                "subreddit": "incels",
                "created_utc": t0 + 4 + (index + 1) * DAY,
            }
        )
    cutoff_a = t0 + 4
    data_end = cutoff_a + 89 * DAY
    cohort = _build(make_contributions(rows), params, data_end=data_end)
    row = cohort.filter(pl.col("author_hash") == "short").row(0, named=True)

    assert row["eligible_a"] is False
    assert row["label_deep_a"] is None
    assert row["label_n_a"] == 20
    assert row["eligible_b"] is False
    assert row["label_deep_b"] is None
