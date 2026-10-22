"""Tests for cohort entry, feature windows, eligibility and labels."""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from radpath.cohort import CohortParams, build_cohort, resolve_data_end

DAY = 86_400
T0 = 1_600_000_000
DATA_END = T0 + 400 * DAY


def _week_count(timestamps: list[int], cutoff: int) -> int:
    """Independent reimplementation of the documented active-week rule."""
    return len({(timestamp - cutoff - 1) // (7 * DAY) for timestamp in timestamps})


@pytest.fixture
def cohort_frame(make_contributions):
    """Mixed cohort: deep user, moderate user, capped window, too-late user."""
    rows: list[dict[str, Any]] = []

    # User A: 5 early target records then 24 over the next 75 days.
    for index in range(5):
        rows.append({"author_hash": "user_a", "subreddit": "incels", "created_utc": T0 + index})
    post_a = [T0 + 5 * DAY + int(index * 75 * DAY / 23) for index in range(24)]
    for timestamp in post_a:
        rows.append({"author_hash": "user_a", "subreddit": "incels", "created_utc": timestamp})
    # Gateway activity: 180 days before entry is included; t0 itself is not.
    for subreddit, offset in (
        ("lonely", -180 * DAY),
        ("lonely", -100 * DAY),
        ("ForeverAlone", -10 * DAY),
        ("socialanxiety", 0),
        ("lonely", 5 * DAY),
    ):
        rows.append({"author_hash": "user_a", "subreddit": subreddit, "created_utc": T0 + offset})

    # User B: 10 daily target records, followed by nothing.
    for index in range(10):
        rows.append(
            {
                "author_hash": "user_b",
                "subreddit": "Braincels",
                "created_utc": T0 + 100 * DAY + index,
            }
        )

    # User C: fewer than N records inside the 14-day cap; day-20 record is post-window.
    for offset_days in (0, 1, 2, 20):
        rows.append(
            {
                "author_hash": "user_c",
                "subreddit": "MGTOW",
                "created_utc": T0 + 150 * DAY + offset_days * DAY,
            }
        )

    # User D: enters too near the end of observation for a full follow-up.
    for offset_days in (0, 1):
        rows.append(
            {
                "author_hash": "user_d",
                "subreddit": "IncelsWithoutHate",
                "created_utc": DATA_END - 10 * DAY + offset_days * DAY,
            }
        )

    # User E: gateway-only, never enters a target community.
    rows.append({"author_hash": "user_e", "subreddit": "lonely", "created_utc": T0})

    return make_contributions(rows), post_a


def test_cohort_membership_and_cutoffs(cohort_frame, params: CohortParams) -> None:
    frame, post_a = cohort_frame
    cohort = build_cohort(
        frame,
        params,
        target_subreddits=["incels", "Braincels", "MGTOW", "IncelsWithoutHate"],
        gateway_subreddits=["lonely", "ForeverAlone", "socialanxiety"],
        data_end_utc=DATA_END,
    )
    assert cohort.height == 4
    assert "user_e" not in cohort["author_hash"].to_list()

    rows = {row["author_hash"]: row for row in cohort.iter_rows(named=True)}
    a = rows["user_a"]
    b = rows["user_b"]
    c = rows["user_c"]
    d = rows["user_d"]

    assert a["t0"] == T0
    assert a["entry_subreddit"] == "incels"
    assert a["cutoff_a"] == T0 + 4
    assert a["cutoff_b"] == T0 - 1
    assert a["n_target_window_a"] == 5
    assert a["n_target_records_total"] == 29
    assert a["eligible_a"] and a["eligible_b"]
    assert a["label_n_a"] == 24
    assert a["label_weeks_a"] == _week_count(post_a, T0 + 4)
    assert a["label_deep_a"] is True
    assert a["label_n_b"] == 29
    assert a["task_b_included"] is True
    assert a["label_deep_b"] is True

    assert a["n_gateway_pre_b"] == 3
    assert a["n_gateway_subreddits_pre_b"] == 2
    assert a["n_gateway_active_weeks_pre_b"] == 3

    assert b["cutoff_a"] == T0 + 100 * DAY + 4
    assert b["n_target_window_a"] == 5
    assert b["label_n_a"] == 5
    assert b["label_deep_a"] is False
    assert b["label_n_b"] == 10
    assert b["task_b_included"] is False
    assert b["label_deep_b"] is None

    # The 14-day cap applies when the Nth contribution has not arrived.
    assert c["cutoff_a"] == T0 + 150 * DAY + 14 * DAY
    assert c["n_target_window_a"] == 3
    assert c["label_n_a"] == 1
    assert c["label_deep_a"] is False
    assert c["label_n_b"] == 4
    assert c["task_b_included"] is False
    assert c["label_deep_b"] is None

    # Ineligible users get null labels, not negative labels.
    assert d["eligible_a"] is False
    assert d["eligible_b"] is False
    assert d["label_deep_a"] is None
    assert d["label_deep_b"] is None


def test_deep_label_requires_both_thresholds(
    make_contributions, params: CohortParams
) -> None:
    rows: list[dict[str, Any]] = []

    def add_user(user: str, t0: int, post_offsets: list[int]) -> None:
        for index in range(5):
            rows.append(
                {"author_hash": user, "subreddit": "incels", "created_utc": t0 + index}
            )
        for offset in post_offsets:
            rows.append(
                {"author_hash": user, "subreddit": "incels", "created_utc": t0 + 4 + offset}
            )

    # Exactly 20 contributions across exactly 4 week bins.
    add_user(
        "exact",
        T0,
        [DAY, 8 * DAY, 15 * DAY, 22 * DAY] + [DAY + 100] * 16,
    )
    # 19 contributions across 4 weeks.
    add_user(
        "count_short",
        T0 + 100 * DAY,
        [DAY, 8 * DAY, 15 * DAY, 22 * DAY] + [DAY + 100] * 15,
    )
    # 20 contributions but only 3 week bins.
    add_user(
        "weeks_short",
        T0 + 200 * DAY,
        [DAY, 8 * DAY, 15 * DAY] + [DAY + 100] * 17,
    )

    cohort = build_cohort(
        make_contributions(rows),
        params,
        target_subreddits=["incels"],
        gateway_subreddits=["lonely"],
        data_end_utc=T0 + 500 * DAY,
    )
    rows_by_user = {row["author_hash"]: row for row in cohort.iter_rows(named=True)}
    assert rows_by_user["exact"]["label_n_a"] == 20
    assert rows_by_user["exact"]["label_weeks_a"] == 4
    assert rows_by_user["exact"]["label_deep_a"] is True
    assert rows_by_user["count_short"]["label_n_a"] == 19
    assert rows_by_user["count_short"]["label_deep_a"] is False
    assert rows_by_user["weeks_short"]["label_n_a"] == 20
    assert rows_by_user["weeks_short"]["label_weeks_a"] == 3
    assert rows_by_user["weeks_short"]["label_deep_a"] is False


def test_resolve_data_end() -> None:
    frame = pl.DataFrame({"created_utc": [T0, T0 + 10]})
    assert resolve_data_end(frame) == T0 + 10
    assert resolve_data_end(frame, T0 + 999) == T0 + 999
    assert resolve_data_end(frame, "1600010000") == 1_600_010_000
    assert resolve_data_end(frame, "2021-01-01T00:00:00Z") == 1_609_459_200
    with pytest.raises(ValueError):
        resolve_data_end(frame, "not-a-date")
    with pytest.raises(ValueError):
        resolve_data_end(pl.DataFrame({"created_utc": []}))


def test_empty_target_set_returns_empty_cohort(
    make_contributions, params: CohortParams
) -> None:
    frame = make_contributions(
        [{"author_hash": "u", "subreddit": "lonely", "created_utc": T0}]
    )
    cohort = build_cohort(
        frame,
        params,
        target_subreddits=["incels"],
        gateway_subreddits=["lonely"],
        data_end_utc=T0 + 100 * DAY,
    )
    assert cohort.height == 0
    assert "cutoff_a" in cohort.columns


def test_target_ending_mid_cohort_truncates_only_affected_windows(
    make_contributions, params: CohortParams
) -> None:
    rows: list[dict[str, Any]] = []
    coverage = {
        "incels": (T0, T0 + 100 * DAY),
        "braincels": (T0, T0 + 500 * DAY),
    }

    def add_user(user: str, subreddit: str, t0: int) -> None:
        for index in range(5):
            rows.append(
                {"author_hash": user, "subreddit": subreddit, "created_utc": t0 + index}
            )
        rows.append(
            {
                "author_hash": user,
                "subreddit": "lonely",
                "created_utc": t0 - 10 * DAY,
            }
        )

    add_user("early", "incels", T0)
    add_user("mid", "braincels", T0 + 50 * DAY)
    add_user("late", "braincels", T0 + 200 * DAY)

    cohort = build_cohort(
        make_contributions(rows),
        params,
        target_subreddits=["incels", "braincels"],
        gateway_subreddits=["lonely"],
        coverage=coverage,
    )
    by_user = {row["author_hash"]: row for row in cohort.iter_rows(named=True)}

    assert by_user["early"]["eligible_a"] is True
    assert by_user["early"]["near_coverage_start"] is True
    assert by_user["early"]["label_deep_a"] is not None

    assert by_user["mid"]["eligible_a"] is False
    assert by_user["mid"]["ineligible_reason_a"] == "followup_truncated_by_coverage_end"
    assert by_user["mid"]["label_deep_a"] is None
    assert by_user["mid"]["near_coverage_start"] is False

    # incels had already ended before this user's window, so it cannot truncate it.
    assert by_user["late"]["eligible_a"] is True
    assert by_user["late"]["ineligible_reason_a"] is None
    assert by_user["late"]["near_coverage_start"] is False


def test_config_generates_robustness_and_comments_only_variants() -> None:
    cfg = {
        "feature_window": {"first_n": 5, "days": 14},
        "pre_entry_days": 180,
        "followup_days": 90,
        "deep_min_contributions": 20,
        "deep_min_active_weeks": 4,
        "robustness": {
            "min_contributions": [10, 20, 50],
            "min_active_weeks": [3, 4, 8],
            "followup_days": [90, 180],
            "comments_only": True,
        },
    }
    params = CohortParams.from_config(cfg)
    names = {variant.name for variant in params.all_variants()}
    assert params.primary_variant.name == "c20_w4_f90"
    assert "c10_w3_f180" in names
    assert "c50_w8_f90" in names
    assert "c20_w4_f90_comments" in names
    assert len(names) == 19  # 18 grid cells plus one comments-only variant
    comments = next(v for v in params.all_variants() if v.comments_only)
    assert comments.kinds == ("comment",)
