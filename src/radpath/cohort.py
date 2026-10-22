"""Cohort construction, feature windows, follow-up eligibility and labels.

Two tasks share one cohort of users (first target-community contribution
defines entry time ``t0``) but use different feature windows:

* Task A (early signal): target-community records up to
  ``cutoff_a = min(timestamp of Nth contribution, t0 + feature_days)``.
* Task B (pre-entry): gateway-community records with
  ``t0 - pre_entry_days <= timestamp < t0``; ``cutoff_b = t0 - 1``.

In both tasks, a feature uses only records at or before the task cutoff and
a label uses only target-community records with timestamps strictly after
the cutoff and at or before ``cutoff + followup_days``. Users whose follow-up
window extends past the end of observation are ineligible and receive null
labels rather than a negative label.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import polars as pl

SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 604_800


@dataclass(frozen=True)
class CohortParams:
    """Research parameters needed to build windows and labels."""

    first_n: int
    feature_days: int
    pre_entry_days: int
    followup_days: int
    deep_min_contributions: int
    deep_min_active_weeks: int

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CohortParams":
        """Build parameters from a validated radpath config."""
        window = cfg["feature_window"]
        return cls(
            first_n=int(window["first_n"]),
            feature_days=int(window["days"]),
            pre_entry_days=int(cfg["pre_entry_days"]),
            followup_days=int(cfg["followup_days"]),
            deep_min_contributions=int(cfg["deep_min_contributions"]),
            deep_min_active_weeks=int(cfg["deep_min_active_weeks"]),
        )


def resolve_data_end(
    contributions: pl.DataFrame,
    configured: Any = None,
) -> int:
    """Resolve the end of observation in epoch seconds.

    Args:
        contributions: Cleaned contribution table.
        configured: Config value. ``None``/empty means use the latest
            timestamp in ``contributions``; numbers are epoch seconds; strings
            may be epoch strings or ISO-8601 dates (assumed UTC if naive).

    Returns:
        End-of-observation timestamp.

    Raises:
        ValueError: If no configured value is available and the table is
            empty, or if an ISO date cannot be parsed.
    """
    if configured is not None and str(configured).strip():
        value = str(configured).strip()
        try:
            return int(float(value))
        except ValueError:
            pass
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid data_end_utc: {configured!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())
    if contributions.height == 0:
        raise ValueError("Cannot infer data end from an empty contribution table.")
    return int(contributions["created_utc"].max())


def build_cohort(
    contributions: pl.DataFrame,
    params: CohortParams,
    target_subreddits: Iterable[str],
    gateway_subreddits: Iterable[str],
    data_end_utc: Any = None,
    split_fractions: Mapping[str, Any] | None = None,
) -> pl.DataFrame:
    """Build one row per cohort user with cutoffs, features and labels.

    Args:
        contributions: Cleaned records with at least ``id``, ``author_hash``,
            ``subreddit``, ``kind``, ``created_utc``.
        params: Window and label parameters.
        target_subreddits: Communities defining cohort entry.
        gateway_subreddits: Communities used for Task B pre-entry counts.
        data_end_utc: Optional end of observation (see
            :func:`resolve_data_end`).
        split_fractions: When provided, add a ``split`` column using
            :func:`radpath.split.temporal_split`.

    Returns:
        User-level DataFrame sorted by ``author_hash``. Label columns are
        null for ineligible users.

    Raises:
        ValueError: If required columns are missing or no target records
            remain.
    """
    required = {"id", "author_hash", "subreddit", "kind", "created_utc"}
    missing = required - set(contributions.columns)
    if missing:
        raise ValueError(f"Missing contribution columns: {sorted(missing)}")

    targets = {name.strip().lower() for name in target_subreddits}
    gateways = {name.strip().lower() for name in gateway_subreddits}
    if not targets:
        raise ValueError("At least one target subreddit is required.")

    if contributions.height == 0:
        return _empty_cohort(include_split=split_fractions is not None)

    frame = contributions.with_columns(
        pl.col("subreddit").str.to_lowercase(),
        pl.col("created_utc").cast(pl.Int64),
    )
    data_end = resolve_data_end(frame, data_end_utc)
    followup_seconds = params.followup_days * SECONDS_PER_DAY

    # Target records define t0. Deduplicate repeated IDs from overlapping dumps.
    core = (
        frame.filter(pl.col("subreddit").is_in(list(targets)))
        .sort(["author_hash", "created_utc", "id"])
        .unique(subset=["kind", "id"], keep="first", maintain_order=True)
    )
    if core.height == 0:
        return _empty_cohort(include_split=split_fractions is not None)

    entry = (
        core.group_by("author_hash", maintain_order=True)
        .agg(pl.col("subreddit").first().alias("entry_subreddit"))
    )
    users = core.group_by("author_hash").agg(
        pl.col("created_utc").min().alias("t0"),
        pl.len().alias("n_target_records_total"),
    ).join(entry, on="author_hash", how="left")

    nth = (
        core.with_columns(pl.col("id").cum_count().over("author_hash").alias("_rank"))
        .filter(pl.col("_rank") == params.first_n)
        .select("author_hash", pl.col("created_utc").alias("_nth_ts"))
    )
    users = users.join(nth, on="author_hash", how="left").with_columns(
        (pl.col("t0") + params.feature_days * SECONDS_PER_DAY).alias("_a_cap"),
    )
    users = users.with_columns(
        pl.min_horizontal(
            pl.col("_a_cap"),
            pl.col("_nth_ts").fill_null(pl.col("_a_cap")),
        ).alias("cutoff_a"),
        (pl.col("t0") - 1).alias("cutoff_b"),
    ).with_columns(
        (pl.col("cutoff_a") + followup_seconds).alias("followup_end_a"),
        (pl.col("cutoff_b") + followup_seconds).alias("followup_end_b"),
    ).with_columns(
        (pl.col("followup_end_a") <= data_end).alias("eligible_a"),
        (pl.col("followup_end_b") <= data_end).alias("eligible_b"),
    )

    # Task A feature count: target records at or before cutoff_a.
    feature_a = (
        core.join(users.select("author_hash", "cutoff_a"), on="author_hash", how="left")
        .filter(pl.col("created_utc") <= pl.col("cutoff_a"))
        .group_by("author_hash")
        .agg(pl.len().alias("n_target_window_a"))
    )

    # Task B feature counts: gateway records in [t0 - pre_entry_days, t0),
    # equivalently at or before cutoff_b = t0 - 1.
    gateway = (
        frame.filter(pl.col("subreddit").is_in(list(gateways)))
        .sort(["author_hash", "created_utc", "id"])
        .unique(subset=["kind", "id"], keep="first", maintain_order=True)
        .join(users.select("author_hash", "t0"), on="author_hash", how="inner")
        .filter(
            (pl.col("created_utc") < pl.col("t0"))
            & (
                pl.col("created_utc")
                >= pl.col("t0") - params.pre_entry_days * SECONDS_PER_DAY
            )
        )
    )
    gateway_features = gateway.group_by("author_hash").agg(
        pl.len().alias("n_gateway_pre_b"),
        pl.col("subreddit").n_unique().alias("n_gateway_subreddits_pre_b"),
        (pl.col("created_utc") // SECONDS_PER_WEEK)
        .n_unique()
        .alias("n_gateway_active_weeks_pre_b"),
    )

    label_a = _window_labels(core, users, "cutoff_a", "followup_end_a", "a")
    label_b = _window_labels(core, users, "cutoff_b", "followup_end_b", "b")

    count_columns = [
        "n_target_window_a",
        "n_gateway_pre_b",
        "n_gateway_subreddits_pre_b",
        "n_gateway_active_weeks_pre_b",
        "label_n_a",
        "label_weeks_a",
        "label_n_b",
        "label_weeks_b",
    ]
    cohort = (
        users.join(feature_a, on="author_hash", how="left")
        .join(gateway_features, on="author_hash", how="left")
        .join(label_a, on="author_hash", how="left")
        .join(label_b, on="author_hash", how="left")
        .with_columns([pl.col(name).fill_null(0) for name in count_columns])
        .with_columns(
            pl.when(pl.col("eligible_a"))
            .then(
                (pl.col("label_n_a") >= params.deep_min_contributions)
                & (pl.col("label_weeks_a") >= params.deep_min_active_weeks)
            )
            .otherwise(None)
            .alias("label_deep_a"),
            pl.when(pl.col("eligible_b"))
            .then(
                (pl.col("label_n_b") >= params.deep_min_contributions)
                & (pl.col("label_weeks_b") >= params.deep_min_active_weeks)
            )
            .otherwise(None)
            .alias("label_deep_b"),
        )
        .drop("_a_cap", "_nth_ts")
        .sort("author_hash")
    )

    if split_fractions is not None:
        from radpath.split import temporal_split

        cohort = temporal_split(cohort, split_fractions)

    return cohort


def _window_labels(
    core: pl.DataFrame,
    users: pl.DataFrame,
    cutoff_col: str,
    end_col: str,
    suffix: str,
) -> pl.DataFrame:
    """Aggregate post-window contribution counts and active weeks.

    Active weeks are 7-day bins anchored at the first second after the
    cutoff, so a 90-day window contains at most 13 bins.
    """
    return (
        core.join(
            users.select("author_hash", cutoff_col, end_col),
            on="author_hash",
            how="left",
        )
        .filter(
            (pl.col("created_utc") > pl.col(cutoff_col))
            & (pl.col("created_utc") <= pl.col(end_col))
        )
        .group_by("author_hash")
        .agg(
            pl.len().alias(f"label_n_{suffix}"),
            (
                (pl.col("created_utc") - pl.col(cutoff_col) - 1)
                // SECONDS_PER_WEEK
            )
            .n_unique()
            .alias(f"label_weeks_{suffix}"),
        )
    )


def _empty_cohort(include_split: bool = False) -> pl.DataFrame:
    """Return an empty cohort frame with the expected schema."""
    schema: dict[str, Any] = {
        "author_hash": pl.String,
        "t0": pl.Int64,
        "n_target_records_total": pl.Int64,
        "entry_subreddit": pl.String,
        "cutoff_a": pl.Int64,
        "followup_end_a": pl.Int64,
        "eligible_a": pl.Boolean,
        "cutoff_b": pl.Int64,
        "followup_end_b": pl.Int64,
        "eligible_b": pl.Boolean,
        "n_target_window_a": pl.Int64,
        "n_gateway_pre_b": pl.Int64,
        "n_gateway_subreddits_pre_b": pl.Int64,
        "n_gateway_active_weeks_pre_b": pl.Int64,
        "label_n_a": pl.Int64,
        "label_weeks_a": pl.Int64,
        "label_deep_a": pl.Boolean,
        "label_n_b": pl.Int64,
        "label_weeks_b": pl.Int64,
        "label_deep_b": pl.Boolean,
    }
    if include_split:
        schema["split"] = pl.String
    return pl.DataFrame(schema=schema)
