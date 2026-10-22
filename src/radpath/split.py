"""User-level temporal train/validation/test splits with a label embargo.

Users are assigned by UTC calendar date of cohort entry (``t0``). All users
with the same entry date share a split, so ties can never straddle a
boundary. Fraction targets are resolved to cumulative user counts and then to
the first date reaching each target.

When a label-window end column is supplied, users in an earlier split whose
label window reaches the start date of the next split are removed from that
split (``split = null``). This prevents training or validation labels from
overlapping the later evaluation period. Row-level random splitting is
intentionally not supported.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Any, Mapping

import polars as pl

LOGGER = logging.getLogger(__name__)

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")


def temporal_split(
    frame: pl.DataFrame,
    fractions: Mapping[str, Any],
    time_col: str = "t0",
    id_col: str = "author_hash",
    label_end_col: str | None = None,
) -> pl.DataFrame:
    """Assign each user to one temporal split, then apply the label embargo.

    Args:
        frame: User-level frame, one row per user.
        fractions: Mapping with positive ``train``, ``val`` and ``test``
            fractions summing to 1.
        time_col: Entry timestamp column used for date thresholds.
        id_col: Unique user identifier retained for deterministic ordering.
        label_end_col: Optional last timestamp of the user's label window. If
            present, users whose window crosses the next split's start date
            are removed from their split.

    Returns:
        Frame sorted by ``(time_col, id_col)`` with a ``split`` column.
        Embargoed users keep their row but have a null ``split``.

    Raises:
        ValueError: If required columns are missing, fractions are invalid,
            or fewer than three distinct entry dates are available.
    """
    missing = {time_col, id_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for temporal split: {sorted(missing)}")
    if frame.height < 3:
        raise ValueError("At least 3 users are required to create three splits.")

    train_fraction, val_fraction, test_fraction = _validate_fractions(fractions)

    ordered = frame.sort([time_col, id_col])
    with_date = ordered.with_columns(
        pl.from_epoch(pl.col(time_col).cast(pl.Int64), time_unit="s")
        .dt.date()
        .alias("_entry_date")
    )
    date_counts = (
        with_date.group_by("_entry_date")
        .len()
        .sort("_entry_date")
    )
    dates: list[date] = date_counts["_entry_date"].to_list()
    counts: list[int] = [int(value) for value in date_counts["len"].to_list()]
    if len(dates) < 3:
        raise ValueError(
            "At least 3 distinct entry dates are required so users sharing a "
            "t0 date can stay in one split."
        )

    cumulative: list[int] = []
    running = 0
    for count in counts:
        running += count
        cumulative.append(running)
    n_users = ordered.height

    # Pick the last date of train and val. Leave at least one date for each
    # later split; the date's full user count stays together.
    train_target = max(1, int(n_users * train_fraction))
    train_index = _first_at_least(cumulative, train_target, 0, len(dates) - 3)

    val_target = max(
        cumulative[train_index] + 1,
        int(n_users * (train_fraction + val_fraction)),
    )
    val_index = _first_at_least(
        cumulative,
        val_target,
        train_index + 1,
        len(dates) - 2,
    )

    train_end = dates[train_index]
    val_end = dates[val_index]
    result = (
        with_date.with_columns(
            pl.when(pl.col("_entry_date") <= pl.lit(train_end))
            .then(pl.lit("train"))
            .when(pl.col("_entry_date") <= pl.lit(val_end))
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .drop("_entry_date")
    )

    if label_end_col is not None:
        result = _apply_embargo(
            result,
            label_end_col=label_end_col,
            val_start_date=dates[train_index + 1],
            test_start_date=dates[val_index + 1],
        )

    _assert_temporal_order(result, time_col)
    return result


def _validate_fractions(fractions: Mapping[str, Any]) -> tuple[float, float, float]:
    """Validate and return the three split fractions."""
    if set(fractions) != set(SPLIT_ORDER):
        raise ValueError("Split fractions must define exactly train, val and test.")
    train_fraction = float(fractions["train"])
    val_fraction = float(fractions["val"])
    test_fraction = float(fractions["test"])
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive.")
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")
    return train_fraction, val_fraction, test_fraction


def _first_at_least(
    cumulative: list[int],
    target: int,
    min_index: int,
    max_index: int,
) -> int:
    """First index in ``[min_index, max_index]`` reaching ``target``."""
    for index in range(min_index, max_index + 1):
        if cumulative[index] >= target:
            return index
    return max_index


def _midnight_utc(day: date) -> int:
    """Epoch seconds at 00:00 UTC on ``day``."""
    return int(datetime.combine(day, time.min, tzinfo=timezone.utc).timestamp())


def _apply_embargo(
    frame: pl.DataFrame,
    label_end_col: str,
    val_start_date: date,
    test_start_date: date,
) -> pl.DataFrame:
    """Null out split assignments whose label window crosses a boundary."""
    if label_end_col not in frame.columns:
        raise ValueError(f"Missing label end column: {label_end_col}")

    val_start = _midnight_utc(val_start_date)
    test_start = _midnight_utc(test_start_date)
    crosses_val = (pl.col("split") == "train") & (
        pl.col(label_end_col).cast(pl.Int64) >= val_start
    )
    crosses_test = (pl.col("split") == "val") & (
        pl.col(label_end_col).cast(pl.Int64) >= test_start
    )
    n_train = int(frame.select(crosses_val.sum()).item())
    n_val = int(frame.select(crosses_test.sum()).item())
    LOGGER.info(
        "split embargo: dropped %d train and %d validation users "
        "(train labels must end before %s; val labels before %s)",
        n_train,
        n_val,
        val_start_date.isoformat(),
        test_start_date.isoformat(),
    )
    return frame.with_columns(
        pl.when(crosses_val | crosses_test)
        .then(None)
        .otherwise(pl.col("split"))
        .alias("split")
    )


def _assert_temporal_order(frame: pl.DataFrame, time_col: str) -> None:
    """Validate split exclusivity and temporal ordering in-place.

    Embargoed rows (null split) are allowed. A split may be empty after the
    embargo; that is a substantive signal that the window lengths are too
    short, and is reported by EDA rather than silently ignored.

    Raises:
        AssertionError: If a user appears twice, no user remains, or a later
            split starts before an earlier split ends.
    """
    assigned = frame.filter(pl.col("split").is_not_null())
    if assigned.height == 0:
        raise AssertionError("The split embargo removed every user.")
    if set(assigned["split"].unique().to_list()) - set(SPLIT_ORDER):
        raise AssertionError("Unexpected split label.")
    if frame.select(pl.col("author_hash").n_unique()).item() != frame.height:
        raise AssertionError("A user appears in more than one split.")

    bounds = {
        name: assigned.filter(pl.col("split") == name)[time_col]
        for name in SPLIT_ORDER
        if assigned.filter(pl.col("split") == name).height
    }
    if not {"train", "test"} <= set(bounds):
        raise AssertionError("Both train and test must remain non-empty.")
    if "train" in bounds and "val" in bounds:
        if bounds["train"].max() > bounds["val"].min():
            raise AssertionError(
                "Training entrants are not earlier than validation entrants."
            )
    if "val" in bounds and "test" in bounds:
        if bounds["val"].max() > bounds["test"].min():
            raise AssertionError(
                "Validation entrants are not earlier than test entrants."
            )
    if "train" in bounds and "test" in bounds:
        if bounds["train"].max() > bounds["test"].min():
            raise AssertionError("Training entrants are not earlier than test entrants.")


def split_summary(frame: pl.DataFrame, label_columns: tuple[str, ...] = ()) -> pl.DataFrame:
    """Summarise split sizes and positive rates per label column.

    Null-split (embargoed) users are excluded. Only splits that remain are
    returned.

    Args:
        frame: Split-assigned user-level frame.
        label_columns: Binary label columns to summarise.

    Returns:
        One row per non-empty split with user counts and positive rates.
    """
    aggregations: list[pl.Expr] = [pl.len().alias("n_users")]
    for column in label_columns:
        if column in frame.columns:
            aggregations.extend(
                [
                    (pl.col(column) == 1).sum().alias(f"{column}_n_positive"),
                    (pl.col(column) == 1).mean().alias(f"{column}_positive_rate"),
                ]
            )
    summary = (
        frame.filter(pl.col("split").is_not_null())
        .group_by("split")
        .agg(aggregations)
    )
    return summary.with_columns(
        pl.col("split").cast(pl.Enum(list(SPLIT_ORDER)))
    ).sort("split")
