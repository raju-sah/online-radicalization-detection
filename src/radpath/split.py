"""User-level temporal train/validation/test splits.

Users are ordered by cohort entry date (``t0``). The earliest entrants form
the training set, the middle group validation, and the latest entrants test.
Ties at a boundary are broken deterministically by hashed user ID; row-level
random splitting is intentionally not supported.
"""

from __future__ import annotations

from typing import Any, Mapping

import polars as pl

SPLIT_ORDER: tuple[str, ...] = ("train", "val", "test")


def temporal_split(
    frame: pl.DataFrame,
    fractions: Mapping[str, Any],
    time_col: str = "t0",
    id_col: str = "author_hash",
) -> pl.DataFrame:
    """Assign each user to exactly one temporal split.

    Args:
        frame: User-level frame, one row per user.
        fractions: Mapping with positive ``train``, ``val`` and ``test``
            fractions summing to 1.
        time_col: Entry timestamp column used for ordering.
        id_col: Unique user identifier used as a deterministic tie-breaker.

    Returns:
        The frame sorted by ``(time_col, id_col)`` with a ``split`` column.

    Raises:
        ValueError: If required columns are missing, fractions are invalid,
            or the frame is too small to populate all three splits.
    """
    missing = {time_col, id_col} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for temporal split: {sorted(missing)}")
    if frame.height < 3:
        raise ValueError("At least 3 users are required to create three splits.")

    train_fraction = float(fractions["train"])
    val_fraction = float(fractions["val"])
    test_fraction = float(fractions["test"])
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive.")
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    ordered = frame.sort([time_col, id_col])
    n_rows = ordered.height
    train_end = int(n_rows * train_fraction)
    val_end = int(n_rows * (train_fraction + val_fraction))

    # Guarantee all three splits are non-empty.
    train_end = min(max(train_end, 1), n_rows - 2)
    val_end = min(max(val_end, train_end + 1), n_rows - 1)

    split_expr = (
        pl.when(pl.int_range(n_rows) < train_end)
        .then(pl.lit("train"))
        .when(pl.int_range(n_rows) < val_end)
        .then(pl.lit("val"))
        .otherwise(pl.lit("test"))
    )
    result = ordered.with_columns(split_expr.alias("split"))

    _assert_temporal_order(result, time_col)
    return result


def _assert_temporal_order(frame: pl.DataFrame, time_col: str) -> None:
    """Validate split exclusivity and temporal ordering in-place.

    Raises:
        AssertionError: If a user appears twice or a later split starts
            before an earlier split ends.
    """
    counts = frame.group_by("split").len()
    if set(counts["split"].to_list()) != set(SPLIT_ORDER):
        raise AssertionError("Split assignment did not produce train/val/test.")
    if frame.select(pl.col("author_hash").n_unique()).item() != frame.height:
        raise AssertionError("A user appears in more than one split.")

    bounds = {
        name: frame.filter(pl.col("split") == name)[time_col]
        for name in SPLIT_ORDER
    }
    if bounds["train"].max() > bounds["val"].min():
        raise AssertionError("Training entrants are not earlier than validation entrants.")
    if bounds["val"].max() > bounds["test"].min():
        raise AssertionError("Validation entrants are not earlier than test entrants.")


def split_summary(frame: pl.DataFrame, label_columns: tuple[str, ...] = ()) -> pl.DataFrame:
    """Summarise split sizes and positive rates per label column.

    Args:
        frame: Split-assigned user-level frame.
        label_columns: Binary label columns to summarise.

    Returns:
        One row per split with user counts and positive rates (0-1).
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
    summary = frame.group_by("split").agg(aggregations)
    return summary.with_columns(
        pl.col("split").cast(pl.Enum(list(SPLIT_ORDER)))
    ).sort("split")
