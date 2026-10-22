"""Shared synthetic test data helpers. No real Reddit data is used."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radpath.cohort import CohortParams  # noqa: E402

DAY = 86_400
T0 = 1_600_000_000


@pytest.fixture
def params() -> CohortParams:
    """Default research parameters used across tests."""
    return CohortParams(
        first_n=5,
        feature_days=14,
        pre_entry_days=180,
        followup_days=90,
        deep_min_contributions=20,
        deep_min_active_weeks=4,
    )


@pytest.fixture
def make_contributions():
    """Factory building a minimal contribution frame from row dictionaries."""

    def _factory(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
        data: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            data.append(
                {
                    "id": row.get("id", f"id_{index:06d}"),
                    "author_hash": row["author_hash"],
                    "subreddit": str(row["subreddit"]).lower(),
                    "kind": row.get("kind", "comment"),
                    "created_utc": int(row["created_utc"]),
                    "title": row.get("title", ""),
                    "selftext": row.get("selftext", ""),
                    "body": row.get("body", "synthetic"),
                    "text": row.get("text", "synthetic"),
                    "has_text": bool(row.get("has_text", True)),
                    "score": int(row.get("score", 0)),
                    "num_comments": int(row.get("num_comments", 0)),
                    "link_id": row.get("link_id"),
                    "parent_id": row.get("parent_id"),
                }
            )
        frame = pl.DataFrame(data)
        return frame.with_columns(
            pl.col("created_utc").cast(pl.Int64),
            pl.col("score").cast(pl.Int64),
            pl.col("num_comments").cast(pl.Int64),
        )

    return _factory
