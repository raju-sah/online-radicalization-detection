"""Tests for per-subreddit coverage and manual overrides."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from radpath.coverage import (
    apply_coverage_overrides,
    coverage_from_contributions,
    load_coverage,
    parse_utc,
    write_coverage,
)

DAY = 86_400
T0 = 1_600_000_000


def test_parse_utc_accepts_epoch_and_iso() -> None:
    assert parse_utc(T0) == T0
    assert parse_utc(str(T0)) == T0
    assert parse_utc("2020-09-13T12:26:40Z") == T0
    with pytest.raises(ValueError):
        parse_utc("not-a-date")


def test_coverage_from_contributions_and_overrides() -> None:
    frame = pl.DataFrame(
        {
            "subreddit": ["Incels", "incels", "braincels"],
            "created_utc": [T0, T0 + 10, T0 + 5 * DAY],
        }
    )
    coverage = coverage_from_contributions(frame, {"incels", "braincels"})
    rows = {row["subreddit"]: row for row in coverage.iter_rows(named=True)}
    assert rows["incels"]["start_utc"] == T0
    assert rows["incels"]["end_utc"] == T0 + 10
    assert rows["incels"]["n_records"] == 2

    overridden = apply_coverage_overrides(
        coverage,
        {"incels": {"end_utc": T0 + 3 * DAY}, "MGTOW": {
            "start_utc": "2019-01-01T00:00:00Z",
            "end_utc": "2019-06-01T00:00:00Z",
        }},
    )
    rows = {row["subreddit"]: row for row in overridden.iter_rows(named=True)}
    assert rows["incels"]["end_utc"] == T0 + 3 * DAY
    assert rows["incels"]["start_utc"] == T0
    assert rows["incels"]["overridden"] is True
    assert rows["mgtow"]["n_records"] == 0
    assert rows["mgtow"]["overridden"] is True


def test_write_and_load_coverage(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"subreddit": ["incels", "incels"], "created_utc": [T0, T0 + 5]}
    )
    coverage = coverage_from_contributions(frame)
    path = write_coverage(tmp_path / "coverage.parquet", coverage)
    loaded = load_coverage(tmp_path)
    assert loaded.equals(pl.read_parquet(path))
    assert loaded.height == 1
