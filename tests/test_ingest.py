"""Tests for streaming ingestion, hashing and cleaning at the boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson
import polars as pl
import pytest
import zstandard as zstd

from radpath.clean import CleanRules
from radpath.config import get_salt
from radpath.ingest import (
    FileSpec,
    hash_author,
    ingest_file,
    load_contributions,
    normalise_created_utc,
)

SALT = "unit-test-salt-0123456789"


def _write_zst(path: Path, records: list[Any]) -> None:
    """Write JSON objects (or raw strings) as a zstandard JSONL file."""
    compressor = zstd.ZstdCompressor(level=3)
    with path.open("wb") as raw, compressor.stream_writer(raw) as stream:
        for record in records:
            payload = record if isinstance(record, str) else orjson.dumps(record).decode("utf-8")
            stream.write(payload.encode("utf-8") + b"\n")


def _submission(
    record_id: str,
    author: str,
    created_utc: Any = 1_600_000_000,
    subreddit: str = "incels",
    title: str = "hello",
    selftext: str = "world",
    score: Any = "7",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "author": author,
        "subreddit": subreddit,
        "created_utc": created_utc,
        "title": title,
        "selftext": selftext,
        "score": score,
        "num_comments": "2",
    }


def test_hash_author_is_salted_deterministic_and_case_insensitive() -> None:
    first = hash_author("Alice", SALT)
    second = hash_author("alice", SALT)
    assert first == second
    assert len(first) == 64
    assert first != hash_author("Alice", "a-different-salt-0123456789")
    assert "alice" not in first
    with pytest.raises(ValueError):
        hash_author("alice", "")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1_600_000_000, 1_600_000_000),
        ("1600000000", 1_600_000_000),
        (1_600_000_000.9, 1_600_000_000),
        ("1600000000.0", 1_600_000_000),
        (None, None),
        ("", None),
        ("nan", None),
        ("inf", None),
        ("not-a-date", None),
        (-5, None),
        (True, None),
    ],
)
def test_normalise_created_utc(value: Any, expected: int | None) -> None:
    assert normalise_created_utc(value) == expected


def test_ingest_hashes_drops_and_normalises(tmp_path: Path) -> None:
    raw_path = tmp_path / "tiny_submissions.zst"
    records: list[Any] = [
        _submission("s1", "Alice", 1_600_000_000, score="11"),
        _submission("s2", "alice", "1600000100"),
        _submission("s3", "[deleted]"),
        _submission("s4", "HelperBot"),
        _submission("s5", "AutoModerator"),
        _submission("s6", "empty_user", title="", selftext=""),
        _submission("s7", "bad_time", created_utc="never"),
        _submission("s8", "other_user", subreddit="AskReddit"),
        "this is not valid json",
        {"id": "s9", "author": "missing_subreddit", "created_utc": 1_600_000_000},
    ]
    _write_zst(raw_path, records)

    interim = tmp_path / "interim"
    stats = ingest_file(
        FileSpec(path=raw_path, kind="submissions"),
        out_dir=interim / "submissions",
        salt=SALT,
        rules=CleanRules(),
        batch_size=1,
        subreddit_filter={"incels"},
    )
    assert stats.total == 10
    assert stats.kept == 3
    assert stats.malformed_json == 1
    assert stats.dropped_deleted_author == 1
    assert stats.dropped_bot_author == 2
    assert stats.kept_empty_text == 1
    assert stats.kept_removed_text == 0
    assert stats.kept_deleted_text == 0
    assert stats.dropped_bad_timestamp == 1
    assert stats.dropped_subreddit == 1
    assert stats.dropped_missing_field == 1
    assert stats.coverage["incels"] == (1_600_000_000, 1_600_000_100, 3)

    output = interim / "submissions" / "tiny_submissions.parquet"
    frame = pl.read_parquet(output)
    assert frame.height == 3
    assert set(frame["author_hash"].to_list()) == {
        hash_author("Alice", SALT),
        hash_author("alice", SALT),
        hash_author("empty_user", SALT),
    }
    assert sorted(frame["created_utc"].to_list()) == [
        1_600_000_000,
        1_600_000_000,
        1_600_000_100,
    ]
    assert sorted(frame["score"].to_list()) == [7, 7, 11]
    assert frame["subreddit"].unique().to_list() == ["incels"]
    assert frame["kind"].unique().to_list() == ["submission"]
    empty_row = frame.filter(pl.col("author_hash") == hash_author("empty_user", SALT))
    assert empty_row["has_text"].to_list() == [False]
    assert empty_row["text"].to_list() == [""]

    # Raw usernames must not survive anywhere in the output.
    for column in frame.columns:
        if frame[column].dtype == pl.String:
            values = [value for value in frame[column].to_list() if value is not None]
            assert not any("alice" in value.lower() for value in values)
            assert not any("helperbot" in value.lower() for value in values)


def test_ingest_comments_and_load_contributions(tmp_path: Path) -> None:
    submissions = tmp_path / "one_submissions.zst"
    comments = tmp_path / "one_comments.zst"
    _write_zst(submissions, [_submission("s1", "poster")])
    _write_zst(
        comments,
        [
            {
                "id": "c1",
                "author": "commenter",
                "subreddit": "Incels",
                "created_utc": 1_600_000_050,
                "body": "a comment",
                "score": 3,
                "link_id": "t3_s1",
                "parent_id": "t3_s1",
            },
            {
                "id": "c2",
                "author": "commenter",
                "subreddit": "Incels",
                "created_utc": 1_600_000_060,
                "body": "[removed]",
                "score": 0,
            },
        ],
    )
    interim = tmp_path / "interim"
    for path, kind in ((submissions, "submissions"), (comments, "comments")):
        ingest_file(
            FileSpec(path=path, kind=kind),
            out_dir=interim / kind,
            salt=SALT,
            subreddit_filter={"incels"},
        )

    frame = load_contributions(interim)
    assert frame.height == 3
    assert set(frame["kind"].to_list()) == {"submission", "comment"}
    assert frame["subreddit"].unique().to_list() == ["incels"]
    assert frame["created_utc"].dtype == pl.Int64
    removed = frame.filter(pl.col("id") == "c2").row(0, named=True)
    assert removed["has_text"] is False
    assert removed["text"] == ""


def test_ingest_respects_overwrite_flag(tmp_path: Path) -> None:
    raw_path = tmp_path / "tiny.zst"
    _write_zst(raw_path, [_submission("s1", "alice")])
    spec = FileSpec(path=raw_path, kind="submissions")
    out_dir = tmp_path / "interim"
    ingest_file(spec, out_dir, salt=SALT, overwrite=True)
    with pytest.raises(FileExistsError):
        ingest_file(spec, out_dir, salt=SALT, overwrite=False)


def test_ingest_reports_distinct_bot_authors(tmp_path: Path) -> None:
    raw_path = tmp_path / "bots.zst"
    _write_zst(
        raw_path,
        [
            _submission("s1", "helper_bot"),
            _submission("s2", "AutoModerator"),
            _submission("s3", "real_user"),
        ],
    )
    stats = ingest_file(
        FileSpec(path=raw_path, kind="submissions"),
        tmp_path / "interim",
        salt=SALT,
    )
    assert stats.bot_suffix_authors == {hash_author("helper_bot", SALT)}
    assert stats.bot_explicit_authors == {hash_author("AutoModerator", SALT)}
    assert stats.seen_authors == {
        hash_author("helper_bot", SALT),
        hash_author("AutoModerator", SALT),
        hash_author("real_user", SALT),
    }
    assert stats.bot_suffix_share == pytest.approx(1 / 3)


def test_write_coverage_includes_configured_subreddits(tmp_path: Path) -> None:
    from radpath.coverage import write_coverage

    raw_path = tmp_path / "one.zst"
    _write_zst(raw_path, [_submission("s1", "alice", subreddit="incels")])
    stats = ingest_file(
        FileSpec(path=raw_path, kind="submissions"),
        tmp_path / "interim" / "submissions",
        salt=SALT,
    )
    path = write_coverage(
        tmp_path / "interim" / "coverage.parquet",
        stats.coverage,
        subreddits={"incels", "Braincels"},
    )
    frame = pl.read_parquet(path)
    rows = {row["subreddit"]: row for row in frame.iter_rows(named=True)}
    assert rows["incels"]["start_utc"] == 1_600_000_000
    assert rows["incels"]["n_records"] == 1
    assert rows["braincels"]["start_utc"] is None
    assert rows["braincels"]["n_records"] == 0


def test_ingest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_file(
            FileSpec(path=tmp_path / "nope.zst", kind="submissions"),
            out_dir=tmp_path / "out",
            salt=SALT,
        )


def test_get_salt_rejects_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RADPATH_SALT", "change-me")
    with pytest.raises(ValueError):
        get_salt()
    monkeypatch.setenv("RADPATH_SALT", "a-real-salt-0123456789")
    assert get_salt() == "a-real-salt-0123456789"


def test_ingest_discards_identifiers_embedded_in_free_text(tmp_path: Path) -> None:
    """Hashing the author must not leave names in persisted content."""
    raw = tmp_path / "mention.zst"
    _write_zst(raw, [_submission(
        "s1", "synthetic_author", title="synthetic_author introduces himself",
        selftext="Reply to u/synthetic_other and https://example.org/profile",
    )])
    ingest_file(FileSpec(raw, "submissions"), tmp_path / "out", SALT)
    frame = pl.read_parquet(tmp_path / "out" / "mention.parquet")
    assert frame["has_text"].to_list() == [True]
    for name in ("title", "selftext", "body", "text"):
        assert frame[name].to_list() == [""]
