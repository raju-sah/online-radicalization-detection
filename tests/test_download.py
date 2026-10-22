"""Tests for the controlled download routes. No real data or network is used."""

from __future__ import annotations

import io
import json
import time
import urllib.error
from pathlib import Path
from typing import Any

import bencodepy
import orjson
import pytest
import zstandard as zstd

from radpath.download import (
    ArcticShiftClient,
    DownloadTarget,
    _load_checkpoint,
    download_api_target,
    load_torrent,
    map_torrent_files,
)


def _torrent_bytes(files: list[tuple[list[str], int]], name: str = "reddit") -> bytes:
    """Build a tiny multi-file torrent payload."""
    return bencodepy.encode(
        {
            b"announce": b"https://example.invalid/announce",
            b"info": {
                b"name": name.encode(),
                b"piece length": 16384,
                b"pieces": b"0" * 20,
                b"files": [{"length": length, "path": [part.encode() for part in path]} for path, length in files],
            },
        }
    )


def _records(ids_and_times: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [
        {"id": record_id, "created_utc": created, "subreddit": "incels", "body": "x"}
        for record_id, created in ids_and_times
    ]


def _read_zst_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as reader:
        text = io.TextIOWrapper(reader, encoding="utf-8")
        return [orjson.loads(line) for line in text if line.strip()]


class FakeClient:
    """Deterministic in-memory replacement for ArcticShiftClient."""

    def __init__(self, pages: list[list[dict[str, Any]]]) -> None:
        self.pages = list(pages)
        self.calls: list[int | None] = []

    def fetch_page(
        self,
        subreddit: str,
        kind: str,
        after: int | None,
        limit: int | str,
        before: int | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(after)
        return self.pages.pop(0) if self.pages else []


def test_torrent_mapping_is_case_insensitive_and_indexes_selected_files(tmp_path: Path) -> None:
    torrent_path = tmp_path / "tiny.torrent"
    torrent_path.write_bytes(
        _torrent_bytes(
            [
                (["data", "Incels_comments.zst"], 101),
                (["data", "braincels", "Braincels_Submissions.zst"], 202),
                (["data", "unrelated_comments.zst"], 303),
            ]
        )
    )

    torrent = load_torrent(torrent_path)
    mapping = map_torrent_files(torrent, ["incels", "Braincels"])

    assert mapping[DownloadTarget("incels", "comments")][0].index == 1
    assert mapping[DownloadTarget("incels", "comments")][0].length == 101
    assert mapping[DownloadTarget("braincels", "submissions")][0].index == 2
    assert all(target.subreddit != "unrelated" for target in mapping)


def test_api_pagination_deduplicates_page_boundary(tmp_path: Path) -> None:
    target = DownloadTarget("incels", "comments")
    now = int(time.time()) - 10_000
    pages = [
        _records([("a", now), ("b", now + 1), ("c", now + 2)]),
        _records([("c", now + 2), ("d", now + 3)]),
    ]
    client = FakeClient(pages)

    result = download_api_target(
        target,
        tmp_path,
        tmp_path / "checkpoints",
        client,  # type: ignore[arg-type]
        limit=3,
        progress=False,
    )

    assert result.status == "complete"
    assert result.record_count == 4
    assert [record["id"] for record in _read_zst_records(tmp_path / target.filename)] == ["a", "b", "c", "d"]
    assert client.calls[:2] == [None, now + 2]
    assert _load_checkpoint(tmp_path / "checkpoints" / "incels_comments.json")["complete"] is True


def test_api_resume_truncates_uncommitted_bytes_and_appends(tmp_path: Path) -> None:
    target = DownloadTarget("incels", "comments")
    now = int(time.time()) - 10_000
    final_path = tmp_path / target.filename
    checkpoint_path = tmp_path / "checkpoints" / target.filename.replace(".zst", ".json")
    checkpoint_path.parent.mkdir(parents=True)
    compressor = zstd.ZstdCompressor(level=3)
    first_frame = compressor.compress(orjson.dumps(_records([("a", now)])[0]) + b"\n")
    final_path.write_bytes(first_frame + b"uncommitted-garbage")
    checkpoint_path.write_text(
        json.dumps(
            {
                "last_utc": now,
                "count": 1,
                "first_utc": now,
                "recent_ids": ["a"],
                "committed_bytes": len(first_frame),
                "complete": False,
            }
        ),
        encoding="utf-8",
    )

    client = FakeClient([_records([("a", now), ("b", now + 1)])])
    result = download_api_target(
        target,
        tmp_path,
        checkpoint_path.parent,
        client,  # type: ignore[arg-type]
        limit=3,
        progress=False,
    )

    assert result.status == "complete"
    assert result.record_count == 2
    assert [record["id"] for record in _read_zst_records(final_path)] == ["a", "b"]
    assert client.calls[0] == now


def test_api_auto_limit_continues_until_empty_page(tmp_path: Path) -> None:
    target = DownloadTarget("incels", "comments")
    now = int(time.time()) - 10_000
    client = FakeClient(
        [
            _records([("a", now)]),
            _records([("b", now + 1)]),
            [],
        ]
    )

    result = download_api_target(
        target,
        tmp_path,
        tmp_path / "checkpoints",
        client,  # type: ignore[arg-type]
        limit="auto",
        progress=False,
    )

    assert result.status == "complete"
    assert result.record_count == 2
    assert [record["id"] for record in _read_zst_records(tmp_path / target.filename)] == ["a", "b"]


def test_time_series_total_sums_points(monkeypatch: pytest.MonkeyPatch) -> None:
    client = ArcticShiftClient("https://example.invalid", "test-agent", min_interval=0.0)
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda url: {"data": [{"date": 1, "value": 3}, {"date": 2, "value": 4}]},
    )
    assert client.time_series_total("incels", "comments") == 7


def test_empty_api_result_is_not_found_and_removes_file(tmp_path: Path) -> None:
    target = DownloadTarget("missing", "submissions")
    client = FakeClient([[]])
    result = download_api_target(
        target,
        tmp_path,
        tmp_path / "checkpoints",
        client,  # type: ignore[arg-type]
        limit=100,
        progress=False,
    )
    assert result.status == "not_found"
    assert not (tmp_path / target.filename).exists()


class _Response:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_retry_after_429_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        calls["count"] += 1
        if calls["count"] < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "0"},
                None,
            )
        return _Response(b'{"data": []}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = ArcticShiftClient(
        "https://example.invalid",
        "test-agent",
        min_interval=0.0,
        max_retries=3,
        backoff_seconds=0.0,
        sleep=sleeps.append,
    )

    assert client.fetch_page("incels", "comments", None, 100) == []
    assert calls["count"] == 3
    assert sleeps == [0.0, 0.0]
