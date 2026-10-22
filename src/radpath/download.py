"""Controlled data acquisition for radpath.

Downloads are allowed only through ``scripts/download_data.py`` and only for
subreddits configured in ``configs/default.yaml``. Two routes are supported:

1. Selected files from the configured Academic Torrents Reddit bundle,
   fetched with ``aria2c``.
2. Arctic Shift subreddit-level API pages, used only for targets absent from
   the torrent (or when ``aria2c`` is unavailable).

No per-user or per-author endpoint is used. API output is streamed directly
to zstandard-compressed NDJSON; checkpoints allow safe resumption.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import bencodepy
import orjson
import zstandard as zstd
from tqdm import tqdm

from radpath.config import PROJECT_ROOT, resolve_paths
from radpath.ingest import normalise_created_utc

LOGGER = logging.getLogger(__name__)

KINDS: tuple[str, ...] = ("comments", "submissions")
DOWNLOAD_PRIORITY: tuple[str, ...] = (
    "incels",
    "braincels",
    "incelswithouthate",
    "mgtow",
    "inceltears",
    "foreveralone",
)
WINDOW_SECONDS = 90 * 86_400
API_ENDPOINTS: dict[str, str] = {
    "comments": "/api/comments/search",
    "submissions": "/api/posts/search",
}
TORRENT_EXTENSIONS = (".zst", ".zstd", ".ndjson", ".jsonl", ".json", ".xz", ".bz2", ".gz")
MANIFEST_STATUSES = ("complete", "not_found", "partial", "error")


class DownloadError(RuntimeError):
    """Base error for acquisition failures."""


class ApiBlockedError(DownloadError):
    """Raised after repeated API rate-limit or server errors."""


class ApiNotFoundError(DownloadError):
    """Raised when an API endpoint returns HTTP 404."""


class SharedRateLimiter:
    """Thread-safe global request limiter.

    All workers must share one instance so the configured minimum interval is
    enforced across the whole process, not per worker. ``wait`` sleeps while
    holding the lock, which deliberately serialises callers: two workers can
    never start requests less than ``min_interval`` seconds apart.
    """

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> float:
        """Block until this caller may request, returning the granted time."""
        if self.min_interval <= 0:
            return self._clock()
        with self._lock:
            now = self._clock()
            delay = self._next_allowed - now
            if delay > 0:
                self._sleep(delay)
                now = self._clock()
            self._next_allowed = now + self.min_interval
            return now


@dataclass(frozen=True)
class DownloadTarget:
    """One configured subreddit/type pair."""

    subreddit: str
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def filename(self) -> str:
        """Output filename, e.g. ``incels_comments.zst``."""
        return f"{self.subreddit}_{self.kind}.zst"


@dataclass(frozen=True)
class TorrentFile:
    """A file inside a torrent, with its 1-based aria2 select index."""

    index: int
    path: tuple[str, ...]
    length: int


@dataclass(frozen=True)
class TorrentMeta:
    """Parsed multi-file torrent metadata."""

    name: str
    files: tuple[TorrentFile, ...]
    is_multi_file: bool

    @property
    def total_bytes(self) -> int:
        """Total size of all files in the torrent."""
        return sum(item.length for item in self.files)


@dataclass
class PlanItem:
    """One acquisition decision shown in the dry-run table."""

    target: DownloadTarget
    route: str
    size: int | None
    status: str = "pending"
    files: tuple[TorrentFile, ...] = ()
    existing_bytes: int | None = None


@dataclass
class ApiResult:
    """Outcome of one API target download."""

    status: str
    bytes: int
    record_count: int
    first_created_utc: int | None
    last_created_utc: int | None


def download_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the validated ``download`` config section."""
    section = cfg.get("download")
    if not isinstance(section, dict):
        raise ValueError("Config must define a download section.")
    required = (
        "torrent_url",
        "arctic_shift_base_url",
        "user_agent",
        "torrent_dir",
        "checkpoint_dir",
        "manifest",
        "api_limit",
        "api_min_interval_seconds",
        "api_max_retries",
        "api_backoff_seconds",
        "min_free_gb",
    )
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"Missing download config keys: {missing}")
    return section


def download_paths(cfg: Mapping[str, Any], download_cfg: Mapping[str, Any] | None = None) -> dict[str, Path]:
    """Resolve acquisition paths against the repository root."""
    section = download_cfg or download_config(cfg)
    base = resolve_paths(cfg)

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path

    paths = {
        "raw": base["raw"],
        "torrent_dir": resolve(section["torrent_dir"]),
        "checkpoint_dir": resolve(section["checkpoint_dir"]),
        "manifest": resolve(section["manifest"]),
        "log_path": resolve(section.get("log_path", "reports/download.log")),
    }
    for key in ("raw", "torrent_dir", "checkpoint_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["log_path"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def configured_targets(
    cfg: Mapping[str, Any],
    only: Sequence[str] | None = None,
    kind: str | None = None,
) -> list[DownloadTarget]:
    """List configured subreddit/type targets, optionally filtered.

    Args:
        cfg: radpath configuration.
        only: Case-insensitive subreddit names to keep. Must all be configured.
        kind: Optional single kind (``comments`` or ``submissions``).

    Returns:
        Deterministic, de-duplicated targets in config order.

    Raises:
        ValueError: If an unknown subreddit or kind is requested.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"--type must be one of {KINDS}")
    names: list[str] = []
    seen: set[str] = set()
    for key in ("target_subreddits", "control_subreddits", "gateway_subreddits"):
        for value in cfg.get(key, []):
            name = str(value).strip().lower()
            if name and name not in seen:
                seen.add(name)
                names.append(name)

    if only:
        requested = {str(value).strip().lower() for value in only}
        unknown = requested - seen
        if unknown:
            raise ValueError(
                "Refusing subreddits not listed in configs/default.yaml: "
                + ", ".join(sorted(unknown))
            )
        names = [name for name in names if name in requested]

    priority = {name: index for index, name in enumerate(DOWNLOAD_PRIORITY)}
    config_order = {name: index for index, name in enumerate(names)}
    names = sorted(
        names,
        key=lambda name: (
            priority.get(name, len(priority)),
            config_order[name],
        ),
    )

    kinds = (kind,) if kind else KINDS
    return [DownloadTarget(name, record_kind) for name in names for record_kind in kinds]


def _decode(value: Any) -> str:
    """Decode bencode bytes to text."""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def load_torrent(path: Path) -> TorrentMeta:
    """Parse a .torrent file into file/index metadata.

    Args:
        path: Local .torrent file.

    Returns:
        Parsed metadata with 1-based file indices matching aria2c.

    Raises:
        DownloadError: If required torrent keys are missing or malformed.
    """
    try:
        decoded = bencodepy.decode(path.read_bytes())
        info = decoded[b"info"]
    except Exception as exc:  # bencodepy raises several parser exception types
        raise DownloadError(f"Could not parse torrent metadata: {path.name}") from exc

    name = _decode(info.get(b"name", b""))
    files_meta = info.get(b"files")
    files: list[TorrentFile] = []
    if files_meta:
        for index, item in enumerate(files_meta, start=1):
            parts = tuple(_decode(part) for part in item[b"path"])
            files.append(TorrentFile(index=index, path=parts, length=int(item[b"length"])))
    elif b"length" in info:
        files.append(TorrentFile(index=1, path=(name,), length=int(info[b"length"])))
    if not files:
        raise DownloadError(f"Torrent contains no files: {path.name}")
    return TorrentMeta(name=name, files=tuple(files), is_multi_file=bool(files_meta))


def _strip_extension(name: str) -> str:
    """Strip a known dump extension from a filename."""
    lowered = name.lower()
    for extension in TORRENT_EXTENSIONS:
        if lowered.endswith(extension):
            return lowered[: -len(extension)]
    return lowered


def _infer_kind(path_parts: Sequence[str]) -> str | None:
    """Infer comments/submissions from a torrent path."""
    stem = _strip_extension(path_parts[-1])
    match = re.search(r"(?:^|[._-])(comments|submissions)(?:$|[._-])", stem)
    if match:
        return match.group(1)
    for kind in KINDS:
        if stem.endswith(kind):
            return kind
    return None


def _subreddit_candidates(path_parts: Sequence[str]) -> list[str]:
    """Return plausible subreddit names from a torrent path."""
    stem = _strip_extension(path_parts[-1])
    match = re.search(r"(?:^|[._-])(comments|submissions)(?:$|[._-])", stem)
    prefix = stem[: match.start()] if match else stem
    for kind in KINDS:
        if prefix.endswith(kind):
            prefix = prefix[: -len(kind)]
    prefix = prefix.strip("._- ")
    for common in ("r_", "r-", "reddit_", "reddit-", "dump_", "dumps_"):
        if prefix.startswith(common):
            prefix = prefix[len(common) :]
    candidates = [prefix] if prefix else []
    candidates.extend(_strip_extension(part) for part in path_parts[:-1])
    return [candidate.strip("._- ") for candidate in candidates if candidate.strip("._- ")]


def _subreddit_score(candidate: str, subreddit: str) -> int:
    """Score a candidate name against a configured subreddit (0 = no match)."""
    candidate_lower = candidate.lower()
    subreddit_lower = subreddit.lower()
    if candidate_lower == subreddit_lower:
        return 2
    pattern = rf"(?:^|[._-]){re.escape(subreddit_lower)}(?:$|[._-])"
    return 1 if re.search(pattern, candidate_lower) else 0


def map_torrent_files(
    torrent: TorrentMeta,
    subreddits: Iterable[str],
    kinds: Iterable[str] = KINDS,
) -> dict[DownloadTarget, tuple[TorrentFile, ...]]:
    """Map configured subreddits/types to torrent files case-insensitively.

    Args:
        torrent: Parsed torrent metadata.
        subreddits: Configured names allowed for acquisition.
        kinds: Record kinds to consider.

    Returns:
        Mapping from target to one or more torrent files. Unmatched targets
        are absent (they should use the API route).
    """
    allowed = {str(value).lower() for value in subreddits}
    allowed_kinds = {str(value) for value in kinds}
    selected: dict[DownloadTarget, list[TorrentFile]] = {}
    for torrent_file in torrent.files:
        kind = _infer_kind(torrent_file.path)
        if kind not in allowed_kinds:
            continue
        scores: dict[str, int] = {}
        for candidate in _subreddit_candidates(torrent_file.path):
            for name in allowed:
                score = _subreddit_score(candidate, name)
                if score:
                    scores[name] = max(scores.get(name, 0), score)
        if not scores:
            continue
        best_score = max(scores.values())
        winners = sorted(name for name, score in scores.items() if score == best_score)
        if len(winners) > 1:
            LOGGER.warning(
                "Ambiguous torrent path %s matches multiple subreddits; skipping",
                "/".join(torrent_file.path),
            )
            continue
        target = DownloadTarget(winners[0], kind)
        selected.setdefault(target, []).append(torrent_file)
    return {target: tuple(files) for target, files in selected.items()}


def fetch_torrent(
    url: str,
    destination: Path,
    user_agent: str,
    timeout: float = 60.0,
    refresh: bool = False,
) -> Path:
    """Download and validate torrent metadata via the project script.

    Args:
        url: Torrent URL from config.
        destination: Local cache path.
        user_agent: Descriptive HTTP User-Agent.
        timeout: Network timeout in seconds.
        refresh: Re-download even when a cached torrent parses.

    Returns:
        Path to a valid cached torrent.

    Raises:
        DownloadError: On network or parse failure.
    """
    if destination.exists() and not refresh:
        try:
            load_torrent(destination)
            return destination
        except DownloadError:
            LOGGER.warning("Cached torrent metadata is invalid; re-downloading.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        load_torrent(temporary)
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise DownloadError(f"Could not fetch torrent metadata: {exc}") from exc
    LOGGER.info("Torrent metadata cached at %s", destination)
    return destination


def fetch_torrent_magnet(
    magnet: str,
    destination: Path,
    user_agent: str,
    timeout: float = 600.0,
) -> Path:
    """Fetch torrent metadata from a magnet URI without downloading payloads.

    aria2c saves the metadata as ``<infohash>.torrent`` under ``destination``'s
    directory. The file is validated and renamed to the configured cache name.

    Raises:
        DownloadError: If aria2c is missing, times out, or saves no valid torrent.
    """
    executable = shutil.which("aria2c")
    if executable is None:
        raise DownloadError("aria2c is not installed; cannot fetch magnet metadata.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in destination.parent.glob("*.torrent")}
    command = [
        executable,
        "--bt-metadata-only=true",
        "--bt-save-metadata=true",
        "--seed-time=0",
        "--file-allocation=none",
        "--summary-interval=30",
        "--console-log-level=notice",
        "--dir=" + str(destination.parent),
        "--user-agent=" + user_agent,
        magnet,
    ]
    LOGGER.info("Fetching torrent metadata from magnet (up to %.0fs)", timeout)
    try:
        completed = subprocess.run(command, check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(f"Magnet metadata fetch timed out after {timeout:.0f}s") from exc
    if completed.returncode != 0:
        LOGGER.warning("aria2c magnet metadata exited with status %s", completed.returncode)
    candidates = [
        path
        for path in destination.parent.glob("*.torrent")
        if path.resolve() not in before and path != destination
    ]
    for candidate in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            load_torrent(candidate)
        except DownloadError:
            continue
        os.replace(candidate, destination)
        LOGGER.info("Torrent metadata from magnet cached at %s", destination)
        return destination
    raise DownloadError("Magnet metadata fetch produced no valid .torrent file.")


def aria2_available() -> bool:
    """Return True when ``aria2c`` is on PATH."""
    return shutil.which("aria2c") is not None


def torrent_source_path(meta: TorrentMeta, torrent_file: TorrentFile, torrent_dir: Path) -> Path:
    """Return the aria2c output path for one torrent file."""
    if meta.is_multi_file:
        return torrent_dir / meta.name / Path(*torrent_file.path)
    return torrent_dir / meta.name


def download_torrent_selection(
    torrent_path: Path,
    meta: TorrentMeta,
    selections: Mapping[DownloadTarget, tuple[TorrentFile, ...]],
    torrent_dir: Path,
) -> None:
    """Run aria2c for the selected torrent file indices."""
    indices = sorted({item.index for files in selections.values() for item in files})
    if not indices:
        return
    executable = shutil.which("aria2c")
    if executable is None:
        raise DownloadError("aria2c is not installed.")
    torrent_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        f"--select-file={','.join(str(index) for index in indices)}",
        "--seed-time=0",
        "--continue=true",
        "--dir=" + str(torrent_dir),
        "--file-allocation=none",
        "--summary-interval=30",
        "--console-log-level=notice",
        "--download-result=full",
        str(torrent_path),
    ]
    LOGGER.info(
        "Starting aria2c for %d torrent file(s), %d selected index(es)",
        len(selections),
        len(indices),
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        LOGGER.warning("aria2c exited with status %s; verifying files.", completed.returncode)


def scan_zst_file(path: Path, progress: bool = True) -> dict[str, Any]:
    """Stream a .zst file to count records and timestamp bounds.

    Returns:
        Dictionary with ``record_count``, ``first_created_utc``,
        ``last_created_utc`` and ``malformed_lines``.
    """
    decompressor = zstd.ZstdDecompressor(max_window_size=2**31)
    record_count = 0
    malformed = 0
    first: int | None = None
    last: int | None = None
    with path.open("rb") as raw:
        with decompressor.stream_reader(raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            iterator = tqdm(
                text_stream,
                desc=f"scan {path.name}",
                unit=" rec",
                mininterval=30,
                disable=not progress,
            )
            for line in iterator:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = orjson.loads(line)
                except orjson.JSONDecodeError:
                    malformed += 1
                    continue
                record_count += 1
                timestamp = normalise_created_utc(record.get("created_utc"))
                if timestamp is not None:
                    first = timestamp if first is None else min(first, timestamp)
                    last = timestamp if last is None else max(last, timestamp)
            iterator.close()
    return {
        "record_count": record_count,
        "first_created_utc": first,
        "last_created_utc": last,
        "malformed_lines": malformed,
    }


def finalise_torrent_target(
    target: DownloadTarget,
    files: Sequence[TorrentFile],
    meta: TorrentMeta,
    torrent_dir: Path,
    raw_dir: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Move/concat torrent outputs into the canonical raw filename.

    Returns:
        Aggregate file statistics for the manifest.
    """
    sources = [torrent_source_path(meta, item, torrent_dir) for item in files]
    expected_size = sum(item.length for item in files)
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise DownloadError(f"Torrent output missing for {target.subreddit}/{target.kind}: {missing}")
    actual_total = sum(source.stat().st_size for source in sources)
    if actual_total != expected_size:
        raise DownloadError(
            f"Incomplete torrent output for {target.subreddit}/{target.kind}: "
            f"{actual_total} of {expected_size} bytes"
        )

    final_path = raw_dir / target.filename
    if final_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {final_path}")
        final_path.unlink()

    if len(sources) == 1:
        os.replace(sources[0], final_path)
    else:
        partial = final_path.with_name(final_path.name + ".part")
        with partial.open("wb") as output:
            for source in sources:
                with source.open("rb") as handle:
                    shutil.copyfileobj(handle, output, length=8 * 1024 * 1024)
        if partial.stat().st_size != expected_size:
            partial.unlink(missing_ok=True)
            raise DownloadError(f"Concatenated size mismatch for {target.subreddit}/{target.kind}")
        os.replace(partial, final_path)
        for source in sources:
            source.unlink(missing_ok=True)

    LOGGER.info("Scanning %s", final_path.name)
    scan = scan_zst_file(final_path)
    return {
        "bytes": final_path.stat().st_size,
        "record_count": scan["record_count"],
        "first_created_utc": scan["first_created_utc"],
        "last_created_utc": scan["last_created_utc"],
        "malformed_lines": scan["malformed_lines"],
    }


class ArcticShiftClient:
    """Rate-limited Arctic Shift client for subreddit-level searches."""

    def __init__(
        self,
        base_url: str,
        user_agent: str,
        timeout: float = 60.0,
        min_interval: float = 1.0,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        limiter: SharedRateLimiter | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout = timeout
        self.min_interval = max(0.0, min_interval)
        self.max_retries = max(0, max_retries)
        self.backoff_seconds = max(0.0, backoff_seconds)
        self._sleep = sleep
        self.rate_limiter = limiter or SharedRateLimiter(
            self.min_interval,
            sleep=sleep,
        )

    def _wait_for_rate_limit(self) -> None:
        """Wait for this worker's turn under the shared global limiter."""
        self.rate_limiter.wait()

    def fetch_page(
        self,
        subreddit: str,
        kind: str,
        after: int | None,
        limit: int | str,
        before: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch one ascending page of subreddit records."""
        if kind not in API_ENDPOINTS:
            raise ValueError(f"Unsupported kind: {kind}")
        params: dict[str, Any] = {
            "subreddit": subreddit,
            "sort": "asc",
            "limit": limit,
        }
        if after is not None:
            params["after"] = int(after)
        if before is not None:
            params["before"] = int(before)
        url = (
            f"{self.base_url}{API_ENDPOINTS[kind]}?"
            f"{urllib.parse.urlencode(params)}"
        )
        try:
            payload = self._request_json(url)
        except DownloadError as exc:
            # Arctic Shift sometimes returns 422 for a boundary timestamp even
            # though the same query works one second earlier. Retry slightly
            # earlier; the caller de-duplicates IDs, and ingest de-duplicates
            # again, so an overlap is safer than dropping records.
            if after is None or "HTTP 422" not in str(exc):
                raise
            retried = False
            for shift in range(1, 4):
                params["after"] = max(0, int(after) - shift)
                retry_url = (
                    f"{self.base_url}{API_ENDPOINTS[kind]}?"
                    f"{urllib.parse.urlencode(params)}"
                )
                LOGGER.warning(
                    "HTTP 422 at after=%s; retrying with after=%s",
                    after,
                    params["after"],
                )
                try:
                    payload = self._request_json(retry_url)
                    retried = True
                    break
                except DownloadError as retry_exc:
                    if "HTTP 422" not in str(retry_exc):
                        raise
            if not retried:
                raise
        if isinstance(payload, dict):
            data = payload.get("data", [])
        elif isinstance(payload, list):
            data = payload
        else:
            raise DownloadError("Unexpected Arctic Shift response shape.")
        if not isinstance(data, list):
            raise DownloadError("Arctic Shift response 'data' is not a list.")
        return data

    def time_series_total(self, subreddit: str, kind: str, precision: str = "year") -> int | None:
        """Return an aggregate record count, or None when no series is available.

        This uses the subreddit-level ``/api/time_series`` count key. It is
        only used for planning; no per-user data is requested.
        """
        series = "posts" if kind == "submissions" else "comments"
        params = {
            "key": f"r/{subreddit}/{series}/count",
            "precision": precision,
        }
        url = f"{self.base_url}/api/time_series?{urllib.parse.urlencode(params)}"
        payload = self._request_json(url)
        if not isinstance(payload, dict):
            return None
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            return None
        total = 0
        for point in data:
            if not isinstance(point, dict):
                return None
            total += int(point.get("value", 0))
        return total

    def _request_json(self, url: str) -> Any:
        """GET JSON with rate limiting, retries and exponential backoff."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read()
                return orjson.loads(body)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise ApiNotFoundError(f"Not found: {url}") from exc
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable:
                    raise DownloadError(f"HTTP {exc.code} for {url}") from exc
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self._retry_delay(exc, attempt)
                LOGGER.warning(
                    "Arctic Shift HTTP %s; retrying in %.1fs (%d/%d)",
                    exc.code,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep(delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError, orjson.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = min(60.0, self.backoff_seconds * (2**attempt))
                LOGGER.warning(
                    "Arctic Shift request failed (%s); retrying in %.1fs (%d/%d)",
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep(delay)
        raise ApiBlockedError(
            f"Arctic Shift repeatedly unavailable after {self.max_retries + 1} attempts: {last_error}"
        )

    def _retry_delay(self, error: urllib.error.HTTPError, attempt: int) -> float:
        """Respect Retry-After when present, otherwise use exponential backoff."""
        header = error.headers.get("Retry-After") if error.headers else None
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass
        return min(60.0, self.backoff_seconds * (2**attempt))


def _checkpoint_path(checkpoint_dir: Path, target: DownloadTarget) -> Path:
    """Return the per-target API checkpoint path."""
    return checkpoint_dir / f"{target.subreddit}_{target.kind}.json"


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    """Load a JSON checkpoint, or None when absent/corrupt."""
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Ignoring corrupt checkpoint: %s", path.name)
        return None
    return value if isinstance(value, dict) else None


def _save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """Atomically write a checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _new_checkpoint() -> dict[str, Any]:
    """Return a fresh API checkpoint."""
    return {
        "last_utc": None,
        "count": 0,
        "first_utc": None,
        "recent_ids": [],
        "committed_bytes": 0,
        "complete": False,
    }


def download_api_target(
    target: DownloadTarget,
    raw_dir: Path,
    checkpoint_dir: Path,
    client: ArcticShiftClient,
    limit: int | str = 100,
    overwrite: bool = False,
    progress: bool = True,
    window_seconds: int = WINDOW_SECONDS,
) -> ApiResult:
    """Download one subreddit/type through the Arctic Shift API.

    Checkpoints store the last timestamp, recent page IDs and the committed
    byte offset. On resume, any bytes beyond the checkpoint are truncated,
    then paging continues with boundary deduplication by ID.
    """
    final_path = raw_dir / target.filename
    checkpoint_path = _checkpoint_path(checkpoint_dir, target)
    checkpoint = _load_checkpoint(checkpoint_path)

    if checkpoint and checkpoint.get("complete") and final_path.exists():
        return ApiResult(
            status="complete",
            bytes=final_path.stat().st_size,
            record_count=int(checkpoint.get("count", 0)),
            first_created_utc=checkpoint.get("first_utc"),
            last_created_utc=checkpoint.get("last_utc"),
        )

    if final_path.exists() and not checkpoint:
        if not overwrite:
            raise FileExistsError(f"Output exists without checkpoint (use --overwrite): {final_path}")
        final_path.unlink()

    if not checkpoint:
        checkpoint = _new_checkpoint()
        final_path.touch()
        _save_checkpoint(checkpoint_path, checkpoint)
    else:
        committed = int(checkpoint.get("committed_bytes", 0))
        if not final_path.exists():
            checkpoint = _new_checkpoint()
            final_path.touch()
            _save_checkpoint(checkpoint_path, checkpoint)
        elif final_path.stat().st_size < committed:
            LOGGER.warning("Output shorter than checkpoint; restarting %s", target.filename)
            checkpoint = _new_checkpoint()
            final_path.write_bytes(b"")
            _save_checkpoint(checkpoint_path, checkpoint)
        elif final_path.stat().st_size > committed:
            LOGGER.warning(
                "Truncating uncommitted bytes from %s (%d -> %d)",
                target.filename,
                final_path.stat().st_size,
                committed,
            )
            with final_path.open("r+b") as handle:
                handle.truncate(committed)

    after = checkpoint.get("last_utc")
    recent_ids = set(str(value) for value in checkpoint.get("recent_ids", []))
    count = int(checkpoint.get("count", 0))
    first_utc = checkpoint.get("first_utc")
    committed = int(checkpoint.get("committed_bytes", 0))
    complete = False
    not_found = False

    compressor = zstd.ZstdCompressor(level=3)
    bar = tqdm(
        desc=f"{target.subreddit}/{target.kind}",
        unit=" rec",
        mininterval=10,
        disable=not progress,
    )
    try:
        with final_path.open("ab") as output:
            window_end: int | None = (
                int(after) + window_seconds if after is not None else None
            )
            while True:
                previous_after = after
                page = client.fetch_page(
                    target.subreddit,
                    target.kind,
                    after,
                    limit,
                    before=window_end,
                )
                if not page:
                    if window_end is None:
                        complete = True
                        break
                    now = int(time.time())
                    if window_end >= now:
                        complete = True
                        break
                    after = window_end
                    window_end += window_seconds
                    continue

                new_records: list[dict[str, Any]] = []
                page_ids: set[str] = set()
                page_timestamps: list[int] = []
                for record in page:
                    record_id = str(record.get("id", "")).strip()
                    if record_id:
                        if record_id in recent_ids or record_id in page_ids:
                            continue
                        page_ids.add(record_id)
                    timestamp = normalise_created_utc(record.get("created_utc"))
                    if timestamp is not None:
                        page_timestamps.append(timestamp)
                    new_records.append(record)

                max_page_ts = max(page_timestamps) if page_timestamps else None
                if new_records:
                    payload = b"".join(orjson.dumps(record) + b"\n" for record in new_records)
                    compressed = compressor.compress(payload)
                    output.write(compressed)
                    output.flush()
                    os.fsync(output.fileno())
                    committed += len(compressed)
                    count += len(new_records)
                    if page_timestamps:
                        page_first = min(page_timestamps)
                        first_utc = page_first if first_utc is None else min(int(first_utc), page_first)
                        after = max_page_ts
                    recent_ids = set(str(record.get("id", "")).strip() for record in page if record.get("id"))
                    checkpoint = {
                        "last_utc": after,
                        "count": count,
                        "first_utc": first_utc,
                        "recent_ids": sorted(recent_ids),
                        "committed_bytes": committed,
                        "complete": False,
                    }
                    _save_checkpoint(checkpoint_path, checkpoint)
                    bar.update(len(new_records))

                if window_end is None:
                    if isinstance(limit, int) and len(page) < limit:
                        complete = True
                        break
                    if max_page_ts is not None:
                        window_end = max_page_ts + window_seconds
                else:
                    if max_page_ts is not None and max_page_ts >= window_end:
                        after = window_end
                        window_end += window_seconds
                        continue
                    if isinstance(limit, int) and len(page) < limit:
                        after = window_end
                        window_end += window_seconds
                        continue
                if max_page_ts is None or (
                    previous_after is not None and max_page_ts <= int(previous_after)
                ):
                    LOGGER.warning(
                        "Pagination stalled for %s/%s at after=%s; stopping incomplete.",
                        target.subreddit,
                        target.kind,
                        previous_after,
                    )
                    complete = False
                    break
                if not new_records:
                    after = max_page_ts
                elif after is None:
                    after = max_page_ts
    finally:
        bar.close()

    checkpoint["complete"] = complete
    checkpoint["committed_bytes"] = committed
    _save_checkpoint(checkpoint_path, checkpoint)

    if complete and count == 0:
        not_found = True
        final_path.unlink(missing_ok=True)
        status = "not_found"
    elif complete:
        status = "complete"
    else:
        status = "partial"

    return ApiResult(
        status=status,
        bytes=final_path.stat().st_size if final_path.exists() else 0,
        record_count=count,
        first_created_utc=first_utc,
        last_created_utc=checkpoint.get("last_utc"),
    )


class DownloadManifest:
    """JSON manifest of completed, partial and missing downloads."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: dict[tuple[str, str], dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Load an existing manifest; ignore corrupt files with a warning."""
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Ignoring corrupt download manifest: %s", self.path.name)
            return
        for entry in payload.get("files", []):
            key = (str(entry.get("subreddit", "")).lower(), str(entry.get("type", "")))
            if all(key):
                self.entries[key] = entry

    def get(self, target: DownloadTarget) -> dict[str, Any] | None:
        """Return a manifest entry for a target, if present."""
        return self.entries.get((target.subreddit, target.kind))

    def upsert(self, entry: Mapping[str, Any]) -> None:
        """Insert or replace one manifest entry and save atomically."""
        key = (str(entry["subreddit"]).lower(), str(entry["type"]))
        self.entries[key] = dict(entry)
        self.save()

    def save(self) -> None:
        """Write the manifest atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": sorted(
                self.entries.values(),
                key=lambda entry: (str(entry.get("subreddit", "")), str(entry.get("type", ""))),
            ),
        }
        temporary = self.path.with_name(self.path.name + ".part")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def manifest_entry(
    target: DownloadTarget,
    route: str,
    path: Path,
    status: str,
    bytes_count: int,
    record_count: int | None,
    first_created_utc: int | None,
    last_created_utc: int | None,
) -> dict[str, Any]:
    """Build a manifest record with the required fields."""
    try:
        relative_path = str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        relative_path = str(path)
    return {
        "subreddit": target.subreddit,
        "type": target.kind,
        "route": route,
        "path": relative_path,
        "bytes": int(bytes_count),
        "record_count": record_count,
        "first_created_utc": first_created_utc,
        "last_created_utc": last_created_utc,
        "download_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": status,
    }


def complete_entry_exists(
    manifest: DownloadManifest,
    target: DownloadTarget,
    raw_dir: Path,
) -> bool:
    """Return True when the manifest says complete and the file matches."""
    entry = manifest.get(target)
    if not entry or entry.get("status") != "complete":
        return False
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return False
    expected = entry.get("bytes")
    return expected is None or path.stat().st_size == int(expected)


def build_api_client(
    cfg: Mapping[str, Any],
    section: Mapping[str, Any] | None = None,
    limiter: SharedRateLimiter | None = None,
) -> ArcticShiftClient:
    """Build a rate-limited Arctic Shift client from config.

    Pass one ``limiter`` to every worker client when running concurrently.
    """
    download_section = section or download_config(cfg)
    return ArcticShiftClient(
        base_url=str(download_section["arctic_shift_base_url"]),
        user_agent=str(download_section["user_agent"]),
        timeout=float(download_section.get("api_timeout_seconds", 60)),
        min_interval=float(download_section["api_min_interval_seconds"]),
        max_retries=int(download_section["api_max_retries"]),
        backoff_seconds=float(download_section["api_backoff_seconds"]),
        limiter=limiter,
    )


def estimate_counts(
    targets: Sequence[DownloadTarget],
    client: ArcticShiftClient,
) -> dict[DownloadTarget, int | None]:
    """Fetch aggregate record counts for planning, without downloading records."""
    counts: dict[DownloadTarget, int | None] = {}
    for target in targets:
        try:
            counts[target] = client.time_series_total(target.subreddit, target.kind)
        except (ApiNotFoundError, DownloadError):
            counts[target] = None
    return counts


def print_counts(counts: Mapping[DownloadTarget, int | None]) -> None:
    """Print aggregate record counts per target."""
    header = f"{'subreddit':<26} {'type':<13} {'records':>12}"
    print(header)
    print("-" * len(header))
    known = 0
    unknown = 0
    for target, count in counts.items():
        if count is None:
            unknown += 1
            text = "unavailable"
        else:
            known += count
            text = f"{count:,}"
        print(f"{target.subreddit:<26} {target.kind:<13} {text:>12}")
    print("-" * len(header))
    print(
        f"{'TOTAL':<26} {len(counts):>13} {known:>12,} known; "
        f"{unknown} unavailable"
    )


def build_plan(
    cfg: Mapping[str, Any],
    manifest: DownloadManifest,
    torrent: TorrentMeta | None,
    targets: Sequence[DownloadTarget],
    raw_dir: Path,
    force_api: bool = False,
) -> list[PlanItem]:
    """Build dry-run/download plan with torrent/API routes and sizes."""
    mapping: dict[DownloadTarget, tuple[TorrentFile, ...]] = {}
    if torrent is not None and not force_api:
        subreddits = {target.subreddit for target in targets}
        mapping = map_torrent_files(torrent, subreddits)

    plan: list[PlanItem] = []
    for target in targets:
        entry = manifest.get(target)
        if complete_entry_exists(manifest, target, raw_dir):
            path = Path(str(entry["path"]))
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            plan.append(
                PlanItem(
                    target=target,
                    route="skip",
                    size=path.stat().st_size,
                    status="complete",
                )
            )
            continue
        if target in mapping:
            files = mapping[target]
            plan.append(
                PlanItem(
                    target=target,
                    route="torrent",
                    size=sum(item.length for item in files),
                    files=files,
                )
            )
        else:
            plan.append(
                PlanItem(
                    target=target,
                    route="api",
                    size=None,
                    status="not_in_torrent" if torrent is not None and not force_api else "api",
                )
            )
        entry = manifest.get(target)
        if entry and entry.get("status") == "not_found":
            plan[-1].status = "not_found"
    return plan


def format_bytes(value: int | None) -> str:
    """Human-readable byte size."""
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"


def print_plan(plan: Sequence[PlanItem], free_bytes: int, raw_dir: Path) -> None:
    """Print the dry-run table and disk summary."""
    header = f"{'subreddit':<26} {'type':<13} {'route':<9} {'size':>12}  status"
    print(header)
    print("-" * len(header))
    known_total = 0
    unknown = 0
    for item in plan:
        size_text = format_bytes(item.size)
        if item.size is None:
            unknown += 1
        elif item.route != "skip":
            known_total += item.size
        print(
            f"{item.target.subreddit:<26} {item.target.kind:<13} "
            f"{item.route:<9} {size_text:>12}  {item.status}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<26} {len(plan):>13} {'':<9} {format_bytes(known_total):>12}  "
        f"known; {unknown} unknown; {len(plan)} file(s)"
    )
    print(f"Free disk on {raw_dir}: {format_bytes(free_bytes)}")


def _disk_guard(plan: Sequence[PlanItem], free_bytes: int, min_free_gb: float) -> tuple[bool, str]:
    """Check whether known downloads leave the requested free-space cushion."""
    required_known = sum(item.size or 0 for item in plan if item.route in ("torrent", "api"))
    unknown = any(item.route == "api" and item.size is None for item in plan)
    reserve = int(min_free_gb * 1024**3)
    available_after = free_bytes - required_known
    if available_after < reserve:
        return (
            False,
            f"Need at least {min_free_gb:.0f} GiB free after download; "
            f"free={format_bytes(free_bytes)}, known download={format_bytes(required_known)}, "
            f"after={format_bytes(available_after)}"
            + (" (API sizes unknown)" if unknown else ""),
        )
    if unknown and free_bytes < reserve:
        return False, f"API sizes are unknown and free space is below {min_free_gb:.0f} GiB."
    return True, "ok"


def _save_torrent_manifest(
    manifest: DownloadManifest,
    target: DownloadTarget,
    stats: Mapping[str, Any],
    raw_dir: Path,
) -> None:
    """Write a completed torrent manifest entry."""
    manifest.upsert(
        manifest_entry(
            target=target,
            route="torrent",
            path=raw_dir / target.filename,
            status="complete",
            bytes_count=int(stats["bytes"]),
            record_count=int(stats["record_count"]),
            first_created_utc=stats.get("first_created_utc"),
            last_created_utc=stats.get("last_created_utc"),
        )
    )


def _run_api_item(
    item: PlanItem,
    paths: Mapping[str, Path],
    client: ArcticShiftClient,
    api_limit: int | str,
    overwrite: bool,
    window_seconds: int,
) -> ApiResult:
    """Run one API target; used by the shared worker pool."""
    return download_api_target(
        item.target,
        paths["raw"],
        paths["checkpoint_dir"],
        client,
        limit=api_limit,
        overwrite=overwrite,
        window_seconds=window_seconds,
    )


def acquire(
    cfg: Mapping[str, Any],
    targets: Sequence[DownloadTarget],
    dry_run: bool,
    force_api: bool = False,
    overwrite: bool = False,
    min_free_gb: float | None = None,
    ignore_disk_check: bool = False,
    refresh_torrent: bool = False,
    workers: int = 3,
) -> list[PlanItem]:
    """Plan and optionally execute downloads for the selected targets.

    API targets run in a small thread pool. Every worker shares one
    :class:`SharedRateLimiter`, so the configured minimum interval is a global
    request rate rather than a per-worker rate.
    """
    workers = max(1, int(workers))
    section = download_config(cfg)
    paths = download_paths(cfg, section)
    manifest = DownloadManifest(paths["manifest"])
    free_bytes = shutil.disk_usage(paths["raw"]).free
    reserve_gb = float(section["min_free_gb"] if min_free_gb is None else min_free_gb)

    torrent: TorrentMeta | None = None
    torrent_path: Path | None = None
    if not force_api and aria2_available():
        torrent_cache = paths["torrent_dir"] / "academic_torrents_reddit.torrent"
        try:
            torrent_path = fetch_torrent(
                str(section["torrent_url"]),
                torrent_cache,
                str(section["user_agent"]),
                timeout=float(section.get("api_timeout_seconds", 60)),
                refresh=refresh_torrent,
            )
            torrent = load_torrent(torrent_path)
            LOGGER.info(
                "Torrent metadata: %s files, %s",
                len(torrent.files),
                format_bytes(torrent.total_bytes),
            )
        except DownloadError as exc:
            magnet = str(section.get("torrent_magnet", "")).strip()
            if magnet and (refresh_torrent or not torrent_cache.exists()):
                LOGGER.warning("%s Retrying via magnet metadata.", exc)
                try:
                    torrent_path = fetch_torrent_magnet(
                        magnet,
                        torrent_cache,
                        str(section["user_agent"]),
                        timeout=float(section.get("torrent_metadata_timeout_seconds", 600)),
                    )
                    torrent = load_torrent(torrent_path)
                    LOGGER.info(
                        "Torrent metadata: %s files, %s",
                        len(torrent.files),
                        format_bytes(torrent.total_bytes),
                    )
                except DownloadError as magnet_exc:
                    LOGGER.warning("%s Falling back to the API route.", magnet_exc)
                    torrent = None
                    torrent_path = None
            else:
                LOGGER.warning("%s Falling back to the API route.", exc)
                torrent = None
                torrent_path = None
    elif not aria2_available():
        LOGGER.warning("aria2c is not installed; using the API route.")
    elif force_api:
        LOGGER.info("--force-api requested; using the API route.")

    plan = build_plan(cfg, manifest, torrent, targets, paths["raw"], force_api=force_api)
    print_plan(plan, free_bytes, paths["raw"])
    if dry_run:
        return plan

    allowed, reason = _disk_guard(plan, free_bytes, reserve_gb)
    if not allowed:
        if not ignore_disk_check:
            raise SystemExit(
                f"Refusing to download: {reason}. Free space or choose a smaller subset with --only, "
                "or pass --ignore-disk-check after reviewing the risk."
            )
        LOGGER.warning("Disk guard overridden: %s", reason)

    active = [item for item in plan if item.route in ("torrent", "api")]
    torrent_items = [item for item in active if item.route == "torrent"]
    api_items = [item for item in active if item.route == "api"]

    if torrent_items and torrent is not None and torrent_path is not None:
        selections = {item.target: item.files for item in torrent_items}
        try:
            download_torrent_selection(
                torrent_path,
                torrent,
                selections,
                paths["torrent_dir"],
            )
            for item in torrent_items:
                try:
                    stats = finalise_torrent_target(
                        item.target,
                        item.files,
                        torrent,
                        paths["torrent_dir"],
                        paths["raw"],
                        overwrite=overwrite,
                    )
                    _save_torrent_manifest(manifest, item.target, stats, paths["raw"])
                    item.status = "complete"
                except Exception as exc:
                    LOGGER.error(
                        "Torrent finalisation failed for %s/%s: %s",
                        item.target.subreddit,
                        item.target.kind,
                        exc,
                    )
                    item.status = "error"
                    manifest.upsert(
                        manifest_entry(
                            item.target,
                            "torrent",
                            paths["raw"] / item.target.filename,
                            "error",
                            0,
                            None,
                            None,
                            None,
                        )
                    )
        except DownloadError as exc:
            LOGGER.error("Torrent route failed: %s", exc)
            for item in torrent_items:
                item.status = "error"
                item.route = "api"

    limiter = SharedRateLimiter(
        float(section["api_min_interval_seconds"])
    )
    client = build_api_client(cfg, section, limiter=limiter)
    raw_limit = section["api_limit"]
    api_limit: int | str = (
        "auto" if isinstance(raw_limit, str) and raw_limit.strip().lower() == "auto" else int(raw_limit)
    )
    api_targets = api_items + [item for item in torrent_items if item.status == "error"]
    LOGGER.info(
        "Starting %d API target(s) with %d worker(s); global limit %.2fs/request",
        len(api_targets),
        min(workers, max(1, len(api_targets))),
        limiter.min_interval,
    )
    blocked_error: ApiBlockedError | None = None
    executor = ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="radpath-download",
    )
    try:
        futures = {
            executor.submit(
                _run_api_item,
                item,
                paths,
                client,
                api_limit,
                overwrite,
                int(section.get("api_window_days", 90)) * 86_400,
            ): item
            for item in api_targets
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                item.status = result.status
                manifest.upsert(
                    manifest_entry(
                        item.target,
                        "api",
                        paths["raw"] / item.target.filename,
                        result.status,
                        result.bytes,
                        result.record_count,
                        result.first_created_utc,
                        result.last_created_utc,
                    )
                )
                LOGGER.info(
                    "API %s/%s: %s (%d records)",
                    item.target.subreddit,
                    item.target.kind,
                    result.status,
                    result.record_count,
                )
            except ApiBlockedError as exc:
                blocked_error = exc
                LOGGER.error(
                    "API blocked repeatedly for %s/%s; stopping downloads: %s",
                    item.target.subreddit,
                    item.target.kind,
                    exc,
                )
                for pending in futures:
                    if pending is not future:
                        pending.cancel()
            except FileExistsError as exc:
                LOGGER.error("%s", exc)
                item.status = "error"
            except Exception as exc:
                LOGGER.error(
                    "API download failed for %s/%s: %s",
                    item.target.subreddit,
                    item.target.kind,
                    exc,
                )
                item.status = "error"
    finally:
        executor.shutdown(wait=True, cancel_futures=blocked_error is not None)
    if blocked_error is not None:
        raise SystemExit(2)

    print_plan(plan, shutil.disk_usage(paths["raw"]).free, paths["raw"])
    return plan
