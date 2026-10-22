"""Stream zstandard-compressed Reddit JSONL into cleaned Parquet.

Raw files are processed line by line; a whole dump is never loaded into
memory. Authors are cleaned (bots/deleted accounts) and then irreversibly
hashed with the salt from ``.env`` before anything is written to disk.

Output layout::

    data/interim/submissions/<source-stem>.parquet
    data/interim/comments/<source-stem>.parquet
"""

from __future__ import annotations

import hashlib
import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Mapping

import orjson
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

from radpath.clean import CleanRules, is_deleted_value, compose_text, drop_reason, bot_rule

LOGGER = logging.getLogger(__name__)

CONTRIBUTION_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("author_hash", pa.string()),
        pa.field("subreddit", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("created_utc", pa.int64()),
        pa.field("title", pa.string()),
        pa.field("selftext", pa.string()),
        pa.field("body", pa.string()),
        pa.field("text", pa.string()),
        pa.field("has_text", pa.bool_()),
        pa.field("score", pa.int64()),
        pa.field("num_comments", pa.int64()),
        pa.field("link_id", pa.string()),
        pa.field("parent_id", pa.string()),
    ]
)

CONTRIBUTION_COLUMNS: tuple[str, ...] = tuple(CONTRIBUTION_SCHEMA.names)
KINDS: tuple[str, ...] = ("submissions", "comments")

MAX_WINDOW_SIZE = 2**31


@dataclass(frozen=True)
class FileSpec:
    """A raw ``.zst`` file and the record kind it contains."""

    path: Path
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")


@dataclass
class IngestStats:
    """Aggregate ingestion counters. Never stores raw usernames.

    ``coverage`` stores aggregate first/last timestamps per subreddit.
    ``seen_authors`` and the bot sets store salted hashes only, so distinct
    author counts can be reported without retaining or logging usernames.
    """

    files: int = 0
    total: int = 0
    kept: int = 0
    malformed_json: int = 0
    dropped_missing_field: int = 0
    dropped_bad_timestamp: int = 0
    dropped_deleted_author: int = 0
    dropped_bot_author: int = 0
    kept_removed_text: int = 0
    kept_deleted_text: int = 0
    kept_empty_text: int = 0
    dropped_subreddit: int = 0
    coverage: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    seen_authors: set[str] = field(default_factory=set)
    bot_suffix_authors: set[str] = field(default_factory=set)
    bot_explicit_authors: set[str] = field(default_factory=set)

    _COUNTERS: ClassVar[tuple[str, ...]] = (
        "files",
        "total",
        "kept",
        "malformed_json",
        "dropped_missing_field",
        "dropped_bad_timestamp",
        "dropped_deleted_author",
        "dropped_bot_author",
        "kept_removed_text",
        "kept_deleted_text",
        "kept_empty_text",
        "dropped_subreddit",
    )

    def add(self, other: "IngestStats") -> None:
        """Add another stats object's counters, coverage and author sets."""
        for name in self._COUNTERS:
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for subreddit, (start, end, count) in other.coverage.items():
            if subreddit in self.coverage:
                old_start, old_end, old_count = self.coverage[subreddit]
                self.coverage[subreddit] = (
                    min(old_start, start),
                    max(old_end, end),
                    old_count + count,
                )
            else:
                self.coverage[subreddit] = (start, end, count)
        self.seen_authors.update(other.seen_authors)
        self.bot_suffix_authors.update(other.bot_suffix_authors)
        self.bot_explicit_authors.update(other.bot_explicit_authors)

    def as_dict(self) -> dict[str, int]:
        """Return counters as a plain dictionary."""
        return {name: int(getattr(self, name)) for name in self._COUNTERS}

    @property
    def bot_suffix_share(self) -> float | None:
        """Share of distinct non-deleted authors dropped by the suffix rule."""
        if not self.seen_authors:
            return None
        return len(self.bot_suffix_authors) / len(self.seen_authors)

    def summary(self) -> str:
        """Compact human-readable summary (aggregate only)."""
        share = self.bot_suffix_share
        share_text = f"{share:.2%}" if share is not None else "n/a"
        return (
            f"files={self.files} read={self.total} kept={self.kept} "
            f"malformed={self.malformed_json} missing={self.dropped_missing_field} "
            f"bad_ts={self.dropped_bad_timestamp} "
            f"deleted_author={self.dropped_deleted_author} "
            f"bot_author={self.dropped_bot_author} "
            f"removed_text={self.kept_removed_text} "
            f"deleted_text={self.kept_deleted_text} "
            f"empty_text={self.kept_empty_text} "
            f"other_subreddit={self.dropped_subreddit} "
            f"distinct_authors={len(self.seen_authors)} "
            f"bot_suffix_authors={len(self.bot_suffix_authors)} ({share_text}) "
            f"bot_explicit_authors={len(self.bot_explicit_authors)}"
        )


def hash_author(author: str, salt: str) -> str:
    """Return a salted SHA-256 hash of a username.

    Reddit usernames are case-insensitive, so they are casefolded before
    hashing to ensure the same account maps to one identifier across dumps.

    Args:
        author: Raw username. Never stored or logged by callers.
        salt: Secret salt from ``RADPATH_SALT``.

    Returns:
        Lower-case hexadecimal SHA-256 digest.
    """
    if not salt:
        raise ValueError("A non-empty salt is required to hash authors.")
    payload = f"{salt}\x1f{str(author).strip().casefold()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalise_created_utc(value: Any) -> int | None:
    """Coerce an inconsistent epoch timestamp to integer UTC seconds.

    Arctic Shift / Pushshift dumps mix strings and numbers across years.
    Returns None for missing, non-finite, or unparseable values.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return int(number)


def _as_int(value: Any, default: int = 0) -> int:
    """Best-effort integer coercion for score-like fields."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return int(number) if math.isfinite(number) else default


def _as_text(value: Any) -> str:
    """Coerce an optional text field to a stripped string."""
    if value is None:
        return ""
    return str(value).strip()


def prepare_record(
    record: Mapping[str, Any],
    kind: str,
    salt: str,
    rules: CleanRules | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalise one raw record and hash its author.

    Args:
        record: Parsed JSON object from a raw dump.
        kind: ``"submissions"`` or ``"comments"``.
        salt: Username hashing salt.
        rules: Cleaning rules; defaults are used when omitted.

    Returns:
        ``(row, None)`` when kept, otherwise ``(None, reason)`` where reason
        is one of ``missing_field``, ``bad_timestamp``, ``deleted_author`` or
        ``bot_author``. Known authors with removed/deleted/empty content are
        kept with ``has_text = False``.
    """
    active_rules = rules or CleanRules()
    record_id = record.get("id")
    subreddit = record.get("subreddit")
    if not record_id or not subreddit:
        return None, "missing_field"

    created_utc = normalise_created_utc(record.get("created_utc"))
    if created_utc is None:
        return None, "bad_timestamp"

    author = record.get("author")
    title = "" if is_deleted_value(record.get("title"), active_rules) else _as_text(record.get("title"))
    selftext = (
        "" if is_deleted_value(record.get("selftext"), active_rules) else _as_text(record.get("selftext"))
    )
    body = "" if is_deleted_value(record.get("body"), active_rules) else _as_text(record.get("body"))

    reason = drop_reason(author, "comment" if kind == "comments" else "submission", title, selftext, body, active_rules)
    if reason:
        return None, reason

    text = compose_text("comment" if kind == "comments" else "submission", title, selftext, body, active_rules)
    row = {
        "id": str(record_id),
        "author_hash": hash_author(str(author), salt),
        "subreddit": str(subreddit).strip().lower(),
        "kind": "comment" if kind == "comments" else "submission",
        "created_utc": created_utc,
        "title": title,
        "selftext": selftext,
        "body": body,
        "text": text,
        "has_text": bool(text.strip()),
        "score": _as_int(record.get("score")),
        "num_comments": _as_int(record.get("num_comments")),
        "link_id": _as_text(record.get("link_id")) or None,
        "parent_id": _as_text(record.get("parent_id")) or None,
    }
    return row, None


def iter_jsonl(path: Path) -> Iterator[dict[str, Any] | None]:
    """Stream one ``.zst`` JSONL file, yielding parsed objects or None.

    Uses a streaming zstandard reader with an enlarged window size, so
    memory use stays proportional to a single line.
    """
    decompressor = zstd.ZstdDecompressor(max_window_size=MAX_WINDOW_SIZE)
    with path.open("rb") as raw:
        with decompressor.stream_reader(raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield orjson.loads(line)
                except orjson.JSONDecodeError:
                    yield None


def _kept_text_case(record: Mapping[str, Any]) -> str:
    """Classify a kept record with no usable text for aggregate logging."""
    values = {
        str(record.get(name)).strip().lower()
        for name in ("title", "selftext", "body")
        if record.get(name) is not None
    }
    if "[removed]" in values:
        return "removed"
    if "[deleted]" in values:
        return "deleted"
    return "empty"


def ingest_file(
    spec: FileSpec,
    out_dir: Path,
    salt: str,
    rules: CleanRules | None = None,
    batch_size: int = 20_000,
    subreddit_filter: set[str] | None = None,
    overwrite: bool = True,
) -> IngestStats:
    """Stream one raw file into a cleaned Parquet batch file.

    Args:
        spec: Raw file and its kind.
        out_dir: Directory for the kind, e.g. ``data/interim/submissions``.
        salt: Secret hashing salt.
        rules: Cleaning rules.
        batch_size: Rows buffered before each Parquet write.
        subreddit_filter: Lower-case subreddits to keep; None keeps all.
        overwrite: Replace an existing output file.

    Returns:
        Aggregate counters and observed coverage for the file.

    Raises:
        FileNotFoundError: If the input file is missing.
        FileExistsError: If output exists and ``overwrite`` is false.
    """
    if not spec.path.exists():
        raise FileNotFoundError(f"Raw file not found: {spec.path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{spec.path.stem}.parquet"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists (use --overwrite): {out_path}")

    active_rules = rules or CleanRules()
    stats = IngestStats(files=1)
    writer = pq.ParquetWriter(out_path, CONTRIBUTION_SCHEMA, compression="zstd")
    batch: list[dict[str, Any]] = []
    wrote_any = False

    def flush() -> None:
        nonlocal batch, wrote_any
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CONTRIBUTION_SCHEMA))
            wrote_any = True
            batch = []

    try:
        for record in iter_jsonl(spec.path):
            stats.total += 1
            if record is None:
                stats.malformed_json += 1
                continue
            subreddit = str(record.get("subreddit", "")).strip().lower()
            if (
                subreddit
                and subreddit_filter is not None
                and subreddit not in subreddit_filter
            ):
                stats.dropped_subreddit += 1
                continue
            raw_author = record.get("author")
            author_hash: str | None = None
            if not is_deleted_value(raw_author, active_rules):
                author_hash = hash_author(str(raw_author), salt)
                stats.seen_authors.add(author_hash)
            row, reason = prepare_record(record, spec.kind, salt, active_rules)
            if row is None:
                counter = {
                    "missing_field": "dropped_missing_field",
                    "bad_timestamp": "dropped_bad_timestamp",
                    "deleted_author": "dropped_deleted_author",
                    "bot_author": "dropped_bot_author",
                }.get(reason or "", "dropped_missing_field")
                setattr(stats, counter, getattr(stats, counter) + 1)
                if reason == "bot_author" and author_hash is not None:
                    if bot_rule(raw_author, active_rules) == "explicit":
                        stats.bot_explicit_authors.add(author_hash)
                    else:
                        stats.bot_suffix_authors.add(author_hash)
                continue
            stats.kept += 1
            if not row["has_text"]:
                case = _kept_text_case(record)
                setattr(stats, f"kept_{case}_text", getattr(stats, f"kept_{case}_text") + 1)
            subreddit_name = str(row["subreddit"])
            timestamp = int(row["created_utc"])
            if subreddit_name in stats.coverage:
                start, end, count = stats.coverage[subreddit_name]
                stats.coverage[subreddit_name] = (
                    min(start, timestamp),
                    max(end, timestamp),
                    count + 1,
                )
            else:
                stats.coverage[subreddit_name] = (timestamp, timestamp, 1)
            batch.append(row)
            if len(batch) >= batch_size:
                flush()
        flush()
        if not wrote_any:
            writer.write_table(pa.Table.from_pylist([], schema=CONTRIBUTION_SCHEMA))
    finally:
        writer.close()

    LOGGER.info("ingested %s -> %s (%s)", spec.path.name, out_path.name, stats.summary())
    return stats


def ingest_files(
    specs: Iterable[FileSpec],
    interim_dir: Path,
    salt: str,
    rules: CleanRules | None = None,
    batch_size: int = 20_000,
    subreddit_filter: set[str] | None = None,
    overwrite: bool = True,
    bot_warn_fraction: float = 0.01,
) -> IngestStats:
    """Ingest several raw files into ``interim_dir/<kind>/``.

    Args:
        specs: Raw files with their kinds.
        interim_dir: Root interim directory from config.
        salt: Secret hashing salt.
        rules: Cleaning rules.
        batch_size: Rows per Parquet write.
        subreddit_filter: Lower-case subreddits to keep.
        overwrite: Replace existing outputs.
        bot_warn_fraction: Warn when the suffix rule drops more than this
            share of distinct non-deleted authors.

    Returns:
        Combined counters and coverage across files.
    """
    combined = IngestStats()
    for spec in specs:
        file_stats = ingest_file(
            spec=spec,
            out_dir=interim_dir / spec.kind,
            salt=salt,
            rules=rules,
            batch_size=batch_size,
            subreddit_filter=subreddit_filter,
            overwrite=overwrite,
        )
        combined.add(file_stats)
    LOGGER.info(
        "distinct authors: seen=%d bot_suffix=%d bot_explicit=%d",
        len(combined.seen_authors),
        len(combined.bot_suffix_authors),
        len(combined.bot_explicit_authors),
    )
    share = combined.bot_suffix_share
    if share is not None and share > bot_warn_fraction:
        LOGGER.warning(
            "bot suffix rule dropped %.2f%% of distinct authors "
            "(threshold %.2f%%); review the suffix rule",
            share * 100,
            bot_warn_fraction * 100,
        )
    return combined


def _empty_contributions() -> pl.DataFrame:
    """Return an empty DataFrame with the contribution schema."""
    integer_columns = {"created_utc", "score", "num_comments"}
    return pl.DataFrame(
        {
            name: pl.Series(
                name,
                [],
                dtype=(
                    pl.Int64
                    if name in integer_columns
                    else pl.Boolean
                    if name == "has_text"
                    else pl.String
                ),
            )
            for name in CONTRIBUTION_COLUMNS
        }
    )


def load_contributions(interim_dir: Path) -> pl.DataFrame:
    """Load all interim Parquet files into one contribution table.

    Args:
        interim_dir: Root interim directory containing ``submissions/`` and
            ``comments/`` subdirectories.

    Returns:
        Concatenated DataFrame sorted by author and timestamp. Returns an
        empty frame when no interim files exist yet.
    """
    paths: list[str] = []
    for kind in KINDS:
        paths.extend(str(path) for path in sorted((interim_dir / kind).glob("*.parquet")))
    if not paths:
        return _empty_contributions()
    frame = pl.concat(
        [pl.read_parquet(path) for path in paths],
        how="vertical_relaxed",
    )
    if "has_text" not in frame.columns:
        frame = frame.with_columns(
            (pl.col("text").str.strip_chars().str.len_chars() > 0).alias("has_text")
        )
    return frame.with_columns(
        pl.col("subreddit").str.to_lowercase(),
        pl.col("created_utc").cast(pl.Int64),
        pl.col("score").cast(pl.Int64),
        pl.col("num_comments").cast(pl.Int64),
        pl.col("has_text").cast(pl.Boolean),
    ).sort(["author_hash", "created_utc", "id"])
