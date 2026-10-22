"""Per-subreddit observation coverage.

Coverage is the interval between the first and last cleaned record observed
for a subreddit. It is written during ingestion and used to decide whether a
follow-up window is fully observed. Manual ``coverage_overrides`` in the
config can shorten or extend a window (for example a known ban date).

Coverage files contain only subreddit names, aggregate counts and timestamps;
no user identifiers are stored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import polars as pl

COVERAGE_SCHEMA: dict[str, Any] = {
    "subreddit": pl.String,
    "start_utc": pl.Int64,
    "end_utc": pl.Int64,
    "n_records": pl.Int64,
    "overridden": pl.Boolean,
}
COVERAGE_COLUMNS: tuple[str, ...] = tuple(COVERAGE_SCHEMA)


def parse_utc(value: Any) -> int:
    """Parse epoch seconds or an ISO-8601 timestamp into epoch seconds.

    Naive ISO timestamps are interpreted as UTC.
    """
    if isinstance(value, bool):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Empty UTC timestamp")
    try:
        return int(float(text))
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid UTC timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def empty_coverage() -> pl.DataFrame:
    """Return an empty coverage frame with the standard schema."""
    return pl.DataFrame(schema=COVERAGE_SCHEMA)


def coverage_from_contributions(
    contributions: pl.DataFrame,
    subreddits: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Compute observed coverage from a cleaned contribution table.

    Args:
        contributions: Frame with ``subreddit`` and ``created_utc``.
        subreddits: Optional lower-case names to keep.

    Returns:
        One row per subreddit with first/last timestamps and record count.
    """
    missing = {"subreddit", "created_utc"} - set(contributions.columns)
    if missing:
        raise ValueError(f"Missing coverage columns: {sorted(missing)}")
    if contributions.height == 0:
        return empty_coverage()

    frame = contributions.select(
        pl.col("subreddit").str.to_lowercase(),
        pl.col("created_utc").cast(pl.Int64),
    )
    if subreddits is not None:
        names = {str(name).strip().lower() for name in subreddits}
        frame = frame.filter(pl.col("subreddit").is_in(list(names)))

    coverage = frame.group_by("subreddit").agg(
        pl.col("created_utc").min().alias("start_utc"),
        pl.col("created_utc").max().alias("end_utc"),
        pl.len().cast(pl.Int64).alias("n_records"),
    )
    return coverage.with_columns(pl.lit(False).alias("overridden")).select(
        COVERAGE_COLUMNS
    ).sort("subreddit")


def normalise_coverage(coverage: Any) -> pl.DataFrame:
    """Coerce a coverage frame or mapping into the standard frame schema.

    Mapping values may be ``(start, end)`` tuples/lists, mappings with
    ``start_utc``/``end_utc`` keys, or source frames.
    """
    if coverage is None:
        return empty_coverage()
    if isinstance(coverage, pl.DataFrame):
        frame = coverage
        if frame.height == 0:
            return empty_coverage()
        if "subreddit" not in frame.columns:
            raise ValueError("Coverage frame must contain a subreddit column.")
        rows: list[dict[str, Any]] = []
        for row in frame.iter_rows(named=True):
            rows.append(
                {
                    "subreddit": str(row["subreddit"]).strip().lower(),
                    "start_utc": (
                        None if row["start_utc"] is None else parse_utc(row["start_utc"])
                    ),
                    "end_utc": (
                        None if row["end_utc"] is None else parse_utc(row["end_utc"])
                    ),
                    "n_records": int(row.get("n_records") or 0),
                    "overridden": bool(row.get("overridden", False)),
                }
            )
        return pl.DataFrame(rows, schema=COVERAGE_SCHEMA).sort("subreddit")

    rows = []
    for subreddit, value in dict(coverage).items():
        if isinstance(value, Mapping):
            start = value.get("start_utc")
            end = value.get("end_utc")
            n_records = int(value.get("n_records") or 0)
        else:
            start, end = value[0], value[1]
            n_records = int(value[2]) if len(value) > 2 else 0
        if start is None or end is None:
            raise ValueError(f"Coverage for {subreddit!r} needs start_utc and end_utc")
        rows.append(
            {
                "subreddit": str(subreddit).strip().lower(),
                "start_utc": parse_utc(start),
                "end_utc": parse_utc(end),
                "n_records": n_records,
                "overridden": False,
            }
        )
    return pl.DataFrame(rows, schema=COVERAGE_SCHEMA).sort("subreddit")


def coverage_ranges(coverage: Any) -> dict[str, tuple[int, int]]:
    """Return ``{subreddit: (start_utc, end_utc)}`` from any coverage input."""
    frame = normalise_coverage(coverage)
    return {
        str(row["subreddit"]): (int(row["start_utc"]), int(row["end_utc"]))
        for row in frame.iter_rows(named=True)
        if row["start_utc"] is not None and row["end_utc"] is not None
    }


def apply_coverage_overrides(
    coverage: Any,
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> pl.DataFrame:
    """Apply manual start/end overrides on top of observed coverage.

    Overrides may add a subreddit that has no observed records. Both ends are
    required for a new subreddit; for an observed one a single end may be
    supplied and the other is retained.
    """
    frame = normalise_coverage(coverage)
    if not overrides:
        return frame

    rows = {str(row["subreddit"]): dict(row) for row in frame.iter_rows(named=True)}
    for subreddit, override in overrides.items():
        name = str(subreddit).strip().lower()
        if not isinstance(override, Mapping):
            raise ValueError(f"coverage_overrides.{subreddit} must be a mapping")
        current = rows.get(name, {"subreddit": name, "n_records": 0, "overridden": False})
        start = override.get("start_utc", current.get("start_utc"))
        end = override.get("end_utc", current.get("end_utc"))
        if start is None or end is None:
            raise ValueError(
                f"coverage_overrides.{subreddit} needs start_utc and end_utc "
                "for a subreddit with no observed coverage"
            )
        current.update(
            {
                "subreddit": name,
                "start_utc": parse_utc(start),
                "end_utc": parse_utc(end),
                "n_records": int(current.get("n_records") or 0),
                "overridden": True,
            }
        )
        rows[name] = current
    return pl.DataFrame(list(rows.values()), schema=COVERAGE_SCHEMA).sort("subreddit")


def write_coverage(
    path: Path,
    coverage: Any,
    subreddits: Iterable[str] | None = None,
) -> Path:
    """Write coverage to ``path`` and return the path.

    When ``subreddits`` is supplied, only those names are written; configured
    subreddits with no kept records still appear with null timestamps.
    """
    frame = normalise_coverage(coverage)
    if subreddits is not None:
        names = sorted({str(name).strip().lower() for name in subreddits})
        observed = frame.filter(pl.col("subreddit").is_in(names))
        missing = [
            name
            for name in names
            if observed.filter(pl.col("subreddit") == name).height == 0
        ]
        if missing:
            frame = pl.concat(
                [
                    observed,
                    pl.DataFrame(
                        {
                            "subreddit": missing,
                            "start_utc": [None] * len(missing),
                            "end_utc": [None] * len(missing),
                            "n_records": [0] * len(missing),
                            "overridden": [False] * len(missing),
                        },
                        schema=COVERAGE_SCHEMA,
                    ),
                ],
                how="vertical",
            ).sort("subreddit")
        else:
            frame = observed
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)
    return path


def load_coverage(interim_dir: Path) -> pl.DataFrame:
    """Load ``<interim>/coverage.parquet`` or return an empty frame."""
    path = Path(interim_dir) / "coverage.parquet"
    if not path.exists():
        return empty_coverage()
    return normalise_coverage(pl.read_parquet(path))
