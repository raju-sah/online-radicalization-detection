"""Leakage-safe feature construction for Task A and Task B.

Every feature for a user is computed only from records inside that user's
feature window:

* Task A: target-community records with ``t0 <= created_utc <= cutoff_a``.
* Task B: gateway-community records with
  ``t0 - pre_entry_days <= created_utc < t0`` (``created_utc <= cutoff_b``).

Features are aggregate behaviour counts and simple text statistics. Raw text is
returned only as an in-memory per-user string for TF-IDF; it is never written
to disk by this module.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import polars as pl

SECONDS_PER_DAY = 86_400
_HOUR = 3_600


def window_records(
    contributions: pl.DataFrame,
    cohort: pl.DataFrame,
    subreddits: Iterable[str],
    task: str,
    pre_entry_days: int = 180,
) -> pl.DataFrame:
    """Return the deduplicated records inside each cohort user's window.

    Args:
        contributions: Cleaned contribution table.
        cohort: Cohort frame with ``author_hash``, ``t0``, ``cutoff_a``,
            ``cutoff_b``.
        subreddits: Subreddits to draw records from (targets for A, gateways
            for B).
        task: ``"a"`` or ``"b"``.
        pre_entry_days: Task B look-back length.

    Returns:
        Contribution rows joined with ``t0`` and the task cutoff.
    """
    if task not in {"a", "b"}:
        raise ValueError("task must be 'a' or 'b'")
    names = [name.strip().lower() for name in subreddits]
    cutoff_col = "cutoff_a" if task == "a" else "cutoff_b"
    frame = (
        contributions.with_columns(pl.col("subreddit").str.to_lowercase())
        .filter(pl.col("subreddit").is_in(names))
        .sort(["author_hash", "created_utc", "id"])
        .unique(subset=["kind", "id"], keep="first", maintain_order=True)
        .join(
            cohort.select("author_hash", "t0", pl.col(cutoff_col).alias("cutoff")),
            on="author_hash",
            how="inner",
        )
        .filter(pl.col("created_utc") <= pl.col("cutoff"))
    )
    if task == "a":
        frame = frame.filter(pl.col("created_utc") >= pl.col("t0"))
    else:
        frame = frame.filter(
            pl.col("created_utc") >= pl.col("t0") - pre_entry_days * SECONDS_PER_DAY
        )
    return frame


def _entropy(counts: Iterable[int]) -> float:
    values = [c for c in counts if c > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum(c / total * math.log2(c / total) for c in values)


def behaviour_features(
    records: pl.DataFrame,
    users: pl.DataFrame,
    prefix: str,
    first_n: int | None = None,
    window_seconds: int | None = None,
) -> pl.DataFrame:
    """Aggregate behaviour features over in-window records.

    Args:
        records: Output of :func:`window_records`.
        users: Frame with one ``author_hash`` row per user to keep (users with
            no records get zeros).
        prefix: Column prefix, e.g. ``"a"`` or ``"b"``.
        first_n: If set, ``time_to_nth`` is the seconds from the first window
            record to the Nth; users who never reach N get ``window_seconds``.
        window_seconds: Censoring value for ``time_to_nth``.

    Returns:
        One row per user in ``users`` with numeric feature columns.
    """
    recs = records.sort(["author_hash", "created_utc"]).with_columns(
        pl.col("created_utc").diff().over("author_hash").alias("_gap"),
        ((pl.col("created_utc") // _HOUR) % 24).alias("_hour"),
        (((pl.col("created_utc") // SECONDS_PER_DAY) + 3) % 7 >= 5).alias("_weekend"),
        (pl.col("kind") == "comment").alias("_is_comment"),
        pl.col("parent_id").fill_null("").str.starts_with("t1_").alias("_reply_comment"),
        (~pl.col("has_text")).alias("_no_text"),
        (pl.col("created_utc") // SECONDS_PER_DAY).alias("_day"),
    )
    base = recs.group_by("author_hash").agg(
        pl.len().alias("n"),
        pl.col("_is_comment").mean().alias("comment_share"),
        pl.col("_day").n_unique().alias("active_days"),
        pl.col("_gap").mean().alias("gap_mean"),
        pl.col("_gap").median().alias("gap_median"),
        pl.col("_gap").std().alias("gap_std"),
        pl.col("_weekend").mean().alias("weekend_share"),
        pl.col("link_id").drop_nulls().n_unique().alias("n_threads"),
        pl.col("subreddit").n_unique().alias("n_subreddits"),
        pl.col("_reply_comment").filter(pl.col("_is_comment")).mean().alias("reply_to_comment_share"),
        pl.col("score").mean().alias("score_mean"),
        pl.col("score").median().alias("score_median"),
        pl.col("_no_text").mean().alias("no_text_share"),
        pl.col("created_utc").min().alias("_first"),
        pl.col("created_utc").max().alias("_last"),
        pl.col("created_utc").sort().slice(first_n - 1, 1).first().alias("_nth")
        if first_n
        else pl.lit(None).alias("_nth"),
    )
    hours = (
        recs.group_by("author_hash", "_hour")
        .len()
        .group_by("author_hash")
        .agg(pl.col("len").alias("_hc"))
        .with_columns(
            pl.col("_hc")
            .map_elements(_entropy, return_dtype=pl.Float64)
            .alias("hour_entropy")
        )
        .select("author_hash", "hour_entropy")
    )
    out = base.join(hours, on="author_hash", how="left")
    if first_n:
        censor = float(window_seconds or 0)
        out = out.with_columns(
            pl.when(pl.col("_nth").is_not_null())
            .then(pl.col("_nth") - pl.col("_first"))
            .otherwise(pl.lit(censor))
            .cast(pl.Float64)
            .alias("time_to_nth"),
            pl.col("_nth").is_not_null().cast(pl.Int64).alias("reached_n"),
        )
    out = out.drop("_first", "_last", "_nth")
    out = users.select("author_hash").join(out, on="author_hash", how="left")
    numeric = [c for c in out.columns if c != "author_hash"]
    out = out.with_columns(pl.col(numeric).cast(pl.Float64).fill_null(0.0).fill_nan(0.0))
    return out.rename({c: f"{prefix}_{c}" for c in numeric})


def text_stat_features(
    records: pl.DataFrame, users: pl.DataFrame, prefix: str
) -> pl.DataFrame:
    """Length and VADER sentiment statistics over has_text=true records."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    texty = records.filter(pl.col("has_text"))
    scores = [analyzer.polarity_scores(t or "")["compound"] for t in texty["text"]]
    texty = texty.with_columns(
        pl.Series("_sent", scores, dtype=pl.Float64),
        pl.col("text").fill_null("").str.len_chars().alias("_len"),
    )
    agg = texty.group_by("author_hash").agg(
        pl.col("_len").mean().alias("len_mean"),
        pl.col("_len").max().alias("len_max"),
        pl.col("_len").median().alias("len_median"),
        pl.col("_sent").mean().alias("sent_mean"),
        pl.col("_sent").min().alias("sent_min"),
    )
    out = users.select("author_hash").join(agg, on="author_hash", how="left")
    numeric = [c for c in out.columns if c != "author_hash"]
    out = out.with_columns(pl.col(numeric).cast(pl.Float64).fill_null(0.0))
    return out.rename({c: f"{prefix}_{c}" for c in numeric})


def user_texts(records: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    """Concatenate in-window text per user (has_text=true only, in memory)."""
    agg = (
        records.filter(pl.col("has_text"))
        .sort(["author_hash", "created_utc"])
        .group_by("author_hash", maintain_order=True)
        .agg(pl.col("text").fill_null("").str.join(" \n ").alias("doc"))
    )
    return users.select("author_hash").join(agg, on="author_hash", how="left").with_columns(
        pl.col("doc").fill_null("")
    )


def gateway_features(
    records: pl.DataFrame,
    users: pl.DataFrame,
    gateways: Iterable[str],
) -> pl.DataFrame:
    """Task B extras: per-gateway counts and recency of last gateway activity."""
    names = [g.strip().lower() for g in gateways]
    counts = (
        records.group_by("author_hash", "subreddit").len()
        .pivot(on="subreddit", index="author_hash", values="len")
    )
    for name in names:
        if name not in counts.columns:
            counts = counts.with_columns(pl.lit(0).alias(name))
    counts = counts.select("author_hash", *names).rename({n: f"b_gw_{n}" for n in names})
    recency = records.group_by("author_hash").agg(
        ((pl.col("t0").first() - pl.col("created_utc").max()) / SECONDS_PER_DAY).alias(
            "b_days_since_last_gateway"
        )
    )
    out = users.select("author_hash").join(counts, on="author_hash", how="left").join(
        recency, on="author_hash", how="left"
    )
    numeric = [c for c in out.columns if c != "author_hash"]
    return out.with_columns(pl.col(numeric).cast(pl.Float64).fill_null(0.0))


def build_task_features(
    contributions: pl.DataFrame,
    cohort: pl.DataFrame,
    task: str,
    targets: Iterable[str],
    gateways: Iterable[str],
    first_n: int = 5,
    feature_days: int = 14,
    pre_entry_days: int = 180,
    with_text_stats: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build behavioural features and per-user text for one task.

    Args:
        contributions: Cleaned contributions.
        cohort: Cohort frame restricted to the users to featurise.
        task: ``"a"`` or ``"b"``.
        targets: Target subreddits.
        gateways: Gateway subreddits.
        first_n: Task A N for time-to-Nth.
        feature_days: Task A window cap in days.
        pre_entry_days: Task B look-back.
        with_text_stats: Compute VADER/length features (slower).

    Returns:
        ``(features, texts)`` keyed by ``author_hash``.
    """
    sources = targets if task == "a" else gateways
    recs = window_records(contributions, cohort, sources, task, pre_entry_days)
    feats = behaviour_features(
        recs,
        cohort,
        task,
        first_n=first_n if task == "a" else None,
        window_seconds=feature_days * SECONDS_PER_DAY,
    )
    if with_text_stats:
        feats = feats.join(text_stat_features(recs, cohort, task), on="author_hash")
    if task == "b":
        feats = feats.join(gateway_features(recs, cohort, gateways), on="author_hash")
    return feats, user_texts(recs, cohort)


def feature_columns(frame: pl.DataFrame) -> list[str]:
    """Numeric feature columns (everything except the key)."""
    return [c for c in frame.columns if c != "author_hash"]


def activity_count_columns(task: str) -> list[str]:
    """Columns used by the activity-count-only baseline."""
    return [f"{task}_n"]


def describe(frame: pl.DataFrame) -> Mapping[str, float]:
    """Aggregate column means, safe to log."""
    return {c: float(frame[c].mean() or 0.0) for c in feature_columns(frame)}
