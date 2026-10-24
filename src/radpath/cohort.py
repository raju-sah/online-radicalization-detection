"""Cohort construction, feature windows, follow-up eligibility and labels.

Two tasks share one cohort of users (first target-community contribution
defines entry time ``t0``) but use different feature windows:

* Task A (early signal): target-community records up to
  ``cutoff_a = min(timestamp of Nth contribution, t0 + feature_days)``.
* Task B (pre-entry): gateway-community records with
  ``t0 - pre_entry_days <= timestamp < t0``; ``cutoff_b = t0 - 1``.

In both tasks, a feature uses only records at or before the task cutoff and
a label uses only target-community records with timestamps strictly after
the cutoff. A label window is eligible only when it ends inside the observed
coverage of every target subreddit that existed during that window; otherwise
labels are null rather than negative.

The primary label is one of several config-driven variants. Every variant is
emitted as columns named after the variant (for example
``label_deep_a_c20_w4_f90``). The primary variant also has short aliases
(``label_deep_a``) for convenience.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import polars as pl

from radpath.coverage import (
    apply_coverage_overrides,
    coverage_from_contributions,
    coverage_ranges,
    normalise_coverage,
    parse_utc,
)

LOGGER = logging.getLogger(__name__)

SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 604_800
FOLLOWUP_TRUNCATED = "followup_truncated_by_coverage_end"


@dataclass(frozen=True)
class LabelVariant:
    """One deep-engagement label definition."""

    name: str
    min_contributions: int
    min_active_weeks: int
    followup_days: int
    kinds: tuple[str, ...] = ("submission", "comment")

    @property
    def comments_only(self) -> bool:
        """Whether this variant counts comments only."""
        return self.kinds == ("comment",)


@dataclass(frozen=True)
class CohortParams:
    """Research parameters needed to build windows and labels."""

    first_n: int
    feature_days: int
    pre_entry_days: int
    followup_days: int
    deep_min_contributions: int
    deep_min_active_weeks: int
    near_coverage_start_days: int = 30
    max_entry_utc: int | None = None
    split_eligible_only: bool = False
    embargo_followup_days: int | None = None
    label_variants: tuple[LabelVariant, ...] = ()

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CohortParams":
        """Build parameters and the full label-variant grid from config."""
        window = cfg["feature_window"]
        min_contributions = int(cfg["deep_min_contributions"])
        min_weeks = int(cfg["deep_min_active_weeks"])
        followup_days = int(cfg["followup_days"])
        primary = LabelVariant(
            name=f"c{min_contributions}_w{min_weeks}_f{followup_days}",
            min_contributions=min_contributions,
            min_active_weeks=min_weeks,
            followup_days=followup_days,
        )
        variants: list[LabelVariant] = [primary]
        seen = {primary.name}

        robustness = cfg.get("robustness", {}) or {}
        for contributions in robustness.get("min_contributions", []):
            for weeks in robustness.get("min_active_weeks", []):
                for days in robustness.get("followup_days", []):
                    variant = LabelVariant(
                        name=f"c{int(contributions)}_w{int(weeks)}_f{int(days)}",
                        min_contributions=int(contributions),
                        min_active_weeks=int(weeks),
                        followup_days=int(days),
                    )
                    if variant.name not in seen:
                        variants.append(variant)
                        seen.add(variant.name)

        if bool(robustness.get("comments_only", False)):
            variant = LabelVariant(
                name=f"{primary.name}_comments",
                min_contributions=primary.min_contributions,
                min_active_weeks=primary.min_active_weeks,
                followup_days=primary.followup_days,
                kinds=("comment",),
            )
            if variant.name not in seen:
                variants.append(variant)

        return cls(
            first_n=int(window["first_n"]),
            feature_days=int(window["days"]),
            pre_entry_days=int(cfg["pre_entry_days"]),
            followup_days=followup_days,
            deep_min_contributions=min_contributions,
            deep_min_active_weeks=min_weeks,
            near_coverage_start_days=int(
                (cfg.get("coverage", {}) or {}).get("near_start_days", 30)
            ),
            split_eligible_only=bool(
                (cfg.get("split_options", {}) or {}).get("eligible_only", False)
            ),
            embargo_followup_days=(
                int(cfg["split_options"]["embargo_followup_days"])
                if (cfg.get("split_options", {}) or {}).get("embargo_followup_days")
                else None
            ),
            max_entry_utc=(
                parse_utc(cfg["entry_cutoff_utc"])
                if cfg.get("entry_cutoff_utc") not in (None, "")
                else None
            ),
            label_variants=tuple(variants),
        )

    @property
    def primary_variant(self) -> LabelVariant:
        """The pre-specified primary label variant."""
        return LabelVariant(
            name=(
                f"c{self.deep_min_contributions}_w{self.deep_min_active_weeks}"
                f"_f{self.followup_days}"
            ),
            min_contributions=self.deep_min_contributions,
            min_active_weeks=self.deep_min_active_weeks,
            followup_days=self.followup_days,
        )

    def all_variants(self) -> tuple[LabelVariant, ...]:
        """Return every configured variant, always including the primary."""
        if not self.label_variants:
            return (self.primary_variant,)
        names = {variant.name for variant in self.label_variants}
        if self.primary_variant.name not in names:
            return (self.primary_variant, *self.label_variants)
        return self.label_variants

    @property
    def max_followup_days(self) -> int:
        """Longest label window across variants, used for split embargo."""
        return max(variant.followup_days for variant in self.all_variants())


def resolve_data_end(
    contributions: pl.DataFrame,
    configured: Any = None,
) -> int:
    """Resolve the end of observation in epoch seconds.

    This helper is retained for callers that need a single global timestamp.
    Cohort eligibility uses per-subreddit coverage instead.

    Args:
        contributions: Cleaned contribution table.
        configured: Config value. ``None``/empty means use the latest
            timestamp in ``contributions``; numbers are epoch seconds; strings
            may be epoch strings or ISO-8601 dates (assumed UTC if naive).

    Returns:
        End-of-observation timestamp.

    Raises:
        ValueError: If no configured value is available and the table is
            empty, or if an ISO date cannot be parsed.
    """
    if configured is not None and str(configured).strip():
        return parse_utc(configured)
    if contributions.height == 0:
        raise ValueError("Cannot infer data end from an empty contribution table.")
    return int(contributions["created_utc"].max())


def build_cohort(
    contributions: pl.DataFrame,
    params: CohortParams,
    target_subreddits: Iterable[str],
    gateway_subreddits: Iterable[str],
    coverage: Any = None,
    coverage_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    split_fractions: Mapping[str, Any] | None = None,
    data_end_utc: Any = None,
) -> pl.DataFrame:
    """Build one row per cohort user with cutoffs, features and labels.

    Args:
        contributions: Cleaned records with at least ``id``, ``author_hash``,
            ``subreddit``, ``kind``, ``created_utc``.
        params: Window and label-variant parameters.
        target_subreddits: Communities defining cohort entry.
        gateway_subreddits: Communities used for Task B pre-entry counts.
        coverage: Per-subreddit coverage frame or mapping. When omitted it is
            inferred from the observed target records.
        coverage_overrides: Manual start/end overrides applied to ``coverage``.
        split_fractions: When provided, add a ``split`` column and embargo
            windows that cross a split boundary.
        data_end_utc: Deprecated global fallback. When supplied without an
            explicit coverage argument, it is used as the end of every target
            subreddit's coverage.

    Returns:
        User-level DataFrame sorted by ``author_hash``. Label columns are null
        for ineligible users or for Task B users without gateway activity.

    Raises:
        ValueError: If required columns are missing or no target records
            remain.
    """
    required = {"id", "author_hash", "subreddit", "kind", "created_utc"}
    missing = required - set(contributions.columns)
    if missing:
        raise ValueError(f"Missing contribution columns: {sorted(missing)}")

    targets = {name.strip().lower() for name in target_subreddits}
    gateways = {name.strip().lower() for name in gateway_subreddits}
    if not targets:
        raise ValueError("At least one target subreddit is required.")

    variants = params.all_variants()

    if contributions.height == 0:
        return _empty_cohort(params, include_split=split_fractions is not None)

    frame = contributions.with_columns(
        pl.col("subreddit").str.to_lowercase(),
        pl.col("created_utc").cast(pl.Int64),
    )

    coverage_frame = (
        coverage_from_contributions(frame, targets)
        if coverage is None
        else normalise_coverage(coverage)
    )
    coverage_frame = apply_coverage_overrides(coverage_frame, coverage_overrides)
    coverage_frame = coverage_frame.filter(
        pl.col("subreddit").is_in(list(targets))
    )
    if data_end_utc is not None and coverage is None:
        # Deprecated compatibility path: a global observation end replaces
        # per-target ends. New code should write/configure coverage instead.
        legacy_end = parse_utc(data_end_utc)
        coverage_frame = coverage_frame.with_columns(
            pl.when(pl.col("subreddit").is_in(list(targets)))
            .then(pl.lit(legacy_end))
            .otherwise(pl.col("end_utc"))
            .alias("end_utc")
        )
    ranges = coverage_ranges(coverage_frame)

    # Target records define t0. Deduplicate repeated IDs from overlapping dumps.
    core = (
        frame.filter(pl.col("subreddit").is_in(list(targets)))
        .sort(["author_hash", "created_utc", "id"])
        .unique(subset=["kind", "id"], keep="first", maintain_order=True)
    )
    if core.height == 0:
        return _empty_cohort(params, include_split=split_fractions is not None)

    entry = (
        core.group_by("author_hash", maintain_order=True)
        .agg(pl.col("subreddit").first().alias("entry_subreddit"))
    )
    users = core.group_by("author_hash").agg(
        pl.col("created_utc").min().alias("t0"),
        pl.len().alias("n_target_records_total"),
    ).join(entry, on="author_hash", how="left")

    if params.max_entry_utc is not None:
        # D3: users entering after the last date every target stream is fully
        # observed have unreliable entry times and labels; drop them.
        users = users.filter(pl.col("t0") <= params.max_entry_utc)
        if users.height == 0:
            return _empty_cohort(params, include_split=split_fractions is not None)

    nth = (
        core.with_columns(pl.col("id").cum_count().over("author_hash").alias("_rank"))
        .filter(pl.col("_rank") == params.first_n)
        .select("author_hash", pl.col("created_utc").alias("_nth_ts"))
    )
    users = users.join(nth, on="author_hash", how="left").with_columns(
        (pl.col("t0") + params.feature_days * SECONDS_PER_DAY).alias("_a_cap"),
    )
    users = users.with_columns(
        pl.min_horizontal(
            pl.col("_a_cap"),
            pl.col("_nth_ts").fill_null(pl.col("_a_cap")),
        ).alias("cutoff_a"),
        (pl.col("t0") - 1).alias("cutoff_b"),
    )

    followups = sorted({variant.followup_days for variant in variants})
    users = _add_followup_eligibility(users, ranges, followups)

    # Task A feature count: target records at or before cutoff_a.
    feature_a = (
        core.join(users.select("author_hash", "cutoff_a"), on="author_hash", how="left")
        .filter(pl.col("created_utc") <= pl.col("cutoff_a"))
        .group_by("author_hash")
        .agg(pl.len().alias("n_target_window_a"))
    )

    # Task B feature counts: gateway records in [t0 - pre_entry_days, t0),
    # equivalently at or before cutoff_b = t0 - 1.
    gateway = (
        frame.filter(pl.col("subreddit").is_in(list(gateways)))
        .sort(["author_hash", "created_utc", "id"])
        .unique(subset=["kind", "id"], keep="first", maintain_order=True)
        .join(users.select("author_hash", "t0"), on="author_hash", how="inner")
        .filter(
            (pl.col("created_utc") < pl.col("t0"))
            & (
                pl.col("created_utc")
                >= pl.col("t0") - params.pre_entry_days * SECONDS_PER_DAY
            )
        )
    )
    gateway_features = gateway.group_by("author_hash").agg(
        pl.len().alias("n_gateway_pre_b"),
        pl.col("subreddit").n_unique().alias("n_gateway_subreddits_pre_b"),
        (pl.col("created_utc") // SECONDS_PER_WEEK)
        .n_unique()
        .alias("n_gateway_active_weeks_pre_b"),
    )

    users = (
        users.join(feature_a, on="author_hash", how="left")
        .join(gateway_features, on="author_hash", how="left")
        .with_columns(
            pl.col("n_target_window_a").fill_null(0),
            pl.col("n_gateway_pre_b").fill_null(0),
            pl.col("n_gateway_subreddits_pre_b").fill_null(0),
            pl.col("n_gateway_active_weeks_pre_b").fill_null(0),
        )
    )

    # Task B cohort restriction: at least one gateway contribution pre-entry.
    for days in followups:
        users = users.with_columns(
            (
                pl.col(f"eligible_b_f{days}")
                & (pl.col("n_gateway_pre_b") >= 1)
            ).alias(f"task_b_included_f{days}")
        )

    # Label counts depend only on follow-up length and record kinds, so each
    # unique combination is aggregated once and reused across threshold variants.
    combos = {
        (variant.followup_days, variant.kinds)
        for variant in variants
    }
    for task, cutoff_col in (("a", "cutoff_a"), ("b", "cutoff_b")):
        for days, kinds in sorted(combos):
            key = _combo_key(task, days, kinds)
            labels = _window_labels(
                core,
                users,
                cutoff_col,
                f"followup_end_{task}_f{days}",
                key,
                kinds,
            )
            users = users.join(labels, on="author_hash", how="left")

    users = users.with_columns(
        [
            pl.col(f"label_n_{_combo_key(task, days, kinds)}").fill_null(0)
            .alias(f"label_n_{task}_{variant.name}")
            for variant in variants
            for task in ("a", "b")
            for days, kinds in [(variant.followup_days, variant.kinds)]
        ]
        + [
            pl.col(f"label_weeks_{_combo_key(task, days, kinds)}").fill_null(0)
            .alias(f"label_weeks_{task}_{variant.name}")
            for variant in variants
            for task in ("a", "b")
            for days, kinds in [(variant.followup_days, variant.kinds)]
        ]
    )

    for variant in variants:
        for task in ("a", "b"):
            eligible_col = (
                f"eligible_{task}_f{variant.followup_days}"
                if task == "a"
                else f"task_b_included_f{variant.followup_days}"
            )
            users = users.with_columns(
                pl.when(pl.col(eligible_col))
                .then(
                    (
                        pl.col(f"label_n_{task}_{variant.name}")
                        >= variant.min_contributions
                    )
                    & (
                        pl.col(f"label_weeks_{task}_{variant.name}")
                        >= variant.min_active_weeks
                    )
                )
                .otherwise(None)
                .alias(f"label_deep_{task}_{variant.name}")
            )

    primary = params.primary_variant
    aliases = [
        pl.col(f"followup_end_a_f{params.followup_days}").alias("followup_end_a"),
        pl.col(f"followup_end_b_f{params.followup_days}").alias("followup_end_b"),
        pl.col(f"eligible_a_f{params.followup_days}").alias("eligible_a"),
        pl.col(f"eligible_b_f{params.followup_days}").alias("eligible_b"),
        pl.col(f"ineligible_reason_a_f{params.followup_days}").alias(
            "ineligible_reason_a"
        ),
        pl.col(f"ineligible_reason_b_f{params.followup_days}").alias(
            "ineligible_reason_b"
        ),
        pl.col(f"task_b_included_f{params.followup_days}").alias("task_b_included"),
        pl.col(f"label_n_a_{primary.name}").alias("label_n_a"),
        pl.col(f"label_weeks_a_{primary.name}").alias("label_weeks_a"),
        pl.col(f"label_deep_a_{primary.name}").alias("label_deep_a"),
        pl.col(f"label_n_b_{primary.name}").alias("label_n_b"),
        pl.col(f"label_weeks_b_{primary.name}").alias("label_weeks_b"),
        pl.col(f"label_deep_b_{primary.name}").alias("label_deep_b"),
    ]
    users = users.with_columns(aliases)

    earliest_target_start = (
        min((start for start, _ in ranges.values()), default=None)
    )
    if earliest_target_start is None:
        users = users.with_columns(pl.lit(False).alias("near_coverage_start"))
    else:
        users = users.with_columns(
            (
                (pl.col("t0") >= earliest_target_start)
                & (
                    pl.col("t0")
                    <= earliest_target_start
                    + params.near_coverage_start_days * SECONDS_PER_DAY
                )
            ).alias("near_coverage_start")
        )

    users = users.with_columns(
        (
            pl.col("cutoff_a")
            + params.max_followup_days * SECONDS_PER_DAY
        ).alias("label_window_end")
    ).drop("_a_cap", "_nth_ts")

    # Drop temporary per-combination count columns now that variants are named.
    temporary = [
        f"label_n_{_combo_key(task, days, kinds)}"
        for task in ("a", "b")
        for days, kinds in combos
    ] + [
        f"label_weeks_{_combo_key(task, days, kinds)}"
        for task in ("a", "b")
        for days, kinds in combos
    ]
    users = users.drop(temporary).sort("author_hash")

    if split_fractions is not None:
        from radpath.split import temporal_split

        embargo_days = params.embargo_followup_days or params.max_followup_days
        users = users.with_columns(
            (pl.col("cutoff_a") + embargo_days * SECONDS_PER_DAY).alias("_embargo_end")
        )
        if params.split_eligible_only:
            # D4: split among users whose primary Task A label is observable.
            eligible = temporal_split(
                users.filter(pl.col("eligible_a")),
                split_fractions,
                label_end_col="_embargo_end",
            ).select("author_hash", "split")
            users = users.join(eligible, on="author_hash", how="left")
        else:
            users = temporal_split(
                users, split_fractions, label_end_col="_embargo_end"
            )
        users = users.drop("_embargo_end").sort("author_hash")

    return users


def _combo_key(task: str, days: int, kinds: tuple[str, ...]) -> str:
    """Column key for one cutoff/task/follow-up/kinds combination."""
    kind_key = "comments" if kinds == ("comment",) else "all"
    return f"{task}_f{int(days)}_{kind_key}"


def _add_followup_eligibility(
    users: pl.DataFrame,
    ranges: Mapping[str, tuple[int, int]],
    followups: Iterable[int],
) -> pl.DataFrame:
    """Add follow-up end dates and coverage eligibility per task and length."""
    for task, cutoff_col in (("a", "cutoff_a"), ("b", "cutoff_b")):
        for days in followups:
            end_col = f"followup_end_{task}_f{days}"
            truncated = pl.lit(False)
            for start, end in ranges.values():
                # Only targets that existed at entry can truncate the label
                # window. A community that launches later cannot explain the
                # user's absence from it before launch.
                relevant = (pl.lit(start) <= pl.col("t0")) & (
                    pl.lit(end) >= pl.col("t0")
                )
                truncated = truncated | (relevant & (pl.col(end_col) > pl.lit(end)))
            users = users.with_columns(
                (pl.col(cutoff_col) + days * SECONDS_PER_DAY).alias(end_col),
            ).with_columns(
                (~truncated).alias(f"eligible_{task}_f{days}"),
                pl.when(truncated)
                .then(pl.lit(FOLLOWUP_TRUNCATED))
                .otherwise(None)
                .alias(f"ineligible_reason_{task}_f{days}"),
            )
    return users


def _window_labels(
    core: pl.DataFrame,
    users: pl.DataFrame,
    cutoff_col: str,
    end_col: str,
    key: str,
    kinds: tuple[str, ...],
) -> pl.DataFrame:
    """Aggregate post-window contribution counts and active weeks.

    Active weeks are 7-day bins anchored at the first second after the
    cutoff, so a 90-day window contains at most 13 bins. ``kinds`` restricts
    the counted records (all records or comments only for the variant).
    """
    filtered = core.filter(pl.col("kind").is_in(list(kinds)))
    return (
        filtered.join(
            users.select("author_hash", cutoff_col, end_col),
            on="author_hash",
            how="left",
        )
        .filter(
            (pl.col("created_utc") > pl.col(cutoff_col))
            & (pl.col("created_utc") <= pl.col(end_col))
        )
        .group_by("author_hash")
        .agg(
            pl.len().alias(f"label_n_{key}"),
            (
                (pl.col("created_utc") - pl.col(cutoff_col) - 1)
                // SECONDS_PER_WEEK
            )
            .n_unique()
            .alias(f"label_weeks_{key}"),
        )
    )


def _empty_cohort(params: CohortParams, include_split: bool = False) -> pl.DataFrame:
    """Return an empty cohort frame with the expected schema."""
    schema: dict[str, Any] = {
        "author_hash": pl.String,
        "t0": pl.Int64,
        "n_target_records_total": pl.Int64,
        "entry_subreddit": pl.String,
        "cutoff_a": pl.Int64,
        "cutoff_b": pl.Int64,
        "n_target_window_a": pl.Int64,
        "n_gateway_pre_b": pl.Int64,
        "n_gateway_subreddits_pre_b": pl.Int64,
        "n_gateway_active_weeks_pre_b": pl.Int64,
        "near_coverage_start": pl.Boolean,
        "label_window_end": pl.Int64,
    }
    followups = sorted({variant.followup_days for variant in params.all_variants()})
    for task in ("a", "b"):
        for days in followups:
            schema[f"followup_end_{task}_f{days}"] = pl.Int64
            schema[f"eligible_{task}_f{days}"] = pl.Boolean
            schema[f"ineligible_reason_{task}_f{days}"] = pl.String
            if task == "b":
                schema[f"task_b_included_f{days}"] = pl.Boolean
    schema.update(
        {
            "followup_end_a": pl.Int64,
            "followup_end_b": pl.Int64,
            "eligible_a": pl.Boolean,
            "eligible_b": pl.Boolean,
            "ineligible_reason_a": pl.String,
            "ineligible_reason_b": pl.String,
            "task_b_included": pl.Boolean,
        }
    )
    for variant in params.all_variants():
        for task in ("a", "b"):
            schema[f"label_n_{task}_{variant.name}"] = pl.Int64
            schema[f"label_weeks_{task}_{variant.name}"] = pl.Int64
            schema[f"label_deep_{task}_{variant.name}"] = pl.Boolean
    schema.update(
        {
            "label_n_a": pl.Int64,
            "label_weeks_a": pl.Int64,
            "label_deep_a": pl.Boolean,
            "label_n_b": pl.Int64,
            "label_weeks_b": pl.Int64,
            "label_deep_b": pl.Boolean,
        }
    )
    if include_split:
        schema["split"] = pl.String
    return pl.DataFrame(schema=schema)
