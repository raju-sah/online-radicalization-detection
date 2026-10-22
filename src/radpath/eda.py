"""Aggregate-only exploratory analysis and plots.

Everything here operates on the user-level cohort and writes only summary
statistics, class balances, coverage summaries and distribution plots. No
usernames or hashed user IDs are exported; per-user rows never leave
``data/processed``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # headless, safe on servers
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from radpath.cohort import CohortParams, LabelVariant  # noqa: E402
from radpath.coverage import normalise_coverage  # noqa: E402
from radpath.split import SPLIT_ORDER  # noqa: E402

TASK_LABELS: dict[str, str] = {"a": "Task A (early)", "b": "Task B (pre-entry)"}
TASK_COLORS: dict[str, str] = {"a": "#2b6cb0", "b": "#c05621"}


def run_eda(
    cohort: pl.DataFrame,
    cfg: Mapping[str, Any],
    out_dir: Path,
    coverage: Any = None,
    data_end_utc: int | None = None,
) -> dict[str, Any]:
    """Compute aggregate summaries and write tables/plots to ``out_dir``.

    Args:
        cohort: Output of :func:`radpath.cohort.build_cohort`, ideally with a
            ``split`` column.
        cfg: radpath config.
        out_dir: Report directory (created if needed).
        coverage: Effective per-subreddit coverage frame or mapping.
        data_end_utc: Deprecated global observation end, retained only for
            callers of older outputs.

    Returns:
        Dictionary of aggregate summary values.

    Raises:
        ValueError: If the cohort is empty or has no eligible users.
    """
    if cohort.height == 0:
        raise ValueError("Cohort is empty; nothing to summarise.")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = CohortParams.from_config(cfg)
    variants = params.all_variants()
    primary = params.primary_variant
    top_n = int(cfg.get("eda", {}).get("top_n_subreddits", 10))

    eligibility = {
        "a": cohort.filter(pl.col("eligible_a")),
        "b": cohort.filter(pl.col("task_b_included")),
    }
    if all(frame.height == 0 for frame in eligibility.values()):
        raise ValueError(
            "No eligible users for Task A or B. Check data coverage and the "
            "followup_days setting; labels are null for insufficient follow-up."
        )

    balance_rows = _class_balance_rows(cohort, variants)
    balance = pl.DataFrame(balance_rows)
    balance.write_csv(out_dir / "class_balance.csv")

    split_rows = _split_balance_rows(cohort, variants)
    split_balance = pl.DataFrame(split_rows)
    split_balance.write_csv(out_dir / "split_balance.csv")

    variant_rows = _variant_balance_rows(cohort, variants)
    variant_balance = pl.DataFrame(variant_rows)
    variant_balance.write_csv(out_dir / "variant_balance.csv")

    ineligibility_rows = _ineligibility_rows(cohort)
    ineligibility = pl.DataFrame(ineligibility_rows)
    ineligibility.write_csv(out_dir / "ineligibility.csv")

    sensitivity_rows = _sensitivity_rows(cohort, cfg)
    sensitivity = pl.DataFrame(sensitivity_rows)
    sensitivity.write_csv(out_dir / "sensitivity.csv")

    coverage_frame = normalise_coverage(coverage)
    coverage_table = _coverage_table(coverage_frame)
    coverage_table.write_csv(out_dir / "coverage.csv")

    monthly = _monthly_entrants(cohort)
    monthly.write_csv(out_dir / "monthly_new_entrants.csv")

    _plot_class_balance(balance, out_dir / "class_balance.png")
    _plot_entry_timeline(cohort, out_dir / "entry_timeline.png")
    _plot_coverage(coverage_frame, cfg, out_dir / "coverage_timeline.png")
    _plot_monthly_entrants(monthly, out_dir / "monthly_new_entrants.png")
    _plot_post_window_counts(
        eligibility,
        primary.min_contributions,
        primary.min_active_weeks,
        out_dir / "post_window_activity.png",
    )
    _plot_active_weeks(
        eligibility,
        primary.min_active_weeks,
        out_dir / "active_weeks.png",
    )
    _plot_gateway_activity(
        eligibility["b"],
        int(cfg["pre_entry_days"]),
        out_dir / "gateway_activity.png",
    )
    _plot_entry_subreddits(cohort, top_n, out_dir / "entry_subreddits.png")
    if "split" in cohort.columns:
        _plot_split_rates(split_balance, out_dir / "split_positive_rates.png")

    summary = _build_summary(
        cohort,
        cfg,
        params,
        balance,
        split_balance,
        ineligibility,
        sensitivity,
        coverage_frame,
        data_end_utc,
    )
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {
        "n_users": cohort.height,
        "class_balance": balance_rows,
        "split_balance": split_rows,
        "variant_balance": variant_rows,
        "ineligibility": ineligibility_rows,
        "sensitivity": sensitivity_rows,
    }


def _class_balance_rows(
    cohort: pl.DataFrame, variants: tuple[LabelVariant, ...]
) -> list[dict[str, Any]]:
    """Class counts and positive rates among eligible users, by task."""
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for task in ("a", "b"):
            eligible_col = "eligible_a" if task == "a" else "task_b_included"
            label_col = f"label_deep_{task}_{variant.name}"
            frame = cohort.filter(pl.col(eligible_col))
            labels = frame.get_column(label_col).drop_nulls()
            n_positive = int(labels.eq(True).sum())
            n_negative = int(labels.eq(False).sum())
            n_eligible = n_positive + n_negative
            rows.append(
                {
                    "task": task,
                    "variant": variant.name,
                    "eligible_users": n_eligible,
                    "positive": n_positive,
                    "negative": n_negative,
                    "positive_rate": (n_positive / n_eligible) if n_eligible else None,
                    "deep_min_contributions": variant.min_contributions,
                    "deep_min_active_weeks": variant.min_active_weeks,
                    "followup_days": variant.followup_days,
                    "kinds": "+".join(variant.kinds),
                }
            )
    return rows


def _split_balance_rows(
    cohort: pl.DataFrame, variants: tuple[LabelVariant, ...]
) -> list[dict[str, Any]]:
    """Class counts per temporal split, task and label variant."""
    if "split" not in cohort.columns:
        return []
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        split_frame = cohort.filter(pl.col("split") == split)
        for variant in variants:
            for task in ("a", "b"):
                label_col = f"label_deep_{task}_{variant.name}"
                labels = split_frame.get_column(label_col).drop_nulls()
                n_positive = int(labels.eq(True).sum())
                n_negative = int(labels.eq(False).sum())
                n_eligible = n_positive + n_negative
                rows.append(
                    {
                        "task": task,
                        "variant": variant.name,
                        "split": split,
                        "n_users_total": split_frame.height,
                        "n_eligible": n_eligible,
                        "positive": n_positive,
                        "negative": n_negative,
                        "positive_rate": (
                            n_positive / n_eligible if n_eligible else None
                        ),
                    }
                )
    return rows


def _variant_balance_rows(
    cohort: pl.DataFrame, variants: tuple[LabelVariant, ...]
) -> list[dict[str, Any]]:
    """Class balance for every label variant, overall and per split."""
    rows: list[dict[str, Any]] = []
    splits = ("all", *SPLIT_ORDER) if "split" in cohort.columns else ("all",)
    for variant in variants:
        for task in ("a", "b"):
            label_col = f"label_deep_{task}_{variant.name}"
            for split in splits:
                frame = (
                    cohort
                    if split == "all"
                    else cohort.filter(pl.col("split") == split)
                )
                labels = frame.get_column(label_col).drop_nulls()
                n_positive = int(labels.eq(True).sum())
                n_negative = int(labels.eq(False).sum())
                n_eligible = n_positive + n_negative
                rows.append(
                    {
                        "task": task,
                        "variant": variant.name,
                        "split": split,
                        "min_contributions": variant.min_contributions,
                        "min_active_weeks": variant.min_active_weeks,
                        "followup_days": variant.followup_days,
                        "kinds": "+".join(variant.kinds),
                        "n_users_total": frame.height,
                        "n_eligible": n_eligible,
                        "positive": n_positive,
                        "negative": n_negative,
                        "positive_rate": (
                            n_positive / n_eligible if n_eligible else None
                        ),
                    }
                )
    return rows


def _ineligibility_rows(cohort: pl.DataFrame) -> list[dict[str, Any]]:
    """Count ineligible users by reason for each task."""
    rows: list[dict[str, Any]] = []
    for task in ("a", "b"):
        reason_col = f"ineligible_reason_{task}"
        counts = (
            cohort.filter(pl.col(reason_col).is_not_null())
            .group_by(reason_col)
            .len()
            .sort("len", descending=True)
        )
        for row in counts.iter_rows(named=True):
            rows.append(
                {
                    "task": task,
                    "reason": row[reason_col],
                    "n_users": int(row["len"]),
                    "share_of_cohort": int(row["len"]) / cohort.height,
                }
            )
        if task == "b":
            rows.append(
                {
                    "task": task,
                    "reason": "no_gateway_activity",
                    "n_users": int(
                        cohort.select(
                            (
                                pl.col("eligible_b")
                                & (pl.col("n_gateway_pre_b") == 0)
                            ).sum()
                        ).item()
                    ),
                    "share_of_cohort": (
                        int(
                            cohort.select(
                                (
                                    pl.col("eligible_b")
                                    & (pl.col("n_gateway_pre_b") == 0)
                                ).sum()
                            ).item()
                        )
                        / cohort.height
                    ),
                }
            )
    return rows


def _sensitivity_rows(
    cohort: pl.DataFrame, cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Primary-label balance with and without near-coverage-start users."""
    if "near_coverage_start" not in cohort.columns:
        return []
    exclude = bool(cfg.get("sensitivity", {}).get("exclude_near_coverage_start", False))
    if not exclude:
        return []
    rows: list[dict[str, Any]] = []
    for task in ("a", "b"):
        eligible_col = "eligible_a" if task == "a" else "task_b_included"
        label_col = f"label_deep_{task}"
        full = cohort.filter(pl.col(eligible_col))
        trimmed = full.filter(~pl.col("near_coverage_start"))
        for name, frame in (("full", full), ("exclude_near_coverage_start", trimmed)):
            labels = frame.get_column(label_col).drop_nulls()
            n_positive = int(labels.eq(True).sum())
            n_negative = int(labels.eq(False).sum())
            n_eligible = n_positive + n_negative
            rows.append(
                {
                    "task": task,
                    "analysis": name,
                    "n_users_total": frame.height,
                    "n_eligible": n_eligible,
                    "positive": n_positive,
                    "negative": n_negative,
                    "positive_rate": (
                        n_positive / n_eligible if n_eligible else None
                    ),
                }
            )
    return rows


def _coverage_table(coverage: pl.DataFrame) -> pl.DataFrame:
    """Human-readable per-subreddit coverage table."""
    if coverage.height == 0:
        return pl.DataFrame(
            schema={
                "subreddit": pl.String,
                "start_utc": pl.Int64,
                "end_utc": pl.Int64,
                "n_records": pl.Int64,
                "overridden": pl.Boolean,
                "start_date": pl.String,
                "end_date": pl.String,
            }
        )
    return coverage.with_columns(
        pl.when(pl.col("start_utc").is_not_null())
        .then(
            pl.from_epoch(pl.col("start_utc"), time_unit="s")
            .dt.strftime("%Y-%m-%d")
        )
        .otherwise(None)
        .alias("start_date"),
        pl.when(pl.col("end_utc").is_not_null())
        .then(
            pl.from_epoch(pl.col("end_utc"), time_unit="s")
            .dt.strftime("%Y-%m-%d")
        )
        .otherwise(None)
        .alias("end_date"),
    ).sort("subreddit")


def _monthly_entrants(cohort: pl.DataFrame) -> pl.DataFrame:
    """Monthly count of new cohort entrants per entry subreddit."""
    if "entry_subreddit" not in cohort.columns:
        return pl.DataFrame(
            schema={"month": pl.String, "entry_subreddit": pl.String, "n_users": pl.Int64}
        )
    return (
        cohort.with_columns(
            pl.from_epoch(pl.col("t0"), time_unit="s")
            .dt.strftime("%Y-%m")
            .alias("month")
        )
        .group_by("month", "entry_subreddit")
        .len()
        .rename({"len": "n_users"})
        .sort(["month", "entry_subreddit"])
    )


def _plot_class_balance(balance: pl.DataFrame, path: Path) -> None:
    """Bar chart of positive vs negative counts for each task (primary)."""
    primary = balance.filter(pl.col("variant") == balance["variant"][0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, task in zip(axes, ("a", "b")):
        row = primary.filter(pl.col("task") == task).row(0, named=True)
        values = [int(row["negative"]), int(row["positive"])]
        bars = axis.bar(
            ["not deep", "deep"], values, color=["#a0aec0", TASK_COLORS[task]]
        )
        axis.set_title(TASK_LABELS[task])
        axis.set_ylabel("eligible users")
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(value),
                ha="center",
                va="bottom",
            )
        rate = row["positive_rate"]
        axis.text(
            0.98,
            0.95,
            f"positive rate: {rate:.1%}" if rate is not None else "no eligible users",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    fig.suptitle("Class balance among eligible users (primary label)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_entry_timeline(cohort: pl.DataFrame, path: Path) -> None:
    """Histogram of cohort entry dates, split by temporal split if present."""
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    dates = [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in cohort["t0"]]
    if "split" in cohort.columns:
        split_values = cohort["split"].fill_null("embargoed")
        for split in (*SPLIT_ORDER, "embargoed"):
            mask = (split_values == split).to_numpy()
            selected = [date for date, keep in zip(dates, mask) if keep]
            if selected:
                axis.hist(selected, bins=30, alpha=0.6, label=split)
        axis.legend(title="temporal split")
    else:
        axis.hist(dates, bins=30, color="#4a5568")
    axis.set_title("Cohort entry dates (first target-community contribution)")
    axis.set_xlabel("entry date (UTC)")
    axis.set_ylabel("users")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_coverage(
    coverage: pl.DataFrame, cfg: Mapping[str, Any], path: Path
) -> None:
    """Horizontal timeline of per-subreddit coverage windows."""
    frame = coverage.filter(
        pl.col("start_utc").is_not_null() & pl.col("end_utc").is_not_null()
    )
    if frame.height == 0:
        return
    targets = {str(name).lower() for name in cfg.get("target_subreddits", [])}
    frame = frame.sort("start_utc")
    fig, axis = plt.subplots(
        figsize=(10, max(3, 0.35 * frame.height + 1.5)),
        constrained_layout=True,
    )
    for index, row in enumerate(frame.iter_rows(named=True)):
        start = mdates.date2num(
            datetime.fromtimestamp(int(row["start_utc"]), tz=timezone.utc)
        )
        end = mdates.date2num(
            datetime.fromtimestamp(int(row["end_utc"]), tz=timezone.utc)
        )
        colour = "#2b6cb0" if row["subreddit"] in targets else "#a0aec0"
        axis.barh(index, max(end - start, 1), left=start, color=colour, height=0.6)
        if row["overridden"]:
            axis.text(
                end,
                index,
                " override",
                va="center",
                ha="left",
                fontsize=7,
                color="#c05621",
            )
    axis.set_yticks(range(frame.height), frame["subreddit"].to_list())
    axis.xaxis_date()
    axis.set_title("Per-subreddit observation coverage (blue = target)")
    axis.set_xlabel("date (UTC)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_monthly_entrants(monthly: pl.DataFrame, path: Path) -> None:
    """Monthly new entrants per entry subreddit."""
    if monthly.height == 0:
        return
    months = sorted(monthly["month"].unique().to_list())
    positions = {month: index for index, month in enumerate(months)}
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    for subreddit in sorted(monthly["entry_subreddit"].unique().to_list()):
        frame = monthly.filter(pl.col("entry_subreddit") == subreddit)
        x = [positions[month] for month in frame["month"].to_list()]
        axis.plot(x, frame["n_users"].to_list(), marker="o", markersize=2, label=subreddit)
    step = max(1, len(months) // 12)
    axis.set_xticks(range(0, len(months), step), months[::step], rotation=45, ha="right")
    axis.set_title("Monthly new cohort entrants by entry subreddit")
    axis.set_ylabel("new users")
    axis.legend(title="entry subreddit", fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_post_window_counts(
    eligibility: Mapping[str, pl.DataFrame],
    min_contributions: int,
    min_weeks: int,
    path: Path,
) -> None:
    """Distribution of post-window contributions and active weeks by task."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, (task, frame) in zip(axes, eligibility.items()):
        counts = frame.get_column(f"label_n_{task}").to_numpy()
        if counts.size == 0:
            axis.set_visible(False)
            continue
        values = np.log1p(counts)
        axis.hist(values, bins=30, color=TASK_COLORS[task], alpha=0.85)
        axis.axvline(np.log1p(min_contributions), color="black", linestyle="--", linewidth=1)
        axis.set_title(TASK_LABELS[task])
        axis.set_xlabel("contributions in label window (log1p)")
        axis.set_ylabel("eligible users")
        axis.text(
            0.98,
            0.95,
            f"threshold: {min_contributions} contributions\nand {min_weeks} active weeks",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    fig.suptitle("Post-window target-community activity")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_active_weeks(
    eligibility: Mapping[str, pl.DataFrame],
    min_weeks: int,
    path: Path,
) -> None:
    """Histogram of distinct active weeks inside each label window."""
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for task, frame in eligibility.items():
        weeks = frame.get_column(f"label_weeks_{task}").to_numpy()
        if weeks.size == 0:
            continue
        bins = np.arange(0, max(14, int(weeks.max()) + 2)) - 0.5
        axis.hist(weeks, bins=bins, alpha=0.55, label=TASK_LABELS[task])
    axis.axvline(min_weeks - 0.5, color="black", linestyle="--", linewidth=1)
    axis.set_title("Distinct active weeks in the label window")
    axis.set_xlabel("active weeks (7-day bins from window start)")
    axis.set_ylabel("eligible users")
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_gateway_activity(frame: pl.DataFrame, pre_entry_days: int, path: Path) -> None:
    """Distribution of Task B pre-entry gateway contribution counts."""
    counts = frame.get_column("n_gateway_pre_b").to_numpy()
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    if counts.size:
        axis.hist(np.log1p(counts), bins=30, color=TASK_COLORS["b"], alpha=0.85)
        zero_share = float((counts == 0).mean())
        axis.text(
            0.98,
            0.95,
            f"no gateway activity: {zero_share:.1%}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
    axis.set_title(f"Task B: gateway contributions in the {pre_entry_days} days before entry")
    axis.set_xlabel("contributions (log1p)")
    axis.set_ylabel("included users")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_entry_subreddits(cohort: pl.DataFrame, top_n: int, path: Path) -> None:
    """Top entry subreddits by cohort users (aggregate counts only)."""
    counts = (
        cohort.group_by("entry_subreddit")
        .len()
        .sort("len", descending=True)
        .head(top_n)
        .sort("len")
    )
    fig, axis = plt.subplots(figsize=(8, max(3, 0.35 * counts.height + 1.5)), constrained_layout=True)
    axis.barh(counts["entry_subreddit"], counts["len"], color="#4a5568")
    axis.set_title("Top entry subreddits")
    axis.set_xlabel("cohort users")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_split_rates(split_balance: pl.DataFrame, path: Path) -> None:
    """Positive rate by temporal split for the primary label and task."""
    if split_balance.height == 0:
        return
    primary_variant = split_balance["variant"][0]
    frame = split_balance.filter(pl.col("variant") == primary_variant)
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    width = 0.35
    x = np.arange(len(SPLIT_ORDER))
    for offset, task in ((-width / 2, "a"), (width / 2, "b")):
        rates = []
        for split in SPLIT_ORDER:
            row = frame.filter(
                (pl.col("split") == split) & (pl.col("task") == task)
            )
            rate = row["positive_rate"][0] if row.height else None
            rates.append(float(rate) if rate is not None else np.nan)
        axis.bar(x + offset, rates, width=width, label=TASK_LABELS[task], color=TASK_COLORS[task])
    axis.set_xticks(x, list(SPLIT_ORDER))
    axis.set_ylabel("positive rate")
    axis.set_title("Primary deep-engagement rate by temporal split")
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_summary(
    cohort: pl.DataFrame,
    cfg: Mapping[str, Any],
    params: CohortParams,
    balance: pl.DataFrame,
    split_balance: pl.DataFrame,
    ineligibility: pl.DataFrame,
    sensitivity: pl.DataFrame,
    coverage: pl.DataFrame,
    data_end_utc: int | None,
) -> str:
    """Build the aggregate Markdown summary."""
    primary = params.primary_variant
    start = datetime.fromtimestamp(int(cohort["t0"].min()), tz=timezone.utc).date()
    end = datetime.fromtimestamp(int(cohort["t0"].max()), tz=timezone.utc).date()
    lines = [
        "# radpath EDA summary",
        "",
        "Aggregate output only. Hashed user IDs are not included.",
        "",
        f"- Users in cohort: **{cohort.height:,}**",
        f"- Entry dates: **{start} to {end}**",
    ]
    if "split" in cohort.columns:
        n_embargoed = int(cohort["split"].is_null().sum())
        lines.append(f"- Embargoed users with null split: **{n_embargoed:,}**")
    if data_end_utc is not None:
        end_date = datetime.fromtimestamp(int(data_end_utc), tz=timezone.utc).date()
        lines.append(f"- Deprecated global observation end: **{end_date}**")
    lines.extend(
        [
            f"- Primary label: **>= {primary.min_contributions} contributions** and "
            f"**>= {primary.min_active_weeks} active weeks** in the "
            f"{primary.followup_days}-day label window",
            f"- Label variants generated: **{len(params.all_variants())}** "
            f"(primary `{primary.name}`)",
            f"- Task A window: first {int(cfg['feature_window']['first_n'])} target "
            f"contributions, capped at {int(cfg['feature_window']['days'])} days",
            f"- Task B window: {int(cfg['pre_entry_days'])} days before entry, gateway subreddits only",
            "- Task B cohort requires at least one pre-entry gateway contribution.",
            "",
            "## Class balance (primary label, eligible users)",
            "",
            "| task | eligible | positive | negative | positive rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    primary_balance = balance.filter(pl.col("variant") == primary.name)
    for row in primary_balance.iter_rows(named=True):
        rate = row["positive_rate"]
        rate_text = f"{rate:.1%}" if rate is not None else "n/a"
        lines.append(
            f"| {TASK_LABELS[str(row['task'])]} | {row['eligible_users']:,} | "
            f"{row['positive']:,} | {row['negative']:,} | {rate_text} |"
        )

    if split_balance.height:
        lines.extend(
            [
                "",
                "## Temporal splits (primary label)",
                "",
                "| split | users | task | eligible | positive rate |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for row in split_balance.filter(
            pl.col("variant") == primary.name
        ).iter_rows(named=True):
            rate = row["positive_rate"]
            rate_text = f"{rate:.1%}" if rate is not None else "n/a"
            lines.append(
                f"| {row['split']} | {row['n_users_total']:,} | "
                f"{TASK_LABELS[str(row['task'])]} | {row['n_eligible']:,} | {rate_text} |"
            )

    if ineligibility.height:
        lines.extend(
            [
                "",
                "## Ineligible / excluded users by reason",
                "",
                "| task | reason | users | share of cohort |",
                "|---|---|---:|---:|",
            ]
        )
        for row in ineligibility.iter_rows(named=True):
            lines.append(
                f"| {TASK_LABELS[str(row['task'])]} | {row['reason']} | "
                f"{row['n_users']:,} | {row['share_of_cohort']:.1%} |"
            )

    if sensitivity.height:
        lines.extend(
            [
                "",
                "## Sensitivity: exclude near-coverage-start users",
                "",
                "| task | analysis | eligible | positive rate |",
                "|---|---|---:|---:|",
            ]
        )
        for row in sensitivity.iter_rows(named=True):
            rate = row["positive_rate"]
            rate_text = f"{rate:.1%}" if rate is not None else "n/a"
            lines.append(
                f"| {TASK_LABELS[str(row['task'])]} | {row['analysis']} | "
                f"{row['n_eligible']:,} | {rate_text} |"
            )

    if coverage.height:
        lines.extend(
            [
                "",
                "## Coverage",
                "",
                "| subreddit | start | end | records | override |",
                "|---|---|---|---:|---|",
            ]
        )
        for row in _coverage_table(coverage).iter_rows(named=True):
            lines.append(
                f"| {row['subreddit']} | {row['start_date'] or 'n/a'} | "
                f"{row['end_date'] or 'n/a'} | {row['n_records']:,} | "
                f"{'yes' if row['overridden'] else 'no'} |"
            )

    task_b_followup = cohort.filter(pl.col("eligible_b"))
    if task_b_followup.height:
        zero_gateway = int((task_b_followup["n_gateway_pre_b"] == 0).sum())
        lines.extend(
            [
                "",
                "## Task B gateway restriction",
                "",
                f"- Follow-up eligible Task B users: **{task_b_followup.height:,}**",
                f"- Excluded for zero pre-entry gateway activity: **{zero_gateway:,}** "
                f"({zero_gateway / task_b_followup.height:.1%})",
            ]
        )

    median_a = cohort.filter(pl.col("eligible_a"))["label_n_a"].median()
    median_b = cohort.filter(pl.col("task_b_included"))["label_n_b"].median()
    lines.extend(
        [
            "",
            "## Aggregate activity",
            "",
            f"- Median primary post-window target contributions: Task A {median_a}, Task B {median_b}",
            "",
            "## Figures",
            "",
            "`class_balance.png`, `entry_timeline.png`, `coverage_timeline.png`, "
            "`monthly_new_entrants.png`, `post_window_activity.png`, "
            "`active_weeks.png`, `gateway_activity.png`, `entry_subreddits.png`, "
            "`split_positive_rates.png` (when splits exist).",
            "",
        ]
    )
    return "\n".join(lines)
