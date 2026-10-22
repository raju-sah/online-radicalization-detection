"""Aggregate-only exploratory analysis and plots.

Everything here operates on the user-level cohort and writes only summary
statistics, class balances, and distribution plots. No usernames or hashed
user IDs are exported; per-user rows never leave ``data/processed``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # headless, safe on servers
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from radpath.split import SPLIT_ORDER  # noqa: E402

TASK_LABELS: dict[str, str] = {"a": "Task A (early)", "b": "Task B (pre-entry)"}
TASK_COLORS: dict[str, str] = {"a": "#2b6cb0", "b": "#c05621"}


def run_eda(
    cohort: pl.DataFrame,
    cfg: Mapping[str, Any],
    out_dir: Path,
    data_end_utc: int | None = None,
) -> dict[str, Any]:
    """Compute aggregate summaries and write plots to ``out_dir``.

    Args:
        cohort: Output of :func:`radpath.cohort.build_cohort`, ideally with a
            ``split`` column.
        cfg: radpath config (uses deep-engagement thresholds and EDA options).
        out_dir: Report directory (created if needed).
        data_end_utc: End of observation, included in the summary if known.

    Returns:
        Dictionary of aggregate summary values.

    Raises:
        ValueError: If the cohort has no eligible users for either task.
    """
    if cohort.height == 0:
        raise ValueError("Cohort is empty; nothing to summarise.")
    out_dir.mkdir(parents=True, exist_ok=True)

    min_contributions = int(cfg["deep_min_contributions"])
    min_weeks = int(cfg["deep_min_active_weeks"])
    top_n = int(cfg.get("eda", {}).get("top_n_subreddits", 10))

    eligibility = {
        task: cohort.filter(pl.col(f"eligible_{task}"))
        for task in ("a", "b")
    }
    if all(frame.height == 0 for frame in eligibility.values()):
        raise ValueError(
            "No eligible users for Task A or B. Check data coverage and the "
            "followup_days setting; labels are null for insufficient follow-up."
        )

    balance_rows = _class_balance_rows(cohort, cfg)
    balance = pl.DataFrame(balance_rows)
    balance.write_csv(out_dir / "class_balance.csv")

    split_rows = _split_balance_rows(cohort, cfg)
    split_balance = pl.DataFrame(split_rows)
    split_balance.write_csv(out_dir / "split_balance.csv")

    _plot_class_balance(balance, out_dir / "class_balance.png")
    _plot_entry_timeline(cohort, out_dir / "entry_timeline.png")
    _plot_post_window_counts(eligibility, min_contributions, min_weeks, out_dir / "post_window_activity.png")
    _plot_active_weeks(eligibility, min_weeks, out_dir / "active_weeks.png")
    _plot_gateway_activity(eligibility["b"], int(cfg["pre_entry_days"]), out_dir / "gateway_activity.png")
    _plot_entry_subreddits(cohort, top_n, out_dir / "entry_subreddits.png")
    if "split" in cohort.columns:
        _plot_split_rates(split_balance, out_dir / "split_positive_rates.png")

    summary = _build_summary(cohort, cfg, balance, split_balance, data_end_utc)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    return {
        "n_users": cohort.height,
        "class_balance": balance_rows,
        "split_balance": split_rows,
    }


def _class_balance_rows(
    cohort: pl.DataFrame, cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Class counts and positive rates among eligible users, by task."""
    rows: list[dict[str, Any]] = []
    for task in ("a", "b"):
        frame = cohort.filter(pl.col(f"eligible_{task}"))
        labels = frame.get_column(f"label_deep_{task}").drop_nulls()
        n_positive = int(labels.eq(True).sum())
        n_negative = int(labels.eq(False).sum())
        n_eligible = n_positive + n_negative
        rows.append(
            {
                "task": task,
                "eligible_users": n_eligible,
                "positive": n_positive,
                "negative": n_negative,
                "positive_rate": (n_positive / n_eligible) if n_eligible else None,
                "deep_min_contributions": int(cfg["deep_min_contributions"]),
                "deep_min_active_weeks": int(cfg["deep_min_active_weeks"]),
            }
        )
    return rows


def _split_balance_rows(
    cohort: pl.DataFrame, cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Class counts per temporal split and task."""
    if "split" not in cohort.columns:
        return []
    rows: list[dict[str, Any]] = []
    for split in SPLIT_ORDER:
        split_frame = cohort.filter(pl.col("split") == split)
        for task in ("a", "b"):
            labels = (
                split_frame.filter(pl.col(f"eligible_{task}"))
                .get_column(f"label_deep_{task}")
                .drop_nulls()
            )
            n_positive = int(labels.eq(True).sum())
            n_negative = int(labels.eq(False).sum())
            n_eligible = n_positive + n_negative
            rows.append(
                {
                    "task": task,
                    "split": split,
                    "n_users_total": split_frame.height,
                    "n_eligible": n_eligible,
                    "positive": n_positive,
                    "negative": n_negative,
                    "positive_rate": (n_positive / n_eligible) if n_eligible else None,
                }
            )
    return rows


def _plot_class_balance(balance: pl.DataFrame, path: Path) -> None:
    """Bar chart of positive vs negative counts for each task."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, task in zip(axes, ("a", "b")):
        row = balance.filter(pl.col("task") == task).row(0, named=True)
        values = [int(row["negative"]), int(row["positive"])]
        bars = axis.bar(["not deep", "deep"], values, color=["#a0aec0", TASK_COLORS[task]])
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
    fig.suptitle("Class balance among eligible users")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_entry_timeline(cohort: pl.DataFrame, path: Path) -> None:
    """Histogram of cohort entry dates, split by temporal split if present."""
    fig, axis = plt.subplots(figsize=(9, 4), constrained_layout=True)
    dates = [datetime.fromtimestamp(int(ts), tz=timezone.utc) for ts in cohort["t0"]]
    if "split" in cohort.columns:
        for split in SPLIT_ORDER:
            mask = (cohort["split"] == split).to_numpy()
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
    axis.set_ylabel("eligible users")
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
    """Positive rate by temporal split and task."""
    if split_balance.height == 0:
        return
    fig, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    width = 0.35
    x = np.arange(len(SPLIT_ORDER))
    for offset, task in ((-width / 2, "a"), (width / 2, "b")):
        rates = []
        for split in SPLIT_ORDER:
            row = split_balance.filter(
                (pl.col("split") == split) & (pl.col("task") == task)
            )
            rate = row["positive_rate"][0] if row.height else None
            rates.append(float(rate) if rate is not None else np.nan)
        axis.bar(x + offset, rates, width=width, label=TASK_LABELS[task], color=TASK_COLORS[task])
    axis.set_xticks(x, list(SPLIT_ORDER))
    axis.set_ylabel("positive rate")
    axis.set_title("Deep-engagement rate by temporal split")
    axis.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_summary(
    cohort: pl.DataFrame,
    cfg: Mapping[str, Any],
    balance: pl.DataFrame,
    split_balance: pl.DataFrame,
    data_end_utc: int | None,
) -> str:
    """Build the aggregate Markdown summary."""
    min_contributions = int(cfg["deep_min_contributions"])
    min_weeks = int(cfg["deep_min_active_weeks"])
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
    if data_end_utc is not None:
        end_date = datetime.fromtimestamp(int(data_end_utc), tz=timezone.utc).date()
        lines.append(f"- End of observation: **{end_date}**")
    lines.extend(
        [
            f"- Deep engagement: **>= {min_contributions} contributions** and "
            f"**>= {min_weeks} active weeks** in the {int(cfg['followup_days'])}-day label window",
            f"- Task A window: first {int(cfg['feature_window']['first_n'])} target "
            f"contributions, capped at {int(cfg['feature_window']['days'])} days",
            f"- Task B window: {int(cfg['pre_entry_days'])} days before entry, gateway subreddits only",
            "",
            "## Class balance (eligible users)",
            "",
            "| task | eligible | positive | negative | positive rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in balance.iter_rows(named=True):
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
                "## Temporal splits",
                "",
                "| split | users | task | eligible | positive rate |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for row in split_balance.iter_rows(named=True):
            rate = row["positive_rate"]
            rate_text = f"{rate:.1%}" if rate is not None else "n/a"
            lines.append(
                f"| {row['split']} | {row['n_users_total']:,} | "
                f"{TASK_LABELS[str(row['task'])]} | {row['n_eligible']:,} | {rate_text} |"
            )

    median_a = cohort.filter(pl.col("eligible_a"))["label_n_a"].median()
    median_b = cohort.filter(pl.col("eligible_b"))["label_n_b"].median()
    gateway_share = (
        cohort.filter(pl.col("eligible_b"))["n_gateway_pre_b"].gt(0).mean()
        if cohort.filter(pl.col("eligible_b")).height
        else None
    )
    lines.extend(
        [
            "",
            "## Aggregate activity",
            "",
            f"- Median post-window target contributions: Task A {median_a}, Task B {median_b}",
        ]
    )
    if gateway_share is not None:
        lines.append(
            f"- Eligible Task B users with any gateway activity: {gateway_share:.1%}"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "`class_balance.png`, `entry_timeline.png`, `post_window_activity.png`, "
            "`active_weeks.png`, `gateway_activity.png`, `entry_subreddits.png`, "
            "`split_positive_rates.png` (when splits exist).",
            "",
        ]
    )
    return "\n".join(lines)
