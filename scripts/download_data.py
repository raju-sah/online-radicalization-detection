#!/usr/bin/env python
"""Download configured subreddit dumps through the approved routes.

This is the **only** permitted data-acquisition entry point. It fetches only
subreddits listed in ``configs/default.yaml`` and never uses per-user or
per-author endpoints.

Examples::

    python scripts/download_data.py --dry-run
    python scripts/download_data.py --yes
    python scripts/download_data.py --dry-run --only incels Braincels --type comments
    python scripts/download_data.py --yes --force-api --only lonely
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radpath.config import load_config  # noqa: E402
from radpath.download import acquire, configured_targets  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the route/size plan without downloading.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually download the planned files.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of configured subreddits (case-insensitive).",
    )
    parser.add_argument(
        "--type",
        choices=("comments", "submissions"),
        default=None,
        help="Restrict to one record type.",
    )
    parser.add_argument(
        "--force-api",
        action="store_true",
        help="Skip torrents and use the Arctic Shift API route.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs that are not marked complete.",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Refuse --yes when free space would drop below this (default: config).",
    )
    parser.add_argument(
        "--ignore-disk-check",
        action="store_true",
        help="Allow --yes despite the free-space cushion check.",
    )
    parser.add_argument(
        "--refresh-torrent",
        action="store_true",
        help="Re-download cached torrent metadata.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run and args.yes:
        parser.error("--dry-run and --yes are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> None:
    """Plan or execute downloads for configured targets."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    cfg = load_config(args.config)
    try:
        targets = configured_targets(cfg, only=args.only, kind=args.type)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not targets:
        raise SystemExit("No configured targets selected.")

    dry_run = not args.yes
    plan = acquire(
        cfg,
        targets,
        dry_run=dry_run,
        force_api=args.force_api,
        overwrite=args.overwrite,
        min_free_gb=args.min_free_gb,
        ignore_disk_check=args.ignore_disk_check,
        refresh_torrent=args.refresh_torrent,
    )
    if dry_run:
        print("Dry run only. Pass --yes to download.")
        return

    final = {(item.target.subreddit, item.target.kind): item.status for item in plan}
    not_found = sorted(
        f"{subreddit}/{kind}" for (subreddit, kind), status in final.items() if status == "not_found"
    )
    incomplete = sorted(
        f"{subreddit}/{kind}"
        for (subreddit, kind), status in final.items()
        if status not in ("complete", "not_found", "skip")
    )
    if not_found:
        print("Not found in both routes: " + ", ".join(not_found))
    if incomplete:
        print("Incomplete/error (rerun to resume): " + ", ".join(incomplete))


if __name__ == "__main__":
    main()
