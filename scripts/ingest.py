#!/usr/bin/env python
"""Stream raw .zst dumps into cleaned, username-hashed Parquet.

Examples::

    python scripts/ingest.py \
        --submissions "data/raw/*submissions*.zst" \
        --comments "data/raw/*comments*.zst"

Pass directories instead of globs to ingest every ``.zst`` inside them.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from radpath.clean import CleanRules  # noqa: E402
from radpath.config import (  # noqa: E402
    configured_subreddits,
    ensure_data_dirs,
    get_salt,
    load_config,
)
from radpath.ingest import FileSpec, ingest_files  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--submissions",
        nargs="+",
        default=[],
        help="Submission .zst files, globs, or directories.",
    )
    parser.add_argument(
        "--comments",
        nargs="+",
        default=[],
        help="Comment .zst files, globs, or directories.",
    )
    parser.add_argument(
        "--no-subreddit-filter",
        action="store_true",
        help="Keep records from all subreddits instead of configured lists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Rows per Parquet write (default: ingest.batch_size).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if an output Parquet file already exists.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def expand_inputs(values: list[str]) -> list[Path]:
    """Expand files, glob patterns and directories into sorted paths."""
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if candidate.is_dir():
            paths.extend(sorted(candidate.glob("*.zst")))
        elif candidate.exists():
            paths.append(candidate)
        else:
            paths.extend(sorted(Path(match) for match in glob.glob(value, recursive=True)))
    return [path for path in paths if path.is_file()]


def main(argv: list[str] | None = None) -> None:
    """Run ingestion for the requested record kinds."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)

    submissions = expand_inputs(args.submissions)
    comments = expand_inputs(args.comments)
    if not submissions and not comments:
        raise SystemExit("No input files found; pass --submissions and/or --comments.")

    for kind, files in (("submissions", submissions), ("comments", comments)):
        stems = [path.stem for path in files]
        duplicates = {stem for stem in stems if stems.count(stem) > 1}
        if duplicates:
            raise SystemExit(f"Duplicate {kind} file stems would collide: {sorted(duplicates)}")

    salt = get_salt()
    rules = CleanRules.from_config(cfg)
    restrict = bool(cfg.get("ingest", {}).get("restrict_to_config_subreddits", True))
    subreddit_filter = (
        None if args.no_subreddit_filter or not restrict else configured_subreddits(cfg)
    )
    batch_size = args.batch_size or int(cfg.get("ingest", {}).get("batch_size", 20_000))

    specs = [
        *(FileSpec(path=path, kind="submissions") for path in submissions),
        *(FileSpec(path=path, kind="comments") for path in comments),
    ]
    stats = ingest_files(
        specs=specs,
        interim_dir=paths["interim"],
        salt=salt,
        rules=rules,
        batch_size=batch_size,
        subreddit_filter=subreddit_filter,
        overwrite=args.overwrite,
    )
    print(f"Ingested {stats.files} file(s): {stats.summary()}")
    print(f"Interim Parquet: {paths['interim']}")


if __name__ == "__main__":
    main()
