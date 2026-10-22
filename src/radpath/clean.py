"""Cleaning rules for raw Reddit records.

Cleaning happens *before* author hashing inside :mod:`radpath.ingest`, because
bot detection needs the raw username. These helpers are pure and do not log
or expose author names.

Deleted/removed/empty *content* is not dropped: it still records a real
account's activity. Only records whose author is unknown or automated, or
whose required identifiers are missing, are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DELETED_AUTHORS = frozenset({"[deleted]", "[removed]"})


@dataclass(frozen=True)
class CleanRules:
    """Configurable record-level cleaning rules."""

    deleted_values: frozenset[str] = frozenset({"[deleted]", "[removed]"})
    bot_name_suffixes: tuple[str, ...] = ("bot",)
    extra_bot_names: frozenset[str] = frozenset({"automoderator"})

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "CleanRules":
        """Build rules from the ``clean`` section of a radpath config."""
        clean_cfg = cfg.get("clean", {})
        suffixes = tuple(
            str(value).strip().lower()
            for value in clean_cfg.get("bot_name_suffixes", ["bot"])
            if str(value).strip()
        )
        extras = frozenset(
            str(value).strip().lower()
            for value in clean_cfg.get("extra_bot_names", ["AutoModerator"])
            if str(value).strip()
        )
        deleted = frozenset(
            str(value).strip().lower()
            for value in clean_cfg.get("deleted_values", ["[deleted]", "[removed]"])
        )
        return cls(
            deleted_values=deleted,
            bot_name_suffixes=suffixes,
            extra_bot_names=extras,
        )


def is_deleted_value(value: Any, rules: CleanRules | None = None) -> bool:
    """Return True for deleted/removed markers and empty values."""
    active_rules = rules or CleanRules()
    if value is None:
        return True
    text = str(value).strip()
    return not text or text.lower() in active_rules.deleted_values


def is_deleted_author(author: Any, rules: CleanRules | None = None) -> bool:
    """Return True when an author is missing or deleted/removed."""
    return is_deleted_value(author, rules)


def bot_rule(author: Any, rules: CleanRules | None = None) -> str | None:
    """Return which bot rule matched an author, or None.

    ``"explicit"`` means an exact configured name; ``"suffix"`` means the
    configurable username-suffix heuristic. Matching is case-insensitive.
    """
    if author is None:
        return None
    active_rules = rules or CleanRules()
    name = str(author).strip().lower()
    if not name:
        return None
    if name in active_rules.extra_bot_names:
        return "explicit"
    if any(name.endswith(suffix) for suffix in active_rules.bot_name_suffixes if suffix):
        return "suffix"
    return None


def is_bot(author: Any, rules: CleanRules | None = None) -> bool:
    """Return True for configured bots and usernames ending in a bot suffix.

    This is deliberately conservative: it is better to drop a borderline
    automated account than to model it as a person, and the rule is documented
    as a limitation.
    """
    return bot_rule(author, rules) is not None


def compose_text(
    kind: str,
    title: Any,
    selftext: Any,
    body: Any,
    rules: CleanRules | None = None,
) -> str:
    """Combine the textual fields of a submission or comment.

    Deleted/removed markers and empty fields are omitted. The result is
    stripped but otherwise left as-is for later (deferred) text work.
    """
    active_rules = rules or CleanRules()
    if kind == "comment":
        return "" if is_deleted_value(body, active_rules) else str(body).strip()

    parts = [
        str(value).strip()
        for value in (title, selftext)
        if value is not None and not is_deleted_value(value, active_rules)
    ]
    return "\n\n".join(parts)


def is_empty_text(
    kind: str,
    title: Any,
    selftext: Any,
    body: Any,
    rules: CleanRules | None = None,
) -> bool:
    """Return True when a record has no usable text for its kind."""
    return not compose_text(kind, title, selftext, body, rules).strip()


def drop_reason(
    author: Any,
    kind: str,
    title: Any,
    selftext: Any,
    body: Any,
    rules: CleanRules | None = None,
) -> str | None:
    """Return a cleaning reason for dropping a record, or None to keep it.

    Possible reasons: ``deleted_author`` and ``bot_author``. Content that is
    deleted, removed or empty is kept with ``has_text = False`` because it is
    still evidence of account activity.
    """
    active_rules = rules or CleanRules()
    if is_deleted_author(author, active_rules):
        return "deleted_author"
    if is_bot(author, active_rules):
        return "bot_author"
    return None
