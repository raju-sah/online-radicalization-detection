"""Tests for record-level cleaning rules."""

from __future__ import annotations

import pytest

from radpath.clean import (
    CleanRules,
    compose_text,
    drop_reason,
    bot_rule,
    is_bot,
    is_deleted_author,
    is_deleted_value,
    is_empty_text,
)


def test_bot_detection_is_case_insensitive_and_configurable() -> None:
    rules = CleanRules(
        bot_name_suffixes=("bot",),
        extra_bot_names=frozenset({"automoderator", "botdefense"}),
    )
    assert is_bot("HelperBot", rules)
    assert is_bot("helperbot", rules)
    assert is_bot("AutoModerator", rules)
    assert is_bot("BotDefense", rules)
    assert not is_bot("Roberta", rules)
    assert not is_bot("human_person", rules)
    assert not is_bot(None, rules)

    no_suffix = CleanRules(bot_name_suffixes=(), extra_bot_names=frozenset())
    assert not is_bot("HelperBot", no_suffix)


def test_bot_rule_separates_explicit_names_from_suffix_heuristic() -> None:
    rules = CleanRules(
        bot_name_suffixes=("bot",),
        extra_bot_names=frozenset({"automoderator"}),
    )
    assert bot_rule("AutoModerator", rules) == "explicit"
    assert bot_rule("HelperBot", rules) == "suffix"
    assert bot_rule("human_person", rules) is None
    assert bot_rule(None, rules) is None


def test_deleted_values_cover_markers_and_blanks() -> None:
    for value in ("[deleted]", "[removed]", "  ", "", None):
        assert is_deleted_value(value)
        assert is_deleted_author(value)
    assert not is_deleted_value("still here")
    assert not is_deleted_author("some_user")

    rules = CleanRules(deleted_values=frozenset({"[gone]"}))
    assert is_deleted_value("[gone]", rules)
    assert not is_deleted_value("[deleted]", rules)


def test_compose_text_ignores_removed_parts() -> None:
    rules = CleanRules()
    assert compose_text("submission", "Title", "Body", "", rules) == "Title\n\nBody"
    assert compose_text("submission", "Title", "[removed]", "", rules) == "Title"
    assert compose_text("submission", "Title", "[deleted]", "", rules) == "Title"
    assert compose_text("comment", None, None, "hello", rules) == "hello"
    assert compose_text("comment", None, None, "[removed]", rules) == ""


def test_empty_text_requires_all_text_fields_to_be_blank() -> None:
    assert is_empty_text("submission", "", "", "")
    assert is_empty_text("submission", "   ", None, None)
    assert is_empty_text("comment", None, None, "[deleted]")
    assert not is_empty_text("submission", "Title only", "", "")
    assert not is_empty_text("submission", "", "Body only", "")
    assert not is_empty_text("comment", None, None, "valid")


@pytest.mark.parametrize(
    ("author", "expected"),
    [
        ("[deleted]", "deleted_author"),
        ("[removed]", "deleted_author"),
        ("AutoModerator", "bot_author"),
        ("SomeBot", "bot_author"),
        ("real_user", None),
    ],
)
def test_drop_reason(author: str, expected: str | None) -> None:
    reason = drop_reason(author, "comment", None, None, "text")
    assert reason == expected


def test_drop_reason_keeps_known_authors_with_empty_or_removed_content() -> None:
    assert drop_reason("real_user", "comment", None, None, "") is None
    assert drop_reason("real_user", "comment", None, None, "[removed]") is None
    assert drop_reason("real_user", "submission", "", "", "") is None
    assert drop_reason("real_user", "comment", None, None, "x") is None


def test_rules_from_config() -> None:
    cfg = {
        "clean": {
            "bot_name_suffixes": ["bot", "_robot"],
            "extra_bot_names": ["AutoModerator", "Helper"],
            "deleted_values": ["[deleted]"],
        }
    }
    rules = CleanRules.from_config(cfg)
    assert is_bot("Helper", rules)
    assert is_bot("name_robot", rules)
    assert is_deleted_value("[deleted]", rules)
    assert not is_deleted_value("[removed]", rules)
