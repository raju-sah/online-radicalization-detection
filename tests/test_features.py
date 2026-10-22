"""Feature windows must not see records after the cutoff (Task A) or t0 (Task B)."""

from __future__ import annotations

import polars as pl

from radpath.cohort import build_cohort
from radpath.features import build_task_features

DAY = 86_400
T0 = 1_600_000_000
TARGETS = ["incels"]
GATEWAYS = ["lonely", "foreveralone"]


def _rows(extra: list[dict]) -> list[dict]:
    rows = [
        {"author_hash": "u", "subreddit": "incels", "created_utc": T0 + i * 3600, "text": f"hello {i}"}
        for i in range(8)
    ]
    rows += [
        {"author_hash": "u", "subreddit": "lonely", "created_utc": T0 - 10 * DAY, "text": "pre"},
        {"author_hash": "u", "subreddit": "foreveralone", "created_utc": T0 - 2 * DAY, "text": "pre2"},
    ]
    return rows + extra


def _feats(make_contributions, params, extra, task):
    frame = make_contributions(_rows(extra))
    cohort = build_cohort(
        frame, params, TARGETS, GATEWAYS, data_end_utc=T0 + 1000 * DAY
    )
    return build_task_features(frame, cohort, task, TARGETS, GATEWAYS)


def test_task_a_ignores_records_after_cutoff(make_contributions, params):
    late = [
        {"author_hash": "u", "subreddit": "incels", "created_utc": T0 + 20 * DAY + i, "text": "zzz leak"}
        for i in range(50)
    ]
    base_f, base_t = _feats(make_contributions, params, [], "a")
    more_f, more_t = _feats(make_contributions, params, late, "a")
    assert base_f.equals(more_f)
    assert base_t.equals(more_t)
    assert base_f["a_n"][0] == 5
    assert "leak" not in more_t["doc"][0]


def test_task_b_uses_only_pre_entry_gateway(make_contributions, params):
    post = [
        {"author_hash": "u", "subreddit": "lonely", "created_utc": T0 + i + 1, "text": "after"}
        for i in range(30)
    ]
    base_f, _ = _feats(make_contributions, params, [], "b")
    more_f, more_t = _feats(make_contributions, params, post, "b")
    assert base_f.equals(more_f)
    assert base_f["b_n"][0] == 2
    assert base_f["b_gw_lonely"][0] == 1
    assert "after" not in more_t["doc"][0]
