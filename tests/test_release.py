"""Release privacy and feature-availability checks on synthetic data only."""
from pathlib import Path
import importlib.util
import polars as pl
import pyarrow.parquet as pq
from radpath.features import build_task_features
from radpath.cohort import build_cohort


def test_release_features_ignore_mutable_metadata(make_contributions, params):
    rows = [{"author_hash":"synthetic", "subreddit":"incels", "created_utc":1600000000+i,
             "score":i, "text":"synthetic mention", "has_text":True} for i in range(5)]
    original = make_contributions(rows)
    cohort = build_cohort(original, params, ["incels"], ["lonely"], data_end_utc=1700000000)
    changed = original.with_columns(pl.lit(99999).alias("score"), pl.lit(False).alias("has_text"),
                                    pl.lit("edited content").alias("text"))
    before, text_before = build_task_features(original, cohort, "a", ["incels"], ["lonely"], creation_time_only=True)
    after, text_after = build_task_features(changed, cohort, "a", ["incels"], ["lonely"], creation_time_only=True)
    assert before.equals(after)
    assert text_before.equals(text_after)
    assert text_before["doc"].to_list() == [""]
    assert not any("score" in c or "text" in c or "sent" in c for c in before.columns)


def test_legacy_minimization_preserves_metadata(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/minimize_text.py"
    spec = importlib.util.spec_from_file_location("minimize_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    file = tmp_path / "synthetic.parquet"
    frame = pl.DataFrame({"author_hash":["synthetic_hash"], "created_utc":[1600000000],
                          "text":["u/synthetic_mention"], "body":["synthetic name"], "has_text":[True]})
    frame.write_parquet(file)
    assert module.minimize(file, 1) == 1
    cleaned = pl.read_parquet(file)
    assert cleaned.drop("text", "body").equals(frame.drop("text", "body"))
    assert cleaned["text"].to_list() == [""]
    assert cleaned["body"].to_list() == [""]
    assert module.minimize(file, 1) == 1
    assert pl.read_parquet(file).equals(cleaned)
