#!/usr/bin/env python
"""Phase 9: aggregate interpretation of the final behavioural LightGBM (Task A).

Refits the model with the config logged in Phase 5 (deterministic seed), then
reports validation-set permutation importance, a SHAP summary on a validation
sample, and test performance/calibration by entry year and era. Everything is
aggregate; no per-user prediction or explanation is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from radpath.cohort import CohortParams  # noqa: E402
from radpath.config import ensure_data_dirs, load_config  # noqa: E402
from radpath.coverage import parse_utc  # noqa: E402
from radpath.features import build_task_features  # noqa: E402
from radpath.ingest import load_contributions  # noqa: E402
from radpath.models import _lgbm, _safe_ap, _safe_auc, ece_score  # noqa: E402


def group_table(frame: pl.DataFrame, p: np.ndarray, group: str, bins: int) -> pl.DataFrame:
    """Aggregate performance and calibration for each group (min 30 users)."""
    rows = []
    y = frame["_y"].to_numpy()
    for g in sorted(frame[group].unique().to_list()):
        m = (frame[group] == g).to_numpy()
        if m.sum() < 30:
            continue
        rows.append({
            group: g, "n": int(m.sum()), "positives": int(y[m].sum()), "pos_rate": float(y[m].mean()),
            "mean_pred": float(p[m].mean()),
            "auroc": _safe_auc(y[m], p[m]) if 0 < y[m].sum() < m.sum() else float("nan"),
            "auprc": _safe_ap(y[m], p[m]), "ece": ece_score(y[m], p[m], bins),
        })
    return pl.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """Run permutation importance, SHAP and subgroup analyses."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset", default="real")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = ensure_data_dirs(cfg)
    params = CohortParams.from_config(cfg)
    mc = cfg["models"]
    seed = int(mc["seed"])
    label = f"label_deep_a_{params.primary_variant.name}"
    results = paths["reports"] / "results" / args.dataset
    logged = pl.read_csv(results / f"metrics_{args.dataset}_a_{label}.csv").filter(pl.col("model") == "behaviour_lgbm")
    conf = json.loads(logged["config"][0])

    cohort = pl.read_parquet(paths["processed"] / "cohort.parquet")
    users = cohort.filter(pl.col(label).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
    feats, _ = build_task_features(
        load_contributions(paths["interim"]), users, "a", cfg["target_subreddits"], cfg["gateway_subreddits"],
        first_n=params.first_n, feature_days=params.feature_days,
    )
    ban = parse_utc(cfg["eras"]["incels_ban_utc"])
    frame = users.select(
        "author_hash", "split", "t0", pl.col(label).cast(pl.Int64).alias("_y"),
        pl.col("t0").map_elements(lambda t: datetime.fromtimestamp(t, timezone.utc).year, return_dtype=pl.Int64).alias("entry_year"),
        pl.when(pl.col("t0") < ban).then(pl.lit("pre_incels_ban")).otherwise(pl.lit("post_incels_ban")).alias("era"),
    ).join(feats, on="author_hash")
    cols = [c for c in feats.columns if c != "author_hash"]
    tr, va, te = (frame.filter(pl.col("split") == s) for s in ("train", "val", "test"))
    ytr = tr["_y"].to_numpy()
    prior = float(ytr.mean())
    model = _lgbm(conf, mc["lgbm_common"], seed, (1 - prior) / prior).fit(tr.select(cols).to_numpy(), ytr)
    Xva, yva = va.select(cols).to_numpy(), va["_y"].to_numpy()
    pte = model.predict_proba(te.select(cols).to_numpy())[:, 1]
    print("refit test AUROC", _safe_auc(te["_y"].to_numpy(), pte), "logged", logged["auroc"][0], flush=True)

    # Permutation importance on VALIDATION (never test), drop in AUPRC.
    rng = np.random.default_rng(seed)
    base = _safe_ap(yva, model.predict_proba(Xva)[:, 1])
    imp = []
    for j, c in enumerate(cols):
        drops = []
        for _ in range(5):
            Xp = Xva.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(base - _safe_ap(yva, model.predict_proba(Xp)[:, 1]))
        imp.append({"feature": c, "auprc_drop_mean": float(np.mean(drops)), "auprc_drop_sd": float(np.std(drops))})
    imp_df = pl.DataFrame(imp).sort("auprc_drop_mean", descending=True)
    imp_df.write_csv(results / "permutation_importance_val.csv")
    print(imp_df.head(10))

    # SHAP summary on a validation sample (aggregate plot only).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    idx = rng.choice(len(Xva), size=min(2000, len(Xva)), replace=False)
    sv = shap.TreeExplainer(model).shap_values(Xva[idx])
    sv = sv[1] if isinstance(sv, list) else sv
    mean_abs = np.abs(sv).mean(axis=0)
    pl.DataFrame({"feature": cols, "mean_abs_shap": mean_abs}).sort("mean_abs_shap", descending=True).write_csv(
        results / "shap_mean_abs_val.csv"
    )
    shap.summary_plot(sv, Xva[idx], feature_names=cols, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(results / "shap_summary_val.png", dpi=150)
    plt.close()

    # Test performance / calibration by entry year and era.
    bins = int(mc["ece_bins"])
    by_year = group_table(te, pte, "entry_year", bins)
    by_era = group_table(te, pte, "era", bins)
    by_year.write_csv(results / "test_by_entry_year.csv")
    by_era.write_csv(results / "test_by_era.csv")
    print(by_year)
    print(by_era)
    # Validation breakdown (descriptive; spans earlier eras than test).
    pva = model.predict_proba(Xva)[:, 1]
    group_table(va, pva, "entry_year", bins).write_csv(results / "val_by_entry_year.csv")
    group_table(va, pva, "era", bins).write_csv(results / "val_by_era.csv")
    group_table(tr, model.predict_proba(tr.select(cols).to_numpy())[:, 1], "entry_year", bins).select(
        "entry_year", "n", "positives", "pos_rate"
    ).write_csv(results / "train_by_entry_year_counts.csv")
    print(pl.read_csv(results / "val_by_entry_year.csv"))
    print(pl.read_csv(results / "val_by_era.csv"))
    print(pl.read_csv(results / "train_by_entry_year_counts.csv"))


if __name__ == "__main__":
    main()
