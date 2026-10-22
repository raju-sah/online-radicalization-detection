"""Baseline and behavioural models with leakage-safe evaluation.

Models: majority class, activity-count-only logistic regression, behavioural
LightGBM, TF-IDF logistic regression and a combined (behaviour + TF-IDF)
logistic regression. Class imbalance is handled with class weights only;
nothing is resampled. Hyperparameters and the F1 decision threshold are chosen
on the validation split; the test split is scored once per final configuration
and every test evaluation is appended to ``reports/test_log.csv``.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("majority", "activity_lr", "behaviour_lgbm", "tfidf_lr", "combined_lr")
TEST_LOG_FIELDS = [
    "timestamp", "dataset", "task", "label", "model", "config_hash",
    "n_test", "pos_test", "auroc", "auprc", "f1", "ece",
]


@dataclass
class Data:
    """Aligned arrays for one split."""

    keys: np.ndarray
    X: np.ndarray
    docs: list[str]
    y: np.ndarray


def ece_score(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pl.DataFrame:
    """Aggregate reliability-curve bins (counts, mean prediction, observed rate)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": b, "n": int(mask.sum()),
                         "mean_pred": float(p[mask].mean()),
                         "observed": float(y[mask].mean())})
    return pl.DataFrame(rows)


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def _safe_ap(y: np.ndarray, p: np.ndarray) -> float:
    return float(average_precision_score(y, p)) if y.sum() > 0 else float("nan")


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on the supplied (validation) data."""
    if y.sum() == 0:
        return 0.5
    best, thr = -1.0, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99))):
        f = f1_score(y, p >= t, zero_division=0)
        if f > best:
            best, thr = f, float(t)
    return thr


def metrics(y: np.ndarray, p: np.ndarray, thr: float, bins: int = 10) -> dict[str, float]:
    """AUROC, AUPRC, F1 at ``thr``, ECE, Brier."""
    return {
        "auroc": _safe_auc(y, p),
        "auprc": _safe_ap(y, p),
        "f1": float(f1_score(y, p >= thr, zero_division=0)),
        "ece": ece_score(y, p, bins),
        "brier": float(np.mean((p - y) ** 2)),
    }


def bootstrap_ci(
    y: np.ndarray, p: np.ndarray, thr: float, n: int, seed: int, bins: int = 10
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs from a user-level bootstrap of the evaluation set."""
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {k: [] for k in ("auroc", "auprc", "f1", "ece")}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0 or y[i].sum() == len(i):
            continue
        m = metrics(y[i], p[i], thr, bins)
        for k in draws:
            draws[k].append(m[k])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"),) * 2
        for k, v in draws.items()
    }


def _fit_lr(X: Any, y: np.ndarray, C: float, seed: int) -> LogisticRegression:
    return LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed).fit(X, y)


def _lgbm(params: Mapping[str, Any], common: Mapping[str, Any], seed: int, pos_weight: float):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        **params, **common, scale_pos_weight=pos_weight, random_state=seed,
        n_jobs=2, verbose=-1,
    )


def fit_predict_all(
    train: Data, val: Data, test: Data, cfg: Mapping[str, Any], count_idx: list[int]
) -> dict[str, dict[str, Any]]:
    """Fit every model (tuning on val only) and return val/test probabilities.

    Returns:
        ``{model: {"val": p_val, "test": p_test, "config": {...}}}``.
    """
    mc = cfg["models"]
    seed = int(mc["seed"])
    out: dict[str, dict[str, Any]] = {}
    prior = float(train.y.mean())
    out["majority"] = {
        "val": np.full(len(val.y), prior), "test": np.full(len(test.y), prior),
        "config": {"strategy": "train_prior"},
    }

    def pick(candidates):
        best = None
        for conf, predict in candidates:
            pv = predict("val")
            score = _safe_ap(val.y, pv)
            if best is None or (score > best[0]):
                best = (score, conf, predict)
        _, conf, predict = best
        return {"val": predict("val"), "test": predict("test"), "config": conf}

    sets = {"train": train, "val": val, "test": test}

    # activity-count-only LR
    scaler = StandardScaler().fit(train.X[:, count_idx])
    def act_cands():
        for C in mc["lr_c_grid"]:
            m = _fit_lr(scaler.transform(train.X[:, count_idx]), train.y, C, seed)
            yield {"C": C}, (lambda s, m=m: m.predict_proba(scaler.transform(sets[s].X[:, count_idx]))[:, 1])
    out["activity_lr"] = pick(act_cands())

    # behavioural LightGBM
    pos_w = float((1 - prior) / max(prior, 1e-9))
    def lgb_cands():
        for g in mc["lgbm_grid"]:
            m = _lgbm(g, mc["lgbm_common"], seed, pos_w).fit(train.X, train.y)
            yield dict(g), (lambda s, m=m: m.predict_proba(sets[s].X)[:, 1])
    out["behaviour_lgbm"] = pick(lgb_cands())

    # TF-IDF fit on train only
    tf = dict(mc["tfidf"]); tf["ngram_range"] = tuple(tf["ngram_range"])
    vec = TfidfVectorizer(**tf)
    try:
        vec.fit(train.docs)
        T = {k: vec.transform(v.docs) for k, v in sets.items()}
    except ValueError:  # empty vocabulary
        T = {k: sp.csr_matrix((len(v.y), 1)) for k, v in sets.items()}
    def tf_cands():
        for C in mc["lr_c_grid"]:
            m = _fit_lr(T["train"], train.y, C, seed)
            yield {"C": C}, (lambda s, m=m: m.predict_proba(T[s])[:, 1])
    out["tfidf_lr"] = pick(tf_cands())

    # combined
    sc2 = StandardScaler().fit(train.X)
    def comb(s):
        return sp.hstack([sp.csr_matrix(sc2.transform(sets[s].X)), T[s]]).tocsr()
    def comb_cands():
        for C in mc["lr_c_grid"]:
            m = _fit_lr(comb("train"), train.y, C, seed)
            yield {"C": C}, (lambda s, m=m: m.predict_proba(comb(s))[:, 1])
    out["combined_lr"] = pick(comb_cands())
    return out


def config_hash(obj: Any) -> str:
    """Short stable hash of a JSON-serialisable configuration."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def log_test_evaluation(path: Path, row: dict[str, Any], allow_repeat: bool) -> None:
    """Append one test evaluation; refuse repeated test scoring of a final config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open() as fh:
            existing = list(csv.DictReader(fh))
    key = ("dataset", "task", "label", "model", "config_hash")
    if not allow_repeat and any(all(r[k] == str(row[k]) for k in key) for r in existing):
        raise RuntimeError(
            f"Test split already evaluated for {[row[k] for k in key]}; "
            "tuning against test results is forbidden."
        )
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TEST_LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in TEST_LOG_FIELDS})


def evaluate_task(
    feats: pl.DataFrame,
    texts: pl.DataFrame,
    cohort: pl.DataFrame,
    task: str,
    label_col: str,
    cfg: Mapping[str, Any],
    results_dir: Path,
    test_log: Path,
    dataset: str,
    exclude_cols: tuple[str, ...] = (),
    allow_repeat: bool = False,
) -> pl.DataFrame:
    """Run all models for one task/label and write aggregate results.

    Args:
        feats: Behavioural features keyed by ``author_hash``.
        texts: In-window text per user (``doc`` column).
        cohort: Cohort with ``split`` and the label column.
        task: ``"a"`` or ``"b"``.
        label_col: Boolean label column; nulls (ineligible) are dropped.
        cfg: Full config.
        results_dir: Output directory (aggregate CSV and reliability data).
        test_log: Path to ``reports/test_log.csv``.
        dataset: ``"real"`` or ``"synthetic"``-style tag recorded in the log.
        exclude_cols: Feature columns to drop (ablations).
        allow_repeat: Permit repeated test evaluation (synthetic only).

    Returns:
        One row per model with test metrics, CIs, threshold and class balance.
    """
    included = f"task_b_included_f{cfg['followup_days']}" if task == "b" else None
    frame = cohort.filter(pl.col(label_col).is_not_null() & pl.col("split").is_in(["train", "val", "test"]))
    if included:
        frame = frame.filter(pl.col(included))
    frame = (
        frame.select("author_hash", "split", pl.col(label_col).cast(pl.Int64).alias("_y"))
        .join(feats, on="author_hash").join(texts, on="author_hash")
    )
    cols = [c for c in feats.columns if c != "author_hash" and c not in exclude_cols]
    count_idx = [cols.index(f"{task}_n")]

    def mk(split: str) -> Data:
        f = frame.filter(pl.col("split") == split)
        return Data(f["author_hash"].to_numpy(), f.select(cols).to_numpy().astype(float),
                    f["doc"].to_list(), f["_y"].to_numpy())

    train, val, test = mk("train"), mk("val"), mk("test")
    for name, d in (("train", train), ("val", val), ("test", test)):
        if len(d.y) == 0 or d.y.sum() == 0 or d.y.sum() == len(d.y):
            raise ValueError(f"Split {name} has a degenerate label distribution ({len(d.y)} users).")

    mc = cfg["models"]
    preds = fit_predict_all(train, val, test, cfg, count_idx)
    rows, rel = [], []
    for model in MODEL_NAMES:
        p = preds[model]
        thr = best_f1_threshold(val.y, p["val"])
        m = metrics(test.y, p["test"], thr, mc["ece_bins"])
        ci = bootstrap_ci(test.y, p["test"], thr, int(mc["bootstrap_n"]), int(mc["seed"]), mc["ece_bins"])
        chash = config_hash({"model": model, **p["config"], "excl": sorted(exclude_cols)})
        row = {
            "dataset": dataset, "task": task, "label": label_col, "model": model,
            "config": json.dumps(p["config"]), "config_hash": chash, "threshold": thr,
            "n_train": len(train.y), "n_val": len(val.y), "n_test": len(test.y),
            "pos_rate_train": float(train.y.mean()), "pos_rate_val": float(val.y.mean()),
            "pos_rate_test": float(test.y.mean()),
            "pos_test": int(test.y.sum()),
            **m,
        }
        for k, (lo, hi) in ci.items():
            row[f"{k}_lo"], row[f"{k}_hi"] = lo, hi
        row["leakage_flag"] = bool(m["auroc"] > float(mc["leakage_auroc_flag"]))
        rows.append(row)
        log_test_evaluation(
            test_log,
            {**row, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "pos_test": row["pos_test"]},
            allow_repeat,
        )
        rt = reliability_table(test.y, p["test"], mc["ece_bins"]).with_columns(
            pl.lit(model).alias("model"))
        rel.append(rt)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = pl.DataFrame(rows)
    tag = f"{dataset}_{task}_{label_col}" + (f"_excl{len(exclude_cols)}" if exclude_cols else "")
    out.write_csv(results_dir / f"metrics_{tag}.csv")
    pl.concat(rel).write_csv(results_dir / f"reliability_{tag}.csv")
    _reliability_plot(pl.concat(rel), results_dir / f"reliability_{tag}.png", tag)
    return out


def _reliability_plot(rel: pl.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for model in MODEL_NAMES:
        d = rel.filter(pl.col("model") == model)
        if d.height > 1:
            ax.plot(d["mean_pred"], d["observed"], marker="o", ms=3, label=model)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
