#!/usr/bin/env python
"""Generate every number, table and figure used by paper/main.tex.

Reads only aggregate files under ``reports/`` (round 2 = dataset ``real_r2``),
``data/processed/cohort_meta.json`` and the snapshot record. Writes
``paper/generated/*.tex`` (numbers as macros plus tables) and copies figures to
``paper/figures``. The outputs derive from real data and are gitignored.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import polars as pl  # noqa: E402

R = PROJECT_ROOT / "reports"
R2 = R / "results" / "real_r2"
GEN = PROJECT_ROOT / "paper" / "generated"
FIG = PROJECT_ROOT / "paper" / "figures"
PRIMARY = "label_deep_a_c20_w4_f90"
MODEL_LABEL = {
    "majority": "Majority class", "activity_lr": "Activity count (LR)",
    "behaviour_lgbm": "Behavioural LightGBM", "tfidf_lr": "TF-IDF (LR)",
    "combined_lr": "Behaviour + TF-IDF (LR)",
    "embedding_lr_minilm": "MiniLM mean embedding (LR)",
    "embedding_lgbm_minilm": "MiniLM mean embedding (LightGBM)",
}
FEATURE_LABEL = {
    "a_n": "contributions in window", "a_n_threads": "distinct threads",
    "a_gap_mean": "mean inter-event gap", "a_reply_to_comment_share": "share replying to comments",
    "a_time_to_nth": "time to 5th contribution", "a_gap_median": "median inter-event gap",
    "a_comment_share": "comment share", "a_active_days": "active days",
    "a_hour_entropy": "hour-of-day entropy", "a_weekend_share": "weekend share",
}
macros: list[str] = []


def num(x: float, d: int = 3) -> str:
    """Format a float."""
    return f"{x:.{d}f}"


def thousands(x: int | float) -> str:
    """Format an integer with thin-space-free commas."""
    return f"{int(x):,}".replace(",", "{,}")


def macro(name: str, value: str) -> None:
    """Register a LaTeX macro."""
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def ci(row: dict, m: str) -> str:
    """'x [lo, hi]' string."""
    return f"{num(row[m])} [{num(row[m + '_lo'])}, {num(row[m + '_hi'])}]"


def esc(s: str) -> str:
    """Escape LaTeX specials in labels."""
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def table(path: Path, header: list[str], rows: list[list[str]], spec: str) -> None:
    """Write a booktabs tabular."""
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Build all generated paper inputs."""
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    meta = json.loads((PROJECT_ROOT / "data/processed/cohort_meta.json").read_text())
    snap = json.loads((PROJECT_ROOT / "data/raw/SNAPSHOT_2026-10-24.json").read_text())["streams"]
    ing = (R / "ingest_real.log").read_text()
    m = re.search(r"read=(\d+) kept=(\d+).*distinct_authors=(\d+)", ing)
    macro("NRead", thousands(m.group(1))); macro("NKept", thousands(m.group(2)))
    macro("NAuthorsAll", thousands(m.group(3)))
    macro("NCohort", thousands(meta["n_users"])); macro("NEligA", thousands(meta["eligible_a"]))
    macro("NDeepA", thousands(meta["positive_a"]))
    macro("PctDeepA", f"{100 * meta['positive_a'] / meta['eligible_a']:.1f}")
    macro("NIncelsComments", thousands(snap["incels_comments"]["records"]))

    gate = pl.read_csv(R / "results" / "gate1_class_balance.csv").filter(pl.col("variant") == "c20_w4_f90")
    for sp, key in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        g = gate.filter(pl.col("split") == sp).to_dicts()[0]
        macro(f"N{key}", thousands(g["n"])); macro(f"Pos{key}", thousands(g["positives"]))
        macro(f"Rate{key}", f"{100 * g['pos_rate']:.1f}")

    a = pl.read_csv(R2 / f"metrics_real_r2_a_{PRIMARY}.csv")
    emb = pl.read_csv(R2 / f"metrics_real_r2_a_{PRIMARY}_minilm.csv")
    main_rows = pl.concat([a, emb], how="diagonal_relaxed").to_dicts()
    by = {r["model"]: r for r in main_rows}
    for k, nm in (("activity_lr", "Act"), ("behaviour_lgbm", "Lgbm"), ("tfidf_lr", "Tfidf"),
                  ("combined_lr", "Comb"), ("embedding_lgbm_minilm", "EmbLgbm"),
                  ("embedding_lr_minilm", "EmbLr")):
        macro(f"Auroc{nm}", num(by[k]["auroc"])); macro(f"AurocCI{nm}", ci(by[k], "auroc"))
        macro(f"Auprc{nm}", num(by[k]["auprc"]))
    macro("AuprcMaj", num(by["majority"]["auprc"]))
    table(GEN / "tab_main.tex", ["Model", "AUROC [95\\% CI]", "AUPRC [95\\% CI]", "F1", "ECE"],
          [[MODEL_LABEL[r["model"]], ci(r, "auroc"), ci(r, "auprc"), num(r["f1"]), num(r["ece"])]
           for r in main_rows], "lcccc")

    b = pl.read_csv(R2 / "metrics_real_r2_b_label_deep_b_c20_w4_f90.csv").to_dicts()
    macro("NTestB", thousands(b[0]["n_test"])); macro("PosTestB", thousands(b[0]["pos_test"]))
    non_maj = [r for r in b if r["model"] != "majority"]
    macro("AurocBLow", num(min(r["auroc"] for r in non_maj)))
    macro("AurocBHigh", num(max(r["auroc"] for r in non_maj)))

    # Round 2 contrast corpus is empty (IncelTears starts after the train period);
    # the valid-ish ablation is the superseded round 1 and is labelled as such.
    vm2 = re.search(r"train docs (\d+), contrast texts (\d+)", (R / "round2.log").read_text())
    macro("NContrastTextsRTwo", thousands(vm2.group(2)))
    R1 = R / "results" / "real_r1_superseded"
    vm1 = re.search(r"train docs (\d+), contrast texts (\d+)", (R / "vocab.log").read_text())
    macro("NVocabDocsROne", thousands(vm1.group(1))); macro("NContrastTextsROne", thousands(vm1.group(2)))
    a1 = pl.read_csv(R1 / f"metrics_real_a_{PRIMARY}.csv").to_dicts()
    base1 = {r["model"]: r for r in a1}
    vocab = pl.read_csv(R1 / "vocab_ablation_summary.csv").to_dicts()
    rows = [["0 (none)", num(base1["tfidf_lr"]["auroc"]), num(base1["combined_lr"]["auroc"])]]
    for k in sorted({r["k_removed"] for r in vocab}):
        t = next(r for r in vocab if r["k_removed"] == k and r["model"] == "tfidf_lr")
        c = next(r for r in vocab if r["k_removed"] == k and r["model"] == "combined_lr")
        rows.append([str(k), num(t["auroc"]), num(c["auroc"])])
    table(GEN / "tab_vocab.tex", ["Top terms removed", "TF-IDF AUROC", "Combined AUROC"], rows, "lcc")

    rob = pl.read_csv(R2 / "robustness_summary.csv").filter(pl.col("model") == "behaviour_lgbm").to_dicts()
    est90 = [r for r in rob if r["analysis"] != "primary"]
    macro("NRobust", str(len(rob)))
    macro("RobAurocLow", num(min(r["auroc"] for r in rob))); macro("RobAurocHigh", num(max(r["auroc"] for r in rob)))
    near = next(r for r in rob if "near" in r["analysis"])
    macro("AurocNear", num(near["auroc"]))
    comm = next(r for r in rob if r["analysis"].endswith("comments"))
    macro("AurocCommentsOnly", num(comm["auroc"]))
    ne = [x for x in (R2 / "robustness_not_estimable.txt").read_text().split() if x]
    macro("NNotEst", str(len(ne)))
    table(GEN / "tab_robust.tex", ["Analysis", "Test positives", "AUROC [95\\% CI]", "AUPRC"],
          [[esc(r["analysis"].replace("primary_excl_near_coverage_start", "primary, excl. near-start")),
            thousands(r["pos_test"]), ci(r, "auroc"), num(r["auprc"])] for r in rob], "lccc")

    imp = pl.read_csv(R2 / "permutation_importance_val.csv").head(6).to_dicts()
    macro("ImpTop", num(imp[0]["auprc_drop_mean"]))
    table(GEN / "tab_importance.tex", ["Feature", "Drop in validation AUPRC"],
          [[FEATURE_LABEL.get(r["feature"], esc(r["feature"])), num(r["auprc_drop_mean"])] for r in imp], "lc")

    mg = pl.read_csv(R / "mgtow" / "results" / "metrics_real_mgtow_a_label_deep_a_c20_w4_f90.csv").to_dicts()
    mgb = {r["model"]: r for r in mg}
    macro("NTestMg", thousands(mg[0]["n_test"])); macro("PosTestMg", thousands(mg[0]["pos_test"]))
    macro("AurocMgLgbm", num(mgb["behaviour_lgbm"]["auroc"])); macro("AurocMgTfidf", num(mgb["tfidf_lr"]["auroc"]))
    macro("AurocMgAct", num(mgb["activity_lr"]["auroc"]))

    table(GEN / "tab_streams.tex", ["Stream", "Records", "Complete", "Observed to"],
          [[esc(k.replace("_", "/")), thousands(v["records"] or 0), "yes" if v["complete"] else "no",
            (__import__("datetime").datetime.fromtimestamp(v["last_utc"], __import__("datetime").UTC).strftime("%Y-%m-%d")
             if v["last_utc"] else "--")] for k, v in snap.items()], "lccc")

    log = pl.read_csv(R / "test_log.csv")
    macro("NTestEvals", str(log.height))
    macro("MaxTestAuroc", num(float(log["auroc"].max())))

    (GEN / "numbers.tex").write_text("\n".join(macros) + "\n")
    for src, dst in ((R2 / "robustness_forest.png", "robustness_forest.png"),
                     (R2 / f"reliability_real_r2_a_{PRIMARY}.png", "reliability.png"),
                     (R2 / "shap_summary_val.png", "shap_summary.png")):
        shutil.copy(src, FIG / dst)
    print(f"wrote {len(macros)} macros and tables to {GEN}")


if __name__ == "__main__":
    main()
