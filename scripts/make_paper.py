#!/usr/bin/env python
"""Generate the current preprint inputs from aggregate release results.

Requires local ignored analysis outputs. No individual records are read. All
numeric paper results are derived from these aggregates, never typed by hand.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from radpath.config import load_config, resolve_paths

LABELS = {"majority":"Majority class", "activity_lr":"Activity LR",
          "behaviour_lr":"Metadata LR", "behaviour_lgbm":"Metadata LightGBM"}


def esc(text: str) -> str:
    """Escape plain table labels for TeX."""
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def integer(value: int | float | str) -> str:
    """Format integer macros without TeX punctuation ambiguity."""
    return f"{int(value):,}".replace(",", "{,}")


def interval(row: dict, metric: str) -> str:
    """Format a point estimate and its saved bootstrap interval."""
    return f"{row[metric]:.3f} [{row[metric+'_lo']:.3f}, {row[metric+'_hi']:.3f}]"


def table(path: Path, header: list[str], rows: list[list[str]], spec: str) -> None:
    """Write one booktabs table with no embedded data identifiers."""
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Build tables, numeric macros, and aggregate-only figures."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/release.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = resolve_paths(cfg)
    reports = paths["reports"]
    dataset = cfg["release_analysis"]["dataset"]
    results = reports / "results" / dataset
    gen, fig = PROJECT_ROOT / "paper/generated", PROJECT_ROOT / "paper/figures"
    gen.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    macros = []
    def macro(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")
    meta = json.loads((paths["processed"] / "cohort_meta.json").read_text())
    snap = json.loads((paths["raw"] / "SNAPSHOT_2026-10-24.json").read_text())["streams"]
    match = re.search(r"read=(\d+) kept=(\d+).*distinct_authors=(\d+)", (reports / "ingest_real.log").read_text())
    if not match:
        raise ValueError("Aggregate ingestion counts unavailable.")
    for name, value in zip(("NRead", "NKept", "NAuthorsAll"), match.groups()):
        macro(name, integer(value))
    for name, key in (("NCohort","n_users"), ("NEligA","eligible_a"), ("NDeepA","positive_a")):
        macro(name, integer(meta[key]))
    macro("PctDeepA", f"{100*meta['positive_a']/meta['eligible_a']:.1f}")
    summary = pl.read_csv(results / "summary.csv")
    primary = summary.filter(pl.col("analysis") == "primary")
    by = {r["model"]:r for r in primary.to_dicts()}
    for model, suffix in (("activity_lr","Act"), ("behaviour_lr","MetaLr"), ("behaviour_lgbm","Lgbm")):
        r = by[model]
        macro("Auroc"+suffix, f"{r['auroc']:.3f}")
        macro("AurocCI"+suffix, interval(r,"auroc"))
        macro("Auprc"+suffix, f"{r['auprc']:.3f}")
    table(gen / "tab_release.tex", ["Model", "AUROC [95\\% CI]", "AUPRC [95\\% CI]", "F1", "ECE", "Brier"],
          [[LABELS[r['model']],interval(r,'auroc'),interval(r,'auprc'),f"{r['f1']:.3f}",f"{r['ece']:.3f}",f"{r['brier']:.3f}"] for r in primary.to_dicts()], "lccccc")
    balance = pl.read_csv(results / "class_balance.csv")
    balance_rows=[]
    def day(value):
        return datetime.fromtimestamp(int(value), timezone.utc).strftime("%Y-%m-%d")
    for split, suffix in (("train","Train"),("val","Val"),("test","Test")):
        row = balance.filter(pl.col("split")==split).row(0,named=True)
        rate=100*row['positives']/row['n']
        for name,val in (("N",row['n']),("Pos",row['positives'])):
            macro(name+suffix,integer(val))
        macro("Rate"+suffix,f"{rate:.1f}")
        balance_rows.append([split,integer(row['n']),integer(row['positives']),f"{rate:.1f}\\%",day(row['entry_start_utc']),day(row['entry_end_utc'])])
    table(gen / "tab_balance.tex",["Split","Users","Positives","Positive rate","Earliest entry","Latest entry"],balance_rows,"lrrrll")
    robust=summary.filter(pl.col("model")=="behaviour_lgbm")
    macro("NRobust",robust.height)
    macro("RobAurocLow",f"{robust['auroc'].min():.3f}")
    macro("RobAurocHigh",f"{robust['auroc'].max():.3f}")
    def analysis_label(row):
        if row['analysis']=='comments_only': return "Comments-only outcome"
        if row['analysis']=='exclude_near_start': return "Exclude near-start entrants"
        match=re.search(r"c(\d+)_w(\d+)_f90",row['label'])
        return f"{match[1]} contributions, {match[2]} weeks"+(" (primary)" if row['analysis']=='primary' else "")
    table(gen / "tab_release_robust.tex",["Analysis","Test users","Positives","AUROC [95\\% CI]","AUPRC","F1"],
          [[analysis_label(r),integer(r['n_test']),integer(r['pos_test']),interval(r,'auroc'),f"{r['auprc']:.3f}",f"{r['f1']:.3f}"] for r in robust.to_dicts()],"lrrccc")
    historical_b=pl.read_csv(reports / "results/real_r2/metrics_real_r2_b_label_deep_b_c20_w4_f90.csv").row(0,named=True)
    macro("NTestB",integer(historical_b['n_test']))
    macro("PosTestB",integer(historical_b['pos_test']))
    table(gen / "tab_streams.tex",["Stream","Records","Retrieval complete","Observed through"],
          [[esc(k.replace('_','/')),integer(v['records'] or 0),'yes' if v['complete'] else 'no',day(v['last_utc']) if v['last_utc'] else '--'] for k,v in snap.items()],"lrcl")
    (gen / "numbers.tex").write_text("\n".join(macros)+"\n")
    rel=pl.read_csv(results / f"reliability_{dataset}_a_label_deep_a_c20_w4_f90_primary.csv")
    fig_cal,ax=plt.subplots(figsize=(3.5,3.1))
    ax.plot([0,1],[0,1],'k--',lw=.8)
    for model,label in LABELS.items():
        rows=rel.filter((pl.col('model')==model)&(pl.col('n')>=cfg['release_analysis']['min_aggregate_cell']))
        ax.plot(rows['mean_pred'],rows['observed'],marker='o',ms=3,label=label)
    ax.set(xlabel='Mean predicted probability',ylabel='Observed positive rate',xlim=(0,1),ylim=(0,1))
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    ax.legend(fontsize=6)
    fig_cal.tight_layout(); fig_cal.savefig(fig/'release_reliability.png',dpi=220); plt.close(fig_cal)
    imp=pl.read_csv(results / 'permutation_importance_val.csv').head(8).reverse()
    fig_imp,ax=plt.subplots(figsize=(3.5,3.2))
    feature_labels = {'a_n':'Contributions', 'a_n_threads':'Distinct threads',
                      'a_time_to_nth':'Time to fifth contribution', 'a_gap_std':'Gap standard deviation',
                      'a_reply_to_comment_share':'Reply-to-comment share', 'a_gap_mean':'Mean gap',
                      'a_gap_median':'Median gap', 'a_hour_entropy':'UTC hour entropy'}
    labels=[feature_labels.get(x,x.removeprefix('a_').replace('_',' ')) for x in imp['feature']]
    ax.barh(labels,imp['auprc_drop_mean'],xerr=imp['auprc_drop_sd'],color='#245b78',capsize=2)
    ax.set_xlabel('Drop in validation AUPRC',fontsize=8)
    ax.set_xlim(min(0.0, (imp['auprc_drop_mean']-imp['auprc_drop_sd']).min())-0.002,
                (imp['auprc_drop_mean']+imp['auprc_drop_sd']).max()+0.004)
    ax.tick_params(labelsize=7)
    fig_imp.tight_layout(); fig_imp.savefig(fig/'release_importance.png',dpi=220); plt.close(fig_imp)
    print(f"generated {len(macros)} numeric macros, 4 tables and 2 aggregate figures")


if __name__ == '__main__':
    main()
