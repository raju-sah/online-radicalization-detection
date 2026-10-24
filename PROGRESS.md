# radpath progress

Last updated: 2026-10-24T15:26Z

This file is the running memory for the autonomous roadmap. Re-read it after a
long context break. The roadmap is in `ROADMAP.md`; the frozen analysis choices
are in `PREREGISTRATION.md`.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 1. Download speed-up + commit | done | Shared 1 req/s limiter, `--workers 3`, priority ordering, duplicate-page advance; download running. |
| 2. r/incels smoke test | done | Ran 2026-10-23 on complete incels comments+submissions; no schema/parse issues. Tag `phase-2`. |
| 3. Synthetic features + baselines | done | `features.py`, `models.py`, `scripts/run_models.py`; verified on a 4,000-user synthetic fixture. Tag `phase-3`. |
| 4. Full real-data run | done | Run on frozen snapshot (DEVIATIONS D1). Gate 1 PASS. Tag `phase-4`. |
| 5. Real-data baselines | done | Task A solid; Task B too small to interpret (test n=13). Tag `phase-5`. |
| 6. Sentence embeddings | done | MiniLM on CPU in ~40 min. Tag `phase-6`. |
| 7. Vocabulary ablation | pending | Depends on Phase 5. |
| 8. Robustness + sensitivity | done | 20 analyses + MGTOW contrast (D2). Tag `phase-8`. |
| 9. Interpretation | done (analysis) | Outputs written; VALIDITY ISSUE found, see below. Tag `phase-9`. Phase 10 not started. |
| 10. Paper draft | done — STOP (Gate 2) | `paper/main.tex` + `scripts/make_paper.py`; awaiting user review. Tag `phase-10`. |

## Current environment

- Git HEAD at Phase 1 record: `26b391f` (before the roadmap commit).
- Download process: PID `3071601`, PGID `3071601`, running
  `scripts/download_data.py --yes --force-api --workers 3`.
- Disk guard: PID `3071679`, stops the download if free space falls below 5 GiB.
- Last disk check: about 22 GiB free on `/`.
- Do not stop or restart the download for roadmap bookkeeping.

## Phase 1 record (done)

Commits already present at the start of this autonomous run:

```
26b391f fix: advance past duplicate-only API pages
fd937ac feat: add shared rate limiter and concurrent download workers
22038fd feat: validate coverage and robustness configuration
8d0ba12 feat: preregister design and extend aggregate EDA
25c52a5 feat: add coverage-aware labels and embargoed date splits
b984265 feat: retain deleted content and track ingestion coverage
8f70ff3 docs: note time-series coverage limits
ac58a81 feat: add magnet fallback and API size planning
4f4f577 docs: document acquisition workflow and rules
0394a3c feat: add controlled torrent and API acquisition
```

Implementation summary:

- One global rate limiter is shared by all workers; the total request rate is
  capped at one request per second.
- `--workers` defaults to 3.
- Download priority is incels, Braincels, IncelsWithoutHate, MGTOW,
  IncelTears, ForeverAlone, then the remaining configured communities.
- Duplicate-only API pages are advanced by one second after a bounded number
  of identical-page retries, which cleared the old `incels/submissions` stall.
- Tests: `pytest -q` reported 62 passed at the end of Phase 1.

Download status when this record was written:

- Five-minute sample: 60,355 to 87,946 records, +27,591, about 91.97 records/s
  combined across workers.
- Checkpoints were still incomplete:
  - `incels_comments`: 86,214 records, incomplete, last_utc 1476482376.
  - `incels_submissions`: 24,328 records, incomplete, last_utc 1495422597.
  - `braincels_comments`: 35,695 records, incomplete, last_utc 1514822246.
  - `braincels_submissions`: 9,845 records, incomplete, last_utc 1519934378.
- The API returned HTTP 422 for one `incels/comments` page; the downloader
  retried with adjusted cursors. This is a known API edge case, not a reason to
  stop the process.

No r/incels smoke test has been run, so Phase 2 is not done. Phase 3 may use
synthetic fixtures only; it must not read real data or write to
`data/processed`.

## Roadmap bookkeeping

- `ROADMAP.md` was created from the user-supplied roadmap and committed as
  `f1d4f1e`.
- Phase 1 tag: `phase-1` (created after this record).
- Next phase to work: Phase 3 on synthetic data, with periodic checks for the
  two incels checkpoints. When both are complete, run the incels-only smoke
  test, fix schema/parse issues, append the result here, and tag `phase-2`.

## Decisions and deviations

- Research scope decision from the roadmap: targets are incels, Braincels, and
  IncelsWithoutHate. MGTOW remains a downloaded comparison community and must
  not define entry or labels. A dated amendment belongs in
  `PREREGISTRATION.md` before any modelling.
- No deviations recorded yet; create `DEVIATIONS.md` when the first change to
  the preregistered plan is needed.

## Open issues

- Phase 2 waits on both incels checkpoints.
- Phase 4 waits on all target and control downloads.
- Any test AUROC above 0.95 on Task A or B is a mandatory stop for leakage
  investigation.

## Phase 3 record (done, 2026-10-22)

- Preregistration Amendment 1: targets = incels, Braincels, IncelsWithoutHate; MGTOW moved to `control_subreddits` (comparison only). Commit `f58cf67`.
- New: `src/radpath/features.py` (activity, interaction, VADER/length text stats, TF-IDF docs, Task B gateway counts and recency; all windowed by cutoff, tested in `tests/test_features.py`), `src/radpath/models.py` (majority, activity-count LR, behavioural LightGBM, TF-IDF LR, combined LR; class weights only; hyperparameters and F1 threshold chosen on validation; ECE; 1,000-resample user-level bootstrap CIs; `reports/test_log.csv` guard that refuses a second test scoring of the same final config), `scripts/run_models.py`, `models:` block in `configs/default.yaml`.
- Synthetic run (scratchpad, not real data): Task A activity-count models reach AUROC/AUPRC 1.0 because the fixture's engagement profiles are deterministic in early volume; flagged by `leakage_flag` as designed. Not evidence about real data. Task B synthetic AUPRC ~0.58-0.62 vs prevalence 0.57.
- `make_fixture.py` gained `--entry-span-days`; synthetic runs need a wide entry span and `robustness.followup_days: [90]` or the 180-day embargo empties validation.
- Tests: 67 passed.
- Real runs must not use `--dataset synthetic*`; real runs exit with STOP if any test AUROC > 0.95.
- Download (PID 3071601) still running; checkpoints exist for incels, braincels, incelswithouthate (comments + submissions); disk 22 GiB free.

## Download issue (2026-10-22 ~16:20Z)

- The `incels/comments` worker died at 21:39 local with a persistent HTTP 422 at after=1476482376 (checkpoint count 86,214, not complete). Other workers continue under PID 3071601 and are not touched.
- Fix committed (not yet running; the live process loads old code): on persistent 422 the client now tries limit=100/10 at the same cursor, then skips forward 1/10/60 s and records the skip in `client.skipped_gaps` (logged as WARNING "coverage gap"). Decision: a gap of seconds is accepted and will be reported in the data section.
- Next action: when PID 3071601 exits, rerun `scripts/download_data.py --yes --force-api --workers 3` to resume from checkpoints (not before, per instruction not to restart).

## Download restart (2026-10-23 ~02:45 local)

- First run (PID 3071601) exited with HTTP 422 errors on 17 of 22 files (log: `reports/download_run1.log`); incels/submissions, incelswithouthate/submissions complete.
- Resumed from checkpoints with the patched 422 handler: PID 3273123 (`--workers 3`), disk guard restarted. ~25-30 rec/s per stream.
- Phase 2 smoke test runs when incels comments+submissions are both complete.

## STOP: Arctic Shift unavailable (2026-10-23 ~10:05 local)

- Run 2 (PID 3273123, ~7 h) added ~1.1M incels/braincels/mgtow comments, then every stream failed with HTTP 422 at ~09:56 (also on first requests with no cursor), and a relaunch at 10:03 got HTTP 500 on all streams ("blocked repeatedly ... stopping downloads").
- State: incels submissions, incelswithouthate submissions, braincels submissions complete. incels comments 912,843 (to 2017-09-14), braincels comments ~0.7M+, mgtow comments ~0.38M+; other streams partial or empty. All checkpoints intact; resume with the same command.
- Per AGENTS.md rule 4 (stop and report if repeatedly blocked; do not evade limits) the download is stopped and nothing is retried. Phase 2/4 blocked. Waiting for the user to decide (wait for API recovery vs. torrent route needing aria2 via pkexec).

## Recovery (2026-10-23 12:48 local)

- Torrent route tried (dry run): magnet metadata found no peers in 600 s, so the script fell back to the API. Torrent not viable now.
- API recovered; resumed with the same command, PID 3685315, disk guard restarted. incels comments back at work (checkpoint ~912k to 2017-09-14).

## Phase 2 record (done, 2026-10-23 ~14:50 local)

- incels comments (1,131,720) and submissions (54,215) complete. Check done while the download (PID 3685317) kept running; the download was not touched.
- Smoke run in `/tmp/radpath-smoke` (config `smoke.yaml`, targets=[incels] only): ingest read 1,185,935 records, kept 906,607 (deleted authors 268,821, bot authors 10,507; 0 malformed, 0 bad timestamps), 39,120 distinct hashed authors; cohort 38,991 users; EDA wrote without error. 45 s total.
- No parse/schema fixes were needed. Results deliberately not interpreted.

## API outage #2 (2026-10-23 19:08 local)

- Run 3 (PID 3685317, 6.3 h) added incels comments to completion, braincels comments to 2018-09-04 (1.68M), mgtow comments to 2017-11-12 (1.15M), incelswithouthate comments to 2020-08-15 (0.75M). At 19:08 all streams failed simultaneously with HTTP 422, same signature as outage #1 (which cleared after ~2.5 h). Complete: 4 of 22 streams.
- Policy: one resume attempt about every 90 min; stop and report if it keeps failing. No evasion.

## Recovery #2 (2026-10-23 20:41 local)

- API recovered after ~1.5 h; resumed with the same command, PID 4044957, disk guard reattached.

## Session restart (2026-10-24 07:45 local)

- The download died silently at 21:15 on 10-23 when the previous Claude session ended (no API error). Restarted as a systemd user unit (`radpath-download`, PID 17646) so it survives session restarts; disk guard is unit `radpath-diskguard`. Status/stop: `systemctl --user status radpath-download`.
- Checkpoints: incels (both), braincels/incelswithouthate submissions complete; comments: braincels 1.74M (to 2018-09-13), mgtow 1.22M (2017-11-28), incelswithouthate 0.81M (2020-09-03). 4 of 22 complete. Free disk 23 GiB.

## Phase 4 record (done, 2026-10-24)

- User decision (2026-10-24, "Freeze now"): stop the download, analyse the partial snapshot. Logged as DEVIATIONS.md D1 (snapshot table, coverage overrides for Braincels end 2018-09-19 and IncelsWithoutHate end 2020-09-17). Download service stopped; snapshot in `data/raw/SNAPSHOT_2026-10-24.json` (gitignored).
- Fixture leftovers in data/interim and data/processed were moved to the scratchpad; real data ingested from the 18 non-empty real streams (fixture_* files excluded).
- Ingest: read 6,571,877, kept 5,228,874; deleted authors 1,282,507, bots 60,496; 0 malformed, 0 bad timestamps; 359,515 distinct hashed authors.
- Cohort: 102,551 users. Task A eligible 67,727 (5,192 deep, 7.7%); 35,545 embargoed (null split). Task B included only 880 (94 deep) because gateway streams are mostly missing.
- Gate 1 (`scripts/check_gate1.py`, `reports/results/gate1_class_balance.csv`): primary c20_w4_f90 train n=28,947 / 1,919 positive (6.6%); val 3,931 / 277 (7.0%); test 10,324 / 870 (8.4%). PASS (>=300 train, >=100 test).

## Phase 5 record (done, 2026-10-24)

Primary label c20_w4_f90, test evaluated once per final config; logged in `reports/test_log.csv`; results in `reports/results/real/` (metrics CSV with bootstrap CIs, reliability CSV/PNG). All test AUROC < 0.95 (no leakage stop).

Task A (train 28,947 @6.6% pos; test 10,324 @8.4% pos, 870 positives):

| Model | AUROC (95% CI hi shown in CSV) | AUPRC | F1 | ECE |
|---|---|---|---|---|
| majority | 0.500 | 0.084 | 0.155 | 0.018 |
| activity-count LR | 0.808 | 0.218 | 0.365 | 0.269 |
| behavioural LightGBM | 0.836 | 0.318 | 0.390 | 0.260 |
| TF-IDF LR | 0.689 | 0.145 | 0.206 | 0.387 |
| combined LR | 0.837 | 0.318 | 0.398 | 0.265 |

- Class-weighted models are badly calibrated in absolute terms (ECE ~0.26-0.39) since weights inflate probabilities; ranking metrics are the meaningful ones. Post-hoc calibration was not preregistered; will not tune on test.
- Task B: only 450 train / 13 test users (1 test positive); numbers are uninterpretable and must be reported as such (caused by D1: gateway streams missing).
- Notable: activity-count alone gives AUROC 0.81, so early volume carries most of the signal; text adds little over behaviour.

## Phase 6 record (done, 2026-10-24)

- all-MiniLM-L6-v2 on CPU (torch 2.14 CPU, max_seq_length 128), 109,369 feature-window texts for 43,202 labelled Task A users, ~40 min, cached resumably as 22 chunks in `data/processed/embeddings` (gitignored). `scripts/run_embeddings.py`, `models.evaluate_dense`.
- Task A test (n=10,324, 870 positives): embedding LR AUROC 0.570 / AUPRC 0.091; embedding LightGBM AUROC 0.812 / AUPRC 0.253 / F1 0.324. Both below the best behavioural model (0.836 / 0.318) and no leakage flag.
- Reading: pooled semantic vectors add nothing beyond behaviour. LR at 0.57 is below TF-IDF LR (0.69); LightGBM likely recovers volume/no-text information from the vectors (users without text get zero vectors), which LR cannot. Not tuned further.

## Phase 7 record (done, 2026-10-24)

- Weighted log-odds (informative Dirichlet prior, Monroe et al. 2008), target = train-split Task A feature-window text (28,947 user docs) vs IncelTears texts (8,608, submissions only, created <= last train entry). No hand-written term list. Ranking in `data/processed/vocab_ranking.csv` (local, gitignored; not reported).
- Text models retrained with top-k terms removed (test, once per config; `reports/results/real/vocab_ablation_summary.csv`):

| k removed | TF-IDF LR AUROC | combined LR AUROC |
|---|---|---|
| 0 (Phase 5) | 0.689 | 0.837 |
| 50 | 0.686 | 0.837 |
| 200 | 0.684 | 0.837 |
| 1000 | 0.691 | 0.837 |

- Reading: no measurable drop, so the (weak) text signal is not carried by the most community-specific vocabulary. Limitation: the contrast corpus is IncelTears submissions only (D1: no IncelTears comments), while target text is mostly comments, so the ranking partly captures submission-vs-comment register; the null result should be read with that in mind.

## Phase 8 record (done, 2026-10-24)

- `scripts/run_robustness.py`: all 18 threshold x follow-up variants, the comments-only variant, and the primary label excluding near_coverage_start users, each tested once (tag `robust` / `excl_near_start`, three models: activity LR, behavioural LightGBM, combined LR). Summary `reports/results/real/robustness_summary.csv` and forest plot `robustness_forest.png`.
- Behavioural LightGBM across the 20 analyses: AUROC 0.774 (c10_w4_f180) to 0.858 (c50_w3_f90); AUPRC 0.149 to 0.408 (AUPRC tracks prevalence). Primary 0.836; comments-only 0.838; excluding near-coverage-start 0.835. 180-day variants are lower and use fewer users (test n=5,889) because of coverage truncation. No leakage flag.
- Task B robustness was not extended: only 450 / 13 train/test users (D1).
- MGTOW contrast (`scripts/run_mgtow_contrast.py`, `configs/mgtow.yaml`, dataset `real_mgtow`, reported separately): comments only, entrants to 2017-09; needed DEVIATIONS D2 (eligible-only split, 90-day grid, 0.5/0.3/0.2 fractions). Test n=4,833, 592 positives (12.2%): activity LR AUROC 0.842, behavioural LightGBM 0.875 (AUPRC 0.486), TF-IDF LR 0.760, combined 0.879 (AUPRC 0.512). Not comparable to the incels runs (different era, splits, and positives rate); only the qualitative pattern (behaviour >> text) matches.

## Phase 9 record (done, 2026-10-24) and STOP: late-period cohort validity

Outputs (`scripts/run_interpretation.py`; refit reproduces logged test AUROC 0.8365 exactly; all in `reports/results/real/`): validation permutation importance (`permutation_importance_val.csv`), SHAP mean-|value| and summary plot (validation sample), test/val by entry year and era.
- Permutation importance (drop in validation AUPRC, behavioural LightGBM): `a_n` (contributions in the window) 0.158 by far; then reply-to-comment share 0.019, distinct threads 0.019, mean inter-event gap 0.017, time-to-Nth 0.014, comment share 0.011; text-derived features ~0.00-0.004.
- Test entrants are all 2019-08 to 2020-06 (n=10,324; 9,601 enter via IncelsWithoutHate, 723 via Braincels); val 2018-09-19 to 2019-02 (2,114 Braincels, 1,817 IWH); train 2014-01 to 2018-02 (24,936 incels). By-year test AUROC 0.86 (2019), 0.82 (2020); val 0.87. Because the whole test split is post r/incels ban there is no within-test pre/post-ban contrast; the era comparison is confounded with the train/test split. Braincels-quarantine split not possible (snapshot).

**VALIDITY ISSUE (needs a decision; not covered by ROADMAP/AGENTS/PREREG, so I stopped).** Braincels comments are observed only to 2018-09-19 (Braincels submissions run to 2019-08-31, when the sub ended), and the coverage override pins its coverage end at 2018-09-19. After that date the cohort rule treats Braincels as "not existing", so (a) users who were really active in Braincels in 2018-09..2019-08 but appear in the data first through IncelsWithoutHate (or via a stray Braincels submission) get a spurious "entry" t0, and (b) their post-cutoff Braincels activity is unobserved, biasing labels toward negative. This affects every val/test user (val >= 2018-09-19, test 2019-08..2020-06), so the headline test numbers (Phase 5-9) are measured on a population of questionable entry validity. It is not leakage (features stay before cutoff) but it is selection/measurement error, found after seeing test results. All test evaluations to date are in `reports/test_log.csv`.

Options (user decision):
- A (recommended): restrict the cohort to entrants with t0 <= 2018-09-19, the last date all target streams are observed; re-split; rerun Phases 4-9 as a documented second round (DEVIATIONS D3, old numbers kept and labelled superseded). Fast (hours). Smaller, older-era sample.
- B: keep as is, disclose as a limitation. Fastest, weakest.
- C: resume the Braincels comments download through 2019-08 (likely 1+ days, API flaky), then rerun. Best data, slow. A and C can be combined (A now, C later).

## Round 2 (D3) STOP: Gate 1 fails mechanically (2026-10-24)

- User chose option A. Implemented `entry_cutoff_utc` (2018-09-19) in config/cohort (+test; 70 tests pass), logged DEVIATIONS D3, preserved round 1 (`data/processed/cohort_r1.parquet`, `embeddings_r1`, `reports/results/real_r1_superseded`, `gate1_class_balance_r1.csv`).
- Round-2 cohort: 76,512 users; Task A eligible 46,726 (3,677 deep); Task B 722 with gateway (80 deep).
- Gate 1 with the standard 70/15/15 split over all entrants: train 24,961 users / 1,473 positives, val 0, test 0 => FAIL. Cause: entrants are heavily concentrated in 2018 (quantile 70% = 2018-05-04, 85% = 2018-07-07) but labels are observable only for t0 <= ~2018-06-21, so val/test slices hold only label-null users; the 180-day embargo from the robustness grid then removes the rest.
- Split alternatives computed on label-eligible users only (`temporal_split`, users/positives):
  - 0.70/0.15/0.15: train 24,955/1,472, val 0, test 6,998/751 (val empty).
  - 0.5/0.25/0.25, embargo by primary 90-day windows only: train 16,002/980, val 3,093/294, test 11,616/1,201 (test = 2018-H1 entrants).
  - 0.5/0.25/0.25, embargo by 180-day windows: train 9,507/634, val 1,569/113, test 11,616/1,201.
- Needs a user decision (splitting rule is part of the preregistration); nothing else has been run in round 2.

## Round 2 record (complete, 2026-10-24) — supersedes the Phase 5-9 numbers above

Decisions: D3 (entrants <= 2018-09-19), D4 (eligible-only 0.50/0.25/0.25 split, 90-day embargo), D5 (near-start flag anchored at 2016-07-01). Dataset tag `real_r2`; results in `reports/results/real_r2/`; round-1 in `reports/results/real_r1_superseded/`; all test evaluations in `reports/test_log.csv` (71 real_r2 rows; none with AUROC > 0.95).

Cohort 76,512; Task A eligible 46,726 (3,677 deep, 7.9%). Gate 1 PASS: train 16,002 / 980 pos (6.1%), val 3,093 / 294 (9.5%), test 11,616 / 1,201 (10.3%). Entrants span 2016-07 to 2018-09 (incels data start 2016-07; train = 2016-2017 incels-era, val 2017-2018, test 2018).

Task A test (AUROC [95% CI] / AUPRC / F1): majority 0.500 / 0.103 / 0.187; activity-count LR 0.788 [0.778,0.798] / 0.231 / 0.379; behavioural LightGBM 0.815 [0.802,0.826] / 0.289 / 0.400; TF-IDF LR 0.696 [0.682,0.711] / 0.197 / 0.246; combined LR 0.834 [0.823,0.846] / 0.331 / 0.435; MiniLM+LightGBM 0.796 [0.785,0.807] / 0.273 / 0.348; MiniLM+LR 0.550 / 0.106 / 0.201. Task B: 142 test users (17 positives), AUROC 0.53-0.61, intervals include 0.5; uninterpretable.
Vocabulary ablation (top 50/200/1000 terms removed): TF-IDF LR 0.696/0.697/0.697, combined 0.834 throughout; no drop. Same IncelTears-submissions-only caveat as round 1 (contrast text count differs only by cohort).
Robustness (behavioural LightGBM, 11 analyses): AUROC 0.802 to 0.840 across 9 threshold variants with 90-day follow-up; comments-only 0.818; excluding near-start users (v2 flag, 893 flagged) 0.831. Nine 180-day variants are NOT ESTIMABLE (windows run past observed coverage given the entry cutoff; test split empty) — see `robustness_not_estimable.txt`.
Interpretation: permutation importance (validation, AUPRC drop): contributions in the window `a_n` 0.102, distinct threads 0.044, mean inter-event gap 0.028, reply-to-comment share 0.021, time-to-Nth 0.019; text features ~0. Test set is entirely 2018 entrants (post r/incels ban) so there is no within-test era contrast; the validation set spans 2017 (AUROC 0.80) and 2018 (n=137, AUROC 0.71).
MGTOW contrast (Phase 8, `real_mgtow`) was not rerun: it uses its own cohort/config (D2) and does not depend on the incels/Braincels cohort rules; behavioural LightGBM AUROC 0.875 there.
Caveats to carry into the paper: incels data effectively start 2016-07 (entry left-censored for early users); Braincels/IWH comments partial; Task B and IncelTears ablation are weak (D1); class-weighted models are miscalibrated (ECE ~0.2-0.4) and no post-hoc calibration was preregistered; early volume (`a_n`) dominates, so the 'early signal' is largely how quickly a user posts.

## Phase 10 record (done, 2026-10-24) — STOP at Gate 2: ready for review

- `paper/main.tex` (single-file two-column article draft; abstract, intro, related work with TODO citations only, data+ethics, methods, results, limitations, conclusion, appendices). `scripts/make_paper.py` generates `paper/generated/numbers.tex` (57 macros) and 6 tables from `reports/` and copies 3 figures to `paper/figures`; no number is typed by hand. Generated tables/figures are gitignored (derived from real data); regenerate with `python scripts/make_paper.py`.
- No LaTeX compiler (pdflatex/latexmk/tectonic) on this machine, so the PDF has NOT been built or visually checked; the layout is a plain two-column article, to be swapped for the official ACL `acl.sty`.
- Items for the user at review:
  1. Round-2 vocabulary ablation is invalid: the IncelTears contrast corpus has 0 texts because IncelTears submissions begin after the training period (train ends ~2017-08). The paper reports the superseded round-1 ablation (8,608 contrast texts) as indicative only. Decision needed: relax the contrast-time restriction (violates "train split only" in spirit) or drop the ablation / get IncelTears comments.
  2. All citations are TODO placeholders (no invented references); ethics/IRB statement is a TODO.
  3. Test numbers rest on D1-D5 deviations (partial data, post-hoc cohort restriction after seeing round-1 test results, unusual split). Disclosed in the paper and DEVIATIONS.md.
  4. Task B is not estimable (142 test users); the 180-day sensitivity variants are not estimable.
  5. Calibration: class-weighted models have ECE 0.2-0.4; no post-hoc calibration (not preregistered).
- Reports/test_log.csv holds every test evaluation (rounds `real`, `real_r2`, `real_mgtow`); max AUROC in any is below 0.95.

## Gate 2 decision (2026-10-24)

- User accepted the recommendation for the vocabulary ablation: keep the round-1 result as indicative only and disclose that the round-2 contrast corpus is empty (current paper text). No new ablation run, no new deviation. Remaining before submission: build the PDF with the official ACL style, fill citation/ethics TODOs.

## Exploratory preprint release preparation (2026-10-24)

- Author confirmed Raju Sah as sole author, Independent researcher, ORCID 0009-0006-0412-6139. No formal ethics review was sought; no external funding or competing interests declared. Manuscript CC BY 4.0, supporting code MIT. AI assistance from OpenAI Codex and Anthropic Claude is disclosed without claiming completed author verification.
- D6 freezes the corrected analysis in `configs/release.yaml`. Dataset `release_counts_v1` uses the existing round-two cohort and split, excludes archived scores/removal status/text, adds metadata-only LR, and corrects the majority decision threshold to 0.5. All 11 declared analyses completed; 44 unique model evaluations are in `reports/test_log.csv`. The prior test population was already inspected, so no independent confirmation is claimed.
- Primary test n=11,616, positives=1,201 (10.3%). Activity LR: AUROC 0.788, AUPRC 0.231, F1 0.379, ECE 0.299, Brier 0.208. Metadata LR: 0.841 / 0.341 / 0.439 / 0.270 / 0.185. Metadata LightGBM: 0.842 / 0.341 / 0.441 / 0.249 / 0.168. The corrected majority baseline has F1 0, AUROC 0.5, AUPRC 0.103, ECE 0.042, Brier 0.094. Calibration remains poor; metadata LightGBM sensitivity AUROC spans 0.829-0.859. No release evaluation crosses the 0.95 trigger.
- Ingestion now discards all free text before persistence. Existing interim files were minimized in 18 streaming rewrites with all 5,228,874 rows retained. The legacy vocabulary list was moved to excluded raw storage. Raw archives remain local; no data was acquired during this work.
- Rewrote the current `paper/main.tex` in place as an exploratory metadata-only preprint. Replaced citation placeholders with eight verified references, added disclosures, narrowed content/causal claims, and described source missingness, post-hoc choices and probabilistic miscalibration. The generator produces 30 numeric macros, four tables and two aggregate-only figures; both figures were inspected. Historical text and embedding results are not used as evidence for the corrected paper.
- Added licenses, citation information, draft Zenodo metadata, tested dependency versions, an environment inventory and an allowlisted package builder. Source-only review archives are under `output/release/` and are ignored by Git. No user-level data, hashes, text, logs, embeddings or secrets are packaged.
- A full repeat of the release command reused all 44 saved evaluations, with no additional test-log rows.
- Tests: 75 passed. Regression checks reproduced the free-text retention and majority/model-selection defects before fixes, then passed. Creation-time feature invariance and idempotent text minimization are tested on synthetic data. `pip check` passes.
- Remaining release blocker: the built-in LaTeX compiler cannot fetch its uncached bundle from `relay.fullyjustified.net`; the bundle endpoint also returned HTTP 403 in a connectivity check. Three compilation attempts failed before TeX processing. No separate PDF was compiled and no other editor tab was opened. The current source remains in the original editor. PDF compilation, whole-document visual inspection, author scientific review, and the final Zenodo publication remain pending. The source-only package is explicitly marked incomplete.
