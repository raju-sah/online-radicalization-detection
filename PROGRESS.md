# radpath progress

Last updated: 2026-10-22T15:55Z

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
| 9. Interpretation | pending | Depends on Phase 5. |
| 10. Paper draft | pending | Depends on Phases 5-9; then STOP at Gate 2. |

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
