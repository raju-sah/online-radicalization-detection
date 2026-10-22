# radpath roadmap (autonomous)

## Working rules
- Follow AGENTS.md and PREREGISTRATION.md at all times.
- After each phase: run all tests, commit with descriptive messages, `git tag phase-N`, and append to PROGRESS.md: what was done, key numbers, decisions made, open issues.
- If your context gets long, re-read AGENTS.md, ROADMAP.md and PROGRESS.md before continuing. PROGRESS.md is your memory.
- Any change to the preregistered plan goes into DEVIATIONS.md with date and reason. Never silently change the primary label, metrics or splits.
- Hyperparameter tuning uses the validation split only. Each final model config is evaluated on the test split once, and every test evaluation is logged in reports/test_log.csv. Never tune against test results.
- Long jobs run in the background (setsid nohup) with logs in reports/; check them periodically.

## Stop conditions (stop and tell me immediately)
- Tests still failing after 3 fix attempts.
- Free disk < 5 GiB.
- Anything needing sudo/pkexec beyond aria2 (ask first; never ask for my password in chat).
- Any AUROC > 0.95 on Task A or B: treat as probable leakage; investigate and report before continuing.
- A decision not covered by ROADMAP.md, AGENTS.md or PREREGISTRATION.md.

## Research scope decision
- target_subreddits = [incels, Braincels, IncelsWithoutHate]. MGTOW is a separate comparison community (downloaded, used only in a later contrast analysis, not for entry/labels). Record this in PREREGISTRATION.md as an amendment dated today, before any modelling.

## Phase 1: Download speed-up + commit
- Commit any uncommitted work.
- Add a global rate limiter shared across workers (max 1 request/second total) and --workers (default 3). Priority: incels, Braincels, IncelsWithoutHate, MGTOW, IncelTears, ForeverAlone. Test the limiter with a mocked clock.
- Stop the running download cleanly, confirm checkpoints, restart with --workers 3 in the background resuming from checkpoints. Keep the disk guard running.
- Log the PID, combined records/s and new ETA in PROGRESS.md.

## Phase 2: Smoke test (as soon as r/incels comments+submissions are complete)
- Ingest only r/incels into /tmp/radpath-smoke, build cohort, run EDA. Fix any schema/parse issues found. Do not interpret results.
- If incels isn't complete yet, do Phase 3 first and come back.

## Phase 3: Features + baselines on SYNTHETIC data (while the download runs)
Features computed strictly within each user's feature window (enforced by tests):
- Activity: contribution count, comment/submission ratio, time to Nth contribution, active days, inter-event time stats, hour-of-day entropy (UTC), weekend share, number of distinct threads and distinct target subreddits.
- Interaction: share of comments replying to comments vs posts, mean/median score, share of has_text=false records (moderation signal).
- Text (has_text=true only): length stats, TF-IDF (fit on train only), VADER sentiment.
- Task B: per-gateway-subreddit counts, recency of last gateway activity, gateway text TF-IDF.
Models: majority class; activity-count-only logistic regression; behavioral LightGBM; TF-IDF logistic regression; combined. Class imbalance via class weights, never resampling before the split.
Evaluation: AUROC, AUPRC, F1 (threshold chosen on validation), calibration (ECE + reliability plot), and 95% CIs from a user-level bootstrap (1,000 resamples). Results to reports/results/ as CSV + figures, aggregate only.

## Phase 4: Full real-data run (only after ALL target+control downloads are complete)
- Full ingest → coverage → cohort → EDA.
- Automatic gate: if Task A has ≥ 300 positive users in train AND ≥ 100 in test under the primary label, continue to Phase 5. Otherwise STOP (Gate 1) and report the class balance per label variant.

## Phase 5: Real-data baselines
- Run the Phase 3 models on real data. Report all results with CIs.

## Phase 6: Sentence embeddings (CPU first)
- Encode feature-window texts with sentence-transformers all-MiniLM-L6-v2 on CPU, cached to disk in batches, resumable. Mean-pool per user, then logistic regression and LightGBM. If the estimated CPU time is > 24h, STOP and tell me (I can run it on Kaggle/Colab).

## Phase 7: Leakage / vocabulary ablation
- Find community-specific vocabulary with weighted log-odds ratios with an informative Dirichlet prior (Monroe et al. 2008), target vs IncelTears, computed on the TRAIN split only. Do not hand-write any slur list.
- Retrain the text models with the top-k terms removed (k = 50, 200, 1000). Report the performance drop.

## Phase 8: Robustness + sensitivity
- All label variants, comments-only variant, excluding near_coverage_start users, Task B. One summary table + forest plot.
- MGTOW contrast: the same pipeline with MGTOW as the target, reported separately.

## Phase 9: Interpretation
- Aggregate feature importance (permutation importance; SHAP summary for LightGBM), calibration by entry year, performance by era (pre/post r/incels ban, Braincels quarantine). Aggregate only — no per-user explanations.

## Phase 10: Paper draft — STOP (Gate 2)
- LaTeX draft in paper/ using a standard ACL-style two-column template: abstract, intro, related work (leave citations as TODO placeholders — do NOT invent references), data + ethics, methods, results, limitations, conclusion. All numbers pulled from reports/results by script, not typed by hand.
- STOP and tell me it's ready for review.

## Not in scope
- No tool, UI or API that scores individual users. Any later interactive page shows aggregate findings only.
