# radpath progress

Last updated: 2026-10-22T15:55Z

This file is the running memory for the autonomous roadmap. Re-read it after a
long context break. The roadmap is in `ROADMAP.md`; the frozen analysis choices
are in `PREREGISTRATION.md`.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 1. Download speed-up + commit | done | Shared 1 req/s limiter, `--workers 3`, priority ordering, duplicate-page advance; download running. |
| 2. r/incels smoke test | pending | Not run: `incels_comments` and `incels_submissions` are not complete. `/tmp/radpath-smoke` is empty. Phase 3 proceeds first, as allowed by the roadmap. |
| 3. Synthetic features + baselines | in progress | Started after Phase 1 record was written. |
| 4. Full real-data run | blocked | Waiting for all target and control downloads. |
| 5. Real-data baselines | pending | Depends on Phase 4 and Gate 1. |
| 6. Sentence embeddings | pending | Depends on Phase 5. |
| 7. Vocabulary ablation | pending | Depends on Phase 5. |
| 8. Robustness + sensitivity | pending | Depends on Phase 5. |
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
