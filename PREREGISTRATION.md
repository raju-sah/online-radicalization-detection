# radpath preregistration

**Date:** 2026-10-22  
**Status:** written before any model training, feature engineering beyond the
cohort counts described here, or test-set evaluation.

This document fixes the analysis choices for the phase-1 cohort. Changes must
be dated and justified in a revision below; the code and
`configs/default.yaml` are the executable source of truth.

## Research question

Among users making a first observed contribution to configured incel/manosphere
target communities, can early within-community behaviour (Task A) or pre-entry
gateway-community behaviour (Task B) predict sustained deep engagement measured
strictly after the feature window?

This is aggregate observational research. It is not a moderation, flagging, or
profiling system, and no per-user predictions will be released.

## Cohort and entry

- Entry time `t0` is the first cleaned contribution in a target subreddit:
  `incels`, `Braincels`, `MGTOW`, `IncelsWithoutHate`.
- Target records are deduplicated by `(kind, id)`.
- Control subreddit for comparison: `IncelTears` only.
- Gateway subreddits: `ForeverAlone`, `lonely`, `socialanxiety`,
  `dating_advice`, `short`, `Tinder`.
- Known authors with `[removed]`, `[deleted]`, or empty content are retained as
  activity records with `has_text = false`; they are excluded from future text
  features only. Authors named `[deleted]`/`[removed]` are dropped because
  they cannot be attributed to one account.
- Bots are dropped by the configured suffix rule plus explicit names. Distinct
  authors dropped by each rule are logged; the run warns if the suffix rule
  drops more than 1% of distinct authors.

## Tasks

### Task A: early signal

Features use target-community records with

```
created_utc <= cutoff_A
cutoff_A = min(timestamp of the 5th target contribution, t0 + 14 days)
```

### Task B: pre-entry

Features use gateway-community records with

```
t0 - 180 days <= created_utc < t0
```

The Task B cohort includes only users with at least one gateway contribution
in that pre-entry window. Users with zero gateway activity are reported as
excluded; they are not treated as missing.

## Label

The primary label is **deep engagement**:

```
label window = (task cutoff, task cutoff + 90 days]
deep = (target contributions in window >= 20)
       AND (distinct active weeks in window >= 4)
```

Submissions and comments have equal weight. Active weeks are 7-day bins
anchored at the first second after the cutoff. Labels use target-community
records only, strictly after the cutoff.

A label is null, rather than negative, when the window is not fully covered by
every target subreddit that existed at entry. Ineligible users are reported by
reason (for example `followup_truncated_by_coverage_end`).

## Robustness grid

All variants are generated from `configs/default.yaml`; label columns are
named by variant.

- Contributions: `{10, 20, 50}`
- Active weeks: `{3, 4, 8}`
- Follow-up: `{90, 180}` days
- Plus a comments-only variant of the primary label.

Variant names follow `c<contributions>_w<weeks>_f<followup>`, with
`_comments` for the comments-only variant. The primary variant is
`c20_w4_f90`.

## Temporal split and embargo

Users are split by UTC calendar date of `t0` into 70% train (earliest), 15%
validation, 15% test (latest). All users sharing a `t0` date are in the same
split. Splitting is never row-random.

Embargo: a training user is removed from the training split when their label
window reaches the validation start date; a validation user is removed when
their label window reaches the test start date. The embargo uses the longest
configured follow-up (180 days) so no robustness variant leaks across a split.
Removed users keep their cohort row but have `split = null`, and counts are
logged and reported.

## Evaluation

Primary metrics, reported with class balance for every split and variant:

- AUROC
- AUPRC
- F1
- Calibration (reliability plot and a proper scoring rule such as Brier
  score)

Mandatory baselines:

1. Majority class.
2. Activity-count-only model (cohort counts available at the task cutoff).

No accuracy-only reporting. The test split is used once, after model selection
on validation data.

## Coverage and sensitivity

Per-subreddit coverage (first/last observed record) is written to
`data/interim/coverage.parquet` during ingestion. Manual
`coverage_overrides` may encode verified ban/quarantine dates.

`near_coverage_start` flags users entering within 30 days of the earliest
target coverage start. The pre-specified sensitivity analysis excludes these
users; it is reported alongside the primary analysis.

## Analysis outputs and ethics

Reports and figures are aggregate only. Usernames are salted-hashed at
ingestion, raw data and secrets are never committed, and no per-user lookup or
prediction export is permitted. Text features are not part of this
preregistration; if added later they require a dated amendment and an explicit
re-identification-risk review.

## Amendments

None before model training.
