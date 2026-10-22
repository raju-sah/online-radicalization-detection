# radpath

**Pathways into online incel communities: predicting deep engagement from
early and pre-entry behaviour on Reddit.**

`radpath` is an observational computational-social-science project. It asks
whether, among users who make a first contribution to incel/manosphere
communities, sustained deep engagement can be predicted from information
available at or before an early cutoff. The output is aggregate research for
a paper, **not** a tool that flags or profiles individuals.

This repository currently implements **phase 1** (configuration, streaming
ingestion, cleaning, cohort and label construction, temporal splits,
aggregate EDA, a synthetic fixture, and tests) and **phase 2** (controlled
data acquisition). Feature engineering beyond cohort counts, models,
embeddings, and any UI are deliberately deferred.

## Research design

### Cohort

A user enters the cohort at `t0`, the timestamp of their earliest cleaned
contribution in a configured **target** subreddit. Contributions are
deduplicated by `(kind, id)`. Control and gateway communities never define
entry.

### Task A — early signal

Features come from target-community contributions inside an early window:

```
cutoff_A = min(timestamp of the Nth target contribution, t0 + 14 days)
```

with `N = 5` by default. If the Nth contribution arrives after 14 days, the
cutoff is `t0 + 14 days`. Features may use records with
`created_utc <= cutoff_A` only.

### Task B — pre-entry

Features come from configured **gateway** communities in the 180 days before
entry:

```
cutoff_B = t0 - 1 second
feature records: t0 - 180 days <= created_utc <= cutoff_B
```

Users with no gateway activity receive zero counts and remain in the cohort
(they are not treated as missing).

### Label (both tasks)

```
label window = (cutoff, cutoff + 90 days]
deep = (target contributions in window >= 20)
       AND (distinct active weeks in window >= 4)
```

Labels use **target-community records only**, strictly after the cutoff.
Active weeks are seven-day bins anchored at the first second after the
cutoff, so a 90-day window has at most 13 bins. Users whose label window ends
after the end of observation are **ineligible and receive a null label**, not
a negative one.

The cohort table exposes both the label components (`label_n_a`,
`label_weeks_a`, ...) and the final `label_deep_a` / `label_deep_b` columns.
Filter to `eligible_a`/`eligible_b` before any modelling.

### Temporal split

Users are ordered by entry date (`t0`), with hashed IDs as a deterministic
tie-breaker, and assigned 70% train (earliest), 15% validation, 15% test
(latest). The split is user-level and is never random at the row level.

## Setup

Requires Python 3.11 and a C build environment for LightGBM (installed for a
later phase).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Generate a long random salt and put it in .env:
python -c "import secrets; print(secrets.token_hex(32))"
```

Edit `.env` so it contains:

```
RADPATH_SALT=<the random value>
```

`RADPATH_SALT` must stay stable for the lifetime of the dataset: changing it
changes every user hash, so previously ingested outputs can no longer be
joined to new data. Keep a secure offline copy.

Raw data and `.env` are gitignored. Never commit either.

## Raw data

Acquire data **only** through the configured script:

```bash
# Show routes, sizes and disk headroom without downloading
python scripts/download_data.py --dry-run

# Show aggregate record counts for planning
python scripts/download_data.py --estimate --only incels Braincels
# Download selected targets (disk guard requires 5 GiB free afterwards)
python scripts/download_data.py --yes --only incels Braincels

# One record type only, or force the API route
python scripts/download_data.py --yes --only lonely --type comments
python scripts/download_data.py --yes --force-api --only lonely
```

The script accepts only subreddits listed in `configs/default.yaml`. It
prefers selected files from the configured Academic Torrents bundle via
`aria2c`, then falls back to Arctic Shift subreddit-level endpoints
(`/api/posts/search`, `/api/comments/search`). No per-user or per-author
endpoint is used. Requests are limited to at most one per second with
exponential backoff; repeated blocks stop the run rather than evading limits.

If the configured torrent is unavailable (for example the Academic Torrents
page returns HTTP 404), the script retries with the configured magnet URI and
then falls back to the Arctic Shift API route. Magnet metadata fetching uses
`aria2c --bt-metadata-only` and gives up after 10 minutes. API sizes are
unknown until downloaded, so use `--estimate` and review the disk guard before
passing `--yes` with many targets. Time-series counts can start in 2018, so
they are planning lower bounds for older subreddits; validate with a pilot.

Outputs are written to `data/raw/<subreddit>_<type>.zst`. API downloads are
resumable through `data/raw/_checkpoints/`; a completed manifest with route,
byte count, record count and timestamp bounds is written to
`data/raw/download_manifest.json`. Logs are username-free. For a long run:

```bash
setsid nohup python scripts/download_data.py --yes \
  > reports/download.log 2>&1 &
```

Placing dumps manually is still supported if you obtained them through the
approved sources. A common layout is one file per subreddit and record kind,
named e.g. `incels_submissions.zst` and `incels_comments.zst`. The pipeline
also accepts directories or `**` globs.

Required submission fields:

```
id, author, subreddit, created_utc, title, selftext, score, num_comments
```

Required comment fields:

```
id, author, subreddit, created_utc, body, score, link_id, parent_id
```

Missing optional fields become empty strings or zero. `created_utc` may be a
number or numeric string; unparseable timestamps are dropped and counted.

Ingestion keeps only subreddits in `target_subreddits`,
`control_subreddits`, `gateway_subreddits` (plus `ingest.extra_subreddits`)
by default. Pass `--no-subreddit-filter` to keep everything.

## Run the pipeline

```bash
source .venv/bin/activate
```

1. **Ingest** raw dumps (streaming; safe for large files):

```bash
python scripts/ingest.py \
  --submissions "data/raw/*_submissions.zst" \
  --comments "data/raw/*_comments.zst"
```

Outputs go to `data/interim/submissions/` and `data/interim/comments/`,
with usernames replaced by salted hashes. If files are nested, use
`"data/raw/**/*_submissions.zst"` (quote globs so Python expands them) or
pass directories with `--submissions data/raw/incels_dumps`.

Before ingesting real data, delete fixture outputs so synthetic and real
records do not mix:

```bash
rm -f data/raw/*.zst
rm -rf data/interim/submissions data/interim/comments data/processed/*
```

2. **Build the cohort and temporal split**:

```bash
python scripts/build_cohort.py --config configs/default.yaml
```

Writes `data/processed/cohort.parquet` and an aggregate
`data/processed/cohort_meta.json`.

3. **Run aggregate EDA**:

```bash
python scripts/run_eda.py --config configs/default.yaml
```

Writes plots, class-balance tables, and `summary.md` to `reports/`.

4. **Run tests**:

```bash
pytest
```

### Synthetic smoke test

```bash
python scripts/make_fixture.py --out-dir data/raw
python scripts/ingest.py \
  --submissions "data/raw/*_submissions.zst" \
  --comments "data/raw/*_comments.zst"
python scripts/build_cohort.py
python scripts/run_eda.py
```

The fixture is entirely synthetic: fake users, fake text, dirty records,
mixed field types, bots, and deliberate engagement trajectories. It is for
pipeline testing only and must never be analysed as data.

## Configuration

All research parameters live in `configs/default.yaml`: target, control, and
gateway subreddit lists; `feature_window.first_n`, `feature_window.days`;
`pre_entry_days`; `followup_days`; `deep_min_contributions`;
`deep_min_active_weeks`; split fractions; seed; and paths.

Defaults:

```yaml
target_subreddits: [incels, Braincels, MGTOW, IncelsWithoutHate]
control_subreddits: [IncelTears, ForeverAlone]
gateway_subreddits: [ForeverAlone, lonely, socialanxiety, dating_advice, short, Tinder]
feature_window: {first_n: 5, days: 14}
pre_entry_days: 180
followup_days: 90
deep_min_contributions: 20
deep_min_active_weeks: 4
seed: 42
```

`data_end_utc` defaults to `null`, meaning the latest timestamp in the
ingested data. For real data, set it explicitly to the end of coverage of the
**target** dumps. Otherwise a gateway dump that extends later than the target
dumps can make follow-up windows look complete when target coverage has
already ended.

## Repository layout

```
AGENTS.md                 project ethics/rigour rules for future sessions
README.md
requirements.txt
pytest.ini
.env.example
.gitignore
configs/default.yaml
data/raw/                 raw .zst dumps (gitignored)
data/interim/             hashed cleaned parquet (gitignored)
data/processed/           cohort parquet/metadata (gitignored)
reports/                  aggregate EDA outputs (gitignored except .gitkeep)
src/radpath/
  __init__.py
  config.py               YAML + .env loading, validation, paths, salt
  clean.py                deleted/bot/empty-text rules
  ingest.py               streaming .zst JSONL -> parquet, salted hashing
  download.py             torrent/API acquisition, checkpoints, manifest
  cohort.py               t0, windows, eligibility, labels, split hook
  split.py                user-level temporal train/val/test
  eda.py                  aggregate summaries and plots
scripts/
  make_fixture.py         synthetic .zst generator
  download_data.py        the only data-acquisition entry point
  ingest.py
  build_cohort.py
  run_eda.py
tests/
  test_ingest.py
  test_clean.py
  test_cohort.py
  test_download.py
  test_no_leakage.py
  test_split.py
```

## Data ethics

These rules are non-negotiable and are duplicated in `AGENTS.md`.

1. **Usernames are replaced at ingestion** with a salted SHA-256 hash. The
   salt is read from `.env` (`RADPATH_SALT`) and is gitignored. Raw usernames
   never appear in `data/interim`, `data/processed`, `reports`, logs, or
   notebooks. Hashing is pseudonymisation, not a guarantee of anonymity;
   treat hashed IDs as sensitive.
2. **Data and secrets are never committed.** `data/`, `.env`, and generated
   parquet/figures are gitignored.
3. **No per-user lookups, scraping, or exports.** No code may look up or
   output information about specific named users. Per-user prediction
   exports are forbidden. Hashed IDs may be used only for evaluation.
4. **Reports and figures are aggregate only.** Do not publish tables or
   plots that could re-identify users; consider small-cell suppression for
   publication (e.g. suppress cells with fewer than 10 users).
5. **No flagging or profiling tool.** This is retrospective, aggregate
   research. Do not deploy any output to identify or act on individuals.

## Scientific rigour and known limitations

- **No leakage is enforced by code and tested.** Features use only records
  at or before the task cutoff; labels use only target records strictly after
  it. `tests/test_no_leakage.py` independently reconstructs both sets.
- **Eligibility depends on dump coverage, not account activity.** A user who
  leaves Reddit is observed as a non-deep user if their 90-day window falls
  inside the dumps. "Deep engagement" therefore means observed sustained
  activity, not necessarily intent or persistence beyond the data.
- **Left truncation.** Users already active in target communities before the
  first dump record are treated as new entrants at their first observed
  record. For real data, consider excluding users whose first observed
  target activity falls within a buffer after the start of data coverage.
- **Bot filtering is a heuristic.** Names ending in a configurable suffix
  (`bot`) plus an explicit list are dropped, as are `[deleted]`/`[removed]`
  authors and records with no usable text. Real automated accounts that do
  not match the rule remain.
- **Username case is folded before hashing** because Reddit usernames are
  case-insensitive. If a source dump inconsistently changes casing, this
  keeps one account as one user.
- **Zero gateway activity is informative, not missing.** Task B includes
  users with no gateway records; a separate "gateway users only" sensitivity
  analysis may be useful.
- **Label thresholds and windows are research choices**, not ground truth.
  They are configurable and should be sensitivity-tested before the paper.
- **No models yet.** Future modelling must report AUROC, AUPRC, F1, and
  calibration with class balance, plus majority-class and activity-only
  baselines.
