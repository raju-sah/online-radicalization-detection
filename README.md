# radpath

Early posting behaviour and later sustained engagement in selected incel-related
Reddit communities. This is an exploratory observational research project by
Raju Sah, Independent researcher ([ORCID](https://orcid.org/0009-0006-0412-6139)).
Community posting is an activity measure. It does not establish ideology,
radicalization, intent, or dangerousness.

The repository contains ingestion, cohort construction, temporal splits,
aggregate EDA, models, synthetic tests, and a preprint source in `paper/main.tex`.
The current release analysis uses posting metadata only. The manuscript and
release are being prepared for author review before Zenodo publication.

## Research design

Targets are incels, Braincels, and IncelsWithoutHate. MGTOW is a separate
comparison community. Entry is the first **observed** cleaned target contribution,
which may differ from true entry because the archive is incomplete.

Task A predictors use target records from entry through the earlier of the fifth
contribution and 14 days. The primary label requires 20 target contributions in
at least four distinct seven-day bins, strictly after the cutoff and within the
next 90 days. Coverage-truncated windows receive null labels. Labels and predictors
are deduplicated by record kind and ID. Task B was planned for gateway activity in
the 180 days before entry, but available coverage does not support a conclusion.

The final cohort restricts entry to 2018-09-19 or earlier. The split uses primary
Task A label-observable users, UTC entry-date groups, nominal fractions
0.50/0.25/0.25, and a 90-day label embargo. All rows for a user share one split.
Hyperparameters and classification thresholds use validation only. A versioned
log records each test evaluation. A new release tag is not an independent test
set: earlier test inspection informed the corrected design.

The corrected predictors exclude scores, removal status and all text-derived
information. `configs/release.yaml` freezes the models, sensitivity scenarios,
seed, tuning grids and bootstrap settings. `DEVIATIONS.md` discloses differences
from the original local plan. Historical text/embedding results are not used as
evidence about whether content adds signal in the corrected analysis.

## Set up Python

Use Python 3.11 in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-release-lock.txt
pytest
```

`requirements.txt` gives compatible dependency ranges. The release lock records
tested direct versions; `release/environment.json` records the installed package
versions and platform. A C build environment may be needed for LightGBM when a
compatible wheel is unavailable. No GPU or deep learning package is needed for
the corrected analysis. `requirements-legacy.txt` describes optional historical
script dependencies; installing them does not restore discarded text.

## Acquire and ingest data

Acquire data only with `scripts/download_data.py`, restricted to subreddits in
`configs/default.yaml`. Review the dry run before a download:

```bash
python scripts/download_data.py --dry-run
python scripts/download_data.py --estimate
python scripts/download_data.py --yes --workers 3
```

The configured torrent route can fall back to Arctic Shift subreddit endpoints.
All API workers share a rate limit of at most one request per second and use
backoff. Repeated blocks stop acquisition. No per-author endpoints are permitted.
Data stay in `data/raw/`, which is excluded from Git and release packages.

Copy `.env.example` to `.env` and set `RADPATH_SALT` locally to a long random
secret. Keep it stable across the dataset lifetime and retain a secure offline
copy. Never put the salt in chat or a release. Then run:

```bash
python scripts/ingest.py --submissions "data/raw/*_submissions.zst" --comments "data/raw/*_comments.zst"
python scripts/build_cohort.py --config configs/default.yaml
python scripts/run_eda.py --config configs/default.yaml
```

Use only the intended frozen real streams. Keep synthetic fixtures in a separate
scratch directory and config; never mix them into real input directories.
Ingestion streams compressed files, normalizes inconsistent field types, hashes
authors, drops configured bot/deleted authors, and discards free text before
writing interim Parquet. Text presence remains an acquisition diagnostic, not a
corrected predictor. For legacy interim files, minimize text before analysis:

```bash
python scripts/minimize_text.py --config configs/release.yaml
```

This command rewrites text fields in place, atomically per file. It preserves
non-text metadata and row order. Raw archives remain in `data/raw/`. The legacy vocabulary word list is moved
to excluded raw storage because its tokens may contain identifiers. Existing
text-based scripts are historical and cannot recover text from minimized files.

## Reproduce the corrected local analysis

With the same locally held snapshot, cohort and configuration:

```bash
python scripts/run_release_analysis.py --config configs/release.yaml
python scripts/make_paper.py --config configs/release.yaml
```

The first command produces aggregate metrics, calibration bins, class balance,
validation permutation importance, a provenance record, and a test log. It never
writes individual features or predictions. It reads a fixed projection that
excludes free text, scores and removal status. Existing evaluations are reused
when provenance matches; changed inputs require a declared new analysis rather
than silently overwriting results or tuning against test outcomes.

The second command regenerates manuscript tables, numeric macros and group-level
figures from local aggregate outputs. Open the existing `paper/main.tex` in the
LaTeX editor and compile it there. Generated inputs are ignored by Git and travel
only in a reviewed manuscript-source archive. Code alone cannot recreate the
unavailable portions of the partial snapshot or independently reproduce the
reported numbers without the same source data.

Synthetic unit tests require neither Reddit data nor a network. A synthetic
fixture generator is included. Its deterministic posting patterns are useful for
pipeline checks, not evidence about real engagement or model generalization.

## Data ethics

These requirements match `AGENTS.md`:

1. Hash authors at ingestion with salted SHA-256 using local `RADPATH_SALT`.
   Discard all free text before persistence because it may contain usernames or
   other identifying details. Never write raw usernames under interim, processed,
   reports, logs, or notebooks. Treat hashes as sensitive pseudonyms.
2. Never commit or release raw dumps, user-level tables, hashes, text, embeddings,
   predictions, secrets, or data-derived logs. Real-data tables and figures are
   excluded from Git. Publication is limited to reviewed aggregate material.
3. Never query named users, author-history endpoints, or individual profiles.
   No individual prediction export is permitted. Hashes are internal join and
   evaluation keys only.
4. Reports and figures contain aggregates. Suppress groups smaller than ten
   users in publication. Plot feature-level averages rather than identifiable
   individual explanations. Include no quotations or linked Reddit records.
5. This is retrospective aggregate research. Do not build or deploy a tool to
   flag, score, moderate, or act on individuals.

No formal ethics review was sought. No institutional approval or exemption is
claimed. The author reports no external funding or competing interests. OpenAI
Codex and Anthropic Claude assisted with development, analysis support,
manuscript preparation and review. The author remains responsible for the work.

## Limitations and release

The snapshot is partial, entry is left-censored, missingness may vary across
communities and periods, and coverage endpoints do not prove complete observation.
The final test population consists of 2018 entrants. Task B, the final vocabulary
ablation and nine 180-day variants are unsupported. Class-weighted probabilities
need separate calibration assessment. Earlier test inspection informed the final
cohort and feature choices, so the results are exploratory.

See `release/RELEASE.md` for the exact packaging commands and review gates. Code
uses MIT (`LICENSE`); the manuscript uses CC BY 4.0 (`paper/LICENSE.md`). Third-party
sources retain their own rights. A DOI is added only after Zenodo supplies it.
