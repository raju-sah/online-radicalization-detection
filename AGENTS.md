# radpath — project rules for agents and contributors

`radpath` studies pathways into online incel/manosphere communities on Reddit.
The research question is aggregate-level: among users making a first
contribution to target communities, can early or pre-entry behaviour predict
sustained deep engagement measured strictly after the feature window?

This is **observational research**, not a moderation, flagging, or profiling
tool. Read the ethics section before touching data or code.

## Data ethics (non-negotiable)

1. **Hash usernames at ingestion.** Every author is replaced with a salted
   SHA-256 hash. The salt lives in `.env` as `RADPATH_SALT`; `.env` is
   gitignored. Raw usernames must never appear in any file under
   `data/interim`, `data/processed`, `reports`, `logs`, or notebooks.
2. **Never commit data.** `data/` and `.env` are gitignored. Do not commit
   raw dumps, parquet outputs, figures derived from real data, or logs.
3. **No per-user lookups or exports.** Do not write code that looks up,
   scrapes, or outputs information about specific named users. No per-user
   prediction exports. Hashed IDs may appear only where evaluation requires
   them.
4. **Aggregates only in reports and figures.** Any published table or plot
   must be at aggregate level and must not permit re-identification.
5. **README stays accurate.** Keep the "Data ethics" section in `README.md`
   in sync with these rules.

When adding a feature, ask: could this increase re-identification risk?
Text features are especially risky; prefer aggregate behaviour counts.

## Scientific rigour (non-negotiable)

1. **No leakage.** A user's features may only use records with
   `created_utc <= feature_cutoff` for Task A, or `created_utc < t0` for
   Task B. Labels use records strictly after the cutoff and within the
   follow-up window. `tests/test_no_leakage.py` must keep passing.
2. **User-level and temporal splits.** Train on earlier entrants, validation
   in the middle, test on later entrants. Never split rows randomly. A user
   appears in exactly one split.
3. **Report AUROC, AUPRC, F1, and calibration.** Never report accuracy alone.
   Always report class balance. (Model code is a later phase; these
   requirements apply as soon as models exist.)
4. **Trivial baselines are mandatory.** At minimum: majority class and an
   activity-count-only model.
5. **Reproducibility.** Fixed seeds; all parameters live in `configs/`, not
   hard-coded. Paths resolve relative to the repository root.

## Data acquisition (non-negotiable)

1. **Download only through `scripts/download_data.py`.** Do not use `curl`,
   `wget`, browser downloads, or ad-hoc Python snippets to acquire raw data.
   Fetching the configured torrent metadata is also done by that script.
2. **Only configured subreddits.** The script may download only subreddits
   listed in `configs/default.yaml` under `target_subreddits`,
   `control_subreddits`, or `gateway_subreddits`. Never add ad-hoc
   subreddits on the command line.
3. **Never use per-user or per-author endpoints.** Only subreddit-level
   endpoints (Arctic Shift `/api/posts/search`, `/api/comments/search`, and
   aggregate `/api/time_series` for planning) are permitted. Do not query
   user profiles, author histories, or any endpoint keyed by username.
4. **Respect rate limits.** Arctic Shift requests are limited to at most one
   request per second with a descriptive User-Agent and exponential backoff.
   Stop and report if repeatedly blocked; do not evade limits.
5. **No credentials or passwords in chat or files.** For system packages,
   `aria2` may be installed via `pkexec apt-get install -y aria2`, which
   opens a graphical password dialog. Never use `sudo -S`, echo a password,
   pipe a password, store a password, or ask for one in chat. If the popup is
   cancelled, fall back to the Arctic Shift API route.
6. **Downloads stay in `data/raw/`** (gitignored). Never commit raw dumps,
   torrent payloads, checkpoints, or manifests containing data-derived
   content. Logs must remain aggregate-only and username-free.

## Workflow expectations

- Python 3.11, virtualenv, `requirements.txt`; no GPU or deep learning.
- Stream `.zst` files line by line. Never load a whole raw file into memory.
- Raw field types are inconsistent across years; normalise types at ingest.
- Keep the package in `src/radpath` with type hints and docstrings; CLI
  entry points in `scripts/` using `argparse`.
- Run `pytest` before committing. Tests must not require real data or network.
- Commit in small, descriptive steps. Never commit `data/`, `.env`, or raw
  usernames.

## Current scope

Phase 1 covers config, ingestion, cleaning, cohort/label construction,
temporal splits, EDA, synthetic fixtures, and tests. Feature engineering
beyond cohort counts, models, embeddings, and UI are explicitly deferred.
