# Deviations from the preregistered plan

## D1 (2026-10-24): freeze the dataset at the partial download

**What changed.** The data acquisition was stopped before all configured streams completed. The analysis uses the snapshot below instead of the full planned corpus.

**Reason.** The Arctic Shift API route failed repeatedly (two full outages and recurring HTTP 422 on several streams); the torrent route found no peers. Completing all 22 streams was estimated at one to several more days. The user chose to proceed on the downloaded data (decision recorded 2026-10-24).

**Effect on the design.** The primary label, features, metrics and splits are unchanged. Cohort labels are coverage-aware: a user's label is null (not negative) when the follow-up window extends past the observed end of any target subreddit that existed at entry. Consequences: the Task A sample is dominated by the pre-ban r/incels era (2014 to Nov 2017); Braincels comments are observed only to the date below; Task B (gateway features) and the IncelTears ablation (Phase 7) rely on partial or missing streams and are reported with that caveat or skipped; the MGTOW contrast is truncated. Data gaps from the API 422 handling (seconds-long skipped windows) are logged in `reports/download*.log`.

**Snapshot (aggregate counts only).**

| Stream | Records | Complete | Observed to |
|---|---|---|---|
| braincels_comments | 1,793,013 | no | 2018-09-19 |
| braincels_submissions | 216,791 | yes | 2019-08-31 |
| dating_advice_comments | 377,323 | no | 2015-11-05 |
| dating_advice_submissions | 261,106 | no | 2019-07-31 |
| foreveralone_comments | 33,134 | no | 2011-08-24 |
| foreveralone_submissions | 0 | no | - |
| incels_comments | 1,131,720 | yes | 2017-11-03 |
| incels_submissions | 54,215 | yes | 2017-11-03 |
| incelswithouthate_comments | 866,775 | no | 2020-09-17 |
| incelswithouthate_submissions | 25,929 | yes | 2020-04-30 |
| inceltears_comments | 0 | no | - |
| inceltears_submissions | 38,940 | no | 2019-04-15 |
| lonely_comments | 75,843 | no | 2017-12-26 |
| lonely_submissions | 18,680 | no | 2018-07-05 |
| mgtow_comments | 1,266,669 | no | 2017-12-12 |
| mgtow_submissions | 6,178 | no | 2015-11-29 |
| short_comments | 0 | no | - |
| short_submissions | 6,202 | no | 2014-08-04 |
| socialanxiety_comments | 370,100 | no | 2019-02-19 |
| socialanxiety_submissions | 0 | no | - |
| tinder_comments | 877 | no | 2013-12-20 |
| tinder_submissions | 28,382 | no | 2015-02-12 |

**Coverage overrides (part of D1).** Because comment streams are incomplete while submission streams are complete, `configs/default.yaml` pins `coverage_overrides` end times for Braincels (2018-09-19) and IncelsWithoutHate (2020-09-17) to the last observed comment. Follow-up windows ending after those dates get null labels.

## D2 (2026-10-24): MGTOW contrast split among label-eligible users

**What changed.** In the Phase 8 MGTOW contrast only, the temporal split (same fractions and embargo rule) is computed over users whose primary Task A label is observable (`eligible_a`), instead of over all entrants.

**Reason.** The MGTOW comment stream ends 2017-12-12 (D1) while most entrants arrive later; the standard split left validation and test with no labelled users (first attempt failed before any model was scored or logged).

**Effect.** Applies only to the separately reported MGTOW contrast; the primary analysis and all other robustness analyses are unchanged. MGTOW is comment-only and covers 2012 to Dec 2017.

**D2 addendum.** Even with the eligible-only split, the 180-day embargo from the robustness grid emptied validation, so `configs/mgtow.yaml` restricts the grid to 90-day follow-up (embargo 90 days). The primary label (90-day) is unchanged.

**D2 addendum 2.** Entrants within the observable MGTOW period are dense late in 2017, so a 15% validation slice (83 days) is narrower than the 90-day label window plus embargo and was fully embargoed. `configs/mgtow.yaml` therefore uses split fractions 0.5/0.3/0.2 for this contrast only.
