# Prepare the exploratory release

Version 0.1.0 contains a preprint and supporting research code. It is not a
peer-reviewed publication or an independent confirmatory replication.

## Verify locally

1. Run `python -m pytest` in the project virtual environment.
2. Run `python scripts/run_release_analysis.py --config configs/release.yaml`
   using the same local snapshot. Existing results must be reused with matching
   provenance; do not change a tag to bypass repeated-test safeguards.
3. Run `python scripts/make_paper.py --config configs/release.yaml`.
4. Compile the existing `paper/main.tex` in the built-in LaTeX editor. Inspect
   every page, including table widths, figure labels, references and appendix
   placement. Source checks do not establish successful compilation.
5. Export the editor's final checked PDF to `paper/main.pdf`. The package builder
   requires this PDF for a complete release. An unavailable compiler is a release
   blocker; it does not justify labelling the manuscript as visually verified.
6. Review the ethics, funding, competing-interest and AI disclosures and confirm
   that the manuscript accurately describes the author's work and decisions.
7. Run `python scripts/build_release.py`. Inspect `output/release/manifest.json`
   and the contents of each archive before upload.

The package builder also supports `--source-only` to prepare code and TeX sources
for review when the editor's PDF compiler is unavailable. That output is labelled
incomplete and must not be treated as a finished publication package.

## Package contents

- `radpath-code-0.1.0.zip`: allowlisted source, tests, configurations, documentation,
  citation information and the MIT license. No local data, Git history or secrets.
- `radpath-manuscript-source-0.1.0.zip`: the current TeX source, its required generated
  aggregate tables, two aggregate figures and the CC BY 4.0 license statement.
- `radpath-preprint-0.1.0.pdf`: the exported, compiled and visually checked paper,
  included only for a complete build.
- `zenodo-metadata.json`: draft descriptive metadata. Update the publication date
  to the actual release date before upload and add the DOI only when assigned.
- `manifest.json` and `SHA256SUMS`: package inventory and integrity checks.

Generated real-data paper inputs and archives are ignored by Git. Raw archives,
user-level Parquet, hashes, record IDs, text, embeddings, predictions, logs,
vocabulary lists and `.env` are excluded. Dataset minimization is distinct from
publication: the code package cannot reconstruct discarded text or the missing
parts of the source archive. `environment.json` records versions without capturing
credentials, environment variables or filesystem paths.

## Zenodo metadata

The main object is a publication of subtype preprint, with Raju Sah as sole
creator, affiliation Independent researcher and ORCID 0009-0006-0412-6139. The
manuscript is CC BY 4.0. Supporting code is separately MIT-licensed inside its
archive; the description must preserve that distinction. No Reddit dataset is
published. If code later receives a separate DOI, relate the records using the
actual assigned identifiers. No DOI is invented or prefilled here.

This task prepares a reviewable local package. Publication follows the author's
review of the actual files and saved Zenodo preview.
