# RPP-FL — Federated IIoT Intrusion Detection (Code Only)

**Repository scope: source code and execution instructions only.** This repository intentionally includes **no datasets, derived/processed data, experimental results, per-seed predictions, split indices, feature masks, checkpoints, or experiment records**.

## Contents

- `RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py` — original source of the 185-run All-Reviewer revision suite. **Unmodified algorithm.**
- `RPP_FL_V7_ALIGNED_COUPLING.py` — source-aligned mechanism-level experiment (five variants, two datasets, three seeds, three conditions = 90 planned scenarios). Calls the unmodified V6 RPP aggregator for the `full` variant.
- `RPP_FL_V7_ALIGNED_COUPLING_Colab.ipynb` — code-only, self-contained Colab notebook embedding the required source code. No saved cell outputs or attachments.
- `requirements.txt` — import dependencies (not an exact historical environment lockfile).
- `.gitignore` — excludes common datasets, generated outputs, and local secrets.
- `SOURCE_CODE_SHA256.txt` — checksums for the three source artifacts.

## How to use

For the original revision suite, open Google Colab with GPU support, mount Google Drive, upload `RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py`, and execute it. It stores generated data and experiment outputs in **your own** Google Drive (`RPP_FL_DATASETS`, `RPP_FL_CACHE_V6_REVIEW`, `RPP_FL_RESULTS_V6_ALL_REVIEWERS`), not this repository. The source may download dataset representations from public third-party mirrors according to its configuration. Review the dataset licenses and exact versions before use.

For the new aligned comparison, open `RPP_FL_V7_ALIGNED_COUPLING_Colab.ipynb` in Colab and run its cells. Alternatively, place both Python scripts in the same directory and run:

```bash
python -u RPP_FL_V7_ALIGNED_COUPLING.py
```

Outputs are saved only to `MyDrive/RPP_FL_RESULTS_V7_ALIGNED_COUPLING` (and prepared data under the user-owned cache). The aligned experiment was **not yet trained** on the real datasets at repository preparation; do not use earlier unaligned 90-run results as results for this code.

## Scientific qualifications

- The original 185-run outputs were audited for saved-result consistency, but **this source copy cannot be cryptographically proven to be the exact Colab-executed bytes** because the historical run configuration did not preserve its source SHA-256.
- The original V6 source and the earlier unaligned V6 Reviewer-3 study were numerically different; the aligned V7 experiment was created to compare mechanisms against the original V6 RPP implementation. No new 90-scenario performance metrics are asserted here.
- `P4P` and `TrustFed-IDS-style` are **independent reimplementations**, not original-author official implementations.
- The source's federated training simulator does not implement a cryptographic secure-aggregation backend. The differential-privacy branch is separate from individual-update robust screening.
- No prepared arrays, raw files, split indices, predictions, result summaries or data-specific metadata are hosted here at the user's request. **Code-only release does not by itself fully satisfy a reviewer request for independently auditable numerical records.**
- Exact raw-data snapshot identities and historic runtime versions remain to be confirmed. Please do not describe this code-only repository as a complete results reproducibility archive.

## Citation / manuscript use

The repository URL and tag can be added to a paper's Code Availability statement **once it has actually been published**, with the above scope limitations. Do not announce a DOI or public artifact not yet created.

## License

No source-code license has yet been selected by the authors. Redistribution and reuse permissions must be chosen explicitly before adding a license file.
