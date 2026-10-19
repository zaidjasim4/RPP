# RPP-FL: Robust and Privacy-Aware Federated Learning for IIoT Anomaly Detection

**Research code, historical experiment provenance, and reproducibility notes**

> **Documentation clarification (19 October 2026).** An earlier version of this README was insufficiently clear about the difference between an **already completed V6 Reviewer-3 mechanism study** and a **later V7 source-aligned runner**. The historical study was executed and produced the three-seed results reported in Section 5.6.1 and Table 8 of the manuscript. The separate V7 runner was subsequently published as code; its execution should not be inferred from those historical results. This correction concerns documentation and attribution, not a claim that the V7 runner itself generated Table 8.

## 1. Project overview

RPP-FL studies federated anomaly detection for Industrial Internet of Things (IIoT) traffic under heterogeneous (non-IID) data and poisoning attacks. The research explores geometric update trust, temporal reputation, collusion-related signals, robust aggregation, and an operationally separate differential-privacy aggregate-release experiment.

The manuscript includes evaluations on CICIoT2023, EdgeIIoTset, TON_IoT, and WUSTL-IIoT. The additional **mechanism-level study documented below is limited to CICIoT2023 and TON_IoT**; it should not be read as a cross-dataset-transfer experiment.

**Repository scope:** GitHub contains source code and execution instructions. The retained historical experiment results are stored **externally in Google Drive**, not checked into this repository. Source availability and numerical auditability are distinct matters.

## 2. Historical mechanism-level experiment: verified provenance

The mechanism-level figures in **Section 5.6.1 / Table 8** came from the **V6 Reviewer-3 event-coupling experiment**. A review of the retained Google Colab notebook and Google Drive artifacts established the following:

| Provenance item | Evidence from the retained records |
|---|---|
| Execution status | `_STATE/execution_state.json` reports `COMPLETED`. |
| Recorded start/resumption | `2026-10-16 07:12:15` (recorded without an explicit time-zone offset). |
| Recorded completion | `2026-10-16 14:37:53` (recorded without an explicit time-zone offset). |
| Independent file metadata | Google Drive metadata places the result files' creation/modification on **16 October 2026**; the final consolidated CSV was created at approximately **14:37:52 UTC**. |
| Number of completed scenarios | **90 of 90**, with 90 unique dataset/scenario keys in the per-seed results. |
| Experimental design | **2 datasets × 5 aggregation configurations × 3 protocol seeds × 3 conditions**. |
| Datasets | `CICIoT2023`, `TON_IoT`. |
| Variants | `simple_temporal`, `uncoupled`, `partial`, `full`, `p4p`. |
| Protocol seeds | `42`, `2026`, `31415`. |
| Conditions | Clean at Dirichlet alpha = 0.5; ALIE poisoning with 20% malicious clients at alpha = 0.5; and ALIE-20% at alpha = 0.1. |
| Training budget | **25 communication rounds** per recorded scenario. |
| Executed aggregation label | `rpp_event` for the event-coupling variants; the independently reimplemented P4P comparator is labeled `p4p`. |

**Archived results folder:** [RPP_FL_RESULTS_V6_REVIEWER3 on Google Drive](https://drive.google.com/drive/folders/1nzivI5IvjbgUTf5LhzTJgNxFkvj7XYur). Access to the linked folder and individual files is subject to Google Drive sharing permissions.

### 2.1 Files supporting the audit

The linked archive includes:

```text
RPP_FL_RESULTS_V6_REVIEWER3/
├── run_config.json
├── all_results_partial.csv
├── ALL_DATASETS_FINAL_RESULTS.csv
├── _STATE/
│   └── execution_state.json
├── R3_TABLES/
│   ├── MECHANISM_PER_SEED.csv
│   ├── MECHANISM_MEAN_SD_CI95.csv
│   ├── FULL_VS_ALTERNATIVES_PAIRED.csv
│   ├── PAIRED_DIFFERENCE_SUMMARY.csv
│   └── ROUND_LEVEL_SECURITY.csv
├── CICIoT2023/
│   └── r3_<variant>_<condition>_seed<seed>/
└── TON_IoT/
    └── r3_<variant>_<condition>_seed<seed>/
```

Individual scenario directories contain `final_result.json`, `COMPLETED.json`, `round_history.csv`, and `best_checkpoint.pt` / `last_checkpoint.pt`. These provide scenario-level results, completion markers, round-wise diagnostics, and model states.

**Interpretation of the records:** `execution_state.json` and Google Drive timestamps support the documented execution history. The `RunFingerprint` fields identify saved experiment configurations but are **not cryptographic hashes of the executed source code**.

### 2.2 Numerical verification of manuscript Table 8

The archived three-seed per-scenario records were regrouped for **TON_IoT, ALIE-20%, Dirichlet alpha = 0.5**. Recalculated arithmetic means agree with Table 8 to the displayed precision:

| Aggregation configuration | Mean F1 (%) | Mean client attack-detection AUC | Mean malicious aggregation weight (%) |
|---|---:|---:|---:|
| Simple robust + temporal trust | 99.195 | 0.258 | 25.161 |
| Uncoupled RPP | 99.192 | 1.000 | 2.583 |
| Partial event coupling | 99.184 | 1.000 | ~0 |
| Full event coupling | 99.180 | 1.000 | ~0 |
| P4P (independent reimplementation) | 99.301 | 0.549 | 13.744 |

For clarity, the full-coupling mean malicious-weight share is approximately **0.000046%**, not mathematically zero. `~0` represents rounding of the measured value. The study does **not** establish that full coupling always outperforms partial coupling or P4P in predictive F1; its observed influence-suppression effect is protocol- and dataset-dependent.

## 3. Historical V6 Reviewer-3 experiment versus the later V7 runner

Both efforts address the **same broad 90-scenario mechanism-comparison design**, but this common design does **not** imply identical source-code execution:

| Aspect | Completed historical V6 Reviewer-3 study | Later V7 aligned source runner |
|---|---|---|
| Purpose | Mechanism-level event-coupling comparison used for Table 8 | Source-aligned comparison built around the original published V6 aggregator |
| Recorded aggregation | Dedicated `rpp_event` path for mechanism variants | `AlignedServer`; the `full` variant delegates to the original V6 `_aggregate_rpp` |
| Saved study results | 90 completed scenarios in the linked Google Drive archive | **No separately verified V7 results asserted by this README** |
| Output directory | `MyDrive/RPP_FL_RESULTS_V6_REVIEWER3` | `MyDrive/RPP_FL_RESULTS_V7_ALIGNED_COUPLING` when executed |
| Source identity | Historical execution-time SHA-256 **not recorded** | Published GitHub revision can be identified through its commit history |

The implementations are related, but **not byte-identical**. Among their code-path differences are median calculation conventions and handling of a zero/initial momentum reference. Consequently, **Table 8 is attributed to the actual completed V6 Reviewer-3 execution, not to an unverified execution of V7**. The original V6 All-Reviewer implementation and the separately executed V6 Reviewer-3 mechanism study should also not be conflated.

### 3.1 GitHub publication timeline

The aligned V7 source was introduced in [this GitHub commit](https://github.com/zaidjasim4/RPP/commit/84e70966b35b4d65517bdc4361700a8eee8c2577), dated **16 October 2026 at 19:19 UTC**. Google Drive metadata for the completed historical results predates that publication on the same date.

The earlier wording **'the aligned experiment was not yet trained'** referred to the later V7 source-aligned runner at the time of repository preparation. It should **not** have been read as a statement that the historical Reviewer-3 experiments underlying Table 8 had never been executed. This distinction was inadequately explained in the original README and is corrected here.

### 3.2 Source-provenance limitations

- The historical configuration file records `code_sha256: null`; no execution-time source SHA-256 was saved. A retrospective byte-for-byte match between the exact running notebook and the public code **cannot be cryptographically established**.
- The preserved execution notebook and outputs support identifying the study as **V6 Reviewer-3**, and saved metrics reproduce the published numerical summary. They do **not** prove numerical equivalence to V7 on real data.
- The historical Python, PyTorch, NumPy, and pandas versions are captured in `run_config.json`, but the repository's `requirements.txt` is not an exact historical environment lockfile.
- Original raw-data snapshot identities, complete published splits, and every historical dependency build are not fully established by this code repository alone.
- Numerical consistency checks document result provenance; they are not a substitute for an independent full replication.

## 4. Repository contents

| File | Role |
|---|---|
| `RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py` | Original V6 All-Reviewer revision-suite implementation (not the executed Reviewer-3 source). |
| `RPP_FL_V7_ALIGNED_COUPLING.py` | Later aligned five-variant / two-dataset / three-seed / three-condition runner. |
| `RPP_FL_V7_ALIGNED_COUPLING_Colab.ipynb` | Self-contained, code-only Google Colab launcher for the aligned runner; saved training outputs are not included. |
| `RPP_FL.py` | Earlier repository source file; do not treat its filename alone as provenance for Table 8. |
| `requirements.txt` | Import/dependency requirements, not a historical environment lockfile. |
| `.gitignore` | Excludes generated artifacts, datasets, and other local files. |
| `README.md` | This project and provenance documentation. |

The historical **Reviewer-3 result archive is external**, and the executed Reviewer-3 notebook is not provided as a separately versioned, execution-hash-verified source artifact in this GitHub repository.

## 5. Running the published code

### 5.1 Original V6 All-Reviewer code

Run `RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py` in a compatible Python/Google Colab environment with required dependencies and access to the configured datasets. This is the original All-Reviewer suite, **not** a command to recreate the historical Reviewer-3 outputs above.

### 5.2 Later aligned V7 experiment (new execution)

Use `RPP_FL_V7_ALIGNED_COUPLING_Colab.ipynb` in Google Colab, or place the V7 and original V6 Python files in the same directory and execute:

```bash
python -u RPP_FL_V7_ALIGNED_COUPLING.py
```

The runner validates the expected SHA-256 of its V6 dependency before proceeding, executes its configured scenarios, and writes **new results** under `MyDrive/RPP_FL_RESULTS_V7_ALIGNED_COUPLING`. Its outputs must be identified as V7 results rather than silently substituted for the historical Reviewer-3 records. Relevant generated files include `ALIGNED_PROTOCOL_MANIFEST.json`, `MECHANISM_PER_SEED.csv`, `MECHANISM_MEAN_SD_CI95.csv`, and paired-comparison summaries.

**Important:** Real dataset access, preprocessing, resource requirements, and environment compatibility remain the responsibility of the reproducing researcher. Dataset download mechanisms may depend on external mirrors; check dataset versions, permissions, and licenses.

## 6. Methodological and privacy qualifications

- P4P and TrustFed-IDS-style references in the experiments are **independent reimplementations**, not original-author software releases.
- The federated simulation does **not** implement or benchmark a cryptographic secure-aggregation protocol.
- Client-level Gaussian differential privacy is evaluated in a **separate aggregate-release mode** with uniform aggregation, not as simultaneous confidential secure aggregation plus Byzantine client screening.
- The mechanism-level study is not evidence of universal adversarial attribution, perfect poisoning defense, or zero-shot transfer across datasets.
- The repository does not contain datasets, preprocessing caches, prediction files, full dataset snapshots, or execution logs; the **externally retained historical results** should be distinguished from the public code release.

## 7. Referencing this research artifact

Please cite the manuscript for scientific claims. When discussing numerical results from **Section 5.6.1 / Table 8**, identify them as outputs from the **completed V6 Reviewer-3 mechanism-level study (16 October 2026)**. When discussing the code in `RPP_FL_V7_ALIGNED_COUPLING.py`, describe it as the **later source-aligned runner** and do not attribute the historical numerical results to a training execution of that file.

No archival DOI, immutable release tag, or third-party replication certification is claimed here.

## 8. License and reuse

No explicit source-code license is currently included. Public visibility alone does not grant general permission to redistribute or reuse the code; contact the maintainers if permission is needed.
