# RPP-FL — Robust and Privacy-Aware Federated Learning for IIoT Anomaly Detection

**Research implementation · Federated intrusion detection · Non-IID evaluation · Poisoning robustness · Explainable aggregation diagnostics**

RPP-FL is an experimental federated learning framework for binary anomaly and intrusion detection in the Industrial Internet of Things (IIoT). It combines a hybrid Transformer–MLP detector with client-update trust assessment, persistent reputation, collusion-aware safeguards, and robust server aggregation. A separate experimental mode studies the privacy–utility trade-off of client-level Gaussian differential privacy.

This repository presents the **RPP-FL research project as one framework**, together with its evaluation code, reproducibility instructions, and documented experimental evidence.

> **Documentation correction — 19 October 2026:** Earlier README wording was ambiguous and could be read as suggesting that the mechanism-level experiments supporting the manuscript had not been executed. **They were executed.** The archived study contains **90 completed scenarios**, individual run results, training histories, checkpoints, and an execution-state record. The three-seed means reported in **Section 5.6.1 and Table 8** were checked against these saved results. The earlier statement about a code-only experimental runner should not have been interpreted as denying the completed underlying study.

## 1. Research objectives

RPP-FL is designed to investigate three related questions:

1. Can a federated hybrid neural detector maintain strong intrusion-detection utility across heterogeneous IIoT traffic data?
2. How do geometric trust, event-triggered evidence, persistent reputation, and robust aggregation affect an attacker's influence when some participating clients are malicious?
3. What privacy–utility trade-offs arise when a separate client-level differentially private aggregate-release mechanism is applied?

The work considers non-IID client partitions, ALIE-style poisoning, robustness stress tests, ablations, matched baselines, and protocol-specific diagnostic metrics. It does **not** claim universal attack identification or cross-dataset zero-shot transfer.

## 2. Framework architecture

### 2.1 Local anomaly detector

The local model combines learned feature-token representations, Transformer-based interactions, and a compact MLP prediction head. Local clients optimize the detector using their own partitioned records without sharing those raw training records as part of standard federated updates.

### 2.2 Federated orchestration

The experimental framework supports configurable client participation, locally trained updates, checkpointing, validation-driven selection, and aggregation across communication rounds. Client partitioning can be configured to assess heterogeneous label distributions.

### 2.3 Trust- and reputation-aware robust aggregation

The proposed RPP aggregation family evaluates client model updates using:

- Robust geometric deviation from a client-update center.
- Consistency with the current center and historical update direction.
- Peer-agreement and compact-minority/collusion-related evidence.
- Persistent client reputation with asymmetric adjustment.
- Reputation- and trust-aware aggregation weights.
- Coordinate-wise winsorization and a median-based aggregation fallback.

The mechanism evaluation separately reports **predictive performance**, **malicious aggregation-weight share**, and **client attack-detection AUC**. These measures answer different questions and should not be treated as interchangeable.

### 2.4 Separate differential-privacy evaluation

The privacy experiments use clipped client updates, uniform aggregation, Gaussian noise, and Rényi differential privacy accounting in a **separate aggregate-release mode**. They do not demonstrate simultaneous cryptographic secure aggregation and individual-client Byzantine screening.

## 3. Datasets and evaluation scope

The code supports the project's dataset-processing and experiment workflows for several public IIoT/IoT intrusion-detection datasets.

| Dataset | Research role |
|---|---|
| CICIoT2023 | Federated detection, matched comparisons, and mechanism evaluation |
| TON_IoT | Federated detection, poisoning robustness, privacy experiments, and mechanism evaluation |
| EdgeIIoTset | Included in the retained comprehensive revision-suite evaluations |
| WUSTL-IIoT | Included in the broader manuscript/dataset-processing workflow; its evaluation results are **not** part of the 185-run archive described below |
| X-IIoTID | Dataset configuration support; no evaluation result is asserted from the cited archives |

**Scope clarification:** The retained **185-run comprehensive revision archive** covers CICIoT2023, TON_IoT, and EdgeIIoTset. Its presence does not independently verify every dataset-wide claim in the manuscript. The separate mechanism-level results cover **CICIoT2023 and TON_IoT**.

Datasets may be accessed through third-party sources according to the execution configuration. Dataset licenses, release identities, and raw-data snapshots should be independently checked before reusing or distributing data.

## 4. Experimental program and evidence

The research contains a comprehensive evaluation program and a focused mechanism-level study. They are related parts of the same RPP-FL investigation, with **separately preserved experiment records**.

| Study component | Recorded scope | Evidence |
|---|---|---|
| Comprehensive reviewer-revision suite | **185 recorded scenarios** | Archived master result rows, scenario manifests, results, and prediction files were checked for internal consistency |
| Mechanism-level aggregation study | **90/90 completed scenarios** | Retained per-seed outputs, execution-state records, training histories, and checkpoints |

### 4.1 Comprehensive evaluation: 185 scenarios

The retained 185-scenario archive consists of:

| Evaluation group | Recorded scenarios |
|---|---:|
| Matched method comparisons | 108 |
| Component ablations | 48 |
| Attack/stress evaluations | 16 |
| Time-separated reference evaluations | 9 |
| Privacy evaluations | 4 |
| **Total** | **185** |

A stored code-to-results audit found 185 expected scenario names, 185 run manifests, 185 final-result records, and 185 test-prediction CSVs. F1 and accuracy were recomputed from the archived predictions, with agreement reported for all 185 saved runs. This is an **archive-consistency audit**, not an independent GPU retraining or proof of identical execution-time source bytes.

The consolidated 185-run archive is **not included in this GitHub repository**, and this README does not imply that all of its data and predictions are publicly deposited here.

### 4.2 Mechanism-level experiment: 90 completed scenarios

The additional mechanism study follows:

| Item | Recorded value |
|---|---|
| Datasets | CICIoT2023; TON_IoT |
| Aggregation configurations | Simple temporal trust; uncoupled RPP; partial event coupling; full event coupling; P4P |
| Random seeds | 42; 2026; 31415 |
| Conditions | Clean (Dirichlet α = 0.5); ALIE-20% (α = 0.5); ALIE-20% (α = 0.1) |
| Per-scenario training budget | 25 communication rounds |
| Completed scenarios | **2 × 5 × 3 × 3 = 90/90** |
| Recorded execution start/resumption | 16 October 2026, 07:12:15 |
| Recorded completion | 16 October 2026, 14:37:53 |
| Execution-state status | `COMPLETED` |

The timestamps in `execution_state.json` have **no explicit time-zone offset**. The archived file metadata records creation or modification of the results on **16 October 2026** (metadata timestamps in UTC).

The original experimental records are retained separately by the research team; they are **not** included in the public GitHub source-code repository.

### 4.3 Evidence retained for the mechanism study

The retained experiment archive includes the following key files:

| Artifact | Purpose |
|---|---|
| `_STATE/execution_state.json` | Execution status, completed scenario identifiers, and recorded start/completion times |
| `run_config.json` | Configuration, seeds, datasets, and execution-environment metadata |
| `ALL_DATASETS_FINAL_RESULTS.csv` | Consolidated run-level results |
| `R3_TABLES/MECHANISM_PER_SEED.csv` | Independent numerical record for each seed and condition |
| `R3_TABLES/MECHANISM_MEAN_SD_CI95.csv` | Means, standard deviations, and reported confidence intervals |
| `R3_TABLES/FULL_VS_ALTERNATIVES_PAIRED.csv` | Paired performance differences |
| `R3_TABLES/PAIRED_DIFFERENCE_SUMMARY.csv` | Grouped paired-difference statistics |
| `R3_TABLES/ROUND_LEVEL_SECURITY.csv` | Round-level security-related measures |
| `<dataset>/<scenario>/final_result.json` | Individual experiment result and configuration fingerprint |
| `<dataset>/<scenario>/COMPLETED.json` | Scenario completion marker |
| `<dataset>/<scenario>/round_history.csv` | Training-round metrics and diagnostic history |
| `<dataset>/<scenario>/best_checkpoint.pt` | Saved best model state |
| `<dataset>/<scenario>/last_checkpoint.pt` | Saved last model state |

The execution-state record lists 90 completed identifiers. The 90 per-seed result rows were checked for unique dataset/scenario identifiers and agreement with the aggregation summaries.

### 4.4 Independent numerical check of manuscript Table 8

For **TON_IoT**, **ALIE-20% poisoning**, and **Dirichlet α = 0.5**, recomputation from the archived records reproduces the manuscript's displayed three-seed means:

| Aggregation configuration | F1 (%) | Client attack-detection AUC | Malicious aggregation weight (%) |
|---|---:|---:|---:|
| Simple robust + temporal trust | 99.195 | 0.258 | 25.161 |
| Uncoupled RPP | 99.192 | 1.000 | 2.583 |
| Partial event coupling | 99.184 | 1.000 | approximately 0 |
| Full event coupling | 99.180 | 1.000 | approximately 0 |
| P4P (independent reimplementation) | 99.301 | 0.549 | 13.744 |

The stored full-coupling malicious-weight mean is approximately **0.000046%**, rather than exactly zero. The evidence supports a **condition-specific reduction in malicious influence**, not general predictive superiority of the full configuration. In this comparison, P4P retained a higher F1-score.

## 5. Reproducibility and source code

### Repository files

| File | Function |
|---|---|
| `RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py` | Main comprehensive experimental framework and revision-suite workflows |
| `RPP_FL_V7_ALIGNED_COUPLING.py` | Mechanism-comparison runner with configured aggregation variants |
| `RPP_FL_V7_ALIGNED_COUPLING_Colab.ipynb` | Google Colab launcher with embedded source for the aligned mechanism runner |
| `RPP_FL.py` | Additional historical research source retained in the repository |
| `requirements.txt` | Package requirements; not an exact historical environment lockfile |
| `.gitignore` | Generated-file and local-file exclusions |
| `README.md` | Project documentation and research-artifact guide |

### 5.1 Running the comprehensive experiment framework

Set up a compatible Python environment, obtain the datasets, and configure their paths and output destinations before running the comprehensive study:

```bash
python -u RPP_FL_V6_ALL_REVIEWERS_ORIGINAL_AUDITED.py
```

The script uses its configured datasets, experiment toggles, and checkpoint/output directories. The **185-scenario count describes the retained, audited research archive**, not a guarantee that every execution with a different configuration will produce the same scenario count or metrics.

### 5.2 Running an aligned mechanism-comparison study

Open `RPP_FL_V7_ALIGNED_COUPLING_Colab.ipynb` in Google Colab; alternatively, place the mechanism runner and its required comprehensive framework source in the same directory and run:

```bash
python -u RPP_FL_V7_ALIGNED_COUPLING.py
```

The supplied runner checks an expected SHA-256 for its source dependency and is configured for **90 experimental scenarios**. By default, newly generated outputs are directed to `MyDrive/RPP_FL_RESULTS_V7_ALIGNED_COUPLING`, including a protocol manifest and statistical summaries.

**Numerical provenance note:** The values shown in Section 4.4 above are reproduced from the **archived completed mechanism-study records**, not claimed as results from a fresh execution of this published runner. The historical results were produced with a dedicated event-coupling implementation; the public aligned runner provides a related, source-aligned reproduction route. Because the historical run did not save an execution-time source SHA-256, byte-exact identity with the currently published sources cannot be established retrospectively. A new execution should be reported with its own manifest and outputs.

## 6. Reproducibility limits and scientific interpretation

This project provides substantial saved experimental evidence, but the following qualifications matter:

- The **90-run archived execution is documented**, with matching per-seed statistics, scenario outputs, training histories, and checkpoints. The archived evidence verifies saved-result consistency; it does not constitute an independent replication by a third party.
- The original 90-run `run_config.json` contains `"code_sha256": null`. Saved configuration fingerprints do **not** serve as cryptographic source-code checksums.
- The 185-run audit supports **stored-artifact integrity** across the audited scenario records and prediction files, but does not establish a cryptographically proven identity of the exact originally executed source.
- Full raw-data snapshot hashes, complete preprocessing provenance, and a historically exact environment lockfile are not available in this public code repository.
- P4P and TrustFed-IDS-style comparators are **independent research reimplementations**, not official original-author releases.
- Robust client screening uses individual-update information; no cryptographic secure-aggregation backend is implemented or benchmarked as operating simultaneously with that screening.
- Differential privacy is assessed under a separately configured aggregate-release protocol.
- High attack-detection AUC or low malicious weight in one tested setting must not be generalized into universal robustness claims.

The code and archived records are research artifacts. Their numerical evidence should be interpreted together with the specific datasets, attacks, partitions, random seeds, and validation protocols employed.

## 7. Availability, manuscript use, and citation

- **Source code:** [github.com/zaidjasim4/RPP](https://github.com/zaidjasim4/RPP)
- **Completed mechanism experiment (90 scenarios):** Archived execution-state records, per-seed result tables, training histories, and checkpoints are retained separately from the public source-code repository.
- **Manuscript mapping:** Section 5.6.1 and Table 8 correspond to the completed, archived mechanism study.
- **Comprehensive revision-suite results (185 scenarios):** Stored results have undergone an artifact-consistency audit, but this repository does not present them as a public, complete reproduction package.

No dataset redistribution permission, archive DOI, immutable release tag, or independent external replication certification is asserted here.

## 8. License

No explicit software license has been selected for this repository. Source visibility on GitHub does not automatically confer redistribution or modification rights; contact the maintainers regarding reuse.

---

**Documentation maintenance:** The README is intended to describe the RPP-FL research system, its supported experimental workflows, and the provenance of results. Scientific values should be cited from the underlying archived records, not inferred merely from the existence of code.
