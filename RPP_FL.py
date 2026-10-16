# -*- coding: utf-8 -*-
"""
RPP-FL V6 — Reviewer-3 Event-Coupling Mechanism Audit (based on supplied V5)
================================================
Datasets:
    1) CICIoT2023
    2) EdgeIIoTset
    3) TON_IoT (network traffic CSVs recommended)
    4) WUSTL_IIoT (WUSTL-IIoT-2021)
    5) X-IIoTID

Purpose
-------
A reproducible Google-Colab-ready research implementation for binary IoT/IIoT
anomaly/intrusion detection with federated learning, non-IID partitions,
robust reputation-aware aggregation, poisoning attacks, optional DP, and
validation-only model/threshold selection.


REVIEWER-3 SCIENTIFIC WARNING
-----------------------------
The uploaded V5 file's legacy `aggregator="rpp"` does not implement the full
five-signal/event-coupled server described in the supplied RPP-FL manuscript.
V6 therefore ADDS the distinct `aggregator="rpp_event"` research reconstruction.
It does not silently relabel old results or claim to reproduce published figures.
All new numerical claims must come from actual V6 runs. Before publication,
confirm the numerical floor and tie conventions against the original code used
for the previously reported results. No cryptographic secure aggregation is used.

Output: Google Drive /MyDrive/RPP_FL_RESULTS_V6_REVIEWER3
  R3_TABLES/MECHANISM_PER_SEED.csv
  R3_TABLES/MECHANISM_MEAN_SD_CI95.csv
  R3_TABLES/FULL_VS_ALTERNATIVES_PAIRED.csv
  R3_TABLES/ROUND_LEVEL_SECURITY.csv
  REPRODUCIBILITY_DRAFT.zip (local artifact; NOT a public repository)

Major corrections vs. the original script
------------------------------------------
* Correct per-dataset labels (e.g. X-IIoTID uses class1, not anomaly_alert).
* Leakage-prone label/ID columns are removed explicitly.
* Preprocessing is fitted ONLY on the training split.
* Real label-skew non-IID partitioning using a Dirichlet distribution.
* Correct FL update sign: W_new = W_global + aggregated_delta.
* Validation chooses the best round and threshold; test is evaluated ONCE.
* Stronger feature-token Transformer + residual MLP + auxiliary reconstruction.
* Robust aggregation combines current robust center, historical momentum,
  dynamic reputation, sample count, and norm consistency.
* Better poisoning suite: sign-flip, scaling, Gaussian, and ALIE-style stealth.
* Optional central/client DP; epsilon is reported only when a valid RDP
  accountant (Opacus) is available. No fake epsilon approximation.
* Multi-dataset summaries, CSV/JSON outputs, seed=42.
* Automatic public-dataset acquisition and persistent caching in Google Drive.
* Prepared-data cache in Drive so preprocessing is not repeated after restart.
* Round-level resumable checkpoints (last + best), including RNG/accountant state.
* Scenario-level execution state: completed runs are skipped automatically.
* Final publication suite: five-dataset main results, reviewer-matched controlled
  baselines, attack/rate robustness, component ablations, non-IID sensitivity,
  and target-epsilon privacy-utility experiments.
* Reviewer-1 addition: a recent poisoning-specific P4P baseline is reimplemented
  under the SAME preprocessing, client partitions, local model, optimizer, client
  sampling seed, malicious-client identities, attack generator, and round budget.
* Reviewer-controlled runs disable early stopping so every compared method receives
  the same 25-round training budget; validation still selects the reported checkpoint.
* A dedicated reviewer table and protocol manifest are exported automatically.
* Leakage-safe deduplication is performed after dropping known IDs/label fields.
* Paper-ready CSV tables are regenerated after every completed scenario.

IMPORTANT SCIENTIFIC NOTE
-------------------------
This code is designed to obtain strong, reproducible results legitimately.
It does NOT guarantee that the proposed method will outperform every paper.
Do not report metrics until they are produced by actual runs.

The training simulator keeps client updates in process memory. Thus, unless an
actual cryptographic transport/secure-aggregation backend is plugged in, the
code must NOT be described as proving end-to-end homomorphic-encryption
confidentiality. An optional Paillier micro-benchmark is provided separately.
"""

from __future__ import annotations

import os
import re
import gc
import glob
import json
import math
import time
import copy
import random
import hashlib
import warnings
import shutil
import sys
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Tuple, Optional, Sequence, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# =============================================================================
# 0. GLOBAL SETTINGS
# =============================================================================

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_ENABLED = bool(torch.cuda.is_available())

# -----------------------------------------------------------------------------
# Colab / Drive paths. You do NOT need to upload the five datasets manually:
# missing data are downloaded automatically into DATA_ROOT, then reused from
# Drive on all later executions. Existing user-provided files are respected.
# -----------------------------------------------------------------------------

MOUNT_GOOGLE_DRIVE = True
DRIVE_ROOT = "/content/drive/MyDrive"
DATA_ROOT = f"{DRIVE_ROOT}/RPP_FL_DATASETS"
RESULT_ROOT = f"{DRIVE_ROOT}/RPP_FL_RESULTS_V6_REVIEWER3"
CACHE_ROOT = f"{DRIVE_ROOT}/RPP_FL_CACHE_V6_REVIEWER3"
STATE_ROOT = f"{RESULT_ROOT}/_STATE"

# If True, the script acquires missing datasets automatically and copies them
# into DATA_ROOT on Drive. Later executions reuse the Drive copies.
AUTO_DOWNLOAD_DATASETS = True
FORCE_REDOWNLOAD = False

# Persistence / resume. Checkpoint is written to Drive after EVERY round by
# default. If Colab disconnects, rerun the same script and it resumes.
AUTO_RESUME = True
FORCE_RETRAIN_COMPLETED = False
CHECKPOINT_EVERY_N_ROUNDS = 1
CACHE_PREPARED_DATA = True

# "colab" uses manageable, public tabular distributions/mirrors. The source
# and provenance of every downloaded dataset are recorded in Drive manifests.
DATA_DOWNLOAD_PROFILE = "colab"

DATASET_PATHS: Dict[str, str] = {
    "CICIoT2023": f"{DATA_ROOT}/CICIoT2023",
    "EdgeIIoTset": f"{DATA_ROOT}/EdgeIIoTset",
    "TON_IoT": f"{DATA_ROOT}/TON_IoT",
    "WUSTL_IIoT": f"{DATA_ROOT}/WUSTL_IIoT",
    "X-IIoTID": f"{DATA_ROOT}/X-IIoTID",
}

# Large datasets are sampled reproducibly for tractable Colab experiments.
# Set None for full data (only recommended with enough RAM/time).
MAX_ROWS_PER_DATASET: Optional[int] = 300_000
CSV_CHUNK_SIZE = 50_000

# Reviewer-3 default: run ONLY mechanism-level experiments, not the enormous
# legacy V5 suite. Matched within seed: model, data, client selection, attack
# *rule*, and training budget. Trajectories diverge across methods, so the
# constructed ALIE vectors need not be literally identical across methods.
RUN_REVIEWER3_MECHANISM_ONLY = True
R3_DATASETS = ("CICIoT2023", "TON_IoT")
R3_SEEDS = (42, 2026, 31415)
R3_VARIANTS = ("simple_temporal", "uncoupled", "partial", "full", "p4p")
R3_INCLUDE_P4P = True
R3_RUN_CLEAN_CONTROL = True
R3_RUN_SEVERE_NONIID = True
# Settings used by the manuscript, where specified. The strictly positive
# MAD numerical floor is otherwise underspecified in manuscript Section 3.4.
R3_MAD_FLOOR_RELATIVE = 0.01
R3_MAD_FLOOR_ABSOLUTE = 1e-10
R3_AUDIT_VERSION = 6

# -----------------------------------------------------------------------------
# FINAL EXPERIMENT PLAN
# -----------------------------------------------------------------------------
# The default profile is publication-oriented but computationally controlled:
#   * Main clean + ALIE(20%) on ALL five datasets.
#   * Controlled baselines and ablations on CICIoT2023 + TON_IoT.
#   * Attack-type and malicious-ratio robustness on CICIoT2023 + TON_IoT.
#   * Non-IID sensitivity on CICIoT2023 + TON_IoT.
#   * Privacy-utility sweep on TON_IoT (local client-level Gaussian DP).
# Every scenario has round-level checkpoint/resume and completed-run skipping.
# Turn individual suites off if Colab time is limited; rerunning later safely
# continues from the saved Drive state.
RUN_MAIN_FIVE_DATASETS = True
RUN_ATTACK_SUITE = True
RUN_MALICIOUS_RATIO_SWEEP = True
# The legacy baseline suite is superseded by the stricter Reviewer-1 matched suite.
RUN_BASELINES = False
RUN_REVIEWER1_MATCHED_BASELINES = True
# Core direct comparison: Clean + ALIE(20%) + severe non-IID (alpha=0.1)+ALIE(20%).
RUN_REVIEWER1_SEVERE_NONIID = True
# Additional P4P-vs-RPP matched runs for Sign-Flip, Scaling, and Gaussian poisoning.
# Leave True for the revision package; turn False only for quick debugging.
RUN_REVIEWER1_EXTENDED_ATTACKS = True
RUN_ABLATION = True
RUN_NONIID_SWEEP = True
RUN_PRIVACY_SWEEP = True

PRIMARY_ATTACK = "alie"      # {alie, signflip, scale, gaussian}
MAIN_ATTACK_FRACTION = 0.20

ABLATION_DATASETS = ("CICIoT2023", "TON_IoT")
BASELINE_DATASETS = ("CICIoT2023", "TON_IoT")
ROBUSTNESS_DATASETS = ("CICIoT2023", "TON_IoT")
NONIID_DATASETS = ("CICIoT2023", "TON_IoT")
PRIVACY_DATASETS = ("TON_IoT",)

ATTACK_TYPES = ("signflip", "scale", "gaussian", "alie")
MALICIOUS_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
NONIID_ALPHAS = (1.0, 0.5, 0.1)
TARGET_EPSILONS = (20.0, 10.0, 5.0, 1.0)

# Reviewer-1 controlled-comparison settings. P4P is a recent poisoning-specific
# FL-IDS defense (Khang et al., JNCA 2026, DOI 10.1016/j.jnca.2026.104502).
# The implementation below follows the publicly documented P4P pipeline:
# historical global-trajectory probe -> server-side cosine response -> L2-norm
# filtering -> ensemble anomaly evidence -> temporal suspicion -> filtered FedAvg.
# It is an independent, transparent reimplementation, not the authors' official code.
P4P_REFERENCE_DOI = "10.1016/j.jnca.2026.104502"
REVIEWER1_SEVERE_ALPHA = 0.10
REVIEWER1_EXTENDED_ATTACK_TYPES = ("signflip", "scale", "gaussian")

# Paillier remains a measured cryptographic micro-benchmark rather than a fake
# claim of end-to-end encrypted training. It is cheap enough to run once.
RUN_PAILLIER_MICROBENCHMARK = True
PAILLIER_BENCHMARK_COORDS = 64
PAILLIER_KEY_BITS = 1024

# =============================================================================
# 1. AUTOMATIC DATA ACQUISITION + DATASET SPECIFICATIONS
# =============================================================================

@dataclass(frozen=True)
class DownloadCandidate:
    handle: str
    kaggle_path: Optional[str] = None
    note: str = ""


# Public tabular distributions/mirrors. KaggleHub is used because it can fetch
# public datasets programmatically in Colab. The exact chosen source is saved
# in <dataset>/_dataset_ready.json for auditability.
#
# CICIoT2023 note: the default Colab profile uses a compact public CSV mirror
# (1.19M rows) derived from CICIoT2023, rather than downloading the original
# ~46.7M-row release. Change the candidate if a full-release mirror is desired.
DOWNLOAD_CANDIDATES: Dict[str, Tuple[DownloadCandidate, ...]] = {
    "CICIoT2023": (
        DownloadCandidate(
            "subhajournal/iotintrusion",
            "IoT_Intrusion.csv",
            "Compact public CICIoT2023 CSV mirror for Colab-scale experiments.",
        ),
        DownloadCandidate(
            "subhajournal/iotintrusion",
            None,
            "Fallback: download the complete compact CICIoT2023 Kaggle package.",
        ),
    ),
    "EdgeIIoTset": (
        DownloadCandidate(
            "mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot",
            "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv",
            "Publisher-associated Edge-IIoTset Kaggle distribution; DNN-ready CSV only.",
        ),
    ),
    "TON_IoT": (
        DownloadCandidate(
            "arnobbhowmik/ton-iot-network-dataset",
            "train_test_network.csv",
            "TON_IoT network-flow CSV distribution; binary label column is retained.",
        ),
        DownloadCandidate(
            "arnobbhowmik/ton-iot-network-dataset",
            None,
            "Fallback: download the complete TON_IoT network Kaggle package.",
        ),
    ),
    "WUSTL_IIoT": (
        DownloadCandidate(
            "annaamalaiu/wustl-iiot-2021-dataset",
            "wustl_iiot_2021.csv",
            "WUSTL-IIoT-2021 CSV mirror; leakage-prone identifier fields are dropped later.",
        ),
        DownloadCandidate(
            "annaamalaiu/wustl-iiot-2021-dataset",
            None,
            "Fallback: download the complete WUSTL-IIoT-2021 Kaggle package.",
        ),
    ),
    "X-IIoTID": (
        # Download the complete author package first instead of assuming an
        # internal filename. Kaggle package filenames have changed across
        # mirrors; recursive tabular discovery below is filename-agnostic.
        DownloadCandidate(
            "munaalhawawreh/xiiotid-iiot-intrusion-dataset",
            None,
            "Complete X-IIoTID Kaggle distribution from the dataset author account.",
        ),
        DownloadCandidate(
            "annaamalai1752/x-iiotd-dataset",
            None,
            "Fallback public X-IIoTID package mirror.",
        ),
    ),
}


def _tabular_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path] if Path(path).suffix.lower() in {".csv", ".parquet", ".pq", ".feather"} else []
    if not os.path.isdir(path):
        return []
    files: List[str] = []
    for pattern in ("**/*.csv", "**/*.parquet", "**/*.pq", "**/*.feather"):
        files.extend(glob.glob(os.path.join(path, pattern), recursive=True))
    return sorted(set(files))


def _copy_download_to_drive(downloaded_path: str, dest_dir: str) -> List[str]:
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    sources = _tabular_files(downloaded_path)
    if not sources and os.path.isdir(downloaded_path):
        # Sometimes a single requested file is nested under the cache directory.
        sources = _tabular_files(downloaded_path)
    if not sources:
        raise FileNotFoundError(f"No tabular file found in downloaded path: {downloaded_path}")

    copied: List[str] = []
    for src in sources:
        # Flatten to the dataset directory. Duplicate basenames get a short hash.
        base = os.path.basename(src)
        dst = os.path.join(dest_dir, base)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            copied.append(dst)
            continue
        if os.path.exists(dst):
            stem, ext = os.path.splitext(base)
            dst = os.path.join(dest_dir, f"{stem}_{stable_hash(src)}{ext}")
        print(f"[DATA] Copying to Drive: {base} ({os.path.getsize(src)/1024**2:.1f} MB)")
        tmp = dst + ".partial"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        copied.append(dst)
    return copied


def ensure_dataset_downloaded(dataset_name: str, force: bool = False) -> str:
    """Download once, persist on Drive, and reuse on every later execution."""
    dest = DATASET_PATHS[dataset_name]
    marker = os.path.join(dest, "_dataset_ready.json")
    existing = _tabular_files(dest)

    if not force and existing:
        print(f"[DATA] {dataset_name}: using Drive cache ({len(existing)} tabular file(s)).")
        if not os.path.exists(marker):
            atomic_json_dump({
                "dataset": dataset_name,
                "status": "ready_existing",
                "files": [os.path.basename(x) for x in existing],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, marker)
        return dest

    if not AUTO_DOWNLOAD_DATASETS:
        raise FileNotFoundError(f"{dataset_name} missing at {dest} and AUTO_DOWNLOAD_DATASETS=False")

    pip_install_if_missing("kagglehub", "kagglehub")
    import kagglehub  # type: ignore

    Path(dest).mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for cand in DOWNLOAD_CANDIDATES[dataset_name]:
        try:
            print(f"[DATA] Downloading {dataset_name} from Kaggle: {cand.handle}")
            if cand.kaggle_path:
                downloaded = kagglehub.dataset_download(
                    cand.handle, path=cand.kaggle_path, force_download=force
                )
            else:
                downloaded = kagglehub.dataset_download(cand.handle, force_download=force)
            copied = _copy_download_to_drive(str(downloaded), dest)
            atomic_json_dump({
                "dataset": dataset_name,
                "status": "ready",
                "kaggle_handle": cand.handle,
                "kaggle_path": cand.kaggle_path,
                "note": cand.note,
                "files": [os.path.basename(x) for x in copied],
                "sizes_bytes": {os.path.basename(x): os.path.getsize(x) for x in copied},
                "download_profile": DATA_DOWNLOAD_PROFILE,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, marker)
            print(f"[DATA] {dataset_name}: cached permanently in {dest}")
            return dest
        except Exception as e:
            last_err = e
            print(f"[WARN] {dataset_name} source failed: {cand.handle} / {cand.kaggle_path}: {e}")

    raise RuntimeError(
        f"Automatic download failed for {dataset_name}. Last error: {last_err}. "
        "If Kaggle requests authentication, add your Kaggle token to Colab once; "
        "the downloaded files will still be persisted in Drive and reused thereafter."
    )


def ensure_all_datasets() -> None:
    print_header("DATASET ACQUISITION / DRIVE CACHE")
    for name in DATASET_PATHS:
        try:
            ensure_dataset_downloaded(name, force=FORCE_REDOWNLOAD)
        except Exception as e:
            print(f"[ERROR] Dataset acquisition failed for {name}: {type(e).__name__}: {e}")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    label_candidates: Tuple[str, ...]
    normal_values: Tuple[str, ...]
    attack_values: Tuple[str, ...] = ()
    leakage_cols: Tuple[str, ...] = ()
    id_cols: Tuple[str, ...] = ()
    label_is_numeric_binary: bool = False


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "CICIoT2023": DatasetSpec(
        name="CICIoT2023",
        label_candidates=("label", "Label", "LABEL"),
        normal_values=("BENIGNTRAFFIC", "BENIGN", "NORMAL", "0", "FALSE"),
        leakage_cols=("attack_type", "attack_cat", "category", "subcategory"),
        id_cols=(
            "Flow ID", "flow_id", "Timestamp", "timestamp",
            "Src IP", "Dst IP", "src_ip", "dst_ip",
        ),
    ),
    "EdgeIIoTset": DatasetSpec(
        name="EdgeIIoTset",
        label_candidates=("Attack_label", "attack_label", "label", "Label"),
        normal_values=("0", "NORMAL", "BENIGN", "FALSE"),
        leakage_cols=("Attack_type", "attack_type", "Attack Type", "attack_cat"),
        id_cols=(
            "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4", "tcp.srcport", "tcp.dstport",
        ),
        label_is_numeric_binary=True,
    ),
    "TON_IoT": DatasetSpec(
        name="TON_IoT",
        label_candidates=("label", "Label", "LABEL", "target", "Target"),
        normal_values=("0", "NORMAL", "BENIGN", "FALSE"),
        leakage_cols=("type", "Type", "attack_type", "Attack_type", "attack_cat"),
        id_cols=(
            "ts", "timestamp", "Timestamp", "src_ip", "dst_ip",
            "src_addr", "dst_addr", "src_port", "dst_port",
        ),
        label_is_numeric_binary=True,
    ),
    "WUSTL_IIoT": DatasetSpec(
        name="WUSTL_IIoT",
        label_candidates=("Traffic", "traffic", "Target", "target", "label", "Label"),
        normal_values=("NORMAL", "NORMAL TRAFFIC", "0", "FALSE", "BENIGN"),
        leakage_cols=("Traffic_type", "Attack_type", "attack_type", "category"),
        # Official WUSTL-IIoT-2021 guidance recommends removing these fields.
        id_cols=("StartTime", "LastTime", "SrcAddr", "DstAddr", "sIpId", "dIpId"),
    ),
    "X-IIoTID": DatasetSpec(
        name="X-IIoTID",
        label_candidates=("class1", "Class1", "CLASS1"),
        normal_values=("NORMAL", "0", "FALSE", "BENIGN"),
        # class2/class3 are higher-resolution labels. anomaly_alert is an IDS
        # alert signal and is removed here to avoid circular target leakage.
        leakage_cols=(
            "class2", "class3", "Class2", "Class3",
            "anomaly_alert", "Anomaly_alert", "anomaly alert",
        ),
        id_cols=(
            "Date", "Timestamp", "Scr_IP", "Des_IP", "Src_IP", "Dst_IP",
            "Scr_port", "Des_port", "src_ip", "dst_ip",
        ),
    ),
}


# =============================================================================
# 2. REPRODUCIBILITY / UTILITIES
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic CuDNN is reproducible but can be slower.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    Path(DATA_ROOT).mkdir(parents=True, exist_ok=True)
    Path(RESULT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
    Path(STATE_ROOT).mkdir(parents=True, exist_ok=True)


def mount_drive_if_needed() -> None:
    if not MOUNT_GOOGLE_DRIVE:
        return
    try:
        from google.colab import drive  # type: ignore
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
    except Exception as e:
        print(f"[INFO] Google Drive mount skipped: {e}")


def stable_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def atomic_json_dump(obj: Any, path: str) -> None:
    """Write JSON atomically; a killed Colab never leaves a valid-looking partial file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_dataframe_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def pip_install_if_missing(import_name: str, package_name: Optional[str] = None) -> None:
    try:
        __import__(import_name)
        return
    except Exception:
        pass
    pkg = package_name or import_name
    print(f"[SETUP] Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def canonical_col_lookup(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    by_lower = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in by_lower:
            return str(by_lower[key])
    return None


def safe_to_numeric(s: pd.Series) -> Tuple[pd.Series, float]:
    out = pd.to_numeric(s, errors="coerce")
    ratio = float(out.notna().mean())
    return out, ratio


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# =============================================================================
# 3. MEMORY-SAFE FILE LOADING
# =============================================================================

def _priority_sample_csv(path: str, k: int, seed: int,
                         chunksize: int = CSV_CHUNK_SIZE) -> pd.DataFrame:
    """Uniform row sample without loading the whole CSV into memory."""
    rng = np.random.default_rng(seed)
    cand_df: Optional[pd.DataFrame] = None
    cand_keys: Optional[np.ndarray] = None

    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunksize):
        if len(chunk) == 0:
            continue
        keys = rng.random(len(chunk), dtype=np.float64)
        if cand_df is None:
            cand_df = chunk.reset_index(drop=True)
            cand_keys = keys
        else:
            cand_df = pd.concat([cand_df, chunk], ignore_index=True)
            cand_keys = np.concatenate([cand_keys, keys])  # type: ignore[arg-type]

        if len(cand_df) > k:
            assert cand_keys is not None
            keep = np.argpartition(cand_keys, kth=k - 1)[:k]
            cand_df = cand_df.iloc[keep].reset_index(drop=True)
            cand_keys = cand_keys[keep]

    if cand_df is None:
        raise ValueError(f"No rows read from {path}")
    if len(cand_df) > k:
        cand_df = cand_df.sample(n=k, random_state=seed).reset_index(drop=True)
    return cand_df


def _read_any_file(path: str, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        if max_rows is None:
            return pd.read_csv(path, low_memory=False)
        return _priority_sample_csv(path, max_rows, seed)
    if ext in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif ext in {".feather", ".ft"}:
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported tabular file: {path}")
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    return df.reset_index(drop=True)


def load_dataset_path(path: str, max_rows: Optional[int], seed: int) -> pd.DataFrame:
    """
    Load one file or recursively load a folder. For multi-file datasets, the
    sampling budget is allocated approximately in proportion to file size,
    avoiding the severe distribution distortion caused by equal-per-file
    sampling when attack files have very different sizes.
    """
    if os.path.isfile(path):
        return _read_any_file(path, max_rows, seed)

    if not os.path.isdir(path):
        raise FileNotFoundError(path)

    files: List[str] = []
    for pattern in ("**/*.csv", "**/*.parquet", "**/*.pq", "**/*.feather"):
        files.extend(glob.glob(os.path.join(path, pattern), recursive=True))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No CSV/Parquet/Feather files under: {path}")

    if max_rows is None:
        frames = [_read_any_file(f, None, seed + i) for i, f in enumerate(files)]
        return pd.concat(frames, ignore_index=True)

    sizes = np.array([max(os.path.getsize(f), 1) for f in files], dtype=np.float64)
    probs = sizes / sizes.sum()
    budgets = np.maximum(100, np.floor(probs * max_rows).astype(int))
    # Normalize the budget back near max_rows.
    if budgets.sum() > max_rows:
        scale = max_rows / budgets.sum()
        budgets = np.maximum(1, np.floor(budgets * scale).astype(int))

    frames = []
    for i, (f, k) in enumerate(zip(files, budgets.tolist())):
        try:
            frames.append(_read_any_file(f, int(k), seed + 1009 * (i + 1)))
        except Exception as e:
            print(f"[WARN] Skipping file {f}: {e}")
    if not frames:
        raise RuntimeError(f"Could not read any file under: {path}")
    df = pd.concat(frames, ignore_index=True)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


# =============================================================================
# 4. LABEL ADAPTERS / LEAKAGE CONTROL
# =============================================================================

def make_binary_labels(df: pd.DataFrame, spec: DatasetSpec) -> Tuple[np.ndarray, str]:
    label_col = canonical_col_lookup(df, spec.label_candidates)
    if label_col is None:
        raise ValueError(
            f"[{spec.name}] Could not find label column. Tried: {spec.label_candidates}. "
            f"Available columns (first 40): {list(df.columns)[:40]}"
        )

    s = df[label_col]
    num, numeric_ratio = safe_to_numeric(s)
    unique_num = sorted(pd.Series(num.dropna().unique()).tolist())

    # Numeric 0/1 labels are handled directly when unambiguous.
    if numeric_ratio > 0.995 and len(unique_num) <= 3 and set(unique_num).issubset({0, 1}):
        y = num.fillna(0).astype(int).to_numpy(dtype=np.int64)
        return y, label_col

    normal = {v.upper().strip() for v in spec.normal_values}
    attack = {v.upper().strip() for v in spec.attack_values}
    st = s.astype(str).str.strip().str.upper()

    if attack:
        unknown = ~(st.isin(normal) | st.isin(attack))
        if unknown.any():
            print(f"[WARN] [{spec.name}] {int(unknown.sum())} unknown label values mapped as attack.")
        y = (~st.isin(normal)).astype(np.int64).to_numpy()
    else:
        # Any non-normal class is an attack. This is appropriate for datasets
        # whose label column contains many attack names.
        y = (~st.isin(normal)).astype(np.int64).to_numpy()

    if len(np.unique(y)) != 2:
        vc = pd.Series(s).value_counts().head(20)
        raise ValueError(
            f"[{spec.name}] Binary mapping did not produce two classes. "
            f"Top raw labels:\n{vc}"
        )
    return y, label_col


def drop_known_leakage(df: pd.DataFrame, spec: DatasetSpec, label_col: str) -> pd.DataFrame:
    out = df.copy()
    to_drop: List[str] = [label_col]

    # Case-insensitive exact matches for explicit leakage / IDs.
    lower_map = {str(c).strip().lower(): c for c in out.columns}
    for name in list(spec.leakage_cols) + list(spec.id_cols):
        col = lower_map.get(name.strip().lower())
        if col is not None:
            to_drop.append(str(col))

    # Remove alternative label columns if present (e.g. WUSTL Traffic + Target).
    for cand in spec.label_candidates:
        col = lower_map.get(cand.strip().lower())
        if col is not None and str(col) != label_col:
            to_drop.append(str(col))

    # Generic ultra-unique identifiers: exact names only, not fuzzy feature names.
    generic_ids = {
        "flow id", "flowid", "srcip", "dstip", "source ip", "destination ip",
        "source_ip", "destination_ip", "mac", "src_mac", "dst_mac"
    }
    for c in out.columns:
        if str(c).strip().lower() in generic_ids:
            to_drop.append(str(c))

    out = out.drop(columns=sorted(set(to_drop)), errors="ignore")
    out = out.dropna(axis=1, how="all")
    return out


# =============================================================================
# 5. LEAKAGE-SAFE PREPROCESSING
# =============================================================================

class LeakageSafePreprocessor:
    """
    Fit only on training data.

    Numeric:
      median imputation -> train quantile clipping -> z-score.
    Categorical:
      low-cardinality -> train-fitted one-hot;
      high-cardinality -> train-fitted frequency encoding.
    Constant / nearly-empty columns are removed.
    """

    def __init__(self, onehot_max_unique: int = 20,
                 clip_q_low: float = 0.001,
                 clip_q_high: float = 0.999,
                 max_missing: float = 0.98):
        self.onehot_max_unique = onehot_max_unique
        self.clip_q_low = clip_q_low
        self.clip_q_high = clip_q_high
        self.max_missing = max_missing

        self.keep_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.lowcat_cols: List[str] = []
        self.highcat_cols: List[str] = []

        self.num_stats: Dict[str, Tuple[float, float, float, float]] = {}
        self.lowcat_levels: Dict[str, List[str]] = {}
        self.highcat_freq: Dict[str, Dict[str, float]] = {}
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "LeakageSafePreprocessor":
        self.keep_cols = []
        for c in df.columns:
            s = df[c]
            if float(s.isna().mean()) >= self.max_missing:
                continue
            # Treat a column as constant using string representation to avoid
            # mixed-type comparison failures.
            if s.astype(str).nunique(dropna=False) <= 1:
                continue
            self.keep_cols.append(str(c))

        work = df[self.keep_cols].copy()

        for c in self.keep_cols:
            s = work[c]
            num, ratio = safe_to_numeric(s)
            if pd.api.types.is_numeric_dtype(s) or ratio >= 0.98:
                self.numeric_cols.append(c)
                x = num.replace([np.inf, -np.inf], np.nan)
                med = float(x.median()) if x.notna().any() else 0.0
                x = x.fillna(med)
                ql = float(x.quantile(self.clip_q_low))
                qh = float(x.quantile(self.clip_q_high))
                if not np.isfinite(ql): ql = med
                if not np.isfinite(qh): qh = med
                if qh < ql: ql, qh = qh, ql
                xc = x.clip(ql, qh)
                mean = float(xc.mean())
                std = float(xc.std(ddof=0))
                if not np.isfinite(std) or std < 1e-8:
                    std = 1.0
                self.num_stats[c] = (med, ql, qh, mean, std)
            else:
                st = s.astype(str).fillna("__NA__")
                nuniq = int(st.nunique())
                if nuniq <= self.onehot_max_unique:
                    self.lowcat_cols.append(c)
                    # Keep deterministic train-only levels.
                    levels = sorted(st.value_counts().index.astype(str).tolist())
                    self.lowcat_levels[c] = levels
                else:
                    self.highcat_cols.append(c)
                    freq = st.value_counts(normalize=True)
                    self.highcat_freq[c] = {str(k): float(v) for k, v in freq.items()}

        names: List[str] = []
        names.extend(self.numeric_cols)
        for c in self.lowcat_cols:
            names.extend([f"{c}=={lv}" for lv in self.lowcat_levels[c]])
        names.extend([f"{c}__freq" for c in self.highcat_cols])
        self.feature_names_ = names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks: List[np.ndarray] = []

        for c in self.numeric_cols:
            med, ql, qh, mean, std = self.num_stats[c]
            if c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
                x = x.fillna(med).clip(ql, qh)
                arr = ((x.to_numpy(dtype=np.float32) - mean) / std).reshape(-1, 1)
            else:
                arr = np.zeros((n, 1), dtype=np.float32)
            blocks.append(arr.astype(np.float32, copy=False))

        for c in self.lowcat_cols:
            levels = self.lowcat_levels[c]
            if c in df.columns:
                st = df[c].astype(str).fillna("__NA__")
            else:
                st = pd.Series(["__MISSING__"] * n)
            for lv in levels:
                blocks.append((st.to_numpy() == lv).astype(np.float32).reshape(-1, 1))

        for c in self.highcat_cols:
            fmap = self.highcat_freq[c]
            if c in df.columns:
                st = df[c].astype(str).fillna("__NA__")
                arr = st.map(fmap).fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1)
            else:
                arr = np.zeros((n, 1), dtype=np.float32)
            blocks.append(arr)

        if not blocks:
            raise ValueError("Preprocessing produced zero features.")
        X = np.concatenate(blocks, axis=1).astype(np.float32, copy=False)
        X = np.nan_to_num(X, nan=0.0, posinf=8.0, neginf=-8.0)
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)


class TrainOnlyFeatureSelector:
    def __init__(self, max_features: int = 192):
        self.max_features = max_features
        self.selector: Optional[SelectKBest] = None
        self.indices_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrainOnlyFeatureSelector":
        k = min(self.max_features, X.shape[1])
        if X.shape[1] <= k:
            self.indices_ = np.arange(X.shape[1], dtype=np.int64)
            return self
        self.selector = SelectKBest(score_func=f_classif, k=k)
        self.selector.fit(X, y)
        self.indices_ = self.selector.get_support(indices=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selector is None:
            return X.astype(np.float32, copy=False)
        return self.selector.transform(X).astype(np.float32, copy=False)


# =============================================================================
# 6. STRATIFIED RAW SPLIT + DIRICHLET NON-IID PARTITION
# =============================================================================

def stratified_train_val_test_indices(y: np.ndarray, seed: int,
                                      train_frac: float = 0.70,
                                      val_frac: float = 0.15,
                                      test_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9
    idx = np.arange(len(y))
    tr, temp = train_test_split(
        idx, test_size=(1.0 - train_frac), random_state=seed, stratify=y
    )
    temp_y = y[temp]
    rel_test = test_frac / (val_frac + test_frac)
    va, te = train_test_split(
        temp, test_size=rel_test, random_state=seed + 1, stratify=temp_y
    )
    return np.asarray(tr), np.asarray(va), np.asarray(te)


def dirichlet_label_partition(y: np.ndarray, n_clients: int, alpha: float,
                              seed: int, min_samples: int = 64,
                              max_tries: int = 100,
                               strict: bool = False) -> Dict[int, np.ndarray]:
    """Standard label-skew Dirichlet partition, reproducible with seed."""
    if n_clients <= 1:
        return {0: np.arange(len(y), dtype=np.int64)}
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be > 0")

    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    for _ in range(max_tries):
        buckets: List[List[int]] = [[] for _ in range(n_clients)]
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            rng.shuffle(cls_idx)
            proportions = rng.dirichlet(np.full(n_clients, alpha, dtype=np.float64))
            # Discourage pathological oversized clients while keeping label skew.
            capacity = np.array([len(b) < len(y) / n_clients * 2.0 for b in buckets], dtype=np.float64)
            proportions = proportions * capacity
            if proportions.sum() <= 0:
                proportions = np.ones(n_clients) / n_clients
            else:
                proportions = proportions / proportions.sum()
            cuts = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
            parts = np.split(cls_idx, cuts)
            for cid, part in enumerate(parts):
                buckets[cid].extend(part.tolist())

        sizes = [len(b) for b in buckets]
        if min(sizes) >= min_samples:
            out = {}
            for cid, b in enumerate(buckets):
                arr = np.asarray(b, dtype=np.int64)
                rng.shuffle(arr)
                out[cid] = arr
            return out

    if strict:
        raise RuntimeError(
            f"Dirichlet partition failed minimum-size constraint after {max_tries} "
            f"attempts (alpha={alpha}, clients={n_clients}). Refusing IID fallback "
            "in Reviewer-3 protocol."
        )
    print("[WARN] Could not satisfy min client size; using stratified IID fallback.")
    all_idx = np.arange(len(y))
    # Reproducible fallback.
    rng.shuffle(all_idx)
    return {i: a.astype(np.int64) for i, a in enumerate(np.array_split(all_idx, n_clients))}


# =============================================================================
# 7. MODEL: FEATURE-TOKEN TRANSFORMER + RESIDUAL MLP + RECON AUXILIARY LOSS
# =============================================================================

class FeatureTokenizer(nn.Module):
    """Each scalar feature gets its own learnable affine tokenization."""
    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.feature_embedding = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0) + self.feature_embedding.unsqueeze(0)


class HybridTabularTransformer(nn.Module):
    """Hybrid supervised detector with optional Transformer branch.

    The ``use_transformer`` switch is used ONLY for the planned ablation. The
    MLP branch remains identical, so the ablation measures the incremental
    value of feature-token self-attention without changing preprocessing.
    """
    def __init__(self, n_features: int, d_model: int = 32, nhead: int = 4,
                 num_layers: int = 2, dim_ff: int = 96, mlp_hidden: int = 128,
                 dropout: float = 0.10, use_transformer: bool = True):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.n_features = n_features
        self.use_transformer = bool(use_transformer)

        self.mlp_branch = nn.Sequential(
            nn.Linear(n_features, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.GELU(),
        )

        if self.use_transformer:
            self.tokenizer = FeatureTokenizer(n_features, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.enc_norm = nn.LayerNorm(d_model)
            classifier_in = 2 * d_model
            self.recon_head = nn.Linear(d_model, 1)
            self.recon_mlp_head = None
        else:
            # Pure residual-MLP ablation: no tokenizer, attention, CLS token,
            # or Transformer parameters are instantiated.
            self.tokenizer = None
            self.cls = None
            self.encoder = None
            self.enc_norm = None
            classifier_in = d_model
            self.recon_head = None
            self.recon_mlp_head = nn.Linear(d_model, n_features)

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mlp_z = self.mlp_branch(x)
        if self.use_transformer:
            assert self.tokenizer is not None and self.cls is not None
            assert self.encoder is not None and self.enc_norm is not None
            assert self.recon_head is not None
            tok = self.tokenizer(x)
            cls = self.cls.expand(x.size(0), -1, -1)
            z = self.encoder(torch.cat([cls, tok], dim=1))
            z = self.enc_norm(z)
            cls_z = z[:, 0]
            feat_z = z[:, 1:]
            logits = self.classifier(torch.cat([cls_z, mlp_z], dim=1)).squeeze(1)
            recon = self.recon_head(feat_z).squeeze(-1)
        else:
            assert self.recon_mlp_head is not None
            logits = self.classifier(mlp_z).squeeze(1)
            recon = self.recon_mlp_head(mlp_z)
        return logits, recon


# =============================================================================
# 8. TRAINING CONFIG
# =============================================================================

@dataclass
class Config:
    # FL
    n_clients: int = 10
    n_rounds: int = 25
    clients_per_round_frac: float = 0.80
    local_epochs: int = 2
    batch_size: int = 256
    lr_local: float = 8e-4
    weight_decay: float = 1e-4
    lr_global: float = 1.0
    fedprox_mu: float = 0.0

    # Data heterogeneity
    dirichlet_alpha: float = 0.5
    min_client_samples: int = 64

    # Model
    max_model_features: int = 192
    d_model: int = 32
    nhead: int = 4
    num_layers: int = 2
    dim_ff: int = 96
    mlp_hidden: int = 128
    dropout: float = 0.10
    recon_lambda: float = 0.03
    focal_gamma: float = 1.5
    grad_clip: float = 2.0
    use_transformer: bool = True

    # Robust aggregation + component ablation switches
    aggregator: str = "rpp"  # rpp, p4p, fedavg, median, trimmed_mean
    use_reputation: bool = True
    use_momentum: bool = True
    use_robust_center: bool = True
    use_norm_consistency: bool = True
    # V6: separate manuscript-derived server aggregator and factorial variants.
    r3_variant: str = "full"
    r3_strict_noniid: bool = False
    update_clip_C: float = 1.0
    beta_momentum: float = 0.90
    alpha_rep: float = 0.80
    rep_init: float = 0.60
    reputation_power: float = 2.0
    trim_ratio: float = 0.20
    trust_sketch_dim: int = 2048
    trust_floor_quantile: float = 0.10

    # Attack
    frac_malicious: float = 0.20
    attack_type: str = "alie"
    signflip_scale: float = 1.0
    scale_attack: float = 4.0
    gaussian_attack_std: float = 1.0
    alie_z: float = 1.5

    # Differential privacy
    # Main detection/robustness runs use no DP so their metrics are an explicit
    # utility upper bound. A separate privacy sweep uses local client-level DP
    # with Opacus RDP accounting and target-epsilon-calibrated noise.
    # mode: none | local_dp | central_dp
    privacy_mode: str = "none"
    dp_noise_multiplier: float = 0.0
    dp_delta: float = 1e-5
    dp_target_epsilon: Optional[float] = None

    # Optimization / evaluation
    early_stop_patience: int = 6
    min_rounds_before_stop: int = 8
    enable_early_stopping: bool = True

    # P4P poisoning-specific baseline (independent matched-protocol reimplementation).
    # These hyperparameters are explicit so the revision is reproducible and does not
    # silently assume unpublished settings from the reference paper.
    p4p_probe_beta: float = 0.90
    p4p_temporal_beta: float = 0.80
    p4p_norm_z_threshold: float = 3.50
    p4p_instant_threshold: float = 0.60
    p4p_temporal_threshold: float = 0.55
    p4p_min_keep_fraction: float = 0.50
    p4p_iforest_estimators: int = 128
    threshold_objective: str = "f1"
    seed: int = SEED


# =============================================================================
# 9. PYTORCH DATASET / FL VECTOR HELPERS
# =============================================================================

class BinaryTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def flatten_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def set_params_from_vector(model: nn.Module, vec_cpu: torch.Tensor) -> None:
    vec = vec_cpu.to(next(model.parameters()).device)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(vec[offset:offset + n].view_as(p))
            offset += n
    if offset != len(vec):
        raise ValueError("Parameter vector length mismatch")


def clip_vector_l2(x: torch.Tensor, C: float) -> torch.Tensor:
    if C <= 0:
        return x
    n = x.norm()
    if n <= C:
        return x
    return x * (C / (n + 1e-12))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    den = float(a.norm() * b.norm())
    if den < 1e-12:
        return 0.0
    return float(torch.dot(a, b) / (den + 1e-12))


# =============================================================================
# 10. LOSS / CLIENT TRAINING
# =============================================================================

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          pos_weight: torch.Tensor,
                          gamma: float = 1.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    p = torch.sigmoid(logits)
    pt = torch.where(targets > 0.5, p, 1.0 - p)
    mod = torch.pow(torch.clamp(1.0 - pt, min=1e-6), gamma)
    return (mod * bce).mean()


class Client:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray,
                 cfg: Config, model_ctor):
        self.cid = cid
        self.cfg = cfg
        self.model_ctor = model_ctor
        self.n_samples = len(X)
        self.y = y
        self.loader = DataLoader(
            BinaryTabularDataset(X, y),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        if pos > 0 and neg > 0:
            pw = float(np.clip(neg / pos, 0.5, 20.0))
        else:
            pw = 1.0
        self.pos_weight = pw

    def local_train_and_delta(self, W_global: torch.Tensor) -> Tuple[torch.Tensor, float]:
        model = self.model_ctor().to(DEVICE)
        set_params_from_vector(model, W_global)
        model.train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.cfg.lr_local,
            weight_decay=self.cfg.weight_decay
        )
        pos_weight = torch.tensor(self.pos_weight, dtype=torch.float32, device=DEVICE)
        global_vec_dev = W_global.to(DEVICE) if self.cfg.fedprox_mu > 0 else None

        scaler = torch.cuda.amp.GradScaler(enabled=AMP_ENABLED)
        t0 = time.perf_counter()

        for _ in range(self.cfg.local_epochs):
            for xb, yb in self.loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=AMP_ENABLED):
                    logits, recon = model(xb)
                    cls_loss = focal_bce_with_logits(
                        logits, yb, pos_weight, gamma=self.cfg.focal_gamma
                    )
                    rec_loss = F.smooth_l1_loss(recon, xb, beta=0.5)
                    loss = cls_loss + self.cfg.recon_lambda * rec_loss

                    if self.cfg.fedprox_mu > 0 and global_vec_dev is not None:
                        local_vec = torch.cat([p.reshape(-1) for p in model.parameters()])
                        prox = 0.5 * self.cfg.fedprox_mu * torch.sum((local_vec - global_vec_dev) ** 2)
                        loss = loss + prox

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()

        delta = flatten_params(model) - W_global
        elapsed = time.perf_counter() - t0
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return delta.detach().cpu(), elapsed


# =============================================================================
# 11. POISONING ATTACKS
# =============================================================================

def apply_poisoning_attack(delta: torch.Tensor, attack_type: str,
                           benign_stack: Optional[torch.Tensor],
                           cfg: Config, generator: torch.Generator) -> torch.Tensor:
    attack_type = attack_type.lower().strip()
    if attack_type in {"none", "clean", ""}:
        return delta

    if attack_type == "signflip":
        return -cfg.signflip_scale * delta

    if attack_type == "scale":
        return cfg.scale_attack * delta

    if attack_type == "gaussian":
        scale = float(delta.std().item())
        if scale < 1e-8:
            scale = max(float(delta.norm().item()) / math.sqrt(max(len(delta), 1)), 1e-4)
        noise = torch.randn(delta.shape, generator=generator, dtype=delta.dtype)
        return delta + cfg.gaussian_attack_std * scale * noise

    if attack_type in {"alie", "stealth"}:
        # A Little Is Enough (ALIE)-style in-distribution model poisoning.
        # The malicious update is crafted from the current benign coordinate
        # mean and standard deviation, making it less obvious than sign-flip.
        if benign_stack is None or benign_stack.shape[0] < 2:
            return -delta
        mu = benign_stack.mean(dim=0)
        sd = benign_stack.std(dim=0, unbiased=False)
        return mu - cfg.alie_z * sd

    raise ValueError(f"Unknown attack type: {attack_type}")


# =============================================================================
# 12. OPTIONAL RDP ACCOUNTANT
# =============================================================================

class PrivacyAccountant:
    """
    Uses Opacus RDPAccountant when installed. If it is unavailable, epsilon is
    deliberately reported as NaN rather than using an invalid hand-written
    approximation.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.accountant = None
        self.available = False
        if cfg.privacy_mode == "none" or cfg.dp_noise_multiplier <= 0:
            return
        try:
            from opacus.accountants import RDPAccountant  # type: ignore
            self.accountant = RDPAccountant()
            self.available = True
        except Exception:
            self.accountant = None

    def step(self, sample_rate: float) -> float:
        if self.cfg.privacy_mode == "none" or self.cfg.dp_noise_multiplier <= 0:
            return float("inf")
        if not self.available or self.accountant is None:
            return float("nan")
        self.accountant.step(
            noise_multiplier=self.cfg.dp_noise_multiplier,
            sample_rate=float(sample_rate),
        )
        return float(self.accountant.get_epsilon(delta=self.cfg.dp_delta))

    def state_dict(self) -> Optional[dict]:
        if not self.available or self.accountant is None:
            return None
        try:
            return self.accountant.state_dict()
        except Exception:
            return None

    def load_state_dict(self, state: Optional[dict]) -> None:
        if state is None or not self.available or self.accountant is None:
            return
        try:
            self.accountant.load_state_dict(state)
        except Exception as e:
            print(f"[WARN] Could not restore privacy accountant state: {e}")


def estimate_rdp_epsilon(noise_multiplier: float, sample_rate: float,
                         steps: int, delta: float) -> float:
    """Return Opacus RDP epsilon for a fixed per-round Gaussian mechanism.

    This is used for planning the privacy sweep. The final reported epsilon is
    still read from the accountant state accumulated during the actual run.
    """
    if noise_multiplier <= 0:
        return float("inf")
    try:
        from opacus.accountants import RDPAccountant  # type: ignore
    except Exception:
        return float("nan")
    acc = RDPAccountant()
    for _ in range(int(steps)):
        acc.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
    return float(acc.get_epsilon(delta=float(delta)))


def calibrate_noise_multiplier(target_epsilon: float, sample_rate: float,
                               steps: int, delta: float) -> float:
    """Binary-search a noise multiplier whose planned RDP epsilon <= target.

    The calibration assumes at most ``steps`` communication rounds. If early
    stopping occurs, the actual final epsilon will be lower than the target.
    """
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")
    lo, hi = 0.05, 1.0
    eps_hi = estimate_rdp_epsilon(hi, sample_rate, steps, delta)
    if not np.isfinite(eps_hi):
        raise RuntimeError("Opacus RDP accountant is unavailable for calibration")
    while eps_hi > target_epsilon and hi < 256.0:
        hi *= 2.0
        eps_hi = estimate_rdp_epsilon(hi, sample_rate, steps, delta)
    if eps_hi > target_epsilon:
        raise RuntimeError(f"Could not calibrate DP noise for epsilon={target_epsilon}")
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        eps_mid = estimate_rdp_epsilon(mid, sample_rate, steps, delta)
        if eps_mid > target_epsilon:
            lo = mid
        else:
            hi = mid
    return float(hi)


# =============================================================================
# 13. SERVER / ROBUST AGGREGATION
# =============================================================================

class Server:
    def __init__(self, n_clients: int, cfg: Config, model_ctor, n_params: int):
        self.cfg = cfg
        self.model_ctor = model_ctor
        self.model = model_ctor().to(DEVICE)
        self.W = flatten_params(self.model).clone()
        self.V = torch.zeros_like(self.W)
        self.reputation: Dict[int, float] = {i: cfg.rep_init for i in range(n_clients)}

        # Independent state for the Reviewer-1 P4P baseline. Keeping P4P state
        # separate from RPP reputation/momentum prevents cross-method leakage.
        self.p4p_probe = torch.zeros_like(self.W)
        self.p4p_suspicion: Dict[int, float] = {i: 0.0 for i in range(n_clients)}
        self.p4p_round = 0
        self.last_agg_diagnostics: Dict[str, Any] = {}

        rng = np.random.default_rng(cfg.seed + 2027)
        k = min(cfg.trust_sketch_dim, n_params)
        self.sketch_idx = torch.tensor(
            np.sort(rng.choice(n_params, size=k, replace=False)), dtype=torch.long
        )

    def _sketch(self, g: torch.Tensor) -> torch.Tensor:
        return g[self.sketch_idx]

    def _prepare_updates(self, updates: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """Prepare client updates without giving every baseline an RPP-specific clip.

        Local-DP requires clipping by definition. In non-DP robustness experiments,
        L2 clipping is part of RPP itself; FedAvg/Median/Trimmed Mean and P4P receive
        the same raw attacked updates. This avoids accidentally strengthening the
        controlled baselines with an undeclared RPP defense component.
        """
        prepared: Dict[int, torch.Tensor] = {}
        for cid, g in updates.items():
            if self.cfg.privacy_mode == "local_dp" and self.cfg.dp_noise_multiplier > 0:
                gc = clip_vector_l2(g, self.cfg.update_clip_C)
                std = self.cfg.dp_noise_multiplier * self.cfg.update_clip_C
                gc = gc + torch.randn_like(gc) * std
            elif self.cfg.aggregator in {"rpp", "rpp_event"}:
                gc = clip_vector_l2(g, self.cfg.update_clip_C)
            else:
                gc = g.clone()
            prepared[cid] = gc
        return prepared

    @staticmethod
    def _rank01(values: np.ndarray) -> np.ndarray:
        """Deterministic rank normalization; higher input -> higher anomaly score."""
        v = np.asarray(values, dtype=np.float64)
        n = len(v)
        if n <= 1:
            return np.zeros(n, dtype=np.float64)
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        return ranks / float(n - 1)

    @staticmethod
    def _robust_standardize_matrix(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        med = np.median(X, axis=0, keepdims=True)
        mad = np.median(np.abs(X - med), axis=0, keepdims=True)
        scale = 1.4826 * mad
        scale = np.where(scale < 1e-8, 1.0, scale)
        Z = (X - med) / scale
        return np.clip(np.nan_to_num(Z, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)

    def _aggregate_p4p(self, updates: Dict[int, torch.Tensor],
                       sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        """Matched-protocol reimplementation of the publicly documented P4P pipeline.

        Reference: Khang et al., J. Network and Computer Applications 251 (2026),
        104502, DOI 10.1016/j.jnca.2026.104502. Publicly documented components are:
        a probe derived from the global historical trajectory, server-side cosine
        response, L2-norm filtering, ensemble anomaly detection, and temporal
        suspicion scoring. The reference paper does not expose an official code path
        through this script, so all instantiation choices below are explicit and saved
        in Config rather than being presented as unpublished author settings.
        """
        cids = list(updates.keys())
        stack = torch.stack([updates[c].detach().cpu() for c in cids], dim=0)
        m = len(cids)
        if m == 1:
            c = cids[0]
            return updates[c].clone(), {c: 1.0}, {c: 1.0}

        # P4P global-trajectory probe. At round 1, where no trajectory exists yet,
        # use the coordinate median only as a neutral bootstrap reference.
        probe = self.p4p_probe.detach().cpu()
        if float(probe.norm()) <= 1e-12:
            probe = torch.median(stack, dim=0).values

        center = torch.median(stack, dim=0).values
        norms = np.asarray([float(x.norm()) for x in stack], dtype=np.float64)
        med_norm = float(np.median(norms))
        mad_norm = float(np.median(np.abs(norms - med_norm)))
        norm_scale = max(1.4826 * mad_norm, 1e-8)
        norm_z = np.abs(norms - med_norm) / norm_scale
        norm_anom = np.clip(norm_z / max(self.cfg.p4p_norm_z_threshold, 1e-6), 0.0, 1.0)
        norm_flag = norm_z > self.cfg.p4p_norm_z_threshold

        probe_cos = np.asarray([cosine(x, probe) for x in stack], dtype=np.float64)
        center_cos = np.asarray([cosine(x, center) for x in stack], dtype=np.float64)
        probe_anom = np.clip((1.0 - probe_cos) / 2.0, 0.0, 1.0)
        center_anom = np.clip((1.0 - center_cos) / 2.0, 0.0, 1.0)
        center_dist = np.asarray([float((x - center).norm()) for x in stack], dtype=np.float64)
        log_norm_ratio = np.log((norms + 1e-12) / (med_norm + 1e-12))

        # Ensemble anomaly detection is intentionally server-side. The detector input
        # uses only update-derived quantities and therefore requires no extra raw data.
        features = np.column_stack([probe_cos, center_cos, log_norm_ratio, center_dist])
        Z = self._robust_standardize_matrix(features)

        if m >= 4:
            iso = IsolationForest(
                n_estimators=max(int(self.cfg.p4p_iforest_estimators), 32),
                contamination="auto",
                random_state=self.cfg.seed + 4000 + self.p4p_round,
                n_jobs=1,
            )
            iso.fit(Z)
            iso_raw = -iso.decision_function(Z)
            iso_score = self._rank01(iso_raw)

            n_neighbors = max(2, min(5, m - 1))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination="auto")
            lof.fit_predict(Z)
            lof_raw = -lof.negative_outlier_factor_
            lof_score = self._rank01(lof_raw)
        else:
            # Very small participation sets are unsafe for density estimators.
            # Fall back to deterministic robust geometric anomaly ranks.
            iso_score = self._rank01(center_dist)
            lof_score = self._rank01(norm_z)

        # Multi-metric instantaneous evidence: probe direction + L2 + two ensemble
        # detectors. Center direction contributes to the anomaly-detector features but
        # is not double-counted as a separate hard filter.
        instant = (
            0.35 * probe_anom
            + 0.25 * norm_anom
            + 0.20 * iso_score
            + 0.20 * lof_score
        )
        instant = np.clip(instant, 0.0, 1.0)

        temporal = np.zeros(m, dtype=np.float64)
        effective = np.zeros(m, dtype=np.float64)
        for i, cid in enumerate(cids):
            old = float(self.p4p_suspicion.get(cid, 0.0))
            new = (
                self.cfg.p4p_temporal_beta * old
                + (1.0 - self.cfg.p4p_temporal_beta) * float(instant[i])
            )
            new = float(np.clip(new, 0.0, 1.0))
            self.p4p_suspicion[cid] = new
            temporal[i] = new
            effective[i] = max(float(instant[i]), new)

        reject = (
            norm_flag
            | (instant >= self.cfg.p4p_instant_threshold)
            | (temporal >= self.cfg.p4p_temporal_threshold)
        )
        keep = ~reject

        # P4P assumes a benign majority. Preserve at least that much participation
        # rather than allowing an unstable detector to eliminate nearly all clients.
        min_keep = max(1, int(math.ceil(self.cfg.p4p_min_keep_fraction * m)))
        if int(keep.sum()) < min_keep:
            order = np.argsort(effective, kind="mergesort")
            keep = np.zeros(m, dtype=bool)
            keep[order[:min_keep]] = True

        kept_cids = [c for c, k in zip(cids, keep.tolist()) if k]
        ns = np.asarray([max(sample_counts[c], 1) for c in kept_cids], dtype=np.float64)
        w_keep = ns / ns.sum()
        delta = torch.zeros_like(self.W)
        weights: Dict[int, float] = {c: 0.0 for c in cids}
        for cid, wi in zip(kept_cids, w_keep.tolist()):
            delta += updates[cid] * float(wi)
            weights[cid] = float(wi)

        trust = {
            cid: float(np.clip(1.0 - effective[i], 0.0, 1.0))
            for i, cid in enumerate(cids)
        }
        self.last_agg_diagnostics = {
            "P4PKept": len(kept_cids),
            "P4PRejected": int(m - len(kept_cids)),
            "P4PMeanInstantSuspicion": float(np.mean(instant)),
            "P4PMeanTemporalSuspicion": float(np.mean(temporal)),
            "P4PMaxNormZ": float(np.max(norm_z)),
            "P4PMeanProbeCosine": float(np.mean(probe_cos)),
        }
        return self._apply_central_dp(delta, weights), trust, weights


    @staticmethod
    def _r3_robust_scale(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Coordinate median and 1.4826*MAD plus documented positive floor.

        Manuscript defines rho=1.4826*MAD+phi but not numeric phi. V6 uses
        phi=max(1e-10, .01 * median(nonzero scale)) for transparent replication.
        This convention MUST be matched to the source of originally reported
        figures before asserting numerical reproduction.
        """
        # Standard median for an even number of clients averages both middle
        # order statistics, unlike torch.median (which chooses the lower one).
        def exact_median(x: torch.Tensor) -> torch.Tensor:
            ordered = torch.sort(x, dim=0).values
            n = ordered.shape[0]
            return ordered[n // 2] if n % 2 else 0.5 * (ordered[n//2 - 1] + ordered[n//2])
        med = exact_median(matrix)
        mad = exact_median((matrix - med).abs())
        scale0 = 1.4826 * mad
        nonzero = scale0[scale0 > 1e-12]
        typical = float(nonzero.median()) if nonzero.numel() else 0.0
        phi = max(R3_MAD_FLOOR_ABSOLUTE, R3_MAD_FLOOR_RELATIVE * typical)
        return med, scale0 + phi

    @staticmethod
    def _r3_collusion_guard(z: torch.Tensor) -> Tuple[List[int], float, float]:
        """Deterministic farthest-seed two-group, minority compactness guard.

        Returns selected-row indices only; never reads attack labels.
        Deterministically breaks assignment ties in favor of first seed.
        """
        m, k = z.shape
        if m < 3:
            return [], float("nan"), float("nan")
        d = torch.cdist(z, z, p=2) / math.sqrt(max(k, 1))
        np_d = d.detach().cpu().numpy()
        np.fill_diagonal(np_d, -1.0)
        ia, ib = np.unravel_index(np.argmax(np_d), np_d.shape)
        # Seed owners remain in their own groups even in coincident data.
        left, right = [int(ia)], [int(ib)]
        for i in range(m):
            if i in (ia, ib):
                continue
            (left if np_d[i, ia] <= np_d[i, ib] else right).append(i)
        small, large = (left, right) if len(left) <= len(right) else (right, left)
        if len(small) > math.ceil(0.40 * m):
            return [], float("nan"), float("nan")
        def pair_med(group: List[int]) -> float:
            if len(group) < 2:
                return 0.0
            pairs = [np_d[group[a], group[b]] for a in range(len(group))
                     for b in range(a + 1, len(group))]
            return float(np.median(pairs))
        ds, dl = pair_med(small), pair_med(large)
        cross = float(np.median([np_d[i,j] for i in small for j in large]))
        denom = dl + 1e-10
        kappa, chi = ds / denom, cross / denom
        if len(small) == 1:
            flagged = chi >= 1.25
        else:
            flagged = (kappa <= 0.60) and (chi >= 0.90)
        return (list(small) if flagged else []), kappa, chi

    def _aggregate_r3_event(self, updates: Dict[int, torch.Tensor],
                            sample_counts: Dict[int, int]) -> Tuple[
                                torch.Tensor, Dict[int, float], Dict[int, float]]:
        """Manuscript-derived, reviewable event coupling experiment.

        Variants:
          simple_temporal = robust center/deviation/momentum, symmetric EMA;
                            robust winsorization + fixed median anchor
          uncoupled       = five trust cues incl. collusion score but no 0.05
                            event gate, no event-reputation penalty, fixed anchor
          partial         = instant gate + reputation event penalty, fixed anchor
          full            = instant gate + reputation event penalty + adaptive anchor
        All variants share L2 clipped input updates, same full-coordinate
        winsorization, sample-support cap, and basic geometric calculations.
        Distinction between `uncoupled` and `full` is intentionally explicit.
        """
        variant = self.cfg.r3_variant
        if variant not in {"simple_temporal", "uncoupled", "partial", "full"}:
            raise ValueError(f"Unknown R3 coupling variant: {variant}")
        cids = list(updates)
        stack = torch.stack([updates[c].detach().cpu() for c in cids], dim=0)
        m = len(cids)
        sketches = stack[:, self.sketch_idx]
        center, scale = self._r3_robust_scale(sketches)
        zs = torch.clamp((sketches - center) / scale, min=-20.0, max=20.0)
        deviation = torch.median(
            torch.clamp(torch.abs(sketches - center) / scale, max=20.0), dim=1
        ).values
        dev_score = torch.exp(-torch.clamp(deviation - 0.70, min=0.0) / 0.75)
        center_cos = torch.tensor([
            max(0.0, cosine(sketches[i], center)) for i in range(m)
        ], dtype=torch.float32)
        hist = self._sketch(self.V)
        # A zero momentum reference has undefined cosine; use neutral score 1.
        hist_cos = torch.tensor([
            max(0.0, cosine(sketches[i], hist)) if float(hist.norm()) > 1e-12
            else 1.0 for i in range(m)
        ], dtype=torch.float32)
        peer = []
        for i in range(m):
            sims = [cosine(sketches[i], sketches[j]) for j in range(m) if j != i]
            peer.append((1.0 + float(np.median(sims))) / 2.0 if sims else 1.0)
        peer_score = torch.tensor(peer, dtype=torch.float32).clamp(0.0, 1.0)

        flagged_rows, kappa, chi = self._r3_collusion_guard(zs)
        flagged = set(flagged_rows)
        flagged_frac = len(flagged_rows) / max(m, 1)
        coll_score = torch.ones(m, dtype=torch.float32)
        for ix in flagged:
            coll_score[ix] = 0.05

        if variant == "simple_temporal":
            # Explicit simpler comparator; no collusion-derived score/gate.
            immediate = 0.40 * center_cos + 0.40 * dev_score + 0.20 * hist_cos
        else:
            immediate = (0.10 * center_cos + 0.10 * hist_cos
                         + 0.35 * dev_score + 0.10 * peer_score
                         + 0.35 * coll_score)
        if variant in {"partial", "full"}:
            for ix in flagged:
                immediate[ix] *= 0.05  # manuscript Eq: multiplicative gate
        immediate = immediate.clamp(0.0, 1.0)

        trust: Dict[int, float] = {}
        for i, cid in enumerate(cids):
            score = float(immediate[i])
            old = float(self.reputation[cid])
            if variant == "simple_temporal":
                new = 0.80 * old + 0.20 * score
            else:
                # Faster decrease (new information 40%), slower recovery (5%).
                alpha = 0.60 if score < old else 0.95
                new = alpha * old + (1.0 - alpha) * score
            if variant in {"partial", "full"} and i in flagged:
                new *= 0.70
            self.reputation[cid] = float(np.clip(new, 0.005, 1.0))
            trust[cid] = score

        med_n = float(np.median([sample_counts[c] for c in cids]))
        raw_w = np.asarray([
            math.sqrt(min(max(sample_counts[cid], 1), 1.5 * med_n))
            * self.reputation[cid] ** 2
            * max(trust[cid], 1e-4) ** 2.5
            for cid in cids
        ], dtype=np.float64)
        if not np.isfinite(raw_w).all() or raw_w.sum() <= 0:
            raise FloatingPointError("Invalid R3 aggregation weights")
        raw_w = raw_w / raw_w.sum()
        weights = {cid: float(w) for cid, w in zip(cids, raw_w)}

        full_med, full_scale = self._r3_robust_scale(stack)
        winsor = torch.maximum(torch.minimum(stack, full_med + 2.5 * full_scale),
                               full_med - 2.5 * full_scale)
        robust = (winsor * torch.from_numpy(raw_w.astype(np.float32))[:, None]).sum(0)
        anchor = min(0.55, max(0.0, 0.10 + flagged_frac)) if variant == "full" else 0.10
        delta = (1.0 - anchor) * robust + anchor * full_med
        self.last_agg_diagnostics = {
            "R3Variant": variant,
            "R3FlaggedCount": int(len(flagged)),
            "R3FlaggedFraction": float(flagged_frac),
            "R3FlaggedClientIDs": json.dumps([int(cids[i]) for i in sorted(flagged)]),
            "R3Kappa": float(kappa),
            "R3Chi": float(chi),
            "R3MedianAnchor": float(anchor),
            "R3MeanTrust": float(np.mean(list(trust.values()))),
            "R3MeanReputation": float(np.mean([self.reputation[c] for c in cids])),
        }
        return delta, trust, weights

    def aggregate(self, updates: Dict[int, torch.Tensor], sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        updates = self._prepare_updates(updates)
        cids = list(updates.keys())
        if not cids:
            raise ValueError("No client updates to aggregate")

        self.last_agg_diagnostics = {}
        if self.cfg.aggregator == "p4p":
            return self._aggregate_p4p(updates, sample_counts)
        if self.cfg.aggregator == "rpp_event":
            return self._aggregate_r3_event(updates, sample_counts)

        if self.cfg.aggregator == "fedavg":
            ns = np.array([sample_counts[c] for c in cids], dtype=np.float64)
            w = ns / ns.sum()
            delta = sum(updates[c] * float(wi) for c, wi in zip(cids, w))
            trust = {c: 1.0 for c in cids}
            weights = {c: float(wi) for c, wi in zip(cids, w)}
            return self._apply_central_dp(delta, weights), trust, weights

        stack = torch.stack([updates[c] for c in cids], dim=0)

        if self.cfg.aggregator == "median":
            delta = torch.median(stack, dim=0).values
            trust = {c: 1.0 for c in cids}
            weights = {c: 1.0 / len(cids) for c in cids}
            return self._apply_central_dp(delta, weights), trust, weights

        if self.cfg.aggregator == "trimmed_mean":
            m = stack.shape[0]
            trim = int(math.floor(self.cfg.trim_ratio * m))
            sorted_vals, _ = torch.sort(stack, dim=0)
            if 2 * trim < m:
                core = sorted_vals[trim:m - trim]
            else:
                core = sorted_vals
            delta = core.mean(dim=0)
            trust = {c: 1.0 for c in cids}
            weights = {c: 1.0 / len(cids) for c in cids}
            return self._apply_central_dp(delta, weights), trust, weights

        if self.cfg.aggregator != "rpp":
            raise ValueError(f"Unknown aggregator: {self.cfg.aggregator}")

        # ---------------- RPP reputation-aware robust aggregation ----------------
        # The four trust components are individually switchable for the planned
        # ablation study. Disabled components are not silently replaced by an
        # equivalent signal; their coefficient is removed and the remaining
        # active coefficients are renormalized.
        sketches = torch.stack([self._sketch(updates[c]) for c in cids], dim=0)
        center = torch.median(sketches, dim=0).values
        norms = torch.tensor([float(s.norm()) for s in sketches], dtype=torch.float32)
        med_norm = float(torch.median(norms).item()) + 1e-12

        hist = self._sketch(self.V)
        hist_valid = float(hist.norm()) > 1e-12

        coeff_center = 0.50 if self.cfg.use_robust_center else 0.0
        coeff_hist = 0.35 if self.cfg.use_momentum else 0.0
        coeff_norm = 0.15 if self.cfg.use_norm_consistency else 0.0
        coeff_sum = coeff_center + coeff_hist + coeff_norm
        if coeff_sum <= 0:
            # If all instantaneous trust signals are disabled, use neutral trust
            # and let sample count / optional reputation determine aggregation.
            coeff_sum = 1.0

        trust_values: Dict[int, float] = {}
        current_scores: List[float] = []
        for i, cid in enumerate(cids):
            s = sketches[i]
            cos_center = max(0.0, cosine(s, center)) if self.cfg.use_robust_center else 0.0
            if self.cfg.use_momentum:
                cos_hist = max(0.0, cosine(s, hist)) if hist_valid else (
                    max(0.0, cosine(s, center)) if self.cfg.use_robust_center else 1.0
                )
            else:
                cos_hist = 0.0
            if self.cfg.use_norm_consistency:
                nr = float(norms[i].item()) / med_norm
                norm_score = math.exp(-abs(math.log(max(nr, 1e-8))))
            else:
                norm_score = 0.0

            if coeff_center + coeff_hist + coeff_norm > 0:
                score = (
                    coeff_center * cos_center
                    + coeff_hist * cos_hist
                    + coeff_norm * norm_score
                ) / (coeff_center + coeff_hist + coeff_norm)
            else:
                score = 1.0
            score = float(np.clip(score, 0.0, 1.0))
            current_scores.append(score)
            trust_values[cid] = score

        # Reputation is a true ablation component. When disabled, historical
        # reputation is fixed at 1 and cannot influence the final weights.
        floor = float(np.quantile(current_scores, self.cfg.trust_floor_quantile))
        for cid in cids:
            score = trust_values[cid]
            if self.cfg.use_reputation:
                self.reputation[cid] = (
                    self.cfg.alpha_rep * self.reputation[cid]
                    + (1.0 - self.cfg.alpha_rep) * score
                )
                if score < floor:
                    self.reputation[cid] *= 0.85
                self.reputation[cid] = float(np.clip(self.reputation[cid], 0.01, 1.0))
            else:
                self.reputation[cid] = 1.0

        raw_w = []
        for cid in cids:
            n_factor = math.sqrt(max(sample_counts[cid], 1))
            r_factor = (
                self.reputation[cid] ** self.cfg.reputation_power
                if self.cfg.use_reputation else 1.0
            )
            # Instantaneous trust remains active in w/o-reputation; this cleanly
            # isolates long-term memory from current-round screening.
            t_factor = max(trust_values[cid], 0.05)
            raw_w.append(n_factor * r_factor * t_factor)
        raw_w = np.asarray(raw_w, dtype=np.float64)
        if not np.isfinite(raw_w).all() or raw_w.sum() <= 0:
            raw_w = np.ones(len(cids), dtype=np.float64)
        raw_w /= raw_w.sum()

        delta = torch.zeros_like(self.W)
        weights: Dict[int, float] = {}
        for cid, wi in zip(cids, raw_w.tolist()):
            delta += updates[cid] * float(wi)
            weights[cid] = float(wi)

        delta = self._apply_central_dp(delta, weights)
        return delta, trust_values, weights

    def _apply_central_dp(self, delta: torch.Tensor, weights: Dict[int, float]) -> torch.Tensor:
        if self.cfg.privacy_mode != "central_dp" or self.cfg.dp_noise_multiplier <= 0:
            return delta
        # Conservative sensitivity proxy for a clipped weighted client update.
        max_w = max(weights.values()) if weights else 1.0
        sensitivity = self.cfg.update_clip_C * max_w
        std = self.cfg.dp_noise_multiplier * sensitivity
        return delta + torch.randn_like(delta) * std

    def apply_update(self, delta: torch.Tensor) -> None:
        # CORRECT sign: client delta = W_local - W_global.
        self.W = self.W + self.cfg.lr_global * delta
        self.V = self.cfg.beta_momentum * self.V + (1.0 - self.cfg.beta_momentum) * delta
        if self.cfg.aggregator == "p4p":
            self.p4p_probe = (
                self.cfg.p4p_probe_beta * self.p4p_probe
                + (1.0 - self.cfg.p4p_probe_beta) * delta.detach().cpu()
            )
            self.p4p_round += 1
        set_params_from_vector(self.model, self.W)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        out = np.empty(len(X), dtype=np.float32)
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size]).to(DEVICE)
            logits, _ = self.model(xb)
            out[s:s + batch_size] = torch.sigmoid(logits).detach().cpu().numpy()
        return out


# =============================================================================
# 14. METRICS / VALIDATION-ONLY THRESHOLD SELECTION
# =============================================================================

def best_threshold_by_f1(y_true: np.ndarray, prob: np.ndarray) -> float:
    p, r, th = precision_recall_curve(y_true, prob)
    if len(th) == 0:
        return 0.5
    f1 = 2.0 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    if not np.isfinite(f1).any():
        return 0.5
    idx = int(np.nanargmax(f1))
    return float(np.clip(th[idx], 0.001, 0.999))


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    pred = (prob >= thr).astype(np.int64)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1v = f1_score(y_true, pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, pred)
    spec = tn / (tn + fp + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    fnr = fn / (fn + tp + 1e-12)

    try:
        auc = roc_auc_score(y_true, prob)
    except Exception:
        auc = float("nan")
    try:
        ap = average_precision_score(y_true, prob)
    except Exception:
        ap = float("nan")

    return {
        "Accuracy": float(acc),
        "BalancedAccuracy": float(bacc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1v),
        "MCC": float(mcc),
        "ROC_AUC": float(auc),
        "PR_AUC": float(ap),
        "Specificity": float(spec),
        "FPR": float(fpr),
        "FNR": float(fnr),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


# =============================================================================
# 15. DATA PREPARATION PER DATASET
# =============================================================================

@dataclass
class PreparedData:
    dataset: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    n_raw_features: int
    n_preprocessed_features: int
    n_selected_features: int
    label_col: str
    raw_rows: int
    audit_info: Optional[dict] = None
    split_indices: Optional[dict] = None


def dataset_source_signature(path: str) -> str:
    files = _tabular_files(path)
    payload = []
    for f in files:
        try:
            payload.append((os.path.relpath(f, path) if os.path.isdir(path) else os.path.basename(f),
                            os.path.getsize(f), int(os.path.getmtime(f))))
        except OSError:
            pass
    return stable_hash(payload)


def prepared_cache_key(dataset_name: str, path: str, cfg: Config) -> str:
    return stable_hash({
        "dataset": dataset_name,
        "source_signature": dataset_source_signature(path),
        "seed": cfg.seed,
        "max_rows": MAX_ROWS_PER_DATASET,
        "max_features": cfg.max_model_features,
        "preprocess_version": R3_AUDIT_VERSION if RUN_REVIEWER3_MECHANISM_ONLY else 4,
    })


def save_prepared_cache(data: PreparedData, cache_file: str, meta_file: str) -> None:
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    local_tmp = os.path.join(tempfile.gettempdir(), f"rpp_{data.dataset}_{os.getpid()}.npz")
    np.savez_compressed(
        local_tmp,
        X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
    )
    if data.split_indices is not None:
        np.savez_compressed(os.path.join(Path(cache_file).parent, "split_indices.npz"),
                            **data.split_indices)
    drive_tmp = cache_file + ".partial"
    shutil.copy2(local_tmp, drive_tmp)
    os.replace(drive_tmp, cache_file)
    try:
        os.remove(local_tmp)
    except OSError:
        pass
    atomic_json_dump({
        "dataset": data.dataset,
        "n_raw_features": data.n_raw_features,
        "n_preprocessed_features": data.n_preprocessed_features,
        "n_selected_features": data.n_selected_features,
        "label_col": data.label_col,
        "raw_rows": data.raw_rows,
        "cache_file": os.path.basename(cache_file),
        "status": "ready",
        "audit_version": R3_AUDIT_VERSION,
        "audit_info": data.audit_info or {},
        "split_indices_file": "split_indices.npz" if data.split_indices else None,
    }, meta_file)


def load_prepared_cache(cache_file: str, meta_file: str) -> Optional[PreparedData]:
    if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            m = json.load(f)
        if RUN_REVIEWER3_MECHANISM_ONLY and (
            m.get("audit_version") != R3_AUDIT_VERSION or
            not (Path(cache_file).parent / "split_indices.npz").is_file()
        ):
            print("[AUDIT] Cache lacks V6 provenance, rebuilding reproducibly.")
            return None
        z = np.load(cache_file, allow_pickle=False)
        return PreparedData(
            dataset=m["dataset"],
            X_train=z["X_train"].astype(np.float32, copy=False),
            y_train=z["y_train"].astype(np.int64, copy=False),
            X_val=z["X_val"].astype(np.float32, copy=False),
            y_val=z["y_val"].astype(np.int64, copy=False),
            X_test=z["X_test"].astype(np.float32, copy=False),
            y_test=z["y_test"].astype(np.int64, copy=False),
            n_raw_features=int(m["n_raw_features"]),
            n_preprocessed_features=int(m["n_preprocessed_features"]),
            n_selected_features=int(m["n_selected_features"]),
            label_col=str(m["label_col"]),
            raw_rows=int(m["raw_rows"]),
            audit_info=m.get("audit_info"),
            split_indices=None,
        )
    except Exception as e:
        print(f"[WARN] Prepared cache invalid; rebuilding: {e}")
        return None


def prepare_dataset(dataset_name: str, path: str, cfg: Config) -> PreparedData:
    spec = DATASET_SPECS[dataset_name]
    print_header(f"PREPARING DATASET: {dataset_name}")
    print(f"Path: {path}")

    df = load_dataset_path(path, MAX_ROWS_PER_DATASET, cfg.seed)
    y, label_col = make_binary_labels(df, spec)
    print(f"Rows loaded: {len(df):,} | attack rate: {y.mean():.2%} | label column: {label_col}")

    # IMPORTANT leakage audit: first remove all known label/ID leakage columns,
    # then deduplicate using ONLY the retained predictive features + target.
    # This prevents rows that differ only by IDs/timestamps from being split
    # across train/validation/test and inflating generalization metrics.
    feature_df = drop_known_leakage(df, spec, label_col).reset_index(drop=True)
    n_raw_features = feature_df.shape[1]
    tmp = feature_df.copy()
    tmp["__RPP_LABEL__"] = y
    before = len(tmp)
    tmp = tmp.drop_duplicates().reset_index(drop=True)
    y = tmp.pop("__RPP_LABEL__").to_numpy(dtype=np.int64)
    feature_df = tmp
    removed = before - len(feature_df)
    print(
        f"Leakage-safe dedup: retained={len(feature_df):,} | removed={removed:,} "
        f"({removed / max(before, 1):.2%})"
    )

    tr_idx, va_idx, te_idx = stratified_train_val_test_indices(y, cfg.seed)
    df_tr = feature_df.iloc[tr_idx].reset_index(drop=True)
    df_va = feature_df.iloc[va_idx].reset_index(drop=True)
    df_te = feature_df.iloc[te_idx].reset_index(drop=True)
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    prep = LeakageSafePreprocessor(onehot_max_unique=20)
    X_tr = prep.fit_transform(df_tr)
    X_va = prep.transform(df_va)
    X_te = prep.transform(df_te)
    n_pre = X_tr.shape[1]

    selector = TrainOnlyFeatureSelector(max_features=cfg.max_model_features)
    selector.fit(X_tr, y_tr)
    X_tr = selector.transform(X_tr)
    X_va = selector.transform(X_va)
    X_te = selector.transform(X_te)

    # Clip standardized extreme values; this stabilizes local training and
    # prevents a few large encoded values from dominating client updates.
    X_tr = np.clip(X_tr, -8.0, 8.0).astype(np.float32)
    X_va = np.clip(X_va, -8.0, 8.0).astype(np.float32)
    X_te = np.clip(X_te, -8.0, 8.0).astype(np.float32)

    selected_idx = selector.indices_.tolist() if selector.indices_ is not None else []
    selected_names = [prep.feature_names_[i] for i in selected_idx]
    audit_info = {
        "dataset": dataset_name,
        "source_signature": dataset_source_signature(path),
        "raw_files": [
            {"relative_path": os.path.relpath(f, path), "size_bytes": os.path.getsize(f)}
            for f in _tabular_files(path)
        ],
        "sampling_seed": cfg.seed,
        "max_rows": MAX_ROWS_PER_DATASET,
        "deduplicated_rows": int(len(feature_df)),
        "duplicates_removed": int(removed),
        "split_reference": "indices within deterministically sampled+deduplicated dataset",
        "split_lengths": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
        "split_sha256": {
            name: hashlib.sha256(arr.astype(np.int64).tobytes()).hexdigest()
            for name, arr in (("train", tr_idx), ("val", va_idx), ("test", te_idx))
        },
        "feature_selection_indices": selected_idx,
        "selected_feature_names": selected_names,
        "removed_columns": sorted(set(df.columns) - set(feature_df.columns) - {label_col}),
        "arrays_sha256": {
            name: hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()
            for name, arr in (("X_train", X_tr), ("y_train", y_tr),
                              ("X_val", X_va), ("y_val", y_va),
                              ("X_test", X_te), ("y_test", y_te))
        },
    }
    print(
        f"Split: train={len(X_tr):,}, val={len(X_va):,}, test={len(X_te):,} | "
        f"features raw={n_raw_features}, preprocessed={n_pre}, selected={X_tr.shape[1]}"
    )

    return PreparedData(
        dataset=dataset_name,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        n_raw_features=n_raw_features,
        n_preprocessed_features=n_pre,
        n_selected_features=X_tr.shape[1],
        label_col=label_col,
        raw_rows=len(feature_df),
        audit_info=audit_info,
        split_indices={
            "train_idx": tr_idx.astype(np.int64),
            "val_idx": va_idx.astype(np.int64),
            "test_idx": te_idx.astype(np.int64),
        },
    )


def load_or_prepare_dataset(dataset_name: str, path: str, cfg: Config) -> PreparedData:
    key = prepared_cache_key(dataset_name, path, cfg)
    cache_dir = os.path.join(CACHE_ROOT, dataset_name, key)
    cache_file = os.path.join(cache_dir, "prepared_arrays.npz")
    meta_file = os.path.join(cache_dir, "prepared_meta.json")

    if CACHE_PREPARED_DATA:
        cached = load_prepared_cache(cache_file, meta_file)
        if cached is not None:
            print_header(f"LOADING PREPARED CACHE: {dataset_name}")
            print(f"Cache: {cache_file}")
            print(
                f"train={len(cached.X_train):,}, val={len(cached.X_val):,}, "
                f"test={len(cached.X_test):,}, features={cached.n_selected_features}"
            )
            return cached

    data = prepare_dataset(dataset_name, path, cfg)
    if CACHE_PREPARED_DATA:
        print(f"[CACHE] Saving prepared {dataset_name} arrays to Drive ...")
        save_prepared_cache(data, cache_file, meta_file)
        print(f"[CACHE] Ready: {cache_file}")
    return data


# =============================================================================
# 16. ONE FEDERATED EXPERIMENT
# =============================================================================

def experiment_group_from_name(name: str) -> str:
    if name.startswith("r3_"):
        return "reviewer3_mechanism"
    if name.startswith("abl_"):
        return "ablation"
    if name.startswith("reviewer1_"):
        return "reviewer1_matched"
    if name.startswith(("fedavg_", "fedprox_", "median_", "trimmed_", "p4p_")):
        return "baseline"
    if name.startswith("attack_") or name.startswith("ratio_"):
        return "robustness"
    if name.startswith("noniid_"):
        return "non_iid"
    if name.startswith("privacy_"):
        return "privacy"
    if name.startswith("proposed_"):
        return "main"
    return "other"


def build_model_ctor(n_features: int, cfg: Config):
    def ctor():
        return HybridTabularTransformer(
            n_features=n_features,
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            num_layers=cfg.num_layers,
            dim_ff=cfg.dim_ff,
            mlp_hidden=cfg.mlp_hidden,
            dropout=cfg.dropout,
            use_transformer=cfg.use_transformer,
        )
    return ctor


def experiment_fingerprint(data: PreparedData, cfg: Config, attacked: bool, scenario_name: str) -> str:
    return stable_hash({
        "dataset": data.dataset,
        "scenario": scenario_name,
        "attacked": attacked,
        "config": asdict(cfg),
        "selected_features": data.n_selected_features,
        "rows": data.raw_rows,
        "data_sha256": (data.audit_info or {}).get("arrays_sha256", {}),
        "checkpoint_version": 6 if cfg.aggregator == "rpp_event" else 5,
    })


def capture_rng_state(rng_py: random.Random, torch_gen: torch.Generator) -> dict:
    state = {
        "python_global": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "rng_py": rng_py.getstate(),
        "torch_gen": torch_gen.get_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[dict], rng_py: random.Random, torch_gen: torch.Generator) -> None:
    if not state:
        return
    try:
        random.setstate(state["python_global"])
        np.random.set_state(state["numpy_global"])
        torch.set_rng_state(state["torch_cpu"])
        rng_py.setstate(state["rng_py"])
        torch_gen.set_state(state["torch_gen"])
        if torch.cuda.is_available() and "torch_cuda" in state:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception as e:
        print(f"[WARN] Could not fully restore RNG state: {e}")


def save_training_checkpoint(path: str, *, fingerprint: str, server: Server,
                             cfg: Config, round_idx: int, history: List[Dict[str, Any]],
                             best_val_f1: float, best_round: int, best_threshold: float,
                             best_W: Optional[torch.Tensor], patience: int,
                             malicious: set, rng_py: random.Random,
                             torch_gen: torch.Generator,
                             accountant: PrivacyAccountant) -> None:
    payload = {
        "checkpoint_version": 6 if cfg.aggregator == "rpp_event" else 5,
        "fingerprint": fingerprint,
        "W": server.W,
        "V": server.V,
        "reputation": server.reputation,
        "p4p_probe": server.p4p_probe,
        "p4p_suspicion": server.p4p_suspicion,
        "p4p_round": server.p4p_round,
        "round": round_idx,
        "history": history,
        "best_val_f1": best_val_f1,
        "best_round": best_round,
        "best_threshold": best_threshold,
        "best_W": best_W,
        "patience": patience,
        "malicious": sorted(malicious),
        "config": asdict(cfg),
        "rng_state": capture_rng_state(rng_py, torch_gen),
        "accountant_state": accountant.state_dict(),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_torch_save(payload, path)


def load_training_checkpoint(path: str, fingerprint: str) -> Optional[dict]:
    if not (AUTO_RESUME and os.path.exists(path)):
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Could not read checkpoint {path}: {e}")
        return None
    if ckpt.get("fingerprint") != fingerprint:
        print("[RESUME] Existing checkpoint belongs to a different configuration; ignoring it.")
        return None
    return ckpt


def load_completed_result(run_dir: str, fingerprint: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(run_dir, "final_result.json")
    if FORCE_RETRAIN_COMPLETED or not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            result = json.load(f)
        if result.get("RunFingerprint") == fingerprint:
            print(f"[SKIP COMPLETED] {result.get('Dataset')} / {result.get('Scenario')}")
            return result
    except Exception:
        pass
    return None


def run_federated_experiment(data: PreparedData, cfg: Config,
                             attacked: bool,
                             scenario_name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    set_seed(cfg.seed)
    model_ctor = build_model_ctor(data.n_selected_features, cfg)
    temp_model = model_ctor()
    n_params = sum(p.numel() for p in temp_model.parameters())
    del temp_model

    partitions = dirichlet_label_partition(
        data.y_train,
        n_clients=cfg.n_clients,
        alpha=cfg.dirichlet_alpha,
        seed=cfg.seed,
        min_samples=cfg.min_client_samples,
        strict=cfg.r3_strict_noniid,
    )

    clients: Dict[int, Client] = {}
    for cid, idx in partitions.items():
        clients[cid] = Client(
            cid, data.X_train[idx], data.y_train[idx], cfg, model_ctor
        )

    server = Server(cfg.n_clients, cfg, model_ctor, n_params)
    accountant = PrivacyAccountant(cfg)

    rng_py = random.Random(cfg.seed + 9001)
    torch_gen = torch.Generator().manual_seed(cfg.seed + 8128)

    all_clients = list(range(cfg.n_clients))
    mal_count = int(round(cfg.frac_malicious * cfg.n_clients)) if attacked else 0
    mal_count = min(max(mal_count, 0), cfg.n_clients - 1 if cfg.n_clients > 1 else 0)
    malicious = set(rng_py.sample(all_clients, k=mal_count)) if mal_count > 0 else set()

    per_round = max(1, int(round(cfg.clients_per_round_frac * cfg.n_clients)))
    per_round = min(per_round, cfg.n_clients)

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_round = 0
    best_threshold = 0.5
    best_W: Optional[torch.Tensor] = None
    patience = 0
    start_round = 1

    run_dir = os.path.join(RESULT_ROOT, data.dataset, scenario_name)
    os.makedirs(run_dir, exist_ok=True)
    fingerprint = experiment_fingerprint(data, cfg, attacked, scenario_name)
    last_ckpt_path = os.path.join(run_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(run_dir, "best_checkpoint.pt")
    history_path = os.path.join(run_dir, "round_history.csv")

    completed = load_completed_result(run_dir, fingerprint)
    if completed is not None:
        hist = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
        return completed, hist

    ckpt = load_training_checkpoint(last_ckpt_path, fingerprint)
    if ckpt is not None:
        server.W = ckpt["W"].clone()
        server.V = ckpt["V"].clone()
        server.reputation = {int(k): float(v) for k, v in ckpt["reputation"].items()}
        pp = ckpt.get("p4p_probe")
        if isinstance(pp, torch.Tensor):
            server.p4p_probe = pp.clone()
        server.p4p_suspicion = {
            int(k): float(v) for k, v in ckpt.get("p4p_suspicion", server.p4p_suspicion).items()
        }
        server.p4p_round = int(ckpt.get("p4p_round", 0))
        set_params_from_vector(server.model, server.W)
        history = list(ckpt.get("history", []))
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0))
        best_round = int(ckpt.get("best_round", 0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        bw = ckpt.get("best_W")
        best_W = bw.clone() if isinstance(bw, torch.Tensor) else None
        patience = int(ckpt.get("patience", 0))
        malicious = set(int(x) for x in ckpt.get("malicious", sorted(malicious)))
        accountant.load_state_dict(ckpt.get("accountant_state"))
        restore_rng_state(ckpt.get("rng_state"), rng_py, torch_gen)
        start_round = int(ckpt.get("round", 0)) + 1
        print(f"[RESUME] {data.dataset}/{scenario_name}: continuing from round {start_round}.")

    print_header(
        f"{data.dataset} | {scenario_name} | agg={cfg.aggregator} | "
        f"alpha={cfg.dirichlet_alpha} | privacy={cfg.privacy_mode}"
    )
    print(f"Model params: {n_params:,} | malicious clients: {sorted(malicious)}")
    print(f"Checkpoint: {last_ckpt_path}")

    already_early_stopped = (
        cfg.enable_early_stopping
        and (start_round - 1) >= cfg.min_rounds_before_stop
        and patience >= cfg.early_stop_patience
    )

    if not already_early_stopped:
        for rnd in range(start_round, cfg.n_rounds + 1):
            t0 = time.perf_counter()
            selected = rng_py.sample(all_clients, k=per_round)

            raw_deltas: Dict[int, torch.Tensor] = {}
            sample_counts: Dict[int, int] = {}
            client_times: List[float] = []

            for cid in selected:
                d, sec = clients[cid].local_train_and_delta(server.W)
                raw_deltas[cid] = d
                sample_counts[cid] = clients[cid].n_samples
                client_times.append(sec)

            benign_selected = [cid for cid in selected if cid not in malicious]
            benign_stack = (
                torch.stack([raw_deltas[c] for c in benign_selected], dim=0)
                if len(benign_selected) >= 2 else None
            )

            attacked_updates: Dict[int, torch.Tensor] = {}
            for cid in selected:
                d = raw_deltas[cid]
                if attacked and cid in malicious:
                    d = apply_poisoning_attack(
                        d, cfg.attack_type, benign_stack, cfg, torch_gen
                    )
                attacked_updates[cid] = d

            delta_global, trust, weights = server.aggregate(attacked_updates, sample_counts)
            server.apply_update(delta_global)

            val_prob = server.predict_proba(data.X_val)
            threshold = best_threshold_by_f1(data.y_val, val_prob)
            val_metrics = compute_metrics(data.y_val, val_prob, threshold)
            eps = accountant.step(per_round / cfg.n_clients)

            rep_m = [server.reputation[c] for c in selected if c in malicious]
            rep_b = [server.reputation[c] for c in selected if c not in malicious]
            rep_sep = (
                float(np.mean(rep_b) - np.mean(rep_m)) if rep_m and rep_b else float("nan")
            )
            trust_m = [trust[c] for c in selected if c in malicious]
            trust_b = [trust[c] for c in selected if c not in malicious]
            trust_sep = (
                float(np.mean(trust_b) - np.mean(trust_m)) if trust_m and trust_b else float("nan")
            )
            # Security diagnostics are evaluated only because the simulator knows
            # which clients were injected as malicious. Higher AUC means the
            # trust score separates malicious from benign updates more reliably.
            if attacked and trust_m and trust_b:
                client_attack_auc = float(roc_auc_score(
                    [1 if c in malicious else 0 for c in selected],
                    [1.0 - float(trust[c]) for c in selected],
                ))
                malicious_weight_share = float(sum(
                    weights[c] for c in selected if c in malicious
                ))
            else:
                client_attack_auc = float("nan")
                malicious_weight_share = 0.0

            round_sec = time.perf_counter() - t0
            row = {
                "Dataset": data.dataset,
                "Scenario": scenario_name,
                "Round": rnd,
                "ValThreshold": threshold,
                **{f"Val_{k}": v for k, v in val_metrics.items()},
                "Epsilon": eps,
                "RoundTime_s": round_sec,
                "AvgClientTime_s": float(np.mean(client_times)),
                "RepSeparation": rep_sep,
                "TrustSeparation": trust_sep,
                "MeanTrust": float(np.mean(list(trust.values()))),
                "MaxAggregationWeight": float(max(weights.values())),
                "ClientAttackDetectAUC": client_attack_auc,
                "MaliciousWeightShare": malicious_weight_share,
                "SelectedClients": json.dumps(sorted(int(c) for c in selected)),
                "SelectedMaliciousClients": json.dumps(sorted(int(c) for c in selected if c in malicious)),
                **server.last_agg_diagnostics,
            }
            history.append(row)

            improved = val_metrics["F1"] > best_val_f1 + 1e-5
            if improved:
                best_val_f1 = val_metrics["F1"]
                best_round = rnd
                best_threshold = threshold
                best_W = server.W.clone()
                patience = 0
            else:
                patience += 1

            # Save both the human-readable round history and a full resumable
            # checkpoint to Drive. The LAST checkpoint is what restart uses.
            atomic_dataframe_csv(pd.DataFrame(history), history_path)
            if rnd % max(CHECKPOINT_EVERY_N_ROUNDS, 1) == 0 or rnd == cfg.n_rounds:
                save_training_checkpoint(
                    last_ckpt_path,
                    fingerprint=fingerprint,
                    server=server,
                    cfg=cfg,
                    round_idx=rnd,
                    history=history,
                    best_val_f1=best_val_f1,
                    best_round=best_round,
                    best_threshold=best_threshold,
                    best_W=best_W,
                    patience=patience,
                    malicious=malicious,
                    rng_py=rng_py,
                    torch_gen=torch_gen,
                    accountant=accountant,
                )
                if improved:
                    # Best checkpoint also contains full resume state; useful if
                    # the last few rounds become unstable.
                    shutil.copy2(last_ckpt_path, best_ckpt_path)

            eps_text = f"{eps:.3f}" if np.isfinite(eps) else ("inf" if np.isinf(eps) else "N/A")
            print(
                f"R{rnd:02d}/{cfg.n_rounds} | Val F1={val_metrics['F1']:.4f} "
                f"Acc={val_metrics['Accuracy']:.4f} AUC={val_metrics['ROC_AUC']:.4f} "
                f"MCC={val_metrics['MCC']:.4f} | thr={threshold:.4f} "
                f"eps={eps_text} | t={round_sec:.1f}s | checkpoint=OK"
            )

            if (
                cfg.enable_early_stopping
                and rnd >= cfg.min_rounds_before_stop
                and patience >= cfg.early_stop_patience
            ):
                print(f"[EARLY STOP] No validation F1 improvement for {patience} rounds.")
                break
    else:
        print("[RESUME] Training had already met the early-stop condition; proceeding to final test.")

    if best_W is None:
        best_W = server.W.clone()
        best_round = max(start_round - 1, 0)
    server.W = best_W
    set_params_from_vector(server.model, best_W)

    # TEST IS EVALUATED ONCE, using validation-selected round and threshold.
    test_prob = server.predict_proba(data.X_test)
    test_metrics = compute_metrics(data.y_test, test_prob, best_threshold)

    history_df = pd.DataFrame(history)
    atomic_dataframe_csv(history_df, history_path)

    result: Dict[str, Any] = {
        "RunFingerprint": fingerprint,
        "Dataset": data.dataset,
        "Scenario": scenario_name,
        "ExperimentGroup": experiment_group_from_name(scenario_name),
        "Aggregator": cfg.aggregator,
        "R3Variant": cfg.r3_variant if scenario_name.startswith("r3_") else "",
        "DefenseMethod": (
            "P4P" if cfg.aggregator == "p4p" else
            "FedProx" if (cfg.aggregator == "fedavg" and cfg.fedprox_mu > 0) else
            "FedAvg" if cfg.aggregator == "fedavg" else
            "CoordinateMedian" if cfg.aggregator == "median" else
            "TrimmedMean" if cfg.aggregator == "trimmed_mean" else
            f"RPP-Event-{cfg.r3_variant}" if cfg.aggregator == "rpp_event" else
            "RPP-FL-Legacy-V5"
        ),
        "AttackType": cfg.attack_type if attacked else "none",
        "MaliciousFraction": cfg.frac_malicious if attacked else 0.0,
        "DirichletAlpha": cfg.dirichlet_alpha,
        "PrivacyMode": cfg.privacy_mode,
        "DPNoiseMultiplier": cfg.dp_noise_multiplier,
        "DPTargetEpsilon": cfg.dp_target_epsilon,
        "UseTransformer": cfg.use_transformer,
        "UseAuxReconstruction": cfg.recon_lambda > 0,
        "UseReputation": cfg.use_reputation,
        "UseMomentum": cfg.use_momentum,
        "UseRobustCenter": cfg.use_robust_center,
        "UseNormConsistency": cfg.use_norm_consistency,
        "FixedRoundBudget": not cfg.enable_early_stopping,
        "ConfiguredRounds": cfg.n_rounds,
        "ProtocolSeed": cfg.seed,
        "R3ProtocolCodeVersion": R3_AUDIT_VERSION if cfg.aggregator == "rpp_event" else None,
        "MaliciousClientIDs": json.dumps(sorted(int(c) for c in malicious)),
        "P4PReferenceDOI": P4P_REFERENCE_DOI if cfg.aggregator == "p4p" else "",
        "P4PImplementation": (
            "Independent public-method reimplementation; same local model/data/protocol"
            if cfg.aggregator == "p4p" else ""
        ),
        "BestRound": best_round,
        "ValBestF1": best_val_f1,
        "Threshold": best_threshold,
        "ModelParams": n_params,
        "ModelSizeMB_FP32": (n_params * 4.0) / (1024.0 ** 2),
        "Rows": data.raw_rows,
        "SelectedFeatures": data.n_selected_features,
        **{f"Test_{k}": v for k, v in test_metrics.items()},
    }

    final_eps = history[-1]["Epsilon"] if history else float("nan")
    result["FinalEpsilon"] = final_eps
    if not history_df.empty:
        result["RoundsExecuted"] = int(history_df["Round"].max())
        result["TotalTrainTime_s"] = float(history_df["RoundTime_s"].sum())
        result["AvgRoundTime_s"] = float(history_df["RoundTime_s"].mean())
        result["AvgClientTime_s"] = float(history_df["AvgClientTime_s"].mean())
        result["ApproxUploadMBPerRound"] = float(
            (n_params * 4.0 * per_round) / (1024.0 ** 2)
        )
        if "ClientAttackDetectAUC" in history_df:
            vals = pd.to_numeric(history_df["ClientAttackDetectAUC"], errors="coerce")
            result["MeanClientAttackDetectAUC"] = float(vals.mean()) if vals.notna().any() else float("nan")
        if "MaliciousWeightShare" in history_df:
            vals = pd.to_numeric(history_df["MaliciousWeightShare"], errors="coerce")
            result["MeanMaliciousWeightShare"] = float(vals.mean()) if vals.notna().any() else float("nan")
        for src, dst in (
            ("P4PKept", "MeanP4PKeptClients"),
            ("P4PRejected", "MeanP4PRejectedClients"),
            ("P4PMeanInstantSuspicion", "MeanP4PInstantSuspicion"),
            ("P4PMeanTemporalSuspicion", "MeanP4PTemporalSuspicion"),
            ("P4PMeanProbeCosine", "MeanP4PProbeCosine"),
            ("R3FlaggedFraction", "MeanR3FlaggedFraction"),
            ("R3MedianAnchor", "MeanR3MedianAnchor"),
            ("R3MeanTrust", "MeanR3Trust"),
            ("R3MeanReputation", "MeanR3Reputation"),
        ):
            if src in history_df:
                vals = pd.to_numeric(history_df[src], errors="coerce")
                result[dst] = float(vals.mean()) if vals.notna().any() else float("nan")

    atomic_json_dump(result, os.path.join(run_dir, "final_result.json"))
    atomic_json_dump({
        "status": "COMPLETED",
        "fingerprint": fingerprint,
        "best_round": best_round,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, os.path.join(run_dir, "COMPLETED.json"))

    print("\nTEST (evaluated once):")
    print(pd.Series(result).to_string())

    del server, clients
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, history_df


# =============================================================================
# 17. OPTIONAL PAILLIER MICRO-BENCHMARK
# =============================================================================

def paillier_microbenchmark(n_coords: int = PAILLIER_BENCHMARK_COORDS,
                            key_bits: int = PAILLIER_KEY_BITS,
                            seed: int = SEED) -> Dict[str, float]:
    """
    Real Paillier timing on a small coordinate sample. This is NOT used to
    pretend the full model was homomorphically encrypted. Report it explicitly
    as a micro-benchmark unless you replace the training transport with full HE.
    """
    try:
        from phe import paillier  # type: ignore
    except Exception as e:
        raise RuntimeError("Install 'phe' first: pip install phe") from e

    rng = np.random.default_rng(seed)
    x = rng.integers(-100000, 100000, size=n_coords, dtype=np.int64)
    t0 = time.perf_counter()
    pub, priv = paillier.generate_paillier_keypair(n_length=key_bits)
    keygen = time.perf_counter() - t0

    t0 = time.perf_counter()
    enc = [pub.encrypt(int(v)) for v in x.tolist()]
    enc_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    dec = np.array([priv.decrypt(c) for c in enc], dtype=np.int64)
    dec_t = time.perf_counter() - t0

    if not np.array_equal(x, dec):
        raise RuntimeError("Paillier round-trip failed")
    return {
        "coords": n_coords,
        "key_bits": key_bits,
        "keygen_s": keygen,
        "encrypt_s": enc_t,
        "decrypt_s": dec_t,
        "encrypt_ms_per_coord": 1000.0 * enc_t / n_coords,
        "decrypt_ms_per_coord": 1000.0 * dec_t / n_coords,
    }


# =============================================================================
# 18. EXPERIMENT SUITE
# =============================================================================

def scenario_configs(base: Config, dataset_name: str) -> List[Tuple[str, Config, bool]]:
    """Build the publication experiment plan for one dataset.

    Main results run on all five datasets. Computationally expensive controlled
    studies run only on the representative datasets configured near the top of
    the file. This keeps the final study broad without multiplying every factor
    into an infeasible Cartesian product.
    """
    out: List[Tuple[str, Config, bool]] = []
    if RUN_REVIEWER3_MECHANISM_ONLY:
        planned = []
        for protocol_seed in R3_SEEDS:
            for variant in R3_VARIANTS:
                if variant == "p4p" and not R3_INCLUDE_P4P:
                    continue
                for name, alpha, attacked in (
                    [("clean", 0.5, False)] if R3_RUN_CLEAN_CONTROL else []
                ) + [("alie20", 0.5, True)] + (
                    [("alie20", 0.1, True)] if R3_RUN_SEVERE_NONIID else []
                ):
                    # Avoid comparing methods run with different data/model
                    # seeds. Client positions/identities/schedules match within
                    # each seed & condition; trajectories naturally diverge.
                    alpha_tag = "a01" if alpha == 0.1 else "a05"
                    scenario = f"r3_{variant}_{name}_{alpha_tag}_seed{protocol_seed}"
                    cfg = replace(
                        base, seed=int(protocol_seed),
                        aggregator="p4p" if variant == "p4p" else "rpp_event",
                        r3_variant=variant, r3_strict_noniid=True,
                        dirichlet_alpha=float(alpha), frac_malicious=0.20,
                        attack_type="alie", enable_early_stopping=False,
                        privacy_mode="none", dp_noise_multiplier=0.0,
                        dp_target_epsilon=None, rep_init=0.90,
                        beta_momentum=0.90, update_clip_C=1.0,
                    )
                    planned.append((scenario, cfg, attacked))
        return planned

    base = replace(
        base,
        frac_malicious=MAIN_ATTACK_FRACTION,
        attack_type=PRIMARY_ATTACK,
        privacy_mode="none",
        dp_noise_multiplier=0.0,
        dp_target_epsilon=None,
    )

    # ---------------- Main five-dataset results ----------------
    if RUN_MAIN_FIVE_DATASETS:
        out.append(("proposed_clean", replace(base, aggregator="rpp"), False))
        out.append((
            f"proposed_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION * 100)}",
            replace(base, aggregator="rpp", attack_type=PRIMARY_ATTACK),
            True,
        ))

    # ---------------- Attack type + malicious ratio ----------------
    if RUN_ATTACK_SUITE and dataset_name in ROBUSTNESS_DATASETS:
        for atk in ATTACK_TYPES:
            name = f"attack_{atk}_m{int(MAIN_ATTACK_FRACTION * 100)}"
            # ALIE 20% is already the main attacked run; do not retrain it.
            if atk == PRIMARY_ATTACK:
                continue
            out.append((
                name,
                replace(base, aggregator="rpp", attack_type=atk,
                        frac_malicious=MAIN_ATTACK_FRACTION),
                True,
            ))

    if RUN_MALICIOUS_RATIO_SWEEP and dataset_name in ROBUSTNESS_DATASETS:
        for frac in MALICIOUS_FRACTIONS:
            if abs(frac - MAIN_ATTACK_FRACTION) < 1e-12:
                continue
            out.append((
                f"ratio_{PRIMARY_ATTACK}_m{int(frac * 100)}",
                replace(base, aggregator="rpp", attack_type=PRIMARY_ATTACK,
                        frac_malicious=float(frac)),
                True,
            ))

    # ---------------- Controlled aggregation baselines ----------------
    if RUN_BASELINES and dataset_name in BASELINE_DATASETS:
        baseline_defs = (
            ("fedavg", dict(aggregator="fedavg", fedprox_mu=0.0)),
            ("fedprox", dict(aggregator="fedavg", fedprox_mu=0.01)),
            ("median", dict(aggregator="median", fedprox_mu=0.0)),
            ("trimmed", dict(aggregator="trimmed_mean", fedprox_mu=0.0)),
        )
        for nm, kw in baseline_defs:
            out.append((f"{nm}_clean", replace(base, **kw), False))
            out.append((
                f"{nm}_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION * 100)}",
                replace(base, attack_type=PRIMARY_ATTACK,
                        frac_malicious=MAIN_ATTACK_FRACTION, **kw),
                True,
            ))

    # ---------------- Reviewer-1 matched poisoning baselines ----------------
    # Direct-comparison runs are deliberately separate from the legacy/main runs.
    # Every method below receives the same 25-round budget (early stopping OFF),
    # the same train/validation/test arrays, the same Dirichlet partition seed,
    # the same client-sampling RNG seed, the same malicious-client identities, and
    # the same attack type/hyperparameters. Validation still selects the checkpoint
    # used for the single final test evaluation.
    if RUN_REVIEWER1_MATCHED_BASELINES and dataset_name in BASELINE_DATASETS:
        reviewer_defs = (
            ("rpp", dict(aggregator="rpp", fedprox_mu=0.0)),
            ("p4p", dict(aggregator="p4p", fedprox_mu=0.0)),
            ("fedavg", dict(aggregator="fedavg", fedprox_mu=0.0)),
            ("fedprox", dict(aggregator="fedavg", fedprox_mu=0.01)),
            ("median", dict(aggregator="median", fedprox_mu=0.0)),
            ("trimmed", dict(aggregator="trimmed_mean", fedprox_mu=0.0)),
        )
        matched_base = replace(base, enable_early_stopping=False)

        # Core reviewer table: clean and the adaptive ALIE-20% condition.
        for nm, kw in reviewer_defs:
            out.append((
                f"reviewer1_{nm}_clean",
                replace(matched_base, **kw),
                False,
            ))
            out.append((
                f"reviewer1_{nm}_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION * 100)}",
                replace(
                    matched_base, attack_type=PRIMARY_ATTACK,
                    frac_malicious=MAIN_ATTACK_FRACTION, **kw
                ),
                True,
            ))

        # Severe heterogeneity is the reviewer-critical stress case: alpha=0.1
        # together with the same ALIE-20% adversarial setting.
        if RUN_REVIEWER1_SEVERE_NONIID:
            for nm, kw in reviewer_defs:
                out.append((
                    f"reviewer1_{nm}_alpha01_{PRIMARY_ATTACK}_m20",
                    replace(
                        matched_base, dirichlet_alpha=REVIEWER1_SEVERE_ALPHA,
                        attack_type=PRIMARY_ATTACK, frac_malicious=0.20, **kw
                    ),
                    True,
                ))

        # Breadth check for the new recent poisoning-specific comparator. These
        # additional matched runs compare P4P and RPP under the remaining attacks
        # already used by this paper, without multiplying every classical baseline.
        if RUN_REVIEWER1_EXTENDED_ATTACKS:
            for atk in REVIEWER1_EXTENDED_ATTACK_TYPES:
                for nm, kw in reviewer_defs[:2]:  # RPP and P4P only
                    out.append((
                        f"reviewer1_{nm}_{atk}_m20",
                        replace(
                            matched_base, attack_type=atk, frac_malicious=0.20, **kw
                        ),
                        True,
                    ))

    # ---------------- Component ablation ----------------
    if RUN_ABLATION and dataset_name in ABLATION_DATASETS:
        # Model-component ablations are measured on the clean task.
        out.append((
            "abl_no_transformer_clean",
            replace(base, use_transformer=False),
            False,
        ))
        out.append((
            "abl_no_aux_reconstruction_clean",
            replace(base, recon_lambda=0.0),
            False,
        ))

        # Robust-aggregation components are measured under adaptive ALIE.
        robust_ablations = (
            ("abl_no_reputation_alie20", dict(use_reputation=False)),
            ("abl_no_momentum_alie20", dict(use_momentum=False)),
            ("abl_no_robust_center_alie20", dict(use_robust_center=False)),
            ("abl_no_norm_consistency_alie20", dict(use_norm_consistency=False)),
        )
        for nm, kw in robust_ablations:
            out.append((
                nm,
                replace(base, attack_type="alie", frac_malicious=0.20, **kw),
                True,
            ))

    # ---------------- Non-IID sensitivity ----------------
    if RUN_NONIID_SWEEP and dataset_name in NONIID_DATASETS:
        for a in NONIID_ALPHAS:
            if abs(a - base.dirichlet_alpha) < 1e-12:
                continue
            out.append((
                f"noniid_alpha{a}_clean",
                replace(base, dirichlet_alpha=float(a)),
                False,
            ))
            out.append((
                f"noniid_alpha{a}_{PRIMARY_ATTACK}_m20",
                replace(base, dirichlet_alpha=float(a), attack_type=PRIMARY_ATTACK,
                        frac_malicious=0.20),
                True,
            ))

    # ---------------- Privacy-utility sweep ----------------
    if RUN_PRIVACY_SWEEP and dataset_name in PRIVACY_DATASETS:
        q = max(1, int(round(base.clients_per_round_frac * base.n_clients))) / base.n_clients
        for target_eps in TARGET_EPSILONS:
            try:
                nm = calibrate_noise_multiplier(
                    target_epsilon=float(target_eps),
                    sample_rate=float(q),
                    steps=base.n_rounds,
                    delta=base.dp_delta,
                )
                planned = estimate_rdp_epsilon(nm, q, base.n_rounds, base.dp_delta)
                print(
                    f"[DP PLAN] {dataset_name}: target eps={target_eps:g}, "
                    f"noise_multiplier={nm:.4f}, planned eps={planned:.4f}"
                )
                out.append((
                    f"privacy_localdp_eps{target_eps:g}",
                    replace(
                        base,
                        privacy_mode="local_dp",
                        dp_noise_multiplier=float(nm),
                        dp_target_epsilon=float(target_eps),
                    ),
                    False,
                ))
            except Exception as e:
                print(f"[WARN] Skipping privacy epsilon={target_eps:g}: {e}")

    # Deduplicate names while preserving order.
    seen = set()
    unique: List[Tuple[str, Config, bool]] = []
    for item in out:
        if item[0] not in seen:
            unique.append(item)
            seen.add(item[0])
    return unique


def write_publication_tables(result_df: pd.DataFrame) -> None:
    """Create paper-ready experiment tables from the master result file."""
    if result_df.empty:
        return
    out_dir = os.path.join(RESULT_ROOT, "PUBLICATION_TABLES")
    os.makedirs(out_dir, exist_ok=True)
    df = result_df.copy()

    # Add degradation relative to the clean proposed model on the same dataset.
    clean = (
        df[df["Scenario"] == "proposed_clean"]
        .drop_duplicates("Dataset", keep="last")
        .set_index("Dataset")
    )
    df["F1DropVsClean_pp"] = np.nan
    df["AccuracyDropVsClean_pp"] = np.nan
    for i, row in df.iterrows():
        ds = row["Dataset"]
        if ds in clean.index:
            df.loc[i, "F1DropVsClean_pp"] = 100.0 * (
                float(clean.loc[ds, "Test_F1"]) - float(row["Test_F1"])
            )
            df.loc[i, "AccuracyDropVsClean_pp"] = 100.0 * (
                float(clean.loc[ds, "Test_Accuracy"]) - float(row["Test_Accuracy"])
            )

    key_cols = [
        "Dataset", "Scenario", "ExperimentGroup", "DefenseMethod", "Aggregator", "AttackType",
        "MaliciousFraction", "DirichletAlpha", "PrivacyMode",
        "FixedRoundBudget", "ConfiguredRounds", "ProtocolSeed", "MaliciousClientIDs",
        "DPTargetEpsilon", "DPNoiseMultiplier", "FinalEpsilon",
        "Test_Accuracy", "Test_BalancedAccuracy", "Test_Precision",
        "Test_Recall", "Test_F1", "Test_MCC", "Test_ROC_AUC", "Test_PR_AUC",
        "Test_FPR", "Test_FNR", "F1DropVsClean_pp", "AccuracyDropVsClean_pp",
        "BestRound", "RoundsExecuted", "TotalTrainTime_s", "AvgRoundTime_s",
        "ModelParams", "ModelSizeMB_FP32", "ApproxUploadMBPerRound",
        "MeanClientAttackDetectAUC", "MeanMaliciousWeightShare",
        "MeanP4PKeptClients", "MeanP4PRejectedClients",
        "MeanP4PInstantSuspicion", "MeanP4PTemporalSuspicion",
        "MeanP4PProbeCosine", "P4PReferenceDOI", "P4PImplementation",
    ]
    key_cols = [c for c in key_cols if c in df.columns]

    main = df[df["Scenario"].isin([
        "proposed_clean", f"proposed_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION*100)}"
    ])].copy()
    atomic_dataframe_csv(main[key_cols], os.path.join(out_dir, "TABLE_1_MAIN_FIVE_DATASETS.csv"))

    baseline_mask = (
        df["Dataset"].isin(BASELINE_DATASETS)
        & (df["ExperimentGroup"].eq("baseline") | df["Scenario"].isin([
            "proposed_clean", f"proposed_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION*100)}"
        ]))
    )
    atomic_dataframe_csv(
        df.loc[baseline_mask, key_cols],
        os.path.join(out_dir, "TABLE_2_CONTROLLED_BASELINES.csv"),
    )

    # Reviewer-1 direct evidence table. Unlike literature-value comparisons, all
    # rows here are generated by this script under the matched experimental protocol.
    reviewer1 = df[
        df["Dataset"].isin(BASELINE_DATASETS)
        & df["ExperimentGroup"].eq("reviewer1_matched")
    ].copy()
    if not reviewer1.empty:
        reviewer1["ReviewerCondition"] = np.select(
            [
                reviewer1["Scenario"].str.contains("_clean", regex=False),
                reviewer1["Scenario"].str.contains("_alpha01_", regex=False),
                reviewer1["Scenario"].str.contains("_alie_m20", regex=False),
                reviewer1["Scenario"].str.contains("_signflip_m20", regex=False),
                reviewer1["Scenario"].str.contains("_scale_m20", regex=False),
                reviewer1["Scenario"].str.contains("_gaussian_m20", regex=False),
            ],
            [
                "Clean alpha=0.5",
                "ALIE-20% alpha=0.1",
                "ALIE-20% alpha=0.5",
                "SignFlip-20% alpha=0.5",
                "Scaling-20% alpha=0.5",
                "Gaussian-20% alpha=0.5",
            ],
            default="Other matched condition",
        )
        reviewer_cols = ["ReviewerCondition"] + key_cols
        reviewer_cols = [c for c in reviewer_cols if c in reviewer1.columns]
        reviewer1 = reviewer1.sort_values(
            ["Dataset", "ReviewerCondition", "DefenseMethod", "Scenario"],
            kind="stable",
        )
        atomic_dataframe_csv(
            reviewer1[reviewer_cols],
            os.path.join(out_dir, "TABLE_R1_MATCHED_POISONING_BASELINES.csv"),
        )

        core = reviewer1[
            reviewer1["ReviewerCondition"].isin([
                "Clean alpha=0.5", "ALIE-20% alpha=0.5", "ALIE-20% alpha=0.1"
            ])
        ].copy()
        atomic_dataframe_csv(
            core[reviewer_cols],
            os.path.join(out_dir, "TABLE_R1_CORE_DIRECT_COMPARISON.csv"),
        )
        # Also overwrite the legacy controlled-baseline table with the stricter
        # reviewer-matched version so the manuscript table is not accidentally
        # regenerated from only FedAvg/FedProx/Median/Trimmed Mean.
        atomic_dataframe_csv(
            core[reviewer_cols],
            os.path.join(out_dir, "TABLE_2_CONTROLLED_BASELINES.csv"),
        )

        p4p_rpp = reviewer1[reviewer1["DefenseMethod"].isin(["RPP-FL", "P4P"])].copy()
        atomic_dataframe_csv(
            p4p_rpp[reviewer_cols],
            os.path.join(out_dir, "TABLE_R1_RPP_VS_P4P_ALL_ATTACKS.csv"),
        )

        atomic_json_dump({
            "purpose": "Reviewer-1 controlled poisoning-specific baseline protocol",
            "direct_comparison": True,
            "datasets": list(BASELINE_DATASETS),
            "defenses": ["RPP-FL", "P4P", "FedAvg", "FedProx", "CoordinateMedian", "TrimmedMean"],
            "same_preprocessed_dataset_object": True,
            "same_dirichlet_partition_seed": SEED,
            "same_local_model": "HybridTabularTransformer",
            "same_optimizer": "AdamW (FedProx adds only its method-specific proximal term)",
            "same_round_budget": int(Config().n_rounds),
            "early_stopping_in_direct_comparison": False,
            "same_client_sampling_seed": SEED + 9001,
            "same_attack_generator_seed": SEED + 8128,
            "same_attack_hyperparameters": True,
            "test_policy": "Validation selects best round/threshold; test evaluated once",
            "p4p_reference_doi": P4P_REFERENCE_DOI,
            "p4p_implementation_note": (
                "Independent reimplementation of publicly documented P4P components; "
                "not claimed to be the authors' official code or unpublished hyperparameters."
            ),
        }, os.path.join(out_dir, "REVIEWER1_PROTOCOL_MANIFEST.json"))

    ablation_mask = (
        df["Dataset"].isin(ABLATION_DATASETS)
        & (df["ExperimentGroup"].eq("ablation") | df["Scenario"].isin([
            "proposed_clean", f"proposed_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION*100)}"
        ]))
    )
    ablation_cols = key_cols + [c for c in [
        "UseTransformer", "UseAuxReconstruction", "UseReputation", "UseMomentum",
        "UseRobustCenter", "UseNormConsistency"
    ] if c in df.columns and c not in key_cols]
    atomic_dataframe_csv(
        df.loc[ablation_mask, ablation_cols],
        os.path.join(out_dir, "TABLE_3_ABLATION_STUDY.csv"),
    )

    robust_mask = (
        df["Dataset"].isin(ROBUSTNESS_DATASETS)
        & (df["ExperimentGroup"].eq("robustness") | df["Scenario"].isin([
            "proposed_clean", f"proposed_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION*100)}"
        ]))
    )
    atomic_dataframe_csv(
        df.loc[robust_mask, key_cols],
        os.path.join(out_dir, "TABLE_4_POISONING_ROBUSTNESS.csv"),
    )

    noniid_mask = (
        df["Dataset"].isin(NONIID_DATASETS)
        & (df["ExperimentGroup"].eq("non_iid") | df["Scenario"].isin([
            "proposed_clean", f"proposed_{PRIMARY_ATTACK}_m{int(MAIN_ATTACK_FRACTION*100)}"
        ]))
    )
    atomic_dataframe_csv(
        df.loc[noniid_mask, key_cols],
        os.path.join(out_dir, "TABLE_5_NONIID_SENSITIVITY.csv"),
    )

    privacy_mask = (
        df["Dataset"].isin(PRIVACY_DATASETS)
        & (df["ExperimentGroup"].eq("privacy") | df["Scenario"].eq("proposed_clean"))
    )
    atomic_dataframe_csv(
        df.loc[privacy_mask, key_cols],
        os.path.join(out_dir, "TABLE_6_PRIVACY_UTILITY.csv"),
    )

    # Compact best-metric summary useful when drafting Results/Discussion.
    summary_cols = [c for c in [
        "Dataset", "Scenario", "Test_Accuracy", "Test_BalancedAccuracy",
        "Test_F1", "Test_MCC", "Test_ROC_AUC", "Test_FPR", "FinalEpsilon"
    ] if c in df.columns]
    atomic_dataframe_csv(
        df[summary_cols],
        os.path.join(out_dir, "ALL_EXPERIMENTS_COMPACT.csv"),
    )
    print(f"[TABLES] Paper-ready CSV tables saved in: {out_dir}")



# =============================================================================
# V6: REVIEWER-3 EXPERIMENT REPORTING AND PREPUBLICATION ARTIFACT
# =============================================================================

def _r3_seed_summary(col: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(col, errors="coerce").dropna().to_numpy(dtype=float)
    n = len(a)
    mean = float(a.mean()) if n else float("nan")
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    if n > 1:
        from scipy.stats import t
        margin = float(t.ppf(0.975, n - 1) * sd / math.sqrt(n))
    else:
        margin = float("nan")
    return {"N": n, "Mean": mean, "SD": sd,
            "CI95_Low": mean - margin, "CI95_High": mean + margin}


def write_reviewer3_tables(df: pd.DataFrame) -> None:
    """Never infer winners; report raw/paired and summary metrics automatically."""
    r3 = df[df["ExperimentGroup"].eq("reviewer3_mechanism")].copy()
    if r3.empty:
        return
    outdir = Path(RESULT_ROOT) / "R3_TABLES"
    outdir.mkdir(parents=True, exist_ok=True)
    r3["F1_percent"] = pd.to_numeric(r3["Test_F1"], errors="coerce") * 100
    r3["MCC_percent"] = pd.to_numeric(r3["Test_MCC"], errors="coerce") * 100
    r3["MaliciousWeight_percent"] = pd.to_numeric(
        r3["MeanMaliciousWeightShare"], errors="coerce") * 100
    r3["Condition"] = r3.apply(lambda r:
        ("clean" if r["AttackType"] == "none" else "alie20")
        + ("_alpha01" if abs(float(r["DirichletAlpha"]) - 0.1) < 1e-8 else "_alpha05"),
        axis=1)
    r3 = r3.sort_values(["Dataset", "Condition", "ProtocolSeed", "R3Variant"])
    atomic_dataframe_csv(r3, str(outdir / "MECHANISM_PER_SEED.csv"))

    measures = ["F1_percent", "MCC_percent", "MeanClientAttackDetectAUC",
                "MaliciousWeight_percent", "MeanR3FlaggedFraction",
                "MeanR3MedianAnchor", "MeanR3Trust"]
    rows = []
    for (ds, cond, variant), block in r3.groupby(
        ["Dataset", "Condition", "R3Variant"], sort=True):
        for measure in measures:
            if measure not in block:
                continue
            s = _r3_seed_summary(block[measure])
            rows.append({"Dataset": ds, "Condition": cond,
                         "Variant": variant, "Metric": measure, **s})
    atomic_dataframe_csv(pd.DataFrame(rows), str(outdir / "MECHANISM_MEAN_SD_CI95.csv"))

    paired = []
    index_cols = ["Dataset", "Condition", "ProtocolSeed"]
    for metric in measures[:4]:
        if metric not in r3:
            continue
        pivot = r3.pivot_table(index=index_cols, columns="R3Variant", values=metric,
                               aggfunc="first")
        if "full" not in pivot:
            continue
        for compared in [v for v in R3_VARIANTS if v != "full" and v in pivot]:
            paired_df = pivot[["full", compared]].dropna()
            for (ds, cond, seed), z in paired_df.iterrows():
                paired.append({"Dataset": ds, "Condition": cond,
                               "Seed": int(seed), "Metric": metric,
                               "Full": float(z["full"]), "Comparator": compared,
                               "ComparatorValue": float(z[compared]),
                               "Delta_FullMinusComparator": float(z["full"]-z[compared])})
    paired_df = pd.DataFrame(paired)
    atomic_dataframe_csv(paired_df, str(outdir / "FULL_VS_ALTERNATIVES_PAIRED.csv"))
    paired_summary=[]
    if not paired_df.empty:
        for (ds, cond, comp, metric), block in paired_df.groupby(
            ["Dataset", "Condition", "Comparator", "Metric"], sort=True):
            paired_summary.append({"Dataset": ds, "Condition": cond,
                                   "Comparator": comp, "Metric": metric,
                                   **_r3_seed_summary(block["Delta_FullMinusComparator"])})
    atomic_dataframe_csv(pd.DataFrame(paired_summary), str(outdir / "PAIRED_DIFFERENCE_SUMMARY.csv"))

    # Security observability must be checked on selected clients each round;
    # export round diagnostics without looking at training targets.
    round_dfs = []
    for rr in r3.itertuples(index=False):
        path = Path(RESULT_ROOT) / rr.Dataset / rr.Scenario / "round_history.csv"
        if path.exists():
            h = pd.read_csv(path)
            wanted = [x for x in [
                "Round", "Val_F1", "MeanTrust", "ClientAttackDetectAUC",
                "MaliciousWeightShare", "R3FlaggedFraction", "R3MedianAnchor",
                "R3FlaggedClientIDs", "SelectedClients", "SelectedMaliciousClients"
            ] if x in h]
            small = h[wanted].copy()
            small.insert(0, "Seed", rr.ProtocolSeed)
            small.insert(0, "Variant", rr.R3Variant)
            small.insert(0, "Condition", rr.Condition)
            small.insert(0, "Dataset", rr.Dataset)
            round_dfs.append(small)
    if round_dfs:
        atomic_dataframe_csv(pd.concat(round_dfs, ignore_index=True),
                             str(outdir / "ROUND_LEVEL_SECURITY.csv"))
    print(f"[R3 TABLES] Updated {outdir} | runs={len(r3)}")


def publish_draft_reproducibility_bundle(state: dict) -> None:
    """Create an honest LOCAL artifact; never claim remote availability."""
    import zipfile
    root = Path(RESULT_ROOT)
    target = root / "REPRODUCIBILITY_DRAFT.zip"
    code_path = Path(__file__) if "__file__" in globals() else None
    instructions = (
        "RPP-FL Reviewer-3 draft audit bundle (NOT public, not published).\n"
        "This is an experimental reconstruction of manuscript server rules; \n"
        "verify implementation against actual historical source before claiming \n"
        "that previous Table 4-8 values have been reproduced.\n"
        "Re-run: Google Colab GPU, execute RPP_FL_V6_REVIEWER3.py.\n"
        "Datasets: CICIoT2023 compact Kaggle mirror and TON_IoT network CSV.\n"
        "Each seed rebuilds deterministic sampling and split.\n"
        "No raw traffic records included. No secure aggregation claimed.\n"
        "Numeric near-zero MAD floor and two-seed tie policies specified in code.\n"
        "Some draft conditions may be incomplete: inspect _STATE/execution_state.json.\n"
    )
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as z:
        z.writestr("README_REPRODUCIBILITY.txt", instructions)
        z.writestr("execution_state.json", json.dumps(state, indent=2))
        if code_path and code_path.is_file():
            z.write(code_path, "RPP_FL_V6_REVIEWER3.py")
        for fname in ["run_config.json", "ALL_DATASETS_FINAL_RESULTS.csv",
                      "all_results_partial.csv"]:
            fp = root / fname
            if fp.exists(): z.write(fp, fname)
        for fp in (root / "R3_TABLES").glob("*.csv"):
            z.write(fp, str(fp.relative_to(root)))
        for ds in R3_DATASETS:
            for fp in (root / ds).glob("r3_*/final_result.json"):
                z.write(fp, str(fp.relative_to(root)))
            for fp in (root / ds).glob("r3_*/round_history.csv"):
                z.write(fp, str(fp.relative_to(root)))
        for ds in R3_DATASETS:
            for fp in (Path(CACHE_ROOT) / ds).glob("*/prepared_meta.json"):
                z.write(fp, str(Path("preprocessing") / ds / fp.parent.name / fp.name))
            for fp in (Path(CACHE_ROOT) / ds).glob("*/split_indices.npz"):
                z.write(fp, str(Path("preprocessing") / ds / fp.parent.name / fp.name))
    print(f"[AUDIT DRAFT] {target}")

def main() -> None:
    set_seed(SEED)
    mount_drive_if_needed()
    ensure_dirs()

    print_header("RPP-FL MULTI-DATASET V5 — REVIEWER-1 MATCHED POISONING BASELINES")
    print(f"Device: {DEVICE} | AMP: {AMP_ENABLED} | Seed: {SEED}")
    print(f"Data cache: {DATA_ROOT}")
    print(f"Results/checkpoints: {RESULT_ROOT}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Optional dependencies. Opacus is installed so the RDP accountant can
    # persist a real accounting state when DP is enabled.
    if not RUN_REVIEWER3_MECHANISM_ONLY:
        try:
            pip_install_if_missing("opacus", "opacus")
        except Exception as e:
            print(f"[WARN] Opacus installation failed; epsilon will be N/A: {e}")

    if AUTO_DOWNLOAD_DATASETS:
        if RUN_REVIEWER3_MECHANISM_ONLY:
            print_header("REVIEWER-3 DATASET ACQUISITION")
            for dataset_name in R3_DATASETS:
                ensure_dataset_downloaded(dataset_name, force=FORCE_REDOWNLOAD)
        else:
            ensure_all_datasets()

    base_cfg = Config(seed=SEED)
    all_results: List[Dict[str, Any]] = []
    partial_path = os.path.join(RESULT_ROOT, "all_results_partial.csv")

    # Reload completed partial results on a new Colab session.
    if AUTO_RESUME and os.path.exists(partial_path):
        try:
            old_df = pd.read_csv(partial_path)
            all_results = old_df.to_dict(orient="records")
            print(f"[RESUME] Loaded {len(all_results)} previously completed result row(s).")
        except Exception as e:
            print(f"[WARN] Could not load partial results: {e}")

    run_meta = {
        "seed": SEED,
        "max_rows_per_dataset": MAX_ROWS_PER_DATASET,
        "dataset_paths": DATASET_PATHS,
        "base_config": asdict(base_cfg),
        "auto_download": AUTO_DOWNLOAD_DATASETS,
        "auto_resume": AUTO_RESUME,
        "r3_new_suite_only": RUN_REVIEWER3_MECHANISM_ONLY,
        "r3_datasets": R3_DATASETS,
        "r3_seeds": R3_SEEDS,
        "r3_variants": R3_VARIANTS,
        "r3_include_p4p": R3_INCLUDE_P4P,
        "r3_mad_floor_relative": R3_MAD_FLOOR_RELATIVE,
        "r3_mad_floor_absolute": R3_MAD_FLOOR_ABSOLUTE,
        "r3_notice": "Numerical convention/source-code verification needed before manuscript claim",
        "checkpoint_every_n_rounds": CHECKPOINT_EVERY_N_ROUNDS,
        "environment": {"python": sys.version, "torch": torch.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if "__file__" in globals() and Path(__file__).is_file() else None,
        "prepared_cache": CACHE_PREPARED_DATA,
        "final_suite": {
            "run_main_five_datasets": RUN_MAIN_FIVE_DATASETS,
            "run_attack_suite": RUN_ATTACK_SUITE,
            "run_malicious_ratio_sweep": RUN_MALICIOUS_RATIO_SWEEP,
            "run_baselines": RUN_BASELINES,
            "run_reviewer1_matched_baselines": RUN_REVIEWER1_MATCHED_BASELINES,
            "run_reviewer1_severe_noniid": RUN_REVIEWER1_SEVERE_NONIID,
            "run_reviewer1_extended_attacks": RUN_REVIEWER1_EXTENDED_ATTACKS,
            "p4p_reference_doi": P4P_REFERENCE_DOI,
            "run_ablation": RUN_ABLATION,
            "run_noniid_sweep": RUN_NONIID_SWEEP,
            "run_privacy_sweep": RUN_PRIVACY_SWEEP,
            "ablation_datasets": ABLATION_DATASETS,
            "baseline_datasets": BASELINE_DATASETS,
            "robustness_datasets": ROBUSTNESS_DATASETS,
            "noniid_datasets": NONIID_DATASETS,
            "privacy_datasets": PRIVACY_DATASETS,
            "target_epsilons": TARGET_EPSILONS,
        },
    }
    atomic_json_dump(run_meta, os.path.join(RESULT_ROOT, "run_config.json"))

    execution_state_path = os.path.join(STATE_ROOT, "execution_state.json")
    execution_state: Dict[str, Any] = {
        "status": "RUNNING",
        "started_or_resumed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_dataset": None,
        "current_scenario": None,
        "completed": [],
    }
    if AUTO_RESUME and os.path.exists(execution_state_path):
        try:
            with open(execution_state_path, "r", encoding="utf-8") as f:
                old_state = json.load(f)
            execution_state["completed"] = old_state.get("completed", [])
        except Exception:
            pass
    atomic_json_dump(execution_state, execution_state_path)

    dataset_list = R3_DATASETS if RUN_REVIEWER3_MECHANISM_ONLY else (
        "CICIoT2023", "EdgeIIoTset", "TON_IoT", "WUSTL_IIoT", "X-IIoTID"
    )
    for dataset_name in dataset_list:
        path = DATASET_PATHS[dataset_name]
        if not _tabular_files(path):
            print(f"\n[SKIP] {dataset_name}: no usable tabular data at -> {path}")
            continue

        execution_state["current_dataset"] = dataset_name
        execution_state["current_scenario"] = "PREPARING"
        atomic_json_dump(execution_state, execution_state_path)

        prepared_by_seed: Dict[int, PreparedData] = {}
        for scenario_name, cfg, attacked in scenario_configs(base_cfg, dataset_name):
            if cfg.seed not in prepared_by_seed:
                try:
                    prepared_by_seed[cfg.seed] = load_or_prepare_dataset(dataset_name, path, cfg)
                except Exception as e:
                    print(f"[ERROR] Failed to prepare {dataset_name} seed={cfg.seed}: {type(e).__name__}: {e}")
                    execution_state.setdefault("failures", []).append({"dataset": dataset_name, "seed": cfg.seed, "error": str(e)})
                    atomic_json_dump(execution_state, execution_state_path)
                    continue
            prepared = prepared_by_seed.get(cfg.seed)
            if prepared is None:
                continue
            execution_state["current_scenario"] = scenario_name
            atomic_json_dump(execution_state, execution_state_path)
            try:
                result, _ = run_federated_experiment(
                    prepared, cfg, attacked=attacked, scenario_name=scenario_name
                )
                # Replace same dataset/scenario row rather than duplicating it on resume.
                all_results = [r for r in all_results if not (
                    str(r.get("Dataset")) == dataset_name and str(r.get("Scenario")) == scenario_name
                )]
                all_results.append(result)
                current_df = pd.DataFrame(all_results)
                atomic_dataframe_csv(current_df, partial_path)
                # Keep paper-ready tables current after every completed scenario;
                # a Colab disconnect therefore never loses the latest summaries.
                if RUN_REVIEWER3_MECHANISM_ONLY:
                    write_reviewer3_tables(current_df)
                else:
                    write_publication_tables(current_df)

                key = f"{dataset_name}/{scenario_name}"
                if key not in execution_state["completed"]:
                    execution_state["completed"].append(key)
                atomic_json_dump(execution_state, execution_state_path)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[OOM] {dataset_name}/{scenario_name}: {e}")
                    print("Try reducing Config.batch_size or max_model_features. The last completed round is already checkpointed.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print(f"[ERROR] {dataset_name}/{scenario_name}: {e}")
                execution_state.setdefault("failures", []).append({"scenario": scenario_name, "dataset": dataset_name, "error": str(e)})
                atomic_json_dump(execution_state, execution_state_path)
            except Exception as e:
                print(f"[ERROR] {dataset_name}/{scenario_name}: {type(e).__name__}: {e}")
                execution_state.setdefault("failures", []).append({"scenario": scenario_name, "dataset": dataset_name, "error": str(e)})
                atomic_json_dump(execution_state, execution_state_path)

        prepared_by_seed.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if all_results:
        result_df = pd.DataFrame(all_results)
        result_path = os.path.join(RESULT_ROOT, "ALL_DATASETS_FINAL_RESULTS.csv")
        atomic_dataframe_csv(result_df, result_path)

        print_header("FINAL MULTI-DATASET RESULTS")
        cols = [
            "Dataset", "Scenario", "Aggregator", "DirichletAlpha",
            "Test_Accuracy", "Test_Precision", "Test_Recall", "Test_F1",
            "Test_MCC", "Test_ROC_AUC", "Test_PR_AUC", "Test_FPR",
            "BestRound", "FinalEpsilon", "ModelParams",
        ]
        cols = [c for c in cols if c in result_df.columns]
        print(result_df[cols].to_string(index=False))
        print(f"\nSaved: {result_path}")
    else:
        print("\nNo experiments completed. Check dataset acquisition logs and label columns.")

    if RUN_REVIEWER3_MECHANISM_ONLY:
        if all_results:
            write_reviewer3_tables(pd.DataFrame(all_results))
        # General formula: one standard attacked + optional clean + severe.
        expected_count = len(R3_DATASETS) * len(R3_SEEDS) * (
            len(R3_VARIANTS) - int(not R3_INCLUDE_P4P)
        ) * (1 + int(R3_RUN_CLEAN_CONTROL) + int(R3_RUN_SEVERE_NONIID))
        completed_r3 = sum(1 for z in all_results if str(z.get("Scenario", "")).startswith("r3_"))
        print(f"[R3] Completed {completed_r3}/{expected_count} scenario(s).")
        print(f"[R3] See {RESULT_ROOT}/R3_TABLES")
        execution_state["status"] = "COMPLETED" if completed_r3 == expected_count else "PARTIAL"
    else:
        execution_state["status"] = "COMPLETED"
    execution_state["current_dataset"] = None
    execution_state["current_scenario"] = None
    execution_state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_json_dump(execution_state, execution_state_path)

    if RUN_PAILLIER_MICROBENCHMARK and not RUN_REVIEWER3_MECHANISM_ONLY:
        print_header("PAILLIER MICRO-BENCHMARK")
        try:
            pip_install_if_missing("phe", "phe")
            b = paillier_microbenchmark()
            print(pd.Series(b).to_string())
            with open(os.path.join(RESULT_ROOT, "paillier_microbenchmark.json"), "w") as f:
                json.dump(b, f, indent=2)
        except Exception as e:
            print(f"[WARN] Paillier benchmark failed: {e}")


if __name__ == "__main__":
    main()
