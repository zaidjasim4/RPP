# -*- coding: utf-8 -*-
"""
RPP-FL Multi-Dataset Experimental Framework V6 — All-Reviewer Revision Suite
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

Major corrections vs. the original script
------------------------------------------
* Correct per-dataset labels (e.g. X-IIoTID uses class1, not anomaly_alert).
* Leakage-prone label/ID columns are removed explicitly.
* Preprocessing is fitted ONLY on the training split.
* Real label-skew non-IID partitioning using a Dirichlet distribution.
* Correct FL update sign: W_new = W_global + aggregated_delta.
* Validation chooses the best round and threshold; test is evaluated ONCE.
* Stronger feature-token Transformer + residual MLP + auxiliary reconstruction.
* The proposed RPP aggregator is synchronized with manuscript Sections 3.4–3.7:
  MAD-based coordinate deviation, center/history consistency, peer agreement,
  compact-minority collusion detection, asymmetric reputation, winsorization,
  and adaptive median anchoring.
* Better poisoning suite: sign-flip, scaling, Gaussian, and ALIE-style stealth.
* A separate central client-level Gaussian-DP aggregate-release branch uses
  an Opacus RDP accountant; it is never presented as simultaneous with RPP
  screening or as a concrete secure-aggregation implementation.
* Multi-seed reviewer experiments, CSV/JSON audit artifacts, and seed-specific
  split/client-partition/prediction files for reproducibility.
* Automatic public-dataset acquisition and persistent caching in Google Drive.
* Prepared-data cache in Drive so preprocessing is not repeated after restart.
* Round-level resumable checkpoints (last + best), including RNG/accountant state.
* Scenario-level execution state: completed runs are skipped automatically.
* Final publication suite: five-dataset main results, reviewer-matched controlled
  baselines, attack/rate robustness, component ablations, non-IID sensitivity,
  and target-epsilon privacy-utility experiments.
* Recent comparator additions: P4P and a TrustFed-IDS-style trust-aware baseline
  are independently reimplemented under matched preprocessing, client partitions,
  local model, optimizer, client sampling, malicious identities, attacks, and budget.
  They are explicitly labeled independent reimplementations, not official code.
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

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
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
RESULT_ROOT = f"{DRIVE_ROOT}/RPP_FL_RESULTS_V6_ALL_REVIEWERS"
CACHE_ROOT = f"{DRIVE_ROOT}/RPP_FL_CACHE_V6_REVIEW"
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

# -----------------------------------------------------------------------------
# ALL-REVIEWER REVISION EXPERIMENT PLAN
# -----------------------------------------------------------------------------
# This V6 script is intentionally focused on the experiments requested in the
# major revision. Existing V4/V5 paper results remain untouched in their own
# Drive folders. All new runs are resumable and saved under V6.
#
# Reviewer requests covered here:
#   R1/R2 baseline strength     -> P4P + TrustFed-IDS-style + classical baselines
#   leakage-resistant split    -> time/group-separated holdout (EdgeIIoTset by default)
#   repeated trials            -> three independent seeds + mean/std/95% CI/statistics
#   attacker attribution       -> selected-malicious fraction, AUC, malicious weight
#   attack effectiveness       -> matched undefended FedAvg references
#   stronger adaptive attack   -> white-box trust-targeted coordinated stress attack
#   reproducibility artifacts  -> split indices, feature masks/names, configs,
#                                  client partitions, round logs and test predictions
#   privacy accounting detail  -> uniform clipped mean, Gaussian central client-DP,
#                                  no sampling amplification claimed in accountant
#
# Three independent seeds satisfy the reviewers' request for repeated trials
# while keeping the Colab revision suite computationally tractable. Increase
# REVISION_SEEDS if additional power is desired; resume/checkpoint logic is safe.
REVISION_SEEDS = (42, 2026, 31415)
REPEATED_DATASETS = ("CICIoT2023", "TON_IoT")
STRICT_SPLIT_DATASETS = ("EdgeIIoTset",)
AUDIT_DATASETS = ("CICIoT2023", "EdgeIIoTset", "TON_IoT", "WUSTL_IIoT", "X-IIoTID")
PRIVACY_DATASETS = ("TON_IoT",)

RUN_REVISION_MATCHED_BASELINES = True
RUN_REVISION_MULTI_SEED_ABLATION = True
RUN_REVISION_HIGH_RATIO_STRESS = True
RUN_REVISION_ADAPTIVE_ATTACK = True
RUN_REVISION_STRICT_SPLIT = True
RUN_REVISION_PRIVACY = True
RUN_PREPROCESSING_AUDIT_ALL_DATASETS = True

PRIMARY_ATTACK = "alie"
MAIN_ATTACK_FRACTION = 0.20
REVIEWER_SEVERE_ALPHA = 0.10
HIGH_MALICIOUS_FRACTIONS = (0.30, 0.40)
TARGET_EPSILONS = (20.0, 10.0, 5.0, 1.0)

# Recent poisoning-specific baseline: P4P.
P4P_REFERENCE_DOI = "10.1016/j.jnca.2026.104502"
# Recent trust-aware IDS comparator. The public article describes multi-criteria
# trust from cosine similarity, normalized validation loss and temporal update
# stability. Because official source code / unpublished hyperparameters are not
# assumed here, this is explicitly reported as a transparent public-description
# reimplementation using the same local detector and data protocol as RPP-FL.
TRUSTFED_REFERENCE_DOI = "10.1016/j.comnet.2026.112430"

# The direct matched comparison uses the four most informative methods across
# all repeated seeds. Classical Median/Trimmed Mean/FedProx are retained for the
# seed-42 direct table without multiplying the entire revision runtime.
REPEATED_MATCHED_METHODS = ("rpp", "p4p", "trustfed", "fedavg")
SEED42_EXTRA_METHODS = ("fedprox", "median", "trimmed")

# Leakage-resistant evaluation: use a time-separated holdout when a reliable
# timestamp exists; otherwise use session/device groups. The strict run is
# compared with the original stratified record split under the same seed.
STRICT_SPLIT_MODE = "leakage_resistant"
DEFAULT_SPLIT_MODE = "record_stratified"

# Opacus RDPAccountant default orders, stated explicitly for reproducibility.
# Source: opacus.accountants.rdp.RDPAccountant.DEFAULT_ALPHAS.
RDP_ORDERS = [1.0 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

# Paillier remains only an optional cryptographic micro-benchmark; it is not
# presented as the secure aggregation implementation of the robust RPP mode.
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




def _first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    by_lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        hit = by_lower.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    return None


def _parse_time_order(df: pd.DataFrame, candidates: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return sortable time values when a timestamp-like column is usable."""
    col = _first_existing_column(df, candidates)
    if col is None:
        return None, None
    s = df[col]
    num = pd.to_numeric(s, errors="coerce")
    if float(num.notna().mean()) >= 0.90 and int(num.nunique(dropna=True)) >= 20:
        vals = num.to_numpy(dtype=np.float64)
        fill = np.nanmedian(vals[np.isfinite(vals)]) if np.isfinite(vals).any() else 0.0
        vals = np.nan_to_num(vals, nan=float(fill))
        return vals, col
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if float(dt.notna().mean()) >= 0.90 and int(dt.nunique(dropna=True)) >= 20:
        vals = dt.astype("int64").to_numpy(dtype=np.float64)
        return vals, col
    return None, None


def _session_group_ids(df: pd.DataFrame, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Build leakage-resistant session/device groups from raw identifiers.

    These identifiers are used ONLY for split construction and are removed before
    model preprocessing. This prevents the model from learning address/session IDs.
    """
    candidates: Dict[str, List[Tuple[str, ...]]] = {
        "CICIoT2023": [
            ("Src IP", "Dst IP", "Src Port", "Dst Port"),
            ("src_ip", "dst_ip", "src_port", "dst_port"),
            ("Src IP",), ("src_ip",),
        ],
        "EdgeIIoTset": [
            ("ip.src_host", "ip.dst_host", "tcp.srcport", "tcp.dstport"),
            ("ip.src_host", "ip.dst_host"),
            ("ip.src_host",),
        ],
        "TON_IoT": [
            ("src_ip", "dst_ip", "src_port", "dst_port"),
            ("src_addr", "dst_addr", "src_port", "dst_port"),
            ("src_ip", "dst_ip"), ("src_addr", "dst_addr"), ("src_ip",),
        ],
        "WUSTL_IIoT": [
            ("SrcAddr", "DstAddr", "Sport", "Dport"),
            ("SrcAddr", "DstAddr"), ("SrcAddr",),
        ],
        "X-IIoTID": [
            ("Scr_IP", "Des_IP", "Scr_port", "Des_port"),
            ("Src_IP", "Dst_IP", "src_port", "dst_port"),
            ("Scr_IP", "Des_IP"), ("Src_IP", "Dst_IP"), ("Scr_IP",),
        ],
    }
    by_lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for spec in candidates.get(dataset_name, []):
        cols = []
        ok = True
        for c in spec:
            hit = by_lower.get(c.strip().lower())
            if hit is None:
                ok = False
                break
            cols.append(hit)
        if not ok:
            continue
        key = df[cols].astype(str).fillna("__NA__").agg("|".join, axis=1)
        nunique = int(key.nunique())
        if 10 <= nunique < len(df) * 0.995:
            codes, _ = pd.factorize(key, sort=True)
            return codes.astype(np.int64), "session_group:" + "+".join(cols)
    return None, None


def leakage_resistant_split_indices(raw_df: pd.DataFrame, y: np.ndarray,
                                    dataset_name: str, seed: int,
                                    train_frac: float = 0.70,
                                    val_frac: float = 0.15,
                                    test_frac: float = 0.15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create a stricter time- or group-separated 70/15/15 holdout.

    Time separation is preferred when a usable timestamp exists because it blocks
    temporally adjacent records from crossing the holdout boundary. Otherwise a
    session/device group split is used. If neither is possible, the function raises
    rather than silently falling back to record-level random splitting.
    """
    time_candidates = {
        "CICIoT2023": ("Timestamp", "timestamp", "ts", "time"),
        "EdgeIIoTset": ("frame.time", "Timestamp", "timestamp", "ts"),
        "TON_IoT": ("ts", "timestamp", "Timestamp"),
        "WUSTL_IIoT": ("StartTime", "LastTime", "timestamp", "Timestamp"),
        "X-IIoTID": ("Timestamp", "Date", "timestamp", "date"),
    }
    tvals, tcol = _parse_time_order(raw_df, time_candidates.get(dataset_name, ()))
    if tvals is not None:
        order = np.argsort(tvals, kind="mergesort")
        n = len(order)
        a = int(round(train_frac * n))
        b = int(round((train_frac + val_frac) * n))
        tr, va, te = order[:a], order[a:b], order[b:]
        # Avoid unusable one-class splits; if this occurs, try group separation.
        if all(len(np.unique(y[idx])) == 2 for idx in (tr, va, te)):
            return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64), {
                "mode": "time_separated", "column": tcol,
                "train_frac": len(tr)/n, "val_frac": len(va)/n, "test_frac": len(te)/n,
                "group_overlap": 0,
            }

    groups, gdesc = _session_group_ids(raw_df, dataset_name)
    if groups is None:
        raise RuntimeError(
            f"{dataset_name}: no reliable timestamp or session/device group columns are available "
            "for a leakage-resistant split in this downloaded representation."
        )

    # Search deterministic GroupShuffleSplit candidates and choose the split with
    # the closest size and class balance to the original dataset. Groups never overlap.
    all_idx = np.arange(len(y))
    target_rate = float(np.mean(y))
    best = None
    for attempt in range(64):
        gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed + attempt)
        tr, temp = next(gss.split(all_idx, y, groups=groups))
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[temp])) < 2:
            continue
        temp_groups = groups[temp]
        rel_val = val_frac / (val_frac + test_frac)
        for attempt2 in range(16):
            gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1000 + 31*attempt + attempt2)
            va_rel, te_rel = next(gss2.split(temp, y[temp], groups=temp_groups))
            va, te = temp[va_rel], temp[te_rel]
            if any(len(np.unique(y[idx])) < 2 for idx in (tr, va, te)):
                continue
            size_pen = abs(len(tr)/len(y)-train_frac) + abs(len(va)/len(y)-val_frac) + abs(len(te)/len(y)-test_frac)
            rate_pen = sum(abs(float(np.mean(y[idx])) - target_rate) for idx in (tr, va, te))
            score = size_pen + 0.25 * rate_pen
            if best is None or score < best[0]:
                best = (score, tr.copy(), va.copy(), te.copy())
    if best is None:
        raise RuntimeError(f"{dataset_name}: could not construct a class-valid group-separated split")
    _, tr, va, te = best
    overlap = (
        len(set(groups[tr]).intersection(set(groups[va])))
        + len(set(groups[tr]).intersection(set(groups[te])))
        + len(set(groups[va]).intersection(set(groups[te])))
    )
    return tr.astype(np.int64), va.astype(np.int64), te.astype(np.int64), {
        "mode": "group_separated", "group_definition": gdesc,
        "train_frac": len(tr)/len(y), "val_frac": len(va)/len(y), "test_frac": len(te)/len(y),
        "group_overlap": int(overlap), "n_groups": int(len(np.unique(groups))),
    }

def dirichlet_label_partition(y: np.ndarray, n_clients: int, alpha: float,
                              seed: int, min_samples: int = 64,
                              max_tries: int = 100) -> Dict[int, np.ndarray]:
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

    # Dataset split / heterogeneity
    split_mode: str = DEFAULT_SPLIT_MODE
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

    # Aggregator: rpp, p4p, trustfed, fedavg, median, trimmed_mean, uniform_mean
    aggregator: str = "rpp"
    trim_ratio: float = 0.20

    # RPP-FL parameters — synchronized with Sections 3.4–3.7 of the manuscript.
    update_clip_C: float = 1.0
    trust_sketch_dim: int = 2048
    rpp_scale_floor_frac: float = 1e-3
    rpp_deviation_R0: float = 0.70
    rpp_deviation_tau: float = 0.75
    rpp_byzantine_tolerance: float = 0.40
    rpp_collusion_kappa_max: float = 0.60
    rpp_collusion_chi_min: float = 0.90
    rpp_collusion_singleton_chi_min: float = 1.25
    rpp_collusion_score_flagged: float = 0.05
    rpp_collusion_trust_gate: float = 0.05
    rpp_rep_init: float = 0.90
    rpp_rep_alpha_decay: float = 0.60
    rpp_rep_alpha_recover: float = 0.95
    rpp_rep_collusion_penalty: float = 0.70
    rpp_rep_min: float = 0.005
    rpp_sample_cap_multiplier: float = 1.50
    rpp_reputation_power: float = 2.0
    rpp_trust_power: float = 2.5
    rpp_trust_floor: float = 1e-4
    rpp_winsor_z: float = 2.5
    rpp_anchor_base: float = 0.10
    rpp_anchor_max: float = 0.55
    beta_momentum: float = 0.90

    # RPP component switches for multi-seed ablations.
    use_robust_deviation: bool = True
    use_center_alignment: bool = True
    use_momentum: bool = True
    use_peer_agreement: bool = True
    use_collusion_guard: bool = True
    use_reputation: bool = True
    use_winsorization: bool = True
    use_median_anchor: bool = True

    # Composite trust coefficients from the manuscript.
    lambda_center: float = 0.10
    lambda_history: float = 0.10
    lambda_deviation: float = 0.35
    lambda_peer: float = 0.10
    lambda_collusion: float = 0.35

    # P4P matched reimplementation settings.
    p4p_probe_beta: float = 0.90
    p4p_temporal_beta: float = 0.80
    p4p_norm_z_threshold: float = 3.50
    p4p_instant_threshold: float = 0.60
    p4p_temporal_threshold: float = 0.55
    p4p_min_keep_fraction: float = 0.50
    p4p_iforest_estimators: int = 128

    # TrustFed-IDS-style public-description reimplementation.
    trustfed_temporal_beta: float = 0.80
    trustfed_cosine_weight: float = 0.40
    trustfed_validation_weight: float = 0.40
    trustfed_stability_weight: float = 0.20
    trustfed_min_trust: float = 0.05
    trustfed_val_max_samples: int = 4096

    # Attack model
    frac_malicious: float = 0.20
    attack_type: str = "alie"
    signflip_scale: float = 1.0
    scale_attack: float = 4.0
    gaussian_attack_std: float = 1.0
    alie_z: float = 1.5
    # White-box adaptive coordinated attack: seeks high RPP trust while maximizing
    # deviation within the manuscript's robust-deviation tolerance.
    adaptive_rpp_max_z: float = 0.68
    adaptive_rpp_min_center_cos: float = 0.85
    adaptive_rpp_min_history_cos: float = 0.50
    adaptive_rpp_jitter: float = 0.005

    # Differential privacy: separate aggregate-release mode only.
    # No sampling amplification is claimed; the RDP accountant composes every
    # released round at sample_rate=1.0, which is conservative for the fixed-size
    # 8/10 client sampling used by the simulator.
    privacy_mode: str = "none"  # none | central_dp
    dp_noise_multiplier: float = 0.0
    dp_delta: float = 1e-5
    dp_target_epsilon: Optional[float] = None
    dp_accountant_sample_rate: float = 1.0

    # Optimization / evaluation
    early_stop_patience: int = 6
    min_rounds_before_stop: int = 8
    enable_early_stopping: bool = False  # matched revision runs use fixed budget
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



def build_adaptive_rpp_coordinated_attack(benign_stack: torch.Tensor, server,
                                           cfg: Config,
                                           generator: torch.Generator) -> torch.Tensor:
    """White-box stress attack that explicitly targets RPP instantaneous trust.

    The attacker is granted the stronger knowledge of the benign updates selected
    in the current round and the public RPP screening rule. It searches for a
    coordinated update that remains inside the robust-deviation tolerance and
    maintains center/momentum cosine constraints while maximizing distance from
    the benign mean. This is deliberately stronger than ALIE and is reported as a
    white-box stress test, not as a realistic black-box attacker assumption.
    """
    if benign_stack is None or benign_stack.shape[0] < 2:
        return torch.zeros_like(server.W)
    B = benign_stack.detach().cpu()
    # Respect the same clipping radius that RPP will apply.
    Bc = torch.stack([clip_vector_l2(x, cfg.update_clip_C) for x in B], dim=0)
    full_center, full_rho, _ = server._robust_center_scale(Bc)
    mean = Bc.mean(dim=0)
    hist = server.V.detach().cpu()
    hist_valid = float(hist.norm()) > 1e-12

    best = full_center.clone()
    best_obj = -float("inf")
    # Candidate family lies at/under the R0 tolerance per coordinate. Search
    # several signed directions; deterministic generator keeps runs reproducible.
    for trial in range(64):
        sign = torch.sign(torch.randn(full_center.shape, generator=generator))
        z = cfg.adaptive_rpp_max_z * (0.55 + 0.45 * (trial / 63.0))
        cand = full_center + z * full_rho * sign
        cand = clip_vector_l2(cand, cfg.update_clip_C)
        sk = server._sketch(cand)
        csk = server._sketch(full_center)
        ccos = cosine(sk, csk)
        hcos = cosine(sk, server._sketch(hist)) if hist_valid else 1.0
        # Explicit constraints against the public center/history trust signals.
        if ccos < cfg.adaptive_rpp_min_center_cos:
            continue
        if hist_valid and hcos < cfg.adaptive_rpp_min_history_cos:
            continue
        # Verify the robust-deviation statistic used by the defense.
        Bsk = torch.stack([server._sketch(x) for x in Bc], dim=0)
        c, rho, _ = server._robust_center_scale(Bsk)
        R = float(torch.median(torch.clamp(torch.abs(sk-c)/rho, 0.0, 20.0)).item())
        if R > cfg.rpp_deviation_R0 + 1e-9:
            continue
        # Maximize deviation from the benign mean while remaining trusted.
        obj = float((cand - mean).norm().item())
        if obj > best_obj:
            best_obj = obj
            best = cand.clone()
    return best


def attack_protocol_manifest(cfg: Config) -> Dict[str, Any]:
    return {
        "attacker_selection": "A fixed malicious subset is sampled once from the full federation using the run seed; identities persist across rounds.",
        "poisoning_frequency": "Every malicious client poisons its update in every round in which it is selected.",
        "participation": f"{cfg.n_clients} total clients; fixed-size {int(round(cfg.clients_per_round_frac*cfg.n_clients))}/{cfg.n_clients} selected per round.",
        "reported_malicious_fraction": "Configured fraction refers to the full federation. Realized selected-client malicious fraction is logged separately every round.",
        "attacker_knowledge": {
            "signflip": "local update only",
            "scale": "local update only",
            "gaussian": "local update only plus configured noise distribution",
            "alie": "strong current-round knowledge: benign update coordinate mean/std in the simulator",
            "adaptive_rpp": "white-box stress test: benign selected updates plus public RPP trust rule and momentum state",
        },
        "formulas": {
            "signflip": "Delta_adv = -s * Delta_local",
            "scale": "Delta_adv = a * Delta_local",
            "gaussian": "Delta_adv = Delta_local + sigma_g * std(Delta_local) * N(0,I)",
            "alie": "Delta_adv = mean(Delta_benign) - z * std(Delta_benign)",
            "adaptive_rpp": "coordinated projected candidate constrained by R<=R0 and minimum center/history cosine; maximizes distance from benign mean",
        },
        "parameters": {
            "signflip_scale": cfg.signflip_scale,
            "scale_attack": cfg.scale_attack,
            "gaussian_attack_std": cfg.gaussian_attack_std,
            "alie_z": cfg.alie_z,
            "adaptive_rpp_max_z": cfg.adaptive_rpp_max_z,
            "adaptive_rpp_min_center_cos": cfg.adaptive_rpp_min_center_cos,
            "adaptive_rpp_min_history_cos": cfg.adaptive_rpp_min_history_cos,
        },
    }


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
        return float(self.accountant.get_epsilon(delta=self.cfg.dp_delta, alphas=RDP_ORDERS))

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
    return float(acc.get_epsilon(delta=float(delta), alphas=RDP_ORDERS))


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
        self.reputation: Dict[int, float] = {i: cfg.rpp_rep_init for i in range(n_clients)}

        # P4P state
        self.p4p_probe = torch.zeros_like(self.W)
        self.p4p_suspicion: Dict[int, float] = {i: 0.0 for i in range(n_clients)}
        self.p4p_round = 0

        # TrustFed-IDS-style state
        self.trustfed_trust: Dict[int, float] = {i: 1.0 for i in range(n_clients)}
        self.trustfed_prev_sketch: Dict[int, Optional[torch.Tensor]] = {i: None for i in range(n_clients)}
        self._val_X: Optional[np.ndarray] = None
        self._val_y: Optional[np.ndarray] = None

        self.last_agg_diagnostics: Dict[str, Any] = {}
        rng = np.random.default_rng(cfg.seed + 2027)
        k = min(cfg.trust_sketch_dim, n_params)
        self.sketch_idx = torch.tensor(
            np.sort(rng.choice(n_params, size=k, replace=False)), dtype=torch.long
        )

    def set_validation_reference(self, X: np.ndarray, y: np.ndarray) -> None:
        # Used only by TrustFed-IDS-style comparator. RPP/P4P do not consume
        # validation labels during aggregation.
        n = min(len(X), int(self.cfg.trustfed_val_max_samples))
        if n < len(X):
            rng = np.random.default_rng(self.cfg.seed + 5511)
            idx = np.sort(rng.choice(len(X), size=n, replace=False))
            self._val_X = np.asarray(X[idx], dtype=np.float32)
            self._val_y = np.asarray(y[idx], dtype=np.int64)
        else:
            self._val_X = np.asarray(X, dtype=np.float32)
            self._val_y = np.asarray(y, dtype=np.int64)

    def _sketch(self, g: torch.Tensor) -> torch.Tensor:
        return g[self.sketch_idx]

    def _prepare_updates(self, updates: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        prepared: Dict[int, torch.Tensor] = {}
        for cid, g in updates.items():
            # RPP clipping is part of the proposed defense. The separate privacy
            # branch also requires clipping for its client-level sensitivity bound.
            if self.cfg.aggregator == "rpp" or self.cfg.privacy_mode == "central_dp":
                prepared[cid] = clip_vector_l2(g, self.cfg.update_clip_C)
            else:
                prepared[cid] = g.clone()
        return prepared

    @staticmethod
    def _rank01(values: np.ndarray) -> np.ndarray:
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

    def _robust_center_scale(self, stack: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        center = torch.median(stack, dim=0).values
        mad = torch.median(torch.abs(stack - center), dim=0).values
        base = 1.4826 * mad
        nz = base[base > 1e-12]
        med_nonzero = float(torch.median(nz).item()) if nz.numel() else 1e-6
        phi = max(1e-8, self.cfg.rpp_scale_floor_frac * med_nonzero)
        rho = base + phi
        return center, rho, phi

    @staticmethod
    def _rms_distance(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean((a - b) ** 2) + 1e-12).item())

    @staticmethod
    def _median_pairwise_distance(Z: torch.Tensor, idx: List[int]) -> float:
        if len(idx) <= 1:
            return 0.0
        vals = []
        for ii in range(len(idx)):
            for jj in range(ii + 1, len(idx)):
                vals.append(Server._rms_distance(Z[idx[ii]], Z[idx[jj]]))
        return float(np.median(vals)) if vals else 0.0

    def _collusion_guard(self, sketches: torch.Tensor, center: torch.Tensor,
                         rho: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        m = sketches.shape[0]
        flags = np.zeros(m, dtype=bool)
        scores = np.ones(m, dtype=np.float64)
        diag = {"RPPKappa": float("nan"), "RPPChi": float("nan"), "RPPFlagged": 0}
        if (not self.cfg.use_collusion_guard) or m < 3:
            return flags, scores, diag

        Z = torch.clamp((sketches - center) / rho, -20.0, 20.0)
        D = np.zeros((m, m), dtype=np.float64)
        for i in range(m):
            for j in range(i + 1, m):
                d = self._rms_distance(Z[i], Z[j])
                D[i, j] = D[j, i] = d
        a, b = np.unravel_index(np.argmax(D), D.shape)
        if a == b:
            return flags, scores, diag
        g1, g2 = [], []
        for i in range(m):
            if D[i, a] <= D[i, b]:
                g1.append(i)
            else:
                g2.append(i)
        if not g1 or not g2:
            return flags, scores, diag
        # Deterministic smaller group; in a tie select the group with lower index seed.
        if len(g1) <= len(g2):
            gs, gl = g1, g2
        else:
            gs, gl = g2, g1
        max_minor = int(math.ceil(self.cfg.rpp_byzantine_tolerance * m))
        if len(gs) > max_minor:
            return flags, scores, diag

        ds = self._median_pairwise_distance(Z, gs)
        dl = self._median_pairwise_distance(Z, gl)
        cross = [D[i, j] for i in gs for j in gl]
        dsl = float(np.median(cross)) if cross else 0.0
        eps = 1e-12
        kappa = ds / (dl + eps)
        chi = dsl / (dl + eps)
        flagged = False
        if len(gs) == 1:
            flagged = chi >= self.cfg.rpp_collusion_singleton_chi_min
        else:
            flagged = (
                kappa <= self.cfg.rpp_collusion_kappa_max
                and chi >= self.cfg.rpp_collusion_chi_min
            )
        if flagged:
            flags[gs] = True
            scores[gs] = self.cfg.rpp_collusion_score_flagged
        diag = {"RPPKappa": float(kappa), "RPPChi": float(chi), "RPPFlagged": int(flags.sum())}
        return flags, scores, diag

    def _aggregate_rpp(self, updates: Dict[int, torch.Tensor],
                       sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        """Exact V6 implementation of manuscript Sections 3.4–3.7."""
        cids = list(updates.keys())
        stack = torch.stack([updates[c].detach().cpu() for c in cids], dim=0)
        sketches = torch.stack([self._sketch(updates[c].detach().cpu()) for c in cids], dim=0)
        m = len(cids)
        center, rho, phi = self._robust_center_scale(sketches)

        # D_i: robust coordinate-deviation trust.
        standardized = torch.clamp(torch.abs(sketches - center) / rho, 0.0, 20.0)
        R = torch.median(standardized, dim=1).values.numpy()
        Dscore = np.exp(-np.maximum(R - self.cfg.rpp_deviation_R0, 0.0) / self.cfg.rpp_deviation_tau)

        # C_i and H_i: current-center and historical-momentum consistency.
        Cscore = np.asarray([max(0.0, cosine(s, center)) for s in sketches], dtype=np.float64)
        hist = self._sketch(self.V)
        hist_valid = float(hist.norm()) > 1e-12
        Hscore = np.asarray([
            max(0.0, cosine(s, hist)) if hist_valid else Cscore[i]
            for i, s in enumerate(sketches)
        ], dtype=np.float64)

        # P_i: median pairwise cosine agreement mapped from [-1,1] to [0,1].
        Pscore = np.ones(m, dtype=np.float64)
        if m > 1:
            for i in range(m):
                sims = [cosine(sketches[i], sketches[j]) for j in range(m) if j != i]
                Pscore[i] = float(np.clip((1.0 + float(np.median(sims))) / 2.0, 0.0, 1.0))

        flags, Gscore, cdiag = self._collusion_guard(sketches, center, rho)

        components = [
            (self.cfg.lambda_center if self.cfg.use_center_alignment else 0.0, Cscore),
            (self.cfg.lambda_history if self.cfg.use_momentum else 0.0, Hscore),
            (self.cfg.lambda_deviation if self.cfg.use_robust_deviation else 0.0, Dscore),
            (self.cfg.lambda_peer if self.cfg.use_peer_agreement else 0.0, Pscore),
            (self.cfg.lambda_collusion if self.cfg.use_collusion_guard else 0.0, Gscore),
        ]
        denom = sum(w for w, _ in components)
        if denom <= 0:
            trust_arr = np.ones(m, dtype=np.float64)
        else:
            trust_arr = sum(w * v for w, v in components) / denom
        if self.cfg.use_collusion_guard:
            trust_arr[flags] *= self.cfg.rpp_collusion_trust_gate
        trust_arr = np.clip(trust_arr, 0.0, 1.0)
        trust = {cid: float(trust_arr[i]) for i, cid in enumerate(cids)}

        # Asymmetric persistent reputation: fast punishment, slow recovery.
        for i, cid in enumerate(cids):
            if self.cfg.use_reputation:
                old = float(self.reputation.get(cid, self.cfg.rpp_rep_init))
                alpha = self.cfg.rpp_rep_alpha_decay if trust_arr[i] < old else self.cfg.rpp_rep_alpha_recover
                rep = alpha * old + (1.0 - alpha) * float(trust_arr[i])
                if flags[i] and self.cfg.use_collusion_guard:
                    rep *= self.cfg.rpp_rep_collusion_penalty
                self.reputation[cid] = float(np.clip(rep, self.cfg.rpp_rep_min, 1.0))
            else:
                self.reputation[cid] = 1.0

        counts = np.asarray([max(sample_counts[c], 1) for c in cids], dtype=np.float64)
        med_n = float(np.median(counts))
        capped = np.minimum(counts, self.cfg.rpp_sample_cap_multiplier * med_n)
        q = np.sqrt(capped)
        if self.cfg.use_reputation:
            q *= np.asarray([self.reputation[c] for c in cids], dtype=np.float64) ** self.cfg.rpp_reputation_power
        q *= np.maximum(trust_arr, self.cfg.rpp_trust_floor) ** self.cfg.rpp_trust_power
        if (not np.isfinite(q).all()) or q.sum() <= 0:
            q = np.ones(m, dtype=np.float64)
        w = q / q.sum()
        weights = {cid: float(w[i]) for i, cid in enumerate(cids)}

        # Full-dimensional winsorization around coordinate median ±2.5 robust scales.
        full_center, full_rho, _ = self._robust_center_scale(stack)
        if self.cfg.use_winsorization:
            lo = full_center - self.cfg.rpp_winsor_z * full_rho
            hi = full_center + self.cfg.rpp_winsor_z * full_rho
            bounded = torch.maximum(torch.minimum(stack, hi), lo)
        else:
            bounded = stack
        robust_mean = torch.sum(bounded * torch.tensor(w, dtype=bounded.dtype).view(-1, 1), dim=0)

        flagged_fraction = float(flags.mean()) if self.cfg.use_collusion_guard else 0.0
        anchor = 0.0
        if self.cfg.use_median_anchor:
            anchor = float(np.clip(self.cfg.rpp_anchor_base + flagged_fraction, 0.0, self.cfg.rpp_anchor_max))
        delta = (1.0 - anchor) * robust_mean + anchor * full_center

        self.last_agg_diagnostics = {
            **cdiag,
            "RPPScaleFloorPhi": float(phi),
            "RPPMeanDeviationR": float(np.mean(R)),
            "RPPMeanDeviationTrust": float(np.mean(Dscore)),
            "RPPMeanCenterTrust": float(np.mean(Cscore)),
            "RPPMeanHistoryTrust": float(np.mean(Hscore)),
            "RPPMeanPeerTrust": float(np.mean(Pscore)),
            "RPPMeanCollusionTrust": float(np.mean(Gscore)),
            "RPPFlaggedFraction": flagged_fraction,
            "RPPAnchorCoeff": anchor,
        }
        return self._apply_central_dp(delta, weights), trust, weights

    def _aggregate_p4p(self, updates: Dict[int, torch.Tensor],
                       sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        """Independent matched-protocol implementation of published P4P components.

        Publicly documented components: historical global-trajectory probe,
        server-computed cosine response, L2-norm filtering, ensemble anomaly
        detection, temporal suspicion scoring, and filtered FedAvg.
        """
        cids = list(updates.keys())
        stack = torch.stack([updates[c].detach().cpu() for c in cids], dim=0)
        m = len(cids)
        if m == 1:
            c = cids[0]
            return updates[c].clone(), {c: 1.0}, {c: 1.0}
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
        center_dist = np.asarray([float((x - center).norm()) for x in stack], dtype=np.float64)
        log_norm_ratio = np.log((norms + 1e-12) / (med_norm + 1e-12))
        features = np.column_stack([probe_cos, center_cos, log_norm_ratio, center_dist])
        Z = self._robust_standardize_matrix(features)
        if m >= 4:
            iso = IsolationForest(
                n_estimators=max(int(self.cfg.p4p_iforest_estimators), 32),
                contamination="auto", random_state=self.cfg.seed + 4000 + self.p4p_round, n_jobs=1,
            )
            iso.fit(Z)
            iso_score = self._rank01(-iso.decision_function(Z))
            lof = LocalOutlierFactor(n_neighbors=max(2, min(5, m - 1)), contamination="auto")
            lof.fit_predict(Z)
            lof_score = self._rank01(-lof.negative_outlier_factor_)
        else:
            iso_score = self._rank01(center_dist)
            lof_score = self._rank01(norm_z)
        instant = np.clip(0.35*probe_anom + 0.25*norm_anom + 0.20*iso_score + 0.20*lof_score, 0.0, 1.0)
        temporal = np.zeros(m, dtype=np.float64)
        effective = np.zeros(m, dtype=np.float64)
        for i, cid in enumerate(cids):
            old = float(self.p4p_suspicion.get(cid, 0.0))
            new = self.cfg.p4p_temporal_beta*old + (1-self.cfg.p4p_temporal_beta)*float(instant[i])
            self.p4p_suspicion[cid] = float(np.clip(new, 0.0, 1.0))
            temporal[i] = self.p4p_suspicion[cid]
            effective[i] = max(float(instant[i]), temporal[i])
        reject = norm_flag | (instant >= self.cfg.p4p_instant_threshold) | (temporal >= self.cfg.p4p_temporal_threshold)
        keep = ~reject
        min_keep = max(1, int(math.ceil(self.cfg.p4p_min_keep_fraction * m)))
        if int(keep.sum()) < min_keep:
            order = np.argsort(effective, kind="mergesort")
            keep = np.zeros(m, dtype=bool); keep[order[:min_keep]] = True
        kept = [c for c, k in zip(cids, keep.tolist()) if k]
        ns = np.asarray([max(sample_counts[c], 1) for c in kept], dtype=np.float64)
        wk = ns / ns.sum()
        delta = torch.zeros_like(self.W)
        weights = {c: 0.0 for c in cids}
        for cid, wi in zip(kept, wk.tolist()):
            delta += updates[cid] * float(wi); weights[cid] = float(wi)
        trust = {cid: float(np.clip(1.0 - effective[i], 0.0, 1.0)) for i, cid in enumerate(cids)}
        self.last_agg_diagnostics = {
            "P4PKept": len(kept), "P4PRejected": int(m-len(kept)),
            "P4PMeanInstantSuspicion": float(np.mean(instant)),
            "P4PMeanTemporalSuspicion": float(np.mean(temporal)),
            "P4PMaxNormZ": float(np.max(norm_z)), "P4PMeanProbeCosine": float(np.mean(probe_cos)),
        }
        return delta, trust, weights

    @torch.no_grad()
    def _validation_loss_for_update(self, delta: torch.Tensor) -> float:
        if self._val_X is None or self._val_y is None or len(self._val_X) == 0:
            return float("nan")
        original = self.W.clone()
        try:
            set_params_from_vector(self.model, original + delta.detach().cpu())
            self.model.eval()
            total, n = 0.0, 0
            for st in range(0, len(self._val_X), 2048):
                xb = torch.from_numpy(self._val_X[st:st+2048]).to(DEVICE)
                yb = torch.from_numpy(self._val_y[st:st+2048].astype(np.float32)).to(DEVICE)
                logits, _ = self.model(xb)
                total += float(F.binary_cross_entropy_with_logits(logits, yb, reduction="sum").item())
                n += len(yb)
            return total / max(n, 1)
        finally:
            set_params_from_vector(self.model, original)

    def _aggregate_trustfed(self, updates: Dict[int, torch.Tensor],
                            sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        """TrustFed-IDS-style matched reimplementation from public method description.

        It uses the three publicly stated trust signals: cosine similarity,
        normalized server validation loss, and historical update stability, then
        performs trust-weighted FedAvg-T. Hyperparameter instantiation is explicit
        in Config and is not claimed to reproduce unpublished author code.
        """
        cids = list(updates.keys())
        sketches = torch.stack([self._sketch(updates[c].detach().cpu()) for c in cids], dim=0)
        center = torch.median(sketches, dim=0).values
        cos_s = np.asarray([(1.0 + cosine(sketches[i], center))/2.0 for i in range(len(cids))], dtype=np.float64)
        losses = np.asarray([self._validation_loss_for_update(updates[c]) for c in cids], dtype=np.float64)
        if np.isfinite(losses).all() and float(np.ptp(losses)) > 1e-12:
            val_s = 1.0 - (losses - losses.min()) / (losses.max() - losses.min())
        else:
            val_s = np.ones(len(cids), dtype=np.float64)
        stab_s = np.ones(len(cids), dtype=np.float64)
        for i, cid in enumerate(cids):
            prev = self.trustfed_prev_sketch.get(cid)
            if isinstance(prev, torch.Tensor) and float(prev.norm()) > 1e-12:
                stab_s[i] = float(np.clip((1.0 + cosine(sketches[i], prev))/2.0, 0.0, 1.0))
        wsum = self.cfg.trustfed_cosine_weight + self.cfg.trustfed_validation_weight + self.cfg.trustfed_stability_weight
        instant = (
            self.cfg.trustfed_cosine_weight*cos_s
            + self.cfg.trustfed_validation_weight*val_s
            + self.cfg.trustfed_stability_weight*stab_s
        ) / max(wsum, 1e-12)
        trust_arr = np.zeros(len(cids), dtype=np.float64)
        for i, cid in enumerate(cids):
            old = float(self.trustfed_trust.get(cid, 1.0))
            new = self.cfg.trustfed_temporal_beta*old + (1-self.cfg.trustfed_temporal_beta)*float(instant[i])
            self.trustfed_trust[cid] = float(np.clip(new, self.cfg.trustfed_min_trust, 1.0))
            self.trustfed_prev_sketch[cid] = sketches[i].clone()
            trust_arr[i] = self.trustfed_trust[cid]
        ns = np.asarray([max(sample_counts[c], 1) for c in cids], dtype=np.float64)
        raw = ns * np.maximum(trust_arr, self.cfg.trustfed_min_trust)
        raw /= raw.sum()
        delta = torch.zeros_like(self.W)
        weights = {}
        for cid, wi in zip(cids, raw.tolist()):
            delta += updates[cid] * float(wi); weights[cid] = float(wi)
        trust = {cid: float(trust_arr[i]) for i, cid in enumerate(cids)}
        self.last_agg_diagnostics = {
            "TrustFedMeanCosine": float(np.mean(cos_s)),
            "TrustFedMeanValidationScore": float(np.mean(val_s)),
            "TrustFedMeanStability": float(np.mean(stab_s)),
            "TrustFedMeanTrust": float(np.mean(trust_arr)),
        }
        return delta, trust, weights

    def aggregate(self, updates: Dict[int, torch.Tensor], sample_counts: Dict[int, int]) -> Tuple[torch.Tensor, Dict[int, float], Dict[int, float]]:
        updates = self._prepare_updates(updates)
        cids = list(updates.keys())
        if not cids:
            raise ValueError("No client updates to aggregate")
        self.last_agg_diagnostics = {}
        if self.cfg.aggregator == "rpp":
            return self._aggregate_rpp(updates, sample_counts)
        if self.cfg.aggregator == "p4p":
            return self._aggregate_p4p(updates, sample_counts)
        if self.cfg.aggregator == "trustfed":
            return self._aggregate_trustfed(updates, sample_counts)

        stack = torch.stack([updates[c] for c in cids], dim=0)
        if self.cfg.aggregator == "uniform_mean":
            delta = stack.mean(dim=0)
            trust = {c: 1.0 for c in cids}
            weights = {c: 1.0/len(cids) for c in cids}
            return self._apply_central_dp(delta, weights), trust, weights
        if self.cfg.aggregator == "fedavg":
            ns = np.array([sample_counts[c] for c in cids], dtype=np.float64)
            w = ns / ns.sum()
            delta = sum(updates[c] * float(wi) for c, wi in zip(cids, w))
            trust = {c: 1.0 for c in cids}
            weights = {c: float(wi) for c, wi in zip(cids, w)}
            return self._apply_central_dp(delta, weights), trust, weights
        if self.cfg.aggregator == "median":
            delta = torch.median(stack, dim=0).values
            trust = {c: 1.0 for c in cids}; weights = {c: 1.0/len(cids) for c in cids}
            return delta, trust, weights
        if self.cfg.aggregator == "trimmed_mean":
            m = stack.shape[0]; trim = int(math.floor(self.cfg.trim_ratio*m))
            sorted_vals, _ = torch.sort(stack, dim=0)
            core = sorted_vals[trim:m-trim] if 2*trim < m else sorted_vals
            delta = core.mean(dim=0)
            trust = {c: 1.0 for c in cids}; weights = {c: 1.0/len(cids) for c in cids}
            return delta, trust, weights
        raise ValueError(f"Unknown aggregator: {self.cfg.aggregator}")

    def _apply_central_dp(self, delta: torch.Tensor, weights: Dict[int, float]) -> torch.Tensor:
        if self.cfg.privacy_mode != "central_dp" or self.cfg.dp_noise_multiplier <= 0:
            return delta
        # For the privacy branch, uniform_mean is required. With m selected clipped
        # clients and add/remove adjacency, sensitivity is C/m as stated in the paper.
        max_w = max(weights.values()) if weights else 1.0
        sensitivity = self.cfg.update_clip_C * max_w
        std = self.cfg.dp_noise_multiplier * sensitivity
        return delta + torch.randn_like(delta) * std

    def apply_update(self, delta: torch.Tensor) -> None:
        self.W = self.W + self.cfg.lr_global * delta
        self.V = self.cfg.beta_momentum * self.V + (1.0 - self.cfg.beta_momentum) * delta
        if self.cfg.aggregator == "p4p":
            self.p4p_probe = self.cfg.p4p_probe_beta*self.p4p_probe + (1-self.cfg.p4p_probe_beta)*delta.detach().cpu()
            self.p4p_round += 1
        set_params_from_vector(self.model, self.W)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        self.model.eval()
        out = np.empty(len(X), dtype=np.float32)
        for st in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[st:st+batch_size]).to(DEVICE)
            logits, _ = self.model(xb)
            out[st:st+len(xb)] = torch.sigmoid(logits).detach().cpu().numpy()
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
    source_rows_before_cap: int
    loaded_rows_after_cap: int
    rows_after_dedup: int
    split_mode: str
    split_meta: Dict[str, Any]
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    selected_feature_names: List[str]
    selected_feature_indices: np.ndarray
    preprocessing_audit: Dict[str, Any]


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


def _fast_csv_row_count(path: str) -> int:
    # Network-traffic CSVs used here do not contain embedded multiline records.
    # Binary newline counting avoids loading tens of millions of rows into RAM.
    n = 0
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            n += block.count(b"\n")
    return max(n - 1, 0)  # header


def dataset_total_source_rows(path: str) -> int:
    total = 0
    for f in _tabular_files(path):
        ext = Path(f).suffix.lower()
        try:
            if ext == ".csv":
                total += _fast_csv_row_count(f)
            elif ext in {".parquet", ".pq"}:
                try:
                    import pyarrow.parquet as pq  # type: ignore
                    total += int(pq.ParquetFile(f).metadata.num_rows)
                except Exception:
                    total += int(len(pd.read_parquet(f)))
            elif ext in {".feather", ".ft"}:
                total += int(len(pd.read_feather(f)))
        except Exception as e:
            print(f"[WARN] Could not count source rows in {f}: {e}")
    return int(total)


def prepared_cache_key(dataset_name: str, path: str, cfg: Config) -> str:
    return stable_hash({
        "dataset": dataset_name,
        "source_signature": dataset_source_signature(path),
        "seed": cfg.seed,
        "split_mode": cfg.split_mode,
        "max_rows": MAX_ROWS_PER_DATASET,
        "max_features": cfg.max_model_features,
        "preprocess_version": 6,
    })


def save_prepared_cache(data: PreparedData, cache_file: str, meta_file: str) -> None:
    Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
    local_tmp = os.path.join(tempfile.gettempdir(), f"rpp_{data.dataset}_{os.getpid()}.npz")
    np.savez_compressed(
        local_tmp,
        X_train=data.X_train, y_train=data.y_train,
        X_val=data.X_val, y_val=data.y_val,
        X_test=data.X_test, y_test=data.y_test,
        train_indices=data.train_indices, val_indices=data.val_indices, test_indices=data.test_indices,
        selected_feature_indices=data.selected_feature_indices,
    )
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
        "source_rows_before_cap": data.source_rows_before_cap,
        "loaded_rows_after_cap": data.loaded_rows_after_cap,
        "rows_after_dedup": data.rows_after_dedup,
        "split_mode": data.split_mode,
        "split_meta": data.split_meta,
        "selected_feature_names": data.selected_feature_names,
        "preprocessing_audit": data.preprocessing_audit,
        "cache_file": os.path.basename(cache_file),
        "status": "ready",
    }, meta_file)


def load_prepared_cache(cache_file: str, meta_file: str) -> Optional[PreparedData]:
    if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
        return None
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            m = json.load(f)
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
            source_rows_before_cap=int(m.get("source_rows_before_cap", m["raw_rows"])),
            loaded_rows_after_cap=int(m.get("loaded_rows_after_cap", m["raw_rows"])),
            rows_after_dedup=int(m.get("rows_after_dedup", m["raw_rows"])),
            split_mode=str(m.get("split_mode", DEFAULT_SPLIT_MODE)),
            split_meta=dict(m.get("split_meta", {})),
            train_indices=z["train_indices"].astype(np.int64),
            val_indices=z["val_indices"].astype(np.int64),
            test_indices=z["test_indices"].astype(np.int64),
            selected_feature_names=list(m.get("selected_feature_names", [])),
            selected_feature_indices=z["selected_feature_indices"].astype(np.int64),
            preprocessing_audit=dict(m.get("preprocessing_audit", {})),
        )
    except Exception as e:
        print(f"[WARN] Prepared cache invalid; rebuilding: {e}")
        return None


def _write_preprocessing_audit(data: PreparedData, cfg: Config, path: str) -> None:
    out_dir = os.path.join(RESULT_ROOT, "REPRODUCIBILITY", data.dataset,
                           f"split_{data.split_mode}", f"seed_{cfg.seed}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    atomic_json_dump({
        "dataset": data.dataset,
        "source_path": path,
        "source_rows_before_300k_cap": data.source_rows_before_cap,
        "rows_loaded_after_cap": data.loaded_rows_after_cap,
        "rows_after_leakage_safe_duplicate_removal": data.rows_after_dedup,
        "split_mode": data.split_mode,
        "split_meta": data.split_meta,
        "train_rows": int(len(data.y_train)),
        "validation_rows": int(len(data.y_val)),
        "test_rows": int(len(data.y_test)),
        "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(data.y_train, return_counts=True))},
        "validation_class_counts": {str(k): int(v) for k, v in zip(*np.unique(data.y_val, return_counts=True))},
        "test_class_counts": {str(k): int(v) for k, v in zip(*np.unique(data.y_test, return_counts=True))},
        "label_column": data.label_col,
        "raw_predictive_features": data.n_raw_features,
        "preprocessed_features": data.n_preprocessed_features,
        "selected_features": data.n_selected_features,
        "selected_feature_names": data.selected_feature_names,
        "preprocessing": data.preprocessing_audit,
        "seed": cfg.seed,
        "max_rows_per_dataset": MAX_ROWS_PER_DATASET,
    }, os.path.join(out_dir, "preprocessing_manifest.json"))
    np.savez_compressed(
        os.path.join(out_dir, "split_indices_and_feature_mask.npz"),
        train_indices=data.train_indices,
        validation_indices=data.val_indices,
        test_indices=data.test_indices,
        selected_feature_indices=data.selected_feature_indices,
    )
    pd.DataFrame({
        "selected_index": np.arange(len(data.selected_feature_names)),
        "feature_name": data.selected_feature_names,
    }).to_csv(os.path.join(out_dir, "selected_features.csv"), index=False)


def prepare_dataset(dataset_name: str, path: str, cfg: Config) -> PreparedData:
    spec = DATASET_SPECS[dataset_name]
    print_header(f"PREPARING DATASET: {dataset_name} | split={cfg.split_mode} | seed={cfg.seed}")
    print(f"Path: {path}")

    source_rows = dataset_total_source_rows(path)
    df = load_dataset_path(path, MAX_ROWS_PER_DATASET, cfg.seed)
    loaded_rows = len(df)
    y, label_col = make_binary_labels(df, spec)
    class_counts_loaded = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"Rows loaded: {len(df):,} / source={source_rows:,} | attack rate: {y.mean():.2%} | label: {label_col}")

    # Preserve raw time/session identifiers only for split construction. They are
    # never sent to the model. Duplicate removal is based solely on retained
    # predictive features + target, so rows differing only by IDs/timestamps collapse.
    raw_for_split = df.reset_index(drop=True).copy()
    feature_df = drop_known_leakage(df, spec, label_col).reset_index(drop=True)
    n_raw_features = feature_df.shape[1]
    tmp = feature_df.copy()
    tmp["__RPP_LABEL__"] = y
    before = len(tmp)
    keep_mask = ~tmp.duplicated(keep="first")
    tmp = tmp.loc[keep_mask].reset_index(drop=True)
    raw_for_split = raw_for_split.loc[keep_mask.to_numpy()].reset_index(drop=True)
    y = tmp.pop("__RPP_LABEL__").to_numpy(dtype=np.int64)
    feature_df = tmp
    class_counts_after_dedup = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    removed = before - len(feature_df)
    print(f"Leakage-safe dedup: retained={len(feature_df):,} | removed={removed:,} ({removed/max(before,1):.2%})")

    if cfg.split_mode == STRICT_SPLIT_MODE:
        tr_idx, va_idx, te_idx, split_meta = leakage_resistant_split_indices(
            raw_for_split, y, dataset_name, cfg.seed
        )
    elif cfg.split_mode == DEFAULT_SPLIT_MODE:
        tr_idx, va_idx, te_idx = stratified_train_val_test_indices(y, cfg.seed)
        split_meta = {
            "mode": "record_stratified",
            "train_frac": len(tr_idx)/len(y), "val_frac": len(va_idx)/len(y), "test_frac": len(te_idx)/len(y),
            "note": "Original record-level stratified split; provided only as the within-dataset reference protocol.",
        }
    else:
        raise ValueError(f"Unknown split_mode={cfg.split_mode}")

    df_tr = feature_df.iloc[tr_idx].reset_index(drop=True)
    df_va = feature_df.iloc[va_idx].reset_index(drop=True)
    df_te = feature_df.iloc[te_idx].reset_index(drop=True)
    y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]

    prep = LeakageSafePreprocessor(onehot_max_unique=20, clip_q_low=0.001, clip_q_high=0.999)
    X_tr = prep.fit_transform(df_tr)
    X_va = prep.transform(df_va)
    X_te = prep.transform(df_te)
    n_pre = X_tr.shape[1]

    selector = TrainOnlyFeatureSelector(max_features=cfg.max_model_features)
    selector.fit(X_tr, y_tr)
    selected_idx = np.asarray(selector.indices_, dtype=np.int64)
    selected_names = [prep.feature_names_[int(i)] for i in selected_idx]
    X_tr = selector.transform(X_tr)
    X_va = selector.transform(X_va)
    X_te = selector.transform(X_te)

    X_tr = np.clip(X_tr, -8.0, 8.0).astype(np.float32)
    X_va = np.clip(X_va, -8.0, 8.0).astype(np.float32)
    X_te = np.clip(X_te, -8.0, 8.0).astype(np.float32)

    preprocessing_audit = {
        "fit_scope": "training partition only",
        "class_counts_after_binary_mapping_before_dedup": class_counts_loaded,
        "class_counts_after_dedup": class_counts_after_dedup,
        "split_index_semantics": "indices refer to the capped, leakage-column-removed and deduplicated dataset used by this run",
        "numeric_rule": "median imputation -> train 0.001/0.999 quantile clipping -> train z-score",
        "categorical_rule": "one-hot when <=20 train levels; otherwise train-fitted frequency encoding",
        "feature_selection": "SelectKBest(f_classif) fit on training only; at most 192 features",
        "final_numeric_clip": [-8.0, 8.0],
        "onehot_max_unique": 20,
        "quantile_clip_low": 0.001,
        "quantile_clip_high": 0.999,
        "numeric_columns": list(prep.numeric_cols),
        "low_cardinality_categorical_columns": list(prep.lowcat_cols),
        "high_cardinality_categorical_columns": list(prep.highcat_cols),
        "selected_feature_indices": selected_idx.tolist(),
    }

    data = PreparedData(
        dataset=dataset_name,
        X_train=X_tr, y_train=y_tr,
        X_val=X_va, y_val=y_va,
        X_test=X_te, y_test=y_te,
        n_raw_features=n_raw_features,
        n_preprocessed_features=n_pre,
        n_selected_features=X_tr.shape[1],
        label_col=label_col,
        raw_rows=len(feature_df),
        source_rows_before_cap=source_rows,
        loaded_rows_after_cap=loaded_rows,
        rows_after_dedup=len(feature_df),
        split_mode=cfg.split_mode,
        split_meta=split_meta,
        train_indices=np.asarray(tr_idx, dtype=np.int64),
        val_indices=np.asarray(va_idx, dtype=np.int64),
        test_indices=np.asarray(te_idx, dtype=np.int64),
        selected_feature_names=selected_names,
        selected_feature_indices=selected_idx,
        preprocessing_audit=preprocessing_audit,
    )
    print(
        f"Split: train={len(X_tr):,}, val={len(X_va):,}, test={len(X_te):,} | "
        f"features raw={n_raw_features}, preprocessed={n_pre}, selected={X_tr.shape[1]} | {split_meta}"
    )
    _write_preprocessing_audit(data, cfg, path)
    return data


def load_or_prepare_dataset(dataset_name: str, path: str, cfg: Config) -> PreparedData:
    key = prepared_cache_key(dataset_name, path, cfg)
    cache_dir = os.path.join(CACHE_ROOT, dataset_name, key)
    cache_file = os.path.join(cache_dir, "prepared_arrays.npz")
    meta_file = os.path.join(cache_dir, "prepared_meta.json")

    if CACHE_PREPARED_DATA:
        cached = load_prepared_cache(cache_file, meta_file)
        if cached is not None:
            print_header(f"LOADING PREPARED CACHE: {dataset_name} | split={cfg.split_mode} | seed={cfg.seed}")
            print(f"Cache: {cache_file}")
            _write_preprocessing_audit(cached, cfg, path)
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
    n = name.lower()
    if "strictsplit" in n: return "leakage_resistant"
    if "ablation" in n or "abl_" in n: return "ablation"
    if "adaptive_rpp" in n or "alie_m30" in n or "alie_m40" in n or "stress" in n: return "robustness_stress"
    if "privacy" in n: return "privacy"
    if n.startswith("rev_") or "matched" in n: return "reviewer_matched"
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
        "checkpoint_version": 6,
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
        "checkpoint_version": 6,
        "fingerprint": fingerprint,
        "W": server.W,
        "V": server.V,
        "reputation": server.reputation,
        "p4p_probe": server.p4p_probe,
        "p4p_suspicion": server.p4p_suspicion,
        "p4p_round": server.p4p_round,
        "trustfed_trust": server.trustfed_trust,
        "trustfed_prev_sketch": server.trustfed_prev_sketch,
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


def _defense_method_name(cfg: Config) -> str:
    if cfg.aggregator == "rpp": return "RPP-FL"
    if cfg.aggregator == "p4p": return "P4P"
    if cfg.aggregator == "trustfed": return "TrustFed-IDS-style"
    if cfg.aggregator == "median": return "CoordinateMedian"
    if cfg.aggregator == "trimmed_mean": return "TrimmedMean"
    if cfg.aggregator == "uniform_mean": return "UniformMean-DP"
    if cfg.aggregator == "fedavg" and cfg.fedprox_mu > 0: return "FedProx"
    if cfg.aggregator == "fedavg": return "FedAvg"
    return cfg.aggregator


def run_federated_experiment(data: PreparedData, cfg: Config,
                             attacked: bool,
                             scenario_name: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    set_seed(cfg.seed)
    model_ctor = build_model_ctor(data.n_selected_features, cfg)
    temp_model = model_ctor()
    n_params = sum(p.numel() for p in temp_model.parameters())
    del temp_model

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

    partitions = dirichlet_label_partition(
        data.y_train, n_clients=cfg.n_clients, alpha=cfg.dirichlet_alpha,
        seed=cfg.seed, min_samples=cfg.min_client_samples,
    )
    np.savez_compressed(
        os.path.join(run_dir, "client_partition_indices.npz"),
        **{f"client_{cid}": np.asarray(idx, dtype=np.int64) for cid, idx in partitions.items()}
    )

    clients: Dict[int, Client] = {
        cid: Client(cid, data.X_train[idx], data.y_train[idx], cfg, model_ctor)
        for cid, idx in partitions.items()
    }
    server = Server(cfg.n_clients, cfg, model_ctor, n_params)
    server.set_validation_reference(data.X_val, data.y_val)
    accountant = PrivacyAccountant(cfg)

    rng_py = random.Random(cfg.seed + 9001)
    torch_gen = torch.Generator().manual_seed(cfg.seed + 8128)
    all_clients = list(range(cfg.n_clients))
    mal_count = int(round(cfg.frac_malicious * cfg.n_clients)) if attacked else 0
    mal_count = min(max(mal_count, 0), cfg.n_clients - 1 if cfg.n_clients > 1 else 0)
    malicious = set(rng_py.sample(all_clients, k=mal_count)) if mal_count > 0 else set()
    per_round = min(max(1, int(round(cfg.clients_per_round_frac * cfg.n_clients))), cfg.n_clients)

    atomic_json_dump({
        "scenario": scenario_name,
        "dataset": data.dataset,
        "split_mode": data.split_mode,
        "split_meta": data.split_meta,
        "config": asdict(cfg),
        "defense_method": _defense_method_name(cfg),
        "malicious_client_ids_full_federation": sorted(int(c) for c in malicious),
        "attack_protocol": attack_protocol_manifest(cfg),
        "test_policy": "validation-only round/threshold selection; test evaluated exactly once after selection",
        "p4p_reference_doi": P4P_REFERENCE_DOI if cfg.aggregator == "p4p" else None,
        "trustfed_reference_doi": TRUSTFED_REFERENCE_DOI if cfg.aggregator == "trustfed" else None,
        "comparison_note": "All matched methods use identical prepared arrays, client partition seed, local model, local optimizer, communication budget, malicious identities, and attack generation seed.",
    }, os.path.join(run_dir, "run_manifest.json"))

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0; best_round = 0; best_threshold = 0.5
    best_W: Optional[torch.Tensor] = None
    patience = 0; start_round = 1

    ckpt = load_training_checkpoint(last_ckpt_path, fingerprint)
    if ckpt is not None:
        server.W = ckpt["W"].clone(); server.V = ckpt["V"].clone()
        server.reputation = {int(k): float(v) for k, v in ckpt.get("reputation", server.reputation).items()}
        pp = ckpt.get("p4p_probe")
        if isinstance(pp, torch.Tensor): server.p4p_probe = pp.clone()
        server.p4p_suspicion = {int(k): float(v) for k, v in ckpt.get("p4p_suspicion", server.p4p_suspicion).items()}
        server.p4p_round = int(ckpt.get("p4p_round", 0))
        server.trustfed_trust = {int(k): float(v) for k, v in ckpt.get("trustfed_trust", server.trustfed_trust).items()}
        prev = ckpt.get("trustfed_prev_sketch", {})
        if isinstance(prev, dict):
            server.trustfed_prev_sketch = {int(k): (v.clone() if isinstance(v, torch.Tensor) else None) for k, v in prev.items()}
        set_params_from_vector(server.model, server.W)
        history = list(ckpt.get("history", []))
        best_val_f1 = float(ckpt.get("best_val_f1", -1.0)); best_round = int(ckpt.get("best_round", 0))
        best_threshold = float(ckpt.get("best_threshold", 0.5))
        bw = ckpt.get("best_W"); best_W = bw.clone() if isinstance(bw, torch.Tensor) else None
        patience = int(ckpt.get("patience", 0))
        malicious = set(int(x) for x in ckpt.get("malicious", sorted(malicious)))
        accountant.load_state_dict(ckpt.get("accountant_state"))
        restore_rng_state(ckpt.get("rng_state"), rng_py, torch_gen)
        start_round = int(ckpt.get("round", 0)) + 1
        print(f"[RESUME] {data.dataset}/{scenario_name}: continuing from round {start_round}.")

    print_header(f"{data.dataset} | {scenario_name} | method={_defense_method_name(cfg)} | alpha={cfg.dirichlet_alpha} | split={data.split_mode}")
    print(f"Model params: {n_params:,} | malicious full-federation clients: {sorted(malicious)}")

    already_early_stopped = (
        cfg.enable_early_stopping and (start_round-1) >= cfg.min_rounds_before_stop
        and patience >= cfg.early_stop_patience
    )
    if not already_early_stopped:
        for rnd in range(start_round, cfg.n_rounds + 1):
            t0 = time.perf_counter()
            selected = rng_py.sample(all_clients, k=per_round)
            selected_malicious = [cid for cid in selected if cid in malicious]

            raw_deltas: Dict[int, torch.Tensor] = {}
            sample_counts: Dict[int, int] = {}
            client_times: List[float] = []
            for cid in selected:
                d, sec = clients[cid].local_train_and_delta(server.W)
                raw_deltas[cid] = d; sample_counts[cid] = clients[cid].n_samples; client_times.append(sec)

            benign_selected = [cid for cid in selected if cid not in malicious]
            benign_stack = torch.stack([raw_deltas[c] for c in benign_selected], dim=0) if len(benign_selected) >= 2 else None
            adaptive_candidate = None
            if attacked and cfg.attack_type.lower() == "adaptive_rpp" and benign_stack is not None:
                adaptive_candidate = build_adaptive_rpp_coordinated_attack(benign_stack, server, cfg, torch_gen)

            attacked_updates: Dict[int, torch.Tensor] = {}
            for cid in selected:
                d = raw_deltas[cid]
                if attacked and cid in malicious:
                    if cfg.attack_type.lower() == "adaptive_rpp" and adaptive_candidate is not None:
                        scale = max(float(adaptive_candidate.std().item()), 1e-8)
                        jitter = torch.randn(adaptive_candidate.shape, generator=torch_gen, dtype=adaptive_candidate.dtype)
                        d = adaptive_candidate + cfg.adaptive_rpp_jitter * scale * jitter
                    else:
                        d = apply_poisoning_attack(d, cfg.attack_type, benign_stack, cfg, torch_gen)
                attacked_updates[cid] = d

            delta_global, trust, weights = server.aggregate(attacked_updates, sample_counts)
            server.apply_update(delta_global)

            val_prob = server.predict_proba(data.X_val)
            threshold = best_threshold_by_f1(data.y_val, val_prob)
            val_metrics = compute_metrics(data.y_val, val_prob, threshold)
            eps = accountant.step(cfg.dp_accountant_sample_rate if cfg.privacy_mode == "central_dp" else per_round/cfg.n_clients)

            trust_m = [trust[c] for c in selected if c in malicious]
            trust_b = [trust[c] for c in selected if c not in malicious]
            if attacked and trust_m and trust_b:
                client_attack_auc = float(roc_auc_score(
                    [1 if c in malicious else 0 for c in selected],
                    [1.0 - float(trust[c]) for c in selected],
                ))
                malicious_weight_share = float(sum(weights.get(c, 0.0) for c in selected if c in malicious))
            else:
                client_attack_auc = float("nan"); malicious_weight_share = 0.0
            rep_m = [server.reputation[c] for c in selected if c in malicious]
            rep_b = [server.reputation[c] for c in selected if c not in malicious]

            row = {
                "Dataset": data.dataset, "Scenario": scenario_name, "Round": rnd,
                "ProtocolSeed": cfg.seed, "SplitMode": data.split_mode,
                "ConfiguredMaliciousFractionFullFederation": cfg.frac_malicious if attacked else 0.0,
                "FullFederationMaliciousCount": len(malicious),
                "SelectedClientCount": len(selected),
                "SelectedMaliciousCount": len(selected_malicious),
                "RealizedMaliciousFractionSelected": len(selected_malicious)/max(len(selected),1),
                "ValThreshold": threshold,
                **{f"Val_{k}": v for k, v in val_metrics.items()},
                "Epsilon": eps,
                "RoundTime_s": time.perf_counter()-t0,
                "AvgClientTime_s": float(np.mean(client_times)),
                "RepSeparation": float(np.mean(rep_b)-np.mean(rep_m)) if rep_m and rep_b else float("nan"),
                "TrustSeparation": float(np.mean(trust_b)-np.mean(trust_m)) if trust_m and trust_b else float("nan"),
                "MeanTrust": float(np.mean(list(trust.values()))),
                "MaxAggregationWeight": float(max(weights.values())) if weights else float("nan"),
                "ClientAttackDetectAUC": client_attack_auc,
                "MaliciousWeightShare": malicious_weight_share,
                "SelectedClients": json.dumps(sorted(int(c) for c in selected)),
                "SelectedMaliciousClients": json.dumps(sorted(int(c) for c in selected_malicious)),
                **server.last_agg_diagnostics,
            }
            history.append(row)

            improved = val_metrics["F1"] > best_val_f1 + 1e-5
            if improved:
                best_val_f1 = val_metrics["F1"]; best_round = rnd; best_threshold = threshold
                best_W = server.W.clone(); patience = 0
            else:
                patience += 1

            atomic_dataframe_csv(pd.DataFrame(history), history_path)
            if rnd % max(CHECKPOINT_EVERY_N_ROUNDS,1) == 0 or rnd == cfg.n_rounds:
                save_training_checkpoint(
                    last_ckpt_path, fingerprint=fingerprint, server=server, cfg=cfg,
                    round_idx=rnd, history=history, best_val_f1=best_val_f1,
                    best_round=best_round, best_threshold=best_threshold, best_W=best_W,
                    patience=patience, malicious=malicious, rng_py=rng_py,
                    torch_gen=torch_gen, accountant=accountant,
                )
                if improved: shutil.copy2(last_ckpt_path, best_ckpt_path)

            print(f"R{rnd:02d}/{cfg.n_rounds} | ValF1={val_metrics['F1']:.4f} AUC={val_metrics['ROC_AUC']:.4f} | selected-mal={len(selected_malicious)}/{len(selected)} | attackAUC={client_attack_auc:.3f} | malWeight={malicious_weight_share:.3f}")
            if cfg.enable_early_stopping and rnd >= cfg.min_rounds_before_stop and patience >= cfg.early_stop_patience:
                break

    if best_W is None:
        best_W = server.W.clone(); best_round = max(start_round-1, 0)
    server.W = best_W; set_params_from_vector(server.model, best_W)
    val_prob_final = server.predict_proba(data.X_val)
    val_pred_final = (val_prob_final >= best_threshold).astype(np.int64)
    atomic_dataframe_csv(pd.DataFrame({
        "deduplicated_row_index": data.val_indices,
        "y_true": data.y_val.astype(np.int64),
        "probability": val_prob_final.astype(np.float64),
        "prediction": val_pred_final,
    }), os.path.join(run_dir, "validation_predictions.csv"))

    test_prob = server.predict_proba(data.X_test)
    test_metrics = compute_metrics(data.y_test, test_prob, best_threshold)
    test_pred = (test_prob >= best_threshold).astype(np.int64)
    atomic_dataframe_csv(pd.DataFrame({
        "deduplicated_row_index": data.test_indices,
        "y_true": data.y_test.astype(np.int64),
        "probability": test_prob.astype(np.float64),
        "prediction": test_pred,
    }), os.path.join(run_dir, "test_predictions.csv"))

    history_df = pd.DataFrame(history); atomic_dataframe_csv(history_df, history_path)
    result: Dict[str, Any] = {
        "RunFingerprint": fingerprint, "Dataset": data.dataset, "Scenario": scenario_name,
        "ExperimentGroup": experiment_group_from_name(scenario_name),
        "Aggregator": cfg.aggregator, "DefenseMethod": _defense_method_name(cfg),
        "AttackType": cfg.attack_type if attacked else "none",
        "MaliciousFraction": cfg.frac_malicious if attacked else 0.0,
        "DirichletAlpha": cfg.dirichlet_alpha, "SplitMode": data.split_mode,
        "SplitStrategy": data.split_meta.get("mode", data.split_mode),
        "PrivacyMode": cfg.privacy_mode, "DPNoiseMultiplier": cfg.dp_noise_multiplier,
        "DPTargetEpsilon": cfg.dp_target_epsilon, "ProtocolSeed": cfg.seed,
        "FixedRoundBudget": not cfg.enable_early_stopping, "ConfiguredRounds": cfg.n_rounds,
        "MaliciousClientIDs": json.dumps(sorted(int(c) for c in malicious)),
        "P4PReferenceDOI": P4P_REFERENCE_DOI if cfg.aggregator == "p4p" else "",
        "TrustFedReferenceDOI": TRUSTFED_REFERENCE_DOI if cfg.aggregator == "trustfed" else "",
        "BestRound": best_round, "ValBestF1": best_val_f1, "Threshold": best_threshold,
        "ModelParams": n_params, "ModelSizeMB_FP32": (n_params*4.0)/(1024.0**2),
        "Rows": data.raw_rows, "SelectedFeatures": data.n_selected_features,
        "UseTransformer": cfg.use_transformer, "UseAuxReconstruction": cfg.recon_lambda > 0,
        "UseRobustDeviation": cfg.use_robust_deviation, "UseCenterAlignment": cfg.use_center_alignment,
        "UseMomentum": cfg.use_momentum, "UsePeerAgreement": cfg.use_peer_agreement,
        "UseCollusionGuard": cfg.use_collusion_guard, "UseReputation": cfg.use_reputation,
        "UseWinsorization": cfg.use_winsorization, "UseMedianAnchor": cfg.use_median_anchor,
        **{f"Test_{k}": v for k, v in test_metrics.items()},
    }
    result["FinalEpsilon"] = history[-1]["Epsilon"] if history else float("nan")
    if not history_df.empty:
        result["RoundsExecuted"] = int(history_df["Round"].max())
        result["TotalTrainTime_s"] = float(history_df["RoundTime_s"].sum())
        result["AvgRoundTime_s"] = float(history_df["RoundTime_s"].mean())
        realized = pd.to_numeric(history_df["RealizedMaliciousFractionSelected"], errors="coerce")
        result["MeanRealizedMaliciousFractionSelected"] = float(realized.mean())
        result["MinRealizedMaliciousFractionSelected"] = float(realized.min())
        result["MaxRealizedMaliciousFractionSelected"] = float(realized.max())
        for src, dst in (
            ("ClientAttackDetectAUC", "MeanClientAttackDetectAUC"),
            ("MaliciousWeightShare", "MeanMaliciousWeightShare"),
            ("RPPFlaggedFraction", "MeanRPPFlaggedFraction"),
            ("RPPAnchorCoeff", "MeanRPPAnchorCoeff"),
            ("P4PMeanTemporalSuspicion", "MeanP4PTemporalSuspicion"),
            ("TrustFedMeanTrust", "MeanTrustFedTrust"),
        ):
            if src in history_df:
                vals = pd.to_numeric(history_df[src], errors="coerce")
                result[dst] = float(vals.mean()) if vals.notna().any() else float("nan")

    atomic_json_dump(result, os.path.join(run_dir, "final_result.json"))
    atomic_json_dump({"status":"COMPLETED", "fingerprint":fingerprint, "best_round":best_round,
                      "finished_at":time.strftime("%Y-%m-%d %H:%M:%S")}, os.path.join(run_dir, "COMPLETED.json"))
    print("\nTEST (evaluated once):"); print(pd.Series(result).to_string())
    del server, clients; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
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
# 18. ALL-REVIEWER REVISION SUITE + STATISTICS
# =============================================================================

def _method_kwargs(name: str) -> Dict[str, Any]:
    return {
        "rpp": dict(aggregator="rpp", fedprox_mu=0.0),
        "p4p": dict(aggregator="p4p", fedprox_mu=0.0),
        "trustfed": dict(aggregator="trustfed", fedprox_mu=0.0),
        "fedavg": dict(aggregator="fedavg", fedprox_mu=0.0),
        "fedprox": dict(aggregator="fedavg", fedprox_mu=0.01),
        "median": dict(aggregator="median", fedprox_mu=0.0),
        "trimmed": dict(aggregator="trimmed_mean", fedprox_mu=0.0),
    }[name]


def revision_repeated_scenarios(base: Config) -> List[Tuple[str, Config, bool]]:
    methods = list(REPEATED_MATCHED_METHODS)
    if base.seed == REVISION_SEEDS[0]:
        methods += list(SEED42_EXTRA_METHODS)
    out: List[Tuple[str, Config, bool]] = []
    conditions = [
        ("clean", dict(dirichlet_alpha=0.5, attack_type="alie", frac_malicious=0.20), False),
        ("alie20", dict(dirichlet_alpha=0.5, attack_type="alie", frac_malicious=0.20), True),
        ("alpha01_alie20", dict(dirichlet_alpha=REVIEWER_SEVERE_ALPHA, attack_type="alie", frac_malicious=0.20), True),
    ]
    for method in methods:
        for label, ckw, attacked in conditions:
            cfg = replace(base, enable_early_stopping=False, **_method_kwargs(method), **ckw)
            out.append((f"rev_matched_{method}_{label}_seed{base.seed}", cfg, attacked))

    if RUN_REVISION_MULTI_SEED_ABLATION:
        # Full RPP rows already exist above; only removal variants are added.
        ablations = [
            ("no_transformer_clean", dict(use_transformer=False), False),
            ("no_reputation_alie20", dict(use_reputation=False), True),
            ("no_momentum_alie20", dict(use_momentum=False), True),
            ("no_deviation_alie20", dict(use_robust_deviation=False), True),
            ("no_peer_alie20", dict(use_peer_agreement=False), True),
            ("no_collusion_alie20", dict(use_collusion_guard=False), True),
            ("no_winsor_alie20", dict(use_winsorization=False), True),
            ("no_median_anchor_alie20", dict(use_median_anchor=False), True),
        ]
        for label, akw, attacked in ablations:
            cfg = replace(base, aggregator="rpp", enable_early_stopping=False,
                          attack_type="alie", frac_malicious=0.20, **akw)
            out.append((f"rev_ablation_{label}_seed{base.seed}", cfg, attacked))

    # Repeat the original RPP non-IID sensitivity points that are not already
    # represented by the alpha=0.5 matched rows. This directly supports the
    # reviewer's request that the principal non-IID conclusions be multi-seed.
    for alpha in (1.0, REVIEWER_SEVERE_ALPHA):
        out.append((
            f"rev_noniid_rpp_alpha{alpha:g}_clean_seed{base.seed}",
            replace(base, aggregator="rpp", dirichlet_alpha=float(alpha),
                    attack_type="alie", frac_malicious=0.20, enable_early_stopping=False),
            False,
        ))
        # alpha=0.1 + ALIE is already present as rev_matched_rpp_alpha01_alie20.
        if abs(float(alpha) - REVIEWER_SEVERE_ALPHA) > 1e-12:
            out.append((
                f"rev_noniid_rpp_alpha{alpha:g}_alie20_seed{base.seed}",
                replace(base, aggregator="rpp", dirichlet_alpha=float(alpha),
                        attack_type="alie", frac_malicious=0.20, enable_early_stopping=False),
                True,
            ))
    return out


def revision_stress_scenarios(base: Config) -> List[Tuple[str, Config, bool]]:
    out = []
    if base.seed != REVISION_SEEDS[0]:
        return out
    if RUN_REVISION_HIGH_RATIO_STRESS:
        for frac in HIGH_MALICIOUS_FRACTIONS:
            for method in ("rpp", "fedavg"):
                out.append((
                    f"rev_stress_{method}_alie_m{int(frac*100)}_seed{base.seed}",
                    replace(base, enable_early_stopping=False, attack_type="alie",
                            frac_malicious=float(frac), **_method_kwargs(method)), True
                ))
    if RUN_REVISION_ADAPTIVE_ATTACK:
        for frac in (0.20, 0.40):
            for method in ("rpp", "fedavg"):
                out.append((
                    f"rev_stress_{method}_adaptive_rpp_m{int(frac*100)}_seed{base.seed}",
                    replace(base, enable_early_stopping=False, attack_type="adaptive_rpp",
                            frac_malicious=float(frac), **_method_kwargs(method)), True
                ))
    return out


def strict_split_scenarios(base: Config) -> List[Tuple[str, Config, bool]]:
    # Directly addresses the 100% EdgeIIoTset leakage concern. We retain only the
    # proposed method here because the question is how its performance changes
    # under a stricter holdout, not another aggregation ranking.
    return [
        (f"rev_strictsplit_rpp_clean_seed{base.seed}", replace(base, aggregator="rpp", enable_early_stopping=False), False),
        (f"rev_strictsplit_rpp_alie20_seed{base.seed}", replace(base, aggregator="rpp", attack_type="alie", frac_malicious=0.20, enable_early_stopping=False), True),
    ]


def privacy_scenarios(base: Config) -> List[Tuple[str, Config, bool]]:
    out = []
    if not RUN_REVISION_PRIVACY or base.seed != REVISION_SEEDS[0]:
        return out
    for target in TARGET_EPSILONS:
        try:
            nm = calibrate_noise_multiplier(float(target), sample_rate=1.0,
                                            steps=base.n_rounds, delta=base.dp_delta)
            out.append((
                f"rev_privacy_uniform_eps{target:g}_seed{base.seed}",
                replace(base, aggregator="uniform_mean", privacy_mode="central_dp",
                        dp_noise_multiplier=float(nm), dp_target_epsilon=float(target),
                        dp_accountant_sample_rate=1.0, enable_early_stopping=False),
                False,
            ))
        except Exception as e:
            print(f"[WARN] privacy target eps={target}: {e}")
    return out


def _ci95(values: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0: return float("nan"), float("nan")
    if len(x) == 1: return float(x[0]), float(x[0])
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    try:
        from scipy.stats import t  # type: ignore
        crit = float(t.ppf(0.975, df=len(x)-1))
    except Exception:
        crit = 1.96
    m = float(np.mean(x)); return m-crit*se, m+crit*se


def _holm_adjust(pvals: List[float]) -> List[float]:
    arr = np.asarray(pvals, dtype=float); n = len(arr)
    out = np.full(n, np.nan)
    valid = np.where(np.isfinite(arr))[0]
    order = valid[np.argsort(arr[valid])]
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (len(order)-rank)*arr[idx])
        val = max(prev, val); prev = val; out[idx] = val
    return out.tolist()


def write_revision_tables(df: pd.DataFrame) -> None:
    out_dir = os.path.join(RESULT_ROOT, "PUBLICATION_TABLES")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if df.empty: return
    atomic_dataframe_csv(df, os.path.join(out_dir, "ALL_REVISION_RUNS.csv"))

    # Multi-seed descriptive statistics.
    metrics = [m for m in ("Test_F1", "Test_MCC", "Test_ROC_AUC", "Test_Accuracy",
                           "MeanClientAttackDetectAUC", "MeanMaliciousWeightShare") if m in df.columns]
    group_cols = ["Dataset", "SplitMode", "DefenseMethod", "AttackType", "MaliciousFraction", "DirichletAlpha"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys)); base["NSeeds"] = int(g["ProtocolSeed"].nunique())
        for met in metrics:
            vals = pd.to_numeric(g[met], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0: continue
            lo, hi = _ci95(vals)
            rows.append({**base, "Metric": met, "N": len(vals), "Mean": float(np.mean(vals)),
                         "Std": float(np.std(vals, ddof=1)) if len(vals)>1 else 0.0,
                         "CI95_Low": lo, "CI95_High": hi})
    stats = pd.DataFrame(rows)
    atomic_dataframe_csv(stats, os.path.join(out_dir, "TABLE_REVISION_MULTI_SEED_MEAN_STD_CI95.csv"))

    # Paired seed-wise RPP comparisons for the central matched conditions.
    pair_rows = []
    focus = df[df["Scenario"].str.contains("rev_matched_", na=False)].copy()
    for met in [m for m in ("Test_F1", "Test_MCC") if m in focus.columns]:
        for cond_keys, g in focus.groupby(["Dataset","SplitMode","AttackType","MaliciousFraction","DirichletAlpha"], dropna=False):
            rpp = g[g["DefenseMethod"] == "RPP-FL"][["ProtocolSeed", met]].rename(columns={met:"RPP"})
            for method in sorted(set(g["DefenseMethod"]) - {"RPP-FL"}):
                oth = g[g["DefenseMethod"] == method][["ProtocolSeed", met]].rename(columns={met:"Other"})
                z = rpp.merge(oth, on="ProtocolSeed", how="inner")
                if len(z) < 2: continue
                diff = z["RPP"].to_numpy(float) - z["Other"].to_numpy(float)
                lo, hi = _ci95(diff)
                try:
                    from scipy.stats import ttest_rel, wilcoxon  # type: ignore
                    pt = float(ttest_rel(z["RPP"], z["Other"]).pvalue)
                    try: pw = float(wilcoxon(diff, zero_method="wilcox").pvalue)
                    except Exception: pw = float("nan")
                except Exception:
                    pt = pw = float("nan")
                sd = float(np.std(diff, ddof=1)) if len(diff)>1 else float("nan")
                dz = float(np.mean(diff)/sd) if np.isfinite(sd) and sd>1e-12 else float("nan")
                pair_rows.append({
                    "Dataset":cond_keys[0], "SplitMode":cond_keys[1], "AttackType":cond_keys[2],
                    "MaliciousFraction":cond_keys[3], "DirichletAlpha":cond_keys[4],
                    "Metric":met, "Comparator":method, "NPairedSeeds":len(z),
                    "MeanDifference_RPP_minus_Comparator":float(np.mean(diff)),
                    "CI95_Diff_Low":lo, "CI95_Diff_High":hi, "Cohen_dz":dz,
                    "Paired_t_p":pt, "Wilcoxon_p":pw,
                })
    pair = pd.DataFrame(pair_rows)
    if not pair.empty:
        pair["Wilcoxon_Holm_p"] = _holm_adjust(pair["Wilcoxon_p"].tolist())
        atomic_dataframe_csv(pair, os.path.join(out_dir, "TABLE_REVISION_PAIRED_STATISTICS.csv"))

    # Core reviewer-controlled baselines.
    core = df[df["Scenario"].str.contains("rev_matched_", na=False)].copy()
    atomic_dataframe_csv(core, os.path.join(out_dir, "TABLE_REVISION_MATCHED_BASELINES_ALL_SEEDS.csv"))

    abl = df[df["Scenario"].str.contains("rev_ablation_", na=False)].copy()
    atomic_dataframe_csv(abl, os.path.join(out_dir, "TABLE_REVISION_ABLATION_ALL_SEEDS.csv"))

    stress = df[df["Scenario"].str.contains("rev_stress_", na=False)].copy()
    atomic_dataframe_csv(stress, os.path.join(out_dir, "TABLE_REVISION_ATTACK_EFFECTIVENESS_AND_STRESS.csv"))

    # Explicit attack effectiveness relative to the same method's matched clean
    # reference. This is especially important for the undefended FedAvg control.
    eff_rows = []
    matched = df[df["Scenario"].str.contains("rev_matched_", na=False)].copy()
    if not matched.empty:
        clean = matched[(matched["AttackType"] == "none") & (matched["DirichletAlpha"] == 0.5)]
        attacked_rows = pd.concat([
            matched[(matched["AttackType"] != "none") & (matched["DirichletAlpha"] == 0.5)],
            stress,
        ], ignore_index=True, sort=False)
        for _, ar in attacked_rows.iterrows():
            ref = clean[(clean["Dataset"] == ar["Dataset"]) &
                        (clean["DefenseMethod"] == ar["DefenseMethod"]) &
                        (clean["ProtocolSeed"] == ar["ProtocolSeed"]) &
                        (clean["SplitMode"] == ar["SplitMode"])]
            if ref.empty: continue
            cr = ref.iloc[0]
            eff_rows.append({
                "Dataset": ar["Dataset"], "ProtocolSeed": ar["ProtocolSeed"],
                "DefenseMethod": ar["DefenseMethod"], "AttackType": ar["AttackType"],
                "MaliciousFraction": ar["MaliciousFraction"], "DirichletAlpha": ar["DirichletAlpha"],
                "Clean_F1": cr.get("Test_F1", np.nan), "Attacked_F1": ar.get("Test_F1", np.nan),
                "F1_Drop": cr.get("Test_F1", np.nan) - ar.get("Test_F1", np.nan),
                "Clean_MCC": cr.get("Test_MCC", np.nan), "Attacked_MCC": ar.get("Test_MCC", np.nan),
                "MCC_Drop": cr.get("Test_MCC", np.nan) - ar.get("Test_MCC", np.nan),
                "AttackDetectionAUC": ar.get("MeanClientAttackDetectAUC", np.nan),
                "MaliciousWeightShare": ar.get("MeanMaliciousWeightShare", np.nan),
                "MeanRealizedMaliciousFractionSelected": ar.get("MeanRealizedMaliciousFractionSelected", np.nan),
                "MinRealizedMaliciousFractionSelected": ar.get("MinRealizedMaliciousFractionSelected", np.nan),
                "MaxRealizedMaliciousFractionSelected": ar.get("MaxRealizedMaliciousFractionSelected", np.nan),
            })
    atomic_dataframe_csv(pd.DataFrame(eff_rows), os.path.join(out_dir, "TABLE_REVISION_ATTACK_EFFECTIVENESS_VS_CLEAN.csv"))

    noniid = df[df["Scenario"].str.contains("rev_noniid_", na=False)].copy()
    atomic_dataframe_csv(noniid, os.path.join(out_dir, "TABLE_REVISION_NONIID_MULTI_SEED.csv"))

    strict = df[df["Scenario"].str.contains("rev_strictsplit_", na=False)].copy()
    atomic_dataframe_csv(strict, os.path.join(out_dir, "TABLE_REVISION_LEAKAGE_RESISTANT_SPLIT.csv"))

    privacy = df[df["Scenario"].str.contains("rev_privacy_", na=False)].copy()
    atomic_dataframe_csv(privacy, os.path.join(out_dir, "TABLE_REVISION_PRIVACY_ACCOUNTING.csv"))

    orders = list(RDP_ORDERS)
    atomic_json_dump({
        "mode": "separate privacy-evaluation branch; NOT simultaneous with RPP screening",
        "adjacency": "client-level add/remove adjacency",
        "delta": Config().dp_delta,
        "clipping_norm_C": Config().update_clip_C,
        "aggregation": "uniform mean of individually clipped client updates; data-dependent trust/reputation disabled",
        "gaussian_noise": "N(0, sigma^2 * (C/m)^2 I) added to released aggregate",
        "client_sampling_in_simulator": "fixed-size without replacement: 8 of 10 clients per round",
        "accounting_sampling_assumption": "sample_rate=1.0; no privacy amplification from client subsampling is claimed",
        "rdp_orders": orders,
        "composed_rounds": Config().n_rounds,
        "secure_sum_assumption": "Only the uniform clipped sum/mean is assumed visible to the server. No concrete cryptographic secure-aggregation protocol is implemented or benchmarked by the training simulator.",
        "robust_mode_observability": "RPP robust mode requires individual clipped updates and is therefore operationally separate from this privacy branch.",
    }, os.path.join(out_dir, "PRIVACY_ACCOUNTING_MANIFEST.json"))

    atomic_json_dump({
        "RPP-FL": {
            "status": "proposed method implementation",
            "note": "Synchronized with manuscript Sections 3.4-3.7 in this revision code."
        },
        "P4P": {
            "status": "independent matched-protocol reimplementation from the published method description; not claimed to be official source code",
            "doi": P4P_REFERENCE_DOI,
            "matched_factors": ["preprocessing", "split", "client partitions", "local detector", "optimizer", "round budget", "client selection", "malicious identities", "attack realization"]
        },
        "TrustFed-IDS-style": {
            "status": "independent matched-protocol reimplementation of the published trust mechanism; original MLP-LSTM local model is intentionally replaced by the common paper local detector for aggregation-only comparison",
            "doi": TRUSTFED_REFERENCE_DOI,
            "trust_components": ["cosine similarity", "normalized server validation loss", "temporal update stability"]
        },
        "classical_baselines": ["FedAvg", "FedProx", "coordinate-wise Median", "Trimmed Mean"],
    }, os.path.join(out_dir, "METHOD_IMPLEMENTATION_MANIFEST.json"))


def _append_or_replace_result(all_results: List[Dict[str, Any]], result: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [r for r in all_results if not (
        str(r.get("Dataset")) == str(result.get("Dataset")) and str(r.get("Scenario")) == str(result.get("Scenario"))
    )] + [result]


def _run_scenarios(prepared: PreparedData, scenarios: List[Tuple[str, Config, bool]],
                   all_results: List[Dict[str, Any]], partial_path: str) -> List[Dict[str, Any]]:
    for scenario_name, cfg, attacked in scenarios:
        try:
            result, _ = run_federated_experiment(prepared, cfg, attacked, scenario_name)
            all_results = _append_or_replace_result(all_results, result)
            cur = pd.DataFrame(all_results)
            atomic_dataframe_csv(cur, partial_path)
            write_revision_tables(cur)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[OOM] {prepared.dataset}/{scenario_name}: {e}")
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            else:
                print(f"[ERROR] {prepared.dataset}/{scenario_name}: {e}")
        except Exception as e:
            print(f"[ERROR] {prepared.dataset}/{scenario_name}: {type(e).__name__}: {e}")
    return all_results


def main() -> None:
    set_seed(SEED); mount_drive_if_needed(); ensure_dirs()
    print_header("RPP-FL V6 — ALL-REVIEWER REVISION SUITE")
    print(f"Device={DEVICE} | seeds={REVISION_SEEDS} | result root={RESULT_ROOT}")
    try: pip_install_if_missing("opacus", "opacus")
    except Exception as e: print(f"[WARN] Opacus unavailable: {e}")
    if AUTO_DOWNLOAD_DATASETS: ensure_all_datasets()

    partial_path = os.path.join(RESULT_ROOT, "all_revision_results_partial.csv")
    all_results: List[Dict[str, Any]] = []
    if AUTO_RESUME and os.path.exists(partial_path):
        try: all_results = pd.read_csv(partial_path).to_dict(orient="records")
        except Exception: pass

    # 1) Reproducibility/preprocessing audit for every available dataset (seed 42).
    if RUN_PREPROCESSING_AUDIT_ALL_DATASETS:
        for dname in AUDIT_DATASETS:
            path = DATASET_PATHS[dname]
            if not _tabular_files(path): continue
            try:
                load_or_prepare_dataset(dname, path, Config(seed=REVISION_SEEDS[0], split_mode=DEFAULT_SPLIT_MODE))
            except Exception as e:
                print(f"[AUDIT WARN] {dname}: {e}")

    # 2) Multi-seed matched baselines + multi-seed ablations on CICIoT2023/TON_IoT.
    if RUN_REVISION_MATCHED_BASELINES:
        for dname in REPEATED_DATASETS:
            path = DATASET_PATHS[dname]
            if not _tabular_files(path): continue
            for seed in REVISION_SEEDS:
                base = Config(seed=seed, split_mode=DEFAULT_SPLIT_MODE)
                try: prepared = load_or_prepare_dataset(dname, path, base)
                except Exception as e:
                    print(f"[PREP ERROR] {dname}/seed{seed}: {e}"); continue
                all_results = _run_scenarios(prepared, revision_repeated_scenarios(base), all_results, partial_path)
                if RUN_REVISION_HIGH_RATIO_STRESS or RUN_REVISION_ADAPTIVE_ATTACK:
                    all_results = _run_scenarios(prepared, revision_stress_scenarios(base), all_results, partial_path)
                del prepared; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    # 3) Leakage-resistant time/group-separated EdgeIIoTset evaluation.
    if RUN_REVISION_STRICT_SPLIT:
        for dname in STRICT_SPLIT_DATASETS:
            path = DATASET_PATHS[dname]
            if not _tabular_files(path): continue
            for seed in REVISION_SEEDS:
                strict_base = Config(seed=seed, split_mode=STRICT_SPLIT_MODE)
                try:
                    strict_data = load_or_prepare_dataset(dname, path, strict_base)
                    all_results = _run_scenarios(strict_data, strict_split_scenarios(strict_base), all_results, partial_path)
                    # matched random-split RPP clean reference for the same seed
                    random_base = Config(seed=seed, split_mode=DEFAULT_SPLIT_MODE)
                    random_data = load_or_prepare_dataset(dname, path, random_base)
                    ref = [(f"rev_strictsplit_reference_random_rpp_clean_seed{seed}", replace(random_base, aggregator="rpp"), False)]
                    all_results = _run_scenarios(random_data, ref, all_results, partial_path)
                except Exception as e:
                    print(f"[STRICT SPLIT WARN] {dname}/seed{seed}: {e}")

    # 4) Separate privacy-accounting branch (one seed; four target epsilons).
    if RUN_REVISION_PRIVACY:
        for dname in PRIVACY_DATASETS:
            path = DATASET_PATHS[dname]
            if not _tabular_files(path): continue
            base = Config(seed=REVISION_SEEDS[0], split_mode=DEFAULT_SPLIT_MODE)
            try:
                data = load_or_prepare_dataset(dname, path, base)
                all_results = _run_scenarios(data, privacy_scenarios(base), all_results, partial_path)
            except Exception as e:
                print(f"[PRIVACY WARN] {dname}: {e}")

    if all_results:
        final_df = pd.DataFrame(all_results)
        final_path = os.path.join(RESULT_ROOT, "ALL_REVIEWER_REVISION_RESULTS.csv")
        atomic_dataframe_csv(final_df, final_path); write_revision_tables(final_df)
        print_header("REVISION SUITE COMPLETE")
        print(f"Saved: {final_path}")
        print(f"Publication tables: {os.path.join(RESULT_ROOT, 'PUBLICATION_TABLES')}")
    else:
        print("No revision experiments completed.")


if __name__ == "__main__":
    main()