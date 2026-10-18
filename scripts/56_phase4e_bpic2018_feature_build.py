#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ETCE Stage56
BPIC2018 Development Feature Build + Frozen Representation Mapping

This stage is DATA/REPRESENTATION ONLY. It does NOT train RF/XGB/MLP/LSTM.

Frozen rules
------------
1. Input must be the successful Stage55 BPIC2018 frozen dataset.
2. Development mapping:
      fit preprocessing ONLY on 2015
      transform 2015 (development_train) and 2016 (development_valid)
3. Final mapping:
      fit preprocessing ONLY on 2015+2016
      transform 2015+2016 (final_train) and 2017 (formal_external_test)
4. 2017 is NEVER used to fit vocabularies, numerical scaling, feature selection,
   class weights, thresholds, hyperparameters, or model architecture.
5. No label-driven feature selection is performed in Stage56.
6. Prefixes remain P5/P10/P15/P20 exactly as frozen by Stage55.

Outputs are shared by RF / XGB / MLP / LSTM.

Tabular representation
----------------------
For each case-prefix:
  fixed numeric prefix descriptors
  +
  count vectors for activity / doctype / subprocess vocabularies.

Sequence representation
-----------------------
For each case-prefix, fixed length = prefix length:
  activity_id
  doctype_id
  subprocess_id
  log1p(delta_seconds)
  mask

The categorical vocabularies contain:
  <PAD> = 0
  <UNK> = 1

No category observed only in validation/test can enter a fitted vocabulary.

Expected Stage55 directory
--------------------------
D:\\ETCE\\results\\phase4\\phase4e_bpic2018_external\\stage55_dataset

Default Stage56 output
----------------------
D:\\ETCE\\results\\phase4\\phase4e_bpic2018_external\\stage56_features
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


STAGE = 56
PREFIXES = (5, 10, 15, 20)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID = 0
UNK_ID = 1

EXPECTED_STAGE55_FILES = (
    "protocol_frozen.json",
    "case_manifest.csv",
    "prefix_manifest.csv",
    "prefix_events.csv.gz",
    "class_stats.csv",
    "consistency_checks.json",
    "dataset_summary.json",
    "sha256_manifest.json",
    "STAGE55_FROZEN_PROTOCOL.md",
)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def norm_text(x: Any) -> str:
    if x is None:
        return ""
    return " ".join(str(x).strip().lower().split())


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_clean_output_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        existing = list(out_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(
                f"Stage56 output already exists: {out_dir}\n"
                "Refusing to overwrite silently. Use --overwrite only for an exact rebuild."
            )
        if overwrite:
            import shutil
            for p in existing:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iso_timestamp(s: str) -> float:
    from datetime import datetime
    t = str(s).strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    dt = datetime.fromisoformat(t)
    return dt.timestamp()


# ---------------------------------------------------------------------
# Stage55 verification
# ---------------------------------------------------------------------

def verify_stage55(stage55_dir: Path) -> Dict[str, Any]:
    for name in EXPECTED_STAGE55_FILES:
        p = stage55_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Missing Stage55 frozen file: {p}")

    checks = read_json(stage55_dir / "consistency_checks.json")
    failed = {k: v for k, v in checks.items() if not v.get("pass", False)}
    if failed:
        raise RuntimeError(
            "Stage55 consistency_checks.json is not fully PASS. "
            "Stage56 must not run from an unfrozen Stage55 dataset."
        )

    summary = read_json(stage55_dir / "dataset_summary.json")
    if not summary.get("all_hard_checks_pass", False):
        raise RuntimeError("Stage55 dataset_summary.json says hard checks did not pass.")

    protocol = read_json(stage55_dir / "protocol_frozen.json")
    if protocol.get("stage") != 55:
        raise RuntimeError("Input protocol is not Stage55.")
    if protocol.get("dataset") != "BPIC2018":
        raise RuntimeError("Stage56 expects BPIC2018 Stage55 data.")
    if protocol.get("prefixes") != [5, 10, 15, 20]:
        raise RuntimeError(f"Unexpected frozen prefixes: {protocol.get('prefixes')}")

    # Verify Stage55 output hashes where possible.
    hash_manifest = read_json(stage55_dir / "sha256_manifest.json")
    hash_records = hash_manifest.get("files", {})
    verified = {}
    for raw_path, meta in hash_records.items():
        p = Path(raw_path)
        # Input XES may not be under Stage55 dir; only verify files that still exist.
        if p.exists():
            actual = sha256_file(p)
            expected = meta.get("sha256")
            ok = actual == expected
            verified[str(p)] = {"pass": ok, "actual": actual, "expected": expected}
            if not ok:
                raise RuntimeError(f"SHA256 mismatch for frozen Stage55 file: {p}")

    return {
        "protocol": protocol,
        "summary": summary,
        "verified_hashes": verified,
    }


# ---------------------------------------------------------------------
# Canonical prefix sample reader
# ---------------------------------------------------------------------

@dataclass
class PrefixSample:
    case_id: str
    case_year: int
    dev_role: str
    final_role: str
    label: int
    label_reason: str
    prefix_len: int
    timestamps: List[float]
    timestamp_raw: List[str]
    activities: List[str]
    doctypes: List[str]
    subprocesses: List[str]


def iter_prefix_samples(prefix_events_path: Path) -> Iterator[PrefixSample]:
    """
    prefix_events.csv.gz is written in Stage55 grouped by case and prefix.
    Stream one case-prefix sample at a time.
    """
    with gzip.open(prefix_events_path, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)

        current_key: Optional[Tuple[str, int]] = None
        rows: List[Dict[str, str]] = []

        def emit(group_rows: List[Dict[str, str]]) -> PrefixSample:
            if not group_rows:
                raise RuntimeError("Internal empty prefix group.")
            first = group_rows[0]
            p = int(first["prefix_len"])
            if len(group_rows) != p:
                raise RuntimeError(
                    f"Prefix row count mismatch for case={first['case_id']} P{p}: "
                    f"{len(group_rows)} rows"
                )

            # Stage55 guarantees prefix_event_pos = 1..p.
            positions = [int(x["prefix_event_pos"]) for x in group_rows]
            if positions != list(range(1, p + 1)):
                raise RuntimeError(
                    f"Non-canonical prefix positions for {first['case_id']} P{p}: {positions}"
                )

            ts_raw = [x["timestamp"] for x in group_rows]
            ts = [parse_iso_timestamp(x) for x in ts_raw]
            if any(b < a for a, b in zip(ts, ts[1:])):
                raise RuntimeError(
                    f"Stage56 found non-monotone frozen prefix timestamps: "
                    f"{first['case_id']} P{p}"
                )

            return PrefixSample(
                case_id=first["case_id"],
                case_year=int(first["case_year"]),
                dev_role=first["dev_role"],
                final_role=first["final_role"],
                label=int(first["label"]),
                label_reason=first["label_reason"],
                prefix_len=p,
                timestamps=ts,
                timestamp_raw=ts_raw,
                activities=[norm_text(x["activity"]) for x in group_rows],
                doctypes=[norm_text(x["doctype"]) for x in group_rows],
                subprocesses=[norm_text(x["subprocess"]) for x in group_rows],
            )

        for row in r:
            key = (row["case_id"], int(row["prefix_len"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                yield emit(rows)
                rows = []
                current_key = key
            rows.append(row)

        if rows:
            yield emit(rows)


# ---------------------------------------------------------------------
# Vocabularies and scalers
# ---------------------------------------------------------------------

def build_vocab(values: Iterable[str]) -> Dict[str, int]:
    counts = Counter(values)
    # Deterministic: frequency desc, lexical asc.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, _ in ordered:
        if token in vocab:
            continue
        vocab[token] = len(vocab)
    return vocab


def vocab_hash(vocab: Dict[str, int]) -> str:
    payload = json.dumps(vocab, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class Standardizer:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self.mean_ = X.mean(axis=0, dtype=np.float64)
        self.scale_ = X.std(axis=0, dtype=np.float64)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Standardizer not fitted")
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "scale": self.scale_.tolist() if self.scale_ is not None else None,
        }


# ---------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------

NUMERIC_FEATURE_NAMES = [
    "prefix_len",
    "elapsed_seconds",
    "mean_delta_seconds",
    "std_delta_seconds",
    "max_delta_seconds",
    "zero_delta_fraction",
    "distinct_activity_count",
    "distinct_doctype_count",
    "distinct_subprocess_count",
    "activity_change_fraction",
    "doctype_change_fraction",
    "subprocess_change_fraction",
]


def fixed_numeric_features(s: PrefixSample) -> np.ndarray:
    ts = np.asarray(s.timestamps, dtype=np.float64)
    if len(ts) > 1:
        deltas = np.diff(ts)
    else:
        deltas = np.zeros((0,), dtype=np.float64)

    def change_fraction(vals: Sequence[str]) -> float:
        if len(vals) <= 1:
            return 0.0
        changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
        return changes / (len(vals) - 1)

    elapsed = float(ts[-1] - ts[0]) if len(ts) else 0.0
    mean_delta = float(deltas.mean()) if len(deltas) else 0.0
    std_delta = float(deltas.std()) if len(deltas) else 0.0
    max_delta = float(deltas.max()) if len(deltas) else 0.0
    zero_frac = float(np.mean(deltas == 0.0)) if len(deltas) else 0.0

    return np.asarray(
        [
            float(s.prefix_len),
            max(0.0, elapsed),
            max(0.0, mean_delta),
            max(0.0, std_delta),
            max(0.0, max_delta),
            zero_frac,
            float(len(set(s.activities))),
            float(len(set(s.doctypes))),
            float(len(set(s.subprocesses))),
            change_fraction(s.activities),
            change_fraction(s.doctypes),
            change_fraction(s.subprocesses),
        ],
        dtype=np.float32,
    )


def count_vector(values: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    # Exclude PAD from count vector; keep UNK as a real column.
    out = np.zeros((len(vocab) - 1,), dtype=np.float32)
    for x in values:
        idx = vocab.get(x, UNK_ID)
        out[idx - 1] += 1.0
    return out


def encode_sequence(values: Sequence[str], vocab: Dict[str, int], length: int) -> np.ndarray:
    arr = np.full((length,), PAD_ID, dtype=np.int32)
    for i, x in enumerate(values[:length]):
        arr[i] = vocab.get(x, UNK_ID)
    return arr


def sequence_delta_log1p(s: PrefixSample, length: int) -> np.ndarray:
    arr = np.zeros((length,), dtype=np.float32)
    if not s.timestamps:
        return arr
    for i in range(1, min(len(s.timestamps), length)):
        delta = max(0.0, s.timestamps[i] - s.timestamps[i - 1])
        arr[i] = np.float32(math.log1p(delta))
    return arr


# ---------------------------------------------------------------------
# Materialize split data
# ---------------------------------------------------------------------

def select_samples(
    all_samples: List[PrefixSample],
    years: Sequence[int],
    prefix: int,
) -> List[PrefixSample]:
    ys = set(years)
    return [s for s in all_samples if s.case_year in ys and s.prefix_len == prefix]


def fit_mapping(samples: Sequence[PrefixSample]) -> Dict[str, Any]:
    activities = (x for s in samples for x in s.activities)
    doctypes = (x for s in samples for x in s.doctypes)
    subprocesses = (x for s in samples for x in s.subprocesses)

    activity_vocab = build_vocab(activities)
    doctype_vocab = build_vocab(doctypes)
    subprocess_vocab = build_vocab(subprocesses)

    X_num = np.vstack([fixed_numeric_features(s) for s in samples]).astype(np.float32)
    scaler = Standardizer().fit(X_num)

    return {
        "activity_vocab": activity_vocab,
        "doctype_vocab": doctype_vocab,
        "subprocess_vocab": subprocess_vocab,
        "scaler": scaler,
    }


def make_feature_names(mapping: Dict[str, Any]) -> List[str]:
    names = list(NUMERIC_FEATURE_NAMES)

    def inv_vocab(vocab: Dict[str, int]) -> List[Tuple[int, str]]:
        return sorted((idx, tok) for tok, idx in vocab.items() if idx != PAD_ID)

    for idx, tok in inv_vocab(mapping["activity_vocab"]):
        names.append(f"activity_count::{tok}")
    for idx, tok in inv_vocab(mapping["doctype_vocab"]):
        names.append(f"doctype_count::{tok}")
    for idx, tok in inv_vocab(mapping["subprocess_vocab"]):
        names.append(f"subprocess_count::{tok}")
    return names


def transform_samples(
    samples: Sequence[PrefixSample],
    mapping: Dict[str, Any],
    prefix: int,
) -> Dict[str, np.ndarray]:
    n = len(samples)

    X_num_raw = np.vstack([fixed_numeric_features(s) for s in samples]).astype(np.float32)
    X_num = mapping["scaler"].transform(X_num_raw)

    X_act = np.vstack(
        [count_vector(s.activities, mapping["activity_vocab"]) for s in samples]
    ).astype(np.float32)
    X_doc = np.vstack(
        [count_vector(s.doctypes, mapping["doctype_vocab"]) for s in samples]
    ).astype(np.float32)
    X_sub = np.vstack(
        [count_vector(s.subprocesses, mapping["subprocess_vocab"]) for s in samples]
    ).astype(np.float32)

    X_tab = np.concatenate([X_num, X_act, X_doc, X_sub], axis=1).astype(np.float32)

    seq_act = np.vstack(
        [encode_sequence(s.activities, mapping["activity_vocab"], prefix) for s in samples]
    ).astype(np.int32)
    seq_doc = np.vstack(
        [encode_sequence(s.doctypes, mapping["doctype_vocab"], prefix) for s in samples]
    ).astype(np.int32)
    seq_sub = np.vstack(
        [encode_sequence(s.subprocesses, mapping["subprocess_vocab"], prefix) for s in samples]
    ).astype(np.int32)
    seq_delta = np.vstack(
        [sequence_delta_log1p(s, prefix) for s in samples]
    ).astype(np.float32)
    seq_mask = np.ones((n, prefix), dtype=np.uint8)

    case_id = np.asarray([s.case_id for s in samples], dtype=str)
    year = np.asarray([s.case_year for s in samples], dtype=np.int16)
    y = np.asarray([s.label for s in samples], dtype=np.int8)
    label_reason = np.asarray([s.label_reason for s in samples], dtype=str)

    return {
        "case_id": case_id,
        "year": year,
        "y": y,
        "label_reason": label_reason,
        "X_tabular": X_tab,
        "X_numeric_raw": X_num_raw,
        "activity_seq": seq_act,
        "doctype_seq": seq_doc,
        "subprocess_seq": seq_sub,
        "delta_log1p_seq": seq_delta,
        "mask": seq_mask,
    }


def unknown_rates(
    samples: Sequence[PrefixSample],
    mapping: Dict[str, Any],
) -> Dict[str, float]:
    def rate(values: Iterable[str], vocab: Dict[str, int]) -> float:
        total = 0
        unk = 0
        for x in values:
            total += 1
            if x not in vocab:
                unk += 1
        return (unk / total) if total else 0.0

    return {
        "activity": rate((x for s in samples for x in s.activities), mapping["activity_vocab"]),
        "doctype": rate((x for s in samples for x in s.doctypes), mapping["doctype_vocab"]),
        "subprocess": rate((x for s in samples for x in s.subprocesses), mapping["subprocess_vocab"]),
    }


def save_npz(path: Path, data: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)


def mapping_metadata(mapping: Dict[str, Any]) -> Dict[str, Any]:
    av = mapping["activity_vocab"]
    dv = mapping["doctype_vocab"]
    sv = mapping["subprocess_vocab"]
    return {
        "activity_vocab": av,
        "doctype_vocab": dv,
        "subprocess_vocab": sv,
        "activity_vocab_sha256": vocab_hash(av),
        "doctype_vocab_sha256": vocab_hash(dv),
        "subprocess_vocab_sha256": vocab_hash(sv),
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "numeric_standardizer": mapping["scaler"].to_dict(),
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="ETCE Stage56 - BPIC2018 Development Feature Build + Frozen Representation Mapping"
    )
    ap.add_argument(
        "--project_root",
        type=Path,
        default=Path(r"D:\ETCE"),
    )
    ap.add_argument(
        "--stage55_dir",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = ap.parse_args()

    project_root = args.project_root.resolve()
    stage55_dir = (
        args.stage55_dir.resolve()
        if args.stage55_dir
        else (
            project_root
            / "results"
            / "phase4"
            / "phase4e_bpic2018_external"
            / "stage55_dataset"
        )
    )
    out_dir = (
        args.out_dir.resolve()
        if args.out_dir
        else (
            project_root
            / "results"
            / "phase4"
            / "phase4e_bpic2018_external"
            / "stage56_features"
        )
    )

    ensure_clean_output_dir(out_dir, args.overwrite)
    t0 = time.time()

    print("=" * 118)
    print("ETCE STAGE56 - BPIC2018 DEVELOPMENT FEATURE BUILD + FROZEN REPRESENTATION MAPPING")
    print("=" * 118)
    print(f"Stage55 : {stage55_dir}")
    print(f"Output  : {out_dir}")

    stage55_verify = verify_stage55(stage55_dir)
    print("[Stage55 gate] PASS")

    # Read all 4 * 43,793 prefix samples. This is about 175k samples and is
    # manageable in memory because each contains at most 20 events.
    samples = list(iter_prefix_samples(stage55_dir / "prefix_events.csv.gz"))
    print(f"Canonical case-prefix samples: {len(samples):,}")

    expected_samples = int(stage55_verify["summary"]["prediction_cases"]) * len(PREFIXES)
    if len(samples) != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples:,} prefix samples from Stage55; got {len(samples):,}"
        )

    # Global frozen audit.
    counts_by_year_prefix = Counter((s.case_year, s.prefix_len) for s in samples)
    labels_by_year_prefix = Counter(
        (s.case_year, s.prefix_len, s.label) for s in samples
    )

    # A case must appear exactly once per prefix.
    case_prefix_seen = Counter((s.case_id, s.prefix_len) for s in samples)
    dup = [k for k, v in case_prefix_seen.items() if v != 1]
    if dup:
        raise RuntimeError(f"Duplicate/missing case-prefix pairs detected: {dup[:20]}")

    # The case set must be identical across P5/P10/P15/P20.
    case_sets = {
        p: {s.case_id for s in samples if s.prefix_len == p}
        for p in PREFIXES
    }
    base_set = case_sets[5]
    if any(case_sets[p] != base_set for p in PREFIXES):
        raise RuntimeError("Case membership differs across frozen prefixes.")

    # -------------------------------------------------------------
    # Development mapping: fit 2015 only
    # -------------------------------------------------------------
    dev_fit_samples = [s for s in samples if s.case_year == 2015]
    dev_mapping = fit_mapping(dev_fit_samples)
    write_json(out_dir / "mapping_development_fit2015.json", mapping_metadata(dev_mapping))

    # -------------------------------------------------------------
    # Final mapping: fit 2015+2016 only
    # -------------------------------------------------------------
    final_fit_samples = [s for s in samples if s.case_year in (2015, 2016)]
    final_mapping = fit_mapping(final_fit_samples)
    write_json(out_dir / "mapping_final_fit2015_2016.json", mapping_metadata(final_mapping))

    # Ensure the final mapping NEVER saw 2017 by construction.
    if any(s.case_year == 2017 for s in final_fit_samples):
        raise RuntimeError("Critical leakage: 2017 entered final mapping fit.")

    feature_names_dev = make_feature_names(dev_mapping)
    feature_names_final = make_feature_names(final_mapping)
    write_json(out_dir / "feature_names_development.json", feature_names_dev)
    write_json(out_dir / "feature_names_final.json", feature_names_final)

    # Save each prefix separately for simple downstream use.
    output_records: Dict[str, Any] = {}
    split_audit: Dict[str, Any] = {}

    for p in PREFIXES:
        print(f"\n[P{p}] building development representation...", flush=True)

        train15 = select_samples(samples, [2015], p)
        valid16 = select_samples(samples, [2016], p)

        dev_train_data = transform_samples(train15, dev_mapping, p)
        dev_valid_data = transform_samples(valid16, dev_mapping, p)

        dev_train_path = out_dir / f"development_P{p}_train2015.npz"
        dev_valid_path = out_dir / f"development_P{p}_valid2016.npz"
        save_npz(dev_train_path, dev_train_data)
        save_npz(dev_valid_path, dev_valid_data)

        print(f"[P{p}] building final-refit representation...", flush=True)

        final_train = select_samples(samples, [2015, 2016], p)
        formal_test = select_samples(samples, [2017], p)

        final_train_data = transform_samples(final_train, final_mapping, p)
        formal_test_data = transform_samples(formal_test, final_mapping, p)

        final_train_path = out_dir / f"final_P{p}_train2015_2016.npz"
        formal_test_path = out_dir / f"final_P{p}_test2017.npz"
        save_npz(final_train_path, final_train_data)
        save_npz(formal_test_path, formal_test_data)

        split_audit[f"P{p}"] = {
            "development_train_2015": {
                "n": len(train15),
                "positive": int(sum(s.label for s in train15)),
                "negative": int(len(train15) - sum(s.label for s in train15)),
            },
            "development_valid_2016": {
                "n": len(valid16),
                "positive": int(sum(s.label for s in valid16)),
                "negative": int(len(valid16) - sum(s.label for s in valid16)),
                "unknown_rates_vs_2015_mapping": unknown_rates(valid16, dev_mapping),
            },
            "final_train_2015_2016": {
                "n": len(final_train),
                "positive": int(sum(s.label for s in final_train)),
                "negative": int(len(final_train) - sum(s.label for s in final_train)),
            },
            "formal_external_test_2017": {
                "n": len(formal_test),
                "positive": int(sum(s.label for s in formal_test)),
                "negative": int(len(formal_test) - sum(s.label for s in formal_test)),
                "unknown_rates_vs_2015_2016_mapping": unknown_rates(formal_test, final_mapping),
            },
            "development_tabular_dim": int(dev_train_data["X_tabular"].shape[1]),
            "final_tabular_dim": int(final_train_data["X_tabular"].shape[1]),
            "sequence_length": p,
        }

        output_records[f"P{p}"] = {
            "development_train": str(dev_train_path),
            "development_valid": str(dev_valid_path),
            "final_train": str(final_train_path),
            "formal_test": str(formal_test_path),
        }

    # -------------------------------------------------------------
    # Hard gates
    # -------------------------------------------------------------
    checks: Dict[str, Dict[str, Any]] = {}

    def add_check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks[name] = {
            "pass": bool(passed),
            "actual": actual,
            "expected": expected,
        }

    # Expected cohort counts are taken from Stage55 output, not re-guessed here.
    pred_year_counts = {
        int(k): int(v)
        for k, v in stage55_verify["summary"]["prediction_year_counts"].items()
    }
    pos_year_counts = {
        int(k): int(v)
        for k, v in stage55_verify["summary"]["positive_by_year_prediction"].items()
    }

    for p in PREFIXES:
        add_check(
            f"P{p}_2015_count",
            counts_by_year_prefix[(2015, p)] == pred_year_counts[2015],
            counts_by_year_prefix[(2015, p)],
            pred_year_counts[2015],
        )
        add_check(
            f"P{p}_2016_count",
            counts_by_year_prefix[(2016, p)] == pred_year_counts[2016],
            counts_by_year_prefix[(2016, p)],
            pred_year_counts[2016],
        )
        add_check(
            f"P{p}_2017_count",
            counts_by_year_prefix[(2017, p)] == pred_year_counts[2017],
            counts_by_year_prefix[(2017, p)],
            pred_year_counts[2017],
        )
        add_check(
            f"P{p}_2015_positive",
            labels_by_year_prefix[(2015, p, 1)] == pos_year_counts[2015],
            labels_by_year_prefix[(2015, p, 1)],
            pos_year_counts[2015],
        )
        add_check(
            f"P{p}_2016_positive",
            labels_by_year_prefix[(2016, p, 1)] == pos_year_counts[2016],
            labels_by_year_prefix[(2016, p, 1)],
            pos_year_counts[2016],
        )
        add_check(
            f"P{p}_2017_positive",
            labels_by_year_prefix[(2017, p, 1)] == pos_year_counts[2017],
            labels_by_year_prefix[(2017, p, 1)],
            pos_year_counts[2017],
        )

    # Verify fit-year policies from the actual sample lists.
    add_check(
        "development_mapping_fit_years",
        sorted({s.case_year for s in dev_fit_samples}) == [2015],
        sorted({s.case_year for s in dev_fit_samples}),
        [2015],
    )
    add_check(
        "final_mapping_fit_years",
        sorted({s.case_year for s in final_fit_samples}) == [2015, 2016],
        sorted({s.case_year for s in final_fit_samples}),
        [2015, 2016],
    )
    add_check(
        "formal_test_not_in_any_mapping_fit",
        not any(s.case_year == 2017 for s in dev_fit_samples + final_fit_samples),
        False if any(s.case_year == 2017 for s in dev_fit_samples + final_fit_samples) else True,
        True,
    )

    all_pass = all(v["pass"] for v in checks.values())
    write_json(out_dir / "consistency_checks.json", checks)
    write_json(out_dir / "split_audit.json", split_audit)

    protocol = {
        "stage": STAGE,
        "name": "BPIC2018 Development Feature Build + Frozen Representation Mapping",
        "dataset": "BPIC2018",
        "created_local": datetime.now().astimezone().isoformat(),
        "source_stage55": str(stage55_dir),
        "prefixes": list(PREFIXES),
        "scope": "representation only; no model training; no tuning",
        "development": {
            "mapping_fit_years": [2015],
            "train_years": [2015],
            "validation_years": [2016],
        },
        "final_refit": {
            "mapping_fit_years": [2015, 2016],
            "train_years": [2015, 2016],
            "formal_external_test_years": [2017],
        },
        "formal_test_prohibitions": [
            "no vocabulary fitting on 2017",
            "no numeric scaler fitting on 2017",
            "no feature selection on 2017",
            "no class-weight selection on 2017",
            "no threshold selection on 2017",
            "no hyperparameter selection on 2017",
            "no model architecture selection on 2017",
        ],
        "tabular_representation": {
            "numeric_features": NUMERIC_FEATURE_NAMES,
            "categorical_features": [
                "activity count vector",
                "doctype count vector",
                "subprocess count vector",
            ],
        },
        "sequence_representation": {
            "categorical_sequences": ["activity", "doctype", "subprocess"],
            "temporal_sequence": "log1p nonnegative inter-event delta seconds",
            "padding": {"token": PAD_TOKEN, "id": PAD_ID},
            "unknown": {"token": UNK_TOKEN, "id": UNK_ID},
        },
        "downstream_models": ["RF", "XGB", "MLP", "LSTM"],
        "label_driven_feature_selection": False,
        "all_hard_checks_pass": all_pass,
    }
    write_json(out_dir / "protocol_frozen.json", protocol)

    summary = {
        "stage": STAGE,
        "dataset": "BPIC2018",
        "total_case_prefix_samples": len(samples),
        "prediction_year_counts": pred_year_counts,
        "positive_by_year_prediction": pos_year_counts,
        "development_mapping": {
            "fit_years": [2015],
            "activity_vocab_size": len(dev_mapping["activity_vocab"]),
            "doctype_vocab_size": len(dev_mapping["doctype_vocab"]),
            "subprocess_vocab_size": len(dev_mapping["subprocess_vocab"]),
            "tabular_dim": len(feature_names_dev),
        },
        "final_mapping": {
            "fit_years": [2015, 2016],
            "activity_vocab_size": len(final_mapping["activity_vocab"]),
            "doctype_vocab_size": len(final_mapping["doctype_vocab"]),
            "subprocess_vocab_size": len(final_mapping["subprocess_vocab"]),
            "tabular_dim": len(feature_names_final),
        },
        "outputs": output_records,
        "all_hard_checks_pass": all_pass,
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    write_json(out_dir / "dataset_summary.json", summary)

    md = f"""# ETCE Stage56 — BPIC2018 Frozen Representation Mapping

## Leakage control

### Development mapping
- Fit preprocessing on **2015 only**
- Train data: **2015**
- Validation data: **2016**

### Final-refit mapping
- Fit preprocessing on **2015 + 2016 only**
- Final train data: **2015 + 2016**
- Formal external TEST: **2017**

The 2017 formal TEST is not used for vocabulary construction, scaler fitting,
feature selection, class-weight selection, threshold selection, hyperparameter
selection, or architecture selection.

## Frozen prefixes
P5 / P10 / P15 / P20.

## Shared outputs
Each prefix produces four NPZ files:
- development train (2015)
- development validation (2016)
- final train (2015+2016)
- formal external TEST (2017)

Each NPZ contains both:
- `X_tabular` for RF/XGB/MLP
- sequence tensors for LSTM

Stage56 gate: **{"PASS" if all_pass else "FAIL"}**
"""
    (out_dir / "STAGE56_FROZEN_REPRESENTATION.md").write_text(md, encoding="utf-8")

    # Hash all Stage56 outputs except the hash manifest itself.
    files_to_hash = sorted(
        p for p in out_dir.iterdir()
        if p.is_file() and p.name != "sha256_manifest.json"
    )
    hash_manifest = {
        "stage": STAGE,
        "algorithm": "SHA256",
        "generated_local": datetime.now().astimezone().isoformat(),
        "files": {
            str(p): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in files_to_hash
        },
    }
    write_json(out_dir / "sha256_manifest.json", hash_manifest)

    print("\n" + "=" * 118)
    print("ETCE STAGE56 SUMMARY")
    print("=" * 118)
    print(f"Prediction cases/year : {pred_year_counts}")
    print(f"Positive/year         : {pos_year_counts}")
    print(
        "Development vocab    : "
        f"activity={len(dev_mapping['activity_vocab'])}, "
        f"doctype={len(dev_mapping['doctype_vocab'])}, "
        f"subprocess={len(dev_mapping['subprocess_vocab'])}"
    )
    print(f"Development tab dim  : {len(feature_names_dev)}")
    print(
        "Final vocab          : "
        f"activity={len(final_mapping['activity_vocab'])}, "
        f"doctype={len(final_mapping['doctype_vocab'])}, "
        f"subprocess={len(final_mapping['subprocess_vocab'])}"
    )
    print(f"Final tab dim        : {len(feature_names_final)}")
    for p in PREFIXES:
        a = split_audit[f"P{p}"]
        print(
            f"P{p}: "
            f"train2015={a['development_train_2015']['n']:,} "
            f"(+{a['development_train_2015']['positive']}), "
            f"valid2016={a['development_valid_2016']['n']:,} "
            f"(+{a['development_valid_2016']['positive']}), "
            f"test2017={a['formal_external_test_2017']['n']:,} "
            f"(+{a['formal_external_test_2017']['positive']})"
        )
        print(
            f"    valid UNK rates: {a['development_valid_2016']['unknown_rates_vs_2015_mapping']}"
        )
        print(
            f"    test  UNK rates: {a['formal_external_test_2017']['unknown_rates_vs_2015_2016_mapping']}"
        )
    print(f"Hard gates          : {'PASS' if all_pass else 'FAIL'}")
    print(f"SHA256              : {out_dir / 'sha256_manifest.json'}")
    print("=" * 118)

    if not all_pass:
        failed = {k: v for k, v in checks.items() if not v["pass"]}
        print("[FAILED CHECKS]")
        for k, v in failed.items():
            print(f"  - {k}: actual={v['actual']} expected={v['expected']}")
        print("\nSTAGE56 FAILED. Do not proceed to model training.")
        return 2

    print(
        "\nSTAGE56 FROZEN SUCCESSFULLY.\n"
        "NEXT: Stage57 may train/tune RF/XGB/MLP/LSTM on 2015->2016 development only. "
        "2017 remains formal external TEST and must not be used for selection."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
