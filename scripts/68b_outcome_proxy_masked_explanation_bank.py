#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Stage 68B: outcome-proxy-masked explanation evidence bank.

Preconditions
-------------
1. Phase 4B-v2 validation multi-rule bank independently PASSED.
2. Phase 4C-v2 D-S protocol is frozen on validation only.
3. Phase 4D-0 formal TEST case selection is already frozen.

Sensitivity target set
----------------------
The same 100 frozen formal TEST cases × 4 black boxes × Prefix-10/15/20
= 1200 targets. Prefix-5 is omitted because Stage 68A audited zero affected
cases at P5.

Only the activity representation and corresponding frozen black boxes change.
For strict comparability, every original Phase-4D process-valid local pool is
loaded from its frozen cache, its three historical offer-outcome identities
are mapped to O_MASKED_OUTCOME, and exact duplicate masked sequences are
removed deterministically. Black-box labels and the deterministic
class-stratified 70/30 local fit/eval split are then recomputed. LORE / DT / BN,
multi-rule extraction, support and fidelity definitions remain unchanged.

No TEST label is used for case selection or rule extraction.

Efficiency extension
--------------------
The script also stores the already-computed local pools, black-box labels,
and exact fit/eval indices in gzip JSON target files. This does NOT alter
the protocol. It only prevents an expensive second reconstruction before
formal frozen D-S fusion.

Run
---
    python -u scripts/68b_outcome_proxy_masked_explanation_bank.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import threading
import time

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ETCE_ROOT", DEFAULT_ROOT)).expanduser().resolve()

LOCAL_DEPS = ROOT / ".deps"
if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas
import scipy
import sklearn
import torch
import xgboost

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase2.models import ActivityLSTM, TabularMLP
from etce.phase4.surrogate_core import sequence_hash
from etce.phase4.stratified_core import run_all_surrogates_stratified
from etce.utils.io import (
    read_json,
    write_csv,
    write_json,
    write_json_atomic,
)

RESULTS3 = ROOT / "results/phase3"

FROZEN_DS = ROOT / "configs/phase4c_ds_frozen_config.json"

PHASE4D = ROOT / "results/phase4/phase4d_formal_test"
SELECTION = PHASE4D / "formal_test_case_selection.json"
ORIGINAL_BANK = PHASE4D / "formal_test_multirule_bank"
ORIGINAL_EVAL_CACHE = ORIGINAL_BANK / "fusion_eval_cache" / "targets"

STAGE68A = ROOT / "results/stage68_outcome_proxy_masking/blackbox"
MASKED_CANONICAL = (
    STAGE68A / "canonical_blackboxes/canonical_manifest_masked.json"
)
STAGE68A_AUDIT = STAGE68A / "audit/validation_stage68a_masked_blackbox.json"

OUT = ROOT / "results/stage68_outcome_proxy_masking/explanation_bank"
CACHE = OUT / "target_cache"
EVAL_CACHE = OUT / "fusion_eval_cache" / "targets"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SURROGATES = ("LORE", "DT", "BN")
MODEL_STEMS = {
    "RF": "rf",
    "XGBoost": "xgboost",
    "MLP": "mlp",
    "LSTM": "lstm",
}

TEST_CASE_COUNT = 100
SENSITIVITY_PREFIXES = (10, 15, 20)
MASKED_ACTIVITIES = frozenset(
    {"O_Accepted", "O_Refused", "O_Cancelled"}
)
MASK_TOKEN = "O_MASKED_OUTCOME"

CACHE_SCHEMA = "stage68b-masked-multirule-target-v1"
EVAL_CACHE_SCHEMA = "stage68b-masked-fusion-eval-target-v1"
PREDICTION_REPRODUCTION_TOL = 1e-6

_STATUS = {
    "target": "startup",
    "stage": "initializing",
    "completed": 0,
    "total": 0,
}
_STATUS_LOCK = threading.Lock()
_STOP_HEARTBEAT = threading.Event()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class MaskedCanonicalPredictor:
    """Load a Stage-68A checkpoint with its masked train-only encoders."""

    def __init__(self, root, model_name, prefix, entry):
        self.root = Path(root)
        self.model_name = model_name
        self.prefix = int(prefix)
        self.seed = int(entry["selected_seed"])
        self.config = dict(entry["hyperparameters"])
        self.encoder = read_json(self.root / entry["encoder_metadata"])
        self.tabular_scaler = joblib.load(self.root / entry["tabular_scaler"])
        self.time_scaler = joblib.load(self.root / entry["sequence_time_scaler"])
        self.checkpoint = self.root / entry["canonical_checkpoint"]
        require(self.checkpoint.is_file(), f"Masked checkpoint missing: {self.checkpoint}")
        require(
            int(self.encoder["prefix"]) == self.prefix,
            "Masked encoder prefix mismatch",
        )
        require(MASK_TOKEN in self.encoder["vocabulary"], "Mask token absent from encoder")
        self.activity_to_id = {
            str(k): int(v) for k, v in self.encoder["activity_to_id"].items()
        }

        if model_name in ("RF", "XGBoost"):
            self.model = joblib.load(self.checkpoint)
        else:
            try:
                payload = torch.load(
                    self.checkpoint, map_location="cpu", weights_only=False
                )
            except TypeError:  # older PyTorch
                payload = torch.load(self.checkpoint, map_location="cpu")
            require(payload.get("model_name") == model_name, "Torch model-name mismatch")
            require(int(payload.get("prefix")) == self.prefix, "Torch prefix mismatch")
            self.config = dict(payload["config"])
            if model_name == "MLP":
                self.model = TabularMLP(
                    int(self.encoder["tabular_feature_count"]),
                    tuple(self.config["hidden_dims"]),
                    float(self.config["dropout"]),
                )
            elif model_name == "LSTM":
                self.model = ActivityLSTM(
                    int(payload["encoder_vocabulary_size"]),
                    int(self.config["embedding_dim"]),
                    2,
                    int(self.config["hidden_size"]),
                    int(self.config["num_layers"]),
                    float(self.config["dropout"]),
                    bool(self.config["bidirectional"]),
                )
            else:
                raise ValueError(model_name)
            self.model.load_state_dict(payload["state_dict"])
            self.model.eval()

    @staticmethod
    def _seconds(delta):
        return float(delta.total_seconds())

    def _encode(self, sequences, timestamps):
        require(len(sequences) == len(timestamps), "Sequence/timestamp batch mismatch")
        tabular_rows = []
        activity_rows = []
        time_rows = []
        vocabulary_size = len(self.encoder["vocabulary"]) + 1

        for sequence, times in zip(sequences, timestamps):
            sequence = tuple(sequence)
            times = tuple(times)
            require(len(sequence) == self.prefix, "Sequence length mismatch")
            require(len(times) == self.prefix, "Timestamp length mismatch")
            ids = np.asarray(
                [self.activity_to_id.get(str(activity), 0) for activity in sequence],
                dtype=np.int64,
            )
            counts = np.bincount(ids, minlength=vocabulary_size).astype(np.float64)
            frequency = counts / self.prefix
            last = np.zeros(vocabulary_size, dtype=np.float64)
            first = np.zeros(vocabulary_size, dtype=np.float64)
            last[ids[-1]] = 1.0
            first[ids[0]] = 1.0
            deltas = np.asarray(
                [0.0]
                + [self._seconds(b - a) for a, b in zip(times, times[1:])],
                dtype=np.float64,
            )
            elapsed = np.asarray(
                [self._seconds(value - times[0]) for value in times],
                dtype=np.float64,
            )
            intervals = deltas[1:]
            temporal = np.asarray(
                [
                    elapsed[-1], intervals.mean(), intervals.std(), intervals.max(),
                    deltas[-1], times[-1].hour, times[-1].weekday(),
                ],
                dtype=np.float64,
            )
            tabular_rows.append(
                np.concatenate((counts, frequency, last, first, temporal))
            )
            activity_rows.append(ids)
            time_rows.append(np.column_stack((deltas, elapsed)))

        tabular = self.tabular_scaler.transform(np.vstack(tabular_rows)).astype(np.float32)
        activities = np.stack(activity_rows).astype(np.int64)
        time_raw = np.vstack(time_rows)
        time_features = self.time_scaler.transform(time_raw).reshape(
            len(sequences), self.prefix, 2
        ).astype(np.float32)
        return tabular, activities, time_features

    def predict_proba(self, sequences, timestamps):
        tabular, activities, time_features = self._encode(sequences, timestamps)
        if self.model_name in ("RF", "XGBoost"):
            return np.asarray(self.model.predict_proba(tabular)[:, 1], dtype=float)

        batch_size = int(self.config.get("batch_size", 256))
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                stop = min(len(sequences), start + batch_size)
                if self.model_name == "MLP":
                    logits = self.model(torch.from_numpy(tabular[start:stop]))
                else:
                    logits = self.model(
                        torch.from_numpy(activities[start:stop]),
                        torch.from_numpy(time_features[start:stop]),
                    )
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probabilities).astype(float)


def fmt_seconds(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def set_status(target=None, stage=None, completed=None, total=None):
    with _STATUS_LOCK:
        if target is not None:
            _STATUS["target"] = target
        if stage is not None:
            _STATUS["stage"] = stage
        if completed is not None:
            _STATUS["completed"] = int(completed)
        if total is not None:
            _STATUS["total"] = int(total)


def heartbeat(started, interval=30):
    while not _STOP_HEARTBEAT.wait(interval):
        with _STATUS_LOCK:
            s = dict(_STATUS)
        elapsed = time.perf_counter() - started
        total = int(s.get("total", 0))
        completed = int(s.get("completed", 0))
        pct = 0.0 if total == 0 else 100.0 * completed / total
        print(
            f"[HEARTBEAT] {completed}/{total} ({pct:.2f}%) | "
            f"elapsed={fmt_seconds(elapsed)} | "
            f"target={s.get('target')} | stage={s.get('stage')}",
            flush=True,
        )


def stable_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def pool_digest(sequences):
    payload = "\n".join(
        sorted(sequence_hash(seq) for seq in sequences)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token(case_id, model, prefix):
    return hashlib.sha256(
        f"{case_id}|{model}|{prefix}".encode("utf-8")
    ).hexdigest()[:20]


def target_cache_path(case_id, model, prefix):
    return CACHE / f"{model}_P{prefix}_{token(case_id, model, prefix)}.json"


def eval_cache_path(case_id, model, prefix):
    return EVAL_CACHE / (
        f"{model}_P{prefix}_{token(case_id, model, prefix)}.json.gz"
    )


def original_eval_cache_path(case_id, model, prefix):
    return ORIGINAL_EVAL_CACHE / (
        f"{model}_P{prefix}_{token(case_id, model, prefix)}.json.gz"
    )


def mask_sequence(sequence):
    return tuple(
        MASK_TOKEN if activity in MASKED_ACTIVITIES else activity
        for activity in sequence
    )


def mask_and_deduplicate_pool(sequences):
    """Map proxy identities to one token and retain first occurrence only."""
    masked = []
    seen = set()
    replacements = 0
    for sequence in sequences:
        sequence = tuple(sequence)
        replacements += sum(a in MASKED_ACTIVITIES for a in sequence)
        transformed = mask_sequence(sequence)
        if transformed not in seen:
            seen.add(transformed)
            masked.append(transformed)
    return masked, replacements


def validate_predictor_against_stage68a(predictor, model, prefix, cases):
    prediction_path = (
        STAGE68A
        / "predictions"
        / f"seed_{predictor.seed}"
        / f"{MODEL_STEMS[model]}_prefix{prefix}_test.csv"
    )
    require(prediction_path.is_file(), f"Stage 68A prediction file missing: {prediction_path}")
    recorded = pandas.read_csv(prediction_path).set_index("case_id")
    ids = [case.case_id for case in cases]
    require(set(ids).issubset(recorded.index.astype(str)), "Stage 68A prediction IDs missing")
    sequences = [mask_sequence(case.activities[:prefix]) for case in cases]
    timestamps = [tuple(case.timestamps[:prefix]) for case in cases]
    recomputed = predictor.predict_proba(sequences, timestamps)
    expected = recorded.loc[ids, "predicted_probability"].to_numpy(dtype=float)
    expected_labels = recorded.loc[ids, "predicted_label"].to_numpy(dtype=int)
    recomputed_labels = (recomputed >= 0.5).astype(int)
    max_diff = float(np.max(np.abs(recomputed - expected)))
    require(
        np.array_equal(recomputed_labels, expected_labels),
        f"Masked predictor class labels changed for {model}/P{prefix}",
    )
    require(
        max_diff <= PREDICTION_REPRODUCTION_TOL,
        (
            f"Masked predictor reproduction failed for {model}/P{prefix}: "
            f"max diff={max_diff}, tolerance={PREDICTION_REPRODUCTION_TOL}"
        ),
    )
    return max_diff


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_gzip_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(
        tmp,
        "wt",
        encoding="utf-8",
        compresslevel=5,
    ) as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    os.replace(tmp, path)


def cache_is_complete(path, case_id, model, prefix):
    try:
        p = read_json(path)
    except Exception:
        return False
    return (
        p.get("schema_version") == CACHE_SCHEMA
        and p.get("case_id") == case_id
        and p.get("model") == model
        and int(p.get("prefix", -1)) == int(prefix)
        and isinstance(p.get("metrics"), list)
        and len(p["metrics"]) == len(SURROGATES)
        and isinstance(p.get("rules"), list)
        and isinstance(p.get("pool_digest"), str)
    )


def eval_cache_is_complete(path, case_id, model, prefix, expected_digest):
    try:
        p = read_gzip_json(path)
    except Exception:
        return False
    return (
        p.get("schema_version") == EVAL_CACHE_SCHEMA
        and p.get("case_id") == case_id
        and p.get("model") == model
        and int(p.get("prefix", -1)) == int(prefix)
        and p.get("pool_digest") == expected_digest
        and isinstance(p.get("pool_sequences"), list)
        and isinstance(p.get("pool_labels"), list)
        and isinstance(p.get("fit_indices"), list)
        and isinstance(p.get("eval_indices"), list)
    )


def rule_core(rule):
    if rule is None:
        return None
    return {
        k: v
        for k, v in rule.items()
        if k != "candidate_id"
    }


def legacy_matches_first(best_rule, cf_rules):
    if best_rule is None and not cf_rules:
        return True
    if best_rule is None or not cf_rules:
        return False
    return rule_core(cf_rules[0]) == rule_core(best_rule)


def rule_record(
    case_id,
    model,
    prefix,
    surrogate,
    candidate_id,
    rule_type,
    rule,
    is_legacy_best=False,
):
    if rule is None:
        return None

    support = int(rule["support"])
    fidelity = rule["fidelity"]

    return {
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "candidate_id": str(candidate_id),
        "rule_type": rule_type,
        "predicted_class": int(rule["predicted_class"]),
        "conditions": rule["conditions"],
        "n_r": support,
        "Fid_r": fidelity,
        "coverage_eval": float(rule["coverage"]),
        "rule_length": int(rule["rule_length"]),
        "witness_sequence": rule.get("witness_sequence"),
        "witness_edit_count": rule.get("witness_edit_count"),
        "is_legacy_best": bool(is_legacy_best),
        "eligible_for_reliability": bool(
            support > 0 and fidelity is not None
        ),
    }


def require_inputs():
    required = [
        ROOT / "data/raw/BPI Challenge 2017.xes.gz",
        ROOT / "results/phase1_2/audit/common_cohort_case_ids.csv",
        FROZEN_DS,
        SELECTION,
        MASKED_CANONICAL,
        STAGE68A_AUDIT,
        ORIGINAL_BANK / "fusion_eval_cache_manifest.json",
    ]
    missing = [
        p.relative_to(ROOT).as_posix()
        for p in required
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required inputs: " + ", ".join(missing)
        )

    frozen = read_json(FROZEN_DS)
    require(frozen.get("status") == "frozen", "D-S protocol is not frozen")
    require(float(frozen.get("lambda")) == 5.0, "Frozen lambda is not 5")
    require(
        frozen.get("selection_scope") == "validation_only",
        "Frozen lambda was not selected validation-only",
    )
    require(
        frozen.get("formal_test_seen_before_freeze") is False,
        "Frozen config indicates formal test was seen before freeze",
    )

    stage68a = read_json(STAGE68A_AUDIT)
    require(stage68a.get("status") == "passed", "Stage 68A is not PASSED")
    require(
        [int(v) for v in stage68a.get("rerun_prefixes", [])]
        == list(SENSITIVITY_PREFIXES),
        "Stage 68A prefix set mismatch",
    )
    mask_audit = stage68a.get("mask_audit", {})
    require(
        mask_audit.get("prefix5_unchanged_by_construction") is True,
        "Stage 68A did not establish that P5 is unaffected",
    )

    original_cache = read_json(ORIGINAL_BANK / "fusion_eval_cache_manifest.json")
    require(original_cache.get("status") == "built", "Original Phase-4D cache is not built")
    require(int(original_cache.get("target_count", -1)) == 1600, "Original cache target count != 1600")
    require(original_cache.get("true_labels_used") is False, "Original cache used true labels")


def main():
    started = time.perf_counter()
    require_inputs()

    frozen = read_json(FROZEN_DS)
    selection = read_json(SELECTION)

    require(selection.get("status") == "frozen", "Formal test selection not frozen")
    require(
        selection.get("source_split") == "test",
        "Formal selection is not TEST",
    )
    require(
        int(selection.get("selected_case_count")) == TEST_CASE_COUNT,
        "Unexpected formal test case count",
    )
    require(
        selection.get("selection_uses_true_label") is False,
        "TEST labels influenced case selection",
    )
    require(
        selection.get("selection_uses_blackbox_prediction") is False
        and selection.get("selection_uses_blackbox_probability") is False,
        "Black-box outputs influenced TEST case selection",
    )
    require(
        selection.get("selection_uses_explanation_output") is False
        and selection.get("selection_uses_surrogate_output") is False,
        "Explanation outputs influenced TEST case selection",
    )
    require(
        [int(v) for v in selection["prefixes"]]
        == [int(v) for v in PREFIXES],
        "Formal selection prefixes differ from frozen Phase-2 prefixes",
    )
    require(
        list(selection["models"]) == list(MODELS),
        "Formal selection models differ",
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    case_by_id = {c.case_id: c for c in cases}

    selected_ids = list(selection["selected_case_ids"])
    require(
        len(selected_ids) == TEST_CASE_COUNT
        and len(set(selected_ids)) == TEST_CASE_COUNT,
        "Frozen TEST selection IDs are not 100 unique cases",
    )

    selected_cases = []
    for cid in selected_ids:
        require(cid in case_by_id, f"Selected case missing from cohort: {cid}")
        case = case_by_id[cid]
        require(case.split == "test", f"Selected case is not TEST: {cid}")
        selected_cases.append(case)

    # Independently reproduce Phase 4D-0 deterministic selection.
    ranked_test = sorted(
        (c for c in cases if c.split == "test"),
        key=lambda c: (stable_rank(c.case_id), c.case_id),
    )
    expected_ids = [c.case_id for c in ranked_test[:TEST_CASE_COUNT]]
    require(
        selected_ids == expected_ids,
        "Frozen Phase 4D-0 TEST selection is not reproducible",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)

    write_json(
        OUT / "runtime_formal_test_multirule.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {
                "joblib": joblib.__version__,
                "numpy": np.__version__,
                "pandas": pandas.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
                "torch": torch.__version__,
                "xgboost": xgboost.__version__,
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    )

    write_json(
        OUT / "formal_test_run_config.json",
        {
            "scope": "Stage 68B outcome-proxy-masked formal TEST evidence bank",
            "selected_case_count": TEST_CASE_COUNT,
            "formal_target_count": (
                TEST_CASE_COUNT * len(MODELS) * len(SENSITIVITY_PREFIXES)
            ),
            "models": list(MODELS),
            "prefixes": list(SENSITIVITY_PREFIXES),
            "surrogates": list(SURROGATES),
            "mask_token": MASK_TOKEN,
            "masked_activities": sorted(MASKED_ACTIVITIES),
            "prefix5_omitted_because_stage68a_affected_count_is_zero": True,
            "neighborhood_source": "frozen original Phase-4D process-valid pools",
            "mask_applied_after_process_valid_pool_generation": True,
            "masked_duplicate_sequences_removed_deterministically": True,
            "stratified_local_split": True,
            "same_phase4b_v2_neighborhood_protocol": True,
            "same_phase4b_v2_blackbox_protocol": False,
            "blackbox_change": "Stage 68A masked canonical black boxes",
            "same_phase4b_v2_surrogate_definitions": True,
            "same_phase4b_v2_surrogate_hyperparameters": True,
            "same_phase4b_v2_process_semantic_constraints": True,
            "same_phase4b_v2_multirule_extraction": True,
            "heldout_eval_rule_metrics": True,
            "frozen_lambda": float(frozen["lambda"]),
            "lambda_selected_here": False,
            "ds_fusion_performed_here": False,
            "test_cases_used": True,
            "test_labels_used_for_selection": False,
            "test_labels_used_for_rule_extraction": False,
            "test_blackbox_outputs_used_for_case_selection": False,
            "fusion_eval_cache_saved": True,
            "hyperparameter_retuning_performed": False,
            "threshold_retuning_performed": False,
        },
    )

    canonical = read_json(MASKED_CANONICAL)
    require(canonical.get("purpose") == "outcome-proxy masking sensitivity", "Masked canonical manifest purpose mismatch")
    require([int(v) for v in canonical.get("prefixes", [])] == list(SENSITIVITY_PREFIXES), "Masked canonical prefix set mismatch")

    expected_targets = TEST_CASE_COUNT * len(MODELS) * len(SENSITIVITY_PREFIXES)
    set_status(completed=0, total=expected_targets)

    hb = threading.Thread(
        target=heartbeat,
        args=(started,),
        daemon=True,
    )
    hb.start()

    done = 0
    reused = 0
    regenerated = 0
    predictor_reproduction_max_abs_diff = {}

    try:
        for model in MODELS:
            for prefix in SENSITIVITY_PREFIXES:
                set_status(
                    target=f"{model}/P{prefix}",
                    stage="load canonical black box",
                )
                predictor = MaskedCanonicalPredictor(
                    ROOT,
                    model,
                    prefix,
                    canonical["items"][model][str(prefix)],
                )
                reproduction_diff = validate_predictor_against_stage68a(
                    predictor, model, prefix, selected_cases
                )
                predictor_reproduction_max_abs_diff[f"{model}/P{prefix}"] = (
                    reproduction_diff
                )

                for case in selected_cases:
                    done += 1
                    label = (
                        f"[{done:04d}/{expected_targets}] "
                        f"{model} P{prefix} {case.case_id}"
                    )
                    set_status(
                        target=label,
                        stage="cache check",
                        completed=done - 1,
                    )

                    tcache = target_cache_path(
                        case.case_id,
                        model,
                        prefix,
                    )

                    if tcache.is_file() and cache_is_complete(
                        tcache,
                        case.case_id,
                        model,
                        prefix,
                    ):
                        p = read_json(tcache)
                        ecache = eval_cache_path(
                            case.case_id,
                            model,
                            prefix,
                        )
                        if eval_cache_is_complete(
                            ecache,
                            case.case_id,
                            model,
                            prefix,
                            p["pool_digest"],
                        ):
                            reused += 1
                            set_status(
                                stage="cached",
                                completed=done,
                            )
                            if done % 25 == 0:
                                elapsed = time.perf_counter() - started
                                print(
                                    f"{label} | cached | "
                                    f"elapsed={fmt_seconds(elapsed)}",
                                    flush=True,
                                )
                            continue

                    target_started = time.perf_counter()

                    original_factual = tuple(case.activities[:prefix])
                    factual = mask_sequence(original_factual)
                    timestamps = tuple(case.timestamps[:prefix])
                    source_cache_path = original_eval_cache_path(
                        case.case_id, model, prefix
                    )
                    require(
                        source_cache_path.is_file(),
                        f"Original Phase-4D cache missing: {source_cache_path}",
                    )
                    source_cache = read_gzip_json(source_cache_path)
                    require(
                        source_cache.get("formal_test_target") is True
                        and source_cache.get("true_label_used") is False,
                        f"Invalid original cache flags: {source_cache_path}",
                    )
                    require(
                        tuple(source_cache.get("factual_sequence", []))
                        == original_factual,
                        f"Original factual sequence mismatch: {case.case_id}/{model}/P{prefix}",
                    )
                    seed = int(source_cache["pool_seed"])

                    set_status(stage="mask and deduplicate frozen original pool")
                    original_pool = [
                        tuple(seq) for seq in source_cache["pool_sequences"]
                    ]
                    pool, replacement_count = mask_and_deduplicate_pool(
                        original_pool
                    )
                    require(
                        all(
                            activity not in MASKED_ACTIVITIES
                            for sequence in pool
                            for activity in sequence
                        ),
                        "An unmasked proxy identity remains in the masked pool",
                    )
                    require(factual in pool, "Masked factual sequence absent from pool")
                    require(len(pool) >= 2, "Masked pool collapsed below two sequences")
                    collapsed_rows = len(original_pool) - len(pool)

                    set_status(stage="black-box predict_proba")
                    probs = np.asarray(
                        predictor.predict_proba(
                            pool,
                            [timestamps] * len(pool),
                        ),
                        dtype=float,
                    )
                    labels = (probs >= 0.5).astype(int)

                    factual_idx = pool.index(factual)
                    factual_prediction = int(labels[factual_idx])

                    set_status(
                        stage="LORE / DT / BN multi-rule extraction"
                    )
                    result = run_all_surrogates_stratified(
                        pool,
                        labels,
                        factual,
                        factual_prediction,
                    )

                    fit_labels = np.asarray(
                        result["fit_labels"],
                        dtype=int,
                    )
                    eval_labels = np.asarray(
                        result["eval_labels"],
                        dtype=int,
                    )

                    metrics = []
                    rules = []
                    total_cf = 0

                    for branch in result["branches"]:
                        surrogate = branch["branch"]

                        require(
                            "counterfactual_rules" in branch,
                            "Phase 4B-v2 multi-rule core is not installed",
                        )

                        fact = branch["factual_rule"]
                        legacy_cf = branch["counterfactual_rule"]
                        cf_rules = list(
                            branch["counterfactual_rules"]
                        )

                        declared = int(
                            branch.get(
                                "counterfactual_rule_count",
                                len(cf_rules),
                            )
                        )
                        require(
                            declared == len(cf_rules),
                            "counterfactual_rule_count mismatch",
                        )

                        candidate_ids = [
                            str(r.get("candidate_id", ""))
                            for r in cf_rules
                        ]
                        expected_candidate_ids = [
                            f"cf_{i:03d}"
                            for i in range(len(cf_rules))
                        ]
                        require(
                            candidate_ids == expected_candidate_ids
                            and len(candidate_ids)
                            == len(set(candidate_ids)),
                            (
                                "Unstable/duplicate candidate_id: "
                                f"{case.case_id}/{model}/P{prefix}/{surrogate}"
                            ),
                        )

                        legacy_match = legacy_matches_first(
                            legacy_cf,
                            cf_rules,
                        )
                        eligible_count = sum(
                            int(r["support"]) > 0
                            and r["fidelity"] is not None
                            for r in cf_rules
                        )

                        metrics.append(
                            {
                                "case_id": case.case_id,
                                "model": model,
                                "prefix": int(prefix),
                                "canonical_seed": int(
                                    predictor.seed
                                ),
                                "pool_seed": int(seed),
                                "surrogate": surrogate,
                                "status": branch["status"],
                                "original_pool_size": len(original_pool),
                                "pool_size": len(pool),
                                "masked_duplicate_rows_removed": collapsed_rows,
                                "masked_activity_replacements": replacement_count,
                                "pool_class0": int(
                                    np.sum(labels == 0)
                                ),
                                "pool_class1": int(
                                    np.sum(labels == 1)
                                ),
                                "fit_size": len(
                                    result["fit_sequences"]
                                ),
                                "eval_size": len(
                                    result["eval_sequences"]
                                ),
                                "fit_class0": int(
                                    np.sum(fit_labels == 0)
                                ),
                                "fit_class1": int(
                                    np.sum(fit_labels == 1)
                                ),
                                "eval_class0": int(
                                    np.sum(eval_labels == 0)
                                ),
                                "eval_class1": int(
                                    np.sum(eval_labels == 1)
                                ),
                                "surrogate_fidelity_eval": (
                                    ""
                                    if branch[
                                        "surrogate_fidelity_eval"
                                    ] is None
                                    else float(
                                        branch[
                                            "surrogate_fidelity_eval"
                                        ]
                                    )
                                ),
                                "factual_rule_available": bool(
                                    fact is not None
                                ),
                                "cf_rule_available": bool(
                                    len(cf_rules) > 0
                                ),
                                "cf_candidate_count": len(
                                    cf_rules
                                ),
                                "cf_eligible_candidate_count": int(
                                    eligible_count
                                ),
                                "cf_multi_rule_available": bool(
                                    len(cf_rules) >= 2
                                ),
                                "legacy_best_rule_available": bool(
                                    legacy_cf is not None
                                ),
                                "legacy_best_matches_first_candidate": bool(
                                    legacy_match
                                ),
                            }
                        )

                        fact_rec = rule_record(
                            case.case_id,
                            model,
                            prefix,
                            surrogate,
                            "fact_000",
                            "factual",
                            fact,
                            False,
                        )
                        if fact_rec is not None:
                            rules.append(fact_rec)

                        for idx, cf in enumerate(cf_rules):
                            rec = rule_record(
                                case.case_id,
                                model,
                                prefix,
                                surrogate,
                                cf["candidate_id"],
                                "counterfactual",
                                cf,
                                is_legacy_best=(
                                    idx == 0 and legacy_match
                                ),
                            )
                            rules.append(rec)

                        total_cf += len(cf_rules)

                    keys = [
                        (
                            r["case_id"],
                            r["model"],
                            r["prefix"],
                            r["surrogate"],
                            r["rule_type"],
                            r["candidate_id"],
                        )
                        for r in rules
                    ]
                    require(
                        len(keys) == len(set(keys)),
                        "Duplicate canonical rule key inside target",
                    )

                    digest = pool_digest(pool)

                    target_payload = {
                        "schema_version": CACHE_SCHEMA,
                        "case_id": case.case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "pool_seed": int(seed),
                        "pool_digest": digest,
                        "original_pool_size": len(original_pool),
                        "pool_size": len(pool),
                        "masked_duplicate_rows_removed": collapsed_rows,
                        "masked_activity_replacements": replacement_count,
                        "pool_class0": int(
                            np.sum(labels == 0)
                        ),
                        "pool_class1": int(
                            np.sum(labels == 1)
                        ),
                        "factual_prediction": factual_prediction,
                        "factual_probability": float(
                            probs[factual_idx]
                        ),
                        "counterfactual_candidate_count": int(
                            total_cf
                        ),
                        "metrics": metrics,
                        "rules": rules,
                    }
                    write_json_atomic(
                        tcache,
                        target_payload,
                    )

                    eval_payload = {
                        "schema_version": EVAL_CACHE_SCHEMA,
                        "case_id": case.case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "pool_seed": int(seed),
                        "pool_digest": digest,
                        "original_pool_size": len(original_pool),
                        "pool_size": len(pool),
                        "masked_duplicate_rows_removed": collapsed_rows,
                        "masked_activity_replacements": replacement_count,
                        "mask_token": MASK_TOKEN,
                        "factual_sequence": list(factual),
                        "factual_prediction": factual_prediction,
                        "factual_probability": float(
                            probs[factual_idx]
                        ),
                        "pool_sequences": [
                            list(seq)
                            for seq in pool
                        ],
                        "pool_labels": [
                            int(v)
                            for v in labels.tolist()
                        ],
                        "fit_indices": [
                            int(v)
                            for v in result["fit_indices"]
                        ],
                        "eval_indices": [
                            int(v)
                            for v in result["eval_indices"]
                        ],
                        "formal_test_target": True,
                        "true_label_used": False,
                        "source_original_eval_cache": str(
                            source_cache_path.relative_to(ROOT).as_posix()
                        ),
                    }
                    write_gzip_json(
                        eval_cache_path(
                            case.case_id,
                            model,
                            prefix,
                        ),
                        eval_payload,
                    )

                    regenerated += 1
                    set_status(
                        stage="target complete",
                        completed=done,
                    )

                    if done % 10 == 0 or done == expected_targets:
                        target_elapsed = (
                            time.perf_counter()
                            - target_started
                        )
                        elapsed = (
                            time.perf_counter()
                            - started
                        )
                        avg = elapsed / done
                        eta = avg * (
                            expected_targets - done
                        )
                        print(
                            f"{label} | pool={len(pool)} | "
                            f"CF={total_cf} | "
                            f"target={target_elapsed:.1f}s | "
                            f"elapsed={fmt_seconds(elapsed)} | "
                            f"ETA={fmt_seconds(eta)}",
                            flush=True,
                        )

    finally:
        _STOP_HEARTBEAT.set()
        hb.join(timeout=1.0)

    # -----------------------------------------------------------------
    # Assemble formal bank from target caches.
    # -----------------------------------------------------------------
    metrics_all = []
    rules_all = []
    manifests = []

    eval_cache_target_count = 0
    eval_cache_total_pool_rows = 0
    eval_cache_total_eval_rows = 0

    for model in MODELS:
        for prefix in SENSITIVITY_PREFIXES:
            for case in selected_cases:
                tcache = target_cache_path(
                    case.case_id,
                    model,
                    prefix,
                )
                require(
                    cache_is_complete(
                        tcache,
                        case.case_id,
                        model,
                        prefix,
                    ),
                    f"Incomplete target cache: {tcache}",
                )
                p = read_json(tcache)

                metrics_all.extend(p["metrics"])
                rules_all.extend(p["rules"])

                manifests.append(
                    {
                        "case_id": p["case_id"],
                        "model": p["model"],
                        "prefix": p["prefix"],
                        "pool_seed": p["pool_seed"],
                        "pool_digest": p["pool_digest"],
                        "original_pool_size": p["original_pool_size"],
                        "pool_size": p["pool_size"],
                        "masked_duplicate_rows_removed": p[
                            "masked_duplicate_rows_removed"
                        ],
                        "masked_activity_replacements": p[
                            "masked_activity_replacements"
                        ],
                        "pool_class0": p["pool_class0"],
                        "pool_class1": p["pool_class1"],
                        "factual_prediction": p[
                            "factual_prediction"
                        ],
                        "factual_probability": p[
                            "factual_probability"
                        ],
                        "counterfactual_candidate_count": p[
                            "counterfactual_candidate_count"
                        ],
                    }
                )

                ecache = eval_cache_path(
                    case.case_id,
                    model,
                    prefix,
                )
                require(
                    eval_cache_is_complete(
                        ecache,
                        case.case_id,
                        model,
                        prefix,
                        p["pool_digest"],
                    ),
                    f"Incomplete eval cache: {ecache}",
                )
                ep = read_gzip_json(ecache)
                eval_cache_target_count += 1
                eval_cache_total_pool_rows += int(
                    ep["pool_size"]
                )
                eval_cache_total_eval_rows += len(
                    ep["eval_indices"]
                )

    require(
        len(manifests) == expected_targets,
        "Formal manifest target count mismatch",
    )
    require(
        len(metrics_all)
        == expected_targets * len(SURROGATES),
        "Formal surrogate metric row count mismatch",
    )
    require(
        eval_cache_target_count == expected_targets,
        "Formal fusion eval cache target count mismatch",
    )

    all_rule_keys = [
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
            r["rule_type"],
            r["candidate_id"],
        )
        for r in rules_all
    ]
    require(
        len(all_rule_keys) == len(set(all_rule_keys)),
        "Duplicate rule key in assembled formal bank",
    )

    write_csv(
        OUT / "neighborhood_manifest_test.csv",
        manifests,
    )
    write_csv(
        OUT / "surrogate_metrics_test.csv",
        metrics_all,
    )

    with (
        OUT / "canonical_rules_test.jsonl"
    ).open("w", encoding="utf-8") as f:
        for r in rules_all:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    cf_rules_all = [
        r
        for r in rules_all
        if r["rule_type"] == "counterfactual"
    ]
    eligible_cf = [
        r
        for r in cf_rules_all
        if bool(r["eligible_for_reliability"])
    ]

    aggregate = []
    for surrogate in SURROGATES:
        group = [
            r
            for r in metrics_all
            if r["surrogate"] == surrogate
        ]
        cf_rows = [
            r
            for r in cf_rules_all
            if r["surrogate"] == surrogate
        ]

        sfid = [
            float(r["surrogate_fidelity_eval"])
            for r in group
            if r["surrogate_fidelity_eval"] != ""
        ]
        supports = [
            float(r["n_r"])
            for r in cf_rows
        ]
        fidelities = [
            float(r["Fid_r"])
            for r in cf_rows
            if r["Fid_r"] is not None
        ]
        counts = [
            int(r["cf_candidate_count"])
            for r in group
        ]

        aggregate.append(
            {
                "surrogate": surrogate,
                "target_surrogate_groups": len(group),
                "mean_surrogate_fidelity_eval": (
                    ""
                    if not sfid
                    else float(np.mean(sfid))
                ),
                "groups_with_cf": int(
                    sum(v > 0 for v in counts)
                ),
                "groups_with_multiple_cf": int(
                    sum(v >= 2 for v in counts)
                ),
                "total_cf_candidates": int(
                    sum(counts)
                ),
                "mean_cf_candidates_per_group": (
                    float(np.mean(counts))
                    if counts
                    else 0.0
                ),
                "max_cf_candidates_in_group": (
                    int(max(counts))
                    if counts
                    else 0
                ),
                "eligible_cf_candidates": int(
                    sum(
                        bool(
                            r["eligible_for_reliability"]
                        )
                        for r in cf_rows
                    )
                ),
                "mean_cf_support_eval": (
                    ""
                    if not supports
                    else float(np.mean(supports))
                ),
                "median_cf_support_eval": (
                    ""
                    if not supports
                    else float(np.median(supports))
                ),
                "mean_cf_fidelity_eval": (
                    ""
                    if not fidelities
                    else float(np.mean(fidelities))
                ),
            }
        )

    write_csv(
        OUT / "surrogate_aggregate_test.csv",
        aggregate,
    )

    counts = [
        int(r["cf_candidate_count"])
        for r in metrics_all
    ]
    distribution = {
        str(k): int(
            sum(v == k for v in counts)
        )
        for k in sorted(set(counts))
    }

    multi_summary = {
        "scope": "Stage 68B outcome-proxy-masked formal TEST multi-rule bank",
        "selected_test_case_count": TEST_CASE_COUNT,
        "target_count": expected_targets,
        "target_surrogate_group_count": len(
            metrics_all
        ),
        "total_canonical_rule_count": len(
            rules_all
        ),
        "total_counterfactual_candidate_count": len(
            cf_rules_all
        ),
        "eligible_counterfactual_candidate_count": len(
            eligible_cf
        ),
        "groups_with_counterfactual_candidate": int(
            sum(v > 0 for v in counts)
        ),
        "groups_with_multiple_counterfactual_candidates": int(
            sum(v >= 2 for v in counts)
        ),
        "max_counterfactual_candidates_in_one_group": (
            int(max(counts))
            if counts
            else 0
        ),
        "counterfactual_candidate_count_distribution": distribution,
        "frozen_lambda": float(
            frozen["lambda"]
        ),
        "lambda_selected_here": False,
        "ds_fusion_performed_here": False,
        "formal_test_cases_used": True,
        "true_labels_used_for_case_selection": False,
        "blackbox_outputs_used_for_case_selection": False,
        "mask_token": MASK_TOKEN,
        "masked_activities": sorted(MASKED_ACTIVITIES),
        "prefixes": list(SENSITIVITY_PREFIXES),
        "original_pool_rows": int(sum(r["original_pool_size"] for r in manifests)),
        "masked_unique_pool_rows": int(sum(r["pool_size"] for r in manifests)),
        "masked_duplicate_rows_removed": int(
            sum(r["masked_duplicate_rows_removed"] for r in manifests)
        ),
        "masked_activity_replacements": int(
            sum(r["masked_activity_replacements"] for r in manifests)
        ),
    }
    write_json(
        OUT / "multi_rule_summary_test.json",
        multi_summary,
    )

    support_values = np.asarray(
        [
            int(r["n_r"])
            for r in eligible_cf
        ],
        dtype=float,
    )

    write_json(
        OUT / "rule_support_summary_test.json",
        {
            "eligible_counterfactual_rule_count": len(
                eligible_cf
            ),
            "support_min": (
                None
                if not len(support_values)
                else float(np.min(support_values))
            ),
            "support_q25": (
                None
                if not len(support_values)
                else float(
                    np.quantile(
                        support_values,
                        0.25,
                    )
                )
            ),
            "support_median": (
                None
                if not len(support_values)
                else float(
                    np.median(support_values)
                )
            ),
            "support_q75": (
                None
                if not len(support_values)
                else float(
                    np.quantile(
                        support_values,
                        0.75,
                    )
                )
            ),
            "support_max": (
                None
                if not len(support_values)
                else float(np.max(support_values))
            ),
            "frozen_lambda": float(
                frozen["lambda"]
            ),
            "lambda_selected_on_validation": True,
            "lambda_selected_here": False,
        },
    )

    write_json(
        OUT / "fusion_eval_cache_manifest.json",
        {
            "schema_version": EVAL_CACHE_SCHEMA,
            "status": "built",
            "target_count": eval_cache_target_count,
            "total_pool_rows": eval_cache_total_pool_rows,
            "total_eval_rows": eval_cache_total_eval_rows,
            "same_targets_as_formal_rule_bank": True,
            "true_labels_used": False,
            "purpose": (
                "re-use exact formal local held-out evaluation "
                "subsets for frozen Stage1/Stage2/Stage3 D-S"
            ),
        },
    )

    write_json(
        OUT / "validation_stage68b_masked_multirule.json",
        {
            "stage": "68B",
            "status": "passed",
            "scope": "internal structural validation of masked sensitivity bank",
            "formal_test_target_count": expected_targets,
            "target_surrogate_group_count": len(metrics_all),
            "canonical_rule_count": len(rules_all),
            "eligible_counterfactual_candidate_count": len(eligible_cf),
            "prefixes": list(SENSITIVITY_PREFIXES),
            "models": list(MODELS),
            "surrogates": list(SURROGATES),
            "frozen_lambda": float(frozen["lambda"]),
            "lambda_selected_here": False,
            "threshold_tuned_here": False,
            "true_labels_used": False,
            "checks": {
                "all_1200_targets_present": len(manifests) == expected_targets,
                "all_3600_target_surrogate_rows_present": len(metrics_all)
                == expected_targets * len(SURROGATES),
                "canonical_rule_keys_unique": len(all_rule_keys)
                == len(set(all_rule_keys)),
                "fusion_eval_cache_complete": eval_cache_target_count
                == expected_targets,
                "unmasked_proxy_identities_absent": True,
                "masked_predictors_reproduce_stage68a_predictions": max(
                    predictor_reproduction_max_abs_diff.values(), default=0.0
                )
                <= PREDICTION_REPRODUCTION_TOL,
            },
            "predictor_reproduction_max_abs_diff": (
                predictor_reproduction_max_abs_diff
            ),
        },
    )

    def fmt(v):
        if v == "":
            return "-"
        return f"{float(v):.4f}"

    report = [
        "# ETCE Stage 68B Outcome-Proxy-Masked Explanation Bank",
        "",
        "> Sensitivity evidence bank using the validation-frozen explanation and fusion protocol.",
        "",
        "## Frozen protocol",
        "",
        f"- Frozen lambda: {float(frozen['lambda']):g}.",
        f"- Selected TEST cases: {TEST_CASE_COUNT}.",
        "- Prefixes: 10, 15, 20 (P5 is unaffected and is omitted).",
        "- Black-boxes: RF, XGBoost, MLP, LSTM.",
        "- Surrogates: LORE, DT, BN.",
        "- Local neighborhoods: the original frozen process-valid pools, masked and deterministically deduplicated.",
        "- Masked black boxes: Stage 68A canonical seeds and frozen threshold 0.50.",
        "- Surrogate definitions, hyperparameters and fidelity evaluation: unchanged.",
        "- TEST true labels were not used for target selection or rule extraction.",
        "- D-S fusion is not performed in this script.",
        "",
        "## Bank diagnostics",
        "",
        f"- Formal local targets: {expected_targets}.",
        f"- Target-surrogate groups: {len(metrics_all)}.",
        f"- Total canonical rules: {len(rules_all)}.",
        f"- Total CF candidates: {len(cf_rules_all)}.",
        f"- Eligible CF candidates: {len(eligible_cf)}.",
        f"- Groups with >=2 CF candidates: {multi_summary['groups_with_multiple_counterfactual_candidates']}.",
        f"- Maximum CF candidates in one group: {multi_summary['max_counterfactual_candidates_in_one_group']}.",
        f"- Original pool rows: {multi_summary['original_pool_rows']}.",
        f"- Unique masked pool rows: {multi_summary['masked_unique_pool_rows']}.",
        f"- Duplicate rows removed after masking: {multi_summary['masked_duplicate_rows_removed']}.",
        "",
        "| Surrogate | Groups | Surrogate fidelity | Groups >=2 CF | Total CF | Mean CF/group | Max CF | Eligible CF | Mean support | Median support | Mean fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in aggregate:
        report.append(
            f"| {row['surrogate']} | "
            f"{row['target_surrogate_groups']} | "
            f"{fmt(row['mean_surrogate_fidelity_eval'])} | "
            f"{row['groups_with_multiple_cf']} | "
            f"{row['total_cf_candidates']} | "
            f"{row['mean_cf_candidates_per_group']:.4f} | "
            f"{row['max_cf_candidates_in_group']} | "
            f"{row['eligible_cf_candidates']} | "
            f"{fmt(row['mean_cf_support_eval'])} | "
            f"{fmt(row['median_cf_support_eval'])} | "
            f"{fmt(row['mean_cf_fidelity_eval'])} |"
        )

    report += [
        "",
        "## Formal fusion cache",
        "",
        f"- Cached targets: {eval_cache_target_count}.",
        f"- Total local pool rows: {eval_cache_total_pool_rows}.",
        f"- Total held-out eval rows: {eval_cache_total_eval_rows}.",
        "",
        "## Stopping point",
        "",
        "Run Stage 68C grouping audit and Stage 68D frozen D-S fusion next. Do not change lambda, BPA, grouping, fusion order, conflict policy, or tie-breaking.",
        "",
    ]

    (
        OUT / "STAGE68B_MASKED_EXPLANATION_BANK_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("ETCE STAGE 68B MASKED EXPLANATION BANK PASSED")
    print("=" * 108)
    print(f"Selected TEST cases: {TEST_CASE_COUNT}")
    print(f"Formal targets: {expected_targets}")
    print(f"Target-surrogate groups: {len(metrics_all)}")
    print(f"Canonical rules: {len(rules_all)}")
    print(f"CF candidates: {len(cf_rules_all)}")
    print(f"Eligible CF candidates: {len(eligible_cf)}")
    print(
        "Groups with >=2 CF candidates: "
        f"{multi_summary['groups_with_multiple_counterfactual_candidates']}"
    )
    print(
        "Max CF candidates/group: "
        f"{multi_summary['max_counterfactual_candidates_in_one_group']}"
    )
    print(f"Fusion eval cache targets: {eval_cache_target_count}")
    print(f"Frozen lambda: {float(frozen['lambda']):g}")
    print(f"Reused target caches this run: {reused}")
    print(f"Regenerated targets this run: {regenerated}")
    print(f"Output: {OUT}")
    print()
    print("NEXT: Stage 68C masked grouping audit.")
    print("Do NOT alter the frozen D-S protocol.")


if __name__ == "__main__":
    main()
