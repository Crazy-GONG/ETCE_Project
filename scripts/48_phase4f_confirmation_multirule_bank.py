#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-1: untouched confirmation multi-rule bank.

Preconditions
-------------
1. Phase 4B-v2 validation multi-rule bank independently PASSED.
2. Phase 4C-v2 D-S protocol is frozen on validation only.
3. Phase 4D-0 original 100-case selection is frozen.
4. Phase 4E selective Stage-3 gate is frozen at 0.50.
5. Phase 4F-0 new 200-case confirmation selection is frozen.

Confirmation target set
-----------------------
200 untouched TEST cases × 4 black-boxes × Prefix-5/10/15/20 = 3200 targets.
The cases are global deterministic SHA-256 ranks 101--300 and have zero
overlap with the original Phase-4D ranks 1--100.

Protocol is IDENTICAL to validated Phase 4B-v2:
- same CanonicalPredictor black boxes;
- same process-valid local-neighborhood generator;
- same pool_size=320, max_edits=3, max_options=8;
- same deterministic model/prefix/case local seed;
- same deterministic class-stratified 70/30 local fit/eval split;
- same LORE / DT / BN definitions and hyperparameters;
- same multi-rule extraction;
- same held-out local eval support n_r, fidelity Fid(r), coverage.

No TEST label, model output, explanation output, or observed Phase-4D/4E
performance is used for confirmation-case selection or rule extraction.

Efficiency extension
--------------------
The script also stores the already-computed local pools, black-box labels,
and exact fit/eval indices in gzip JSON target files. This does NOT alter
the protocol. It only prevents an expensive second reconstruction before
confirmation fusion and the pre-frozen selective Stage-3 analysis.

Run
---
    python -u scripts/48_phase4f_confirmation_multirule_bank.py
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
from etce.phase3.core import CanonicalPredictor, load_process_constraints
from etce.phase4.surrogate_core import (
    generate_process_valid_pool,
    sequence_hash,
)
from etce.phase4.stratified_core import run_all_surrogates_stratified
from etce.utils.io import (
    read_json,
    write_csv,
    write_json,
    write_json_atomic,
)

RESULTS3 = ROOT / "results/phase3"

PHASE4B_V2 = ROOT / "results/phase4/phase4b_v2_multirule_bank"
FROZEN_DS = ROOT / "configs/phase4c_ds_frozen_config.json"

PHASE4D = ROOT / "results/phase4/phase4d_formal_test"
ORIGINAL_SELECTION = PHASE4D / "formal_test_case_selection.json"
PHASE4E = ROOT / "results/phase4/phase4e_selective_stage3"
ADAPTIVE_CONFIG = PHASE4E / "selective_stage3_frozen_config.json"
PHASE4F = ROOT / "results/phase4/phase4f_confirmation"
SELECTION = PHASE4F / "confirmation_case_selection.json"

OUT = PHASE4F / "confirmation_multirule_bank"
CACHE = OUT / "target_cache"
EVAL_CACHE = OUT / "fusion_eval_cache" / "targets"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SURROGATES = ("LORE", "DT", "BN")

ORIGINAL_CASE_COUNT = 100
CONFIRMATION_CASE_COUNT = 200
SELECTIVE_STAGE3_THRESHOLD = 0.50
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0

CACHE_SCHEMA = "phase4f-confirmation-multirule-target-v1"
EVAL_CACHE_SCHEMA = "phase4f-confirmation-fusion-eval-target-v1"

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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_seed(model, prefix, case_id):
    text = f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


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
        ROOT / "results/phase2b/logs/validation_phase2b.json",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "phase3b/validation_phase3b.json",
        PHASE4B_V2 / "validation_phase4b_v2.json",
        FROZEN_DS,
        ORIGINAL_SELECTION,
        ADAPTIVE_CONFIG,
        SELECTION,
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

    for name, path in (
        ("phase2b", required[2]),
        ("phase3b", required[4]),
        ("phase4b_v2", required[5]),
    ):
        require(
            read_json(path).get("status") == "passed",
            f"Required validated phase is not PASSED: {name}",
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

    adaptive = read_json(ADAPTIVE_CONFIG)
    require(
        adaptive.get("status") == "validation_frozen_formal_exploratory",
        "Selective Stage-3 protocol is not frozen",
    )
    require(
        float(adaptive.get("gate", {}).get("pre_mean_fidelity_threshold"))
        == SELECTIVE_STAGE3_THRESHOLD,
        "Selective Stage-3 threshold is not 0.50",
    )


def main():
    started = time.perf_counter()
    require_inputs()

    frozen = read_json(FROZEN_DS)
    original_selection = read_json(ORIGINAL_SELECTION)
    adaptive = read_json(ADAPTIVE_CONFIG)
    selection = read_json(SELECTION)

    require(selection.get("status") == "frozen", "Confirmation selection not frozen")
    require(
        selection.get("source_split") == "test",
        "Confirmation selection is not TEST",
    )
    require(
        int(selection.get("selected_case_count")) == CONFIRMATION_CASE_COUNT,
        "Unexpected confirmation case count",
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
        "Explanation outputs influenced confirmation selection",
    )
    require(
        selection.get("selection_uses_phase4d_or_phase4e_performance") is False,
        "Observed Phase-4D/4E performance influenced confirmation selection",
    )
    require(
        [int(v) for v in selection["prefixes"]]
        == [int(v) for v in PREFIXES],
        "Confirmation prefixes differ from frozen Phase-2 prefixes",
    )
    require(
        list(selection["models"]) == list(MODELS),
        "Confirmation models differ",
    )
    require(
        int(selection.get("global_sha256_rank_start")) == 101
        and int(selection.get("global_sha256_rank_end")) == 300,
        "Confirmation deterministic rank interval is not 101--300",
    )
    require(
        int(selection.get("overlap_with_original_phase4d")) == 0,
        "Confirmation selection reports overlap with Phase 4D",
    )
    require(
        float(selection.get("selective_stage3_pre_fidelity_threshold"))
        == SELECTIVE_STAGE3_THRESHOLD,
        "Confirmation selection threshold is not 0.50",
    )
    require(
        float(adaptive["gate"]["pre_mean_fidelity_threshold"])
        == SELECTIVE_STAGE3_THRESHOLD,
        "Adaptive config threshold changed after case freeze",
    )
    require(
        selection.get("original_selection_sha256")
        == file_sha256(ORIGINAL_SELECTION),
        "Original Phase-4D selection changed after Phase-4F freeze",
    )
    require(
        selection.get("frozen_ds_config_sha256") == file_sha256(FROZEN_DS),
        "Frozen D-S config changed after Phase-4F freeze",
    )
    require(
        selection.get("adaptive_config_sha256")
        == file_sha256(ADAPTIVE_CONFIG),
        "Selective Stage-3 config changed after Phase-4F freeze",
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    case_by_id = {c.case_id: c for c in cases}

    selected_ids = list(selection["selected_case_ids"])
    require(
        len(selected_ids) == CONFIRMATION_CASE_COUNT
        and len(set(selected_ids)) == CONFIRMATION_CASE_COUNT,
        "Frozen confirmation IDs are not 200 unique cases",
    )

    selected_cases = []
    for cid in selected_ids:
        require(cid in case_by_id, f"Selected case missing from cohort: {cid}")
        case = case_by_id[cid]
        require(case.split == "test", f"Selected case is not TEST: {cid}")
        selected_cases.append(case)

    # Independently reproduce both Phase 4D and Phase 4F selections.
    ranked_test = sorted(
        (c for c in cases if c.split == "test"),
        key=lambda c: (stable_rank(c.case_id), c.case_id),
    )
    original_ids = list(original_selection["selected_case_ids"])
    expected_original_ids = [
        c.case_id for c in ranked_test[:ORIGINAL_CASE_COUNT]
    ]
    require(
        original_ids == expected_original_ids,
        "Original Phase 4D ranks 1--100 are not reproducible",
    )
    expected_ids = [
        c.case_id
        for c in ranked_test[
            ORIGINAL_CASE_COUNT:
            ORIGINAL_CASE_COUNT + CONFIRMATION_CASE_COUNT
        ]
    ]
    require(
        selected_ids == expected_ids,
        "Frozen Phase 4F ranks 101--300 are not reproducible",
    )
    require(
        set(selected_ids).isdisjoint(original_ids),
        "Confirmation cohort overlaps original Phase 4D cohort",
    )

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    EVAL_CACHE.mkdir(parents=True, exist_ok=True)

    write_json(
        OUT / "runtime_confirmation_multirule.json",
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
        OUT / "confirmation_run_config.json",
        {
            "scope": "untouched confirmation multi-rule evidence bank",
            "selected_case_count": CONFIRMATION_CASE_COUNT,
            "confirmation_target_count": (
                CONFIRMATION_CASE_COUNT * len(MODELS) * len(PREFIXES)
            ),
            "models": list(MODELS),
            "prefixes": [int(v) for v in PREFIXES],
            "surrogates": list(SURROGATES),
            "pool_size": POOL_SIZE,
            "max_edits": MAX_EDITS,
            "max_options_per_position": MAX_OPTIONS,
            "pool_seed_base": POOL_SEED_BASE,
            "stratified_local_split": True,
            "same_phase4b_v2_neighborhood_protocol": True,
            "same_phase4b_v2_blackbox_protocol": True,
            "same_phase4b_v2_surrogate_definitions": True,
            "same_phase4b_v2_surrogate_hyperparameters": True,
            "same_phase4b_v2_process_semantic_constraints": True,
            "same_phase4b_v2_multirule_extraction": True,
            "heldout_eval_rule_metrics": True,
            "frozen_lambda": float(frozen["lambda"]),
            "frozen_selective_stage3_threshold": (
                SELECTIVE_STAGE3_THRESHOLD
            ),
            "lambda_selected_here": False,
            "selective_threshold_selected_here": False,
            "ds_fusion_performed_here": False,
            "selective_gate_applied_here": False,
            "test_cases_used": True,
            "test_labels_used_for_selection": False,
            "test_labels_used_for_rule_extraction": False,
            "test_blackbox_outputs_used_for_case_selection": False,
            "fusion_eval_cache_saved_without_protocol_change": True,
        },
    )

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)

    expected_targets = (
        CONFIRMATION_CASE_COUNT * len(MODELS) * len(PREFIXES)
    )
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

    try:
        for model in MODELS:
            for prefix in PREFIXES:
                set_status(
                    target=f"{model}/P{prefix}",
                    stage="load canonical black box",
                )
                predictor = CanonicalPredictor(
                    ROOT,
                    model,
                    prefix,
                    canonical["items"][model][str(prefix)],
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

                    factual = tuple(case.activities[:prefix])
                    timestamps = tuple(case.timestamps[:prefix])
                    seed = local_seed(
                        model,
                        prefix,
                        case.case_id,
                    )

                    set_status(stage="generate_process_valid_pool")
                    pool = generate_process_valid_pool(
                        factual,
                        constraints,
                        rng_seed=seed,
                        pool_size=POOL_SIZE,
                        max_edits=MAX_EDITS,
                        max_options_per_position=MAX_OPTIONS,
                    )

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
                                "pool_size": len(pool),
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
                        "pool_size": len(pool),
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
                        "pool_size": len(pool),
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
                        "confirmation_target": True,
                        "true_label_used": False,
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
    # Assemble confirmation bank from target caches.
    # -----------------------------------------------------------------
    metrics_all = []
    rules_all = []
    manifests = []

    eval_cache_target_count = 0
    eval_cache_total_pool_rows = 0
    eval_cache_total_eval_rows = 0

    for model in MODELS:
        for prefix in PREFIXES:
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
                        "pool_size": p["pool_size"],
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
        "Confirmation manifest target count mismatch",
    )
    require(
        len(metrics_all)
        == expected_targets * len(SURROGATES),
        "Confirmation surrogate metric row count mismatch",
    )
    require(
        eval_cache_target_count == expected_targets,
        "Confirmation fusion eval cache target count mismatch",
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
        "Duplicate rule key in assembled confirmation bank",
    )

    write_csv(
        OUT / "neighborhood_manifest_confirmation.csv",
        manifests,
    )
    write_csv(
        OUT / "surrogate_metrics_confirmation.csv",
        metrics_all,
    )

    with (
        OUT / "canonical_rules_confirmation.jsonl"
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
        OUT / "surrogate_aggregate_confirmation.csv",
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
        "scope": "untouched confirmation multi-rule bank",
        "selected_confirmation_case_count": CONFIRMATION_CASE_COUNT,
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
        "confirmation_test_cases_used": True,
        "true_labels_used_for_case_selection": False,
        "blackbox_outputs_used_for_case_selection": False,
    }
    write_json(
        OUT / "multi_rule_summary_confirmation.json",
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
        OUT / "rule_support_summary_confirmation.json",
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
            "same_targets_as_confirmation_rule_bank": True,
            "true_labels_used": False,
            "purpose": (
                "re-use exact confirmation local held-out evaluation "
                "subsets for frozen Stage1/Stage2/Stage3 D-S and the "
                "pre-frozen selective Stage-3 gate"
            ),
        },
    )

    def fmt(v):
        if v == "":
            return "-"
        return f"{float(v):.4f}"

    report = [
        "# ETCE Phase 4F-1 Untouched Confirmation Multi-rule Bank",
        "",
        "> Confirmation evidence bank built after lambda=5 and the selective Stage-3 threshold=0.50 were frozen.",
        "",
        "## Frozen protocol",
        "",
        f"- Frozen lambda: {float(frozen['lambda']):g}.",
        f"- Selected untouched TEST cases: {CONFIRMATION_CASE_COUNT}.",
        "- Deterministic SHA-256 ranks: 101--300; overlap with original Phase 4D: 0.",
        "- Prefixes: 5, 10, 15, 20.",
        "- Black-boxes: RF, XGBoost, MLP, LSTM.",
        "- Surrogates: LORE, DT, BN.",
        "- Local neighborhood/split/surrogate/multi-rule protocol: identical to validated Phase 4B-v2.",
        "- TEST true labels were not used for target selection or rule extraction.",
        "- Phase-4D/4E observed performance was not used for case selection.",
        "- D-S fusion is not performed in this script.",
        "",
        "## Bank diagnostics",
        "",
        f"- Confirmation local targets: {expected_targets}.",
        f"- Target-surrogate groups: {len(metrics_all)}.",
        f"- Total canonical rules: {len(rules_all)}.",
        f"- Total CF candidates: {len(cf_rules_all)}.",
        f"- Eligible CF candidates: {len(eligible_cf)}.",
        f"- Groups with >=2 CF candidates: {multi_summary['groups_with_multiple_counterfactual_candidates']}.",
        f"- Maximum CF candidates in one group: {multi_summary['max_counterfactual_candidates_in_one_group']}.",
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
        "## Confirmation fusion cache",
        "",
        f"- Cached targets: {eval_cache_target_count}.",
        f"- Total local pool rows: {eval_cache_total_pool_rows}.",
        f"- Total held-out eval rows: {eval_cache_total_eval_rows}.",
        "",
        "## Stopping point",
        "",
        "Run an independent confirmation-bank validator next. Do not change the frozen lambda, selective threshold, BPA, grouping, fusion order, conflict policy, or tie-breaking in response to these confirmation-bank diagnostics.",
        "",
    ]

    (
        OUT / "PHASE4F_CONFIRMATION_MULTIRULE_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("ETCE PHASE 4F-1 CONFIRMATION MULTI-RULE BANK COMPLETE")
    print("=" * 108)
    print(f"Selected untouched TEST cases: {CONFIRMATION_CASE_COUNT}")
    print("Global deterministic ranks: 101--300")
    print("Overlap with original Phase 4D: 0")
    print(f"Confirmation targets: {expected_targets}")
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
    print(f"Frozen selective Stage-3 threshold: {SELECTIVE_STAGE3_THRESHOLD:.2f}")
    print(f"Reused target caches this run: {reused}")
    print(f"Regenerated targets this run: {regenerated}")
    print(f"Output: {OUT}")
    print()
    print("NEXT: independent Phase 4F confirmation-bank validator.")
    print("Do NOT alter the frozen D-S or selective Stage-3 protocol.")


if __name__ == "__main__":
    main()
