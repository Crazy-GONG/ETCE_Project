#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETCE Phase 4F-2: independent validator for the untouched confirmation
multi-rule bank.

This validator independently reproduces, for all 3200 frozen confirmation
targets:
- the original Phase 4D ranks 1--100 and Phase 4F ranks 101--300;
- process-valid local neighborhood;
- canonical black-box probabilities / labels;
- deterministic class-stratified 70/30 split;
- LORE / DT / BN fits;
- factual rule and ALL counterfactual candidates;
- deterministic candidate IDs/order;
- held-out n_r / Fid(r) / coverage;
- witness sequence / edit count;
- saved confirmation fusion-eval cache.

It also recomputes:
- multi_rule_summary_confirmation.json
- rule_support_summary_confirmation.json

No protocol selection/tuning is performed here.
The frozen D-S configuration and selective Stage-3 threshold must remain
unchanged.

Run:
    python -u scripts/49_phase4f_validate_confirmation_multirule_bank.py
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
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

import numpy as np

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase3.core import CanonicalPredictor, load_process_constraints
from etce.phase4.surrogate_core import generate_process_valid_pool, sequence_hash
from etce.phase4.stratified_core import run_all_surrogates_stratified
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
PHASE4B_V2 = ROOT / "results/phase4/phase4b_v2_multirule_bank"
PHASE4D = ROOT / "results/phase4/phase4d_formal_test"
ORIGINAL_SELECTION = PHASE4D / "formal_test_case_selection.json"
PHASE4E = ROOT / "results/phase4/phase4e_selective_stage3"
ADAPTIVE_CONFIG = PHASE4E / "selective_stage3_frozen_config.json"
PHASE4F = ROOT / "results/phase4/phase4f_confirmation"
SELECTION = PHASE4F / "confirmation_case_selection.json"
BANK = PHASE4F / "confirmation_multirule_bank"

MANIFEST = BANK / "neighborhood_manifest_confirmation.csv"
METRICS = BANK / "surrogate_metrics_confirmation.csv"
RULES = BANK / "canonical_rules_confirmation.jsonl"
MULTI_SUMMARY = BANK / "multi_rule_summary_confirmation.json"
SUPPORT_SUMMARY = BANK / "rule_support_summary_confirmation.json"
RUN_CONFIG = BANK / "confirmation_run_config.json"
FUSION_CACHE_MANIFEST = BANK / "fusion_eval_cache_manifest.json"
FUSION_CACHE_DIR = BANK / "fusion_eval_cache/targets"

FROZEN_DS = ROOT / "configs/phase4c_ds_frozen_config.json"
VALIDATOR_CACHE = BANK / "validator_cache"
VALIDATION_OUT = BANK / "validation_phase4f_confirmation_multirule.json"
REPORT = BANK / "PHASE4F_CONFIRMATION_MULTIRULE_VALIDATION.md"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SURROGATES = ("LORE", "DT", "BN")
ORIGINAL_CASE_COUNT = 100
CONFIRMATION_CASE_COUNT = 200
SELECTIVE_STAGE3_THRESHOLD = 0.50
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0

PROB_TOL = 1e-9
FLOAT_TOL = 1e-10
EVAL_CACHE_SCHEMA = "phase4f-confirmation-fusion-eval-target-v1"
CACHE_SCHEMA = "phase4f-confirmation-independent-validator-v1"

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


def close(a, b, tol=FLOAT_TOL):
    if a in (None, "") or b in (None, ""):
        return a in (None, "") and b in (None, "")
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


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


def local_seed(model, prefix, case_id):
    text = f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def pool_digest(sequences):
    payload = "\n".join(sorted(sequence_hash(seq) for seq in sequences))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def target_token(case_id, model, prefix):
    return hashlib.sha256(
        f"{case_id}|{model}|{prefix}".encode("utf-8")
    ).hexdigest()[:20]


def eval_cache_path(case_id, model, prefix):
    return FUSION_CACHE_DIR / (
        f"{model}_P{prefix}_{target_token(case_id, model, prefix)}.json.gz"
    )


def validator_cache_path(case_id, model, prefix):
    return VALIDATOR_CACHE / (
        f"{model}_P{prefix}_{target_token(case_id, model, prefix)}.json"
    )


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL line {i}: {path}") from exc
    return out


def read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def json_normalize(obj):
    return json.loads(json.dumps(obj, ensure_ascii=False, sort_keys=True))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint():
    paths = [
        ROOT / "src/etce/phase4/surrogate_core.py",
        ROOT / "src/etce/phase4/stratified_core.py",
        ORIGINAL_SELECTION,
        ADAPTIVE_CONFIG,
        SELECTION,
        FROZEN_DS,
        MANIFEST,
        METRICS,
        RULES,
        MULTI_SUMMARY,
        SUPPORT_SUMMARY,
        RUN_CONFIG,
        FUSION_CACHE_MANIFEST,
    ]
    h = hashlib.sha256()
    for p in paths:
        require(p.is_file(), f"Fingerprint input missing: {p}")
        h.update(str(p.relative_to(ROOT)).encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def cache_passed(path, fingerprint, case_id, model, prefix):
    if not path.is_file():
        return False
    try:
        p = read_json(path)
    except Exception:
        return False
    return (
        p.get("schema_version") == CACHE_SCHEMA
        and p.get("source_fingerprint") == fingerprint
        and p.get("status") == "passed"
        and p.get("case_id") == case_id
        and p.get("model") == model
        and int(p.get("prefix", -1)) == int(prefix)
    )


def rule_core(rule):
    if rule is None:
        return None
    return {k: v for k, v in rule.items() if k != "candidate_id"}


def legacy_matches_first(best_rule, cf_rules):
    if best_rule is None and not cf_rules:
        return True
    if best_rule is None or not cf_rules:
        return False
    return json_normalize(rule_core(cf_rules[0])) == json_normalize(
        rule_core(best_rule)
    )


def expected_rule_record(
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
        "conditions": json_normalize(rule["conditions"]),
        "n_r": support,
        "Fid_r": fidelity,
        "coverage_eval": float(rule["coverage"]),
        "rule_length": int(rule["rule_length"]),
        "witness_sequence": json_normalize(rule.get("witness_sequence")),
        "witness_edit_count": rule.get("witness_edit_count"),
        "is_legacy_best": bool(is_legacy_best),
        "eligible_for_reliability": bool(
            support > 0 and fidelity is not None
        ),
    }


def compare_rule(saved, expected, key):
    require(saved is not None, f"Missing saved rule: {key}")

    exact_fields = (
        "case_id",
        "model",
        "prefix",
        "surrogate",
        "candidate_id",
        "rule_type",
        "predicted_class",
        "rule_length",
        "witness_edit_count",
        "is_legacy_best",
        "eligible_for_reliability",
    )
    for field in exact_fields:
        require(
            saved.get(field) == expected.get(field),
            f"Rule field mismatch {field}: {key}",
        )

    require(
        json_normalize(saved.get("conditions"))
        == json_normalize(expected.get("conditions")),
        f"Rule conditions mismatch: {key}",
    )
    require(
        json_normalize(saved.get("witness_sequence"))
        == json_normalize(expected.get("witness_sequence")),
        f"Rule witness_sequence mismatch: {key}",
    )
    require(
        int(saved["n_r"]) == int(expected["n_r"]),
        f"Rule n_r mismatch: {key}",
    )
    require(
        close(saved.get("Fid_r"), expected.get("Fid_r")),
        f"Rule Fid_r mismatch: {key}",
    )
    require(
        close(saved.get("coverage_eval"), expected.get("coverage_eval")),
        f"Rule coverage mismatch: {key}",
    )


def main():
    started = time.perf_counter()

    required = [
        ROOT / "data/raw/BPI Challenge 2017.xes.gz",
        ROOT / "results/phase1_2/audit/common_cohort_case_ids.csv",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        PHASE4B_V2 / "validation_phase4b_v2.json",
        FROZEN_DS,
        ORIGINAL_SELECTION,
        ADAPTIVE_CONFIG,
        SELECTION,
        MANIFEST,
        METRICS,
        RULES,
        MULTI_SUMMARY,
        SUPPORT_SUMMARY,
        RUN_CONFIG,
        FUSION_CACHE_MANIFEST,
    ]
    missing = [str(p) for p in required if not p.is_file()]
    require(not missing, "Missing inputs:\n" + "\n".join(missing))

    phase4b = read_json(PHASE4B_V2 / "validation_phase4b_v2.json")
    require(phase4b.get("status") == "passed", "Phase 4B-v2 not PASSED")

    frozen = read_json(FROZEN_DS)
    require(frozen.get("status") == "frozen", "D-S protocol not frozen")
    require(float(frozen.get("lambda")) == 5.0, "Frozen lambda != 5")
    require(
        frozen.get("selection_scope") == "validation_only",
        "Frozen lambda not validation-only",
    )
    require(
        frozen.get("formal_test_seen_before_freeze") is False,
        "Formal TEST was seen before D-S freeze",
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

    selection = read_json(SELECTION)
    require(
        selection.get("status") == "frozen",
        "Confirmation selection not frozen",
    )
    require(
        selection.get("schema_version")
        == "etce-phase4f-confirmation-selection-v1",
        "Unexpected confirmation selection schema",
    )
    require(selection.get("source_split") == "test", "Selection is not TEST")
    require(
        selection.get("selection_uses_true_label") is False
        and selection.get("selection_uses_blackbox_prediction") is False
        and selection.get("selection_uses_blackbox_probability") is False
        and selection.get("selection_uses_explanation_output") is False
        and selection.get("selection_uses_surrogate_output") is False
        and selection.get("selection_uses_phase4d_or_phase4e_performance")
        is False,
        "Forbidden information influenced confirmation target selection",
    )
    require(
        int(selection.get("selected_case_count")) == CONFIRMATION_CASE_COUNT,
        "Unexpected confirmation case count",
    )
    require(
        int(selection.get("confirmation_target_count"))
        == CONFIRMATION_CASE_COUNT * len(MODELS) * len(PREFIXES),
        "Unexpected frozen confirmation target count",
    )
    require(
        int(selection.get("global_sha256_rank_start")) == 101
        and int(selection.get("global_sha256_rank_end")) == 300,
        "Confirmation rank interval is not 101--300",
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
        selection.get("original_selection_sha256")
        == sha256_file(ORIGINAL_SELECTION),
        "Original Phase-4D selection changed after Phase-4F freeze",
    )
    require(
        selection.get("frozen_ds_config_sha256") == sha256_file(FROZEN_DS),
        "Frozen D-S config changed after Phase-4F freeze",
    )
    require(
        selection.get("adaptive_config_sha256")
        == sha256_file(ADAPTIVE_CONFIG),
        "Selective Stage-3 config changed after Phase-4F freeze",
    )

    run_config = read_json(RUN_CONFIG)
    require(
        int(run_config["selected_case_count"]) == CONFIRMATION_CASE_COUNT,
        "Confirmation bank selected_case_count mismatch",
    )
    require(
        int(run_config["confirmation_target_count"])
        == CONFIRMATION_CASE_COUNT * len(MODELS) * len(PREFIXES),
        "Confirmation bank target count mismatch",
    )
    require(
        run_config["same_phase4b_v2_neighborhood_protocol"] is True
        and run_config["same_phase4b_v2_blackbox_protocol"] is True
        and run_config["same_phase4b_v2_surrogate_definitions"] is True
        and run_config["same_phase4b_v2_surrogate_hyperparameters"] is True
        and run_config["same_phase4b_v2_process_semantic_constraints"] is True
        and run_config["same_phase4b_v2_multirule_extraction"] is True
        and run_config["heldout_eval_rule_metrics"] is True,
        "Confirmation bank protocol differs from Phase 4B-v2",
    )
    require(
        float(run_config["frozen_lambda"]) == 5.0
        and run_config["lambda_selected_here"] is False
        and run_config["ds_fusion_performed_here"] is False
        and float(run_config["frozen_selective_stage3_threshold"])
        == SELECTIVE_STAGE3_THRESHOLD
        and run_config["selective_threshold_selected_here"] is False
        and run_config["selective_gate_applied_here"] is False,
        "Confirmation bank altered a frozen protocol",
    )
    require(
        run_config["test_labels_used_for_selection"] is False
        and run_config["test_labels_used_for_rule_extraction"] is False
        and run_config[
            "test_blackbox_outputs_used_for_case_selection"
        ] is False,
        "Confirmation bank used forbidden TEST information",
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    case_by_id = {c.case_id: c for c in cases}

    test_cases = [c for c in cases if c.split == "test"]
    require(len(test_cases) == 5276, "Unexpected TEST cohort size")

    ranked_test = sorted(
        test_cases,
        key=lambda c: (stable_rank(c.case_id), c.case_id),
    )
    original_selection = read_json(ORIGINAL_SELECTION)
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
    selected_ids = list(selection["selected_case_ids"])
    require(
        selected_ids == expected_ids,
        "Phase 4F deterministic TEST case selection mismatch",
    )
    require(
        len(selected_ids) == CONFIRMATION_CASE_COUNT
        and len(set(selected_ids)) == CONFIRMATION_CASE_COUNT,
        "Frozen confirmation IDs are not 200 unique cases",
    )
    require(
        set(selected_ids).isdisjoint(original_ids),
        "Confirmation cohort overlaps original Phase 4D cohort",
    )
    require(
        all(
            cid in case_by_id and case_by_id[cid].split == "test"
            for cid in selected_ids
        ),
        "Confirmation selection contains a missing or non-TEST case",
    )
    require(
        tuple(int(v) for v in selection["prefixes"])
        == tuple(int(v) for v in PREFIXES)
        == (5, 10, 15, 20),
        "Frozen prefix protocol mismatch",
    )
    require(list(selection["models"]) == list(MODELS), "Model list mismatch")

    manifest_rows = read_csv(MANIFEST)
    metric_rows = read_csv(METRICS)
    saved_rules = read_jsonl(RULES)

    expected_targets = (
        CONFIRMATION_CASE_COUNT * len(MODELS) * len(PREFIXES)
    )
    require(
        len(manifest_rows) == expected_targets,
        "Confirmation neighborhood manifest row count mismatch",
    )
    require(
        len(metric_rows) == expected_targets * len(SURROGATES),
        "Confirmation surrogate metric row count mismatch",
    )

    manifest_index = {}
    for r in manifest_rows:
        key = (r["case_id"], r["model"], int(r["prefix"]))
        require(key not in manifest_index, f"Duplicate manifest key: {key}")
        manifest_index[key] = r

    metric_index = {}
    for r in metric_rows:
        key = (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
        )
        require(key not in metric_index, f"Duplicate metric key: {key}")
        metric_index[key] = r

    rule_index = {}
    for r in saved_rules:
        key = (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
            r["rule_type"],
            r["candidate_id"],
        )
        require(key not in rule_index, f"Duplicate saved rule key: {key}")
        rule_index[key] = r

    selected_set = set(selected_ids)
    require(
        all(
            r["case_id"] in selected_set
            and r["model"] in MODELS
            and int(r["prefix"]) in PREFIXES
            for r in saved_rules
        ),
        "Saved rule outside frozen confirmation target set",
    )

    fusion_manifest = read_json(FUSION_CACHE_MANIFEST)
    require(
        fusion_manifest.get("schema_version") == EVAL_CACHE_SCHEMA
        and fusion_manifest.get("status") == "built"
        and int(fusion_manifest["target_count"]) == expected_targets
        and fusion_manifest["same_targets_as_confirmation_rule_bank"] is True
        and fusion_manifest["true_labels_used"] is False,
        "Confirmation fusion eval cache manifest invalid",
    )

    fingerprint = source_fingerprint()
    VALIDATOR_CACHE.mkdir(parents=True, exist_ok=True)

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)

    total = expected_targets
    completed = 0
    reused_validator_cache = 0
    independently_recomputed = 0
    max_prob_diff = 0.0

    reproduced_rule_keys = set()
    recomputed_metric_rows = []

    set_status(completed=0, total=total)
    hb = threading.Thread(target=heartbeat, args=(started,), daemon=True)
    hb.start()

    try:
        for model in MODELS:
            for prefix in PREFIXES:
                predictor = CanonicalPredictor(
                    ROOT,
                    model,
                    prefix,
                    canonical["items"][model][str(prefix)],
                )

                for case_id in selected_ids:
                    completed += 1
                    key3 = (case_id, model, int(prefix))
                    label = (
                        f"[{completed:04d}/{total}] "
                        f"{model} P{prefix} {case_id}"
                    )
                    vcache = validator_cache_path(case_id, model, prefix)

                    set_status(
                        target=label,
                        stage="validator cache check",
                        completed=completed - 1,
                    )

                    if cache_passed(
                        vcache,
                        fingerprint,
                        case_id,
                        model,
                        prefix,
                    ):
                        payload = read_json(vcache)
                        for rk in payload["reproduced_rule_keys"]:
                            reproduced_rule_keys.add(tuple(rk))
                        recomputed_metric_rows.extend(payload["metric_rows"])
                        max_prob_diff = max(
                            max_prob_diff,
                            float(payload["factual_probability_abs_diff"]),
                        )
                        reused_validator_cache += 1
                        set_status(
                            stage="validator cached PASS",
                            completed=completed,
                        )
                        if completed % 25 == 0:
                            elapsed = time.perf_counter() - started
                            print(
                                f"{label} | validator-cached PASS | "
                                f"elapsed={fmt_seconds(elapsed)}",
                                flush=True,
                            )
                        continue

                    case = case_by_id[case_id]
                    factual = tuple(case.activities[:prefix])
                    timestamps = tuple(case.timestamps[:prefix])
                    seed = local_seed(model, prefix, case_id)

                    set_status(stage="regenerate process-valid pool")
                    pool = generate_process_valid_pool(
                        factual,
                        constraints,
                        rng_seed=seed,
                        pool_size=POOL_SIZE,
                        max_edits=MAX_EDITS,
                        max_options_per_position=MAX_OPTIONS,
                    )
                    digest = pool_digest(pool)

                    rec_manifest = manifest_index[key3]
                    require(
                        digest == rec_manifest["pool_digest"],
                        f"Neighborhood digest mismatch: {key3}",
                    )
                    require(
                        len(pool) == int(rec_manifest["pool_size"]),
                        f"Neighborhood size mismatch: {key3}",
                    )
                    require(
                        int(rec_manifest["pool_seed"]) == int(seed),
                        f"Pool seed mismatch: {key3}",
                    )

                    set_status(stage="recompute canonical black-box outputs")
                    probs = np.asarray(
                        predictor.predict_proba(
                            pool,
                            [timestamps] * len(pool),
                        ),
                        dtype=float,
                    )
                    labels = (probs >= 0.5).astype(int)
                    factual_idx = pool.index(factual)
                    factual_prob = float(probs[factual_idx])
                    factual_pred = int(labels[factual_idx])

                    diff = abs(
                        factual_prob
                        - float(rec_manifest["factual_probability"])
                    )
                    max_prob_diff = max(max_prob_diff, diff)
                    require(
                        diff <= PROB_TOL,
                        f"Factual probability mismatch: {key3}",
                    )
                    require(
                        factual_pred
                        == int(rec_manifest["factual_prediction"]),
                        f"Factual prediction mismatch: {key3}",
                    )
                    require(
                        int(np.sum(labels == 0))
                        == int(rec_manifest["pool_class0"])
                        and int(np.sum(labels == 1))
                        == int(rec_manifest["pool_class1"]),
                        f"Pool class inventory mismatch: {key3}",
                    )

                    set_status(stage="refit LORE / DT / BN independently")
                    result = run_all_surrogates_stratified(
                        pool,
                        labels,
                        factual,
                        factual_pred,
                    )

                    fit_labels = np.asarray(result["fit_labels"], dtype=int)
                    eval_labels = np.asarray(result["eval_labels"], dtype=int)

                    target_rule_keys = []
                    target_metric_rows = []
                    total_cf = 0

                    for branch in result["branches"]:
                        surrogate = branch["branch"]
                        mkey = (
                            case_id,
                            model,
                            int(prefix),
                            surrogate,
                        )
                        saved_metric = metric_index[mkey]

                        require(
                            "counterfactual_rules" in branch,
                            "Installed Phase 4 core is not multi-rule",
                        )

                        fact = branch["factual_rule"]
                        legacy_cf = branch["counterfactual_rule"]
                        cf_rules = list(branch["counterfactual_rules"])

                        declared = int(
                            branch.get(
                                "counterfactual_rule_count",
                                len(cf_rules),
                            )
                        )
                        require(
                            declared == len(cf_rules),
                            f"Declared CF count mismatch: {mkey}",
                        )

                        candidate_ids = [
                            str(r.get("candidate_id", ""))
                            for r in cf_rules
                        ]
                        expected_ids_cf = [
                            f"cf_{i:03d}" for i in range(len(cf_rules))
                        ]
                        require(
                            candidate_ids == expected_ids_cf
                            and len(candidate_ids) == len(set(candidate_ids)),
                            f"Candidate IDs/order mismatch: {mkey}",
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

                        exact_metric = {
                            "case_id": case_id,
                            "model": model,
                            "prefix": int(prefix),
                            "canonical_seed": int(predictor.seed),
                            "pool_seed": int(seed),
                            "surrogate": surrogate,
                            "status": branch["status"],
                            "pool_size": len(pool),
                            "pool_class0": int(np.sum(labels == 0)),
                            "pool_class1": int(np.sum(labels == 1)),
                            "fit_size": len(result["fit_sequences"]),
                            "eval_size": len(result["eval_sequences"]),
                            "fit_class0": int(np.sum(fit_labels == 0)),
                            "fit_class1": int(np.sum(fit_labels == 1)),
                            "eval_class0": int(np.sum(eval_labels == 0)),
                            "eval_class1": int(np.sum(eval_labels == 1)),
                            "surrogate_fidelity_eval": (
                                ""
                                if branch["surrogate_fidelity_eval"] is None
                                else float(branch["surrogate_fidelity_eval"])
                            ),
                            "factual_rule_available": bool(fact is not None),
                            "cf_rule_available": bool(len(cf_rules) > 0),
                            "cf_candidate_count": len(cf_rules),
                            "cf_eligible_candidate_count": int(eligible_count),
                            "cf_multi_rule_available": bool(len(cf_rules) >= 2),
                            "legacy_best_rule_available": bool(
                                legacy_cf is not None
                            ),
                            "legacy_best_matches_first_candidate": bool(
                                legacy_match
                            ),
                        }

                        integer_fields = (
                            "canonical_seed",
                            "pool_seed",
                            "pool_size",
                            "pool_class0",
                            "pool_class1",
                            "fit_size",
                            "eval_size",
                            "fit_class0",
                            "fit_class1",
                            "eval_class0",
                            "eval_class1",
                            "cf_candidate_count",
                            "cf_eligible_candidate_count",
                        )
                        for fld in integer_fields:
                            require(
                                int(saved_metric[fld])
                                == int(exact_metric[fld]),
                                f"Metric integer mismatch {fld}: {mkey}",
                            )

                        bool_fields = (
                            "factual_rule_available",
                            "cf_rule_available",
                            "cf_multi_rule_available",
                            "legacy_best_rule_available",
                            "legacy_best_matches_first_candidate",
                        )
                        for fld in bool_fields:
                            saved_bool = str(
                                saved_metric[fld]
                            ).strip().lower() in ("true", "1")
                            require(
                                saved_bool == bool(exact_metric[fld]),
                                f"Metric bool mismatch {fld}: {mkey}",
                            )

                        require(
                            saved_metric["status"] == exact_metric["status"],
                            f"Metric status mismatch: {mkey}",
                        )
                        require(
                            close(
                                saved_metric["surrogate_fidelity_eval"],
                                exact_metric["surrogate_fidelity_eval"],
                            ),
                            f"Surrogate fidelity mismatch: {mkey}",
                        )

                        target_metric_rows.append(exact_metric)

                        fact_expected = expected_rule_record(
                            case_id,
                            model,
                            prefix,
                            surrogate,
                            "fact_000",
                            "factual",
                            fact,
                            False,
                        )
                        if fact_expected is not None:
                            rkey = (
                                case_id,
                                model,
                                int(prefix),
                                surrogate,
                                "factual",
                                "fact_000",
                            )
                            compare_rule(
                                rule_index.get(rkey),
                                fact_expected,
                                rkey,
                            )
                            reproduced_rule_keys.add(rkey)
                            target_rule_keys.append(list(rkey))

                        for idx_cf, cf in enumerate(cf_rules):
                            cid = cf["candidate_id"]
                            expected = expected_rule_record(
                                case_id,
                                model,
                                prefix,
                                surrogate,
                                cid,
                                "counterfactual",
                                cf,
                                is_legacy_best=(
                                    idx_cf == 0 and legacy_match
                                ),
                            )
                            rkey = (
                                case_id,
                                model,
                                int(prefix),
                                surrogate,
                                "counterfactual",
                                cid,
                            )
                            compare_rule(
                                rule_index.get(rkey),
                                expected,
                                rkey,
                            )
                            reproduced_rule_keys.add(rkey)
                            target_rule_keys.append(list(rkey))

                        total_cf += len(cf_rules)

                    require(
                        total_cf
                        == int(rec_manifest["counterfactual_candidate_count"]),
                        f"Target total CF count mismatch: {key3}",
                    )

                    set_status(stage="validate fusion-eval cache")
                    epath = eval_cache_path(case_id, model, prefix)
                    require(epath.is_file(), f"Eval cache missing: {epath}")
                    ep = read_gzip_json(epath)

                    require(
                        ep.get("schema_version")
                        == EVAL_CACHE_SCHEMA,
                        f"Eval cache schema mismatch: {key3}",
                    )
                    require(
                        ep["case_id"] == case_id
                        and ep["model"] == model
                        and int(ep["prefix"]) == int(prefix),
                        f"Eval cache identity mismatch: {key3}",
                    )
                    require(
                        ep["pool_digest"] == digest,
                        f"Eval cache digest mismatch: {key3}",
                    )
                    require(
                        json_normalize(ep["pool_sequences"])
                        == json_normalize([list(seq) for seq in pool]),
                        f"Eval cache pool sequences mismatch: {key3}",
                    )
                    require(
                        [int(v) for v in ep["pool_labels"]]
                        == [int(v) for v in labels.tolist()],
                        f"Eval cache labels mismatch: {key3}",
                    )
                    require(
                        [int(v) for v in ep["fit_indices"]]
                        == [int(v) for v in result["fit_indices"]],
                        f"Eval cache fit indices mismatch: {key3}",
                    )
                    require(
                        [int(v) for v in ep["eval_indices"]]
                        == [int(v) for v in result["eval_indices"]],
                        f"Eval cache eval indices mismatch: {key3}",
                    )
                    require(
                        json_normalize(ep["factual_sequence"])
                        == json_normalize(list(factual)),
                        f"Eval cache factual sequence mismatch: {key3}",
                    )
                    require(
                        int(ep["factual_prediction"]) == factual_pred,
                        f"Eval cache factual prediction mismatch: {key3}",
                    )
                    require(
                        abs(float(ep["factual_probability"]) - factual_prob)
                        <= PROB_TOL,
                        f"Eval cache factual probability mismatch: {key3}",
                    )
                    require(
                        ep.get("true_label_used") is False,
                        f"Eval cache indicates true label usage: {key3}",
                    )

                    recomputed_metric_rows.extend(target_metric_rows)

                    write_json(
                        vcache,
                        {
                            "schema_version": CACHE_SCHEMA,
                            "source_fingerprint": fingerprint,
                            "status": "passed",
                            "case_id": case_id,
                            "model": model,
                            "prefix": int(prefix),
                            "pool_digest": digest,
                            "factual_probability_abs_diff": diff,
                            "reproduced_rule_keys": target_rule_keys,
                            "metric_rows": target_metric_rows,
                        },
                    )

                    independently_recomputed += 1
                    set_status(stage="target PASS", completed=completed)

                    if completed % 10 == 0 or completed == total:
                        elapsed = time.perf_counter() - started
                        avg = elapsed / completed
                        eta = avg * (total - completed)
                        print(
                            f"{label} | PASS | "
                            f"elapsed={fmt_seconds(elapsed)} | "
                            f"ETA={fmt_seconds(eta)}",
                            flush=True,
                        )

    finally:
        _STOP_HEARTBEAT.set()
        hb.join(timeout=1.0)

    require(
        len(reproduced_rule_keys) == len(saved_rules),
        (
            "Reproduced canonical rule count mismatch: "
            f"{len(reproduced_rule_keys)} != {len(saved_rules)}"
        ),
    )
    require(
        set(reproduced_rule_keys) == set(rule_index),
        "Saved/reproduced canonical rule key sets differ",
    )
    require(
        len(recomputed_metric_rows)
        == expected_targets * len(SURROGATES),
        "Recomputed surrogate metric count mismatch",
    )

    cf_rules_all = [
        r for r in saved_rules if r["rule_type"] == "counterfactual"
    ]
    eligible_cf = [
        r for r in cf_rules_all if bool(r["eligible_for_reliability"])
    ]

    counts = [int(r["cf_candidate_count"]) for r in recomputed_metric_rows]
    distribution = {
        str(k): int(sum(v == k for v in counts))
        for k in sorted(set(counts))
    }

    recomputed_multi = {
        "selected_confirmation_case_count": CONFIRMATION_CASE_COUNT,
        "target_count": expected_targets,
        "target_surrogate_group_count": len(recomputed_metric_rows),
        "total_canonical_rule_count": len(saved_rules),
        "total_counterfactual_candidate_count": len(cf_rules_all),
        "eligible_counterfactual_candidate_count": len(eligible_cf),
        "groups_with_counterfactual_candidate": int(
            sum(v > 0 for v in counts)
        ),
        "groups_with_multiple_counterfactual_candidates": int(
            sum(v >= 2 for v in counts)
        ),
        "max_counterfactual_candidates_in_one_group": (
            int(max(counts)) if counts else 0
        ),
        "counterfactual_candidate_count_distribution": distribution,
    }

    saved_multi = read_json(MULTI_SUMMARY)
    for fld, val in recomputed_multi.items():
        require(
            saved_multi.get(fld) == val,
            f"multi_rule_summary_confirmation mismatch: {fld}",
        )
    require(
        float(saved_multi["frozen_lambda"]) == 5.0
        and saved_multi["lambda_selected_here"] is False
        and saved_multi["ds_fusion_performed_here"] is False
        and saved_multi["confirmation_test_cases_used"] is True,
        "Confirmation multi-rule summary scope/frozen protocol mismatch",
    )

    supports = np.asarray(
        [int(r["n_r"]) for r in eligible_cf],
        dtype=float,
    )
    saved_support = read_json(SUPPORT_SUMMARY)
    require(
        int(saved_support["eligible_counterfactual_rule_count"])
        == len(eligible_cf),
        "Support summary eligible CF count mismatch",
    )
    if len(supports):
        checks = {
            "support_min": float(np.min(supports)),
            "support_q25": float(np.quantile(supports, 0.25)),
            "support_median": float(np.median(supports)),
            "support_q75": float(np.quantile(supports, 0.75)),
            "support_max": float(np.max(supports)),
        }
        for fld, val in checks.items():
            require(
                close(saved_support.get(fld), val),
                f"Support summary mismatch: {fld}",
            )
    require(
        float(saved_support["frozen_lambda"]) == 5.0
        and saved_support["lambda_selected_on_validation"] is True
        and saved_support["lambda_selected_here"] is False,
        "Support summary lambda state mismatch",
    )

    validation = {
        "scope": (
            "ETCE Phase 4F untouched confirmation multi-rule bank "
            "independent validation"
        ),
        "status": "passed",
        "source_fingerprint": fingerprint,
        "selected_confirmation_case_count": CONFIRMATION_CASE_COUNT,
        "confirmation_target_count": expected_targets,
        "target_surrogate_group_count": len(recomputed_metric_rows),
        "canonical_rule_count": len(saved_rules),
        "counterfactual_candidate_count": len(cf_rules_all),
        "eligible_counterfactual_candidate_count": len(eligible_cf),
        "groups_with_multiple_counterfactual_candidates": int(
            recomputed_multi[
                "groups_with_multiple_counterfactual_candidates"
            ]
        ),
        "max_counterfactual_candidates_in_one_group": int(
            recomputed_multi[
                "max_counterfactual_candidates_in_one_group"
            ]
        ),
        "frozen_lambda": 5.0,
        "frozen_selective_stage3_threshold": SELECTIVE_STAGE3_THRESHOLD,
        "max_factual_probability_abs_diff": float(max_prob_diff),
        "probability_tolerance": PROB_TOL,
        "targets_independently_recomputed_this_run": independently_recomputed,
        "targets_reused_from_validator_cache": reused_validator_cache,
        "checks": {
            "phase4d_ranks_1_100_reproduced": True,
            "phase4f_ranks_101_300_reproduced": True,
            "zero_overlap_with_phase4d": True,
            "all_3200_neighborhoods_reproduced": True,
            "all_pool_digests_reproduced": True,
            "all_blackbox_labels_and_factual_probabilities_reproduced": True,
            "all_stratified_local_splits_reproduced": True,
            "all_surrogate_fits_reproduced": True,
            "all_factual_and_multi_counterfactual_rules_reproduced": True,
            "candidate_ids_unique_ordered_and_deterministic": True,
            "heldout_support_fidelity_coverage_reproduced": True,
            "witnesses_reproduced": True,
            "fusion_eval_cache_reproduced": True,
            "multi_rule_summary_recomputed": True,
            "support_summary_recomputed": True,
            "lambda_remained_frozen_at_5": True,
            "selective_stage3_threshold_remained_frozen_at_0_50": True,
            "no_ds_protocol_retuning_performed": True,
            "no_selective_gate_retuning_performed": True,
        },
        "confirmation_test_cases_used": True,
        "confirmation_true_labels_used_for_selection_or_rules": False,
        "next_phase": (
            "confirmation target-local grouping audit followed by the frozen "
            "lambda=5 Stage1/Stage2/Stage3 D-S fusion and the pre-frozen "
            "selective Stage-3 rule"
        ),
    }

    write_json(VALIDATION_OUT, validation)

    report = f"""# ETCE Phase 4F Untouched Confirmation Multi-rule Bank Validation

Independent validation status: **PASSED**

- Frozen untouched TEST cases: {CONFIRMATION_CASE_COUNT}
- Deterministic ranks: 101--300
- Overlap with original Phase 4D: 0
- Confirmation targets: {expected_targets}
- Target-surrogate groups: {len(recomputed_metric_rows)}
- Canonical rules: {len(saved_rules)}
- Counterfactual candidates: {len(cf_rules_all)}
- Eligible counterfactual candidates: {len(eligible_cf)}
- Groups with >=2 CF candidates: {recomputed_multi['groups_with_multiple_counterfactual_candidates']}
- Maximum CF candidates/group: {recomputed_multi['max_counterfactual_candidates_in_one_group']}
- Frozen lambda: 5
- Frozen selective Stage-3 threshold: {SELECTIVE_STAGE3_THRESHOLD:.2f}
- Maximum factual probability absolute difference: {max_prob_diff:.3e}

The validator independently regenerated every process-valid neighborhood,
canonical black-box output, deterministic stratified local split, LORE/DT/BN
fit, factual rule, all multi-rule counterfactual candidates, candidate IDs,
held-out support/fidelity/coverage, witnesses, and the confirmation
fusion-evaluation cache. The original and confirmation selections were
independently reproduced from TEST case IDs only.

No lambda, BPA, grouping, fusion-order, conflict, representative-rule, or
tie-breaking parameter was re-selected using confirmation results. The
selective Stage-3 threshold remained frozen at 0.50.

**Next:** run the confirmation target-local grouping audit, then apply the
already frozen lambda=5 D-S protocol and selective Stage-3 rule exactly once.
"""
    REPORT.write_text(report, encoding="utf-8")

    print()
    print("=" * 108)
    print(
        "ETCE PHASE 4F-2 CONFIRMATION MULTI-RULE BANK VALIDATION PASSED"
    )
    print("=" * 108)
    print(f"Confirmation targets reproduced: {expected_targets}")
    print(f"Target-surrogate groups: {len(recomputed_metric_rows)}")
    print(f"Canonical rules reproduced: {len(saved_rules)}")
    print(f"CF candidates reproduced: {len(cf_rules_all)}")
    print(f"Eligible CF candidates: {len(eligible_cf)}")
    print(
        "Groups with >=2 CF candidates: "
        f"{recomputed_multi['groups_with_multiple_counterfactual_candidates']}"
    )
    print(
        "Max CF candidates/group: "
        f"{recomputed_multi['max_counterfactual_candidates_in_one_group']}"
    )
    print(
        "Max factual probability abs diff: "
        f"{max_prob_diff:.3e}"
    )
    print(f"Validator-cache reused: {reused_validator_cache}")
    print(f"Independently recomputed this run: {independently_recomputed}")
    print(f"Validation: {VALIDATION_OUT}")
    print()
    print("NEXT: confirmation target-local grouping audit.")
    print(
        "Frozen lambda remains 5 and the selective Stage-3 threshold remains "
        "0.50; no protocol retuning is permitted."
    )


if __name__ == "__main__":
    main()
