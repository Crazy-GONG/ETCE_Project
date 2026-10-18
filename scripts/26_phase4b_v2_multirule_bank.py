"""ETCE Phase 4B-v2: validation-only multi-rule calibration bank.

This phase EXTENDS the already validated Phase 4B rule-extraction interface:
- same 100 unused validation cases;
- same RF / XGBoost / MLP / LSTM canonical black boxes;
- same Prefix-5/10/15/20;
- same process-valid neighborhood generation;
- same deterministic class-stratified 70/30 local split;
- same LORE / DT / BN surrogate fits and hyperparameters;
- same held-out eval subset for n_r, Fid(r), and coverage.

Only the rule-bank interface changes: every genuine counterfactual candidate
returned by each target-local surrogate is retained. The legacy best
``counterfactual_rule`` remains available inside the surrogate core for
backward compatibility, but this writer stores ``counterfactual_rules``.

No test case is touched and lambda is NOT selected here.

Run:
    python -u scripts/26_phase4b_v2_multirule_bank.py

The script is resumable at target level via its own v2 cache directory.
"""
from __future__ import annotations

import csv
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
from etce.phase4.surrogate_core import generate_process_valid_pool, sequence_hash
from etce.phase4.stratified_core import run_all_surrogates_stratified
from etce.utils.io import read_json, write_csv, write_json, write_json_atomic

RESULTS3 = ROOT / "results/phase3"
PHASE4A = ROOT / "results/phase4/phase4a"
PHASE4A1 = ROOT / "results/phase4/phase4a1"
PHASE4B = ROOT / "results/phase4/phase4b_validation_bank"
OUT = ROOT / "results/phase4/phase4b_v2_multirule_bank"
CACHE = OUT / "target_cache"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SURROGATES = ("LORE", "DT", "BN")
CALIBRATION_CASE_COUNT = 100
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
CACHE_SCHEMA = "phase4b-v2-multirule-target-v1"

_STATUS = {"target": "startup", "stage": "initializing", "completed": 0, "total": 0}
_STATUS_LOCK = threading.Lock()
_STOP_HEARTBEAT = threading.Event()


def _fmt_seconds(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _set_status(target=None, stage=None, completed=None, total=None):
    with _STATUS_LOCK:
        if target is not None:
            _STATUS["target"] = target
        if stage is not None:
            _STATUS["stage"] = stage
        if completed is not None:
            _STATUS["completed"] = int(completed)
        if total is not None:
            _STATUS["total"] = int(total)
        _STATUS["updated"] = time.perf_counter()


def _heartbeat(started, interval=30):
    while not _STOP_HEARTBEAT.wait(interval):
        with _STATUS_LOCK:
            snap = dict(_STATUS)
        elapsed = time.perf_counter() - started
        total = snap.get("total", 0)
        completed = snap.get("completed", 0)
        pct = 0.0 if not total else 100.0 * completed / total
        print(
            f"[HEARTBEAT] {completed}/{total} ({pct:.2f}%) | "
            f"elapsed={_fmt_seconds(elapsed)} | "
            f"target={snap.get('target')} | stage={snap.get('stage')}",
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


def cache_name(case_id, model, prefix):
    token = hashlib.sha256(
        f"{case_id}|{model}|{prefix}".encode("utf-8")
    ).hexdigest()[:20]
    return CACHE / f"{model}_P{prefix}_{token}.json"


def cache_is_complete(path, case_id, model, prefix):
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        payload.get("schema_version") == CACHE_SCHEMA
        and payload.get("case_id") == case_id
        and payload.get("model") == model
        and int(payload.get("prefix", -1)) == int(prefix)
        and isinstance(payload.get("metrics"), list)
        and len(payload["metrics"]) == len(SURROGATES)
        and isinstance(payload.get("rules"), list)
        and isinstance(payload.get("pool_digest"), str)
    )


def require_inputs():
    required = [
        ROOT / "data/raw/BPI Challenge 2017.xes.gz",
        ROOT / "results/phase1_2/audit/common_cohort_case_ids.csv",
        ROOT / "results/phase2b/logs/validation_phase2b.json",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "phase3b/validation_phase3b.json",
        PHASE4A / "pilot_case_selection.json",
        PHASE4A1 / "validation_phase4a1.json",
        PHASE4B / "validation_phase4b.json",
        PHASE4B / "calibration_case_selection.json",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    for name, path in (
        ("phase2b", required[2]),
        ("phase3b", required[4]),
        ("phase4a1", required[6]),
        ("phase4b", required[7]),
    ):
        if read_json(path).get("status") != "passed":
            raise RuntimeError(f"Required validated phase is not passed: {name}")


def _rule_core(rule):
    if rule is None:
        return None
    return {
        key: value
        for key, value in rule.items()
        if key != "candidate_id"
    }


def _legacy_matches_first(best_rule, cf_rules):
    if best_rule is None and not cf_rules:
        return True
    if best_rule is None or not cf_rules:
        return False
    return _rule_core(cf_rules[0]) == _rule_core(best_rule)


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
    return {
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "candidate_id": str(candidate_id),
        "rule_type": rule_type,
        "predicted_class": int(rule["predicted_class"]),
        "conditions": rule["conditions"],
        "n_r": int(rule["support"]),
        "Fid_r": rule["fidelity"],
        "coverage_eval": float(rule["coverage"]),
        "rule_length": int(rule["rule_length"]),
        "witness_sequence": rule.get("witness_sequence"),
        "witness_edit_count": rule.get("witness_edit_count"),
        "is_legacy_best": bool(is_legacy_best),
        "eligible_for_reliability": bool(
            int(rule["support"]) > 0 and rule["fidelity"] is not None
        ),
    }


def main():
    started = time.perf_counter()
    require_inputs()

    old_selection = read_json(PHASE4B / "calibration_case_selection.json")
    pilot_ids = set(read_json(PHASE4A / "pilot_case_selection.json")["case_ids"])
    cases, _ = load_frozen_common_cohort(ROOT)
    remaining_validation = sorted(
        (
            case
            for case in cases
            if case.split == "val" and case.case_id not in pilot_ids
        ),
        key=lambda case: (stable_rank(case.case_id), case.case_id),
    )
    calibration_cases = remaining_validation[:CALIBRATION_CASE_COUNT]
    calibration_ids = [case.case_id for case in calibration_cases]
    if calibration_ids != old_selection["case_ids"]:
        raise AssertionError("Phase 4B-v2 calibration cases differ from validated Phase 4B")

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    write_json(
        OUT / "runtime_phase4b_v2.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device_protocol": "same CanonicalPredictor protocol as validated Phase 4B",
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
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                )
            },
            "project_root_from_environment": "ETCE_ROOT" in os.environ,
        },
    )

    write_json(
        OUT / "calibration_case_selection.json",
        {
            "split": "validation",
            "case_count": CALIBRATION_CASE_COUNT,
            "case_ids": calibration_ids,
            "identical_to_validated_phase4b_selection": True,
            "phase4a_pilot_case_ids_excluded": sorted(pilot_ids),
            "true_labels_used_for_selection": False,
            "blackbox_outputs_used_for_selection": False,
            "test_cases_used": False,
        },
    )

    write_json(
        OUT / "phase4b_v2_run_config.json",
        {
            "scope": "validation-only multi-rule calibration bank",
            "source_protocol": "validated Phase 4B + multi-rule extraction extension",
            "models": list(MODELS),
            "prefixes": list(PREFIXES),
            "surrogates": list(SURROGATES),
            "calibration_case_count": CALIBRATION_CASE_COUNT,
            "pool_size": POOL_SIZE,
            "max_edits": MAX_EDITS,
            "max_options_per_position": MAX_OPTIONS,
            "pool_seed_base": POOL_SEED_BASE,
            "stratified_local_split": True,
            "same_local_neighborhood_protocol": True,
            "same_blackbox_protocol": True,
            "same_surrogate_fit_definitions": True,
            "same_surrogate_hyperparameters": True,
            "same_process_semantic_constraints": True,
            "heldout_eval_rule_metrics": True,
            "multi_rule_extraction_extension": True,
            "legacy_best_rule_preserved": True,
            "candidate_id_added": True,
            "lambda_selected_here": False,
            "bpa_or_ds_fusion_performed_here": False,
            "test_cases_used": False,
            "paper_test_statistics": False,
        },
    )

    canonical = read_json(RESULTS3 / "canonical_blackboxes/canonical_manifest.json")
    constraints = load_process_constraints(RESULTS3)
    expected_targets = CALIBRATION_CASE_COUNT * len(MODELS) * len(PREFIXES)

    _set_status(completed=0, total=expected_targets)
    hb = threading.Thread(target=_heartbeat, args=(started,), daemon=True)
    hb.start()

    done = 0
    try:
        for model in MODELS:
            for prefix in PREFIXES:
                _set_status(target=f"{model}/P{prefix}", stage="load canonical black box")
                predictor = CanonicalPredictor(
                    ROOT, model, prefix, canonical["items"][model][str(prefix)]
                )

                for case in calibration_cases:
                    done += 1
                    target_label = f"[{done:04d}/{expected_targets}] {model} P{prefix} {case.case_id}"
                    _set_status(target=target_label, stage="cache check", completed=done - 1)
                    target_cache = cache_name(case.case_id, model, prefix)
                    if target_cache.is_file() and cache_is_complete(
                        target_cache, case.case_id, model, prefix
                    ):
                        _set_status(stage="cached", completed=done)
                        if done % 25 == 0:
                            elapsed = time.perf_counter() - started
                            print(
                                f"{target_label} | cached | elapsed={_fmt_seconds(elapsed)}",
                                flush=True,
                            )
                        continue

                    factual = tuple(case.activities[:prefix])
                    timestamps = tuple(case.timestamps[:prefix])
                    seed = local_seed(model, prefix, case.case_id)

                    target_started = time.perf_counter()
                    _set_status(stage="generate_process_valid_pool")
                    pool = generate_process_valid_pool(
                        factual,
                        constraints,
                        rng_seed=seed,
                        pool_size=POOL_SIZE,
                        max_edits=MAX_EDITS,
                        max_options_per_position=MAX_OPTIONS,
                    )

                    _set_status(stage="black-box predict_proba")
                    probs = predictor.predict_proba(pool, [timestamps] * len(pool))
                    labels = (np.asarray(probs) >= 0.5).astype(int)
                    factual_idx = pool.index(factual)
                    factual_label = int(labels[factual_idx])

                    _set_status(stage="LORE / DT / BN multi-rule extraction")
                    result = run_all_surrogates_stratified(
                        pool, labels, factual, factual_label
                    )
                    fit_labels = np.asarray(result["fit_labels"], dtype=int)
                    eval_labels = np.asarray(result["eval_labels"], dtype=int)

                    metrics = []
                    rules = []
                    target_candidate_total = 0
                    for branch in result["branches"]:
                        surrogate = branch["branch"]
                        if "counterfactual_rules" not in branch:
                            raise RuntimeError(
                                "Phase 4B-v2 core is not installed: branch lacks "
                                "counterfactual_rules. Replace surrogate_core.py and "
                                "stratified_core.py with the v2 versions first."
                            )
                        fact = branch["factual_rule"]
                        legacy_cf = branch["counterfactual_rule"]
                        cf_rules = list(branch["counterfactual_rules"])
                        declared = int(branch.get("counterfactual_rule_count", len(cf_rules)))
                        if declared != len(cf_rules):
                            raise AssertionError("counterfactual_rule_count mismatch")

                        ids = [str(rule.get("candidate_id", "")) for rule in cf_rules]
                        expected_ids = [f"cf_{i:03d}" for i in range(len(cf_rules))]
                        if ids != expected_ids or len(ids) != len(set(ids)):
                            raise AssertionError(
                                f"Unstable/duplicate candidate_id: {case.case_id}/{model}/P{prefix}/{surrogate}"
                            )

                        eligible_count = sum(
                            int(rule["support"]) > 0 and rule["fidelity"] is not None
                            for rule in cf_rules
                        )
                        legacy_match = _legacy_matches_first(legacy_cf, cf_rules)
                        metrics.append(
                            {
                                "case_id": case.case_id,
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
                                    "" if branch["surrogate_fidelity_eval"] is None
                                    else float(branch["surrogate_fidelity_eval"])
                                ),
                                "factual_rule_available": bool(fact is not None),
                                "cf_rule_available": bool(len(cf_rules) > 0),
                                "cf_candidate_count": len(cf_rules),
                                "cf_eligible_candidate_count": int(eligible_count),
                                "cf_multi_rule_available": bool(len(cf_rules) >= 2),
                                "legacy_best_rule_available": bool(legacy_cf is not None),
                                "legacy_best_matches_first_candidate": bool(legacy_match),
                            }
                        )

                        fact_rec = rule_record(
                            case.case_id, model, prefix, surrogate,
                            "fact_000", "factual", fact, False,
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
                                is_legacy_best=(idx == 0 and legacy_match),
                            )
                            rules.append(rec)
                        target_candidate_total += len(cf_rules)

                    keys = [
                        (
                            r["case_id"], r["model"], r["prefix"], r["surrogate"],
                            r["rule_type"], r["candidate_id"],
                        )
                        for r in rules
                    ]
                    if len(keys) != len(set(keys)):
                        raise AssertionError("Duplicate canonical rule key inside target")

                    payload = {
                        "schema_version": CACHE_SCHEMA,
                        "case_id": case.case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "pool_seed": int(seed),
                        "pool_digest": pool_digest(pool),
                        "pool_size": len(pool),
                        "pool_class0": int(np.sum(labels == 0)),
                        "pool_class1": int(np.sum(labels == 1)),
                        "factual_prediction": factual_label,
                        "factual_probability": float(probs[factual_idx]),
                        "counterfactual_candidate_count": int(target_candidate_total),
                        "metrics": metrics,
                        "rules": rules,
                    }
                    write_json_atomic(target_cache, payload)

                    _set_status(stage="target complete", completed=done)
                    if done % 10 == 0 or done == expected_targets:
                        target_elapsed = time.perf_counter() - target_started
                        elapsed = time.perf_counter() - started
                        avg = elapsed / done
                        eta = avg * (expected_targets - done)
                        print(
                            f"{target_label} | pool={len(pool)} | "
                            f"CF candidates={target_candidate_total} | "
                            f"target={target_elapsed:.1f}s | "
                            f"elapsed={_fmt_seconds(elapsed)} | ETA={_fmt_seconds(eta)}",
                            flush=True,
                        )
    finally:
        _STOP_HEARTBEAT.set()
        hb.join(timeout=1.0)

    metrics_all = []
    rules_all = []
    manifests = []
    for model in MODELS:
        for prefix in PREFIXES:
            for case in calibration_cases:
                payload = read_json(cache_name(case.case_id, model, prefix))
                if payload.get("schema_version") != CACHE_SCHEMA:
                    raise AssertionError("Unexpected v2 target-cache schema")
                metrics_all.extend(payload["metrics"])
                rules_all.extend(payload["rules"])
                manifests.append(
                    {
                        "case_id": payload["case_id"],
                        "model": payload["model"],
                        "prefix": payload["prefix"],
                        "pool_seed": payload["pool_seed"],
                        "pool_digest": payload["pool_digest"],
                        "pool_size": payload["pool_size"],
                        "pool_class0": payload["pool_class0"],
                        "pool_class1": payload["pool_class1"],
                        "factual_prediction": payload["factual_prediction"],
                        "factual_probability": payload["factual_probability"],
                        "counterfactual_candidate_count": payload["counterfactual_candidate_count"],
                    }
                )

    all_keys = [
        (
            r["case_id"], r["model"], int(r["prefix"]), r["surrogate"],
            r["rule_type"], r["candidate_id"],
        )
        for r in rules_all
    ]
    if len(all_keys) != len(set(all_keys)):
        raise AssertionError("Duplicate rule key in assembled Phase 4B-v2 bank")

    write_csv(OUT / "neighborhood_manifest.csv", manifests)
    write_csv(OUT / "surrogate_metrics_calibration.csv", metrics_all)
    with (OUT / "canonical_rules_calibration.jsonl").open("w", encoding="utf-8") as stream:
        for rule in rules_all:
            stream.write(json.dumps(rule, ensure_ascii=False, sort_keys=True) + "\n")

    cf_rules_all = [r for r in rules_all if r["rule_type"] == "counterfactual"]
    eligible_cf = [r for r in cf_rules_all if bool(r["eligible_for_reliability"])]

    aggregate = []
    for surrogate in SURROGATES:
        group = [r for r in metrics_all if r["surrogate"] == surrogate]
        cf_rows = [
            r for r in cf_rules_all if r["surrogate"] == surrogate
        ]
        sfid = [
            float(r["surrogate_fidelity_eval"])
            for r in group if r["surrogate_fidelity_eval"] != ""
        ]
        supports = [float(r["n_r"]) for r in cf_rows]
        fidelities = [
            float(r["Fid_r"]) for r in cf_rows if r["Fid_r"] is not None
        ]
        counts = [int(r["cf_candidate_count"]) for r in group]
        aggregate.append(
            {
                "surrogate": surrogate,
                "target_surrogate_groups": len(group),
                "mean_surrogate_fidelity_eval": "" if not sfid else float(np.mean(sfid)),
                "groups_with_cf": int(sum(v > 0 for v in counts)),
                "groups_with_multiple_cf": int(sum(v >= 2 for v in counts)),
                "total_cf_candidates": int(sum(counts)),
                "mean_cf_candidates_per_group": float(np.mean(counts)) if counts else 0.0,
                "max_cf_candidates_in_group": int(max(counts)) if counts else 0,
                "eligible_cf_candidates": int(
                    sum(bool(r["eligible_for_reliability"]) for r in cf_rows)
                ),
                "mean_cf_support_eval": "" if not supports else float(np.mean(supports)),
                "median_cf_support_eval": "" if not supports else float(np.median(supports)),
                "mean_cf_fidelity_eval": "" if not fidelities else float(np.mean(fidelities)),
            }
        )
    write_csv(OUT / "surrogate_aggregate_calibration.csv", aggregate)

    counts = [int(r["cf_candidate_count"]) for r in metrics_all]
    distribution = {
        str(k): int(sum(v == k for v in counts))
        for k in sorted(set(counts))
    }
    multi_rule_summary = {
        "target_count": expected_targets,
        "target_surrogate_group_count": len(metrics_all),
        "total_canonical_rule_count": len(rules_all),
        "total_counterfactual_candidate_count": len(cf_rules_all),
        "eligible_counterfactual_candidate_count": len(eligible_cf),
        "groups_with_counterfactual_candidate": int(sum(v > 0 for v in counts)),
        "groups_with_multiple_counterfactual_candidates": int(sum(v >= 2 for v in counts)),
        "max_counterfactual_candidates_in_one_group": int(max(counts)) if counts else 0,
        "counterfactual_candidate_count_distribution": distribution,
        "stage1_non_degeneracy_not_asserted_here": True,
        "next_required_check": "target-local Stage-1 grouping audit",
        "test_cases_used": False,
    }
    write_json(OUT / "multi_rule_summary.json", multi_rule_summary)

    support_values = np.asarray([int(r["n_r"]) for r in eligible_cf], dtype=float)
    support_summary = {
        "eligible_counterfactual_rule_count": len(eligible_cf),
        "support_min": None if not len(support_values) else float(np.min(support_values)),
        "support_q25": None if not len(support_values) else float(np.quantile(support_values, 0.25)),
        "support_median": None if not len(support_values) else float(np.median(support_values)),
        "support_q75": None if not len(support_values) else float(np.quantile(support_values, 0.75)),
        "support_max": None if not len(support_values) else float(np.max(support_values)),
        "lambda_candidate_grid_for_next_phase": [1, 5, 10, 20, 50],
        "lambda_selected": None,
    }
    write_json(OUT / "rule_support_summary.json", support_summary)

    def fmt(value):
        if value == "":
            return "-"
        return f"{float(value):.4f}"

    report = [
        "# ETCE Phase 4B-v2 Multi-rule Validation Bank",
        "",
        "> Validation-only extension of the already validated Phase 4B. No test case is used.",
        "",
        "## Frozen protocol",
        "",
        f"- Cases: the identical {CALIBRATION_CASE_COUNT} validation cases used in validated Phase 4B.",
        "- Black-boxes: RF, XGBoost, MLP, LSTM.",
        "- Prefixes: 5, 10, 15, 20.",
        "- Neighborhood/process constraints: unchanged.",
        "- Local split: deterministic class-stratified 70/30, factual retained in fit.",
        "- Surrogate fits/hyperparameters: unchanged.",
        "- Rule support/fidelity/coverage: held-out local eval subset only.",
        "- Extension: retain all deterministic, de-duplicated genuine CF candidates per target-local surrogate.",
        "- The legacy best-rule interface remains preserved for backward compatibility.",
        "",
        "## Multi-rule diagnostics",
        "",
        f"- Local targets: {expected_targets}",
        f"- Target-surrogate groups: {len(metrics_all)}",
        f"- Total CF candidates: {len(cf_rules_all)}",
        f"- Eligible CF candidates: {len(eligible_cf)}",
        f"- Groups with >=2 CF candidates: {multi_rule_summary['groups_with_multiple_counterfactual_candidates']}",
        f"- Maximum CF candidates in one group: {multi_rule_summary['max_counterfactual_candidates_in_one_group']}",
        "",
        "| Surrogate | Groups | Surrogate fidelity | Groups >=2 CF | Total CF | Mean CF/group | Max CF | Eligible CF | Mean support | Median support | Mean fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        report.append(
            f"| {row['surrogate']} | {row['target_surrogate_groups']} | "
            f"{fmt(row['mean_surrogate_fidelity_eval'])} | "
            f"{row['groups_with_multiple_cf']} | {row['total_cf_candidates']} | "
            f"{row['mean_cf_candidates_per_group']:.4f} | {row['max_cf_candidates_in_group']} | "
            f"{row['eligible_cf_candidates']} | {fmt(row['mean_cf_support_eval'])} | "
            f"{fmt(row['median_cf_support_eval'])} | {fmt(row['mean_cf_fidelity_eval'])} |"
        )
    report += [
        "",
        "## Stopping point",
        "",
        "Do not freeze lambda yet. First run the independent Phase 4B-v2 validator, then rerun the target-local Stage-1 grouping audit. Only after Stage 1 is shown to be non-degenerate should lambda sensitivity be recomputed on this v2 bank.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 4B-v2 validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]
    (OUT / "PHASE4B_V2_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    print("\nPhase 4B-v2 multi-rule bank complete.")
    print(f"Output: {OUT}")
    print(f"Total CF candidates: {len(cf_rules_all)}")
    print(
        "Groups with >=2 CF candidates: "
        f"{multi_rule_summary['groups_with_multiple_counterfactual_candidates']}"
    )
    print("Run scripts/27_validate_phase4b_v2.py next.")


if __name__ == "__main__":
    main()
