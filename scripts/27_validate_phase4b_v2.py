"""Independent validator for ETCE Phase 4B-v2 multi-rule validation bank.

The validator independently reconstructs every one of the 1600 target-local
neighborhoods, black-box labels, deterministic stratified split, LORE/DT/BN
fits, factual rules, and ALL counterfactual candidates. It verifies held-out
n_r/Fid(r)/coverage and candidate IDs against the saved v2 bank.

No test case is touched.

Run:
    python -u scripts/27_validate_phase4b_v2.py

A validator-only cache allows safe resume after interruption. Cache validity is
bound to hashes of the v2 core source files and the saved v2 rule bank.
"""
from __future__ import annotations

import csv
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
from etce.utils.io import read_json, write_json, write_json_atomic

RESULTS3 = ROOT / "results/phase3"
PHASE4A = ROOT / "results/phase4/phase4a"
PHASE4A1 = ROOT / "results/phase4/phase4a1"
PHASE4B = ROOT / "results/phase4/phase4b_validation_bank"
OUT = ROOT / "results/phase4/phase4b_v2_multirule_bank"
VCACHE = OUT / "validator_cache"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SURROGATES = ("LORE", "DT", "BN")
CALIBRATION_CASE_COUNT = 100
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
PROB_TOL = 1e-5
VALIDATION_CACHE_SCHEMA = "phase4b-v2-independent-validator-v1"

_STATUS = {"target": "startup", "stage": "initializing", "completed": 0, "total": 0}
_STATUS_LOCK = threading.Lock()
_STOP_HEARTBEAT = threading.Event()


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, tol=1e-8):
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def prob_close(a, b):
    return math.isclose(float(a), float(b), rel_tol=PROB_TOL, abs_tol=PROB_TOL)


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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_cache_name(case_id, model, prefix):
    token = hashlib.sha256(
        f"{case_id}|{model}|{prefix}".encode("utf-8")
    ).hexdigest()[:20]
    return VCACHE / f"{model}_P{prefix}_{token}.json"


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


def _rule_core(rule):
    if rule is None:
        return None
    return {key: value for key, value in rule.items() if key != "candidate_id"}


def _legacy_matches_first(best_rule, cf_rules):
    if best_rule is None and not cf_rules:
        return True
    if best_rule is None or not cf_rules:
        return False
    return _rule_core(cf_rules[0]) == _rule_core(best_rule)


def compare_rule(saved, actual):
    require(saved["candidate_id"] == actual["candidate_id"], "candidate_id mismatch")
    require(saved["rule_type"] == actual["rule_type"], "rule_type mismatch")
    require(int(saved["predicted_class"]) == int(actual["predicted_class"]), "rule class mismatch")
    require(saved["conditions"] == actual["conditions"], "rule conditions mismatch")
    require(int(saved["n_r"]) == int(actual["n_r"]), "rule support mismatch")
    if actual["Fid_r"] is None:
        require(saved["Fid_r"] is None, "rule fidelity null mismatch")
    else:
        require(close(saved["Fid_r"], actual["Fid_r"]), "rule fidelity mismatch")
    require(close(saved["coverage_eval"], actual["coverage_eval"]), "rule coverage mismatch")
    require(int(saved["rule_length"]) == int(actual["rule_length"]), "rule length mismatch")
    require(saved.get("witness_sequence") == actual.get("witness_sequence"), "witness sequence mismatch")
    require(saved.get("witness_edit_count") == actual.get("witness_edit_count"), "witness edit mismatch")
    require(bool(saved["is_legacy_best"]) == bool(actual["is_legacy_best"]), "legacy-best flag mismatch")
    require(
        bool(saved["eligible_for_reliability"]) == bool(actual["eligible_for_reliability"]),
        "eligibility mismatch",
    )


def main():
    started = time.perf_counter()
    required = [
        PHASE4B / "validation_phase4b.json",
        PHASE4A1 / "validation_phase4a1.json",
        OUT / "PHASE4B_V2_REPORT.md",
        OUT / "calibration_case_selection.json",
        OUT / "phase4b_v2_run_config.json",
        OUT / "neighborhood_manifest.csv",
        OUT / "surrogate_metrics_calibration.csv",
        OUT / "canonical_rules_calibration.jsonl",
        OUT / "surrogate_aggregate_calibration.csv",
        OUT / "multi_rule_summary.json",
        OUT / "rule_support_summary.json",
    ]
    require(all(path.is_file() for path in required), "Required Phase 4B-v2 artifact missing")
    require(read_json(PHASE4B / "validation_phase4b.json").get("status") == "passed", "Source Phase 4B is not passed")
    require(read_json(PHASE4A1 / "validation_phase4a1.json").get("status") == "passed", "Phase 4A.1 is not passed")

    config = read_json(OUT / "phase4b_v2_run_config.json")
    require(
        config.get("multi_rule_extraction_extension") is True
        and config.get("heldout_eval_rule_metrics") is True
        and config.get("same_local_neighborhood_protocol") is True
        and config.get("same_blackbox_protocol") is True
        and config.get("same_surrogate_fit_definitions") is True
        and config.get("same_surrogate_hyperparameters") is True
        and config.get("lambda_selected_here") is False
        and config.get("bpa_or_ds_fusion_performed_here") is False
        and config.get("test_cases_used") is False,
        "Phase 4B-v2 protocol/config mismatch",
    )

    pilot_ids = set(read_json(PHASE4A / "pilot_case_selection.json")["case_ids"])
    cases, _ = load_frozen_common_cohort(ROOT)
    expected_cases = sorted(
        (
            case for case in cases
            if case.split == "val" and case.case_id not in pilot_ids
        ),
        key=lambda case: (stable_rank(case.case_id), case.case_id),
    )[:CALIBRATION_CASE_COUNT]
    expected_ids = [case.case_id for case in expected_cases]
    case_by_id = {case.case_id: case for case in expected_cases}

    selection = read_json(OUT / "calibration_case_selection.json")
    old_selection = read_json(PHASE4B / "calibration_case_selection.json")
    require(selection["case_ids"] == expected_ids, "V2 calibration selection mismatch")
    require(selection["case_ids"] == old_selection["case_ids"], "V2 selection differs from validated Phase 4B")
    require(selection["test_cases_used"] is False, "Test case used in selection")

    manifest = rows(OUT / "neighborhood_manifest.csv")
    metrics = rows(OUT / "surrogate_metrics_calibration.csv")
    expected_targets = CALIBRATION_CASE_COUNT * len(MODELS) * len(PREFIXES)
    require(len(manifest) == expected_targets, "Neighborhood manifest target count mismatch")
    require(len(metrics) == expected_targets * len(SURROGATES), "Surrogate metric count mismatch")

    manifest_index = {
        (r["case_id"], r["model"], int(r["prefix"])): r
        for r in manifest
    }
    metric_index = {
        (r["case_id"], r["model"], int(r["prefix"]), r["surrogate"]): r
        for r in metrics
    }
    require(len(manifest_index) == len(manifest), "Duplicate manifest key")
    require(len(metric_index) == len(metrics), "Duplicate metric key")

    saved_rules = []
    with (OUT / "canonical_rules_calibration.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                saved_rules.append(json.loads(line))

    rules_by_target = {}
    rule_keys = set()
    for rule in saved_rules:
        key = (
            rule["case_id"], rule["model"], int(rule["prefix"]), rule["surrogate"],
            rule["rule_type"], rule["candidate_id"],
        )
        require(key not in rule_keys, "Duplicate saved canonical rule key")
        rule_keys.add(key)
        tkey = (rule["case_id"], rule["model"], int(rule["prefix"]))
        rules_by_target.setdefault(tkey, []).append(rule)

    source_fingerprint = {
        "surrogate_core_sha256": file_sha256(ROOT / "src/etce/phase4/surrogate_core.py"),
        "stratified_core_sha256": file_sha256(ROOT / "src/etce/phase4/stratified_core.py"),
        "rule_bank_sha256": file_sha256(OUT / "canonical_rules_calibration.jsonl"),
        "manifest_sha256": file_sha256(OUT / "neighborhood_manifest.csv"),
    }
    VCACHE.mkdir(parents=True, exist_ok=True)

    canonical = read_json(RESULTS3 / "canonical_blackboxes/canonical_manifest.json")
    constraints = load_process_constraints(RESULTS3)

    _set_status(completed=0, total=expected_targets)
    hb = threading.Thread(target=_heartbeat, args=(started,), daemon=True)
    hb.start()

    max_prob_diff = 0.0
    reproduced_rule_count = 0
    multi_groups = 0
    max_cf_per_group = 0
    done = 0

    try:
        for model in MODELS:
            for prefix in PREFIXES:
                _set_status(target=f"{model}/P{prefix}", stage="load canonical black box")
                predictor = CanonicalPredictor(
                    ROOT, model, prefix, canonical["items"][model][str(prefix)]
                )

                for case_id in expected_ids:
                    done += 1
                    tkey = (case_id, model, int(prefix))
                    target_label = f"[{done:04d}/{expected_targets}] {model} P{prefix} {case_id}"
                    cache_path = validation_cache_name(case_id, model, prefix)
                    _set_status(target=target_label, stage="validator cache check", completed=done - 1)

                    cached = None
                    if cache_path.is_file():
                        try:
                            candidate = read_json(cache_path)
                            if (
                                candidate.get("schema_version") == VALIDATION_CACHE_SCHEMA
                                and candidate.get("source_fingerprint") == source_fingerprint
                                and candidate.get("case_id") == case_id
                                and candidate.get("model") == model
                                and int(candidate.get("prefix", -1)) == int(prefix)
                                and candidate.get("status") == "passed"
                            ):
                                cached = candidate
                        except Exception:
                            cached = None

                    if cached is not None:
                        reproduced_rule_count += int(cached["reproduced_rule_count"])
                        max_prob_diff = max(max_prob_diff, float(cached["max_prob_diff"]))
                        multi_groups += int(cached["multi_rule_groups"])
                        max_cf_per_group = max(max_cf_per_group, int(cached["max_cf_per_group"]))
                        _set_status(stage="cached validation PASS", completed=done)
                        if done % 25 == 0:
                            print(f"{target_label} | validation cached PASS", flush=True)
                        continue

                    case = case_by_id[case_id]
                    factual = tuple(case.activities[:prefix])
                    timestamps = tuple(case.timestamps[:prefix])
                    seed = local_seed(model, prefix, case_id)

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

                    rec_manifest = manifest_index[tkey]
                    require(rec_manifest["pool_digest"] == pool_digest(pool), "Neighborhood digest mismatch")
                    require(int(rec_manifest["pool_size"]) == len(pool), "Neighborhood size mismatch")
                    require(int(rec_manifest["pool_class0"]) == int(np.sum(labels == 0)), "Neighborhood class0 mismatch")
                    require(int(rec_manifest["pool_class1"]) == int(np.sum(labels == 1)), "Neighborhood class1 mismatch")
                    diff = abs(float(rec_manifest["factual_probability"]) - float(probs[factual_idx]))
                    require(prob_close(rec_manifest["factual_probability"], probs[factual_idx]), "Factual probability mismatch")
                    max_prob_diff = max(max_prob_diff, diff)

                    _set_status(stage="LORE / DT / BN reconstruction")
                    result = run_all_surrogates_stratified(pool, labels, factual, factual_label)
                    fit_labels = np.asarray(result["fit_labels"], dtype=int)
                    eval_labels = np.asarray(result["eval_labels"], dtype=int)

                    actual_rules = []
                    target_multi_groups = 0
                    target_max_cf = 0
                    target_candidate_total = 0
                    for branch in result["branches"]:
                        surrogate = branch["branch"]
                        require("counterfactual_rules" in branch, "V2 core missing counterfactual_rules")
                        fact = branch["factual_rule"]
                        legacy_cf = branch["counterfactual_rule"]
                        cf_rules = list(branch["counterfactual_rules"])
                        require(
                            int(branch.get("counterfactual_rule_count", len(cf_rules))) == len(cf_rules),
                            "Declared candidate count mismatch",
                        )
                        ids = [str(rule.get("candidate_id", "")) for rule in cf_rules]
                        require(ids == [f"cf_{i:03d}" for i in range(len(cf_rules))], "candidate_id ordering mismatch")
                        require(len(ids) == len(set(ids)), "Duplicate candidate_id")

                        legacy_match = _legacy_matches_first(legacy_cf, cf_rules)
                        rec = metric_index[(case_id, model, int(prefix), surrogate)]
                        sfid = branch["surrogate_fidelity_eval"]
                        if sfid is None:
                            require(rec["surrogate_fidelity_eval"] == "", "Surrogate fidelity null mismatch")
                        else:
                            require(close(rec["surrogate_fidelity_eval"], sfid), "Surrogate fidelity mismatch")
                        require(int(rec["pool_size"]) == len(pool), "Metric pool size mismatch")
                        require(int(rec["fit_size"]) == len(result["fit_sequences"]), "Metric fit size mismatch")
                        require(int(rec["eval_size"]) == len(result["eval_sequences"]), "Metric eval size mismatch")
                        require(int(rec["fit_class0"]) == int(np.sum(fit_labels == 0)), "fit class0 mismatch")
                        require(int(rec["fit_class1"]) == int(np.sum(fit_labels == 1)), "fit class1 mismatch")
                        require(int(rec["eval_class0"]) == int(np.sum(eval_labels == 0)), "eval class0 mismatch")
                        require(int(rec["eval_class1"]) == int(np.sum(eval_labels == 1)), "eval class1 mismatch")
                        require(int(rec["cf_candidate_count"]) == len(cf_rules), "CF candidate count mismatch")
                        eligible_count = sum(
                            int(rule["support"]) > 0 and rule["fidelity"] is not None
                            for rule in cf_rules
                        )
                        require(int(rec["cf_eligible_candidate_count"]) == eligible_count, "eligible candidate count mismatch")
                        require(
                            str(rec["cf_multi_rule_available"]).lower() == str(bool(len(cf_rules) >= 2)).lower(),
                            "multi-rule availability mismatch",
                        )
                        require(
                            str(rec["legacy_best_matches_first_candidate"]).lower() == str(bool(legacy_match)).lower(),
                            "legacy-best compatibility mismatch",
                        )

                        fact_rec = rule_record(
                            case_id, model, prefix, surrogate,
                            "fact_000", "factual", fact, False,
                        )
                        if fact_rec is not None:
                            actual_rules.append(fact_rec)
                        for idx, cf in enumerate(cf_rules):
                            actual_rules.append(
                                rule_record(
                                    case_id, model, prefix, surrogate,
                                    cf["candidate_id"], "counterfactual", cf,
                                    is_legacy_best=(idx == 0 and legacy_match),
                                )
                            )
                        target_candidate_total += len(cf_rules)
                        target_multi_groups += int(len(cf_rules) >= 2)
                        target_max_cf = max(target_max_cf, len(cf_rules))

                    saved_target_rules = rules_by_target.get(tkey, [])
                    saved_map = {
                        (r["surrogate"], r["rule_type"], r["candidate_id"]): r
                        for r in saved_target_rules
                    }
                    actual_map = {
                        (r["surrogate"], r["rule_type"], r["candidate_id"]): r
                        for r in actual_rules
                    }
                    require(set(saved_map) == set(actual_map), "Saved/reproduced candidate key set mismatch")
                    for key in sorted(actual_map):
                        compare_rule(saved_map[key], actual_map[key])

                    require(
                        int(rec_manifest["counterfactual_candidate_count"]) == target_candidate_total,
                        "Manifest target CF candidate count mismatch",
                    )

                    target_payload = {
                        "schema_version": VALIDATION_CACHE_SCHEMA,
                        "source_fingerprint": source_fingerprint,
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "status": "passed",
                        "reproduced_rule_count": len(actual_rules),
                        "max_prob_diff": float(diff),
                        "multi_rule_groups": int(target_multi_groups),
                        "max_cf_per_group": int(target_max_cf),
                    }
                    write_json_atomic(cache_path, target_payload)

                    reproduced_rule_count += len(actual_rules)
                    multi_groups += target_multi_groups
                    max_cf_per_group = max(max_cf_per_group, target_max_cf)
                    _set_status(stage="target validation PASS", completed=done)
                    if done % 10 == 0 or done == expected_targets:
                        elapsed = time.perf_counter() - started
                        avg = elapsed / done
                        eta = avg * (expected_targets - done)
                        print(
                            f"{target_label} | PASS | CF candidates={target_candidate_total} | "
                            f"elapsed={_fmt_seconds(elapsed)} | ETA={_fmt_seconds(eta)}",
                            flush=True,
                        )
    finally:
        _STOP_HEARTBEAT.set()
        hb.join(timeout=1.0)

    require(reproduced_rule_count == len(saved_rules), "Saved/reproduced canonical rule count mismatch")

    eligible_cf = [
        r for r in saved_rules
        if r["rule_type"] == "counterfactual" and bool(r["eligible_for_reliability"])
    ]
    all_cf = [r for r in saved_rules if r["rule_type"] == "counterfactual"]
    supports = np.asarray([int(r["n_r"]) for r in eligible_cf], dtype=float)
    summary = read_json(OUT / "rule_support_summary.json")
    require(int(summary["eligible_counterfactual_rule_count"]) == len(eligible_cf), "Eligible CF count mismatch")
    if len(supports):
        require(close(summary["support_min"], np.min(supports)), "support min mismatch")
        require(close(summary["support_q25"], np.quantile(supports, 0.25)), "support q25 mismatch")
        require(close(summary["support_median"], np.median(supports)), "support median mismatch")
        require(close(summary["support_q75"], np.quantile(supports, 0.75)), "support q75 mismatch")
        require(close(summary["support_max"], np.max(supports)), "support max mismatch")
    require(summary["lambda_candidate_grid_for_next_phase"] == [1, 5, 10, 20, 50], "Lambda grid mismatch")
    require(summary["lambda_selected"] is None, "Lambda was selected prematurely")

    multi = read_json(OUT / "multi_rule_summary.json")
    require(int(multi["total_counterfactual_candidate_count"]) == len(all_cf), "Multi-rule total CF mismatch")
    require(int(multi["eligible_counterfactual_candidate_count"]) == len(eligible_cf), "Multi-rule eligible CF mismatch")
    require(int(multi["groups_with_multiple_counterfactual_candidates"]) == multi_groups, "Multi-rule group count mismatch")
    require(int(multi["max_counterfactual_candidates_in_one_group"]) == max_cf_per_group, "Max CF/group mismatch")
    require(multi_groups > 0, "Phase 4B-v2 exported no multi-rule target-local surrogate groups")

    checks = {
        "source_phase4b_passed": True,
        "same_100_validation_cases_reproduced": True,
        "all_1600_neighborhoods_reproduced": True,
        "all_blackbox_labels_and_factual_probabilities_reproduced": True,
        "stratified_local_splits_reproduced": True,
        "all_surrogate_fits_reproduced": True,
        "all_factual_and_multi_counterfactual_rules_reproduced": True,
        "candidate_ids_unique_and_deterministic": True,
        "heldout_support_fidelity_coverage_reproduced": True,
        "multi_rule_extension_nonempty": True,
        "lambda_not_selected": True,
        "test_cases_used": False,
    }
    result = {
        "scope": "ETCE Phase 4B-v2 multi-rule validation calibration bank",
        "status": "passed",
        "checks": checks,
        "calibration_case_count": CALIBRATION_CASE_COUNT,
        "local_target_count": expected_targets,
        "target_surrogate_group_count": len(metrics),
        "canonical_rule_count": len(saved_rules),
        "counterfactual_candidate_count": len(all_cf),
        "eligible_counterfactual_candidate_count": len(eligible_cf),
        "groups_with_multiple_counterfactual_candidates": multi_groups,
        "max_counterfactual_candidates_in_one_group": max_cf_per_group,
        "probability_validation": {
            "tolerance": PROB_TOL,
            "max_factual_probability_abs_diff": max_prob_diff,
        },
        "test_cases_used": False,
    }
    write_json(OUT / "validation_phase4b_v2.json", result)

    report_path = OUT / "PHASE4B_V2_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    old = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 4B-v2 validation.\n"
        "<!-- VALIDATION_END -->"
    )
    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. The identical 100-case validation selection, "
        "all 1600 deterministic process-valid neighborhoods, black-box labels, "
        "stratified local surrogate fits, factual rules, all multi-rule counterfactual "
        "candidates, candidate IDs, held-out support `n_r`, held-out fidelity `Fid(r)`, "
        "coverage, witnesses, and reliability-eligibility flags were independently "
        "reproduced. No test case was used.\n"
        "<!-- VALIDATION_END -->"
    )
    require(old in report, "Phase 4B-v2 report validation marker missing/already replaced")
    report_path.write_text(report.replace(old, new), encoding="utf-8")

    print("\n" + "=" * 96)
    print("PHASE 4B-v2 VALIDATION PASSED")
    print("=" * 96)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nNext: rerun target-local Stage-1 grouping audit on this v2 bank.")


if __name__ == "__main__":
    main()
