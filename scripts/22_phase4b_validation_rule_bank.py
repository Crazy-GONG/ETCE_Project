"""ETCE Phase 4B: expanded validation calibration rule bank.

Purpose
-------
Freeze Phase 4A.1 and build a larger rule-evidence bank using ONLY previously
unused validation cases. No test case is touched here.

This bank will be used in the next phase to choose the support discount lambda
and to calibrate BPA / evidence fusion before any formal test-time fusion.

Run:
    python scripts/22_phase4b_validation_rule_bank.py

The script is resumable at the target level through the target cache directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from pathlib import Path
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ETCE_ROOT", DEFAULT_ROOT)).expanduser().resolve()
LOCAL_DEPS = ROOT / ".deps"
# `.deps` is the optional Windows-local wheel directory used by the original
# workstation. Never place it on sys.path on Linux servers.
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
from etce.phase3.core import (
    CanonicalPredictor,
    load_process_constraints,
)
from etce.phase4.surrogate_core import (
    generate_process_valid_pool,
    sequence_hash,
)
from etce.phase4.stratified_core import (
    run_all_surrogates_stratified,
)
from etce.utils.io import read_json, write_csv, write_json, write_json_atomic

RESULTS3 = ROOT / "results/phase3"
PHASE4A = ROOT / "results/phase4/phase4a"
PHASE4A1 = ROOT / "results/phase4/phase4a1"
OUT = ROOT / "results/phase4/phase4b_validation_bank"
CACHE = OUT / "target_cache"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
CALIBRATION_CASE_COUNT = 100
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0


def stable_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def local_seed(model, prefix, case_id):
    text = f"{POOL_SEED_BASE}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(text).hexdigest()[:16], 16)


def pool_digest(sequences):
    payload = "\n".join(
        sorted(sequence_hash(seq) for seq in sequences)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_name(case_id, model, prefix):
    token = hashlib.sha256(
        f"{case_id}|{model}|{prefix}".encode("utf-8")
    ).hexdigest()[:20]
    return CACHE / f"{model}_P{prefix}_{token}.json"


def cache_is_complete(path, case_id, model, prefix):
    """Accept only a complete cache for the expected target."""
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        payload.get("case_id") == case_id
        and payload.get("model") == model
        and int(payload.get("prefix", -1)) == int(prefix)
        and isinstance(payload.get("metrics"), list)
        and len(payload["metrics"]) == 3
        and isinstance(payload.get("rules"), list)
        and isinstance(payload.get("pool_digest"), str)
    )


def require_server_inputs():
    required = [
        ROOT / "data/raw/BPI Challenge 2017.xes.gz",
        ROOT / "results/phase1_2/audit/common_cohort_case_ids.csv",
        ROOT / "results/phase2b/logs/validation_phase2b.json",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "phase3b/validation_phase3b.json",
        PHASE4A / "pilot_case_selection.json",
        PHASE4A1 / "validation_phase4a1.json",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required server inputs: " + ", ".join(missing))
    gates = {
        "phase2b": read_json(required[2]),
        "phase3b": read_json(required[4]),
        "phase4a1": read_json(required[6]),
    }
    failed = [name for name, payload in gates.items() if payload.get("status") != "passed"]
    if failed:
        raise RuntimeError("Required validated phase is not passed: " + ", ".join(failed))


def rule_record(case_id, model, prefix, surrogate, rule_type, rule):
    if rule is None:
        return None
    return {
        "case_id": case_id,
        "model": model,
        "prefix": int(prefix),
        "surrogate": surrogate,
        "rule_type": rule_type,
        "predicted_class": int(rule["predicted_class"]),
        "conditions": rule["conditions"],
        "n_r": int(rule["support"]),
        "Fid_r": rule["fidelity"],
        "coverage_eval": float(rule["coverage"]),
        "rule_length": int(rule["rule_length"]),
        "witness_sequence": rule.get("witness_sequence"),
        "witness_edit_count": rule.get("witness_edit_count"),
        "eligible_for_reliability": bool(
            int(rule["support"]) > 0
            and rule["fidelity"] is not None
        ),
    }


def main():
    require_server_inputs()
    p4a1 = read_json(
        PHASE4A1 / "validation_phase4a1.json"
    )
    if p4a1.get("status") != "passed":
        raise RuntimeError(
            "Validated Phase 4A.1 is required before Phase 4B"
        )

    pilot_ids = set(
        read_json(
            PHASE4A / "pilot_case_selection.json"
        )["case_ids"]
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    remaining_validation = sorted(
        (
            case
            for case in cases
            if case.split == "val"
            and case.case_id not in pilot_ids
        ),
        key=lambda case: (
            stable_rank(case.case_id),
            case.case_id,
        ),
    )
    calibration_cases = remaining_validation[
        :CALIBRATION_CASE_COUNT
    ]
    calibration_ids = [
        case.case_id
        for case in calibration_cases
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    write_json(
        OUT / "runtime_phase4b.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device_protocol": "cpu",
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
            "project_root_from_environment": "ETCE_ROOT" in os.environ,
        },
    )

    write_json(
        OUT / "calibration_case_selection.json",
        {
            "split": "validation",
            "case_count": CALIBRATION_CASE_COUNT,
            "case_ids": calibration_ids,
            "phase4a_pilot_case_ids_excluded": sorted(pilot_ids),
            "selection_rule": (
                "Exclude all Phase 4A/4A.1 pilot cases, then take the first "
                "100 remaining frozen validation cases sorted by SHA256(case_id)."
            ),
            "true_labels_used_for_selection": False,
            "blackbox_outputs_used_for_selection": False,
            "test_cases_used": False,
        },
    )

    write_json(
        OUT / "phase4b_run_config.json",
        {
            "scope": "expanded validation calibration rule bank",
            "source_protocol": "validated Phase 4A.1",
            "models": list(MODELS),
            "prefixes": list(PREFIXES),
            "calibration_case_count": CALIBRATION_CASE_COUNT,
            "pool_size": POOL_SIZE,
            "max_edits": MAX_EDITS,
            "max_options_per_position": MAX_OPTIONS,
            "pool_seed_base": POOL_SEED_BASE,
            "stratified_local_split": True,
            "same_surrogate_definitions": True,
            "same_surrogate_hyperparameters": True,
            "same_process_semantic_constraints": True,
            "lambda_selected_here": False,
            "bpa_or_ds_fusion_performed_here": False,
            "test_cases_used": False,
            "paper_test_statistics": False,
        },
    )

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)

    expected_targets = (
        CALIBRATION_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES)
    )
    done = 0

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            for case in calibration_cases:
                done += 1
                target_cache = cache_name(
                    case.case_id,
                    model,
                    prefix,
                )
                if target_cache.is_file() and cache_is_complete(
                    target_cache, case.case_id, model, prefix
                ):
                    if done % 25 == 0:
                        print(
                            f"[{done:04d}/{expected_targets}] cached "
                            f"{model} P{prefix} {case.case_id}",
                            flush=True,
                        )
                    continue

                factual = tuple(
                    case.activities[:prefix]
                )
                timestamps = tuple(
                    case.timestamps[:prefix]
                )
                seed = local_seed(
                    model,
                    prefix,
                    case.case_id,
                )

                pool = generate_process_valid_pool(
                    factual,
                    constraints,
                    rng_seed=seed,
                    pool_size=POOL_SIZE,
                    max_edits=MAX_EDITS,
                    max_options_per_position=MAX_OPTIONS,
                )
                probs = predictor.predict_proba(
                    pool,
                    [timestamps] * len(pool),
                )
                labels = (
                    np.asarray(probs) >= 0.5
                ).astype(int)

                factual_idx = pool.index(factual)
                factual_label = int(
                    labels[factual_idx]
                )

                result = run_all_surrogates_stratified(
                    pool,
                    labels,
                    factual,
                    factual_label,
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
                for branch in result["branches"]:
                    surrogate = branch["branch"]
                    fact = branch["factual_rule"]
                    cf = branch["counterfactual_rule"]

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
                            "factual_rule_support_eval": (
                                ""
                                if fact is None
                                else int(fact["support"])
                            ),
                            "factual_rule_fidelity_eval": (
                                ""
                                if (
                                    fact is None
                                    or fact["fidelity"] is None
                                )
                                else float(
                                    fact["fidelity"]
                                )
                            ),
                            "cf_rule_available": bool(
                                cf is not None
                            ),
                            "cf_rule_support_eval": (
                                ""
                                if cf is None
                                else int(cf["support"])
                            ),
                            "cf_rule_fidelity_eval": (
                                ""
                                if (
                                    cf is None
                                    or cf["fidelity"] is None
                                )
                                else float(
                                    cf["fidelity"]
                                )
                            ),
                            "cf_rule_length": (
                                ""
                                if cf is None
                                else int(cf["rule_length"])
                            ),
                            "cf_witness_edit_count": (
                                ""
                                if (
                                    cf is None
                                    or "witness_edit_count" not in cf
                                )
                                else int(
                                    cf["witness_edit_count"]
                                )
                            ),
                        }
                    )

                    fact_rec = rule_record(
                        case.case_id,
                        model,
                        prefix,
                        surrogate,
                        "factual",
                        fact,
                    )
                    cf_rec = rule_record(
                        case.case_id,
                        model,
                        prefix,
                        surrogate,
                        "counterfactual",
                        cf,
                    )
                    if fact_rec is not None:
                        rules.append(fact_rec)
                    if cf_rec is not None:
                        rules.append(cf_rec)

                payload = {
                    "case_id": case.case_id,
                    "model": model,
                    "prefix": int(prefix),
                    "pool_seed": int(seed),
                    "pool_digest": pool_digest(pool),
                    "pool_size": len(pool),
                    "pool_class0": int(
                        np.sum(labels == 0)
                    ),
                    "pool_class1": int(
                        np.sum(labels == 1)
                    ),
                    "factual_prediction": factual_label,
                    "factual_probability": float(
                        probs[factual_idx]
                    ),
                    "metrics": metrics,
                    "rules": rules,
                }
                write_json_atomic(
                    target_cache,
                    payload,
                )

                if done % 10 == 0 or done == expected_targets:
                    print(
                        f"[{done:04d}/{expected_targets}] "
                        f"{model} P{prefix} {case.case_id}: "
                        f"pool={len(pool)} "
                        f"class=({int(np.sum(labels==0))},"
                        f"{int(np.sum(labels==1))})",
                        flush=True,
                    )

    # Assemble deterministic final files from caches.
    metrics_all = []
    rules_all = []
    manifests = []

    for model in MODELS:
        for prefix in PREFIXES:
            for case in calibration_cases:
                payload = read_json(
                    cache_name(
                        case.case_id,
                        model,
                        prefix,
                    )
                )
                metrics_all.extend(
                    payload["metrics"]
                )
                rules_all.extend(
                    payload["rules"]
                )
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
                        "factual_prediction": payload[
                            "factual_prediction"
                        ],
                        "factual_probability": payload[
                            "factual_probability"
                        ],
                    }
                )

    write_csv(
        OUT / "neighborhood_manifest.csv",
        manifests,
    )
    write_csv(
        OUT / "surrogate_metrics_calibration.csv",
        metrics_all,
    )

    with (
        OUT / "canonical_rules_calibration.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as stream:
        for rule in rules_all:
            stream.write(
                json.dumps(
                    rule,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    # Aggregate diagnostics, but do not use them to tune surrogate definitions.
    aggregate = []
    for surrogate in ("LORE", "DT", "BN"):
        group = [
            r
            for r in metrics_all
            if r["surrogate"] == surrogate
        ]
        sfid = [
            float(r["surrogate_fidelity_eval"])
            for r in group
            if r["surrogate_fidelity_eval"] != ""
        ]
        cf_avail = [
            bool(r["cf_rule_available"])
            for r in group
        ]
        cf_support = [
            float(r["cf_rule_support_eval"])
            for r in group
            if r["cf_rule_support_eval"] != ""
        ]
        cf_fid = [
            float(r["cf_rule_fidelity_eval"])
            for r in group
            if r["cf_rule_fidelity_eval"] != ""
        ]
        aggregate.append(
            {
                "surrogate": surrogate,
                "targets": len(group),
                "mean_surrogate_fidelity_eval": (
                    ""
                    if not sfid
                    else float(np.mean(sfid))
                ),
                "cf_rule_availability": float(
                    np.mean(cf_avail)
                ),
                "mean_cf_rule_support_eval": (
                    ""
                    if not cf_support
                    else float(np.mean(cf_support))
                ),
                "median_cf_rule_support_eval": (
                    ""
                    if not cf_support
                    else float(np.median(cf_support))
                ),
                "mean_cf_rule_fidelity_eval": (
                    ""
                    if not cf_fid
                    else float(np.mean(cf_fid))
                ),
            }
        )
    write_csv(
        OUT / "surrogate_aggregate_calibration.csv",
        aggregate,
    )

    # Rule-support distribution for the next lambda-selection stage.
    eligible_cf = [
        r
        for r in rules_all
        if r["rule_type"] == "counterfactual"
        and r["eligible_for_reliability"]
    ]
    support_values = np.asarray(
        [int(r["n_r"]) for r in eligible_cf],
        dtype=float,
    )
    support_summary = {
        "eligible_counterfactual_rule_count": len(
            eligible_cf
        ),
        "support_min": (
            None
            if len(support_values) == 0
            else float(np.min(support_values))
        ),
        "support_q25": (
            None
            if len(support_values) == 0
            else float(
                np.quantile(support_values, 0.25)
            )
        ),
        "support_median": (
            None
            if len(support_values) == 0
            else float(np.median(support_values))
        ),
        "support_q75": (
            None
            if len(support_values) == 0
            else float(
                np.quantile(support_values, 0.75)
            )
        ),
        "support_max": (
            None
            if len(support_values) == 0
            else float(np.max(support_values))
        ),
        "lambda_candidate_grid_for_next_phase": [
            1, 5, 10, 20, 50
        ],
        "lambda_selected": None,
    }
    write_json(
        OUT / "rule_support_summary.json",
        support_summary,
    )

    report = [
        "# ETCE Phase 4B Expanded Validation Rule Bank",
        "",
        "> Calibration-only evidence bank. No test case is used.",
        "",
        "## Frozen protocol",
        "",
        f"- Cases: {CALIBRATION_CASE_COUNT} validation cases not used in Phase 4A/4A.1.",
        "- Black-boxes: RF, XGBoost, MLP, LSTM.",
        "- Prefixes: 5, 10, 15, 20.",
        "- Local neighborhood and process/semantic constraints: unchanged from validated Phase 4A.1.",
        "- Local split: deterministic class-stratified 70/30.",
        "- Surrogates and hyperparameters: unchanged.",
        "",
        "## Purpose",
        "",
        "This phase expands the validation-only rule evidence bank before selecting the support-discount parameter lambda.",
        "Each canonical rule stores `C_r`, predicted class `y_r`, held-out local support `n_r`, and held-out rule fidelity `Fid(r)`.",
        "Rules with `n_r=0` or undefined fidelity are retained for audit but are marked ineligible for reliability/BPA construction.",
        "",
        "## Aggregate diagnostics",
        "",
        "| Surrogate | Targets | Surrogate fidelity | CF rule availability | Mean CF support | Median CF support | Mean CF fidelity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        if v == "":
            return "-"
        return f"{float(v):.4f}"

    for r in aggregate:
        report.append(
            f"| {r['surrogate']} | {r['targets']} | "
            f"{fmt(r['mean_surrogate_fidelity_eval'])} | "
            f"{r['cf_rule_availability']:.4f} | "
            f"{fmt(r['mean_cf_rule_support_eval'])} | "
            f"{fmt(r['median_cf_rule_support_eval'])} | "
            f"{fmt(r['mean_cf_rule_fidelity_eval'])} |"
        )

    report += [
        "",
        "## Next phase",
        "",
        "Do not inspect the formal test rule bank yet.",
        "The next phase will use this validation bank to select `lambda` from `{1,5,10,20,50}` and then freeze the support-aware reliability/BPA protocol before formal test-time fusion.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 4B validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]

    (
        OUT / "PHASE4B_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("Phase 4B validation rule bank complete.")
    print(OUT / "PHASE4B_REPORT.md")
    print("Run scripts/23_validate_phase4b.py next.")


if __name__ == "__main__":
    main()
