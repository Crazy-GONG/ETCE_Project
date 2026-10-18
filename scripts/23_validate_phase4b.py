"""Independent validator for ETCE Phase 4B validation rule bank.

Run:
    python scripts/23_validate_phase4b.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("ETCE_ROOT", DEFAULT_ROOT)).expanduser().resolve()
LOCAL_DEPS = ROOT / ".deps"
# `.deps` contains workstation-specific Windows wheels and must not shadow the
# active Linux virtual environment on a server.
if sys.platform == "win32" and LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(LOCAL_DEPS))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

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
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
PHASE4A = ROOT / "results/phase4/phase4a"
PHASE4A1 = ROOT / "results/phase4/phase4a1"
OUT = ROOT / "results/phase4/phase4b_validation_bank"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
CALIBRATION_CASE_COUNT = 100
POOL_SIZE = 320
MAX_EDITS = 3
MAX_OPTIONS = 8
POOL_SEED_BASE = 0
PROB_TOL = 1e-5


def rows(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


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


def close(a, b, tol=1e-8):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=tol,
        abs_tol=tol,
    )


def prob_close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=PROB_TOL,
        abs_tol=PROB_TOL,
    )


def main():
    checks = {}

    required = [
        OUT / "PHASE4B_REPORT.md",
        OUT / "calibration_case_selection.json",
        OUT / "phase4b_run_config.json",
        OUT / "neighborhood_manifest.csv",
        OUT / "surrogate_metrics_calibration.csv",
        OUT / "surrogate_aggregate_calibration.csv",
        OUT / "canonical_rules_calibration.jsonl",
        OUT / "rule_support_summary.json",
        OUT / "runtime_phase4b.json",
        PHASE4A1 / "validation_phase4a1.json",
    ]
    require(
        all(path.is_file() for path in required),
        "Required Phase 4B artifact missing",
    )
    require(
        read_json(
            PHASE4A1 / "validation_phase4a1.json"
        ).get("status") == "passed",
        "Phase 4A.1 validation is not passed",
    )
    checks["required_outputs_and_source_phase_validated"] = True

    runtime = read_json(OUT / "runtime_phase4b.json")
    require(
        runtime.get("device_protocol") == "cpu"
        and all(
            runtime.get("versions", {}).get(name)
            for name in (
                "joblib", "numpy", "pandas", "scipy",
                "scikit_learn", "torch", "xgboost",
            )
        ),
        "Server runtime metadata is incomplete",
    )
    checks["portable_server_runtime_recorded"] = True

    config = read_json(
        OUT / "phase4b_run_config.json"
    )
    require(
        config["same_surrogate_definitions"] is True
        and config["same_surrogate_hyperparameters"] is True
        and config["same_process_semantic_constraints"] is True
        and config["stratified_local_split"] is True,
        "Frozen Phase 4A.1 protocol changed",
    )
    require(
        config["lambda_selected_here"] is False
        and config["bpa_or_ds_fusion_performed_here"] is False
        and config["test_cases_used"] is False,
        "Phase 4B scope violation",
    )
    checks["frozen_calibration_protocol_ok"] = True

    pilot_ids = set(
        read_json(
            PHASE4A / "pilot_case_selection.json"
        )["case_ids"]
    )
    cases, _ = load_frozen_common_cohort(ROOT)
    expected_cases = sorted(
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
    )[:CALIBRATION_CASE_COUNT]
    expected_ids = [
        case.case_id
        for case in expected_cases
    ]

    selection = read_json(
        OUT / "calibration_case_selection.json"
    )
    require(
        selection["case_ids"] == expected_ids,
        "Calibration case selection mismatch",
    )
    require(
        not (set(expected_ids) & pilot_ids),
        "Phase 4A pilot case leaked into Phase 4B calibration bank",
    )
    require(
        selection["true_labels_used_for_selection"] is False
        and selection["blackbox_outputs_used_for_selection"] is False
        and selection["test_cases_used"] is False,
        "Calibration case selection used forbidden information",
    )
    checks["unused_validation_case_selection_reproduced"] = True

    manifest = rows(
        OUT / "neighborhood_manifest.csv"
    )
    metrics = rows(
        OUT / "surrogate_metrics_calibration.csv"
    )
    expected_targets = (
        CALIBRATION_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES)
    )
    require(
        len(manifest) == expected_targets,
        "Neighborhood manifest target count mismatch",
    )
    require(
        len(metrics) == expected_targets * 3,
        "Surrogate metric row count mismatch",
    )

    manifest_index = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
        ): r
        for r in manifest
    }
    metric_index = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
        ): r
        for r in metrics
    }

    saved_rules = []
    with (
        OUT / "canonical_rules_calibration.jsonl"
    ).open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            line = line.strip()
            if line:
                saved_rules.append(json.loads(line))

    rule_index = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
            r["rule_type"],
        ): r
        for r in saved_rules
    }

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)
    case_by_id = {
        case.case_id: case
        for case in expected_cases
    }

    max_prob_diff = 0.0
    reproduced_rules = 0

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            for case_id in expected_ids:
                case = case_by_id[case_id]
                factual = tuple(
                    case.activities[:prefix]
                )
                timestamps = tuple(
                    case.timestamps[:prefix]
                )
                seed = local_seed(
                    model,
                    prefix,
                    case_id,
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

                rec_manifest = manifest_index[
                    (
                        case_id,
                        model,
                        int(prefix),
                    )
                ]
                require(
                    rec_manifest["pool_digest"]
                    == pool_digest(pool),
                    "Neighborhood sequence digest mismatch",
                )
                require(
                    int(rec_manifest["pool_size"])
                    == len(pool),
                    "Neighborhood size mismatch",
                )
                require(
                    int(rec_manifest["pool_class0"])
                    == int(np.sum(labels == 0))
                    and int(rec_manifest["pool_class1"])
                    == int(np.sum(labels == 1)),
                    "Neighborhood class counts mismatch",
                )
                diff = abs(
                    float(
                        rec_manifest[
                            "factual_probability"
                        ]
                    )
                    - float(
                        probs[factual_idx]
                    )
                )
                max_prob_diff = max(
                    max_prob_diff,
                    diff,
                )
                require(
                    prob_close(
                        rec_manifest[
                            "factual_probability"
                        ],
                        probs[factual_idx],
                    ),
                    "Factual probability mismatch",
                )

                result = run_all_surrogates_stratified(
                    pool,
                    labels,
                    factual,
                    factual_label,
                )

                for branch in result["branches"]:
                    surrogate = branch["branch"]
                    rec = metric_index[
                        (
                            case_id,
                            model,
                            int(prefix),
                            surrogate,
                        )
                    ]

                    sfid = branch[
                        "surrogate_fidelity_eval"
                    ]
                    if sfid is None:
                        require(
                            rec[
                                "surrogate_fidelity_eval"
                            ] == "",
                            "Surrogate fidelity null mismatch",
                        )
                    else:
                        require(
                            close(
                                rec[
                                    "surrogate_fidelity_eval"
                                ],
                                sfid,
                            ),
                            "Surrogate fidelity mismatch",
                        )

                    for rule_type, rule in (
                        (
                            "factual",
                            branch["factual_rule"],
                        ),
                        (
                            "counterfactual",
                            branch["counterfactual_rule"],
                        ),
                    ):
                        key = (
                            case_id,
                            model,
                            int(prefix),
                            surrogate,
                            rule_type,
                        )
                        saved = rule_index.get(key)

                        if rule is None:
                            require(
                                saved is None,
                                "Unexpected saved canonical rule",
                            )
                            continue

                        require(
                            saved is not None,
                            "Expected canonical rule missing",
                        )
                        require(
                            int(saved["predicted_class"])
                            == int(rule["predicted_class"]),
                            "Canonical rule class mismatch",
                        )
                        require(
                            saved["conditions"]
                            == rule["conditions"],
                            "Canonical rule condition mismatch",
                        )
                        require(
                            int(saved["n_r"])
                            == int(rule["support"]),
                            "Canonical rule support mismatch",
                        )

                        if rule["fidelity"] is None:
                            require(
                                saved["Fid_r"] is None,
                                "Canonical rule fidelity null mismatch",
                            )
                        else:
                            require(
                                close(
                                    saved["Fid_r"],
                                    rule["fidelity"],
                                ),
                                "Canonical rule fidelity mismatch",
                            )

                        expected_eligible = bool(
                            int(rule["support"]) > 0
                            and rule["fidelity"] is not None
                        )
                        require(
                            bool(
                                saved[
                                    "eligible_for_reliability"
                                ]
                            )
                            == expected_eligible,
                            "Rule reliability eligibility mismatch",
                        )
                        reproduced_rules += 1

    require(
        reproduced_rules == len(saved_rules),
        "Saved/reproduced canonical rule count mismatch",
    )
    checks["all_validation_neighborhoods_and_rules_reproduced"] = True

    # Support summary must be reproducible from eligible CF rules.
    eligible_cf = [
        r
        for r in saved_rules
        if r["rule_type"] == "counterfactual"
        and bool(
            r["eligible_for_reliability"]
        )
    ]
    supports = np.asarray(
        [int(r["n_r"]) for r in eligible_cf],
        dtype=float,
    )
    summary = read_json(
        OUT / "rule_support_summary.json"
    )
    require(
        int(
            summary[
                "eligible_counterfactual_rule_count"
            ]
        )
        == len(eligible_cf),
        "Eligible CF rule count mismatch",
    )
    if len(supports):
        require(
            close(summary["support_min"], np.min(supports))
            and close(
                summary["support_median"],
                np.median(supports),
            )
            and close(summary["support_max"], np.max(supports)),
            "Rule support summary mismatch",
        )
    require(
        summary["lambda_candidate_grid_for_next_phase"]
        == [1, 5, 10, 20, 50]
        and summary["lambda_selected"] is None,
        "Lambda grid/selection state mismatch",
    )
    checks["rule_support_inventory_recomputed"] = True

    result = {
        "scope": "ETCE Phase 4B validation calibration rule bank",
        "status": "passed",
        "checks": checks,
        "calibration_case_count": CALIBRATION_CASE_COUNT,
        "local_target_count": expected_targets,
        "surrogate_metric_rows": len(metrics),
        "canonical_rule_count": len(saved_rules),
        "eligible_counterfactual_rule_count": len(eligible_cf),
        "probability_validation": {
            "tolerance": PROB_TOL,
            "max_factual_probability_abs_diff": max_prob_diff,
        },
        "test_cases_used": False,
    }

    write_json(
        OUT / "validation_phase4b.json",
        result,
    )

    report_path = OUT / "PHASE4B_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    old = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 4B validation.\n"
        "<!-- VALIDATION_END -->"
    )
    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. The 100-case calibration selection, "
        "all deterministic process-valid neighborhoods, black-box labels, "
        "stratified local surrogate fits, canonical rules, held-out support "
        "`n_r`, held-out fidelity `Fid(r)`, and reliability-eligibility flags "
        "were independently reproduced. No test case was used.\n"
        "<!-- VALIDATION_END -->"
    )
    require(
        old in report,
        "Phase 4B report validation marker missing",
    )
    report_path.write_text(
        report.replace(old, new),
        encoding="utf-8",
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
