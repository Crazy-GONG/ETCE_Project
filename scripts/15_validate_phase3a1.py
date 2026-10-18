"""Independent validator for ETCE Phase 3A.1.

Checks:
- same paired pilot cases as validated Phase 3A
- same canonical checkpoints
- probability reproducibility
- actual prediction flip
- DFG validity
- timestamp validity
- immutable historical terminal events
- no forbidden mutation target
- event-family preservation at every edited position
- paired summary recomputation

Run:
    python scripts/15_validate_phase3a1.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

from etce.phase2.data import PREFIXES, load_frozen_common_cohort
from etce.phase3.core import (
    CanonicalPredictor,
    DIRECT_PROXIES,
    TERMINAL_OFFER_ACTIVITIES,
    flow_valid,
    load_process_constraints,
    sequence_edit_count,
    temporal_valid,
)
from etce.phase3.semantic_core import family_preserved
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
PROBABILITY_TOL = 1e-5


def rows(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b, tol=1e-8):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=tol,
        abs_tol=tol,
    )


def probability_close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=PROBABILITY_TOL,
        abs_tol=PROBABILITY_TOL,
    )


def validate():
    out = RESULTS3 / "phase3a1"
    checks = {}

    required = [
        out / "PHASE3A1_REPORT.md",
        out / "counterfactual_semantic_targets.csv",
        out / "counterfactual_semantic_summary.csv",
        out / "phase3a1_overall_comparison.json",
        out / "phase3a1_run_config.json",
        out / "pilot_case_selection.json",
        RESULTS3 / "phase3a/validation_phase3a.json",
        RESULTS3 / "phase3a/counterfactual_pilot_summary.csv",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "constraints/process_constraints.json",
    ]
    require(
        all(path.is_file() for path in required),
        "Required Phase 3A.1 artifact missing",
    )
    checks["required_outputs_exist"] = True

    phase3a_validation = read_json(
        RESULTS3 / "phase3a/validation_phase3a.json"
    )
    require(
        phase3a_validation.get("status") == "passed",
        "Phase 3A validation is not passed",
    )

    old_selection = read_json(
        RESULTS3 / "phase3a/pilot_case_selection.json"
    )
    new_selection = read_json(
        out / "pilot_case_selection.json"
    )
    require(
        new_selection["case_ids"] == old_selection["case_ids"],
        "Phase 3A.1 does not use exactly the Phase 3A pilot cases",
    )
    checks["paired_pilot_cases_identical"] = True

    config = read_json(
        out / "phase3a1_run_config.json"
    )
    require(
        config["same_pilot_case_ids"] is True
        and config["same_canonical_blackboxes"] is True
        and config["same_train_only_dfg"] is True
        and config["same_position_support_rule"] is True
        and config["same_timestamps"] is True,
        "Paired-design metadata is incomplete",
    )
    require(
        config["true_label_used_by_search"] is False
        and config["test_metrics_used_by_search"] is False,
        "Search improperly used label/test metric information",
    )
    checks["paired_design_metadata_ok"] = True

    constraints = load_process_constraints(RESULTS3)
    canonical = read_json(
        RESULTS3
        / "canonical_blackboxes/canonical_manifest.json"
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    case_by_id = {
        case.case_id: case
        for case in cases
        if case.split == "test"
    }

    targets = rows(
        out / "counterfactual_semantic_targets.csv"
    )

    expected_target_rows = (
        len(old_selection["case_ids"])
        * len(MODELS)
        * len(PREFIXES)
    )
    require(
        len(targets) == expected_target_rows,
        f"Expected {expected_target_rows} target rows",
    )

    max_factual_diff = 0.0
    max_cf_diff = 0.0

    for model in MODELS:
        for prefix in PREFIXES:
            group = [
                r
                for r in targets
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]

            require(
                [r["case_id"] for r in group]
                == old_selection["case_ids"],
                f"Case ordering mismatch: {model}/P{prefix}",
            )

            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            factual_acts = []
            factual_times = []
            cf_acts = []
            cf_times = []
            successful_rows = []

            for r in group:
                case = case_by_id[
                    r["case_id"]
                ]
                factual = tuple(
                    case.activities[:prefix]
                )
                times = tuple(
                    case.timestamps[:prefix]
                )

                require(
                    tuple(
                        json.loads(
                            r["factual_activities"]
                        )
                    )
                    == factual,
                    "Factual activity mismatch",
                )
                require(
                    temporal_valid(times),
                    "Factual timestamps invalid",
                )

                factual_acts.append(
                    factual
                )
                factual_times.append(
                    times
                )

                if r["success"].lower() != "true":
                    continue

                cf = tuple(
                    json.loads(
                        r[
                            "counterfactual_activities"
                        ]
                    )
                )

                require(
                    len(cf) == prefix,
                    "Counterfactual length mismatch",
                )
                require(
                    flow_valid(
                        cf,
                        constraints["edge_count"],
                    ),
                    "Successful counterfactual violates train-only DFG",
                )
                require(
                    temporal_valid(times),
                    "Successful counterfactual violates temporal validity",
                )

                changed = [
                    i
                    for i, (old, new) in enumerate(
                        zip(
                            factual,
                            cf,
                        )
                    )
                    if old != new
                ]

                require(
                    len(changed)
                    == int(
                        r["edit_count"]
                    ),
                    "Edit count mismatch",
                )
                require(
                    close(
                        len(changed) / prefix,
                        r[
                            "normalized_sequence_distance"
                        ],
                    ),
                    "Normalized sequence distance mismatch",
                )
                require(
                    close(
                        r["temporal_distance"],
                        0.0,
                    ),
                    "Phase 3A.1 temporal distance must remain zero",
                )

                for i in changed:
                    old = factual[i]
                    new = cf[i]

                    require(
                        new not in (
                            DIRECT_PROXIES
                            | TERMINAL_OFFER_ACTIVITIES
                        ),
                        "Successful counterfactual introduces forbidden outcome activity",
                    )
                    require(
                        family_preserved(
                            old,
                            new,
                        ),
                        f"Event family violation: {old} -> {new}",
                    )

                for i, old in enumerate(
                    factual
                ):
                    if old in (
                        DIRECT_PROXIES
                        | TERMINAL_OFFER_ACTIVITIES
                    ):
                        require(
                            cf[i] == old,
                            "Immutable historical terminal event changed",
                        )

                require(
                    r[
                        "event_family_preserved"
                    ].lower()
                    == "true",
                    "Recorded family-preservation flag is false",
                )
                require(
                    r[
                        "counterfactual_flow_valid"
                    ].lower()
                    == "true",
                    "Recorded flow-validity flag is false",
                )
                require(
                    r[
                        "counterfactual_temporal_valid"
                    ].lower()
                    == "true",
                    "Recorded temporal-validity flag is false",
                )
                require(
                    r[
                        "immutable_history_preserved"
                    ].lower()
                    == "true",
                    "Recorded immutability flag is false",
                )

                cf_acts.append(
                    cf
                )
                cf_times.append(
                    times
                )
                successful_rows.append(
                    r
                )

            factual_probs = (
                predictor.predict_proba(
                    factual_acts,
                    factual_times,
                )
            )
            for r, p in zip(
                group,
                factual_probs,
            ):
                recorded = float(
                    r["factual_probability"]
                )
                diff = abs(
                    recorded
                    - float(p)
                )
                max_factual_diff = max(
                    max_factual_diff,
                    diff,
                )
                require(
                    probability_close(
                        recorded,
                        p,
                    ),
                    (
                        f"Factual probability mismatch "
                        f"{model}/P{prefix}: "
                        f"abs_diff={diff:.3e}"
                    ),
                )

                pred = int(
                    float(p) >= 0.5
                )
                require(
                    pred
                    == int(
                        r["factual_prediction"]
                    ),
                    "Factual hard prediction mismatch",
                )
                require(
                    1 - pred
                    == int(
                        r["target_label"]
                    ),
                    "Target label mismatch",
                )

            if successful_rows:
                cf_probs = (
                    predictor.predict_proba(
                        cf_acts,
                        cf_times,
                    )
                )
                for r, p in zip(
                    successful_rows,
                    cf_probs,
                ):
                    recorded = float(
                        r[
                            "counterfactual_probability"
                        ]
                    )
                    diff = abs(
                        recorded
                        - float(p)
                    )
                    max_cf_diff = max(
                        max_cf_diff,
                        diff,
                    )
                    require(
                        probability_close(
                            recorded,
                            p,
                        ),
                        (
                            f"CF probability mismatch "
                            f"{model}/P{prefix}: "
                            f"abs_diff={diff:.3e}"
                        ),
                    )

                    pred = int(
                        float(p) >= 0.5
                    )
                    require(
                        pred
                        == int(
                            r[
                                "counterfactual_prediction"
                            ]
                        ),
                        "CF hard prediction mismatch",
                    )
                    require(
                        pred
                        == int(
                            r["target_label"]
                        ),
                        "Successful counterfactual did not flip prediction",
                    )

    checks[
        "prediction_flips_and_all_constraints_reproduced"
    ] = True

    # Recompute paired summaries.
    old_summary_rows = rows(
        RESULTS3
        / "phase3a/counterfactual_pilot_summary.csv"
    )
    old_summary = {
        (
            r["model"],
            int(r["prefix"]),
        ): r
        for r in old_summary_rows
    }

    new_summary_rows = rows(
        out
        / "counterfactual_semantic_summary.csv"
    )
    require(
        len(new_summary_rows)
        == len(MODELS)
        * len(PREFIXES),
        "Phase 3A.1 summary row count mismatch",
    )
    new_summary = {
        (
            r["model"],
            int(r["prefix"]),
        ): r
        for r in new_summary_rows
    }

    total_success = 0
    old_total_success = 0

    for model in MODELS:
        for prefix in PREFIXES:
            group = [
                r
                for r in targets
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]
            success = [
                r
                for r in group
                if r["success"].lower()
                == "true"
            ]

            sr = new_summary[
                (
                    model,
                    int(prefix),
                )
            ]
            old = old_summary[
                (
                    model,
                    int(prefix),
                )
            ]

            semantic_gsr = (
                len(success)
                / len(group)
            )
            baseline_gsr = float(
                old[
                    "generation_success_rate"
                ]
            )

            require(
                int(
                    sr["phase3a_successful"]
                )
                == int(
                    old["successful"]
                ),
                "Baseline success count mismatch in paired summary",
            )
            require(
                close(
                    sr["phase3a_gsr"],
                    baseline_gsr,
                ),
                "Baseline GSR mismatch in paired summary",
            )
            require(
                int(
                    sr["phase3a1_successful"]
                )
                == len(success),
                "Semantic success count mismatch",
            )
            require(
                close(
                    sr["phase3a1_gsr"],
                    semantic_gsr,
                ),
                "Semantic GSR mismatch",
            )
            require(
                close(
                    sr["delta_gsr"],
                    semantic_gsr
                    - baseline_gsr,
                ),
                "Delta GSR mismatch",
            )

            if success:
                require(
                    all(
                        r[
                            "event_family_preserved"
                        ].lower()
                        == "true"
                        for r in success
                    ),
                    "Family-validity summary mismatch",
                )

            total_success += len(
                success
            )
            old_total_success += int(
                old["successful"]
            )

    overall = read_json(
        out
        / "phase3a1_overall_comparison.json"
    )
    require(
        int(
            overall[
                "phase3a_successful"
            ]
        )
        == old_total_success,
        "Overall baseline success mismatch",
    )
    require(
        int(
            overall[
                "phase3a1_successful"
            ]
        )
        == total_success,
        "Overall semantic success mismatch",
    )
    require(
        close(
            overall[
                "phase3a1_gsr"
            ],
            total_success
            / len(targets),
        ),
        "Overall semantic GSR mismatch",
    )

    checks[
        "paired_summary_and_overall_comparison_recomputed"
    ] = True

    result = {
        "scope": (
            "ETCE Phase 3A.1"
        ),
        "status": (
            "passed"
        ),
        "checks": checks,
        "target_rows": len(
            targets
        ),
        "summary_rows": len(
            new_summary_rows
        ),
        "phase3a_successful": (
            old_total_success
        ),
        "phase3a1_successful": (
            total_success
        ),
        "phase3a_gsr": (
            old_total_success
            / len(targets)
        ),
        "phase3a1_gsr": (
            total_success
            / len(targets)
        ),
        "delta_gsr": (
            total_success
            - old_total_success
        )
        / len(targets),
        "probability_validation": {
            "tolerance": (
                PROBABILITY_TOL
            ),
            "max_factual_probability_abs_diff": (
                max_factual_diff
            ),
            "max_counterfactual_probability_abs_diff": (
                max_cf_diff
            ),
        },
    }

    write_json(
        out
        / "validation_phase3a1.json",
        result,
    )

    report_path = (
        out
        / "PHASE3A1_REPORT.md"
    )
    report = report_path.read_text(
        encoding="utf-8"
    )

    old_marker = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 3A.1 validation.\n"
        "<!-- VALIDATION_END -->"
    )
    new_marker = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. "
        "The paired pilot cases, canonical checkpoints, probabilities, "
        "prediction flips, train-only DFG validity, temporal validity, "
        "immutable historical events, forbidden-target exclusion, "
        "event-family preservation, and paired summary statistics "
        "were independently reproduced.\n"
        "<!-- VALIDATION_END -->"
    )

    require(
        old_marker in report,
        "Phase 3A.1 report validation marker missing",
    )
    report_path.write_text(
        report.replace(
            old_marker,
            new_marker,
        ),
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
    validate()
