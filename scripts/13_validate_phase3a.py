"""Independent validation for ETCE Phase 3A.

Run:
    python scripts/13_validate_phase3a.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

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
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
PILOT_CASE_COUNT = 20
FORMAL_SEEDS = (13, 21, 42, 73, 101)


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def stable_case_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def close(a, b, tol=1e-8):
    """Strict tolerance for deterministic scalar/statistical checks."""
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def probability_close(a, b, tol=1e-5):
    """Tolerance for float32 neural-network probabilities.

    Phase 3A may evaluate one case at a time during search, whereas validation
    recomputes several cases in one batch. PyTorch Linear/LSTM kernels can use
    different GEMV/GEMM execution paths, producing harmless float32 differences
    around 1e-7--1e-6 even with identical weights and inputs.

    Hard predictions and target flips are still checked exactly.
    """
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def validate():
    checks = {}
    out = RESULTS3 / "phase3a"
    max_factual_probability_abs_diff = 0.0
    max_counterfactual_probability_abs_diff = 0.0

    required = [
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "constraints/process_constraints.json",
        RESULTS3 / "constraints/dfg_edges.csv",
        RESULTS3 / "constraints/activity_position_counts.csv",
        out / "counterfactual_pilot_targets.csv",
        out / "counterfactual_pilot_summary.csv",
        out / "pilot_case_selection.json",
        out / "phase3a_run_config.json",
        out / "PHASE3A_REPORT.md",
    ]
    require(all(p.is_file() for p in required), "Required Phase 3A artifact missing")
    checks["required_outputs_exist"] = True

    # Recompute canonical checkpoint selection from validation metrics only.
    canonical = read_json(RESULTS3 / "canonical_blackboxes/canonical_manifest.json")
    phase2b_metrics = rows(ROOT / "results/phase2b/metrics/formal_metrics_per_seed.csv")

    for model in MODELS:
        for prefix in PREFIXES:
            group = [
                r for r in phase2b_metrics
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
                and r["split"] == "val"
            ]
            require(
                sorted(int(r["seed"]) for r in group) == list(FORMAL_SEEDS),
                f"Formal validation seeds missing: {model}/P{prefix}"
            )
            mean = sum(float(r["macro_f1"]) for r in group) / len(group)

            selected = sorted(
                group,
                key=lambda r: (
                    abs(float(r["macro_f1"]) - mean),
                    int(r["seed"]),
                ),
            )[0]

            item = canonical["items"][model][str(prefix)]

            require(
                int(item["selected_seed"]) == int(selected["seed"]),
                f"Canonical seed mismatch: {model}/P{prefix}"
            )

            require(
                close(item["validation_mean_macro_f1"], mean),
                f"Canonical validation mean mismatch: {model}/P{prefix}"
            )

    checks["canonical_selection_reproduced_without_test"] = True

    # Verify process constraints metadata.
    constraints = load_process_constraints(RESULTS3)

    require(
        constraints["scope"] == "train_only_pre_label_boundary",
        "Process constraint scope is not train-only pre-label"
    )

    forbidden = set(constraints["forbidden_mutation_targets"])

    require(
        forbidden == set(DIRECT_PROXIES | TERMINAL_OFFER_ACTIVITIES),
        "Forbidden mutation targets changed"
    )

    require(
        not (set(constraints["mutable_activities"]) & forbidden),
        "Forbidden mutation target appears in mutable activity set"
    )

    checks["process_constraints_scope_and_forbidden_targets_ok"] = True

    # Pilot case selection must be label/model independent.
    cases, _ = load_frozen_common_cohort(ROOT)

    test_cases = sorted(
        (c for c in cases if c.split == "test"),
        key=lambda c: (stable_case_rank(c.case_id), c.case_id),
    )

    expected_ids = [
        c.case_id
        for c in test_cases[:PILOT_CASE_COUNT]
    ]

    selection = read_json(
        out / "pilot_case_selection.json"
    )

    require(
        selection["case_ids"] == expected_ids,
        "Pilot case selection mismatch"
    )

    require(
        selection["true_labels_used_for_selection"] is False,
        "Labels used for pilot selection"
    )

    require(
        selection["test_metrics_used_for_selection"] is False,
        "Test metrics used for pilot selection"
    )

    checks["pilot_case_selection_reproduced"] = True

    case_by_id = {
        c.case_id: c
        for c in cases
    }

    targets = rows(
        out / "counterfactual_pilot_targets.csv"
    )

    expected_rows = (
        PILOT_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES)
    )

    require(
        len(targets) == expected_rows,
        f"Expected {expected_rows} target rows"
    )

    # Recompute factual and CF probabilities in model/prefix batches.
    for model in MODELS:
        for prefix in PREFIXES:

            group = [
                r for r in targets
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]

            require(
                len(group) == PILOT_CASE_COUNT,
                f"Target count mismatch {model}/P{prefix}"
            )

            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            require(
                all(
                    int(r["canonical_seed"]) == predictor.seed
                    for r in group
                ),
                f"Canonical seed metadata mismatch {model}/P{prefix}"
            )

            factual_acts = []
            factual_times = []

            cf_acts = []
            cf_times = []
            cf_rows = []

            for r in group:

                case = case_by_id[r["case_id"]]

                acts = tuple(
                    case.activities[:prefix]
                )

                times = tuple(
                    case.timestamps[:prefix]
                )

                recorded_factual = tuple(
                    json.loads(
                        r["factual_activities"]
                    )
                )

                require(
                    recorded_factual == acts,
                    "Factual activity sequence mismatch"
                )

                require(
                    temporal_valid(times),
                    "Factual temporal invalidity"
                )

                factual_acts.append(
                    acts
                )

                factual_times.append(
                    times
                )

                if r["success"].lower() == "true":

                    cf = tuple(
                        json.loads(
                            r["counterfactual_activities"]
                        )
                    )

                    require(
                        len(cf) == prefix,
                        "Counterfactual length mismatch"
                    )

                    require(
                        flow_valid(
                            cf,
                            constraints["edge_count"]
                        ),
                        "Successful CF violates DFG"
                    )

                    require(
                        temporal_valid(times),
                        "Successful CF violates temporal order"
                    )

                    require(
                        not any(
                            new in (
                                DIRECT_PROXIES
                                | TERMINAL_OFFER_ACTIVITIES
                            )
                            and new != old
                            for old, new in zip(
                                acts,
                                cf
                            )
                        ),
                        "Successful CF introduces a forbidden mutation target"
                    )

                    for old, new in zip(
                        acts,
                        cf
                    ):
                        if old in (
                            DIRECT_PROXIES
                            | TERMINAL_OFFER_ACTIVITIES
                        ):
                            require(
                                old == new,
                                "Immutable historical outcome changed"
                            )

                    edits = sequence_edit_count(
                        acts,
                        cf
                    )

                    require(
                        edits == int(
                            r["edit_count"]
                        ),
                        "Edit count mismatch"
                    )

                    require(
                        close(
                            edits / prefix,
                            r[
                                "normalized_sequence_distance"
                            ],
                        ),
                        "Normalized sequence distance mismatch"
                    )

                    require(
                        close(
                            r["temporal_distance"],
                            0.0
                        ),
                        "Phase 3A temporal distance must be zero"
                    )

                    cf_acts.append(
                        cf
                    )

                    cf_times.append(
                        times
                    )

                    cf_rows.append(
                        r
                    )

            # -------------------------------------------------------------
            # Recompute factual probabilities
            # -------------------------------------------------------------
            factual_probs = predictor.predict_proba(
                factual_acts,
                factual_times
            )

            for r, p in zip(
                group,
                factual_probs
            ):

                recorded_p = float(
                    r["factual_probability"]
                )

                recomputed_p = float(
                    p
                )

                diff = abs(
                    recorded_p
                    - recomputed_p
                )

                max_factual_probability_abs_diff = max(
                    max_factual_probability_abs_diff,
                    diff
                )

                require(
                    probability_close(
                        recorded_p,
                        recomputed_p
                    ),
                    (
                        f"Factual probability mismatch "
                        f"{model}/P{prefix}: "
                        f"recorded={recorded_p:.10f}, "
                        f"recomputed={recomputed_p:.10f}, "
                        f"abs_diff={diff:.3e}"
                    )
                )

                pred = int(
                    recomputed_p >= 0.5
                )

                require(
                    pred == int(
                        r["factual_prediction"]
                    ),
                    "Factual hard prediction mismatch"
                )

                require(
                    1 - pred
                    == int(
                        r["target_label"]
                    ),
                    "Target label mismatch"
                )

            # -------------------------------------------------------------
            # Recompute successful CF probabilities
            # -------------------------------------------------------------
            if cf_rows:

                cf_probs = predictor.predict_proba(
                    cf_acts,
                    cf_times
                )

                for r, p in zip(
                    cf_rows,
                    cf_probs
                ):

                    recorded_p = float(
                        r[
                            "counterfactual_probability"
                        ]
                    )

                    recomputed_p = float(
                        p
                    )

                    diff = abs(
                        recorded_p
                        - recomputed_p
                    )

                    max_counterfactual_probability_abs_diff = max(
                        max_counterfactual_probability_abs_diff,
                        diff
                    )

                    require(
                        probability_close(
                            recorded_p,
                            recomputed_p
                        ),
                        (
                            f"CF probability mismatch "
                            f"{model}/P{prefix}: "
                            f"recorded={recorded_p:.10f}, "
                            f"recomputed={recomputed_p:.10f}, "
                            f"abs_diff={diff:.3e}"
                        )
                    )

                    cf_pred = int(
                        recomputed_p >= 0.5
                    )

                    require(
                        cf_pred
                        == int(
                            r[
                                "counterfactual_prediction"
                            ]
                        ),
                        "CF hard prediction mismatch"
                    )

                    require(
                        cf_pred
                        == int(
                            r["target_label"]
                        ),
                        "Successful CF did not flip to target"
                    )

    checks[
        "all_successful_counterfactuals_reproduced_and_valid"
    ] = True

    # -------------------------------------------------------------
    # Recompute summary
    # -------------------------------------------------------------
    summary = rows(
        out / "counterfactual_pilot_summary.csv"
    )

    require(
        len(summary)
        == len(MODELS)
        * len(PREFIXES),
        "Summary row count mismatch"
    )

    sindex = {
        (
            r["model"],
            int(r["prefix"])
        ): r
        for r in summary
    }

    for model in MODELS:
        for prefix in PREFIXES:

            group = [
                r for r in targets
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]

            success = [
                r for r in group
                if r["success"].lower()
                == "true"
            ]

            sr = sindex[
                (
                    model,
                    prefix
                )
            ]

            require(
                int(
                    sr["targets"]
                )
                == len(group),
                "Summary target mismatch"
            )

            require(
                int(
                    sr["successful"]
                )
                == len(success),
                "Summary success mismatch"
            )

            require(
                close(
                    sr[
                        "generation_success_rate"
                    ],
                    len(success)
                    / len(group),
                ),
                "Summary GSR mismatch"
            )

            if success:

                require(
                    all(
                        r[
                            "counterfactual_flow_valid"
                        ].lower()
                        == "true"
                        for r in success
                    ),
                    "Summary contains flow-invalid success"
                )

                require(
                    all(
                        r[
                            "counterfactual_temporal_valid"
                        ].lower()
                        == "true"
                        for r in success
                    ),
                    "Summary contains temporal-invalid success"
                )

                require(
                    all(
                        r[
                            "immutable_history_preserved"
                        ].lower()
                        == "true"
                        for r in success
                    ),
                    "Summary contains immutable-history violation"
                )

    checks[
        "summary_recomputed"
    ] = True

    # -------------------------------------------------------------
    # Final validation result
    # -------------------------------------------------------------
    result = {
        "scope": (
            "ETCE Phase 3A"
        ),

        "status": (
            "passed"
        ),

        "checks": (
            checks
        ),

        "probability_validation": {
            "tolerance": (
                1e-5
            ),

            "reason": (
                "float32 batch-size/GEMV-GEMM numerical tolerance; "
                "hard predictions are checked exactly"
            ),

            "max_factual_probability_abs_diff": (
                max_factual_probability_abs_diff
            ),

            "max_counterfactual_probability_abs_diff": (
                max_counterfactual_probability_abs_diff
            ),
        },

        "target_rows": (
            len(targets)
        ),

        "summary_rows": (
            len(summary)
        ),

        "pilot_case_count": (
            PILOT_CASE_COUNT
        ),

        "models": (
            list(MODELS)
        ),

        "prefixes": (
            list(PREFIXES)
        ),
    }

    write_json(
        out / "validation_phase3a.json",
        result
    )

    # -------------------------------------------------------------
    # Update report validation marker
    # -------------------------------------------------------------
    report_path = (
        out / "PHASE3A_REPORT.md"
    )

    report = report_path.read_text(
        encoding="utf-8"
    )

    old = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 3A validation.\n"
        "<!-- VALIDATION_END -->"
    )

    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. "
        "Canonical checkpoint selection, "
        "train-only process constraints, "
        "pilot-case selection, "
        "black-box probabilities, "
        "prediction flips, "
        "DFG validity, "
        "temporal validity, "
        "immutability, "
        "and summary statistics "
        "were independently reproduced.\n"
        "<!-- VALIDATION_END -->"
    )

    require(
        old in report,
        "Report validation marker missing"
    )

    report_path.write_text(
        report.replace(
            old,
            new
        ),
        encoding="utf-8"
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )


if __name__ == "__main__":
    validate()