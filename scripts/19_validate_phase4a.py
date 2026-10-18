"""Independent validator for ETCE Phase 4A.

Run:
    python scripts/19_validate_phase4a.py
"""
from __future__ import annotations

import csv
import gzip
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
)
from etce.phase3.semantic_core import family_preserved
from etce.phase4.surrogate_core import (
    run_all_surrogates,
)
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
OUT = ROOT / "results/phase4/phase4a"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
PILOT_CASE_COUNT = 8
PROB_TOL = 1e-5


def rows(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(
                stream
            )
        )


def gzip_rows(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(
            csv.DictReader(
                stream
            )
        )


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


def prob_close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=PROB_TOL,
        abs_tol=PROB_TOL,
    )


def stable_rank(case_id):
    return hashlib.sha256(
        case_id.encode("utf-8")
    ).hexdigest()


def normalize_rule(rule):
    if rule is None:
        return None
    output = {
        "predicted_class": int(
            rule[
                "predicted_class"
            ]
        ),
        "conditions": rule[
            "conditions"
        ],
        "support_eval": int(
            rule["support"]
        ),
        "fidelity_eval": rule[
            "fidelity"
        ],
        "coverage_eval": float(
            rule["coverage"]
        ),
        "rule_length": int(
            rule["rule_length"]
        ),
        "witness_sequence": rule.get(
            "witness_sequence"
        ),
        "witness_edit_count": rule.get(
            "witness_edit_count"
        ),
    }
    return output


def validate():
    checks = {}

    required = [
        OUT / "PHASE4A_REPORT.md",
        OUT / "pilot_case_selection.json",
        OUT / "phase4a_run_config.json",
        OUT / "local_neighborhoods.csv.gz",
        OUT / "surrogate_metrics.csv",
        OUT / "surrogate_aggregate.csv",
        OUT / "canonical_rules.jsonl",
        RESULTS3 / "phase3b/validation_phase3b.json",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "constraints/process_constraints.json",
    ]
    require(
        all(
            path.is_file()
            for path in required
        ),
        "Required Phase 4A artifact missing",
    )
    checks[
        "required_outputs_exist"
    ] = True

    require(
        read_json(
            RESULTS3
            / "phase3b/validation_phase3b.json"
        ).get("status")
        == "passed",
        "Phase 3B validation is not passed",
    )

    selection = read_json(
        OUT
        / "pilot_case_selection.json"
    )
    cases, _ = (
        load_frozen_common_cohort(
            ROOT
        )
    )
    expected = sorted(
        (
            case
            for case in cases
            if case.split == "val"
        ),
        key=lambda case: (
            stable_rank(
                case.case_id
            ),
            case.case_id,
        ),
    )[:PILOT_CASE_COUNT]
    expected_ids = [
        case.case_id
        for case in expected
    ]

    require(
        selection["case_ids"]
        == expected_ids,
        "Phase 4A validation case selection mismatch",
    )
    require(
        selection[
            "true_labels_used_for_selection"
        ]
        is False
        and selection[
            "blackbox_outputs_used_for_selection"
        ]
        is False
        and selection[
            "test_cases_used"
        ]
        is False,
        "Phase 4A pilot selection used forbidden information",
    )
    checks[
        "validation_only_pilot_selection_reproduced"
    ] = True

    constraints = (
        load_process_constraints(
            RESULTS3
        )
    )
    canonical = read_json(
        RESULTS3
        / "canonical_blackboxes/canonical_manifest.json"
    )
    case_by_id = {
        case.case_id: case
        for case in expected
    }

    neighborhood = gzip_rows(
        OUT
        / "local_neighborhoods.csv.gz"
    )
    metrics = rows(
        OUT
        / "surrogate_metrics.csv"
    )
    saved_rules = []
    with (
        OUT
        / "canonical_rules.jsonl"
    ).open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            line = line.strip()
            if line:
                saved_rules.append(
                    json.loads(line)
                )

    # Index neighborhoods.
    grouped = {}
    for row in neighborhood:
        key = (
            row["case_id"],
            row["model"],
            int(row["prefix"]),
        )
        grouped.setdefault(
            key,
            [],
        ).append(row)

    expected_targets = (
        PILOT_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES)
    )
    require(
        len(grouped)
        == expected_targets,
        f"Expected {expected_targets} local neighborhoods",
    )

    metric_index = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
        ): r
        for r in metrics
    }
    require(
        len(metric_index)
        == expected_targets * 3,
        "Surrogate metric inventory mismatch",
    )

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

    max_prob_diff = 0.0
    reproduced_rules = {}

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical[
                    "items"
                ][model][str(prefix)],
            )

            for case_id in expected_ids:
                key = (
                    case_id,
                    model,
                    int(prefix),
                )
                rows_local = grouped[
                    key
                ]
                case = case_by_id[
                    case_id
                ]
                factual = tuple(
                    case.activities[:prefix]
                )
                timestamps = tuple(
                    case.timestamps[:prefix]
                )

                sequences = [
                    tuple(
                        json.loads(
                            r["activities"]
                        )
                    )
                    for r in rows_local
                ]
                recorded_probs = np.asarray(
                    [
                        float(
                            r[
                                "bb_probability"
                            ]
                        )
                        for r in rows_local
                    ],
                    dtype=float,
                )
                recorded_labels = np.asarray(
                    [
                        int(
                            r[
                                "bb_label"
                            ]
                        )
                        for r in rows_local
                    ],
                    dtype=int,
                )

                require(
                    len(set(sequences))
                    == len(sequences),
                    "Duplicate sequence in local neighborhood",
                )
                require(
                    factual
                    in set(
                        sequences
                    ),
                    "Factual sequence missing from local neighborhood",
                )

                for seq in sequences:
                    require(
                        flow_valid(
                            seq,
                            constraints[
                                "edge_count"
                            ],
                        ),
                        "Local neighborhood contains DFG-invalid sequence",
                    )
                    require(
                        sequence_edit_count(
                            factual,
                            seq,
                        )
                        <= 3,
                        "Local neighborhood exceeds max edits",
                    )
                    for old, new in zip(
                        factual,
                        seq,
                    ):
                        if old in (
                            TERMINAL_OFFER_ACTIVITIES
                            | DIRECT_PROXIES
                        ):
                            require(
                                new == old,
                                "Immutable historical outcome changed in neighborhood",
                            )
                        elif new != old:
                            require(
                                new not in (
                                    TERMINAL_OFFER_ACTIVITIES
                                    | DIRECT_PROXIES
                                ),
                                "Forbidden outcome activity introduced in neighborhood",
                            )
                            require(
                                family_preserved(
                                    old,
                                    new,
                                ),
                                "Event-family violation in neighborhood",
                            )

                recomputed = (
                    predictor.predict_proba(
                        sequences,
                        [timestamps]
                        * len(sequences),
                    )
                )
                diff = float(
                    np.max(
                        np.abs(
                            recorded_probs
                            - recomputed
                        )
                    )
                )
                max_prob_diff = max(
                    max_prob_diff,
                    diff,
                )
                require(
                    all(
                        prob_close(a, b)
                        for a, b in zip(
                            recorded_probs,
                            recomputed,
                        )
                    ),
                    f"Black-box probability mismatch: {model}/P{prefix}/{case_id}",
                )
                require(
                    np.array_equal(
                        recorded_labels,
                        (
                            recomputed
                            >= 0.5
                        ).astype(int),
                    ),
                    "Neighborhood black-box label mismatch",
                )

                factual_idx = sequences.index(
                    factual
                )
                factual_label = int(
                    recorded_labels[
                        factual_idx
                    ]
                )

                result = run_all_surrogates(
                    sequences,
                    recorded_labels,
                    factual,
                    factual_label,
                )

                for branch in result[
                    "branches"
                ]:
                    surrogate = branch[
                        "branch"
                    ]
                    metric = metric_index[
                        (
                            case_id,
                            model,
                            int(prefix),
                            surrogate,
                        )
                    ]

                    expected_fid = branch[
                        "surrogate_fidelity_eval"
                    ]
                    if expected_fid is None:
                        require(
                            metric[
                                "surrogate_fidelity_eval"
                            ]
                            == "",
                            "Surrogate fidelity null mismatch",
                        )
                    else:
                        require(
                            close(
                                metric[
                                    "surrogate_fidelity_eval"
                                ],
                                expected_fid,
                            ),
                            "Surrogate fidelity mismatch",
                        )

                    for rule_type, rule in (
                        (
                            "factual",
                            branch[
                                "factual_rule"
                            ],
                        ),
                        (
                            "counterfactual",
                            branch[
                                "counterfactual_rule"
                            ],
                        ),
                    ):
                        rkey = (
                            case_id,
                            model,
                            int(prefix),
                            surrogate,
                            rule_type,
                        )
                        if rule is None:
                            require(
                                rkey
                                not in rule_index,
                                "Unexpected saved rule exists",
                            )
                            continue

                        normalized = (
                            normalize_rule(
                                rule
                            )
                        )
                        saved = rule_index.get(
                            rkey
                        )
                        require(
                            saved is not None,
                            "Expected canonical rule missing",
                        )
                        require(
                            int(
                                saved[
                                    "predicted_class"
                                ]
                            )
                            == normalized[
                                "predicted_class"
                            ],
                            "Rule predicted class mismatch",
                        )
                        require(
                            saved[
                                "conditions"
                            ]
                            == normalized[
                                "conditions"
                            ],
                            "Rule conditions mismatch",
                        )
                        require(
                            int(
                                saved[
                                    "support_eval"
                                ]
                            )
                            == normalized[
                                "support_eval"
                            ],
                            "Rule support mismatch",
                        )
                        if (
                            normalized[
                                "fidelity_eval"
                            ]
                            is None
                        ):
                            require(
                                saved[
                                    "fidelity_eval"
                                ]
                                is None,
                                "Rule fidelity null mismatch",
                            )
                        else:
                            require(
                                close(
                                    saved[
                                        "fidelity_eval"
                                    ],
                                    normalized[
                                        "fidelity_eval"
                                    ],
                                ),
                                "Rule fidelity mismatch",
                            )
                        reproduced_rules[
                            rkey
                        ] = True

    checks[
        "all_neighborhoods_process_semantic_and_blackbox_valid"
    ] = True
    checks[
        "all_surrogates_and_canonical_rules_reproduced"
    ] = True

    require(
        len(reproduced_rules)
        == len(saved_rules),
        "Saved/reproduced canonical rule count mismatch",
    )

    result = {
        "scope": "ETCE Phase 4A",
        "status": "passed",
        "checks": checks,
        "pilot_case_count": PILOT_CASE_COUNT,
        "local_target_count": expected_targets,
        "surrogate_metric_rows": len(
            metrics
        ),
        "canonical_rule_count": len(
            saved_rules
        ),
        "neighborhood_rows": len(
            neighborhood
        ),
        "probability_validation": {
            "tolerance": PROB_TOL,
            "max_probability_abs_diff": (
                max_prob_diff
            ),
        },
    }

    write_json(
        OUT
        / "validation_phase4a.json",
        result,
    )

    report_path = (
        OUT
        / "PHASE4A_REPORT.md"
    )
    report = report_path.read_text(
        encoding="utf-8"
    )
    old = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 4A validation.\n"
        "<!-- VALIDATION_END -->"
    )
    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. Validation-only case selection, "
        "all process/semantic neighborhood constraints, black-box probabilities, "
        "local fit/eval partitioning, all three surrogate branches, canonical "
        "factual/counterfactual rules, held-out rule support and rule fidelity "
        "were independently reproduced.\n"
        "<!-- VALIDATION_END -->"
    )
    require(
        old in report,
        "Phase 4A validation marker missing",
    )
    report_path.write_text(
        report.replace(
            old,
            new,
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
