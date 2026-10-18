"""Independent validator for ETCE Phase 3B formal counterfactual evaluation.

Run:
    python scripts/17_validate_phase3b.py
"""
from __future__ import annotations

import csv
import hashlib
import itertools
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
from etce.phase3.semantic_core import family_preserved
from etce.utils.io import read_json, write_json

RESULTS3 = ROOT / "results/phase3"
OUT = RESULTS3 / "phase3b"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SEARCH_SEEDS = (13, 21, 42, 73, 101)
FORMAL_CASE_COUNT = 100
PROB_TOL = 1e-5
T_CRIT_95_DF4 = 2.7764451051977987


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


def prob_close(a, b):
    return math.isclose(
        float(a),
        float(b),
        rel_tol=PROB_TOL,
        abs_tol=PROB_TOL,
    )


def stable_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def mean_std_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half = float(
        T_CRIT_95_DF4
        * std
        / math.sqrt(len(values))
    )
    return mean, std, mean - half, mean + half


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def validate():
    checks = {}
    required = [
        OUT / "PHASE3B_REPORT.md",
        OUT / "formal_case_selection.json",
        OUT / "phase3b_run_config.json",
        OUT / "formal_counterfactual_runs.csv",
        OUT / "formal_metrics_per_seed.csv",
        OUT / "formal_metrics_summary.csv",
        OUT / "cross_seed_stability.csv",
        OUT / "phase3b_overall.json",
        RESULTS3 / "phase3a1/validation_phase3a1.json",
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json",
        RESULTS3 / "constraints/process_constraints.json",
    ]
    require(
        all(path.is_file() for path in required),
        "Required Phase 3B artifact missing",
    )
    checks["required_outputs_exist"] = True

    require(
        read_json(
            RESULTS3 / "phase3a1/validation_phase3a1.json"
        ).get("status") == "passed",
        "Phase 3A.1 validation is not passed",
    )

    cases, _ = load_frozen_common_cohort(ROOT)
    pilot_ids = set(
        read_json(
            RESULTS3 / "phase3a/pilot_case_selection.json"
        )["case_ids"]
    )
    expected_cases = sorted(
        (
            c for c in cases
            if c.split == "test"
            and c.case_id not in pilot_ids
        ),
        key=lambda c: (
            stable_rank(c.case_id),
            c.case_id,
        ),
    )[:FORMAL_CASE_COUNT]
    expected_ids = [
        c.case_id
        for c in expected_cases
    ]

    selection = read_json(
        OUT / "formal_case_selection.json"
    )
    require(
        selection["case_ids"] == expected_ids,
        "Formal case selection mismatch",
    )
    require(
        not (set(expected_ids) & pilot_ids),
        "Pilot case leaked into Phase 3B formal set",
    )
    require(
        selection["true_labels_used_for_selection"] is False
        and selection["model_outputs_used_for_selection"] is False
        and selection["test_metrics_used_for_selection"] is False,
        "Formal case selection used forbidden information",
    )
    checks["formal_unseen_case_selection_reproduced"] = True

    config = read_json(
        OUT / "phase3b_run_config.json"
    )
    require(
        config["search_seeds"] == list(SEARCH_SEEDS),
        "Formal search seeds changed",
    )
    require(
        config["paper_statistics"] is True,
        "Phase 3B not marked as paper statistics",
    )
    require(
        config["event_family_preservation"] is True
        and config["same_position_support"] is True
        and config["train_only_dfg"] is True,
        "Frozen semantic/process constraints changed",
    )
    require(
        config["true_label_used_by_search"] is False
        and config["test_metrics_used_by_search"] is False,
        "Search used forbidden information",
    )
    checks["frozen_formal_protocol_ok"] = True

    constraints = load_process_constraints(RESULTS3)
    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    case_by_id = {
        case.case_id: case
        for case in expected_cases
    }

    raw = rows(
        OUT / "formal_counterfactual_runs.csv"
    )
    expected_rows = (
        FORMAL_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES)
        * len(SEARCH_SEEDS)
    )
    require(
        len(raw) == expected_rows,
        f"Expected {expected_rows} raw formal rows",
    )
    keys = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            int(r["search_seed_base"]),
        )
        for r in raw
    }
    require(
        len(keys) == expected_rows,
        "Duplicate/missing formal run key",
    )

    max_factual_diff = 0.0
    max_cf_diff = 0.0

    # Reproduce black-box probabilities and all constraints.
    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            group = [
                r for r in raw
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]

            factual_acts = []
            factual_times = []
            cf_acts = []
            cf_times = []
            cf_rows = []

            for r in group:
                case = case_by_id[r["case_id"]]
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
                    ) == factual,
                    "Factual activity mismatch",
                )
                require(
                    temporal_valid(times),
                    "Factual timestamp invalidity",
                )

                factual_acts.append(factual)
                factual_times.append(times)

                success = (
                    r["success"].lower()
                    == "true"
                )

                if not success:
                    require(
                        r["failure_category"] in (
                            "no_admissible_mutation",
                            "admissible_candidates_no_flip",
                        ),
                        "Invalid failure category",
                    )
                    if r["failure_category"] == "no_admissible_mutation":
                        require(
                            int(r["candidates_evaluated"]) <= 1,
                            "No-mutation failure evaluated more than factual candidate",
                        )
                    else:
                        require(
                            int(r["candidates_evaluated"]) > 1,
                            "No-flip failure has no admissible candidates",
                        )
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
                    "CF length mismatch",
                )
                require(
                    flow_valid(
                        cf,
                        constraints["edge_count"],
                    ),
                    "Successful CF violates train-only DFG",
                )
                require(
                    temporal_valid(times),
                    "Successful CF violates temporal order",
                )

                changed = [
                    i
                    for i, (old, new) in enumerate(
                        zip(factual, cf)
                    )
                    if old != new
                ]
                require(
                    len(changed)
                    == int(r["edit_count"]),
                    "Edit count mismatch",
                )
                require(
                    close(
                        len(changed) / prefix,
                        r[
                            "normalized_sequence_distance"
                        ],
                    ),
                    "Normalized distance mismatch",
                )
                require(
                    close(
                        r["temporal_distance"],
                        0.0,
                    ),
                    "Temporal distance changed",
                )

                for i in changed:
                    old = factual[i]
                    new = cf[i]
                    require(
                        new not in (
                            DIRECT_PROXIES
                            | TERMINAL_OFFER_ACTIVITIES
                        ),
                        "Forbidden outcome activity introduced",
                    )
                    require(
                        family_preserved(old, new),
                        f"Family violation: {old} -> {new}",
                    )

                for i, old in enumerate(factual):
                    if old in (
                        DIRECT_PROXIES
                        | TERMINAL_OFFER_ACTIVITIES
                    ):
                        require(
                            cf[i] == old,
                            "Immutable historical terminal event changed",
                        )

                cf_acts.append(cf)
                cf_times.append(times)
                cf_rows.append(r)

            factual_probs = predictor.predict_proba(
                factual_acts,
                factual_times,
            )
            for r, p in zip(group, factual_probs):
                diff = abs(
                    float(r["factual_probability"])
                    - float(p)
                )
                max_factual_diff = max(
                    max_factual_diff,
                    diff,
                )
                require(
                    prob_close(
                        r["factual_probability"],
                        p,
                    ),
                    f"Factual probability mismatch {model}/P{prefix}",
                )
                pred = int(
                    float(p) >= 0.5
                )
                require(
                    pred
                    == int(
                        r["factual_prediction"]
                    ),
                    "Factual prediction mismatch",
                )
                require(
                    1 - pred
                    == int(
                        r["target_label"]
                    ),
                    "Target label mismatch",
                )

            if cf_rows:
                cf_probs = predictor.predict_proba(
                    cf_acts,
                    cf_times,
                )
                for r, p in zip(
                    cf_rows,
                    cf_probs,
                ):
                    diff = abs(
                        float(
                            r[
                                "counterfactual_probability"
                            ]
                        )
                        - float(p)
                    )
                    max_cf_diff = max(
                        max_cf_diff,
                        diff,
                    )
                    require(
                        prob_close(
                            r[
                                "counterfactual_probability"
                            ],
                            p,
                        ),
                        f"CF probability mismatch {model}/P{prefix}",
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
                        "Successful CF did not flip prediction",
                    )

    checks[
        "all_formal_predictions_and_constraints_reproduced"
    ] = True

    # Recompute per-seed metrics.
    per_seed = rows(
        OUT / "formal_metrics_per_seed.csv"
    )
    require(
        len(per_seed)
        == len(MODELS)
        * len(PREFIXES)
        * len(SEARCH_SEEDS),
        "Per-seed summary count mismatch",
    )
    pindex = {
        (
            r["model"],
            int(r["prefix"]),
            int(r["search_seed"]),
        ): r
        for r in per_seed
    }

    for model in MODELS:
        for prefix in PREFIXES:
            for seed in SEARCH_SEEDS:
                group = [
                    r for r in raw
                    if r["model"] == model
                    and int(r["prefix"]) == int(prefix)
                    and int(r["search_seed_base"]) == int(seed)
                ]
                success = [
                    r for r in group
                    if r["success"].lower()
                    == "true"
                ]
                rec = pindex[
                    (
                        model,
                        int(prefix),
                        int(seed),
                    )
                ]
                require(
                    int(rec["targets"])
                    == len(group),
                    "Per-seed target count mismatch",
                )
                require(
                    int(rec["successful"])
                    == len(success),
                    "Per-seed success count mismatch",
                )
                require(
                    close(
                        rec["gsr"],
                        len(success) / len(group),
                    ),
                    "Per-seed GSR mismatch",
                )

    checks[
        "per_seed_metrics_recomputed"
    ] = True

    # Recompute mean/std/CI.
    summary = rows(
        OUT / "formal_metrics_summary.csv"
    )
    require(
        len(summary)
        == len(MODELS)
        * len(PREFIXES),
        "Formal summary count mismatch",
    )
    sindex = {
        (
            r["model"],
            int(r["prefix"]),
        ): r
        for r in summary
    }

    metric_names = (
        "gsr",
        "mean_edit_count_successes",
        "mean_normalized_distance_successes",
        "mean_target_margin_successes",
        "mean_candidates_evaluated",
    )

    for model in MODELS:
        for prefix in PREFIXES:
            group = [
                r for r in per_seed
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]
            rec = sindex[
                (
                    model,
                    int(prefix),
                )
            ]
            for metric in metric_names:
                vals = [
                    float(r[metric])
                    for r in group
                    if r[metric] != ""
                ]
                mean, std, lo, hi = mean_std_ci(vals)
                require(
                    close(
                        rec[f"{metric}_mean"],
                        mean,
                    ),
                    f"{metric} mean mismatch",
                )
                require(
                    close(
                        rec[f"{metric}_std"],
                        std,
                    ),
                    f"{metric} std mismatch",
                )
                require(
                    close(
                        rec[f"{metric}_ci95_low"],
                        lo,
                    ),
                    f"{metric} CI low mismatch",
                )
                require(
                    close(
                        rec[f"{metric}_ci95_high"],
                        hi,
                    ),
                    f"{metric} CI high mismatch",
                )

    checks[
        "mean_std_ci_recomputed"
    ] = True

    # Recompute stability.
    stability = rows(
        OUT / "cross_seed_stability.csv"
    )
    require(
        len(stability)
        == FORMAL_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES),
        "Stability row count mismatch",
    )
    stindex = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
        ): r
        for r in stability
    }

    for model in MODELS:
        for prefix in PREFIXES:
            for case_id in expected_ids:
                group = sorted(
                    [
                        r for r in raw
                        if r["case_id"] == case_id
                        and r["model"] == model
                        and int(r["prefix"]) == int(prefix)
                    ],
                    key=lambda r: int(
                        r["search_seed_base"]
                    ),
                )
                success = [
                    r for r in group
                    if r["success"].lower()
                    == "true"
                ]
                seqs = [
                    r[
                        "counterfactual_activities"
                    ]
                    for r in success
                ]
                pos = [
                    json.loads(
                        r["changed_positions"]
                    )
                    for r in success
                ]
                pairs = [
                    jaccard(a, b)
                    for a, b in itertools.combinations(
                        pos,
                        2,
                    )
                ]

                rec = stindex[
                    (
                        case_id,
                        model,
                        int(prefix),
                    )
                ]
                require(
                    int(
                        rec["successful_seeds"]
                    )
                    == len(success),
                    "Stability success count mismatch",
                )
                require(
                    close(
                        rec[
                            "success_rate_across_seeds"
                        ],
                        len(success) / len(SEARCH_SEEDS),
                    ),
                    "Stability success rate mismatch",
                )
                require(
                    int(
                        rec[
                            "unique_counterfactuals"
                        ]
                    )
                    == len(set(seqs)),
                    "Unique CF count mismatch",
                )
                if pairs:
                    require(
                        close(
                            rec[
                                "mean_pairwise_modified_position_jaccard"
                            ],
                            float(np.mean(pairs)),
                        ),
                        "Jaccard stability mismatch",
                    )

    checks[
        "cross_seed_stability_recomputed"
    ] = True

    result = {
        "scope": "ETCE Phase 3B",
        "status": "passed",
        "checks": checks,
        "formal_case_count": FORMAL_CASE_COUNT,
        "pilot_cases_excluded": len(pilot_ids),
        "raw_run_rows": len(raw),
        "per_seed_rows": len(per_seed),
        "summary_rows": len(summary),
        "stability_rows": len(stability),
        "probability_validation": {
            "tolerance": PROB_TOL,
            "max_factual_probability_abs_diff": max_factual_diff,
            "max_counterfactual_probability_abs_diff": max_cf_diff,
        },
    }

    write_json(
        OUT / "validation_phase3b.json",
        result,
    )

    report_path = (
        OUT / "PHASE3B_REPORT.md"
    )
    report = report_path.read_text(
        encoding="utf-8"
    )
    old = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 3B validation.\n"
        "<!-- VALIDATION_END -->"
    )
    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. Formal unseen-case selection, "
        "all black-box probabilities, prediction flips, semantic/process "
        "constraints, failure categories, per-seed metrics, mean/std/95% CI, "
        "and cross-seed stability were independently reproduced.\n"
        "<!-- VALIDATION_END -->"
    )
    require(
        old in report,
        "Phase 3B report validation marker missing",
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
    validate()
