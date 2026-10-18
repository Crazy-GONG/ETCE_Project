"""ETCE Phase 3A.1 semantic-family constraint pilot.

This is a paired re-run of the validated Phase 3A pilot:
- same 20 pilot cases
- same canonical black-box checkpoints
- same prefixes
- same GA/search budget
- same train-only DFG and position support
- same timestamps

Only one rule is added:
    A_* -> A_*
    O_* -> O_*
    W_* -> W_*

Run:
    python scripts/14_phase3a1_semantic_constraint.py
"""
from __future__ import annotations

import csv
import hashlib
import json
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
    temporal_valid,
)
from etce.phase3.semantic_core import (
    activity_family,
    family_preserved,
    semantic_evolutionary_counterfactual_search,
)
from etce.utils.io import read_json, write_csv, write_json

RESULTS3 = ROOT / "results/phase3"
MODELS = ("RF", "XGBoost", "MLP", "LSTM")

# Keep Phase 3A search budget unchanged.
SEARCH_SEED = 0
POPULATION_SIZE = 40
GENERATIONS = 20
MUTATION_RATE = 0.70
CROSSOVER_RATE = 0.50
MAX_OPTIONS = 8


def csv_rows(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def main():
    phase3a_validation = read_json(
        RESULTS3 / "phase3a/validation_phase3a.json"
    )
    if phase3a_validation.get("status") != "passed":
        raise RuntimeError(
            "Validated Phase 3A is required before Phase 3A.1"
        )

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)

    selection = read_json(
        RESULTS3 / "phase3a/pilot_case_selection.json"
    )
    selected_ids = list(selection["case_ids"])

    cases, _ = load_frozen_common_cohort(ROOT)
    case_by_id = {
        case.case_id: case
        for case in cases
        if case.split == "test"
    }

    pilot_cases = []
    for case_id in selected_ids:
        if case_id not in case_by_id:
            raise AssertionError(
                f"Phase 3A pilot case missing from frozen test set: {case_id}"
            )
        pilot_cases.append(case_by_id[case_id])

    baseline_summary_rows = csv_rows(
        RESULTS3 / "phase3a/counterfactual_pilot_summary.csv"
    )
    baseline_summary = {
        (r["model"], int(r["prefix"])): r
        for r in baseline_summary_rows
    }

    out_dir = RESULTS3 / "phase3a1"
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        out_dir / "pilot_case_selection.json",
        {
            "source": "results/phase3/phase3a/pilot_case_selection.json",
            "selection_rule": (
                "Exactly reuse validated Phase 3A pilot cases for paired comparison."
            ),
            "case_ids": selected_ids,
            "pilot_case_count": len(selected_ids),
            "true_labels_used_for_selection": False,
            "test_metrics_used_for_selection": False,
        },
    )

    target_rows = []
    total = (
        len(MODELS)
        * len(PREFIXES)
        * len(pilot_cases)
    )
    done = 0

    for model_name in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model_name,
                prefix,
                canonical["items"][model_name][str(prefix)],
            )

            for case in pilot_cases:
                done += 1

                activities = tuple(
                    case.activities[:prefix]
                )
                timestamps = tuple(
                    case.timestamps[:prefix]
                )

                # Preserve the same deterministic seed construction as Phase 3A.
                seed_material = (
                    f"{SEARCH_SEED}|{model_name}|{prefix}|{case.case_id}"
                ).encode("utf-8")
                local_seed = int(
                    hashlib.sha256(seed_material).hexdigest()[:16],
                    16,
                )

                result = semantic_evolutionary_counterfactual_search(
                    predictor,
                    activities,
                    timestamps,
                    constraints,
                    rng_seed=local_seed,
                    population_size=POPULATION_SIZE,
                    generations=GENERATIONS,
                    mutation_rate=MUTATION_RATE,
                    crossover_rate=CROSSOVER_RATE,
                    max_options_per_position=MAX_OPTIONS,
                )

                cf_activities = result[
                    "counterfactual_activities"
                ]

                if cf_activities is not None:
                    changed_positions = [
                        i + 1
                        for i, (old, new) in enumerate(
                            zip(
                                activities,
                                cf_activities,
                            )
                        )
                        if old != new
                    ]

                    changed_from = [
                        activities[i - 1]
                        for i in changed_positions
                    ]
                    changed_to = [
                        cf_activities[i - 1]
                        for i in changed_positions
                    ]

                    changed_families = [
                        activity_family(
                            activities[i - 1]
                        )
                        for i in changed_positions
                    ]

                    family_ok = all(
                        family_preserved(
                            activities[i - 1],
                            cf_activities[i - 1],
                        )
                        for i in changed_positions
                    )

                    cf_flow = flow_valid(
                        cf_activities,
                        constraints["edge_count"],
                    )
                    cf_temporal = temporal_valid(
                        timestamps
                    )

                    immutable_preserved = all(
                        cf_activities[i] == factual
                        for i, factual in enumerate(
                            activities
                        )
                        if factual in (
                            TERMINAL_OFFER_ACTIVITIES
                            | DIRECT_PROXIES
                        )
                    )
                else:
                    changed_positions = []
                    changed_from = []
                    changed_to = []
                    changed_families = []
                    family_ok = None
                    cf_flow = None
                    cf_temporal = None
                    immutable_preserved = None

                target_rows.append(
                    {
                        "case_id": case.case_id,
                        "true_label_audit_only": int(case.label),
                        "model": model_name,
                        "prefix": int(prefix),
                        "canonical_seed": int(predictor.seed),
                        "search_seed": int(local_seed),
                        "success": bool(result["success"]),
                        "factual_prediction": int(
                            result["factual_prediction"]
                        ),
                        "target_label": int(
                            result["target_label"]
                        ),
                        "factual_probability": float(
                            result["factual_probability"]
                        ),
                        "factual_flow_valid": bool(
                            result["factual_flow_valid"]
                        ),
                        "factual_temporal_valid": bool(
                            temporal_valid(timestamps)
                        ),
                        "counterfactual_prediction": (
                            ""
                            if result["counterfactual_prediction"] is None
                            else int(
                                result[
                                    "counterfactual_prediction"
                                ]
                            )
                        ),
                        "counterfactual_probability": (
                            ""
                            if result["counterfactual_probability"] is None
                            else float(
                                result[
                                    "counterfactual_probability"
                                ]
                            )
                        ),
                        "edit_count": (
                            ""
                            if result["edit_count"] is None
                            else int(
                                result["edit_count"]
                            )
                        ),
                        "normalized_sequence_distance": (
                            ""
                            if result[
                                "normalized_sequence_distance"
                            ] is None
                            else float(
                                result[
                                    "normalized_sequence_distance"
                                ]
                            )
                        ),
                        "temporal_distance": float(
                            result["temporal_distance"]
                        ),
                        "counterfactual_flow_valid": (
                            ""
                            if cf_flow is None
                            else bool(cf_flow)
                        ),
                        "counterfactual_temporal_valid": (
                            ""
                            if cf_temporal is None
                            else bool(cf_temporal)
                        ),
                        "immutable_history_preserved": (
                            ""
                            if immutable_preserved is None
                            else bool(immutable_preserved)
                        ),
                        "event_family_preserved": (
                            ""
                            if family_ok is None
                            else bool(family_ok)
                        ),
                        "process_support": (
                            ""
                            if result["process_support"] is None
                            else float(
                                result["process_support"]
                            )
                        ),
                        "generation_found": (
                            ""
                            if result["generation_found"] is None
                            else int(
                                result["generation_found"]
                            )
                        ),
                        "candidates_evaluated": int(
                            result[
                                "candidates_evaluated"
                            ]
                        ),
                        "changed_positions": json.dumps(
                            changed_positions
                        ),
                        "changed_from": json.dumps(
                            changed_from,
                            ensure_ascii=False,
                        ),
                        "changed_to": json.dumps(
                            changed_to,
                            ensure_ascii=False,
                        ),
                        "changed_families": json.dumps(
                            changed_families,
                            ensure_ascii=False,
                        ),
                        "factual_activities": json.dumps(
                            activities,
                            ensure_ascii=False,
                        ),
                        "counterfactual_activities": (
                            ""
                            if cf_activities is None
                            else json.dumps(
                                cf_activities,
                                ensure_ascii=False,
                            )
                        ),
                        "timestamps_preserved_exactly": bool(
                            result["success"]
                        ),
                    }
                )

                print(
                    f"[{done:03d}/{total}] "
                    f"{model_name} P{prefix} {case.case_id}: "
                    f"success={result['success']} "
                    f"edits={result['edit_count']} "
                    f"eval={result['candidates_evaluated']}",
                    flush=True,
                )

    write_csv(
        out_dir / "counterfactual_semantic_targets.csv",
        target_rows,
    )

    summary_rows = []
    for model in MODELS:
        for prefix in PREFIXES:
            group = [
                r
                for r in target_rows
                if r["model"] == model
                and int(r["prefix"]) == int(prefix)
            ]
            successes = [
                r
                for r in group
                if r["success"]
            ]

            edits = np.asarray(
                [
                    int(r["edit_count"])
                    for r in successes
                ],
                dtype=float,
            )
            distances = np.asarray(
                [
                    float(
                        r[
                            "normalized_sequence_distance"
                        ]
                    )
                    for r in successes
                ],
                dtype=float,
            )
            evaluated = np.asarray(
                [
                    int(
                        r["candidates_evaluated"]
                    )
                    for r in group
                ],
                dtype=float,
            )

            base = baseline_summary[
                (model, int(prefix))
            ]
            baseline_success = int(
                base["successful"]
            )
            baseline_gsr = float(
                base["generation_success_rate"]
            )
            semantic_gsr = (
                len(successes)
                / len(group)
            )

            summary_rows.append(
                {
                    "model": model,
                    "prefix": int(prefix),
                    "targets": len(group),
                    "phase3a_successful": baseline_success,
                    "phase3a_gsr": baseline_gsr,
                    "phase3a1_successful": len(successes),
                    "phase3a1_gsr": semantic_gsr,
                    "delta_gsr": (
                        semantic_gsr
                        - baseline_gsr
                    ),
                    "success_retention_ratio": (
                        ""
                        if baseline_success == 0
                        else len(successes)
                        / baseline_success
                    ),
                    "factual_flow_valid_rate": (
                        sum(
                            bool(
                                r[
                                    "factual_flow_valid"
                                ]
                            )
                            for r in group
                        )
                        / len(group)
                    ),
                    "counterfactual_flow_valid_rate_successes": (
                        ""
                        if not successes
                        else sum(
                            bool(
                                r[
                                    "counterfactual_flow_valid"
                                ]
                            )
                            for r in successes
                        )
                        / len(successes)
                    ),
                    "counterfactual_temporal_valid_rate_successes": (
                        ""
                        if not successes
                        else sum(
                            bool(
                                r[
                                    "counterfactual_temporal_valid"
                                ]
                            )
                            for r in successes
                        )
                        / len(successes)
                    ),
                    "immutable_history_preserved_rate_successes": (
                        ""
                        if not successes
                        else sum(
                            bool(
                                r[
                                    "immutable_history_preserved"
                                ]
                            )
                            for r in successes
                        )
                        / len(successes)
                    ),
                    "event_family_preserved_rate_successes": (
                        ""
                        if not successes
                        else sum(
                            bool(
                                r[
                                    "event_family_preserved"
                                ]
                            )
                            for r in successes
                        )
                        / len(successes)
                    ),
                    "mean_edit_count_successes": (
                        ""
                        if not successes
                        else float(
                            edits.mean()
                        )
                    ),
                    "mean_normalized_sequence_distance_successes": (
                        ""
                        if not successes
                        else float(
                            distances.mean()
                        )
                    ),
                    "mean_candidates_evaluated": float(
                        evaluated.mean()
                    ),
                }
            )

    write_csv(
        out_dir / "counterfactual_semantic_summary.csv",
        summary_rows,
    )

    # Overall paired comparison.
    baseline_total = sum(
        int(r["successful"])
        for r in baseline_summary_rows
    )
    semantic_total = sum(
        int(r["phase3a1_successful"])
        for r in summary_rows
    )
    target_total = len(target_rows)

    overall = {
        "targets": target_total,
        "phase3a_successful": baseline_total,
        "phase3a_gsr": baseline_total / target_total,
        "phase3a1_successful": semantic_total,
        "phase3a1_gsr": semantic_total / target_total,
        "delta_gsr": (
            semantic_total
            - baseline_total
        ) / target_total,
        "success_retention_ratio": (
            semantic_total
            / baseline_total
            if baseline_total > 0
            else None
        ),
    }
    write_json(
        out_dir / "phase3a1_overall_comparison.json",
        overall,
    )

    write_json(
        out_dir / "phase3a1_run_config.json",
        {
            "paired_source_phase": "Phase 3A",
            "same_pilot_case_ids": True,
            "same_canonical_blackboxes": True,
            "same_prefixes": list(PREFIXES),
            "same_train_only_dfg": True,
            "same_position_support_rule": True,
            "same_timestamps": True,
            "only_new_constraint": (
                "event-family preservation: A_*->A_*, "
                "O_*->O_*, W_*->W_*"
            ),
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "crossover_rate": CROSSOVER_RATE,
            "max_options_per_position": MAX_OPTIONS,
            "true_label_used_by_search": False,
            "test_metrics_used_by_search": False,
            "paper_statistics": False,
        },
    )

    report = [
        "# ETCE Phase 3A.1 Semantic-Family Constraint Pilot",
        "",
        "> Paired follow-up to validated Phase 3A. This is still a method/software pilot, not final paper statistics.",
        "",
        "## Only change from Phase 3A",
        "",
        "- `A_*` may only be replaced by `A_*`.",
        "- `O_*` may only be replaced by `O_*`.",
        "- `W_*` may only be replaced by `W_*`.",
        "- All other cohort/model/prefix/DFG/timestamp/search-budget settings are unchanged.",
        "",
        "## Paired results",
        "",
        "| Model | Prefix | Phase3A GSR | Phase3A.1 GSR | Delta | CF DFG valid | Family valid | Mean edits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value):
        if value == "":
            return "-"
        return f"{float(value):.4f}"

    for r in summary_rows:
        report.append(
            f"| {r['model']} | {r['prefix']} | "
            f"{r['phase3a_gsr']:.4f} | "
            f"{r['phase3a1_gsr']:.4f} | "
            f"{r['delta_gsr']:+.4f} | "
            f"{fmt(r['counterfactual_flow_valid_rate_successes'])} | "
            f"{fmt(r['event_family_preserved_rate_successes'])} | "
            f"{fmt(r['mean_edit_count_successes'])} |"
        )

    report += [
        "",
        "## Overall",
        "",
        f"- Phase 3A: {baseline_total}/{target_total} successful, GSR={baseline_total/target_total:.4f}.",
        f"- Phase 3A.1: {semantic_total}/{target_total} successful, GSR={semantic_total/target_total:.4f}.",
        f"- Delta GSR: {(semantic_total-baseline_total)/target_total:+.4f}.",
        (
            f"- Success retention: {semantic_total/baseline_total:.4f}."
            if baseline_total
            else "- Success retention: not defined."
        ),
        "",
        "## Interpretation boundary",
        "",
        "Phase 3A.1 strengthens local process semantics by preserving the BPIC2017 event family at every edited position.",
        "A successful counterfactual still means only that the frozen black-box prediction flips while satisfying the train-only DFG, same-position support, immutable-history, timestamp-preservation, and family-preservation constraints.",
        "It does not establish causal recourse or human actionability.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 3A.1 validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]

    (
        out_dir
        / "PHASE3A1_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("Phase 3A.1 complete.")
    print(
        json.dumps(
            overall,
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        out_dir
        / "PHASE3A1_REPORT.md"
    )


if __name__ == "__main__":
    main()
