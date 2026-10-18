"""ETCE Phase 3A constrained evolutionary counterfactual pilot.

Run only after:
    python scripts/10_select_canonical_blackboxes.py
    python scripts/11_build_process_constraints.py

Then:
    python scripts/12_phase3a_counterfactual_pilot.py

This is a PILOT. It is not final paper statistics.
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
    TERMINAL_OFFER_ACTIVITIES,
    DIRECT_PROXIES,
    evolutionary_counterfactual_search,
    flow_valid,
    load_process_constraints,
    temporal_valid,
)
from etce.utils.io import read_json, write_csv, write_json

RESULTS3 = ROOT / "results/phase3"
MODELS = ("RF", "XGBoost", "MLP", "LSTM")
PILOT_CASE_COUNT = 20
SEARCH_SEED = 0
POPULATION_SIZE = 40
GENERATIONS = 20
MUTATION_RATE = 0.70
CROSSOVER_RATE = 0.50
MAX_OPTIONS = 8


def stable_case_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def main():
    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)

    cases, _ = load_frozen_common_cohort(ROOT)
    test_cases = sorted(
        (c for c in cases if c.split == "test"),
        key=lambda c: (stable_case_rank(c.case_id), c.case_id),
    )
    pilot_cases = test_cases[:PILOT_CASE_COUNT]

    selected_ids = [c.case_id for c in pilot_cases]
    write_json(
        RESULTS3 / "phase3a/pilot_case_selection.json",
        {
            "selection_rule": (
                "Select first 20 test case IDs after sorting by SHA256(case_id), "
                "independent of true label and model output."
            ),
            "pilot_case_count": PILOT_CASE_COUNT,
            "case_ids": selected_ids,
            "true_labels_used_for_selection": False,
            "test_metrics_used_for_selection": False,
        },
    )

    rows = []
    total = len(MODELS) * len(PREFIXES) * len(pilot_cases)
    done = 0

    for model_name in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model_name,
                prefix,
                canonical["items"][model_name][str(prefix)],
            )

            for case_index, case in enumerate(pilot_cases):
                done += 1
                activities = tuple(case.activities[:prefix])
                timestamps = tuple(case.timestamps[:prefix])

                # Deterministic search seed is based only on pilot seed/model/prefix/case ID.
                seed_material = (
                    f"{SEARCH_SEED}|{model_name}|{prefix}|{case.case_id}"
                ).encode("utf-8")
                local_seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)

                result = evolutionary_counterfactual_search(
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

                cf_activities = result["counterfactual_activities"]
                if cf_activities is not None:
                    changed_positions = [
                        i + 1
                        for i, (a, b) in enumerate(zip(activities, cf_activities))
                        if a != b
                    ]
                    changed_from = [activities[i - 1] for i in changed_positions]
                    changed_to = [cf_activities[i - 1] for i in changed_positions]
                    cf_flow = flow_valid(
                        cf_activities, constraints["edge_count"]
                    )
                    cf_temporal = temporal_valid(timestamps)
                    immutable_preserved = all(
                        cf_activities[i] == factual
                        for i, factual in enumerate(activities)
                        if factual in TERMINAL_OFFER_ACTIVITIES | DIRECT_PROXIES
                    )
                else:
                    changed_positions = []
                    changed_from = []
                    changed_to = []
                    cf_flow = None
                    cf_temporal = None
                    immutable_preserved = None

                row = {
                    "case_id": case.case_id,
                    "true_label_audit_only": int(case.label),
                    "model": model_name,
                    "prefix": int(prefix),
                    "canonical_seed": int(predictor.seed),
                    "search_seed": int(local_seed),
                    "success": bool(result["success"]),
                    "factual_prediction": int(result["factual_prediction"]),
                    "target_label": int(result["target_label"]),
                    "factual_probability": float(result["factual_probability"]),
                    "factual_flow_valid": bool(result["factual_flow_valid"]),
                    "factual_temporal_valid": bool(temporal_valid(timestamps)),
                    "counterfactual_prediction": (
                        "" if result["counterfactual_prediction"] is None
                        else int(result["counterfactual_prediction"])
                    ),
                    "counterfactual_probability": (
                        "" if result["counterfactual_probability"] is None
                        else float(result["counterfactual_probability"])
                    ),
                    "edit_count": (
                        "" if result["edit_count"] is None
                        else int(result["edit_count"])
                    ),
                    "normalized_sequence_distance": (
                        "" if result["normalized_sequence_distance"] is None
                        else float(result["normalized_sequence_distance"])
                    ),
                    "temporal_distance": float(result["temporal_distance"]),
                    "counterfactual_flow_valid": (
                        "" if cf_flow is None else bool(cf_flow)
                    ),
                    "counterfactual_temporal_valid": (
                        "" if cf_temporal is None else bool(cf_temporal)
                    ),
                    "immutable_history_preserved": (
                        "" if immutable_preserved is None
                        else bool(immutable_preserved)
                    ),
                    "process_support": (
                        "" if result["process_support"] is None
                        else float(result["process_support"])
                    ),
                    "generation_found": (
                        "" if result["generation_found"] is None
                        else int(result["generation_found"])
                    ),
                    "candidates_evaluated": int(result["candidates_evaluated"]),
                    "changed_positions": json.dumps(changed_positions),
                    "changed_from": json.dumps(changed_from, ensure_ascii=False),
                    "changed_to": json.dumps(changed_to, ensure_ascii=False),
                    "factual_activities": json.dumps(activities, ensure_ascii=False),
                    "counterfactual_activities": (
                        "" if cf_activities is None
                        else json.dumps(cf_activities, ensure_ascii=False)
                    ),
                    "timestamps_preserved_exactly": bool(result["success"]),
                }
                rows.append(row)

                print(
                    f"[{done:03d}/{total}] {model_name} P{prefix} "
                    f"{case.case_id}: success={result['success']} "
                    f"edits={result['edit_count']} "
                    f"eval={result['candidates_evaluated']}",
                    flush=True,
                )

    out_dir = RESULTS3 / "phase3a"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "counterfactual_pilot_targets.csv", rows)

    # Summary per model/prefix.
    summary = []
    for model in MODELS:
        for prefix in PREFIXES:
            group = [
                r for r in rows
                if r["model"] == model and int(r["prefix"]) == int(prefix)
            ]
            successes = [r for r in group if r["success"]]
            edits = np.asarray([int(r["edit_count"]) for r in successes], dtype=float)
            distances = np.asarray(
                [float(r["normalized_sequence_distance"]) for r in successes],
                dtype=float,
            )
            evaluated = np.asarray(
                [int(r["candidates_evaluated"]) for r in group], dtype=float
            )
            summary.append(
                {
                    "model": model,
                    "prefix": int(prefix),
                    "targets": len(group),
                    "successful": len(successes),
                    "generation_success_rate": len(successes) / len(group),
                    "factual_flow_valid_rate": sum(
                        bool(r["factual_flow_valid"]) for r in group
                    ) / len(group),
                    "counterfactual_flow_valid_rate_successes": (
                        "" if not successes
                        else sum(bool(r["counterfactual_flow_valid"]) for r in successes)
                        / len(successes)
                    ),
                    "counterfactual_temporal_valid_rate_successes": (
                        "" if not successes
                        else sum(bool(r["counterfactual_temporal_valid"]) for r in successes)
                        / len(successes)
                    ),
                    "immutable_history_preserved_rate_successes": (
                        "" if not successes
                        else sum(bool(r["immutable_history_preserved"]) for r in successes)
                        / len(successes)
                    ),
                    "mean_edit_count_successes": (
                        "" if not successes else float(edits.mean())
                    ),
                    "mean_normalized_sequence_distance_successes": (
                        "" if not successes else float(distances.mean())
                    ),
                    "mean_candidates_evaluated": float(evaluated.mean()),
                }
            )

    write_csv(out_dir / "counterfactual_pilot_summary.csv", summary)

    report_lines = [
        "# ETCE Phase 3A Counterfactual Neighborhood Pilot",
        "",
        "> Phase 3A is a software/method pilot. These are not final paper statistics.",
        "",
        "## Protocol",
        "",
        f"- Pilot test cases: {PILOT_CASE_COUNT}, selected by SHA256(case_id), independent of labels/model outputs.",
        "- Canonical black-box checkpoint: validation-mean representative seed selected without test metrics.",
        "- Search: process-constrained evolutionary search.",
        "- Candidate representation: fixed-length activity sequence.",
        "- Phase 3A edit operator: activity substitution only.",
        "- Existing historical terminal offer activities are immutable.",
        "- Mutation targets exclude `A_Pending/A_Denied/A_Cancelled` and `O_Accepted/O_Refused/O_Cancelled`.",
        "- Every counterfactual candidate must satisfy the train-only pre-label DFG.",
        "- Replacement activity must also have appeared at the same prefix position in training.",
        "- Factual timestamps are preserved exactly; Phase 3A does not claim temporal recourse.",
        f"- Population={POPULATION_SIZE}, generations={GENERATIONS}, mutation_rate={MUTATION_RATE}, crossover_rate={CROSSOVER_RATE}.",
        "",
        "## Pilot results",
        "",
        "| Model | Prefix | Targets | Success | GSR | Factual DFG valid | CF DFG valid | CF temporal valid | Mean edits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        def fmt(v):
            return "-" if v == "" else f"{float(v):.4f}"
        report_lines.append(
            f"| {r['model']} | {r['prefix']} | {r['targets']} | {r['successful']} | "
            f"{r['generation_success_rate']:.4f} | {r['factual_flow_valid_rate']:.4f} | "
            f"{fmt(r['counterfactual_flow_valid_rate_successes'])} | "
            f"{fmt(r['counterfactual_temporal_valid_rate_successes'])} | "
            f"{fmt(r['mean_edit_count_successes'])} |"
        )

    report_lines += [
        "",
        "## Interpretation boundary",
        "",
        "A successful Phase 3A counterfactual means the frozen black-box hard prediction flipped while the generated activity sequence satisfied the train-derived DFG constraint.",
        "This pilot does **not** yet establish actionability, causal recourse, or Petri-net conformance. Timestamps are preserved rather than optimized.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 3A validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]
    (out_dir / "PHASE3A_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    write_json(
        out_dir / "phase3a_run_config.json",
        {
            "pilot_case_count": PILOT_CASE_COUNT,
            "search_seed_role": "software/method pilot only",
            "search_seed_base": SEARCH_SEED,
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "crossover_rate": CROSSOVER_RATE,
            "max_options_per_position": MAX_OPTIONS,
            "true_label_used_by_search": False,
            "test_metrics_used_by_search": False,
            "timestamps_modified": False,
            "paper_statistics": False,
        },
    )

    print("Phase 3A pilot complete.")
    print(out_dir / "PHASE3A_REPORT.md")


if __name__ == "__main__":
    main()
