"""ETCE Phase 3B formal counterfactual-generation evaluation.

Frozen method:
    validated Phase 3A.1 semantic-family-preserving constrained search.

Formal evaluation design:
- Exclude all 20 Phase 3A/3A.1 pilot cases.
- Select 100 previously unseen test cases by SHA256(case_id), independent of labels/predictions.
- Use all 4 canonical black-box models and Prefix-5/10/15/20.
- Use five formal search seeds: 13, 21, 42, 73, 101.
- Do not tune search parameters.
- Report GSR, minimality, validity, failure categories and cross-seed stability.

Run:
    python scripts/16_phase3b_formal_counterfactual.py

The script is resumable: completed keys already present in the raw CSV are skipped.
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
    temporal_valid,
)
from etce.phase3.semantic_core import (
    family_preserved,
    semantic_evolutionary_counterfactual_search,
)
from etce.utils.io import read_json, write_csv, write_json

RESULTS3 = ROOT / "results/phase3"
OUT = RESULTS3 / "phase3b"

MODELS = ("RF", "XGBoost", "MLP", "LSTM")
SEARCH_SEEDS = (13, 21, 42, 73, 101)
FORMAL_CASE_COUNT = 100

# Frozen from validated Phase 3A.1.
POPULATION_SIZE = 40
GENERATIONS = 20
MUTATION_RATE = 0.70
CROSSOVER_RATE = 0.50
MAX_OPTIONS = 8

T_CRIT_95_DF4 = 2.7764451051977987


def rows(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def stable_rank(case_id):
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()


def local_search_seed(base_seed, model, prefix, case_id):
    material = f"{base_seed}|{model}|{prefix}|{case_id}".encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:16], 16)


def changed_positions(factual, counterfactual):
    return [
        i + 1
        for i, (old, new) in enumerate(zip(factual, counterfactual))
        if old != new
    ]


def append_csv(path, row, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def mean_std_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) <= 1:
        return mean, 0.0, mean, mean
    std = float(values.std(ddof=1))
    half = float(T_CRIT_95_DF4 * std / math.sqrt(len(values)))
    return mean, std, mean - half, mean + half


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    phase3a1_validation = read_json(
        RESULTS3 / "phase3a1/validation_phase3a1.json"
    )
    if phase3a1_validation.get("status") != "passed":
        raise RuntimeError("Validated Phase 3A.1 is required before Phase 3B")

    canonical = read_json(
        RESULTS3 / "canonical_blackboxes/canonical_manifest.json"
    )
    constraints = load_process_constraints(RESULTS3)

    # Pilot cases are permanently excluded from formal Phase 3B evaluation.
    pilot = read_json(
        RESULTS3 / "phase3a/pilot_case_selection.json"
    )
    pilot_ids = set(pilot["case_ids"])

    cases, _ = load_frozen_common_cohort(ROOT)
    remaining_test = sorted(
        (
            c for c in cases
            if c.split == "test"
            and c.case_id not in pilot_ids
        ),
        key=lambda c: (stable_rank(c.case_id), c.case_id),
    )

    formal_cases = remaining_test[:FORMAL_CASE_COUNT]
    formal_ids = [c.case_id for c in formal_cases]

    OUT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT / "formal_case_selection.json",
        {
            "case_count": FORMAL_CASE_COUNT,
            "case_ids": formal_ids,
            "selection_rule": (
                "Exclude all Phase 3A/3A.1 pilot cases, then select the first "
                "100 remaining frozen test cases after sorting by SHA256(case_id)."
            ),
            "pilot_case_ids_excluded": sorted(pilot_ids),
            "true_labels_used_for_selection": False,
            "model_outputs_used_for_selection": False,
            "test_metrics_used_for_selection": False,
        },
    )

    write_json(
        OUT / "phase3b_run_config.json",
        {
            "source_method": "validated Phase 3A.1",
            "formal_case_count": FORMAL_CASE_COUNT,
            "search_seeds": list(SEARCH_SEEDS),
            "models": list(MODELS),
            "prefixes": list(PREFIXES),
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "mutation_rate": MUTATION_RATE,
            "crossover_rate": CROSSOVER_RATE,
            "max_options_per_position": MAX_OPTIONS,
            "event_family_preservation": True,
            "same_position_support": True,
            "train_only_dfg": True,
            "timestamps_modified": False,
            "true_label_used_by_search": False,
            "test_metrics_used_by_search": False,
            "pilot_cases_excluded_from_formal_evaluation": True,
            "paper_statistics": True,
        },
    )

    raw_path = OUT / "formal_counterfactual_runs.csv"
    existing = rows(raw_path)
    completed = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            int(r["search_seed_base"]),
        )
        for r in existing
    }

    fieldnames = [
        "case_id",
        "true_label_audit_only",
        "model",
        "prefix",
        "canonical_seed",
        "search_seed_base",
        "search_seed_local",
        "success",
        "failure_category",
        "factual_prediction",
        "target_label",
        "factual_probability",
        "factual_margin",
        "factual_flow_valid",
        "factual_temporal_valid",
        "counterfactual_prediction",
        "counterfactual_probability",
        "counterfactual_target_margin",
        "edit_count",
        "normalized_sequence_distance",
        "temporal_distance",
        "counterfactual_flow_valid",
        "counterfactual_temporal_valid",
        "immutable_history_preserved",
        "event_family_preserved",
        "process_support",
        "generation_found",
        "candidates_evaluated",
        "changed_positions",
        "changed_from",
        "changed_to",
        "factual_activities",
        "counterfactual_activities",
        "timestamps_preserved_exactly",
    ]

    expected = (
        FORMAL_CASE_COUNT
        * len(MODELS)
        * len(PREFIXES)
        * len(SEARCH_SEEDS)
    )
    done = len(completed)

    for model in MODELS:
        for prefix in PREFIXES:
            predictor = CanonicalPredictor(
                ROOT,
                model,
                prefix,
                canonical["items"][model][str(prefix)],
            )

            for seed_base in SEARCH_SEEDS:
                for case in formal_cases:
                    key = (
                        case.case_id,
                        model,
                        int(prefix),
                        int(seed_base),
                    )
                    if key in completed:
                        continue

                    activities = tuple(case.activities[:prefix])
                    timestamps = tuple(case.timestamps[:prefix])
                    seed_local = local_search_seed(
                        seed_base,
                        model,
                        prefix,
                        case.case_id,
                    )

                    result = semantic_evolutionary_counterfactual_search(
                        predictor,
                        activities,
                        timestamps,
                        constraints,
                        rng_seed=seed_local,
                        population_size=POPULATION_SIZE,
                        generations=GENERATIONS,
                        mutation_rate=MUTATION_RATE,
                        crossover_rate=CROSSOVER_RATE,
                        max_options_per_position=MAX_OPTIONS,
                    )

                    cf = result["counterfactual_activities"]

                    if cf is None:
                        positions = []
                        changed_from = []
                        changed_to = []
                        cf_flow = ""
                        cf_temporal = ""
                        immutable_ok = ""
                        family_ok = ""
                        failure_category = (
                            "no_admissible_mutation"
                            if int(result["candidates_evaluated"]) <= 1
                            else "admissible_candidates_no_flip"
                        )
                    else:
                        positions = changed_positions(activities, cf)
                        changed_from = [
                            activities[i - 1]
                            for i in positions
                        ]
                        changed_to = [
                            cf[i - 1]
                            for i in positions
                        ]

                        cf_flow = bool(
                            flow_valid(
                                cf,
                                constraints["edge_count"],
                            )
                        )
                        cf_temporal = bool(
                            temporal_valid(timestamps)
                        )
                        immutable_ok = all(
                            cf[i] == factual
                            for i, factual in enumerate(activities)
                            if factual in (
                                TERMINAL_OFFER_ACTIVITIES
                                | DIRECT_PROXIES
                            )
                        )
                        family_ok = all(
                            family_preserved(
                                activities[i - 1],
                                cf[i - 1],
                            )
                            for i in positions
                        )
                        failure_category = ""

                    factual_p = float(
                        result["factual_probability"]
                    )
                    cf_p = (
                        ""
                        if result["counterfactual_probability"] is None
                        else float(
                            result["counterfactual_probability"]
                        )
                    )
                    target = int(result["target_label"])

                    row = {
                        "case_id": case.case_id,
                        "true_label_audit_only": int(case.label),
                        "model": model,
                        "prefix": int(prefix),
                        "canonical_seed": int(predictor.seed),
                        "search_seed_base": int(seed_base),
                        "search_seed_local": int(seed_local),
                        "success": bool(result["success"]),
                        "failure_category": failure_category,
                        "factual_prediction": int(
                            result["factual_prediction"]
                        ),
                        "target_label": target,
                        "factual_probability": factual_p,
                        "factual_margin": abs(factual_p - 0.5),
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
                                result["counterfactual_prediction"]
                            )
                        ),
                        "counterfactual_probability": cf_p,
                        "counterfactual_target_margin": (
                            ""
                            if cf_p == ""
                            else (
                                cf_p - 0.5
                                if target == 1
                                else 0.5 - cf_p
                            )
                        ),
                        "edit_count": (
                            ""
                            if result["edit_count"] is None
                            else int(result["edit_count"])
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
                        "counterfactual_flow_valid": cf_flow,
                        "counterfactual_temporal_valid": cf_temporal,
                        "immutable_history_preserved": immutable_ok,
                        "event_family_preserved": family_ok,
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
                            result["candidates_evaluated"]
                        ),
                        "changed_positions": json.dumps(
                            positions
                        ),
                        "changed_from": json.dumps(
                            changed_from,
                            ensure_ascii=False,
                        ),
                        "changed_to": json.dumps(
                            changed_to,
                            ensure_ascii=False,
                        ),
                        "factual_activities": json.dumps(
                            activities,
                            ensure_ascii=False,
                        ),
                        "counterfactual_activities": (
                            ""
                            if cf is None
                            else json.dumps(
                                cf,
                                ensure_ascii=False,
                            )
                        ),
                        "timestamps_preserved_exactly": bool(
                            result["success"]
                        ),
                    }

                    append_csv(
                        raw_path,
                        row,
                        fieldnames,
                    )
                    completed.add(key)
                    done += 1

                    if done % 25 == 0 or done == expected:
                        print(
                            f"[{done}/{expected}] "
                            f"{model} P{prefix} seed={seed_base} "
                            f"success={result['success']}",
                            flush=True,
                        )

    raw = rows(raw_path)
    if len(raw) != expected:
        raise AssertionError(
            f"Phase 3B incomplete: {len(raw)}/{expected} rows"
        )

    # ------------------------------------------------------------------
    # Per-seed model/prefix metrics
    # ------------------------------------------------------------------
    per_seed = []
    for model in MODELS:
        for prefix in PREFIXES:
            for seed_base in SEARCH_SEEDS:
                group = [
                    r for r in raw
                    if r["model"] == model
                    and int(r["prefix"]) == int(prefix)
                    and int(r["search_seed_base"]) == int(seed_base)
                ]
                success = [
                    r for r in group
                    if r["success"].lower() == "true"
                ]
                no_mut = [
                    r for r in group
                    if r["failure_category"] == "no_admissible_mutation"
                ]
                no_flip = [
                    r for r in group
                    if r["failure_category"] == "admissible_candidates_no_flip"
                ]

                edit_values = [
                    float(r["edit_count"])
                    for r in success
                ]
                dist_values = [
                    float(r["normalized_sequence_distance"])
                    for r in success
                ]
                eval_values = [
                    float(r["candidates_evaluated"])
                    for r in group
                ]
                margin_values = [
                    float(r["counterfactual_target_margin"])
                    for r in success
                ]

                per_seed.append(
                    {
                        "model": model,
                        "prefix": int(prefix),
                        "search_seed": int(seed_base),
                        "targets": len(group),
                        "successful": len(success),
                        "gsr": len(success) / len(group),
                        "no_admissible_mutation": len(no_mut),
                        "admissible_candidates_no_flip": len(no_flip),
                        "flow_valid_rate_successes": (
                            ""
                            if not success
                            else sum(
                                r["counterfactual_flow_valid"].lower()
                                == "true"
                                for r in success
                            )
                            / len(success)
                        ),
                        "temporal_valid_rate_successes": (
                            ""
                            if not success
                            else sum(
                                r["counterfactual_temporal_valid"].lower()
                                == "true"
                                for r in success
                            )
                            / len(success)
                        ),
                        "family_valid_rate_successes": (
                            ""
                            if not success
                            else sum(
                                r["event_family_preserved"].lower()
                                == "true"
                                for r in success
                            )
                            / len(success)
                        ),
                        "immutable_rate_successes": (
                            ""
                            if not success
                            else sum(
                                r["immutable_history_preserved"].lower()
                                == "true"
                                for r in success
                            )
                            / len(success)
                        ),
                        "mean_edit_count_successes": (
                            ""
                            if not edit_values
                            else float(np.mean(edit_values))
                        ),
                        "mean_normalized_distance_successes": (
                            ""
                            if not dist_values
                            else float(np.mean(dist_values))
                        ),
                        "mean_target_margin_successes": (
                            ""
                            if not margin_values
                            else float(np.mean(margin_values))
                        ),
                        "mean_candidates_evaluated": float(
                            np.mean(eval_values)
                        ),
                    }
                )

    write_csv(
        OUT / "formal_metrics_per_seed.csv",
        per_seed,
    )

    # ------------------------------------------------------------------
    # Mean/std/95% CI across search seeds
    # ------------------------------------------------------------------
    summary = []
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
            item = {
                "model": model,
                "prefix": int(prefix),
                "n_search_seeds": len(group),
                "targets_per_seed": int(group[0]["targets"]),
            }
            for metric in metric_names:
                vals = [
                    float(r[metric])
                    for r in group
                    if r[metric] != ""
                ]
                if not vals:
                    item[f"{metric}_mean"] = ""
                    item[f"{metric}_std"] = ""
                    item[f"{metric}_ci95_low"] = ""
                    item[f"{metric}_ci95_high"] = ""
                    continue
                mean, std, lo, hi = mean_std_ci(vals)
                item[f"{metric}_mean"] = mean
                item[f"{metric}_std"] = std
                item[f"{metric}_ci95_low"] = lo
                item[f"{metric}_ci95_high"] = hi

            item["mean_no_admissible_mutation"] = float(
                np.mean(
                    [
                        int(r["no_admissible_mutation"])
                        for r in group
                    ]
                )
            )
            item["mean_admissible_candidates_no_flip"] = float(
                np.mean(
                    [
                        int(r["admissible_candidates_no_flip"])
                        for r in group
                    ]
                )
            )
            summary.append(item)

    write_csv(
        OUT / "formal_metrics_summary.csv",
        summary,
    )

    # ------------------------------------------------------------------
    # Cross-seed stability per target/model/prefix
    # ------------------------------------------------------------------
    stability = []
    for model in MODELS:
        for prefix in PREFIXES:
            for case_id in formal_ids:
                group = sorted(
                    [
                        r for r in raw
                        if r["model"] == model
                        and int(r["prefix"]) == int(prefix)
                        and r["case_id"] == case_id
                    ],
                    key=lambda r: int(r["search_seed_base"]),
                )
                successful = [
                    r for r in group
                    if r["success"].lower() == "true"
                ]

                cf_sequences = [
                    r["counterfactual_activities"]
                    for r in successful
                ]
                position_sets = [
                    json.loads(r["changed_positions"])
                    for r in successful
                ]

                pairwise_jaccards = [
                    jaccard(a, b)
                    for a, b in itertools.combinations(
                        position_sets,
                        2,
                    )
                ]

                if cf_sequences:
                    counts = {}
                    for seq in cf_sequences:
                        counts[seq] = counts.get(seq, 0) + 1
                    modal_count = max(counts.values())
                else:
                    modal_count = 0

                stability.append(
                    {
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "successful_seeds": len(successful),
                        "success_rate_across_seeds": (
                            len(successful)
                            / len(SEARCH_SEEDS)
                        ),
                        "unique_counterfactuals": len(
                            set(cf_sequences)
                        ),
                        "modal_counterfactual_fraction_among_successes": (
                            ""
                            if not successful
                            else modal_count / len(successful)
                        ),
                        "mean_pairwise_modified_position_jaccard": (
                            ""
                            if not pairwise_jaccards
                            else float(
                                np.mean(pairwise_jaccards)
                            )
                        ),
                    }
                )

    write_csv(
        OUT / "cross_seed_stability.csv",
        stability,
    )

    # Overall summary, still stratified enough for reporting.
    overall = {
        "formal_case_count": FORMAL_CASE_COUNT,
        "pilot_case_count_excluded": len(pilot_ids),
        "raw_run_rows": len(raw),
        "search_seeds": list(SEARCH_SEEDS),
        "models": list(MODELS),
        "prefixes": list(PREFIXES),
        "all_successful_counterfactuals_expected_valid_by_construction": True,
    }
    write_json(
        OUT / "phase3b_overall.json",
        overall,
    )

    lines = [
        "# ETCE Phase 3B Formal Counterfactual-Generation Evaluation",
        "",
        "## Frozen protocol",
        "",
        f"- Formal cases: {FORMAL_CASE_COUNT} previously unseen test cases.",
        "- All 20 Phase 3A/3A.1 pilot cases are excluded.",
        f"- Search seeds: {list(SEARCH_SEEDS)}.",
        "- Models: RF, XGBoost, MLP, LSTM.",
        "- Prefixes: 5, 10, 15, 20.",
        "- Generator: validated Phase 3A.1 semantic-family-preserving constrained evolutionary search.",
        "- No hyperparameter/search-budget tuning is performed in Phase 3B.",
        "",
        "## Mean ± std across five search seeds",
        "",
        "| Model | Prefix | GSR | Mean edits | Norm. distance | Target margin | Candidates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for r in summary:
        lines.append(
            f"| {r['model']} | {r['prefix']} | "
            f"{float(r['gsr_mean']):.4f} ± {float(r['gsr_std']):.4f} | "
            f"{float(r['mean_edit_count_successes_mean']):.4f} ± "
            f"{float(r['mean_edit_count_successes_std']):.4f} | "
            f"{float(r['mean_normalized_distance_successes_mean']):.4f} ± "
            f"{float(r['mean_normalized_distance_successes_std']):.4f} | "
            f"{float(r['mean_target_margin_successes_mean']):.4f} ± "
            f"{float(r['mean_target_margin_successes_std']):.4f} | "
            f"{float(r['mean_candidates_evaluated_mean']):.2f} ± "
            f"{float(r['mean_candidates_evaluated_std']):.2f} |"
        )

    lines += [
        "",
        "## Validity and interpretation",
        "",
        "Every reported successful counterfactual is required to satisfy the train-only DFG, timestamp-order preservation, immutable historical terminal events, forbidden-target exclusion, and event-family preservation.",
        "Failure is explicitly separated into `no_admissible_mutation` and `admissible_candidates_no_flip`.",
        "Cross-seed stability is stored in `cross_seed_stability.csv` using modified-position Jaccard and exact counterfactual repetition.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 3B validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]

    (
        OUT / "PHASE3B_REPORT.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Phase 3B formal run complete.")
    print(OUT / "PHASE3B_REPORT.md")
    print("Run scripts/17_validate_phase3b.py next.")


if __name__ == "__main__":
    main()
