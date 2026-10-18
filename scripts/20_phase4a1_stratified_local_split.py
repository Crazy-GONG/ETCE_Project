"""ETCE Phase 4A.1 paired stratified local-evidence refinement.

Uses exactly the validated Phase 4A local neighborhoods.
The ONLY methodological change is the local fit/eval split:
deterministic class-stratified 70/30 instead of label-independent hash split.

Run:
    python scripts/20_phase4a1_stratified_local_split.py
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from etce.phase2.data import load_frozen_common_cohort
from etce.phase4.stratified_core import run_all_surrogates_stratified
from etce.utils.io import read_json, write_csv, write_json

PHASE4A = ROOT / "results/phase4/phase4a"
OUT = ROOT / "results/phase4/phase4a1"
SURROGATES = ("LORE", "DT", "BN")


def gzip_rows(path):
    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def rows(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def main():
    validation = read_json(
        PHASE4A / "validation_phase4a.json"
    )
    if validation.get("status") != "passed":
        raise RuntimeError(
            "Validated Phase 4A is required before Phase 4A.1"
        )

    selection = read_json(
        PHASE4A / "pilot_case_selection.json"
    )
    pilot_ids = list(selection["case_ids"])

    cases, _ = load_frozen_common_cohort(ROOT)
    case_by_id = {
        case.case_id: case
        for case in cases
        if case.case_id in set(pilot_ids)
    }

    neighborhood = gzip_rows(
        PHASE4A / "local_neighborhoods.csv.gz"
    )
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

    old_metrics = rows(
        PHASE4A / "surrogate_metrics.csv"
    )
    old_index = {
        (
            r["case_id"],
            r["model"],
            int(r["prefix"]),
            r["surrogate"],
        ): r
        for r in old_metrics
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        OUT / "phase4a1_run_config.json",
        {
            "paired_source": "validated Phase 4A",
            "same_pilot_cases": True,
            "same_local_neighborhoods": True,
            "same_blackbox_probabilities": True,
            "same_surrogate_definitions": True,
            "same_surrogate_hyperparameters": True,
            "only_change": (
                "deterministic class-stratified local 70/30 fit/eval split; "
                "factual remains fit; singleton class remains fit"
            ),
            "true_outcome_labels_used_for_split": False,
            "blackbox_local_labels_used_for_stratification": True,
            "test_cases_used": False,
            "paper_statistics": False,
        },
    )

    metrics = []
    rules = []

    ordered_keys = sorted(
        grouped,
        key=lambda x: (
            x[1],
            x[2],
            x[0],
        ),
    )

    for done, key in enumerate(
        ordered_keys,
        1,
    ):
        case_id, model, prefix = key
        local = grouped[key]
        case = case_by_id[case_id]
        factual = tuple(
            case.activities[:prefix]
        )

        sequences = [
            tuple(
                json.loads(
                    row["activities"]
                )
            )
            for row in local
        ]
        labels = np.asarray(
            [
                int(row["bb_label"])
                for row in local
            ],
            dtype=int,
        )

        factual_idx = sequences.index(
            factual
        )
        factual_label = int(
            labels[factual_idx]
        )

        result = run_all_surrogates_stratified(
            sequences,
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

        for branch in result["branches"]:
            surrogate = branch["branch"]
            factual_rule = branch["factual_rule"]
            cf_rule = branch["counterfactual_rule"]
            old = old_index[
                (
                    case_id,
                    model,
                    int(prefix),
                    surrogate,
                )
            ]

            row = {
                "case_id": case_id,
                "model": model,
                "prefix": int(prefix),
                "surrogate": surrogate,
                "pool_size": len(sequences),
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
                "status": branch["status"],
                "surrogate_fidelity_eval": (
                    ""
                    if branch["surrogate_fidelity_eval"] is None
                    else float(
                        branch["surrogate_fidelity_eval"]
                    )
                ),
                "factual_rule_available": bool(
                    factual_rule is not None
                ),
                "factual_rule_support_eval": (
                    ""
                    if factual_rule is None
                    else int(
                        factual_rule["support"]
                    )
                ),
                "factual_rule_fidelity_eval": (
                    ""
                    if (
                        factual_rule is None
                        or factual_rule["fidelity"] is None
                    )
                    else float(
                        factual_rule["fidelity"]
                    )
                ),
                "factual_rule_length": (
                    ""
                    if factual_rule is None
                    else int(
                        factual_rule["rule_length"]
                    )
                ),
                "cf_rule_available": bool(
                    cf_rule is not None
                ),
                "cf_rule_support_eval": (
                    ""
                    if cf_rule is None
                    else int(
                        cf_rule["support"]
                    )
                ),
                "cf_rule_fidelity_eval": (
                    ""
                    if (
                        cf_rule is None
                        or cf_rule["fidelity"] is None
                    )
                    else float(
                        cf_rule["fidelity"]
                    )
                ),
                "cf_rule_length": (
                    ""
                    if cf_rule is None
                    else int(
                        cf_rule["rule_length"]
                    )
                ),
                "cf_witness_edit_count": (
                    ""
                    if (
                        cf_rule is None
                        or "witness_edit_count" not in cf_rule
                    )
                    else int(
                        cf_rule["witness_edit_count"]
                    )
                ),
                "phase4a_status": old["status"],
                "phase4a_factual_rule_available": old[
                    "factual_rule_available"
                ],
                "phase4a_cf_rule_available": old[
                    "cf_rule_available"
                ],
            }
            metrics.append(row)

            for rule_type, rule in (
                ("factual", factual_rule),
                ("counterfactual", cf_rule),
            ):
                if rule is None:
                    continue
                rules.append(
                    {
                        "case_id": case_id,
                        "model": model,
                        "prefix": int(prefix),
                        "surrogate": surrogate,
                        "rule_type": rule_type,
                        "predicted_class": int(
                            rule["predicted_class"]
                        ),
                        "conditions": rule["conditions"],
                        "support_eval": int(
                            rule["support"]
                        ),
                        "fidelity_eval": rule["fidelity"],
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
                )

        print(
            f"[{done:03d}/{len(ordered_keys)}] "
            f"{model} P{prefix} {case_id}: "
            f"pool=({int(np.sum(labels==0))},"
            f"{int(np.sum(labels==1))}) "
            f"fit=({int(np.sum(fit_labels==0))},"
            f"{int(np.sum(fit_labels==1))}) "
            f"eval=({int(np.sum(eval_labels==0))},"
            f"{int(np.sum(eval_labels==1))})",
            flush=True,
        )

    write_csv(
        OUT / "surrogate_metrics_stratified.csv",
        metrics,
    )

    with (
        OUT / "canonical_rules_stratified.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as stream:
        for rule in rules:
            stream.write(
                json.dumps(
                    rule,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    aggregates = []
    old_aggregate = {
        r["surrogate"]: r
        for r in rows(
            PHASE4A / "surrogate_aggregate.csv"
        )
    }

    for surrogate in SURROGATES:
        group = [
            r
            for r in metrics
            if r["surrogate"] == surrogate
        ]
        sfid = [
            float(r["surrogate_fidelity_eval"])
            for r in group
            if r["surrogate_fidelity_eval"] != ""
        ]
        ffid = [
            float(r["factual_rule_fidelity_eval"])
            for r in group
            if r["factual_rule_fidelity_eval"] != ""
        ]
        cfid = [
            float(r["cf_rule_fidelity_eval"])
            for r in group
            if r["cf_rule_fidelity_eval"] != ""
        ]

        fact_av = float(
            np.mean(
                [
                    bool(r["factual_rule_available"])
                    for r in group
                ]
            )
        )
        cf_av = float(
            np.mean(
                [
                    bool(r["cf_rule_available"])
                    for r in group
                ]
            )
        )

        old = old_aggregate[surrogate]
        aggregates.append(
            {
                "surrogate": surrogate,
                "targets": len(group),
                "phase4a_mean_surrogate_fidelity_eval": float(
                    old["mean_surrogate_fidelity_eval"]
                ),
                "phase4a1_mean_surrogate_fidelity_eval": (
                    ""
                    if not sfid
                    else float(np.mean(sfid))
                ),
                "phase4a_factual_rule_availability": float(
                    old["factual_rule_availability"]
                ),
                "phase4a1_factual_rule_availability": fact_av,
                "delta_factual_rule_availability": (
                    fact_av
                    - float(
                        old["factual_rule_availability"]
                    )
                ),
                "phase4a_cf_rule_availability": float(
                    old["cf_rule_availability"]
                ),
                "phase4a1_cf_rule_availability": cf_av,
                "delta_cf_rule_availability": (
                    cf_av
                    - float(
                        old["cf_rule_availability"]
                    )
                ),
                "phase4a1_mean_factual_rule_fidelity_eval": (
                    ""
                    if not ffid
                    else float(np.mean(ffid))
                ),
                "phase4a1_mean_cf_rule_fidelity_eval": (
                    ""
                    if not cfid
                    else float(np.mean(cfid))
                ),
            }
        )

    write_csv(
        OUT / "surrogate_aggregate_stratified.csv",
        aggregates,
    )

    report = [
        "# ETCE Phase 4A.1 Stratified Local Evidence Pilot",
        "",
        "> Paired refinement of validated Phase 4A. Same neighborhoods and surrogate models; only the local fit/eval partition changes.",
        "",
        "## Motivation",
        "",
        "Phase 4A showed that some local pools contained both black-box classes, but the label-independent 70/30 hash split occasionally placed the minority class entirely outside the fitting subset. That creates an avoidable `single_class_fit` failure unrelated to process validity.",
        "",
        "Phase 4A.1 therefore stratifies the local 70/30 split by **black-box local label**. The true process outcome label is not used.",
        "",
        "Rules and fidelity are still evaluated on held-out local samples.",
        "",
        "## Paired aggregate comparison",
        "",
        "| Surrogate | P4A surrogate fid. | P4A.1 surrogate fid. | Factual availability P4A→P4A.1 | CF availability P4A→P4A.1 | P4A.1 factual rule fid. | P4A.1 CF rule fid. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(v):
        if v == "":
            return "-"
        return f"{float(v):.4f}"

    for r in aggregates:
        report.append(
            f"| {r['surrogate']} | "
            f"{r['phase4a_mean_surrogate_fidelity_eval']:.4f} | "
            f"{fmt(r['phase4a1_mean_surrogate_fidelity_eval'])} | "
            f"{r['phase4a_factual_rule_availability']:.4f}→"
            f"{r['phase4a1_factual_rule_availability']:.4f} | "
            f"{r['phase4a_cf_rule_availability']:.4f}→"
            f"{r['phase4a1_cf_rule_availability']:.4f} | "
            f"{fmt(r['phase4a1_mean_factual_rule_fidelity_eval'])} | "
            f"{fmt(r['phase4a1_mean_cf_rule_fidelity_eval'])} |"
        )

    report += [
        "",
        "## Interpretation boundary",
        "",
        "This refinement does not expand the neighborhood, alter process constraints, change a surrogate hyperparameter, or use test data. It only prevents an existing black-box class from being lost from the fitting subset because of an unstratified local partition.",
        "",
        "<!-- VALIDATION_START -->",
        "Awaiting Phase 4A.1 validation.",
        "<!-- VALIDATION_END -->",
        "",
    ]

    (
        OUT / "PHASE4A1_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("Phase 4A.1 complete.")
    print(OUT / "PHASE4A1_REPORT.md")
    print("Run scripts/21_validate_phase4a1.py next.")


if __name__ == "__main__":
    main()
