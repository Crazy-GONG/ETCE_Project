"""Independent validator for ETCE Phase 4A.1.

Run:
    python scripts/21_validate_phase4a1.py
"""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".deps"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from etce.phase2.data import load_frozen_common_cohort
from etce.phase4.stratified_core import run_all_surrogates_stratified
from etce.utils.io import read_json, write_json

PHASE4A = ROOT / "results/phase4/phase4a"
OUT = ROOT / "results/phase4/phase4a1"


def rows(path):
    with Path(path).open(
        encoding="utf-8",
        newline="",
    ) as stream:
        return list(csv.DictReader(stream))


def gzip_rows(path):
    with gzip.open(
        path,
        "rt",
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


def validate():
    checks = {}

    required = [
        OUT / "PHASE4A1_REPORT.md",
        OUT / "phase4a1_run_config.json",
        OUT / "surrogate_metrics_stratified.csv",
        OUT / "surrogate_aggregate_stratified.csv",
        OUT / "canonical_rules_stratified.jsonl",
        PHASE4A / "validation_phase4a.json",
        PHASE4A / "local_neighborhoods.csv.gz",
        PHASE4A / "surrogate_metrics.csv",
    ]
    require(
        all(path.is_file() for path in required),
        "Required Phase 4A.1 artifact missing",
    )
    require(
        read_json(
            PHASE4A / "validation_phase4a.json"
        ).get("status") == "passed",
        "Phase 4A is not validated",
    )
    checks["required_outputs_and_source_phase_validated"] = True

    config = read_json(
        OUT / "phase4a1_run_config.json"
    )
    require(
        config["same_pilot_cases"] is True
        and config["same_local_neighborhoods"] is True
        and config["same_blackbox_probabilities"] is True
        and config["same_surrogate_definitions"] is True
        and config["same_surrogate_hyperparameters"] is True,
        "Paired Phase 4A.1 design changed more than local split",
    )
    require(
        config["true_outcome_labels_used_for_split"] is False
        and config["blackbox_local_labels_used_for_stratification"] is True
        and config["test_cases_used"] is False,
        "Invalid Phase 4A.1 split metadata",
    )
    checks["paired_protocol_metadata_ok"] = True

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

    metrics = rows(
        OUT / "surrogate_metrics_stratified.csv"
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
        == len(grouped) * 3,
        "Phase 4A.1 metric inventory mismatch",
    )

    saved_rules = []
    with (
        OUT / "canonical_rules_stratified.jsonl"
    ).open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            line = line.strip()
            if line:
                saved_rules.append(
                    json.loads(line)
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

    reproduced_rule_count = 0
    pools_with_both_classes = 0
    pools_with_both_classes_fit_both = 0

    for key, local in grouped.items():
        case_id, model, prefix = key
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
        factual_label = int(
            labels[
                sequences.index(factual)
            ]
        )

        result = run_all_surrogates_stratified(
            sequences,
            labels,
            factual,
            factual_label,
        )

        pool_classes = set(
            labels.tolist()
        )
        fit_classes = set(
            np.asarray(
                result["fit_labels"],
                dtype=int,
            ).tolist()
        )
        if len(pool_classes) == 2:
            pools_with_both_classes += 1
            require(
                len(fit_classes) == 2,
                f"Stratified split lost a present class: {key}",
            )
            pools_with_both_classes_fit_both += 1

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

            expected = branch[
                "surrogate_fidelity_eval"
            ]
            if expected is None:
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
                        expected,
                    ),
                    "Surrogate fidelity mismatch",
                )

            for rule_type, rule in (
                ("factual", branch["factual_rule"]),
                ("counterfactual", branch["counterfactual_rule"]),
            ):
                rkey = (
                    case_id,
                    model,
                    int(prefix),
                    surrogate,
                    rule_type,
                )
                saved = rule_index.get(rkey)
                if rule is None:
                    require(
                        saved is None,
                        "Unexpected saved rule",
                    )
                    continue

                require(
                    saved is not None,
                    "Expected rule missing",
                )
                require(
                    int(saved["predicted_class"])
                    == int(rule["predicted_class"]),
                    "Rule class mismatch",
                )
                require(
                    saved["conditions"]
                    == rule["conditions"],
                    "Rule conditions mismatch",
                )
                require(
                    int(saved["support_eval"])
                    == int(rule["support"]),
                    "Rule support mismatch",
                )
                if rule["fidelity"] is None:
                    require(
                        saved["fidelity_eval"] is None,
                        "Rule fidelity null mismatch",
                    )
                else:
                    require(
                        close(
                            saved["fidelity_eval"],
                            rule["fidelity"],
                        ),
                        "Rule fidelity mismatch",
                    )
                reproduced_rule_count += 1

    require(
        pools_with_both_classes
        == pools_with_both_classes_fit_both,
        "Not every two-class pool retained two classes in fit",
    )
    checks[
        "all_present_blackbox_classes_retained_in_fit"
    ] = True
    checks[
        "all_stratified_surrogates_and_rules_reproduced"
    ] = True

    require(
        reproduced_rule_count
        == len(saved_rules),
        "Saved/reproduced rule count mismatch",
    )

    result = {
        "scope": "ETCE Phase 4A.1",
        "status": "passed",
        "checks": checks,
        "local_target_count": len(grouped),
        "two_class_local_pools": pools_with_both_classes,
        "two_class_fit_subsets": pools_with_both_classes_fit_both,
        "surrogate_metric_rows": len(metrics),
        "canonical_rule_count": len(saved_rules),
    }

    write_json(
        OUT / "validation_phase4a1.json",
        result,
    )

    report_path = (
        OUT / "PHASE4A1_REPORT.md"
    )
    report = report_path.read_text(
        encoding="utf-8"
    )
    old = (
        "<!-- VALIDATION_START -->\n"
        "Awaiting Phase 4A.1 validation.\n"
        "<!-- VALIDATION_END -->"
    )
    new = (
        "<!-- VALIDATION_START -->\n"
        "Post-run validation: **PASSED**. The paired Phase 4A neighborhoods were "
        "kept unchanged; every black-box class present in a local pool was retained "
        "in the fitting subset, and all stratified surrogate metrics and canonical "
        "rules were independently reproduced.\n"
        "<!-- VALIDATION_END -->"
    )
    require(
        old in report,
        "Phase 4A.1 report validation marker missing",
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
